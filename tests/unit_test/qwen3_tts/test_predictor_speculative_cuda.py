# SPDX-License-Identifier: Apache-2.0
"""CUDA full-frame and persistent-graph gates for Predictor speculation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from sglang.srt.runtime_context import get_context
from torch import nn

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.models.qwen3_tts import sampling_kernels as sampling_kernels_module
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    build_speculative_probability_candidates,
    build_speculative_probability_candidates_radix,
    fused_raw_logit_block_k,
    sample_from_logits_with_seed_top_k_top_p,
)
from sglang_omni.models.qwen3_tts.sglang_model import (
    Qwen3TTSCodecTables,
    Qwen3TTSTalker,
)
from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    SpeculativeAcceptanceBatch,
    accept_speculative_candidates_compact,
    accept_speculative_candidates_dense_indexed,
    sample_compact_probability_rows_fused,
)
from sglang_omni.models.qwen3_tts.speculative_commit import (
    commit_speculative_codes,
    commit_speculative_state,
)
from sglang_omni.models.qwen3_tts.speculative_prepare import (
    SpeculativeVerifierInputs,
    prepare_speculative_verification_inputs,
)
from sglang_omni.models.qwen3_tts.speculative_rng import (
    speculative_uniforms,
    speculative_uniforms_all_purposes,
    speculative_uniforms_frame_pool,
)
from sglang_omni.models.qwen3_tts.speculative_runtime import (
    accept_speculative_candidates,
    expand_compact_probability_rows,
    sample_compact_probability_rows,
)
from sglang_omni.models.qwen3_tts.speculative_workspace import (
    PredictorSpeculativeWorkspace,
)
from sglang_omni.utils.predictor_grouped_heads import grouped_lm_head_linear
from sglang_omni.utils.qwen3_tts_speculative_frame_observation import (
    observe_speculative_rounds,
)
from sglang_omni.vendor.sglang.layers import RMSNorm
from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
    DTYPE,
    FUSED_HIDDEN,
    TupleLinear,
    real_shape_layer,
    real_shape_talker,
)

pytestmark = pytest.mark.accelerator
CODEBOOK_VOCAB_SIZE = 2048
TALKER_HIDDEN_SIZE = 2048


def identity_qk_norm(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    head_dim: int,
    alt_stream: torch.cuda.Stream | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(q_norm, nn.Module)
    assert isinstance(k_norm, nn.Module)
    assert head_dim > 0
    assert alt_stream is None or isinstance(alt_stream, torch.cuda.Stream)
    return q, k


def build_speculative_talker(
    device: torch.device, batch_size: int = 1
) -> Qwen3TTSTalker:
    layer_stack = [real_shape_layer(device) for _ in range(5)]
    final_norm = RMSNorm(FUSED_HIDDEN, eps=1e-6).to(device=device, dtype=DTYPE)
    talker = real_shape_talker(device, layer_stack, final_norm, fused=True)
    talker.model = SimpleNamespace(
        codec_embedding=SimpleNamespace(
            weight=torch.empty(
                CODEBOOK_VOCAB_SIZE, TALKER_HIDDEN_SIZE, device=device, dtype=DTYPE
            )
        )
    )
    talker.config = SimpleNamespace(
        num_code_groups=16,
        code_predictor_config=SimpleNamespace(vocab_size=CODEBOOK_VOCAB_SIZE),
    )
    talker.predictor_device = talker.predictor_k_cache.device
    talker.predictor_device_module = torch.get_device_module(device)
    talker.predictor_max_batch_size = batch_size
    talker.predictor_graph_batch_sizes = (batch_size,)
    talker.predictor_capture_stream = None
    talker.predictor_graph_pool = None
    talker.output_codes = torch.full(
        (batch_size, 16), -1, device=device, dtype=torch.long
    )
    talker.output_embeds = torch.zeros(
        (batch_size, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
    )
    talker.predictor_embedding_buffer = torch.empty(
        (batch_size, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
    )
    small_to_mtp_projection = nn.Linear(TALKER_HIDDEN_SIZE, FUSED_HIDDEN, bias=True).to(
        device=device, dtype=DTYPE
    )
    talker.predictor_projected_embeddings = None
    talker.predictor_projected_buffer = torch.empty(
        (batch_size, FUSED_HIDDEN), device=device, dtype=DTYPE
    )
    talker.predictor_cutedsl_layers = None
    talker.sub_batch_size = batch_size
    talker.sub_temperature_tensor = torch.full((batch_size,), 0.9, device=device)
    talker.sub_top_p_tensor = torch.ones((batch_size,), device=device)
    talker.sub_top_k_tensor = torch.full(
        (batch_size,), 4, device=device, dtype=torch.long
    )
    talker.sub_sampling_seed_tensor = torch.full(
        (batch_size,), 1729, device=device, dtype=torch.long
    )
    talker.sub_speculative_seed_tensor = torch.full(
        (batch_size,), 8123, device=device, dtype=torch.long
    )
    talker.sub_speculative_round_index_tensor = None
    talker.sub_speculative_uniforms_tensor = None
    talker.sub_speculative_rng_pool_tensor = None
    talker.sub_seed_offsets = torch.arange(1, 16, device=device, dtype=torch.long)
    talker.sub_do_sample_tensor = torch.ones(
        (batch_size,), device=device, dtype=torch.bool
    )
    talker.sub_request_ids = tuple(
        f"unseeded-full-frame-test-{row}" for row in range(batch_size)
    )
    talker.sub_user_seed_rows = tuple(False for _ in range(batch_size))
    talker.sub_sample_rows = tuple(True for _ in range(batch_size))
    talker.sub_has_sampled_rows = True
    talker.sub_has_argmax_rows = False
    talker.sub_sampled_has_top_p = False
    talker.sub_sampled_max_top_k = 4
    talker.sub_sampled_has_unbounded_top_k = False
    talker.code_predictor_spec_requested_width = 0
    talker.code_predictor_spec_effective_width = 0
    talker.code_predictor_spec_backend = "cuda_graph"
    talker.code_predictor_spec_opening_pair_graph_enabled = False
    talker.code_predictor_spec_nvtx_enabled = False
    talker.code_predictor_spec_rng_backend = "per_round"
    talker.code_predictor_spec_lm_head_backend = "grouped_triton"
    talker.code_predictor_spec_probability_backend = "split_top_k"
    talker.code_predictor_spec_probability_backend_execution_counts = {
        "split_top_k": 0,
        "radix_compact": 0,
        "radix_compact_4bit": 0,
    }
    talker.code_predictor_spec_probability_backend_fallback_count = 0
    talker.code_predictor_spec_acceptance_backend = "compact"
    talker.code_predictor_spec_acceptance_backend_execution_counts = {
        "compact": 0,
        "dense_indexed": 0,
    }
    talker.code_predictor_spec_dense_draft_probabilities = None
    talker.code_predictor_spec_dense_target_probabilities = None
    talker.code_predictor_spec_dense_draft_previous_ids = None
    talker.code_predictor_spec_dense_target_previous_ids = None
    talker.code_predictor_spec_dense_current_ids = None
    talker.code_predictor_spec_dense_candidate_offsets = None
    talker.code_predictor_spec_dense_candidate_capacity = 0
    talker.code_predictor_spec_fallback_reason = None
    talker.code_predictor_spec_graphs = {}
    talker.code_predictor_spec_opening_graphs = {}
    talker.code_predictor_spec_opening_graph_disabled = set()
    talker.code_predictor_spec_round_graphs = {}
    talker.code_predictor_spec_round_graph_disabled = set()
    talker.code_predictor_spec_input_buffers = {}
    talker.code_predictor_spec_effective_width_histogram = {}
    talker.code_predictor_spec_graph_replay_count = 0
    talker.code_predictor_spec_eager_round_count = 0
    talker.code_predictor_spec_graph_capture_failure_count = 0
    talker.code_predictor_spec_full_round_graph_replay_count = 0
    talker.code_predictor_spec_opening_graph_replay_count = 0
    talker.code_predictor_spec_full_round_graph_capture_count = 0
    talker.code_predictor_spec_full_round_graph_capture_failure_count = 0
    talker.code_predictor_spec_opening_graph_capture_count = 0
    talker.code_predictor_spec_opening_graph_capture_failure_count = 0
    talker.code_predictor_spec_full_round_graph_capture_time_ms = 0.0
    talker.code_predictor_spec_opening_graph_capture_time_ms = 0.0
    talker.code_predictor_spec_graph_memory_delta_bytes = 0
    talker.code_predictor_spec_graph_capture_time_ms = 0.0
    talker.code_predictor_spec_verifier_graph_capture_time_ms = 0.0
    talker.code_predictor_spec_configuration_time_ms = 0.0
    talker.code_predictor_output_frame_count = 0
    talker.code_predictor_spec_frame_count = 0
    talker.code_predictor_spec_round_count = 0
    talker.code_predictor_spec_proposed_count = 0
    talker.code_predictor_spec_accepted_count = 0
    talker.code_predictor_spec_committed_count = 0
    talker.code_predictor_spec_bonus_count = 0
    talker.code_predictor_spec_rejection_count = 0
    talker.code_predictor_spec_packed_forward_count = 0
    talker.code_predictor_spec_packed_query_slots = 0
    talker.code_predictor_spec_padded_query_slots = 0
    talker.code_predictor_spec_fused_predictor_forward_count = 0
    talker.code_predictor_generic_forward_count = 0
    talker.code_predictor_spec_standalone_correction_forward_count = 0
    talker.code_predictor_spec_correction_forward_watch = False
    talker.code_predictor_spec_draft_head_evaluation_count = 0
    talker.code_predictor_spec_target_head_evaluation_count = 0
    talker.code_predictor_spec_native_fallback_count = 0
    talker.code_predictor_spec_lm_head_weights = None
    talker.code_predictor_spec_metadata_device = None
    talker.code_predictor_spec_metadata_host = None

    codec_tables = Qwen3TTSCodecTables(15, CODEBOOK_VOCAB_SIZE, TALKER_HIDDEN_SIZE).to(
        device=device, dtype=DTYPE
    )
    talker.code_predictor.model.codec_embedding = codec_tables
    talker.code_predictor.small_to_mtp_projection = small_to_mtp_projection
    talker.code_predictor.project_input = small_to_mtp_projection
    talker.predictor_projected_embeddings = small_to_mtp_projection(
        codec_tables.weight[:-1]
    ).contiguous()
    talker.code_predictor.lm_head = nn.ModuleList(
        [
            TupleLinear(FUSED_HIDDEN, CODEBOOK_VOCAB_SIZE).to(
                device=device, dtype=DTYPE
            )
            for _ in range(15)
        ]
    )
    layer0_embedding = nn.Embedding(CODEBOOK_VOCAB_SIZE, TALKER_HIDDEN_SIZE).to(
        device=device, dtype=DTYPE
    )

    def get_layer0_embeddings() -> nn.Embedding:
        return layer0_embedding

    talker.get_input_embeddings = get_layer0_embeddings
    talker.predictor_graphs = {}
    talker.predictor_graph_disabled = set()
    talker.predictor_graph_failure_count = 0
    talker.predictor_graph_capture_count = 0
    talker.predictor_graph_replay_count = 0
    talker.predictor_graph_startup_count = 0
    talker.predictor_graph_capacity_fallback_count = 0
    talker.predictor_graph_capacity_warned = False
    talker.predictor_eager_count = 0
    talker.predictor_graph_enabled = False
    talker.decode_prep_rids = None
    talker.decode_prep_epoch = 0
    return talker


def configure_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    execution_state = SimpleNamespace(
        graph=SimpleNamespace(disable_cuda_graph=False),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    monkeypatch.setattr(sglang_model_module, "get_exec", lambda: execution_state)
    monkeypatch.setattr(sglang_model_module, "QTTS_BACKEND", "default")
    monkeypatch.setattr(
        sglang_model_module,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )
    monkeypatch.setattr(
        sglang_model_module,
        "apply_qk_norm",
        identity_qk_norm,
    )


def configure_unseeded_request(talker: Qwen3TTSTalker, batch_size: int = 1) -> None:
    talker.sub_request_ids = tuple(
        f"unseeded-full-frame-test-{row}" for row in range(batch_size)
    )
    talker.sub_user_seed_rows = tuple(False for _ in range(batch_size))
    talker.sub_sample_rows = tuple(True for _ in range(batch_size))
    talker.sub_has_sampled_rows = True
    talker.sub_has_argmax_rows = False
    talker.sub_sampled_has_top_p = False
    talker.sub_sampled_max_top_k = 4
    talker.sub_sampled_has_unbounded_top_k = False


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_opening_pair_graph_propagates_fatal_cuda_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        device = torch.device("cuda")
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(
            4,
            backend="eager",
            opening_pair_graph=True,
        )
        workspace = talker.ensure_code_predictor_speculative_workspace()
        signature = ("sampled", 4, False, False, False)
        monkeypatch.setitem(
            talker.__dict__,
            "predictor_graph_signature",
            lambda batch_size, semantic_positions: signature,
        )
        monkeypatch.setattr(
            sglang_model_module.current_platform,
            "get_device_graph_backend",
            lambda graph_device: object(),
        )
        monkeypatch.setitem(
            talker.__dict__,
            "execute_speculative_opening_pair",
            lambda *args: None,
        )

        def fatal_capture(*args, **kwargs):
            raise RuntimeError("CUDA error 719: unspecified launch failure")

        monkeypatch.setitem(
            talker.__dict__, "capture_code_predictor_spec_graph", fatal_capture
        )
        layer0_codes = torch.zeros((1, 1), device=device, dtype=torch.long)
        talker_hidden = torch.zeros(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.zeros((1,), device=device, dtype=torch.long)

        with pytest.raises(RuntimeError, match="unspecified launch failure"):
            talker.try_replay_speculative_opening_graph(
                layer0_codes, talker_hidden, semantic_positions, workspace
            )

    assert talker.code_predictor_spec_opening_graph_capture_failure_count == 1


def verify_complete_frame(
    talker: Qwen3TTSTalker,
    codes: torch.Tensor,
    summed_embedding: torch.Tensor,
) -> None:
    assert codes.ndim == 3 and codes.shape[1:] == (16, 1)
    assert bool(((codes >= 0) & (codes < CODEBOOK_VOCAB_SIZE)).all())
    assert summed_embedding.shape == (codes.shape[0], 1, TALKER_HIDDEN_SIZE)
    expected_embedding = talker.get_input_embeddings()(codes[:, :1, 0])[:, 0]
    codec_tables = talker.code_predictor.model.codec_embedding.weight
    for codebook_index in range(1, 16):
        expected_embedding.add_(
            codec_tables[codebook_index - 1, codes[:, codebook_index, 0]]
        )
    torch.testing.assert_close(
        summed_embedding[:, 0], expected_embedding, atol=0.0, rtol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_packed_predictor_matches_sequential_native_hidden_and_kv_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        device = torch.device("cuda")
        torch.manual_seed(108)
        talker = build_speculative_talker(device)
        query_embeddings = torch.randn((1, 3, FUSED_HIDDEN), device=device, dtype=DTYPE)
        packed_positions = torch.tensor([2, 3, -1], device=device, dtype=torch.long)
        packed_hiddens = talker.predictor_forward_packed_tokens(
            token_embeds=query_embeddings,
            positions=packed_positions,
            batch_size=1,
        )
        packed_key_prefix = talker.predictor_k_cache[:, :1, :4].clone()
        packed_value_prefix = talker.predictor_v_cache[:, :1, :4].clone()

        talker.predictor_k_cache.zero_()
        talker.predictor_v_cache.zero_()
        native_hiddens = torch.cat(
            [
                talker.predictor_forward_tokens(
                    token_embeds=query_embeddings[:, offset : offset + 1],
                    batch_size=1,
                    cache_len=offset + 2,
                )
                for offset in range(2)
            ],
            dim=1,
        )

        torch.testing.assert_close(
            packed_hiddens[:, :2], native_hiddens, atol=2e-2, rtol=2e-2
        )
        for layer_index, lm_head in enumerate(talker.code_predictor.lm_head[:2]):
            packed_logits, _ = lm_head(packed_hiddens[:, layer_index : layer_index + 1])
            native_logits, _ = lm_head(native_hiddens[:, layer_index : layer_index + 1])
            torch.testing.assert_close(
                packed_logits, native_logits, atol=2e-2, rtol=2e-2
            )
        else:
            pass
        torch.testing.assert_close(
            talker.predictor_k_cache[:, :1, :4],
            packed_key_prefix,
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            talker.predictor_v_cache[:, :1, :4],
            packed_value_prefix,
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            packed_hiddens[:, 1], native_hiddens[:, 1], atol=2e-2, rtol=2e-2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_unseeded_eager_speculation_runs_a_complete_predictor_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(982)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(4, backend="eager")
        layer0_codes = torch.tensor([[9]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([7], device=device, dtype=torch.long)

        codes, summed_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, codes, summed_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == 1
    assert stats["native_fallback_count"] == 0
    assert stats["packed_transformer_forwards"] > 0
    assert stats["generic_predictor_forwards"] == 0
    assert stats["committed_codes"] == 15
    assert stats["round_committed_codes"] == 14


@pytest.mark.parametrize(
    "execution_backend", ["eager", "cuda_graph", "full_round_graph"]
)
@pytest.mark.parametrize("top_k", [4, 50, 128])
@pytest.mark.parametrize("acceptance_backend", ["compact", "dense_indexed"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_full_frame_w1_tail_supports_non_power_of_two_top_k(
    monkeypatch: pytest.MonkeyPatch,
    execution_backend: str,
    top_k: int,
    acceptance_backend: str,
) -> None:
    if execution_backend == "full_round_graph" and acceptance_backend != "compact":
        pytest.skip("full-round graph initially supports compact acceptance only")
    device = torch.device("cuda")
    torch.manual_seed(318)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.sub_top_k_tensor.fill_(top_k)
        talker.sub_sampled_max_top_k = top_k
        probability_backend = "radix_compact" if top_k == 4 else "split_top_k"
        talker.configure_code_predictor_speculation(
            4,
            backend=execution_backend,
            probability_backend=probability_backend,
            acceptance_backend=acceptance_backend,
            opening_pair_graph=execution_backend == "full_round_graph",
        )

        probability_call_count = 0
        candidate_view_call_count = 0
        scratch_view_call_count = 0

        if execution_backend == "full_round_graph":
            original_candidate_view = sglang_model_module.compact_candidate_view
            original_scratch_view = sglang_model_module.compact_scratch_view

            def count_candidate_views(
                storage: torch.Tensor, row_count: int, candidate_count: int
            ) -> torch.Tensor:
                nonlocal candidate_view_call_count
                candidate_view_call_count += 1
                return original_candidate_view(storage, row_count, candidate_count)

            def count_scratch_views(
                storage: torch.Tensor, row_count: int, columns_per_row: int
            ) -> torch.Tensor:
                nonlocal scratch_view_call_count
                scratch_view_call_count += 1
                return original_scratch_view(storage, row_count, columns_per_row)

            monkeypatch.setattr(
                sglang_model_module,
                "compact_candidate_view",
                count_candidate_views,
            )
            monkeypatch.setattr(
                sglang_model_module,
                "compact_scratch_view",
                count_scratch_views,
            )
        else:
            pass

        def controlled_probability_builder(
            logits: torch.Tensor,
            temperatures: torch.Tensor,
            top_ks: torch.Tensor,
            top_ps: torch.Tensor,
            active_rows: torch.Tensor,
            *,
            max_top_k: int,
            has_top_p: bool,
            has_unbounded_top_k: bool,
            backend: str,
            probability_output: torch.Tensor | None = None,
            token_id_output: torch.Tensor | None = None,
            split_candidate_keys: torch.Tensor | None = None,
            radix_candidate_keys: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, str]:
            nonlocal probability_call_count
            assert temperatures.shape == top_ks.shape == top_ps.shape
            assert temperatures.shape == active_rows.shape
            assert has_top_p is False
            assert has_unbounded_top_k is False
            probability_call_count += 1
            candidate_count = fused_raw_logit_block_k(max_top_k)
            assert candidate_count is not None
            if split_candidate_keys is not None:
                assert split_candidate_keys.shape[0] == logits.shape[0]
            else:
                pass
            if radix_candidate_keys is not None:
                assert radix_candidate_keys.shape[0] == logits.shape[0]
            else:
                pass
            if probability_output is None:
                probabilities = torch.zeros(
                    (logits.shape[0], candidate_count),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                probabilities = probability_output
                probabilities.fill_(0.0)
            if token_id_output is None:
                token_ids = torch.full(
                    (logits.shape[0], candidate_count),
                    -1,
                    device=device,
                    dtype=torch.int32,
                )
            else:
                token_ids = token_id_output
                token_ids.fill_(-1)
            token_ids[:, 0] = 0
            if logits.shape[0] == 2:
                token_ids[:, 1] = 1
                probabilities[:, :2] = 0.5
            else:
                probabilities[:, 0] = 1.0
            probabilities[~active_rows] = 0.0
            actual_backend = (
                "split_top_k"
                if backend in ("radix_compact", "radix_compact_4bit")
                and max_top_k <= 32
                else backend
            )
            return probabilities, token_ids, actual_backend

        def fixed_uniforms(
            seeds: torch.Tensor,
            semantic_positions: torch.Tensor,
            round_index: torch.Tensor,
            num_code_groups: int,
            proposal_width: int,
            verify_width: int,
            output: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            assert output is not None
            assert seeds.shape == semantic_positions.shape == (1,)
            assert round_index.ndim == 0
            assert num_code_groups == 16
            output.zero_()
            output[:, 0, :proposal_width] = 0.1
            output[:, 1, :proposal_width] = 0.9
            output[:, 2, :verify_width] = 0.1
            output[:, 3, :verify_width] = 0.1
            return (
                output[:, 0, :proposal_width],
                output[:, 1, :proposal_width],
                output[:, 2, :verify_width],
                output[:, 3, :verify_width],
            )

        monkeypatch.setattr(
            sglang_model_module,
            "build_speculative_probability_candidates_selected",
            controlled_probability_builder,
        )
        monkeypatch.setattr(
            sglang_model_module, "speculative_uniforms_all_purposes", fixed_uniforms
        )
        layer0_codes = torch.tensor([[9]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([7], device=device, dtype=torch.long)

        frame_multiplier = 1
        if execution_backend == "full_round_graph":
            with observe_speculative_rounds(talker) as round_observations:
                codes, summed_embedding = talker.code_predictor_forward(
                    layer0_codes, talker_hidden, semantic_positions
                )
                verify_complete_frame(talker, codes, summed_embedding)
                first_codes = codes.clone()
                first_embedding = summed_embedding.clone()
                first_rounds = list(round_observations)
                warmed_probability_calls = probability_call_count
                warmed_candidate_view_calls = candidate_view_call_count
                warmed_scratch_view_calls = scratch_view_call_count
                repeat_codes, repeat_embedding = talker.code_predictor_forward(
                    layer0_codes, talker_hidden, semantic_positions
                )
                verify_complete_frame(talker, repeat_codes, repeat_embedding)
                repeat_rounds = round_observations[len(first_rounds) :]
                assert len(first_rounds) == len(repeat_rounds) == 5
                for first_round, repeat_round in zip(
                    first_rounds, repeat_rounds, strict=True
                ):
                    assert first_round.verify_width == repeat_round.verify_width
                    assert first_round.committed_count == repeat_round.committed_count
                    assert first_round.accepted_count == repeat_round.accepted_count
                    assert first_round.starting_depth == repeat_round.starting_depth
                    assert first_round.ending_depth == repeat_round.ending_depth
                    assert (
                        first_round.logical_kv_length == repeat_round.logical_kv_length
                    )
                    assert torch.equal(
                        first_round.output_codes[:, : first_round.ending_depth + 1],
                        repeat_round.output_codes[:, : repeat_round.ending_depth + 1],
                    )
                    for field in (
                        "metadata",
                        "pending_tokens",
                        "active_rows",
                        "preceding_hidden",
                        "keys",
                        "values",
                        "logits",
                        "probabilities",
                        "token_ids",
                        "valid_queries",
                        "head_indices",
                        "draft_tokens",
                        "committed_tokens",
                    ):
                        assert torch.equal(
                            getattr(first_round, field), getattr(repeat_round, field)
                        )
                else:
                    pass
                assert torch.equal(first_codes, repeat_codes)
                assert torch.equal(first_embedding, repeat_embedding)
                assert probability_call_count == warmed_probability_calls
                assert candidate_view_call_count == warmed_candidate_view_calls
                assert scratch_view_call_count == warmed_scratch_view_calls
            frame_multiplier = 2
        else:
            codes, summed_embedding = talker.code_predictor_forward(
                layer0_codes, talker_hidden, semantic_positions
            )
            verify_complete_frame(talker, codes, summed_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["nvtx_enabled"] is False
    if execution_backend == "full_round_graph":
        assert probability_call_count > 0
    else:
        assert probability_call_count == 9
    assert stats["frames"] == frame_multiplier
    assert stats["native_fallback_count"] == 0
    assert stats["effective_width_histogram"] == {
        "1": frame_multiplier,
        "2": frame_multiplier,
        "4": 3 * frame_multiplier,
    }
    assert stats["probability_backend_execution_counts"] == {
        "split_top_k": 9 * frame_multiplier,
        "radix_compact": 0,
        "radix_compact_4bit": 0,
    }
    assert stats["probability_backend_fallbacks"] == (
        9 * frame_multiplier if top_k == 4 else 0
    )
    assert stats["acceptance_backend"] == acceptance_backend
    assert stats["acceptance_backend_execution_counts"] == {
        "compact": 5 * frame_multiplier if acceptance_backend == "compact" else 0,
        "dense_indexed": (
            5 * frame_multiplier if acceptance_backend == "dense_indexed" else 0
        ),
    }
    if execution_backend == "eager":
        assert stats["eager_verifier_rounds"] == 5
        assert stats["verifier_graph_replays"] == 0
        assert stats["verifier_graph_replay_coverage"] == 0.0
    elif execution_backend == "cuda_graph":
        assert stats["eager_verifier_rounds"] == 0
        assert stats["verifier_graph_replays"] == 5
        assert stats["verifier_graph_replay_coverage"] == 1.0
    else:
        assert stats["eager_verifier_rounds"] == 0
        assert stats["verifier_graph_replays"] == 0
        assert stats["full_round_graph_replays"] == 5 * frame_multiplier
        assert stats["full_round_graph_replay_coverage"] == 1.0
        assert stats["full_round_graph_captures"] == 3
        assert stats["full_round_graph_capture_failures"] == 0
        assert stats["opening_pair_graph_replays"] == frame_multiplier
        assert stats["opening_pair_graph_replay_coverage"] == 1.0
        assert stats["opening_pair_graph_captures"] == 1
        assert stats["opening_pair_graph_capture_failures"] == 0


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_full_round_graph_rejects_invalid_probability_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(
            4,
            backend="full_round_graph",
            acceptance_backend="compact",
        )

        def invalid_target_probability_builder(
            logits: torch.Tensor,
            temperatures: torch.Tensor,
            top_ks: torch.Tensor,
            top_ps: torch.Tensor,
            active_rows: torch.Tensor,
            *,
            max_top_k: int,
            has_top_p: bool,
            has_unbounded_top_k: bool,
            backend: str,
            probability_output: torch.Tensor | None = None,
            token_id_output: torch.Tensor | None = None,
            split_candidate_keys: torch.Tensor | None = None,
            radix_candidate_keys: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, str]:
            assert probability_output is not None
            assert token_id_output is not None
            assert temperatures.shape == top_ks.shape == top_ps.shape
            assert temperatures.shape == active_rows.shape
            assert max_top_k == 4
            assert has_top_p is False
            assert has_unbounded_top_k is False
            assert split_candidate_keys is None or (
                split_candidate_keys.shape[0] == logits.shape[0]
            )
            assert radix_candidate_keys is None or (
                radix_candidate_keys.shape[0] == logits.shape[0]
            )
            probabilities = probability_output
            token_ids = token_id_output
            probabilities.zero_()
            token_ids.fill_(-1)
            token_ids[:, 0] = 0
            if logits.shape[0] != 4:
                probabilities[:, 0] = 1.0
            else:
                pass
            probabilities[~active_rows] = 0.0
            return probabilities, token_ids, backend

        monkeypatch.setattr(
            sglang_model_module,
            "build_speculative_probability_candidates_selected",
            invalid_target_probability_builder,
        )
        layer0_codes = torch.tensor([[9]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([7], device=device, dtype=torch.long)

        with pytest.raises(RuntimeError, match="invalid probability mass"):
            talker.code_predictor_forward(
                layer0_codes, talker_hidden, semantic_positions
            )

    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == 0
    assert stats["rounds"] == 0
    assert stats["full_round_graph_replays"] == 1
    assert stats["full_round_graph_captures"] == 1
    assert stats["probability_backend_execution_counts"]["split_top_k"] == 2
    assert stats["acceptance_backend_execution_counts"]["compact"] == 1


@pytest.mark.parametrize(
    ("capture_error", "fatal_capture"),
    [
        ("recoverable capture failure", False),
        ("CUDA error 719: unspecified launch failure", True),
    ],
)
@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_full_round_graph_capture_failure_falls_back_or_propagates(
    monkeypatch: pytest.MonkeyPatch,
    capture_error: str,
    fatal_capture: bool,
) -> None:
    device = torch.device("cuda")
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(
            4,
            backend="full_round_graph",
            acceptance_backend="compact",
        )
        talker.code_predictor_spec_opening_pair_graph_enabled = False
        original_opening_pair = talker.execute_speculative_opening_pair
        opening_pair_calls = 0

        def count_opening_pair_calls(
            layer0_codes: torch.Tensor,
            talker_hidden: torch.Tensor,
            semantic_positions: torch.Tensor,
            workspace: PredictorSpeculativeWorkspace,
        ) -> None:
            nonlocal opening_pair_calls
            assert workspace.device == layer0_codes.device
            opening_pair_calls += 1
            original_opening_pair(
                layer0_codes,
                talker_hidden,
                semantic_positions,
                workspace,
            )

        def fail_capture(
            run_once: sglang_model_module.SpeculativeGraphCaptureOperation,
            *,
            graph_pool: sglang_model_module.DeviceGraphPool | None = None,
        ) -> tuple[
            sglang_model_module.ReplayableGraph,
            sglang_model_module.SpeculativeGraphCaptureOutputs,
            float,
            int,
        ]:
            assert callable(run_once)
            assert graph_pool is not None
            raise RuntimeError(capture_error)

        monkeypatch.setitem(
            talker.__dict__,
            "execute_speculative_opening_pair",
            count_opening_pair_calls,
        )
        monkeypatch.setitem(
            talker.__dict__, "capture_code_predictor_spec_graph", fail_capture
        )
        monkeypatch.setattr(
            sglang_model_module.current_platform,
            "get_device_graph_backend",
            lambda graph_device: object(),
        )
        layer0_codes = torch.tensor([[9]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([7], device=device, dtype=torch.long)

        if fatal_capture:
            with pytest.raises(RuntimeError, match="unspecified launch failure"):
                talker.code_predictor_forward(
                    layer0_codes, talker_hidden, semantic_positions
                )
        else:
            codes, summed_embedding = talker.code_predictor_forward(
                layer0_codes, talker_hidden, semantic_positions
            )
            verify_complete_frame(talker, codes, summed_embedding)

    stats = talker.predictor_speculative_stats()
    assert opening_pair_calls == 1
    assert stats["native_fallback_count"] == 0
    assert stats["full_round_graph_replays"] == 0
    if fatal_capture:
        assert stats["frames"] == 0
        assert stats["full_round_graph_capture_failures"] == 1
        assert stats["eager_verifier_rounds"] == 0
    else:
        assert stats["frames"] == 1
        assert stats["full_round_graph_capture_failures"] == len(
            stats["effective_width_histogram"]
        )
        assert stats["rounds"] > 0
        assert stats["eager_verifier_rounds"] == stats["rounds"]


@pytest.mark.parametrize("forced_rejection_offset", [0, 1, 2])
@pytest.mark.parametrize("lm_head_mode", ["zero", "nonzero_depth_distinct"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_forced_code_replay_matches_native_targets_and_valid_kv_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    forced_rejection_offset: int,
    lm_head_mode: str,
) -> None:
    device = torch.device("cuda")
    batch_size = 2
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=batch_size):
        talker = build_speculative_talker(device, batch_size)
        if lm_head_mode == "zero":
            for lm_head in talker.code_predictor.lm_head:
                lm_head.weight.zero_()
            else:
                pass
        else:
            torch.manual_seed(90210)
            for head_index, lm_head in enumerate(talker.code_predictor.lm_head):
                lm_head.weight.normal_(mean=0.0, std=1e-3 * (head_index + 1))
            else:
                pass
            assert all(
                bool((head.weight != 0).any()) for head in talker.code_predictor.lm_head
            )
            assert all(
                not torch.equal(left.weight, right.weight)
                for left, right in zip(
                    talker.code_predictor.lm_head,
                    talker.code_predictor.lm_head[1:],
                    strict=False,
                )
            )
        configure_unseeded_request(talker, batch_size)
        talker.configure_code_predictor_speculation(4, backend="eager")

        native_grouped_head = sglang_model_module.grouped_lm_head_linear
        native_probability_builder = (
            sglang_model_module.build_speculative_probability_candidates_selected
        )
        native_acceptance = sglang_model_module.accept_speculative_candidates_compact
        native_packed_forward = talker.predictor_forward_packed_tokens
        round_verification_widths: list[int] = []
        packed_hiddens: list[torch.Tensor] = []
        packed_keys: list[torch.Tensor] = []
        packed_values: list[torch.Tensor] = []
        proposal_head_hiddens: dict[int, torch.Tensor] = {}
        target_logits: list[torch.Tensor] = []
        target_probabilities: list[torch.Tensor] = []
        target_token_ids: list[torch.Tensor] = []
        draft_tokens_by_round: list[torch.Tensor] = []
        acceptance_records: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []

        def fixed_uniforms(
            seeds: torch.Tensor,
            semantic_positions: torch.Tensor,
            round_index: torch.Tensor,
            num_code_groups: int,
            current_proposal_width: int,
            current_verify_width: int,
            output: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            assert output is not None
            assert seeds.shape == semantic_positions.shape == (batch_size,)
            assert num_code_groups == 16
            assert current_verify_width == current_proposal_width + 1
            output[:batch_size].zero_()
            current_round = int(round_index.item())
            if current_round < 3:
                rejection_offset = forced_rejection_offset if current_round == 0 else 0
                assert rejection_offset < current_proposal_width
                output[1, 1, rejection_offset] = 0.5
            else:
                pass
            return (
                output[:batch_size, 0, :current_proposal_width],
                output[:batch_size, 1, :current_proposal_width],
                output[:batch_size, 2, :current_verify_width],
                output[:batch_size, 3, :current_verify_width],
            )

        def record_packed_forward(
            token_embeds: torch.Tensor,
            positions: torch.Tensor,
            batch_size: int,
        ) -> torch.Tensor:
            verification_width = int(token_embeds.shape[1])
            assert token_embeds.shape[0] == batch_size
            assert positions.shape == (batch_size * verification_width,)
            round_verification_widths.append(verification_width)
            hidden_states = native_packed_forward(
                token_embeds=token_embeds,
                positions=positions,
                batch_size=batch_size,
            )
            packed_hiddens.append(hidden_states.clone())
            packed_keys.append(talker.predictor_k_cache.clone())
            packed_values.append(talker.predictor_v_cache.clone())
            return hidden_states

        def record_grouped_head(
            hidden_states: torch.Tensor,
            packed_weights: torch.Tensor,
            hidden_indices: torch.Tensor | None,
            head_indices: torch.Tensor,
            active_rows: torch.Tensor,
            *,
            backend: str,
            output: torch.Tensor | None = None,
        ) -> torch.Tensor:
            logits = native_grouped_head(
                hidden_states,
                packed_weights,
                hidden_indices,
                head_indices,
                active_rows,
                backend=backend,
                output=output,
            )
            if len(round_verification_widths) > len(target_logits):
                round_index = len(target_logits)
                verification_width = round_verification_widths[round_index]
                assert hidden_states.shape[0] == batch_size * verification_width
                target_logits.append(
                    logits.view(batch_size, verification_width, -1).clone()
                )
            else:
                proposal_head_hiddens[len(round_verification_widths)] = (
                    hidden_states.clone()
                )
            return logits

        def record_probability_builder(
            logits: torch.Tensor,
            temperatures: torch.Tensor,
            top_ks: torch.Tensor,
            top_ps: torch.Tensor,
            active_rows: torch.Tensor,
            *,
            max_top_k: int,
            has_top_p: bool,
            has_unbounded_top_k: bool,
            backend: str,
            probability_output: torch.Tensor | None = None,
            token_id_output: torch.Tensor | None = None,
            split_candidate_keys: torch.Tensor | None = None,
            radix_candidate_keys: torch.Tensor | None = None,
        ) -> (
            tuple[
                torch.Tensor,
                torch.Tensor,
                Literal["split_top_k", "radix_compact", "radix_compact_4bit"],
            ]
            | None
        ):
            candidates = native_probability_builder(
                logits,
                temperatures,
                top_ks,
                top_ps,
                active_rows,
                max_top_k=max_top_k,
                has_top_p=has_top_p,
                has_unbounded_top_k=has_unbounded_top_k,
                backend=backend,
                probability_output=probability_output,
                token_id_output=token_id_output,
                split_candidate_keys=split_candidate_keys,
                radix_candidate_keys=radix_candidate_keys,
            )
            if candidates is None:
                return None
            else:
                pass
            probabilities, token_ids, actual_backend = candidates
            if len(round_verification_widths) > len(target_probabilities):
                round_index = len(target_probabilities)
                verification_width = round_verification_widths[round_index]
                assert logits.shape[0] == batch_size * verification_width
                target_probabilities.append(
                    probabilities.view(batch_size, verification_width, -1).clone()
                )
                target_token_ids.append(
                    token_ids.view(batch_size, verification_width, -1).clone()
                )
                return candidates
            else:
                assert logits.shape[0] % batch_size == 0
                assert logits.shape[0] > 0
                controlled_probabilities = probabilities
                controlled_probabilities[active_rows] = 0.0
                controlled_probabilities[active_rows, 0] = 1.0
                assert not bool((controlled_probabilities[~active_rows] != 0.0).any())
                return controlled_probabilities, token_ids, actual_backend

        def record_acceptance(
            draft_tokens: torch.Tensor,
            draft_probabilities: torch.Tensor,
            draft_token_ids: torch.Tensor,
            row_target_probabilities: torch.Tensor,
            row_target_token_ids: torch.Tensor,
            proposal_counts: torch.Tensor,
            active_rows: torch.Tensor,
            acceptance_uniforms: torch.Tensor,
            correction_uniforms: torch.Tensor,
            bonus_uniforms: torch.Tensor,
            outputs: SpeculativeAcceptanceBatch | None = None,
        ) -> SpeculativeAcceptanceBatch:
            round_index = len(acceptance_records)
            verification_width = round_verification_widths[round_index]
            current_proposal_width = verification_width - 1
            assert draft_tokens.shape == (batch_size, current_proposal_width)
            assert row_target_probabilities.shape[0] == batch_size * verification_width
            assert all(
                bool(((uniforms >= 0.0) & (uniforms < 1.0)).all())
                for uniforms in (
                    acceptance_uniforms,
                    correction_uniforms,
                    bonus_uniforms,
                )
            )
            if lm_head_mode == "zero" and round_index < 3:
                rejection_offset = forced_rejection_offset if round_index == 0 else 0
                assert bool(active_rows[1])
                assert int(proposal_counts[1].item()) > rejection_offset
                draft_row = current_proposal_width + rejection_offset
                target_row = verification_width + rejection_offset
                draft_token = draft_tokens[1, rejection_offset]
                draft_token_probability = draft_probabilities[draft_row][
                    draft_token_ids[draft_row] == draft_token
                ].sum()
                target_token_probability = row_target_probabilities[target_row][
                    row_target_token_ids[target_row] == draft_token
                ].sum()
                assert not torch.equal(
                    draft_probabilities[draft_row],
                    row_target_probabilities[target_row],
                )
                assert float(draft_token_probability.item()) == 1.0
                assert float(target_token_probability.item()) < float(
                    draft_token_probability.item()
                )
                acceptance_ratio = min(
                    1.0,
                    float(target_token_probability.item())
                    / float(draft_token_probability.item()),
                )
                rejection_uniform = float(
                    acceptance_uniforms[1, rejection_offset].item()
                )
                assert acceptance_ratio < rejection_uniform < 1.0
            else:
                pass
            decision = native_acceptance(
                draft_tokens,
                draft_probabilities,
                draft_token_ids,
                row_target_probabilities,
                row_target_token_ids,
                proposal_counts,
                active_rows,
                acceptance_uniforms,
                correction_uniforms,
                bonus_uniforms,
                outputs=outputs,
            )

            # Independent acceptance oracle. The fixed test tape uses
            # zero correction/bonus uniforms, so inverse-CDF selects
            # the first positive mass in target candidate order.
            assert bool((correction_uniforms == 0).all())
            assert bool((bonus_uniforms == 0).all())
            assert not bool(decision.invalid_rows.any())

            for row in range(batch_size):
                if not bool(active_rows[row]):
                    assert int(decision.accepted_draft_counts[row]) == 0
                    assert int(decision.committed_token_counts[row]) == 0
                    assert not bool(decision.rejected_rows[row])
                    assert not bool(decision.bonus_rows[row])
                    continue

                proposal_count = int(proposal_counts[row])
                expected_accepted = 0
                expected_tokens = []
                rejected = False
                rejected_q = {}

                for offset in range(proposal_count):
                    q_row = row * current_proposal_width + offset
                    p_row = row * verification_width + offset

                    q = {
                        token: mass
                        for token, mass in zip(
                            draft_token_ids[q_row].tolist(),
                            draft_probabilities[q_row].tolist(),
                            strict=True,
                        )
                        if mass > 0.0
                    }
                    p_dist = {
                        token: mass
                        for token, mass in zip(
                            row_target_token_ids[p_row].tolist(),
                            row_target_probabilities[p_row].tolist(),
                            strict=True,
                        )
                        if mass > 0.0
                    }

                    token = int(draft_tokens[row, offset])
                    q_mass = q.get(token, 0.0)
                    p_mass = p_dist.get(token, 0.0)

                    assert q_mass > 0.0

                    ratio = min(1.0, p_mass / q_mass)
                    uniform = float(acceptance_uniforms[row, offset])

                    if uniform >= ratio:
                        rejected = True
                        rejected_q = q
                        break

                    expected_tokens.append(token)
                    expected_accepted += 1

                selected_row = row * verification_width + expected_accepted

                final_candidates = []
                for token, mass in zip(
                    row_target_token_ids[selected_row].tolist(),
                    row_target_probabilities[selected_row].tolist(),
                    strict=True,
                ):
                    residual = (
                        max(mass - rejected_q.get(token, 0.0), 0.0)
                        if rejected
                        else mass
                    )
                    if residual > 0.0:
                        final_candidates.append(token)

                assert final_candidates
                expected_tokens.append(final_candidates[0])

                assert int(decision.accepted_draft_counts[row]) == expected_accepted
                assert int(decision.committed_token_counts[row]) == len(expected_tokens)
                assert bool(decision.rejected_rows[row]) == rejected
                assert bool(decision.bonus_rows[row]) == (not rejected)
                assert (
                    decision.committed_tokens[row, : len(expected_tokens)].tolist()
                    == expected_tokens
                )

            draft_tokens_by_round.append(draft_tokens.clone())
            acceptance_records.append(
                (
                    proposal_counts.clone(),
                    active_rows.clone(),
                    decision.accepted_draft_counts.clone(),
                    decision.committed_token_counts.clone(),
                    decision.committed_tokens.clone(),
                    decision.bonus_rows.clone(),
                    decision.rejected_rows.clone(),
                )
            )
            return decision

        monkeypatch.setattr(
            sglang_model_module, "speculative_uniforms_all_purposes", fixed_uniforms
        )
        monkeypatch.setitem(
            talker.__dict__,
            "predictor_forward_packed_tokens",
            record_packed_forward,
        )
        monkeypatch.setattr(
            sglang_model_module, "grouped_lm_head_linear", record_grouped_head
        )
        monkeypatch.setattr(
            sglang_model_module,
            "build_speculative_probability_candidates_selected",
            record_probability_builder,
        )
        monkeypatch.setattr(
            sglang_model_module,
            "accept_speculative_candidates_compact",
            record_acceptance,
        )

        layer0_codes = torch.tensor([[9], [17]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (batch_size, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([7, 11], device=device, dtype=torch.long)
        codes, summed_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        codes = codes.clone()
        verify_complete_frame(talker, codes, summed_embedding)

        assert len(round_verification_widths) == len(acceptance_records)
        assert len(round_verification_widths) == len(packed_hiddens)
        assert len(round_verification_widths) == len(target_logits)
        assert len(round_verification_widths) == len(target_probabilities)
        assert len(round_verification_widths) == len(target_token_ids)
        assert len(round_verification_widths) == len(draft_tokens_by_round)
        assert len(round_verification_widths) >= 3
        first_proposal_width = round_verification_widths[0] - 1
        assert forced_rejection_offset < first_proposal_width
        if lm_head_mode == "zero":
            for round_index in range(3):
                (
                    proposal_counts,
                    _,
                    accepted_counts,
                    _,
                    _,
                    bonus_rows,
                    rejected_rows,
                ) = acceptance_records[round_index]
                forced_offset = forced_rejection_offset if round_index == 0 else 0
                assert rejected_rows.tolist() == [False, True]
                assert accepted_counts.tolist() == [
                    int(proposal_counts[0].item()),
                    forced_offset,
                ]
                assert bonus_rows.tolist() == [True, False]

        talker.predictor_k_cache.zero_()
        talker.predictor_v_cache.zero_()
        layer0_embeddings = talker.get_input_embeddings()(layer0_codes).to(
            dtype=talker.predictor_k_cache.dtype
        )
        pair_embeddings = talker.code_predictor.project_input(
            torch.cat(
                (talker_hidden.to(dtype=layer0_embeddings.dtype), layer0_embeddings),
                dim=1,
            )
        )
        initial_hidden_states = talker.predictor_forward_tokens(
            token_embeds=pair_embeddings, batch_size=batch_size, cache_len=0
        )
        native_initial_hiddens = initial_hidden_states[:, 1].clone()
        native_hiddens: dict[int, torch.Tensor] = {}
        native_logits: dict[int, torch.Tensor] = {}
        native_keys: dict[int, torch.Tensor] = {
            0: talker.predictor_k_cache[:, :, :2].clone()
        }
        native_values: dict[int, torch.Tensor] = {
            0: talker.predictor_v_cache[:, :, :2].clone()
        }
        for code_depth in range(1, 15):
            code_tokens = codes[:, code_depth, 0]
            projected_embeddings = talker.predictor_projected_embeddings[
                code_depth - 1, code_tokens
            ].unsqueeze(1)
            hidden_states = talker.predictor_forward_tokens(
                token_embeds=projected_embeddings,
                batch_size=batch_size,
                cache_len=code_depth + 1,
            )
            logits, _ = talker.code_predictor.lm_head[code_depth](hidden_states)
            native_hiddens[code_depth] = hidden_states[:, 0].clone()
            native_logits[code_depth] = logits[:, 0].clone()
            valid_cache_length = code_depth + 2
            native_keys[code_depth] = talker.predictor_k_cache[
                :, :, :valid_cache_length
            ].clone()
            native_values[code_depth] = talker.predictor_v_cache[
                :, :, :valid_cache_length
            ].clone()
        else:
            pass

        committed_depths = [1, 1]
        starting_depths_by_round: list[list[int]] = []
        for round_index, decision_record in enumerate(acceptance_records):
            (
                proposal_counts,
                active_rows,
                accepted_counts,
                committed_counts,
                committed_tokens,
                bonus_rows,
                rejected_rows,
            ) = decision_record
            verification_width = round_verification_widths[round_index]
            current_proposal_width = verification_width - 1
            assert bool((proposal_counts <= current_proposal_width).all())
            starting_depths_by_round.append(list(committed_depths))
            for row in range(batch_size):
                starting_depth = committed_depths[row]
                accepted_count = int(accepted_counts[row].item())
                commit_count = int(committed_counts[row].item())
                if active_rows[row]:
                    expected_codes = codes[
                        row,
                        starting_depth + 1 : starting_depth + 1 + commit_count,
                        0,
                    ]
                    torch.testing.assert_close(
                        committed_tokens[row, :commit_count], expected_codes
                    )
                    for query_offset in range(accepted_count + 1):
                        code_depth = starting_depth + query_offset
                        torch.testing.assert_close(
                            packed_hiddens[round_index][row, query_offset],
                            native_hiddens[code_depth][row],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                        torch.testing.assert_close(
                            target_logits[round_index][row, query_offset],
                            native_logits[code_depth][row],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                    else:
                        pass
                else:
                    pass
                new_depth = starting_depth + commit_count
                valid_cache_length = new_depth + 1
                expected_cache_depth = new_depth - 1
                torch.testing.assert_close(
                    packed_keys[round_index][:, row : row + 1, :valid_cache_length],
                    native_keys[expected_cache_depth][:, row : row + 1],
                    atol=2e-2,
                    rtol=2e-2,
                )
                torch.testing.assert_close(
                    packed_values[round_index][:, row : row + 1, :valid_cache_length],
                    native_values[expected_cache_depth][:, row : row + 1],
                    atol=2e-2,
                    rtol=2e-2,
                )
                committed_depths[row] = new_depth
            else:
                pass
        else:
            pass

        assert committed_depths == [15, 15]
        assert len(proposal_head_hiddens) == sum(
            verification_width > 1 for verification_width in round_verification_widths
        )
        for round_index, decision_record in enumerate(acceptance_records):
            (
                proposal_counts,
                active_rows,
                _,
                _,
                _,
                _,
                _,
            ) = decision_record
            current_proposal_width = round_verification_widths[round_index] - 1
            if current_proposal_width > 0:
                proposal_hidden_states = proposal_head_hiddens[round_index]
                if round_index == 0:
                    expected_proposal_hiddens = native_initial_hiddens
                else:
                    starting_depths = starting_depths_by_round[round_index]
                    expected_proposal_hiddens = torch.stack(
                        [
                            native_hiddens[starting_depths[row] - 1][row]
                            for row in range(batch_size)
                        ]
                    )
                    previous_accepted_counts = acceptance_records[round_index - 1][2]
                for row in range(batch_size):
                    if active_rows[row]:
                        torch.testing.assert_close(
                            proposal_hidden_states[row],
                            expected_proposal_hiddens[row],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                        if round_index > 0:
                            torch.testing.assert_close(
                                proposal_hidden_states[row],
                                packed_hiddens[round_index - 1][
                                    row, int(previous_accepted_counts[row].item())
                                ],
                                atol=2e-2,
                                rtol=2e-2,
                            )
                        else:
                            pass
                    else:
                        pass
            else:
                pass

            verification_width = round_verification_widths[round_index]
            assert target_logits[round_index].shape[1] == verification_width
            assert target_probabilities[round_index].shape[1] == verification_width
            start_depths = starting_depths_by_round[round_index]
            for row in range(batch_size):
                if active_rows[row]:
                    starting_depth = start_depths[row]
                    proposal_count = int(proposal_counts[row].item())
                    talker.predictor_k_cache.zero_()
                    talker.predictor_v_cache.zero_()
                    row_pair_embeddings = talker.code_predictor.project_input(
                        torch.cat(
                            (
                                talker_hidden[row : row + 1].to(
                                    dtype=layer0_embeddings.dtype
                                ),
                                layer0_embeddings[row : row + 1],
                            ),
                            dim=1,
                        )
                    )
                    talker.predictor_forward_tokens(
                        token_embeds=row_pair_embeddings,
                        batch_size=1,
                        cache_len=0,
                    )
                    for code_depth in range(1, starting_depth):
                        code_token = codes[row, code_depth, 0]
                        projected_embeddings = talker.predictor_projected_embeddings[
                            code_depth - 1, code_token
                        ].view(1, 1, -1)
                        talker.predictor_forward_tokens(
                            token_embeds=projected_embeddings,
                            batch_size=1,
                            cache_len=code_depth + 1,
                        )
                    else:
                        pass
                    for query_offset in range(proposal_count + 1):
                        code_depth = starting_depth + query_offset
                        if query_offset == 0:
                            query_token = codes[row, starting_depth, 0]
                        else:
                            query_token = draft_tokens_by_round[round_index][
                                row, query_offset - 1
                            ]
                        projected_embeddings = talker.predictor_projected_embeddings[
                            code_depth - 1, query_token
                        ].view(1, 1, -1)
                        sequential_hidden = talker.predictor_forward_tokens(
                            token_embeds=projected_embeddings,
                            batch_size=1,
                            cache_len=code_depth + 1,
                        )
                        sequential_logits, _ = talker.code_predictor.lm_head[
                            code_depth
                        ](sequential_hidden)
                        torch.testing.assert_close(
                            packed_hiddens[round_index][row, query_offset],
                            sequential_hidden[0, 0],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                        torch.testing.assert_close(
                            target_logits[round_index][row, query_offset],
                            sequential_logits[0, 0],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                        expected_candidates = build_speculative_probability_candidates(
                            sequential_logits.reshape(1, -1).contiguous(),
                            talker.sub_temperature_tensor[row : row + 1].contiguous(),
                            talker.sub_top_k_tensor[row : row + 1].contiguous(),
                            talker.sub_top_p_tensor[row : row + 1].contiguous(),
                            torch.ones((1,), device=device, dtype=torch.bool),
                            max_top_k=int(talker.sub_sampled_max_top_k),
                            has_top_p=bool(talker.sub_sampled_has_top_p),
                            has_unbounded_top_k=bool(
                                talker.sub_sampled_has_unbounded_top_k
                            ),
                        )
                        assert expected_candidates is not None
                        expected_probabilities, expected_token_ids = expected_candidates
                        torch.testing.assert_close(
                            target_probabilities[round_index][row, query_offset],
                            expected_probabilities[0],
                            atol=1e-6,
                            rtol=1e-6,
                        )
                        assert torch.equal(
                            target_token_ids[round_index][row, query_offset],
                            expected_token_ids[0],
                        )
                        valid_cache_length = code_depth + 2
                        torch.testing.assert_close(
                            packed_keys[round_index][
                                :, row : row + 1, :valid_cache_length
                            ],
                            talker.predictor_k_cache[:, :1, :valid_cache_length],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                        torch.testing.assert_close(
                            packed_values[round_index][
                                :, row : row + 1, :valid_cache_length
                            ],
                            talker.predictor_v_cache[:, :1, :valid_cache_length],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                    else:
                        pass
                else:
                    assert int(proposal_counts[row].item()) == 0
            else:
                pass

        stats = talker.predictor_speculative_stats()
        assert stats["frames"] == batch_size
        expected_rejections = sum(
            int(record[6].sum().item()) for record in acceptance_records
        )
        assert stats["rejections"] == expected_rejections
        if lm_head_mode == "zero":
            assert expected_rejections == 3
        assert stats["standalone_correction_forward_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_packed_predictor_masks_positive_positions_past_physical_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=2):
        talker = build_speculative_talker(device, batch_size=2)
        cache_sentinel = 37.0
        talker.predictor_k_cache.fill_(cache_sentinel)
        talker.predictor_v_cache.fill_(cache_sentinel)
        cache_size = talker.predictor_k_cache.shape[2]
        query_embeddings = torch.randn((2, 4, FUSED_HIDDEN), device=device, dtype=DTYPE)
        query_positions = torch.tensor(
            [
                cache_size - 1,
                cache_size,
                cache_size + 1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            device=device,
            dtype=torch.long,
        )

        talker.predictor_forward_packed_tokens(
            token_embeds=query_embeddings,
            positions=query_positions,
            batch_size=2,
        )
        assert torch.all(talker.predictor_k_cache[:, 1] == cache_sentinel)
        assert torch.all(talker.predictor_v_cache[:, 1] == cache_sentinel)
        assert torch.all(talker.predictor_k_cache[:, 0, :-1] == cache_sentinel)
        assert torch.all(talker.predictor_v_cache[:, 0, :-1] == cache_sentinel)
        assert torch.any(
            talker.predictor_k_cache[:, 0, cache_size - 1] != cache_sentinel
        )
        assert torch.any(
            talker.predictor_v_cache[:, 0, cache_size - 1] != cache_sentinel
        )
        with pytest.raises(ValueError, match="align with query rows"):
            talker.predictor_forward_packed_tokens(
                token_embeds=query_embeddings,
                positions=query_positions[:-1],
                batch_size=2,
            )
        with pytest.raises(ValueError, match="integer dtype"):
            talker.predictor_forward_packed_tokens(
                token_embeds=query_embeddings,
                positions=query_positions.to(torch.float32),
                batch_size=2,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_speculative_verifier_graph_replays_the_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(618)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(4, backend="eager")
        layer0_codes = torch.tensor([[14]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([12], device=device, dtype=torch.long)
        eager_codes, eager_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        eager_codes = eager_codes.clone()
        eager_embedding = eager_embedding.clone()
        talker.configure_code_predictor_speculation(4, backend="cuda_graph")
        assert talker.code_predictor_spec_graphs
        assert all(
            graph.token_embeds.shape[-1] == FUSED_HIDDEN
            for graph in talker.code_predictor_spec_graphs.values()
        )
        graph_codes, graph_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, graph_codes, graph_embedding)

    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embedding, eager_embedding)
    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == 2
    assert stats["native_fallback_count"] == 0
    assert stats["graph_captures"] > 0
    assert stats["graph_replays"] > 0
    assert stats["packed_transformer_forwards"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_verifier_graph_stats_report_mixed_eager_and_graph_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(619)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(4, backend="cuda_graph")
        native_resolver = talker.resolve_code_predictor_speculative_round_path
        eager_rounds_remaining = 1

        def resolve_one_round_eagerly(
            batch_size: int, requested_width: int
        ) -> tuple[int, object | None]:
            nonlocal eager_rounds_remaining
            if eager_rounds_remaining > 0:
                eager_rounds_remaining -= 1
                return requested_width, None
            return native_resolver(batch_size, requested_width)

        monkeypatch.setitem(
            talker.__dict__,
            "resolve_code_predictor_speculative_round_path",
            resolve_one_round_eagerly,
        )
        layer0_codes = torch.tensor([[14]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([12], device=device, dtype=torch.long)
        codes, summed_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, codes, summed_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["eager_verifier_rounds"] >= 1
    assert stats["verifier_graph_replays"] >= 1
    assert 0.0 < stats["verifier_graph_replay_coverage"] < 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_explicit_seed_fallback_does_not_block_later_unseeded_speculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(771)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        talker.configure_code_predictor_speculation(4, backend="eager")
        layer0_codes = torch.tensor([[3]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([5], device=device, dtype=torch.long)
        talker.sub_user_seed_rows = (True,)
        seeded_codes, seeded_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, seeded_codes, seeded_embedding)
        configure_unseeded_request(talker)
        unseeded_codes, unseeded_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, unseeded_codes, unseeded_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == 1
    assert stats["native_fallback_count"] == 1
    assert stats["native_fallback_fraction"] == 0.5
    assert stats["packed_transformer_forwards"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_mixed_explicit_seed_batch_falls_back_then_unseeded_batch_speculates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(913)
    configure_runtime_context(monkeypatch)
    batch_size = 2
    with get_context().override_server_args(max_running_requests=batch_size):
        talker = build_speculative_talker(device, batch_size)
        talker.configure_code_predictor_speculation(4, backend="eager")
        talker.sub_user_seed_rows = (False, True)
        layer0_codes = torch.tensor([[3], [14]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (batch_size, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([5, 6], device=device, dtype=torch.long)

        mixed_codes, mixed_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, mixed_codes, mixed_embedding)
        fallback_stats = talker.predictor_speculative_stats()
        assert fallback_stats["frames"] == 0
        assert fallback_stats["native_fallback_count"] == batch_size
        assert fallback_stats["native_fallback_fraction"] == 1.0

        configure_unseeded_request(talker, batch_size)
        unseeded_codes, unseeded_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, unseeded_codes, unseeded_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == batch_size
    assert stats["native_fallback_count"] == batch_size
    assert stats["native_fallback_fraction"] == 0.5
    assert stats["packed_transformer_forwards"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_speculative_frozen_frame_smoke_executes_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(2011)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        layer0_codes = torch.tensor([[6]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (1, 1, TALKER_HIDDEN_SIZE), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([19], device=device, dtype=torch.long)
        talker.configure_code_predictor_speculation(4, backend="eager")
        workspace_pointers: tuple[int, ...] | None = None
        for _ in range(4):
            talker.code_predictor_forward(
                layer0_codes, talker_hidden, semantic_positions
            )
            workspace = talker.code_predictor_spec_active_workspace
            assert workspace is not None
            round_workspace = workspace.rounds[4]
            current_workspace_pointers = (
                round_workspace.proposal_logits.data_ptr(),
                round_workspace.target_logits.data_ptr(),
                round_workspace.proposal_probabilities.data_ptr(),
                round_workspace.target_probabilities.data_ptr(),
                round_workspace.proposal_tokens.data_ptr(),
                round_workspace.acceptance_outputs.committed_tokens.data_ptr(),
            )
            if workspace_pointers is None:
                workspace_pointers = current_workspace_pointers
            else:
                assert current_workspace_pointers == workspace_pointers
        else:
            pass
        speculative_stats = talker.predictor_speculative_stats()
        if speculative_stats["frames"] <= 0:
            raise AssertionError("speculative frame smoke executed zero frames")
        else:
            pass
        if speculative_stats["native_fallback_count"] != 0:
            raise AssertionError("speculative frame smoke executed native fallback")
        else:
            pass
        assert speculative_stats["frames"] == 4
        assert speculative_stats["workspace_allocations"] == 1
        assert speculative_stats["workspace_bytes"] > 0


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_acceptance_matches_dense_reference_at_every_rejection_position() -> (
    None
):
    device = torch.device("cuda")
    batch_size = 6
    draft_width = 3
    verify_width = draft_width + 1
    candidate_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    draft_probabilities = (
        torch.tensor([0.7, 0.1, 0.1, 0.1], device=device, dtype=torch.float32)
        .expand(batch_size * draft_width, -1)
        .contiguous()
    )
    draft_token_ids = candidate_ids.expand(batch_size * draft_width, -1).contiguous()
    target_probabilities = (
        torch.tensor([0.9, 0.05, 0.03, 0.02], device=device, dtype=torch.float32)
        .expand(batch_size * verify_width, -1)
        .clone()
    )
    target_token_ids = candidate_ids.expand(batch_size * verify_width, -1).clone()
    for row in range(draft_width):
        target_row = row * verify_width + row
        target_probabilities[target_row] = 0.25
        target_token_ids[target_row] = torch.tensor(
            [4, 5, 6, 7], device=device, dtype=torch.long
        )
    else:
        pass
    target_probabilities[3 * verify_width + draft_width] = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float32
    )
    target_token_ids[3 * verify_width + draft_width] = torch.tensor(
        [4, 5, 6, 7], device=device, dtype=torch.long
    )
    target_probabilities[4 * verify_width + 1] = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float32
    )
    target_token_ids[4 * verify_width + 1] = torch.tensor(
        [4, 5, 6, 7], device=device, dtype=torch.long
    )
    draft_tokens = torch.zeros(
        (batch_size, draft_width), device=device, dtype=torch.long
    )
    proposal_counts = torch.tensor([3, 3, 3, 3, 1, 0], device=device, dtype=torch.long)
    active_rows = torch.tensor([True, True, True, True, True, False], device=device)
    uniform_storage = torch.zeros(
        (batch_size, 4, verify_width), device=device, dtype=torch.float32
    )
    acceptance_uniforms = uniform_storage[:, 1, :draft_width]
    for row in range(draft_width):
        acceptance_uniforms[row, row] = 0.5
    else:
        pass
    correction_uniforms = uniform_storage[:, 2, :verify_width]
    bonus_uniforms = uniform_storage[:, 3, :verify_width]
    bonus_uniforms[3, draft_width] = 0.35
    bonus_uniforms[4, 1] = 0.25

    compact_result = accept_speculative_candidates_compact(
        draft_tokens,
        draft_probabilities,
        draft_token_ids,
        target_probabilities,
        target_token_ids,
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
    )
    dense_result = accept_speculative_candidates(
        draft_tokens,
        expand_compact_probability_rows(
            draft_probabilities, draft_token_ids, vocab_size=8
        ).view(batch_size, draft_width, 8),
        expand_compact_probability_rows(
            target_probabilities, target_token_ids, vocab_size=8
        ).view(batch_size, verify_width, 8),
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
    )

    assert torch.equal(compact_result.committed_tokens, dense_result.committed_tokens)
    assert torch.equal(
        compact_result.accepted_draft_counts, dense_result.accepted_draft_counts
    )
    assert torch.equal(
        compact_result.committed_token_counts, dense_result.committed_token_counts
    )
    assert torch.equal(compact_result.rejected_rows, dense_result.rejected_rows)
    assert torch.equal(compact_result.bonus_rows, dense_result.bonus_rows)
    assert not bool(compact_result.invalid_rows.any())
    assert compact_result.accepted_draft_counts[:draft_width].tolist() == [0, 1, 2]
    assert compact_result.committed_tokens[3, -1].item() == 6
    assert compact_result.accepted_draft_counts[4:].tolist() == [1, 0]
    assert compact_result.committed_token_counts[4:].tolist() == [2, 0]
    assert compact_result.committed_tokens[4, 1].item() == 5


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_acceptance_w1_samples_target_and_flags_invalid_mass() -> None:
    device = torch.device("cuda")
    target_probabilities = torch.tensor(
        [[0.4, 0.6, 0.0, 0.0], [0.2, 0.3, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    target_token_ids = torch.tensor(
        [[5, 6, -1, -1], [5, 6, -1, -1]], device=device, dtype=torch.long
    )
    result = accept_speculative_candidates_compact(
        torch.empty((2, 0), device=device, dtype=torch.long),
        torch.empty((0, 4), device=device, dtype=torch.float32),
        torch.empty((0, 4), device=device, dtype=torch.long),
        target_probabilities,
        target_token_ids,
        torch.zeros((2,), device=device, dtype=torch.long),
        torch.tensor([True, True], device=device),
        torch.empty((2, 0), device=device, dtype=torch.float32),
        torch.zeros((2, 1), device=device, dtype=torch.float32),
        torch.tensor([[0.99999], [0.5]], device=device, dtype=torch.float32),
    )

    assert result.committed_tokens.shape == (2, 1)
    assert result.committed_tokens[0, 0].item() == 6
    assert result.committed_token_counts.tolist() == [1, 0]
    assert result.invalid_rows.tolist() == [False, True]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("target_probabilities", "dense_target_values"),
    [
        ([0.25, 0.25], [0.5, 0.5]),
        ([0.5, 0.5], [0.0, 0.0]),
    ],
)
def test_dense_indexed_w1_invalid_target_commits_nothing(
    target_probabilities: list[float], dense_target_values: list[float]
) -> None:
    device = torch.device("cuda")
    candidate_count = 2
    target_probs = torch.tensor(
        [target_probabilities], device=device, dtype=torch.float32
    )
    target_ids = torch.tensor([[0, 1]], device=device, dtype=torch.long)
    dense_target = torch.zeros((1, 1, 8), device=device, dtype=torch.float32)
    dense_target[0, 0, :2] = torch.tensor(
        dense_target_values, device=device, dtype=torch.float32
    )
    result = accept_speculative_candidates_dense_indexed(
        torch.empty((1, 0), device=device, dtype=torch.long),
        torch.empty((0, candidate_count), device=device, dtype=torch.float32),
        torch.empty((0, candidate_count), device=device, dtype=torch.long),
        target_probs,
        target_ids,
        torch.zeros((1,), device=device, dtype=torch.long),
        torch.ones((1,), device=device, dtype=torch.bool),
        torch.empty((1, 0), device=device, dtype=torch.float32),
        torch.zeros((1, 1), device=device, dtype=torch.float32),
        torch.full((1, 1), 0.5, device=device, dtype=torch.float32),
        torch.empty((1, 0, 8), device=device, dtype=torch.float32),
        dense_target,
        torch.arange(candidate_count, device=device, dtype=torch.long),
    )

    assert result.invalid_rows.tolist() == [True]
    assert result.committed_token_counts.tolist() == [0]
    assert result.committed_tokens.tolist() == [[0]]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_radix_probability_selection_reports_split_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    logits = torch.zeros((1, CODEBOOK_VOCAB_SIZE), device=device, dtype=DTYPE)
    temperatures = torch.ones((1,), device=device, dtype=torch.float32)
    top_ks = torch.full((1,), 4, device=device, dtype=torch.long)
    top_ps = torch.ones((1,), device=device, dtype=torch.float32)
    active_rows = torch.ones((1,), device=device, dtype=torch.bool)
    candidate_count = fused_raw_logit_block_k(4)
    assert candidate_count == 32
    split_probabilities = torch.zeros(
        (1, candidate_count), device=device, dtype=torch.float32
    )
    split_probabilities[:, 0] = 1.0
    split_token_ids = torch.full(
        (1, candidate_count), -1, device=device, dtype=torch.int32
    )
    split_token_ids[:, 0] = 0

    def split_builder(
        candidate_logits: torch.Tensor,
        candidate_temperatures: torch.Tensor,
        candidate_top_ks: torch.Tensor,
        candidate_top_ps: torch.Tensor,
        candidate_active_rows: torch.Tensor,
        *,
        max_top_k: int,
        has_top_p: bool,
        has_unbounded_top_k: bool,
        probability_output: torch.Tensor | None = None,
        token_id_output: torch.Tensor | None = None,
        split_candidate_keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert candidate_logits is logits
        assert candidate_temperatures is temperatures
        assert candidate_top_ks is top_ks
        assert candidate_top_ps is top_ps
        assert candidate_active_rows is active_rows
        assert max_top_k == 4
        assert has_top_p is False
        assert has_unbounded_top_k is False
        assert probability_output is None
        assert token_id_output is None
        assert split_candidate_keys is None
        return split_probabilities, split_token_ids

    monkeypatch.setattr(
        sampling_kernels_module,
        "build_speculative_probability_candidates",
        split_builder,
    )
    candidates = (
        sampling_kernels_module.build_speculative_probability_candidates_selected(
            logits,
            temperatures,
            top_ks,
            top_ps,
            active_rows,
            max_top_k=4,
            has_top_p=False,
            has_unbounded_top_k=False,
            backend="radix_compact",
        )
    )

    assert candidates is not None
    assert candidates[0] is split_probabilities
    assert candidates[1] is split_token_ids
    assert candidates[2] == "split_top_k"


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_residual_normalizes_subnormal_probability_mass() -> None:
    device = torch.device("cuda")
    draft_probabilities = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32
    )
    draft_token_ids = torch.tensor([[0, -1, -1, -1]], device=device, dtype=torch.long)
    target_probabilities = torch.tensor(
        [
            [0.99999994, 0.5e-40, 1.5e-40, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        device=device,
        dtype=torch.float32,
    )
    target_token_ids = torch.tensor([[0, 1, 2, -1], [3, 4, 5, 6]], device=device)
    result = accept_speculative_candidates_compact(
        torch.tensor([[0]], device=device, dtype=torch.long),
        draft_probabilities,
        draft_token_ids,
        target_probabilities,
        target_token_ids,
        torch.ones((1,), device=device, dtype=torch.long),
        torch.ones((1,), device=device, dtype=torch.bool),
        torch.tensor([[0.99999994]], device=device, dtype=torch.float32),
        torch.tensor([[0.1, 0.1]], device=device, dtype=torch.float32),
        torch.tensor([[0.1, 0.1]], device=device, dtype=torch.float32),
    )

    assert result.committed_tokens[0, 0].item() == 1
    assert result.rejected_rows.tolist() == [True]
    assert not bool(result.invalid_rows.any())


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_acceptance_top_k50_padding_and_unsorted_support_match_dense() -> None:
    device = torch.device("cuda")
    batch_size = 2
    draft_width = 1
    verify_width = 2
    candidate_count = 64
    draft_probabilities = torch.zeros(
        (batch_size, candidate_count), device=device, dtype=torch.float32
    )
    draft_token_ids = torch.full(
        (batch_size, candidate_count), 5, device=device, dtype=torch.long
    )
    draft_probabilities[0, 45] = 0.6
    draft_token_ids[0, 45] = 5
    draft_probabilities[0, 2] = 0.4
    draft_token_ids[0, 2] = 6
    draft_probabilities[1, 8] = 0.4
    draft_token_ids[1, 8] = 6
    draft_probabilities[1, 17] = 0.6
    draft_token_ids[1, 17] = 5
    target_probabilities = torch.zeros(
        (batch_size * verify_width, candidate_count),
        device=device,
        dtype=torch.float32,
    )
    target_token_ids = torch.full(
        (batch_size * verify_width, candidate_count),
        5,
        device=device,
        dtype=torch.long,
    )
    target_probabilities[0, 40] = 0.3
    target_probabilities[0, 1] = 0.2
    target_token_ids[0, 1] = 7
    target_probabilities[0, 25] = 0.5
    target_token_ids[0, 25] = 8
    target_probabilities[1, 3] = 0.25
    target_token_ids[1, 3] = 1
    target_probabilities[1, 55] = 0.75
    target_token_ids[1, 55] = 14
    target_probabilities[2, 30] = 0.9
    target_token_ids[2, 30] = 6
    target_probabilities[2, 20] = 0.1
    target_probabilities[3, 44] = 0.7
    target_token_ids[3, 44] = 12
    target_probabilities[3, 4] = 0.3
    target_token_ids[3, 4] = 13
    draft_tokens = torch.tensor([[5], [6]], device=device, dtype=torch.long)
    proposal_counts = torch.ones((batch_size,), device=device, dtype=torch.long)
    active_rows = torch.ones((batch_size,), device=device, dtype=torch.bool)
    acceptance_uniforms = torch.tensor([[0.4], [0.0]], device=device)
    correction_uniforms = torch.tensor([[0.99, 0.0], [0.0, 0.0]], device=device)
    bonus_uniforms = torch.tensor([[0.0, 0.0], [0.0, 0.8]], device=device)

    compact_result = accept_speculative_candidates_compact(
        draft_tokens,
        draft_probabilities,
        draft_token_ids,
        target_probabilities,
        target_token_ids,
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
    )
    dense_result = accept_speculative_candidates(
        draft_tokens,
        expand_compact_probability_rows(
            draft_probabilities, draft_token_ids, vocab_size=16
        ).view(batch_size, draft_width, 16),
        expand_compact_probability_rows(
            target_probabilities, target_token_ids, vocab_size=16
        ).view(batch_size, verify_width, 16),
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
    )

    # CDF order differs between compact and dense sampling.
    # Equal distributions do not imply identical draws.
    assert compact_result.committed_tokens.tolist() == [[5, 1], [6, 12]]
    assert dense_result.committed_tokens.tolist() == [[5, 1], [6, 13]]

    for result in (compact_result, dense_result):
        assert result.accepted_draft_counts.tolist() == [1, 1]
        assert result.committed_token_counts.tolist() == [2, 2]
        assert result.rejected_rows.tolist() == [False, False]
        assert result.bonus_rows.tolist() == [True, True]
        assert not bool(result.invalid_rows.any())

    # Deterministic quantile sweep:
    # Row 0 rejects token 5, exercising duplicate padding.
    # Residual support: token 7 = 2/7, token 8 = 5/7.
    # Row 1 accepts and samples the unsorted bonus support.
    draw_count = 70

    draws = (
        torch.arange(
            draw_count,
            device=device,
            dtype=torch.float32,
        )
        + 0.5
    ) / draw_count

    grid_uniforms = (
        draws.repeat_interleave(batch_size)
        .unsqueeze(1)
        .expand(-1, verify_width)
        .contiguous()
    )

    grid_acceptance = acceptance_uniforms.clone()
    grid_acceptance[0, 0] = 0.75
    grid_acceptance = grid_acceptance.repeat(draw_count, 1)

    grid_draft_tokens = draft_tokens.repeat(draw_count, 1)
    grid_draft_probs = draft_probabilities.repeat(draw_count, 1)
    grid_draft_ids = draft_token_ids.repeat(draw_count, 1)
    grid_target_probs = target_probabilities.repeat(draw_count, 1)
    grid_target_ids = target_token_ids.repeat(draw_count, 1)
    grid_proposal_counts = proposal_counts.repeat(draw_count)
    grid_active_rows = active_rows.repeat(draw_count)

    grid_batch_size = batch_size * draw_count

    compact_grid = accept_speculative_candidates_compact(
        grid_draft_tokens,
        grid_draft_probs,
        grid_draft_ids,
        grid_target_probs,
        grid_target_ids,
        grid_proposal_counts,
        grid_active_rows,
        grid_acceptance,
        grid_uniforms,
        grid_uniforms,
    )

    dense_grid = accept_speculative_candidates(
        grid_draft_tokens,
        expand_compact_probability_rows(
            grid_draft_probs,
            grid_draft_ids,
            vocab_size=16,
        ).view(grid_batch_size, draft_width, 16),
        expand_compact_probability_rows(
            grid_target_probs,
            grid_target_ids,
            vocab_size=16,
        ).view(grid_batch_size, verify_width, 16),
        grid_proposal_counts,
        grid_active_rows,
        grid_acceptance,
        grid_uniforms,
        grid_uniforms,
    )

    expected_residual_counts = [0] * 16
    expected_residual_counts[7] = 20
    expected_residual_counts[8] = 50

    expected_bonus_counts = [0] * 16
    expected_bonus_counts[12] = 49
    expected_bonus_counts[13] = 21

    for result in (compact_grid, dense_grid):
        assert not bool(result.invalid_rows.any())

        assert result.accepted_draft_counts.tolist() == ([0, 1] * draw_count)
        assert result.committed_token_counts.tolist() == ([1, 2] * draw_count)
        assert result.rejected_rows.tolist() == ([True, False] * draw_count)
        assert result.bonus_rows.tolist() == ([False, True] * draw_count)

        codes = result.committed_tokens.view(draw_count, batch_size, verify_width)

        residual_tokens = codes[:, 0, 0]
        bonus_tokens = codes[:, 1, 1]

        assert bool((codes[:, 1, 0] == 6).all())

        residual_counts = torch.bincount(residual_tokens, minlength=16).tolist()

        bonus_counts = torch.bincount(bonus_tokens, minlength=16).tolist()

        assert residual_counts == expected_residual_counts
        assert bonus_counts == expected_bonus_counts

    print("UNSORTED_COMPACT_DISTRIBUTION_PARITY_PASS")


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_compact_proposals_respect_persistent_rng_row_stride() -> None:
    device = torch.device("cuda")
    batch_size = 2
    proposal_width = 3
    token_ids = torch.tensor([2, 5, 7, 9], device=device, dtype=torch.int32)
    probabilities = (
        torch.tensor([0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float32)
        .expand(batch_size * proposal_width, -1)
        .contiguous()
    )
    compact_ids = token_ids.expand(batch_size * proposal_width, -1).contiguous()
    uniform_storage = torch.zeros((batch_size, 4, 5), device=device)
    uniforms = uniform_storage[:, 0, :proposal_width]
    uniforms[0] = torch.tensor([0.0, 0.1, 0.95], device=device)
    uniforms[1] = torch.tensor([0.19, 0.21, 0.99], device=device)
    active_rows = torch.tensor([[True, True, True], [True, False, True]], device=device)

    sampled_tokens = sample_compact_probability_rows_fused(
        probabilities, compact_ids, uniforms, active_rows
    )
    expected_tokens = sample_compact_probability_rows(
        probabilities,
        compact_ids,
        uniforms.contiguous().view(-1),
        active_rows.view(-1),
    ).view(batch_size, proposal_width)

    assert torch.equal(sampled_tokens, expected_tokens)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_speculative_probability_rows_match_native_seeded_sampler_distribution() -> (
    None
):
    device = torch.device("cuda")
    sample_count = 2048
    base_logits = torch.full((1, 2048), -8.0, device=device, dtype=torch.bfloat16)
    base_logits[0, :8] = torch.tensor(
        [4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 0.0, -1.0],
        device=device,
        dtype=torch.bfloat16,
    )
    logits = base_logits.expand(sample_count, -1).contiguous()
    temperatures = torch.full((sample_count,), 0.7, device=device)
    top_ks = torch.full((sample_count,), 8, device=device, dtype=torch.long)
    top_ps = torch.full((sample_count,), 0.75, device=device)
    active_rows = torch.ones((sample_count,), device=device, dtype=torch.bool)
    candidates = build_speculative_probability_candidates(
        logits,
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=8,
        has_top_p=True,
        has_unbounded_top_k=False,
    )
    assert candidates is not None
    probabilities, token_ids = candidates
    expected_distribution = torch.zeros(2048, device=device)
    expected_distribution.scatter_add_(0, token_ids[0].long(), probabilities[0])

    seeds = torch.arange(1, sample_count + 1, device=device, dtype=torch.long)
    positions = torch.zeros_like(seeds)
    samples = sample_from_logits_with_seed_top_k_top_p(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        max_top_k=8,
        has_top_p=True,
    )
    assert samples is not None
    observed_distribution = torch.bincount(samples, minlength=2048).float()
    observed_distribution /= sample_count
    torch.testing.assert_close(
        observed_distribution,
        expected_distribution,
        atol=0.045,
        rtol=0.0,
    )


def assert_radix_distribution_parity(
    radix_probabilities,
    radix_token_ids,
    split_probabilities,
    split_token_ids,
):
    assert radix_probabilities.shape == split_probabilities.shape

    for probabilities in (
        radix_probabilities,
        split_probabilities,
    ):
        assert bool(torch.isfinite(probabilities).all())
        assert bool((probabilities >= 0).all())

        torch.testing.assert_close(
            probabilities.double().sum(dim=-1),
            torch.ones(
                probabilities.shape[0],
                device=probabilities.device,
                dtype=torch.float64,
            ),
            atol=1e-6,
            rtol=0.0,
        )

    radix_positive = radix_probabilities > 0
    split_positive = split_probabilities > 0

    assert torch.equal(radix_positive, split_positive)

    assert torch.equal(
        radix_token_ids[radix_positive],
        split_token_ids[split_positive],
    )

    for row in range(radix_probabilities.shape[0]):
        ids = radix_token_ids[row, radix_positive[row]]
        assert bool(((ids >= 0) & (ids < 2048)).all())
        assert ids.unique().numel() == ids.numel()

    # Allow FP32 reduction-order rounding only.
    torch.testing.assert_close(
        radix_probabilities,
        split_probabilities,
        atol=0.0,
        rtol=1e-6,
    )

    diff = (radix_probabilities.double() - split_probabilities.double()).abs()

    tv = 0.5 * diff.sum(dim=-1)

    assert bool((tv <= 5e-7).all()), tv

    print(
        "RADIX_DISTRIBUTION_PARITY_PASS",
        "rows=",
        radix_probabilities.shape[0],
        "max_abs=",
        float(diff.max()),
        "max_tv=",
        float(tv.max()),
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_radix_probability_selector_matches_production_split_top_k() -> None:
    device = torch.device("cuda")
    torch.manual_seed(613)
    row_count = 4
    logits = torch.randn((row_count, 2048), device=device).to(torch.bfloat16)
    logits[0, :4] = 2.0
    temperatures = torch.tensor([0.7, 0.9, 1.1, 0.8], device=device)
    top_ks = torch.tensor([64, 50, 33, 48], device=device, dtype=torch.long)
    top_ps = torch.tensor([0.9, 0.8, 1.0, 0.95], device=device)
    active_rows = torch.ones((row_count,), device=device, dtype=torch.bool)
    split_candidates = build_speculative_probability_candidates(
        logits,
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=64,
        has_top_p=True,
        has_unbounded_top_k=False,
    )
    radix_candidates = build_speculative_probability_candidates_radix(
        logits,
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=64,
        has_top_p=True,
        has_unbounded_top_k=False,
    )
    radix_4bit_candidates = build_speculative_probability_candidates_radix(
        logits,
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=64,
        has_top_p=True,
        has_unbounded_top_k=False,
        radix_bits=4,
    )
    assert split_candidates is not None
    assert radix_candidates is not None
    assert radix_4bit_candidates is not None
    split_probabilities, split_token_ids = split_candidates
    radix_probabilities, radix_token_ids = radix_candidates
    assert torch.equal(radix_token_ids, split_token_ids)
    assert_radix_distribution_parity(
        radix_probabilities,
        radix_token_ids,
        split_probabilities,
        split_token_ids,
    )
    assert torch.equal(radix_4bit_candidates[1], split_token_ids)
    assert_radix_distribution_parity(
        radix_4bit_candidates[0],
        radix_4bit_candidates[1],
        split_probabilities,
        split_token_ids,
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("max_top_k", "row_count"),
    [
        (50, 1),
        (50, 8),
        (50, 16),
        (64, 1),
        (64, 8),
        (64, 16),
        (128, 1),
        (128, 8),
        (128, 16),
    ],
)
@pytest.mark.parametrize("radix_bits", [8, 4])
def test_radix_probability_support_compaction_matches_split_for_small_batches(
    max_top_k: int, row_count: int, radix_bits: int
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(1024 + max_top_k)
    logits = torch.randn((row_count, 2048), device=device).to(torch.bfloat16)
    temperatures = torch.full((row_count,), 0.9, device=device, dtype=torch.float32)
    top_ks = torch.full((row_count,), max_top_k, device=device, dtype=torch.long)
    top_ps = torch.ones((row_count,), device=device, dtype=torch.float32)
    active_rows = torch.ones((row_count,), device=device, dtype=torch.bool)
    split_candidates = build_speculative_probability_candidates(
        logits,
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=max_top_k,
        has_top_p=False,
        has_unbounded_top_k=False,
    )
    radix_candidates = build_speculative_probability_candidates_radix(
        logits,
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=max_top_k,
        has_top_p=False,
        has_unbounded_top_k=False,
        radix_bits=radix_bits,
    )

    assert split_candidates is not None
    assert radix_candidates is not None
    split_probabilities, split_token_ids = split_candidates
    radix_probabilities, radix_token_ids = radix_candidates
    assert_radix_distribution_parity(
        radix_probabilities,
        radix_token_ids,
        split_probabilities,
        split_token_ids,
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_speculative_rng_matches_four_murmur_reference_domains() -> None:
    device = torch.device("cuda")
    seeds = torch.tensor([17, 2**40 + 29, 2**52 + 91], device=device, dtype=torch.long)
    semantic_positions = torch.tensor([0, 19, 2048], device=device, dtype=torch.long)
    round_index = torch.tensor(3, device=device, dtype=torch.long)
    proposal_width = 3
    verify_width = proposal_width + 1
    actual = speculative_uniforms_all_purposes(
        seeds,
        semantic_positions,
        round_index,
        num_code_groups=16,
        proposal_width=proposal_width,
        verify_width=verify_width,
    )
    purposes = ("proposal", "acceptance", "correction", "bonus")
    column_counts = (proposal_width, proposal_width, verify_width, verify_width)
    for actual_uniforms, purpose, column_count in zip(
        actual, purposes, column_counts, strict=True
    ):
        expected = speculative_uniforms(
            seeds,
            semantic_positions,
            int(round_index.item()),
            num_code_groups=16,
            column_count=column_count,
            purpose=purpose,
        )
        assert torch.equal(actual_uniforms, expected)
    else:
        pass


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frame_rng_pool_matches_each_round_and_dynamic_width_prefix() -> None:
    device = torch.device("cuda")
    seeds = torch.tensor([17, 2**40 + 29], device=device, dtype=torch.long)
    semantic_positions = torch.tensor([0, 19], device=device, dtype=torch.long)
    num_code_groups = 16
    max_rounds = num_code_groups - 2
    max_verify_width = 4
    pool_storage = torch.empty(
        (2, 4, max_rounds, max_verify_width), device=device, dtype=torch.float32
    )
    actual_pool = speculative_uniforms_frame_pool(
        seeds,
        semantic_positions,
        num_code_groups,
        max_rounds,
        max_verify_width,
        pool_storage,
    )

    for round_index in range(max_rounds):
        round_value = torch.tensor(round_index, device=device, dtype=torch.long)
        expected = speculative_uniforms_all_purposes(
            seeds,
            semantic_positions,
            round_value,
            num_code_groups,
            proposal_width=3,
            verify_width=4,
        )
        actual = (
            actual_pool[:, 0, round_index, :3],
            actual_pool[:, 1, round_index, :3],
            actual_pool[:, 2, round_index, :4],
            actual_pool[:, 3, round_index, :4],
        )
        for actual_draws, expected_draws in zip(actual, expected, strict=True):
            assert torch.equal(actual_draws, expected_draws)
        else:
            pass
        assert torch.equal(actual_pool[:, 0, round_index, :1], expected[0][:, :1])
    else:
        pass


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warp_gemv_grouped_head_matches_existing_bf16_projection() -> None:
    device = torch.device("cuda")
    torch.manual_seed(1729)
    hidden_states = torch.randn((8, 1024), device=device, dtype=torch.bfloat16)
    packed_weights = torch.randn((3, 2048, 1024), device=device, dtype=torch.bfloat16)
    hidden_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    head_indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], device=device)
    active_rows = torch.tensor(
        [True, True, True, True, True, False, True, False], device=device
    )

    baseline_logits = grouped_lm_head_linear(
        hidden_states,
        packed_weights,
        hidden_indices,
        head_indices,
        active_rows,
        backend="grouped_triton",
    )
    warp_logits = grouped_lm_head_linear(
        hidden_states,
        packed_weights,
        hidden_indices,
        head_indices,
        active_rows,
        backend="warp_gemv",
    )
    warp_accumulate_logits = grouped_lm_head_linear(
        hidden_states,
        packed_weights,
        hidden_indices,
        head_indices,
        active_rows,
        backend="warp_accumulate",
    )

    # Independent high-precision oracle.
    reference = torch.zeros_like(baseline_logits, dtype=torch.float64)
    abs_product_sum = torch.zeros_like(reference)
    native_logits = torch.zeros_like(baseline_logits)

    for row in active_rows.nonzero(as_tuple=True)[0].tolist():
        h = hidden_states[hidden_indices[row]]
        w = packed_weights[head_indices[row]]

        products = w.double() * h.double()
        reference[row] = products.sum(dim=-1)
        abs_product_sum[row] = products.abs().sum(dim=-1)

        native_logits[row] = torch.nn.functional.linear(h, w)

    rounded_reference = reference.to(torch.bfloat16)

    # Use a conservative rounding bound for both tiled and warp-local accumulators.
    unit_roundoff = 2.0**-24
    gamma = 64 * unit_roundoff / (1 - 64 * unit_roundoff)
    fp32_error_bound = gamma * abs_product_sum

    for name, logits in (
        ("grouped_triton", baseline_logits),
        ("warp_gemv", warp_logits),
        ("warp_accumulate", warp_accumulate_logits),
    ):
        assert bool(torch.isfinite(logits).all())
        assert bool((logits[~active_rows] == 0).all())

        values = logits.double()

        previous = torch.nextafter(
            logits, torch.full_like(logits, -float("inf"))
        ).double()

        following = torch.nextafter(
            logits, torch.full_like(logits, float("inf"))
        ).double()

        lower = (previous + values) * 0.5
        upper = (following + values) * 0.5

        outside_rounding_interval = torch.maximum(
            lower - reference,
            reference - upper,
        ).clamp_min(0)

        violations = outside_rounding_interval > fp32_error_bound

        print(
            "GROUPED_GEMV_ORACLE",
            name,
            "rounded_mismatch=",
            int((logits != rounded_reference).sum()),
            "max_abs_fp64_error=",
            float((values - reference).abs().max()),
            "rounding_bound_violations=",
            int(violations.sum()),
        )

        assert not bool(violations.any()), name

    pairwise_error = (warp_logits.float() - baseline_logits.float()).abs()

    print(
        "GROUPED_GEMV_PAIRWISE",
        "mismatched=",
        int((pairwise_error != 0).sum()),
        "max_abs=",
        float(pairwise_error.max()),
    )

    # Logit closeness alone does not establish sampling parity.
    # Exercise small top-k and the production default top-k.
    row_count, vocab_size = baseline_logits.shape

    for top_k in (4, 50, 128):
        for top_p in (1.0, 0.9):
            temperatures = torch.linspace(0.6, 1.4, row_count, device=device)
            top_ks = torch.full(
                (row_count,),
                top_k,
                device=device,
                dtype=torch.long,
            )
            top_ps = torch.full((row_count,), top_p, device=device)

            distributions = {}

            for name, logits in (
                ("native", native_logits),
                ("grouped_triton", baseline_logits),
                ("warp_gemv", warp_logits),
                ("warp_accumulate", warp_accumulate_logits),
            ):
                candidates = build_speculative_probability_candidates(
                    logits.contiguous(),
                    temperatures,
                    top_ks,
                    top_ps,
                    active_rows,
                    max_top_k=top_k,
                    has_top_p=top_p < 1.0,
                    has_unbounded_top_k=False,
                )

                assert candidates is not None

                probabilities, token_ids = candidates

                positive = probabilities > 0

                assert bool(torch.isfinite(probabilities).all())
                assert bool((probabilities >= 0).all())

                assert bool(
                    (
                        (token_ids[positive] >= 0) & (token_ids[positive] < vocab_size)
                    ).all()
                )

                dense = torch.zeros(
                    (row_count, vocab_size),
                    device=device,
                    dtype=torch.float32,
                )

                safe_ids = torch.where(
                    positive,
                    token_ids.long(),
                    torch.zeros_like(token_ids.long()),
                )

                dense.scatter_add_(1, safe_ids, probabilities)

                torch.testing.assert_close(
                    dense.sum(dim=-1),
                    active_rows.float(),
                    atol=1e-6,
                    rtol=0.0,
                )

                distributions[name] = dense

            for left, right in (
                ("grouped_triton", "native"),
                ("warp_gemv", "native"),
                ("warp_accumulate", "native"),
                ("warp_gemv", "grouped_triton"),
                ("warp_accumulate", "grouped_triton"),
            ):
                a = distributions[left]
                b = distributions[right]

                diff = (a.double() - b.double()).abs()
                tv = 0.5 * diff.sum(dim=-1)

                support_mismatch = int(((a > 0) != (b > 0)).sum())

                print(
                    "GROUPED_GEMV_DISTRIBUTION",
                    left,
                    right,
                    "top_k=",
                    top_k,
                    "top_p=",
                    top_p,
                    "support_mismatch=",
                    support_mismatch,
                    "max_abs=",
                    float(diff.max()),
                    "max_tv=",
                    float(tv.max()),
                )

                assert support_mismatch == 0

                torch.testing.assert_close(
                    a,
                    b,
                    atol=3e-5,
                    rtol=3e-4,
                )

                assert bool((tv <= 1e-4).all())

    print("WARP_GEMV_NUMERICAL_PARITY_PASS")


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_commit_preserves_bf16_order_and_writes_only_valid_codes() -> None:
    device = torch.device("cuda")
    torch.manual_seed(312)
    codec_tables = torch.randn((15, 17, 32), device=device, dtype=torch.bfloat16)
    committed_tokens = torch.tensor(
        [[3, 8, 4, 0], [5, 11, 0, 0], [1, 2, 3, 4]],
        device=device,
        dtype=torch.long,
    )
    commit_counts = torch.tensor([3, 2, 0], device=device, dtype=torch.long)
    committed_depths = torch.tensor([1, 5, 10], device=device, dtype=torch.long)
    output_codes = torch.full((3, 16), -1, device=device, dtype=torch.long)
    summed_embeddings = torch.randn((3, 32), device=device, dtype=torch.bfloat16)
    expected_codes = output_codes.clone()
    expected_embeddings = summed_embeddings.clone()
    for row in range(3):
        for offset in range(int(commit_counts[row].item())):
            output_depth = int(committed_depths[row].item()) + 1 + offset
            token_id = int(committed_tokens[row, offset].item())
            expected_codes[row, output_depth] = token_id
            expected_embeddings[row].add_(codec_tables[output_depth - 1, token_id])
        else:
            pass
    else:
        pass

    commit_speculative_codes(
        committed_tokens,
        commit_counts,
        committed_depths,
        codec_tables,
        output_codes,
        summed_embeddings,
    )

    assert torch.equal(output_codes, expected_codes)
    assert torch.equal(summed_embeddings, expected_embeddings)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_state_transition_carries_hidden_and_advances_only_committed_prefix() -> (
    None
):
    device = torch.device("cuda")
    batch_size = 3
    verify_width = 4
    hidden_size = 32
    committed_tokens = torch.tensor(
        [[3, 8, 4, 0], [5, 11, 0, 0], [1, 2, 3, 4]],
        device=device,
        dtype=torch.long,
    )
    accepted_draft_counts = torch.tensor([2, 1, 0], device=device, dtype=torch.long)
    committed_token_counts = torch.tensor([3, 2, 0], device=device, dtype=torch.long)
    proposal_counts = torch.tensor([3, 1, 0], device=device, dtype=torch.long)
    bonus_rows = torch.tensor([False, True, False], device=device)
    rejected_rows = torch.tensor([True, False, False], device=device)
    invalid_rows = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    valid_query_counts = torch.tensor([4, 2, 0], device=device, dtype=torch.long)
    verified_hiddens = torch.randn(
        (batch_size, verify_width, hidden_size), device=device, dtype=DTYPE
    )
    pending_tokens = torch.tensor([-1, -2, -3], device=device, dtype=torch.long)
    preceding_hidden = torch.randn(
        (batch_size, hidden_size), device=device, dtype=DTYPE
    )
    expected_preceding_hidden = preceding_hidden.clone()
    expected_preceding_hidden[0].copy_(verified_hiddens[0, 2])
    expected_preceding_hidden[1].copy_(verified_hiddens[1, 1])
    committed_depths = torch.tensor([1, 5, 15], device=device, dtype=torch.long)
    logical_kv_lengths = torch.tensor([2, 6, 16], device=device, dtype=torch.long)
    active_rows = torch.tensor([True, True, False], device=device)
    metadata = torch.empty((batch_size, 9), device=device, dtype=torch.long)

    commit_speculative_state(
        committed_tokens,
        accepted_draft_counts,
        committed_token_counts,
        proposal_counts,
        bonus_rows,
        rejected_rows,
        invalid_rows,
        valid_query_counts,
        verified_hiddens,
        pending_tokens,
        preceding_hidden,
        committed_depths,
        logical_kv_lengths,
        active_rows,
        metadata,
        num_code_groups=16,
    )

    assert pending_tokens.tolist() == [4, 11, -3]
    assert committed_depths.tolist() == [4, 7, 15]
    assert logical_kv_lengths.tolist() == [5, 8, 16]
    assert active_rows.tolist() == [True, True, False]
    assert torch.equal(preceding_hidden, expected_preceding_hidden)
    assert metadata.tolist() == [
        [2, 1, 3, 3, 0, 1, 4, 4, 0],
        [1, 1, 1, 2, 1, 0, 2, 7, 0],
        [0, 0, 0, 0, 0, 0, 0, 15, 0],
    ]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_verifier_preparation_matches_projected_codec_lookup() -> None:
    device = torch.device("cuda")
    batch_size = 3
    verify_width = 4
    hidden_size = 16
    projected_tables = torch.randn(
        (14, 32, hidden_size), device=device, dtype=torch.bfloat16
    )
    pending_tokens = torch.tensor([4, 5, 6], device=device, dtype=torch.long)
    draft_tokens = torch.tensor(
        [[7, 8, 9], [10, 11, 12], [13, 14, 15]], device=device, dtype=torch.long
    )
    committed_depths = torch.tensor([1, 3, 13], device=device, dtype=torch.long)
    logical_kv_lengths = torch.tensor([2, 7, 15], device=device, dtype=torch.long)
    active_rows = torch.tensor([True, True, False], device=device)
    proposal_counts = torch.tensor([3, 1, 0], device=device, dtype=torch.long)
    outputs = SpeculativeVerifierInputs(
        token_embeddings=torch.empty(
            (batch_size, verify_width, hidden_size),
            device=device,
            dtype=torch.bfloat16,
        ),
        positions=torch.empty(
            batch_size * verify_width, device=device, dtype=torch.long
        ),
        target_head_indices=torch.empty(
            (batch_size, verify_width), device=device, dtype=torch.long
        ),
        valid_queries=torch.empty(
            (batch_size, verify_width), device=device, dtype=torch.bool
        ),
    )

    assert prepare_speculative_verification_inputs(
        pending_tokens,
        draft_tokens,
        committed_depths,
        logical_kv_lengths,
        active_rows,
        proposal_counts,
        projected_tables,
        outputs,
    )
    expected_valid = torch.tensor(
        [[True, True, True, True], [True, True, False, False], [False] * 4],
        device=device,
    )
    expected_positions = torch.tensor(
        [[2, 3, 4, 5], [7, 8, -1, -1], [-1] * 4], device=device
    )
    expected_heads = torch.tensor([[1, 2, 3, 4], [3, 4, 0, 0], [0] * 4], device=device)
    query_tokens = torch.cat((pending_tokens[:, None], draft_tokens), dim=1)
    query_depths = (
        committed_depths[:, None] + torch.arange(verify_width, device=device)[None, :]
    )
    expected_embeddings = projected_tables[
        (query_depths - 1).clamp(min=0, max=projected_tables.shape[0] - 1),
        query_tokens,
    ]
    expected_embeddings = torch.where(
        expected_valid.unsqueeze(-1),
        expected_embeddings,
        torch.zeros_like(expected_embeddings),
    )

    assert torch.equal(
        outputs.positions.view(batch_size, verify_width), expected_positions
    )
    assert torch.equal(outputs.target_head_indices, expected_heads)
    assert torch.equal(outputs.valid_queries, expected_valid)
    assert torch.equal(outputs.token_embeddings, expected_embeddings)

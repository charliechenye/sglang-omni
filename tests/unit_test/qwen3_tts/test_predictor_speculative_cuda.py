# SPDX-License-Identifier: Apache-2.0
"""CUDA full-frame and persistent-graph gates for Predictor speculation."""

from __future__ import annotations

from types import SimpleNamespace

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
from sglang_omni.utils.predictor_grouped_heads import grouped_lm_head_linear
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
    talker.predictor_device = device
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
    talker.code_predictor_spec_probability_backend = "split_top_k"
    talker.code_predictor_spec_probability_backend_execution_counts = {
        "split_top_k": 0,
        "radix_compact": 0,
    }
    talker.code_predictor_spec_probability_backend_fallback_count = 0
    talker.code_predictor_spec_fallback_reason = None
    talker.code_predictor_spec_graphs = {}
    talker.code_predictor_spec_effective_width_histogram = {}
    talker.code_predictor_spec_graph_replay_count = 0
    talker.code_predictor_spec_eager_round_count = 0
    talker.code_predictor_spec_graph_capture_failure_count = 0
    talker.code_predictor_spec_graph_memory_delta_bytes = 0
    talker.code_predictor_spec_graph_capture_time_ms = 0.0
    talker.code_predictor_spec_configuration_time_ms = 0.0
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
def test_packed_predictor_matches_sequential_native_hidden_and_kv_prefixes() -> None:
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
        torch.testing.assert_close(packed_logits, native_logits, atol=2e-2, rtol=2e-2)
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


@pytest.mark.parametrize("execution_backend", ["eager", "cuda_graph"])
@pytest.mark.parametrize("top_k", [4, 50])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_full_frame_w1_tail_supports_non_power_of_two_top_k(
    monkeypatch: pytest.MonkeyPatch, execution_backend: str, top_k: int
) -> None:
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
        )

        probability_call_count = 0

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
        ) -> tuple[torch.Tensor, torch.Tensor, str]:
            nonlocal probability_call_count
            assert temperatures.shape == top_ks.shape == top_ps.shape
            assert temperatures.shape == active_rows.shape
            assert has_top_p is False
            assert has_unbounded_top_k is False
            probability_call_count += 1
            candidate_count = fused_raw_logit_block_k(max_top_k)
            assert candidate_count is not None
            probabilities = torch.zeros(
                (logits.shape[0], candidate_count),
                device=device,
                dtype=torch.float32,
            )
            token_ids = torch.full(
                (logits.shape[0], candidate_count),
                -1,
                device=device,
                dtype=torch.int32,
            )
            token_ids[:, 0] = 0
            if probability_call_count == 8:
                token_ids[:, 1] = 1
                probabilities[:, :2] = 0.5
            else:
                probabilities[:, 0] = 1.0
            probabilities[~active_rows] = 0.0
            actual_backend = "split_top_k" if backend == "radix_compact" else backend
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

        codes, summed_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, codes, summed_embedding)

    stats = talker.predictor_speculative_stats()
    assert probability_call_count == 9
    assert stats["frames"] == 1
    assert stats["native_fallback_count"] == 0
    assert stats["effective_width_histogram"] == {"1": 1, "2": 1, "4": 3}
    assert stats["probability_backend_execution_counts"] == {
        "split_top_k": 9,
        "radix_compact": 0,
    }
    assert stats["probability_backend_fallbacks"] == (9 if top_k == 4 else 0)
    if execution_backend == "eager":
        assert stats["eager_verifier_rounds"] == 5
        assert stats["verifier_graph_replays"] == 0
    else:
        assert stats["eager_verifier_rounds"] == 0
        assert stats["verifier_graph_replays"] == 5


@pytest.mark.parametrize("forced_rejection_offset", [0, 1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_forced_code_replay_matches_native_targets_and_valid_kv_prefixes(
    monkeypatch: pytest.MonkeyPatch, forced_rejection_offset: int
) -> None:
    device = torch.device("cuda")
    batch_size = 2
    verify_width = 4
    proposal_width = verify_width - 1
    target_row_count = batch_size * verify_width
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=batch_size):
        talker = build_speculative_talker(device, batch_size)
        for lm_head in talker.code_predictor.lm_head:
            lm_head.weight.zero_()
        else:
            pass
        configure_unseeded_request(talker, batch_size)
        talker.configure_code_predictor_speculation(verify_width, backend="eager")

        native_grouped_head = sglang_model_module.grouped_lm_head_linear
        native_probability_builder = (
            sglang_model_module.build_speculative_probability_candidates_selected
        )
        native_acceptance = sglang_model_module.accept_speculative_candidates_compact
        native_packed_forward = talker.predictor_forward_packed_tokens
        packed_hiddens: list[torch.Tensor] = []
        packed_keys: list[torch.Tensor] = []
        packed_values: list[torch.Tensor] = []
        target_logits: list[torch.Tensor] = []
        target_probabilities: list[torch.Tensor] = []
        target_token_ids: list[torch.Tensor] = []
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
            assert current_proposal_width == proposal_width
            assert current_verify_width == verify_width
            output[:batch_size].zero_()
            current_round = int(round_index.item())
            if current_round == 0:
                output[1, 1, forced_rejection_offset] = 1.0
            elif current_round in (1, 2):
                output[1, 1, 0] = 1.0
            else:
                pass
            return (
                output[:batch_size, 0, :proposal_width],
                output[:batch_size, 1, :proposal_width],
                output[:batch_size, 2, :verify_width],
                output[:batch_size, 3, :verify_width],
            )

        def record_packed_forward(
            token_embeds: torch.Tensor,
            positions: torch.Tensor,
            current_batch_size: int,
        ) -> torch.Tensor:
            hidden_states = native_packed_forward(
                token_embeds=token_embeds,
                positions=positions,
                batch_size=current_batch_size,
            )
            packed_hiddens.append(hidden_states.clone())
            packed_keys.append(talker.predictor_k_cache.clone())
            packed_values.append(talker.predictor_v_cache.clone())
            return hidden_states

        def record_grouped_head(
            hidden_states: torch.Tensor,
            packed_weights: torch.Tensor,
            hidden_indices: torch.Tensor,
            head_indices: torch.Tensor,
            active_rows: torch.Tensor,
            *,
            backend: str,
        ) -> torch.Tensor:
            logits = native_grouped_head(
                hidden_states,
                packed_weights,
                hidden_indices,
                head_indices,
                active_rows,
                backend=backend,
            )
            if hidden_states.shape[0] == target_row_count:
                target_logits.append(logits.view(batch_size, verify_width, -1).clone())
            else:
                pass
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
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
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
            )
            if logits.shape[0] == target_row_count and candidates is not None:
                probabilities, token_ids = candidates
                target_probabilities.append(
                    probabilities.view(batch_size, verify_width, -1).clone()
                )
                target_token_ids.append(
                    token_ids.view(batch_size, verify_width, -1).clone()
                )
            else:
                pass
            return candidates

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
        ) -> SpeculativeAcceptanceBatch:
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
            )
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
        monkeypatch.setattr(
            talker, "predictor_forward_packed_tokens", record_packed_forward
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

        assert acceptance_records[0][2].tolist() == [
            proposal_width,
            forced_rejection_offset,
        ]
        assert acceptance_records[0][5].tolist() == [True, False]
        assert acceptance_records[0][6].tolist() == [False, True]
        assert acceptance_records[0][0].tolist() == [proposal_width, proposal_width]
        assert acceptance_records[3][0].tolist() == [1, proposal_width]

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
        talker.predictor_forward_tokens(
            token_embeds=pair_embeddings, batch_size=batch_size, cache_len=0
        )
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
            assert bool((proposal_counts <= proposal_width).all())
            if round_index < 3:
                assert rejected_rows.tolist() == [False, True]
            else:
                pass
            if round_index == 0:
                assert bonus_rows.tolist() == [True, False]
            else:
                pass
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
                        expected_candidates = build_speculative_probability_candidates(
                            native_logits[code_depth][row : row + 1].contiguous(),
                            torch.full((1,), 0.9, device=device, dtype=torch.float32),
                            torch.full((1,), 4, device=device, dtype=torch.long),
                            torch.ones((1,), device=device, dtype=torch.float32),
                            torch.ones((1,), device=device, dtype=torch.bool),
                            max_top_k=4,
                            has_top_p=False,
                            has_unbounded_top_k=False,
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
                    else:
                        pass
                else:
                    pass
                new_depth = starting_depth + commit_count
                valid_cache_length = new_depth + 1
                expected_cache_depth = new_depth - 1
                torch.testing.assert_close(
                    packed_keys[round_index][:, row : row + 1, :valid_cache_length],
                    native_keys[expected_cache_depth][row : row + 1],
                    atol=2e-2,
                    rtol=2e-2,
                )
                torch.testing.assert_close(
                    packed_values[round_index][:, row : row + 1, :valid_cache_length],
                    native_values[expected_cache_depth][row : row + 1],
                    atol=2e-2,
                    rtol=2e-2,
                )
                committed_depths[row] = new_depth
            else:
                pass
        else:
            pass

        assert committed_depths == [15, 15]
        stats = talker.predictor_speculative_stats()
        assert stats["frames"] == batch_size
        assert stats["rejections"] == 3
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
            batch_size=1,
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
        for _ in range(4):
            talker.code_predictor_forward(
                layer0_codes, talker_hidden, semantic_positions
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert candidate_logits is logits
        assert candidate_temperatures is temperatures
        assert candidate_top_ks is top_ks
        assert candidate_top_ps is top_ps
        assert candidate_active_rows is active_rows
        assert max_top_k == 4
        assert has_top_p is False
        assert has_unbounded_top_k is False
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

    assert torch.equal(compact_result.committed_tokens, dense_result.committed_tokens)
    assert torch.equal(
        compact_result.accepted_draft_counts, dense_result.accepted_draft_counts
    )
    assert compact_result.accepted_draft_counts.tolist() == [1, 1]
    assert not bool(compact_result.invalid_rows.any())


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
    assert split_candidates is not None
    assert radix_candidates is not None
    split_probabilities, split_token_ids = split_candidates
    radix_probabilities, radix_token_ids = radix_candidates
    assert torch.equal(radix_token_ids, split_token_ids)
    torch.testing.assert_close(
        radix_probabilities, split_probabilities, atol=0.0, rtol=0.0
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
def test_radix_probability_support_compaction_matches_split_for_small_batches(
    max_top_k: int, row_count: int
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
    )

    assert split_candidates is not None
    assert radix_candidates is not None
    split_probabilities, split_token_ids = split_candidates
    radix_probabilities, radix_token_ids = radix_candidates
    torch.testing.assert_close(
        radix_probabilities.sum(dim=1),
        torch.ones((row_count,), device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    for row in range(row_count):
        split_positive = split_probabilities[row] > 0.0
        radix_positive = radix_probabilities[row] > 0.0
        assert torch.equal(
            radix_token_ids[row, radix_positive],
            split_token_ids[row, split_positive],
        )
        torch.testing.assert_close(
            radix_probabilities[row, radix_positive],
            split_probabilities[row, split_positive],
            atol=0.0,
            rtol=0.0,
        )
        assert bool((radix_token_ids[row, radix_positive] >= 0).all())
    else:
        pass


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

    assert torch.equal(warp_logits, baseline_logits)


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

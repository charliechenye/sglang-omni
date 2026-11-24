# SPDX-License-Identifier: Apache-2.0
"""CUDA full-frame and persistent-graph gates for Predictor speculation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.runtime_context import get_context
from torch import nn

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    build_speculative_probability_candidates,
    build_speculative_probability_candidates_radix,
    sample_from_logits_with_seed_top_k_top_p,
)
from sglang_omni.models.qwen3_tts.sglang_model import (
    Qwen3TTSCodecTables,
    Qwen3TTSTalker,
)
from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    SpeculativeAcceptanceBatch,
    sample_compact_probability_rows_fused,
)
from sglang_omni.models.qwen3_tts.speculative_rng import (
    speculative_uniforms,
    speculative_uniforms_all_purposes,
)
from sglang_omni.models.qwen3_tts.speculative_runtime import (
    accept_speculative_candidates,
    accept_speculative_candidates_compact,
    commit_speculative_codes,
    expand_compact_probability_rows,
    sample_compact_probability_rows,
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
    talker.code_predictor_spec_fallback_reason = None
    talker.code_predictor_spec_graphs = {}
    talker.code_predictor_spec_effective_width_histogram = {}
    talker.code_predictor_spec_graph_replay_count = 0
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
            sglang_model_module.build_speculative_probability_candidates
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
        ) -> torch.Tensor:
            logits = native_grouped_head(
                hidden_states,
                packed_weights,
                hidden_indices,
                head_indices,
                active_rows,
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
            "build_speculative_probability_candidates",
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
    assert compact_result.accepted_draft_counts[:draft_width].tolist() == [0, 1, 2]
    assert compact_result.committed_tokens[3, -1].item() == 6
    assert compact_result.accepted_draft_counts[4:].tolist() == [1, 0]
    assert compact_result.committed_token_counts[4:].tolist() == [2, 0]
    assert compact_result.committed_tokens[4, 1].item() == 5


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

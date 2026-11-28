# SPDX-License-Identifier: Apache-2.0
"""CUDA correctness gates for the supported P16 sampling and commit kernels."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    build_speculative_probability_candidates,
    sample_from_logits_with_seed_top_k_top_p,
)
from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    accept_speculative_candidates_compact,
)
from sglang_omni.models.qwen3_tts.speculative_commit import commit_speculative_codes
from sglang_omni.models.qwen3_tts.speculative_prepare import (
    SpeculativeVerifierInputs,
    prepare_speculative_verification_inputs,
)
from sglang_omni.models.qwen3_tts.speculative_rng import (
    SPECULATIVE_DRAW_SALTS,
    speculative_uniforms,
    speculative_uniforms_frame_pool,
)
from sglang_omni.vendor.sglang.layers import RMSNorm
from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
    DTYPE,
    FUSED_HIDDEN,
    real_shape_layer,
    real_shape_talker,
)

pytestmark = pytest.mark.accelerator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_packed_predictor_matches_sequential_native_hidden_and_kv_prefixes() -> None:
    device = torch.device("cuda")
    torch.manual_seed(108)
    layers = [real_shape_layer(device) for _ in range(5)]
    final_norm = RMSNorm(FUSED_HIDDEN, eps=1e-6).to(device=device, dtype=DTYPE)
    talker = real_shape_talker(device, layers, final_norm, fused=True)
    talker.predictor_device = device
    talker.code_predictor_spec_correction_forward_watch = False
    talker.code_predictor_spec_standalone_correction_forward_count = 0
    talker.code_predictor_generic_forward_count = 0
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
    torch.testing.assert_close(
        talker.predictor_k_cache[:, :1, :4], packed_key_prefix, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        talker.predictor_v_cache[:, :1, :4], packed_value_prefix, atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_acceptance_rejects_each_proposal_position_and_accepts_bonus() -> None:
    device = torch.device("cuda")
    batch_size = 3
    draft_width = 3
    verify_width = draft_width + 1
    candidate_ids = torch.tensor([0, 1], device=device, dtype=torch.int32)
    draft_probabilities = (
        torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
        .expand(batch_size * draft_width, -1)
        .contiguous()
    )
    draft_token_ids = candidate_ids.expand(batch_size * draft_width, -1).contiguous()
    target_probabilities = torch.full(
        (batch_size * verify_width, 2), 0.5, device=device, dtype=torch.float32
    )
    target_token_ids = candidate_ids.expand(batch_size * verify_width, -1).contiguous()
    target_probabilities[1] = torch.tensor([0.25, 0.75], device=device)
    target_probabilities[4] = torch.tensor([0.1, 0.9], device=device)
    target_probabilities[8:11] = torch.tensor([1.0, 0.0], device=device)
    target_probabilities[11] = torch.tensor([0.0, 1.0], device=device)
    draft_tokens = torch.zeros(
        (batch_size, draft_width), device=device, dtype=torch.long
    )
    proposal_counts = torch.full(
        (batch_size,), draft_width, device=device, dtype=torch.long
    )
    active_rows = torch.ones((batch_size,), device=device, dtype=torch.bool)
    acceptance_uniforms = torch.tensor(
        [[0.1, 0.9, 0.1], [0.5, 0.1, 0.1], [0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    correction_uniforms = torch.zeros((batch_size, verify_width), device=device)
    bonus_uniforms = torch.zeros_like(correction_uniforms)

    result = accept_speculative_candidates_compact(
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

    assert result.accepted_draft_counts.tolist() == [1, 0, 3]
    assert result.committed_token_counts.tolist() == [2, 1, 4]
    assert result.rejected_rows.tolist() == [True, True, False]
    assert result.bonus_rows.tolist() == [False, False, True]
    assert result.invalid_rows.tolist() == [False, False, False]
    assert result.committed_tokens.tolist() == [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_w1_samples_target_and_rejects_invalid_target_mass() -> None:
    device = torch.device("cuda")
    draft_tokens = torch.empty((1, 0), device=device, dtype=torch.long)
    draft_probabilities = torch.empty((0, 2), device=device, dtype=torch.float32)
    draft_token_ids = torch.empty((0, 2), device=device, dtype=torch.long)
    target_token_ids = torch.tensor([[5, 6]], device=device, dtype=torch.long)
    proposal_counts = torch.zeros((1,), device=device, dtype=torch.long)
    active_rows = torch.ones((1,), device=device, dtype=torch.bool)
    acceptance_uniforms = torch.empty((1, 0), device=device)
    correction_uniforms = torch.zeros((1, 1), device=device)
    bonus_uniforms = torch.tensor([[0.7]], device=device)

    valid = accept_speculative_candidates_compact(
        draft_tokens,
        draft_probabilities,
        draft_token_ids,
        torch.tensor([[0.25, 0.75]], device=device),
        target_token_ids,
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
    )
    invalid = accept_speculative_candidates_compact(
        draft_tokens,
        draft_probabilities,
        draft_token_ids,
        torch.zeros((1, 2), device=device),
        target_token_ids,
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
    )

    assert valid.committed_tokens.tolist() == [[6]]
    assert valid.committed_token_counts.tolist() == [1]
    assert valid.invalid_rows.tolist() == [False]
    assert invalid.committed_tokens.tolist() == [[0]]
    assert invalid.committed_token_counts.tolist() == [0]
    assert invalid.invalid_rows.tolist() == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_top_k_probability_support_matches_seeded_native_samples() -> None:
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
        observed_distribution, expected_distribution, atol=0.045, rtol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frame_rng_pool_matches_each_round_and_draw_domain() -> None:
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
        expected = tuple(
            speculative_uniforms(
                seeds,
                semantic_positions,
                round_index,
                num_code_groups,
                (
                    max_verify_width - 1
                    if purpose in ("proposal", "acceptance")
                    else max_verify_width
                ),
                purpose,
            )
            for purpose in SPECULATIVE_DRAW_SALTS
        )
        actual = (
            actual_pool[:, 0, round_index, :3],
            actual_pool[:, 1, round_index, :3],
            actual_pool[:, 2, round_index, :4],
            actual_pool[:, 3, round_index, :4],
        )
        for actual_draws, expected_draws in zip(actual, expected, strict=True):
            assert torch.equal(actual_draws, expected_draws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_commit_preserves_bf16_order_and_only_writes_committed_codes() -> None:
    device = torch.device("cuda")
    codec_tables = torch.randn((15, 17, 32), device=device, dtype=torch.bfloat16)
    committed_tokens = torch.tensor([[3, 8, 4, 0]], device=device, dtype=torch.long)
    commit_counts = torch.tensor([3], device=device, dtype=torch.long)
    committed_depths = torch.tensor([1], device=device, dtype=torch.long)
    output_codes = torch.full((1, 16), -1, device=device, dtype=torch.long)
    summed_embeddings = torch.randn((1, 32), device=device, dtype=torch.bfloat16)
    expected_codes = output_codes.clone()
    expected_embeddings = summed_embeddings.clone()
    for offset in range(3):
        output_depth = 2 + offset
        token_id = int(committed_tokens[0, offset].item())
        expected_codes[0, output_depth] = token_id
        expected_embeddings[0].add_(codec_tables[output_depth - 1, token_id])

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_verifier_preparation_uses_projected_codec_lookup_for_one_request() -> None:
    device = torch.device("cuda")
    verify_width = 4
    hidden_size = 16
    projected_tables = torch.randn(
        (14, 32, hidden_size), device=device, dtype=torch.bfloat16
    )
    pending_tokens = torch.tensor([4], device=device, dtype=torch.long)
    draft_tokens = torch.tensor([[7, 8, 9]], device=device, dtype=torch.long)
    committed_depths = torch.tensor([1], device=device, dtype=torch.long)
    logical_kv_lengths = torch.tensor([2], device=device, dtype=torch.long)
    active_rows = torch.tensor([True], device=device)
    proposal_counts = torch.tensor([3], device=device, dtype=torch.long)
    outputs = SpeculativeVerifierInputs(
        token_embeddings=torch.empty(
            (1, verify_width, hidden_size), device=device, dtype=torch.bfloat16
        ),
        positions=torch.empty(verify_width, device=device, dtype=torch.long),
        target_head_indices=torch.empty(
            (1, verify_width), device=device, dtype=torch.long
        ),
        valid_queries=torch.empty((1, verify_width), device=device, dtype=torch.bool),
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
    expected_tokens = torch.cat((pending_tokens[:, None], draft_tokens), dim=1)
    expected_depths = (
        committed_depths[:, None] + torch.arange(verify_width, device=device)[None, :]
    )
    assert torch.equal(outputs.positions, torch.tensor([2, 3, 4, 5], device=device))
    assert torch.equal(
        outputs.target_head_indices, torch.tensor([[1, 2, 3, 4]], device=device)
    )
    assert bool(outputs.valid_queries.all())
    assert torch.equal(
        outputs.token_embeddings, projected_tables[expected_depths - 1, expected_tokens]
    )

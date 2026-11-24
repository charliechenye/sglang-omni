# SPDX-License-Identifier: Apache-2.0
"""CPU correctness gates for the CodePredictor speculative reference path."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.utils.predictor_speculation import (
    acceptance_probability,
    probabilities_from_logits_cpu,
    residual_distribution_cpu,
    resolve_speculative_capability,
    run_speculative_frame_reference,
    sample_categorical_cpu,
    speculative_acceptance_cpu,
    speculative_resources_requested,
    SpeculativeRowState,
)


def one_hot(token: int, vocab_size: int) -> torch.Tensor:
    probabilities = torch.zeros(vocab_size, dtype=torch.float64)
    probabilities[token] = 1.0
    return probabilities


def test_cpu_reference_probability_filtering_handles_ties_and_signed_zero() -> None:
    logits = torch.tensor([2.0, 1.0, 1.0, 0.0, -1.0], dtype=torch.float32)

    probabilities = probabilities_from_logits_cpu(
        logits,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
    )

    assert probabilities[0] > 0.0
    assert probabilities[1] > 0.0
    assert probabilities[2:].sum() == 0.0
    signed_zero = probabilities_from_logits_cpu(
        torch.tensor([0.0, -0.0], dtype=torch.float32),
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    assert signed_zero.tolist() == pytest.approx([0.5, 0.5], abs=1e-14)
    top_p_probabilities = probabilities_from_logits_cpu(
        torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        temperature=1.0,
        top_k=0,
        top_p=0.6,
    )
    assert top_p_probabilities[0] == pytest.approx(1.0)
    assert top_p_probabilities[1:].sum() == 0.0
    assert top_p_probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("logits", "temperature", "top_k", "top_p"),
    [
        (torch.tensor([float("-inf"), float("-inf")]), 1.0, 0, 1.0),
        (torch.tensor([0.0, float("nan")]), 1.0, 0, 1.0),
        (torch.tensor([0.0, 1.0]), 0.0, 0, 1.0),
        (torch.tensor([0.0, 1.0]), 1.0, -1, 1.0),
        (torch.tensor([0.0, 1.0]), 1.0, 0, 1.1),
    ],
)
def test_cpu_reference_probability_filtering_rejects_invalid_inputs(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> None:
    with pytest.raises(ValueError):
        probabilities_from_logits_cpu(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )


def test_acceptance_residual_and_bonus_edge_cases() -> None:
    assert acceptance_probability(0.2, 0.4) == pytest.approx(0.5)
    assert acceptance_probability(0.9, 0.4) == pytest.approx(1.0)

    equal = one_hot(2, 4)
    assert torch.equal(residual_distribution_cpu(equal, equal), equal)
    equal_distribution_decision = speculative_acceptance_cpu(
        torch.tensor([2]),
        equal.unsqueeze(0),
        torch.stack((equal, equal)),
        torch.tensor([0.999], dtype=torch.float64),
        torch.tensor([0.1, 0.3], dtype=torch.float64),
    )
    assert equal_distribution_decision.accepted_draft_count == 1
    assert not equal_distribution_decision.rejected
    assert torch.equal(
        residual_distribution_cpu(one_hot(1, 4), one_hot(0, 4)), one_hot(1, 4)
    )

    no_proposals = speculative_acceptance_cpu(
        torch.empty((0,), dtype=torch.long),
        torch.empty((0, 4), dtype=torch.float64),
        one_hot(3, 4).unsqueeze(0),
        torch.empty((0,), dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64),
    )
    assert no_proposals.committed_tokens == (3,)
    assert no_proposals.accepted_draft_count == 0
    assert no_proposals.bonus_sampled
    assert not no_proposals.rejected


def test_rejects_each_proposal_position_and_accepts_full_window() -> None:
    vocab_size = 6
    draft_tokens = torch.tensor([0, 1, 2], dtype=torch.long)
    draft_probabilities = torch.stack(
        [one_hot(token, vocab_size) for token in draft_tokens]
    )
    target_probabilities = torch.stack(
        [one_hot(token, vocab_size) for token in (3, 4, 5, 2)]
    )

    for rejection_index in range(3):
        target = target_probabilities.clone()
        for accepted_index in range(rejection_index):
            target[accepted_index] = one_hot(
                int(draft_tokens[accepted_index]), vocab_size
            )
        target[rejection_index] = one_hot(3 + rejection_index, vocab_size)
        target[rejection_index + 1 :] = one_hot(0, vocab_size)
        acceptance_uniforms = torch.zeros(3, dtype=torch.float64)
        acceptance_uniforms[rejection_index] = 0.5
        decision = speculative_acceptance_cpu(
            draft_tokens,
            draft_probabilities,
            target,
            acceptance_uniforms,
            torch.zeros(4, dtype=torch.float64),
        )
        assert decision.rejected
        assert decision.accepted_draft_count == rejection_index
        assert decision.committed_tokens == tuple(
            [int(token) for token in draft_tokens[:rejection_index]]
            + [3 + rejection_index]
        )

    accepted = speculative_acceptance_cpu(
        draft_tokens,
        draft_probabilities,
        torch.stack(
            [one_hot(token, vocab_size) for token in (0, 1, 2, 5)]
        ),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(4, dtype=torch.float64),
    )
    assert not accepted.rejected
    assert accepted.bonus_sampled
    assert accepted.accepted_draft_count == 3
    assert accepted.committed_tokens == (0, 1, 2, 5)


def test_sequential_target_distribution_matches_speculative_sampling() -> None:
    target = torch.tensor([0.17, 0.31, 0.52], dtype=torch.float64)
    draft = torch.tensor([0.55, 0.30, 0.15], dtype=torch.float64)
    draft_token = torch.tensor([0], dtype=torch.long)
    draft_probabilities = draft.unsqueeze(0)
    target_probabilities = torch.stack((target, target))
    generator = torch.Generator().manual_seed(916)
    sample_count = 4000
    observed = torch.zeros(3, dtype=torch.float64)
    for _ in range(sample_count):
        uniforms = torch.rand(2, generator=generator, dtype=torch.float64)
        acceptance = torch.rand(1, generator=generator, dtype=torch.float64)
        decision = speculative_acceptance_cpu(
            draft_token,
            draft_probabilities,
            target_probabilities,
            acceptance,
            uniforms,
        )
        observed[decision.committed_tokens[0]] += 1
    observed /= sample_count
    torch.testing.assert_close(observed, target, atol=0.025, rtol=0.0)


def test_complete_frame_state_machine_commits_only_surviving_codes() -> None:
    batch_size = 3
    verify_width = 3
    num_groups = 5
    vocab_size = 4
    initial_codes = torch.zeros((batch_size, 2), dtype=torch.long)
    initial_hidden = torch.zeros((batch_size, 1), dtype=torch.float64)
    initial_sum = torch.full((batch_size, 1), 100.0, dtype=torch.float64)
    embedding_tables = torch.arange(
        (num_groups - 1) * vocab_size, dtype=torch.float64
    ).reshape(num_groups - 1, vocab_size, 1)
    verify_round = 0

    def provide_drafts(
        states: list[SpeculativeRowState], proposal_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(states) == batch_size
        distributions = torch.zeros(
            (batch_size, verify_width - 1, vocab_size), dtype=torch.float64
        )
        tokens = torch.zeros((batch_size, verify_width - 1), dtype=torch.long)
        for row, count_tensor in enumerate(proposal_counts):
            for proposal_index in range(int(count_tensor)):
                distributions[row, proposal_index] = one_hot(0, vocab_size)
        return distributions, tokens

    def verify_packed(
        pending: torch.Tensor,
        drafts: torch.Tensor,
        proposal_counts: torch.Tensor,
        active_rows: torch.Tensor,
        kv_lengths: torch.Tensor,
        preceding_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal verify_round
        expected_kv_lengths = (2, 2, 2) if verify_round == 0 else (3, 5, 4)
        assert tuple(int(length) for length in kv_lengths) == expected_kv_lengths
        assert tuple(int(count) for count in proposal_counts) == (
            (2, 2, 2) if verify_round == 0 else (1, 0, 0)
        )
        assert tuple(bool(active) for active in active_rows) == (
            (True, True, True) if verify_round == 0 else (True, False, True)
        )
        if verify_round == 1:
            assert preceding_hidden[0, 0] == 0.0
            assert preceding_hidden[2, 0] == 21.0
        else:
            pass
        probabilities = torch.stack(
            [
                torch.stack([one_hot(0, vocab_size) for _ in range(verify_width)])
                for _ in range(batch_size)
            ]
        )
        if verify_round == 0:
            probabilities[0, 0] = one_hot(1, vocab_size)
            probabilities[1, 2] = one_hot(3, vocab_size)
            probabilities[2, 1] = one_hot(2, vocab_size)
        else:
            probabilities[0, 1] = one_hot(2, vocab_size)
            probabilities[2, 0] = one_hot(3, vocab_size)
        hidden = torch.empty((batch_size, verify_width, 1), dtype=torch.float64)
        for row in range(batch_size):
            for token_index in range(verify_width):
                hidden[row, token_index, 0] = (
                    verify_round * 100 + row * 10 + token_index
                )
        verify_round += 1
        return probabilities, hidden

    def uniforms(
        round_index: int, proposal_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acceptance = torch.zeros((batch_size, verify_width - 1), dtype=torch.float64)
        if round_index == 0:
            acceptance[0, 0] = 0.5
            acceptance[2, 1] = 0.5
        else:
            pass
        sampling = torch.zeros((batch_size, verify_width), dtype=torch.float64)
        return acceptance, sampling

    result = run_speculative_frame_reference(
        initial_codes=initial_codes,
        initial_hidden_before_pending=initial_hidden,
        initial_embedding_sum=initial_sum,
        codec_embedding_tables=embedding_tables,
        num_code_groups=num_groups,
        verify_width=verify_width,
        draft_provider=provide_drafts,
        verify_provider=verify_packed,
        uniform_provider=uniforms,
    )

    assert result.codes == (
        (0, 0, 1, 0, 2),
        (0, 0, 0, 0, 3),
        (0, 0, 0, 2, 3),
    )
    assert result.rounds == 2
    assert result.proposed_draft_tokens == 7
    assert result.accepted_draft_tokens == 5
    assert result.bonus_tokens == 3
    assert result.rejections == 2
    expected_sums = torch.tensor([[127.0], [127.0], [129.0]], dtype=torch.float64)
    torch.testing.assert_close(result.summed_embeddings, expected_sums)


def test_frame_with_one_remaining_code_uses_bonus_without_proposals() -> None:
    verify_width = 4
    vocab_size = 3

    def provide_drafts(
        states: list[SpeculativeRowState], proposal_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert tuple(int(count) for count in proposal_counts) == (0,)
        return (
            torch.zeros((1, verify_width - 1, vocab_size), dtype=torch.float64),
            torch.zeros((1, verify_width - 1), dtype=torch.long),
        )

    def verify_packed(
        pending: torch.Tensor,
        drafts: torch.Tensor,
        proposal_counts: torch.Tensor,
        active_rows: torch.Tensor,
        kv_lengths: torch.Tensor,
        preceding_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert tuple(int(token) for token in pending) == (1,)
        assert tuple(bool(active) for active in active_rows) == (True,)
        assert tuple(int(length) for length in kv_lengths) == (2,)
        probabilities = torch.stack(
            [one_hot(2, vocab_size) for _ in range(verify_width)]
        ).unsqueeze(0)
        hidden = torch.zeros((1, verify_width, 1), dtype=torch.float64)
        return probabilities, hidden

    def uniforms(
        round_index: int, proposal_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((1, verify_width - 1), dtype=torch.float64),
            torch.zeros((1, verify_width), dtype=torch.float64),
        )

    result = run_speculative_frame_reference(
        initial_codes=torch.tensor([[0, 1]], dtype=torch.long),
        initial_hidden_before_pending=torch.zeros((1, 1), dtype=torch.float64),
        initial_embedding_sum=torch.tensor([[2.0]], dtype=torch.float64),
        codec_embedding_tables=torch.ones((2, vocab_size, 1), dtype=torch.float64),
        num_code_groups=3,
        verify_width=verify_width,
        draft_provider=provide_drafts,
        verify_provider=verify_packed,
        uniform_provider=uniforms,
    )

    assert result.codes == ((0, 1, 2),)
    assert result.rounds == 1
    assert result.proposed_draft_tokens == 0
    assert result.accepted_draft_tokens == 0
    assert result.bonus_tokens == 1
    assert result.summed_embeddings.item() == 3.0


def test_current_model_frame_commits_exactly_16_codes() -> None:
    batch_size = 1
    verify_width = 8
    num_groups = 16
    vocab_size = 2
    initial_codes = torch.zeros((batch_size, 2), dtype=torch.long)
    initial_hidden = torch.zeros((batch_size, 1), dtype=torch.float64)
    initial_sum = torch.zeros((batch_size, 1), dtype=torch.float64)
    embedding_tables = torch.ones(
        (num_groups - 1, vocab_size, 1), dtype=torch.float64
    )

    def provide_drafts(
        states: list[SpeculativeRowState], proposal_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distributions = torch.zeros(
            (batch_size, verify_width - 1, vocab_size), dtype=torch.float64
        )
        distributions[:, :, 0] = 1.0
        tokens = torch.zeros((batch_size, verify_width - 1), dtype=torch.long)
        return distributions, tokens

    def verify_packed(
        pending: torch.Tensor,
        drafts: torch.Tensor,
        proposal_counts: torch.Tensor,
        active_rows: torch.Tensor,
        kv_lengths: torch.Tensor,
        preceding_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.zeros(
            (batch_size, verify_width, vocab_size), dtype=torch.float64
        )
        probabilities[:, :, 0] = 1.0
        for row, count_tensor in enumerate(proposal_counts):
            probabilities[row, int(count_tensor), 0] = 0.0
            probabilities[row, int(count_tensor), 1] = 1.0
        hidden = torch.zeros((batch_size, verify_width, 1), dtype=torch.float64)
        return probabilities, hidden

    def uniforms(
        round_index: int, proposal_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((batch_size, verify_width - 1), dtype=torch.float64),
            torch.zeros((batch_size, verify_width), dtype=torch.float64),
        )

    result = run_speculative_frame_reference(
        initial_codes=initial_codes,
        initial_hidden_before_pending=initial_hidden,
        initial_embedding_sum=initial_sum,
        codec_embedding_tables=embedding_tables,
        num_code_groups=num_groups,
        verify_width=verify_width,
        draft_provider=provide_drafts,
        verify_provider=verify_packed,
        uniform_provider=uniforms,
    )

    assert len(result.codes[0]) == num_groups
    assert result.rounds == 2
    assert result.codes[0][-1] == 1
    assert result.summed_embeddings.item() == num_groups - 2


def test_invalid_zero_probability_proposal_is_rejected_before_ratio() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        speculative_acceptance_cpu(
            torch.tensor([0]),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )


def test_categorical_sampler_never_returns_zero_mass_tokens() -> None:
    probabilities = torch.tensor([0.0, 0.2, 0.0, 0.8], dtype=torch.float64)
    assert sample_categorical_cpu(probabilities, 0.0) == 1
    assert sample_categorical_cpu(probabilities, 0.2) == 3


def test_speculation_width_defaults_off_and_validates_range() -> None:
    assert not speculative_resources_requested(0)
    assert speculative_resources_requested(2)
    assert speculative_resources_requested(8)
    for invalid_width in (1, 9, -1, 2.0, True):
        with pytest.raises(ValueError):
            speculative_resources_requested(invalid_width)


def test_capability_resolution_reports_platform_sampling_capacity_fallbacks() -> None:
    base = {
        "requested_width": 4,
        "is_cuda": True,
        "is_bfloat16": True,
        "tensor_parallel_size": 1,
        "fused_predictor_available": True,
        "batch_size": 4,
        "packed_row_capacity": 64,
        "query_heads": 16,
        "kv_heads": 8,
        "attention_tile_capacity": 16,
        "vocab_size": 2048,
        "graph_available": True,
        "sampling_supported": True,
        "has_explicit_seed": False,
        "has_greedy_rows": False,
        "deterministic_mode": False,
    }
    supported = resolve_speculative_capability(**base)
    assert supported.effective_width == 4
    assert supported.fallback_reason is None

    cpu = resolve_speculative_capability(**{**base, "is_cuda": False})
    assert cpu.effective_width == 0
    assert cpu.fallback_reason == "unsupported-device-backend"

    sampling = resolve_speculative_capability(
        **{**base, "sampling_supported": False}
    )
    assert sampling.effective_width == 0
    assert sampling.fallback_reason == "unsupported-sampling-configuration"

    capacity = resolve_speculative_capability(
        **{**base, "packed_row_capacity": 15}
    )
    assert capacity.effective_width == 0
    assert capacity.fallback_reason == "packed-row-capacity-exceeded"

    seeded = resolve_speculative_capability(
        **{**base, "has_explicit_seed": True}
    )
    assert seeded.effective_width == 0
    assert seeded.fallback_reason == "seeded-request-requires-native-decoding"

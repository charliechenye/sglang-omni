# SPDX-License-Identifier: Apache-2.0
"""CPU correctness gates for the CodePredictor speculative reference path."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_tts.speculative_runtime import (
    accept_speculative_candidates,
    expand_compact_probability_rows,
    sample_compact_probability_rows,
    speculative_sampling_fallback_reason,
)
from sglang_omni.models.qwen3_tts.speculative_prepare import (
    SpeculativeVerifierInputs,
)
from sglang_omni.models.qwen3_tts.speculative_workspace import (
    PredictorSpeculativeWorkspace,
)
from sglang_omni.utils.predictor_grouped_heads import grouped_lm_head_linear
from sglang_omni.utils.predictor_speculation import (
    SpeculativeRowState,
    acceptance_probability,
    is_fatal_cuda_capture_error,
    probabilities_from_logits_cpu,
    residual_distribution_cpu,
    resolve_speculative_capability,
    resolve_speculative_round_width,
    run_speculative_frame_reference,
    sample_categorical_cpu,
    speculative_acceptance_cpu,
    speculative_resources_requested,
)


def one_hot(token: int, vocab_size: int) -> torch.Tensor:
    probabilities = torch.zeros(vocab_size, dtype=torch.float64)
    probabilities[token] = 1.0
    return probabilities


def test_cpu_grouped_lm_head_selects_rows_and_masks_padding() -> None:
    torch.manual_seed(5)
    hidden_states = torch.randn(3, 7)
    head_weights = torch.randn(4, 11, 7)
    hidden_indices = torch.tensor([2, 0, 1, 99])
    head_indices = torch.tensor([1, 3, 0, 99])
    active_rows = torch.tensor([True, True, False, False])

    actual = grouped_lm_head_linear(
        hidden_states, head_weights, hidden_indices, head_indices, active_rows
    )
    expected = torch.stack(
        [
            torch.nn.functional.linear(hidden_states[2], head_weights[1]),
            torch.nn.functional.linear(hidden_states[0], head_weights[3]),
            torch.zeros(head_weights.shape[1]),
            torch.zeros(head_weights.shape[1]),
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_cpu_grouped_lm_head_writes_to_persistent_output() -> None:
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    head_weights = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    hidden_indices = torch.tensor([2, 0])
    head_indices = torch.tensor([1, 0])
    active_rows = torch.tensor([True, True])
    output = torch.empty((2, 5), dtype=torch.float32)

    actual = grouped_lm_head_linear(
        hidden_states,
        head_weights,
        hidden_indices,
        head_indices,
        active_rows,
        output=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(
        actual,
        torch.stack(
            [
                torch.nn.functional.linear(hidden_states[2], head_weights[1]),
                torch.nn.functional.linear(hidden_states[0], head_weights[0]),
            ]
        ),
    )


def test_cpu_speculative_workspace_keeps_round_pointers_for_mixed_widths() -> None:
    max_batch_size = 3
    max_verify_width = 4
    hidden_size = 8
    verifier_inputs = {
        verify_width: SpeculativeVerifierInputs(
            token_embeddings=torch.empty(
                max_batch_size, verify_width, hidden_size, dtype=torch.bfloat16
            ),
            positions=torch.empty(max_batch_size * verify_width, dtype=torch.long),
            target_head_indices=torch.empty(
                max_batch_size, verify_width, dtype=torch.long
            ),
            valid_queries=torch.empty(
                max_batch_size, verify_width, dtype=torch.bool
            ),
        )
        for verify_width in range(1, max_verify_width + 1)
    }
    workspace = PredictorSpeculativeWorkspace(
        max_batch_size=max_batch_size,
        max_verify_width=max_verify_width,
        vocab_size=16,
        hidden_size=hidden_size,
        hidden_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        max_top_k=50,
        candidate_width=64,
        split_top_k_chunks=8,
        seed_rows=torch.zeros(max_batch_size, dtype=torch.long),
        round_index=torch.zeros((), dtype=torch.long),
        uniforms=torch.empty(max_batch_size, 4, max_verify_width),
        rng_pool=None,
        metadata_device=torch.empty(max_batch_size, 9, dtype=torch.long),
        metadata_host=torch.empty(max_batch_size, 9, dtype=torch.long),
        verifier_input_buffers=verifier_inputs,
    )
    round_workspace = workspace.rounds[3]
    stable_pointers = (
        round_workspace.proposal_logits.data_ptr(),
        round_workspace.target_logits.data_ptr(),
        round_workspace.proposal_probabilities.data_ptr(),
        round_workspace.target_probabilities.data_ptr(),
        round_workspace.proposal_tokens.data_ptr(),
        round_workspace.acceptance_outputs.committed_tokens.data_ptr(),
        round_workspace.verifier_inputs.token_embeddings.data_ptr(),
    )
    live_batch_size = 2
    live_proposal_rows = live_batch_size * round_workspace.proposal_width
    live_target_rows = live_batch_size * round_workspace.verify_width

    assert workspace.rounds[1].proposal_tokens.shape == (max_batch_size, 0)
    assert round_workspace.proposal_logits[:live_proposal_rows].is_contiguous()
    assert round_workspace.target_logits[:live_target_rows].is_contiguous()
    assert round_workspace.split_candidate_keys is not None
    assert workspace.candidate_width == 64
    assert workspace.memory_bytes > 0
    assert stable_pointers == (
        round_workspace.proposal_logits.data_ptr(),
        round_workspace.target_logits.data_ptr(),
        round_workspace.proposal_probabilities.data_ptr(),
        round_workspace.target_probabilities.data_ptr(),
        round_workspace.proposal_tokens.data_ptr(),
        round_workspace.acceptance_outputs.committed_tokens.data_ptr(),
        round_workspace.verifier_inputs.token_embeddings.data_ptr(),
    )


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


def test_probability_reference_accepts_native_selected_tie_order() -> None:
    logits = torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    native_candidate_order = torch.tensor([0, 3, 1, 2], dtype=torch.long)

    stable_reference = probabilities_from_logits_cpu(
        logits,
        temperature=1.0,
        top_k=4,
        top_p=0.75,
    )
    selected_native_order = probabilities_from_logits_cpu(
        logits,
        temperature=1.0,
        top_k=4,
        top_p=0.75,
        ordered_token_ids=native_candidate_order,
    )

    assert stable_reference[3] == 0.0
    assert selected_native_order[3] > 0.0
    assert selected_native_order[2] == 0.0
    assert selected_native_order.sum() == pytest.approx(1.0)


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
        torch.stack([one_hot(token, vocab_size) for token in (0, 1, 2, 5)]),
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
        proposal_uniform, acceptance_uniform = torch.rand(
            2, generator=generator, dtype=torch.float64
        )
        draft_token = torch.tensor(
            [sample_categorical_cpu(draft, float(proposal_uniform))], dtype=torch.long
        )
        sampling_uniforms = torch.rand(2, generator=generator, dtype=torch.float64)
        decision = speculative_acceptance_cpu(
            draft_token,
            draft_probabilities,
            target_probabilities,
            acceptance_uniform.unsqueeze(0),
            sampling_uniforms,
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
        assert all(
            state.valid_kv_length == state.committed_predictor_code_count + 1
            for state in states
        )
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
        valid_query_counts: torch.Tensor,
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
        assert tuple(int(count) for count in valid_query_counts) == (
            (3, 3, 3) if verify_round == 0 else (2, 0, 1)
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
    assert result.accepted_draft_tokens == 4
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
        assert states[0].valid_proposal_count == 0
        assert states[0].valid_query_count == 1
        assert tuple(int(count) for count in proposal_counts) == (0,)
        return (
            torch.zeros((1, verify_width - 1, vocab_size), dtype=torch.float64),
            torch.zeros((1, verify_width - 1), dtype=torch.long),
        )

    def verify_packed(
        pending: torch.Tensor,
        drafts: torch.Tensor,
        proposal_counts: torch.Tensor,
        valid_query_counts: torch.Tensor,
        active_rows: torch.Tensor,
        kv_lengths: torch.Tensor,
        preceding_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert tuple(int(token) for token in pending) == (1,)
        assert tuple(int(count) for count in valid_query_counts) == (1,)
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
    embedding_tables = torch.ones((num_groups - 1, vocab_size, 1), dtype=torch.float64)

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
        valid_query_counts: torch.Tensor,
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


def test_speculative_width_shrinks_from_staged_heterogeneous_depths() -> None:
    committed_depths = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
    active_rows = torch.tensor([True, True, True, True, False])
    selected_widths = [
        resolve_speculative_round_width(
            configured_width=4,
            backend_capacity_width=4,
            num_code_groups=8,
            committed_depths=torch.tensor([depth], dtype=torch.long),
            active_rows=torch.tensor([True]),
        )
        for depth in (3, 4, 5, 6)
    ]

    assert selected_widths == [4, 3, 2, 1]
    assert (
        resolve_speculative_round_width(
            configured_width=4,
            backend_capacity_width=3,
            num_code_groups=8,
            committed_depths=committed_depths,
            active_rows=active_rows,
        )
        == 3
    )
    assert (
        resolve_speculative_round_width(
            configured_width=4,
            backend_capacity_width=4,
            num_code_groups=8,
            committed_depths=committed_depths,
            active_rows=torch.zeros_like(active_rows),
        )
        == 0
    )


def test_speculative_width_uses_host_rows_after_metadata_transfer() -> None:
    assert (
        resolve_speculative_round_width(
            configured_width=4,
            backend_capacity_width=4,
            num_code_groups=8,
            committed_depths=[4, 5, 6],
            active_rows=[True, True, False],
        )
        == 3
    )


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
        "grouped_lm_head_supported": True,
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

    sampling = resolve_speculative_capability(**{**base, "sampling_supported": False})
    assert sampling.effective_width == 0
    assert sampling.fallback_reason == "unsupported-sampling-configuration"

    grouped_head = resolve_speculative_capability(
        **{**base, "grouped_lm_head_supported": False}
    )
    assert grouped_head.effective_width == 0
    assert grouped_head.fallback_reason == "grouped-lm-head-unavailable"

    capacity = resolve_speculative_capability(**{**base, "packed_row_capacity": 15})
    assert capacity.effective_width == 0
    assert capacity.fallback_reason == "packed-row-capacity-exceeded"

    seeded = resolve_speculative_capability(**{**base, "has_explicit_seed": True})
    assert seeded.effective_width == 0
    assert seeded.fallback_reason == "seeded-request-requires-native-decoding"


@pytest.mark.parametrize(
    "error_message",
    [
        "CUDA error: an illegal memory access was encountered",
        "device-side assert triggered",
        "unspecified launch failure",
        "CUDA error 700",
        "CUDA error 719",
        "CUDA context is destroyed",
    ],
)
def test_fatal_cuda_capture_errors_are_not_retryable(error_message: str) -> None:
    assert is_fatal_cuda_capture_error(RuntimeError(error_message))


def test_ordinary_capture_error_can_use_narrower_graph() -> None:
    assert not is_fatal_cuda_capture_error(
        RuntimeError("capture shape is unsupported at requested width")
    )


def test_batched_speculative_decision_handles_zero_tail_rejection_and_bonus() -> None:
    draft_tokens = torch.tensor([[0, 0], [0, 0], [0, 1]], dtype=torch.long)
    draft_probabilities = torch.zeros((3, 2, 4), dtype=torch.float32)
    draft_probabilities[:, :, 0] = 1.0
    draft_probabilities[2, 1].zero_()
    draft_probabilities[2, 1, 1] = 1.0
    target_probabilities = torch.zeros((3, 3, 4), dtype=torch.float32)
    target_probabilities[0, 0, 2] = 1.0
    target_probabilities[1, 0, 0] = 1.0
    target_probabilities[1, 1, 3] = 1.0
    target_probabilities[2, 0, 0] = 1.0
    target_probabilities[2, 1, 2] = 1.0
    target_probabilities[2, 2, 1] = 1.0

    decision = accept_speculative_candidates(
        draft_tokens,
        draft_probabilities,
        target_probabilities,
        torch.tensor([0, 1, 2]),
        torch.tensor([True, True, True]),
        torch.tensor([[0.5, 0.5], [0.1, 0.5], [0.1, 0.5]]),
        torch.zeros((3, 3)),
        torch.zeros((3, 3)),
    )

    assert decision.committed_tokens.tolist() == [[2, 0, 0], [0, 3, 0], [0, 2, 0]]
    assert decision.accepted_draft_counts.tolist() == [0, 1, 1]
    assert decision.committed_token_counts.tolist() == [1, 2, 2]
    assert decision.bonus_rows.tolist() == [True, True, False]
    assert decision.rejected_rows.tolist() == [False, False, True]


def test_compact_probability_expansion_and_request_eligibility() -> None:
    probabilities = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    token_ids = torch.tensor([[3, 1], [2, 0]])
    dense = expand_compact_probability_rows(probabilities, token_ids, 5)
    assert dense.tolist() == [[0.0, 0.75, 0.0, 0.25, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]

    ordinary_unseeded = speculative_sampling_fallback_reason(
        ("request-a", "request-b"),
        (False, False),
        (True, True),
        has_unbounded_top_k=False,
        deterministic_mode=False,
    )
    assert ordinary_unseeded is None
    explicit_seed = speculative_sampling_fallback_reason(
        ("request-a",),
        (True,),
        (True,),
        has_unbounded_top_k=False,
        deterministic_mode=False,
    )
    assert explicit_seed == "explicit-seed-requires-native-decoding"
    mixed_seed_batch = speculative_sampling_fallback_reason(
        ("request-a", "request-b"),
        (False, True),
        (True, True),
        has_unbounded_top_k=False,
        deterministic_mode=False,
    )
    assert mixed_seed_batch == "explicit-seed-requires-native-decoding"
    mixed_sampling = speculative_sampling_fallback_reason(
        ("request-a", "request-b"),
        (False, False),
        (True, False),
        has_unbounded_top_k=False,
        deterministic_mode=False,
    )
    assert mixed_sampling == "greedy-or-mixed-sampling-requires-native-decoding"


def test_compact_sampler_clamps_rounding_tail_to_last_positive_candidate() -> None:
    probabilities = torch.tensor([[0.1, 0.2, 0.6999999, 0.0]])
    token_ids = torch.tensor([[11, 13, 17, 19]], dtype=torch.long)
    uniforms = torch.tensor([0.99999994], dtype=torch.float32)
    active_rows = torch.tensor([True])

    actual = sample_compact_probability_rows(
        probabilities, token_ids, uniforms, active_rows
    )

    assert actual.tolist() == [17]

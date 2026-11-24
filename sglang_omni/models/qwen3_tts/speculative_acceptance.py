# SPDX-License-Identifier: Apache-2.0
"""Fused compact acceptance for Qwen3-TTS speculative rounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def sample_compact_probability_rows_kernel(
        probabilities,
        token_ids,
        uniforms,
        active_rows,
        sampled_tokens,
        PROPOSAL_WIDTH: tl.constexpr,
        CANDIDATE_COUNT: tl.constexpr,
        UNIFORM_STRIDE_ROW: tl.constexpr,
        UNIFORM_STRIDE_COLUMN: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidate_offsets = tl.arange(0, BLOCK_CANDIDATES)
        candidate_mask = candidate_offsets < CANDIDATE_COUNT
        batch_row = row // PROPOSAL_WIDTH
        proposal_offset = row % PROPOSAL_WIDTH
        uniform = tl.load(
            uniforms
            + batch_row * UNIFORM_STRIDE_ROW
            + proposal_offset * UNIFORM_STRIDE_COLUMN
        )
        is_active = tl.load(active_rows + row).to(tl.int1)
        row_probabilities = tl.load(
            probabilities + row * CANDIDATE_COUNT + candidate_offsets,
            mask=candidate_mask,
            other=0.0,
        ).to(tl.float32)
        cumulative = tl.cumsum(row_probabilities, axis=0)
        selected_rank = tl.sum((cumulative <= uniform).to(tl.int32), axis=0)
        last_positive_rank = tl.max(
            tl.where(row_probabilities > 0.0, candidate_offsets, 0), axis=0
        )
        selected_rank = tl.minimum(selected_rank, last_positive_rank)
        selected_rank = tl.minimum(selected_rank, CANDIDATE_COUNT - 1)
        row_token_ids = tl.load(
            token_ids + row * CANDIDATE_COUNT + candidate_offsets,
            mask=candidate_mask,
            other=0,
        ).to(tl.int32)
        selected_token = tl.sum(
            tl.where(candidate_offsets == selected_rank, row_token_ids, 0), axis=0
        )
        tl.store(sampled_tokens + row, tl.where(is_active, selected_token, 0))

    @triton.jit
    def compact_speculative_acceptance_kernel(
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
        committed_tokens,
        accepted_draft_counts,
        committed_token_counts,
        rejected_rows,
        bonus_rows,
        DRAFT_WIDTH: tl.constexpr,
        VERIFY_WIDTH: tl.constexpr,
        CANDIDATE_COUNT: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        ACCEPTANCE_UNIFORM_STRIDE_ROW: tl.constexpr,
        ACCEPTANCE_UNIFORM_STRIDE_COLUMN: tl.constexpr,
        CORRECTION_UNIFORM_STRIDE_ROW: tl.constexpr,
        CORRECTION_UNIFORM_STRIDE_COLUMN: tl.constexpr,
        BONUS_UNIFORM_STRIDE_ROW: tl.constexpr,
        BONUS_UNIFORM_STRIDE_COLUMN: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidate_offsets = tl.arange(0, BLOCK_CANDIDATES)
        candidate_mask = candidate_offsets < CANDIDATE_COUNT
        is_active = tl.load(active_rows + row).to(tl.int1)
        proposal_count = tl.load(proposal_counts + row).to(tl.int32)
        first_rejection = DRAFT_WIDTH
        has_rejection = False

        for draft_offset in tl.static_range(0, DRAFT_WIDTH):
            draft_token = tl.load(
                draft_tokens + row * DRAFT_WIDTH + draft_offset
            ).to(tl.int32)
            draft_row = row * DRAFT_WIDTH + draft_offset
            target_row = row * VERIFY_WIDTH + draft_offset
            draft_ids = tl.load(
                draft_token_ids + draft_row * CANDIDATE_COUNT + candidate_offsets,
                mask=candidate_mask,
                other=-1,
            ).to(tl.int32)
            draft_values = tl.load(
                draft_probabilities
                + draft_row * CANDIDATE_COUNT
                + candidate_offsets,
                mask=candidate_mask,
                other=0.0,
            ).to(tl.float32)
            target_ids = tl.load(
                target_token_ids + target_row * CANDIDATE_COUNT + candidate_offsets,
                mask=candidate_mask,
                other=-1,
            ).to(tl.int32)
            target_values = tl.load(
                target_probabilities
                + target_row * CANDIDATE_COUNT
                + candidate_offsets,
                mask=candidate_mask,
                other=0.0,
            ).to(tl.float32)
            draft_token_probability = tl.sum(
                tl.where(draft_ids == draft_token, draft_values, 0.0), axis=0
            )
            target_token_probability = tl.sum(
                tl.where(target_ids == draft_token, target_values, 0.0), axis=0
            )
            safe_draft_probability = tl.where(
                draft_token_probability > 0.0, draft_token_probability, 1.0
            )
            acceptance_ratio = tl.where(
                draft_token_probability > 0.0,
                tl.minimum(1.0, target_token_probability / safe_draft_probability),
                0.0,
            )
            acceptance_uniform = tl.load(
                acceptance_uniforms
                + row * ACCEPTANCE_UNIFORM_STRIDE_ROW
                + draft_offset * ACCEPTANCE_UNIFORM_STRIDE_COLUMN
            )
            draft_is_valid = is_active & (draft_offset < proposal_count)
            rejected = draft_is_valid & (acceptance_uniform >= acceptance_ratio)
            first_rejection = tl.minimum(
                first_rejection,
                tl.where(rejected, draft_offset, DRAFT_WIDTH),
            )
            has_rejection = has_rejection | rejected

        accepted_count = tl.where(
            has_rejection, first_rejection, proposal_count
        ).to(tl.int32)
        accepted_count = tl.where(is_active, accepted_count, 0)
        correction_row = row * VERIFY_WIDTH + accepted_count
        target_ids = tl.load(
            target_token_ids + correction_row * CANDIDATE_COUNT + candidate_offsets,
            mask=candidate_mask,
            other=2147483647,
        ).to(tl.int32)
        target_values = tl.load(
            target_probabilities
            + correction_row * CANDIDATE_COUNT
            + candidate_offsets,
            mask=candidate_mask,
            other=0.0,
        ).to(tl.float32)
        safe_draft_offset = tl.minimum(accepted_count, DRAFT_WIDTH - 1)
        draft_row = row * DRAFT_WIDTH + safe_draft_offset
        draft_ids = tl.load(
            draft_token_ids + draft_row * CANDIDATE_COUNT + candidate_offsets,
            mask=candidate_mask,
            other=2147483647,
        ).to(tl.int32)
        draft_values = tl.load(
            draft_probabilities + draft_row * CANDIDATE_COUNT + candidate_offsets,
            mask=candidate_mask,
            other=0.0,
        ).to(tl.float32)

        lower_bounds = tl.full((BLOCK_CANDIDATES,), 0, tl.int32)
        upper_bounds = tl.full((BLOCK_CANDIDATES,), CANDIDATE_COUNT, tl.int32)
        for _ in tl.static_range(0, SEARCH_STEPS):
            middle = (lower_bounds + upper_bounds) // 2
            safe_middle = tl.minimum(middle, CANDIDATE_COUNT - 1)
            middle_ids = tl.gather(draft_ids, safe_middle, axis=0)
            search_right = (middle < CANDIDATE_COUNT) & (middle_ids < target_ids)
            lower_bounds = tl.where(search_right, middle + 1, lower_bounds)
            upper_bounds = tl.where(search_right, upper_bounds, middle)
        else:
            pass
        matched_ranks = tl.minimum(lower_bounds, CANDIDATE_COUNT - 1)
        matched_ids = tl.gather(draft_ids, matched_ranks, axis=0)
        matched_values = tl.gather(draft_values, matched_ranks, axis=0)
        draft_values_for_target = tl.where(
            matched_ids == target_ids, matched_values, 0.0
        )
        residual_values = tl.maximum(target_values - draft_values_for_target, 0.0)
        residual_mass = tl.sum(residual_values, axis=0)
        use_residual = has_rejection & (residual_mass > 0.0)
        selected_values = tl.where(use_residual, residual_values, target_values)
        selected_mass = tl.sum(selected_values, axis=0)
        safe_selected_mass = tl.maximum(selected_mass, 1.1754943508222875e-38)
        normalized_values = selected_values / safe_selected_mass
        cumulative_values = tl.cumsum(normalized_values, axis=0)
        correction_uniform = tl.load(
            correction_uniforms
            + row * CORRECTION_UNIFORM_STRIDE_ROW
            + accepted_count * CORRECTION_UNIFORM_STRIDE_COLUMN
        )
        bonus_uniform = tl.load(
            bonus_uniforms
            + row * BONUS_UNIFORM_STRIDE_ROW
            + accepted_count * BONUS_UNIFORM_STRIDE_COLUMN
        )
        selected_uniform = tl.where(has_rejection, correction_uniform, bonus_uniform)
        selected_rank = tl.sum(
            (cumulative_values <= selected_uniform).to(tl.int32), axis=0
        )
        last_positive_rank = tl.max(
            tl.where(normalized_values > 0.0, candidate_offsets, 0), axis=0
        )
        selected_rank = tl.minimum(selected_rank, last_positive_rank)
        final_token = tl.gather(target_ids, selected_rank, axis=0)
        final_token = tl.where(is_active, final_token, 0)

        output_offsets = tl.arange(0, BLOCK_WIDTH)
        draft_output_mask = output_offsets < DRAFT_WIDTH
        committed_drafts = tl.load(
            draft_tokens + row * DRAFT_WIDTH + tl.minimum(output_offsets, DRAFT_WIDTH - 1),
            mask=draft_output_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            committed_tokens + row * VERIFY_WIDTH + output_offsets,
            tl.where(output_offsets < accepted_count, committed_drafts, 0),
            mask=output_offsets < VERIFY_WIDTH,
        )
        tl.store(
            committed_tokens + row * VERIFY_WIDTH + accepted_count,
            final_token.to(tl.int64),
            mask=is_active,
        )
        tl.store(accepted_draft_counts + row, accepted_count)
        tl.store(
            committed_token_counts + row,
            tl.where(is_active, accepted_count + 1, 0),
        )
        tl.store(rejected_rows + row, (has_rejection & is_active).to(tl.int64))
        tl.store(bonus_rows + row, ((~has_rejection) & is_active).to(tl.int64))


else:
    sample_compact_probability_rows_kernel = None
    compact_speculative_acceptance_kernel = None


@dataclass(frozen=True)
class SpeculativeAcceptanceBatch:
    committed_tokens: torch.Tensor
    accepted_draft_counts: torch.Tensor
    committed_token_counts: torch.Tensor
    rejected_rows: torch.Tensor
    bonus_rows: torch.Tensor


def accept_speculative_candidates_compact(
    draft_tokens: torch.Tensor,
    draft_probabilities: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probabilities: torch.Tensor,
    target_token_ids: torch.Tensor,
    proposal_counts: torch.Tensor,
    active_rows: torch.Tensor,
    acceptance_uniforms: torch.Tensor,
    correction_uniforms: torch.Tensor,
    bonus_uniforms: torch.Tensor,
) -> SpeculativeAcceptanceBatch:
    """Run compact first-rejection and residual sampling in one Triton launch."""
    if draft_tokens.ndim != 2 or draft_probabilities.ndim != 2:
        raise ValueError("compact speculative draft tensors have invalid ranks")
    else:
        pass
    batch_size, draft_width = draft_tokens.shape
    if draft_width <= 0 or draft_probabilities.shape[0] != batch_size * draft_width:
        raise ValueError("compact speculative draft rows do not match draft tokens")
    else:
        pass
    candidate_count = draft_probabilities.shape[1]
    verify_width = draft_width + 1
    if candidate_count <= 0:
        raise ValueError("compact speculative supports must contain candidates")
    elif draft_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("draft tokens must use an integer dtype")
    elif draft_token_ids.shape != draft_probabilities.shape:
        raise ValueError("draft token IDs do not match compact probabilities")
    elif target_probabilities.ndim != 2 or target_probabilities.shape[0] != batch_size * verify_width:
        raise ValueError("compact target rows must include a trailing bonus row")
    elif target_token_ids.shape != target_probabilities.shape:
        raise ValueError("target token IDs do not match compact probabilities")
    elif target_probabilities.shape[1] != candidate_count:
        raise ValueError("draft and target compact supports must have equal widths")
    elif proposal_counts.shape != (batch_size,) or active_rows.shape != (batch_size,):
        raise ValueError("proposal metadata must contain one value per batch row")
    elif proposal_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("proposal counts must use an integer dtype")
    elif active_rows.dtype is not torch.bool:
        raise ValueError("active-row metadata must use a boolean dtype")
    elif acceptance_uniforms.shape != (batch_size, draft_width):
        raise ValueError("acceptance uniforms must match the draft window")
    elif correction_uniforms.shape != (batch_size, verify_width):
        raise ValueError("correction uniforms must match the verify window")
    elif bonus_uniforms.shape != (batch_size, verify_width):
        raise ValueError("bonus uniforms must match the verify window")
    else:
        pass
    round_tensors = (
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
    if any(tensor.device != draft_tokens.device for tensor in round_tensors):
        raise ValueError("speculative round tensors must share one device")
    elif draft_probabilities.dtype is not torch.float32 or target_probabilities.dtype is not torch.float32:
        raise ValueError("speculative compact probabilities must use float32")
    elif (
        acceptance_uniforms.dtype is not torch.float32
        or correction_uniforms.dtype is not torch.float32
        or bonus_uniforms.dtype is not torch.float32
    ):
        raise ValueError("speculative uniforms must use float32")
    elif draft_token_ids.dtype not in (torch.int32, torch.int64) or target_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compact token IDs must use an integer dtype")
    elif not draft_tokens.is_contiguous():
        raise ValueError("draft token rows must be contiguous")
    elif not proposal_counts.is_contiguous() or not active_rows.is_contiguous():
        raise ValueError("proposal metadata must be contiguous")
    elif not draft_tokens.is_cuda or compact_speculative_acceptance_kernel is None:
        raise RuntimeError("compact speculative acceptance requires CUDA and Triton")
    else:
        pass

    sorted_draft_token_ids, draft_sort_order = torch.sort(
        draft_token_ids.to(torch.int32), dim=-1
    )
    sorted_draft_probabilities = draft_probabilities.gather(1, draft_sort_order)
    sorted_target_token_ids, target_sort_order = torch.sort(
        target_token_ids.to(torch.int32), dim=-1
    )
    sorted_target_probabilities = target_probabilities.gather(1, target_sort_order)
    committed_tokens = torch.empty(
        (batch_size, verify_width), device=draft_tokens.device, dtype=torch.long
    )
    accepted_draft_counts = torch.empty_like(proposal_counts)
    committed_token_counts = torch.empty_like(proposal_counts)
    rejected_rows = torch.empty_like(proposal_counts)
    bonus_rows = torch.empty_like(proposal_counts)
    compact_speculative_acceptance_kernel[(batch_size,)](
        draft_tokens,
        sorted_draft_probabilities,
        sorted_draft_token_ids,
        sorted_target_probabilities,
        sorted_target_token_ids,
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
        committed_tokens,
        accepted_draft_counts,
        committed_token_counts,
        rejected_rows,
        bonus_rows,
        DRAFT_WIDTH=draft_width,
        VERIFY_WIDTH=verify_width,
        CANDIDATE_COUNT=candidate_count,
        BLOCK_WIDTH=triton.next_power_of_2(verify_width),
        BLOCK_CANDIDATES=triton.next_power_of_2(candidate_count),
        SEARCH_STEPS=candidate_count.bit_length(),
        ACCEPTANCE_UNIFORM_STRIDE_ROW=acceptance_uniforms.stride(0),
        ACCEPTANCE_UNIFORM_STRIDE_COLUMN=acceptance_uniforms.stride(1),
        CORRECTION_UNIFORM_STRIDE_ROW=correction_uniforms.stride(0),
        CORRECTION_UNIFORM_STRIDE_COLUMN=correction_uniforms.stride(1),
        BONUS_UNIFORM_STRIDE_ROW=bonus_uniforms.stride(0),
        BONUS_UNIFORM_STRIDE_COLUMN=bonus_uniforms.stride(1),
        num_warps=4,
    )
    return SpeculativeAcceptanceBatch(
        committed_tokens=committed_tokens,
        accepted_draft_counts=accepted_draft_counts,
        committed_token_counts=committed_token_counts,
        rejected_rows=rejected_rows.to(torch.bool),
        bonus_rows=bonus_rows.to(torch.bool),
    )


def sample_compact_probability_rows_fused(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    uniforms: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Sample compact proposal rows with a single inverse-CDF launch."""
    if probabilities.ndim != 2 or token_ids.shape != probabilities.shape:
        raise ValueError("compact probabilities and token IDs must have equal 2D shapes")
    elif uniforms.ndim != 2 or active_rows.shape != uniforms.shape:
        raise ValueError("proposal uniforms and active rows must have equal 2D shapes")
    else:
        pass
    batch_size, proposal_width = uniforms.shape
    row_count, candidate_count = probabilities.shape
    if proposal_width <= 0 or row_count != batch_size * proposal_width:
        raise ValueError("compact probability rows do not match proposal uniforms")
    elif candidate_count <= 0:
        raise ValueError("compact probability rows must contain candidates")
    elif token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compact token IDs must use an integer dtype")
    elif active_rows.dtype is not torch.bool:
        raise ValueError("proposal active rows must use a boolean dtype")
    elif probabilities.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("compact probabilities must use a floating-point dtype")
    elif uniforms.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("proposal uniforms must use a floating-point dtype")
    elif any(
        tensor.device != probabilities.device
        for tensor in (token_ids, uniforms, active_rows)
    ):
        raise ValueError("compact proposal tensors must share one device")
    elif not probabilities.is_cuda or sample_compact_probability_rows_kernel is None:
        raise RuntimeError("compact proposal sampling requires CUDA and Triton")
    elif not probabilities.is_contiguous() or not token_ids.is_contiguous():
        raise ValueError("compact probability rows and token IDs must be contiguous")
    elif not active_rows.is_contiguous():
        raise ValueError("compact proposal active rows must be contiguous")
    else:
        pass
    if row_count == 0:
        return torch.empty((batch_size, proposal_width), device=probabilities.device, dtype=torch.long)
    else:
        pass
    sampled_tokens = torch.empty(
        (row_count,), device=probabilities.device, dtype=torch.long
    )
    sample_compact_probability_rows_kernel[(row_count,)](
        probabilities,
        token_ids,
        uniforms,
        active_rows.view(-1),
        sampled_tokens,
        PROPOSAL_WIDTH=proposal_width,
        CANDIDATE_COUNT=candidate_count,
        UNIFORM_STRIDE_ROW=uniforms.stride(0),
        UNIFORM_STRIDE_COLUMN=uniforms.stride(1),
        BLOCK_CANDIDATES=triton.next_power_of_2(candidate_count),
        num_warps=4,
    )
    return sampled_tokens.view(batch_size, proposal_width)

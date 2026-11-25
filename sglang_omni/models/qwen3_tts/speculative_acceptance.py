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
    def sort_compact_support_by_token_id(
        token_ids,
        probabilities,
        CANDIDATE_COUNT: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
    ):
        candidate_offsets = tl.arange(0, BLOCK_CANDIDATES)
        candidate_mask = candidate_offsets < CANDIDATE_COUNT
        probability_values = tl.where(candidate_mask, probabilities, 0.0).to(tl.float32)
        sorted_token_keys = tl.where(
            candidate_mask & (probability_values > 0.0) & (token_ids >= 0),
            token_ids,
            2147483647,
        ).to(tl.uint64)
        probability_bits = probability_values.to(tl.uint32, bitcast=True)
        descending_probability_bits = (~probability_bits).to(tl.uint64)
        packed_keys = (sorted_token_keys << 32) | descending_probability_bits
        sorted_keys = tl.sort(packed_keys, descending=False)
        sorted_token_ids = (sorted_keys >> 32).to(tl.int32)
        sorted_probability_bits = ~((sorted_keys & 0xFFFFFFFF).to(tl.uint32))
        sorted_probabilities = sorted_probability_bits.to(tl.float32, bitcast=True)
        return sorted_token_ids, sorted_probabilities

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
        invalid_rows,
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
        invalid_distribution = False

        for draft_offset in tl.static_range(0, DRAFT_WIDTH):
            draft_token = tl.load(draft_tokens + row * DRAFT_WIDTH + draft_offset).to(
                tl.int32
            )
            draft_row = row * DRAFT_WIDTH + draft_offset
            target_row = row * VERIFY_WIDTH + draft_offset
            draft_ids = tl.load(
                draft_token_ids + draft_row * CANDIDATE_COUNT + candidate_offsets,
                mask=candidate_mask,
                other=-1,
            ).to(tl.int32)
            draft_values = tl.load(
                draft_probabilities + draft_row * CANDIDATE_COUNT + candidate_offsets,
                mask=candidate_mask,
                other=0.0,
            ).to(tl.float32)
            target_ids = tl.load(
                target_token_ids + target_row * CANDIDATE_COUNT + candidate_offsets,
                mask=candidate_mask,
                other=-1,
            ).to(tl.int32)
            target_values = tl.load(
                target_probabilities + target_row * CANDIDATE_COUNT + candidate_offsets,
                mask=candidate_mask,
                other=0.0,
            ).to(tl.float32)
            draft_mass = tl.sum(draft_values, axis=0)
            target_mass = tl.sum(target_values, axis=0)
            draft_invalid = (
                (
                    tl.sum(
                        ((draft_values < 0.0) | (draft_values != draft_values)).to(
                            tl.int32
                        ),
                        axis=0,
                    )
                    > 0
                )
                | (tl.abs(draft_mass - 1.0) > 1.0e-4)
                | (
                    tl.sum(
                        ((draft_values > 0.0) & (draft_ids < 0)).to(tl.int32), axis=0
                    )
                    > 0
                )
            )
            target_invalid = (
                (
                    tl.sum(
                        ((target_values < 0.0) | (target_values != target_values)).to(
                            tl.int32
                        ),
                        axis=0,
                    )
                    > 0
                )
                | (tl.abs(target_mass - 1.0) > 1.0e-4)
                | (
                    tl.sum(
                        ((target_values > 0.0) & (target_ids < 0)).to(tl.int32), axis=0
                    )
                    > 0
                )
            )
            invalid_distribution = invalid_distribution | (
                (is_active & (draft_offset < proposal_count))
                & (draft_invalid | target_invalid)
            )
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

        accepted_count = tl.where(has_rejection, first_rejection, proposal_count).to(
            tl.int32
        )
        accepted_count = tl.where(is_active, accepted_count, 0)
        correction_row = row * VERIFY_WIDTH + accepted_count
        target_ids = tl.load(
            target_token_ids + correction_row * CANDIDATE_COUNT + candidate_offsets,
            mask=candidate_mask,
            other=2147483647,
        ).to(tl.int32)
        target_values = tl.load(
            target_probabilities + correction_row * CANDIDATE_COUNT + candidate_offsets,
            mask=candidate_mask,
            other=0.0,
        ).to(tl.float32)
        target_mass = tl.sum(target_values, axis=0)
        target_invalid = (
            (
                tl.sum(
                    ((target_values < 0.0) | (target_values != target_values)).to(
                        tl.int32
                    ),
                    axis=0,
                )
                > 0
            )
            | (tl.abs(target_mass - 1.0) > 1.0e-4)
            | (
                tl.sum(((target_values > 0.0) & (target_ids < 0)).to(tl.int32), axis=0)
                > 0
            )
        )
        invalid_distribution = invalid_distribution | (is_active & target_invalid)
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
        if has_rejection:
            residual_draft_row = row * DRAFT_WIDTH + accepted_count
            residual_draft_ids = tl.load(
                draft_token_ids
                + residual_draft_row * CANDIDATE_COUNT
                + candidate_offsets,
                mask=candidate_mask,
                other=-1,
            ).to(tl.int32)
            residual_draft_values = tl.load(
                draft_probabilities
                + residual_draft_row * CANDIDATE_COUNT
                + candidate_offsets,
                mask=candidate_mask,
                other=0.0,
            ).to(tl.float32)
            sorted_draft_ids, sorted_draft_values = sort_compact_support_by_token_id(
                residual_draft_ids,
                residual_draft_values,
                CANDIDATE_COUNT,
                BLOCK_CANDIDATES,
            )
            lower_bounds = tl.full((BLOCK_CANDIDATES,), 0, tl.int32)
            upper_bounds = tl.full((BLOCK_CANDIDATES,), CANDIDATE_COUNT, tl.int32)
            for _ in tl.static_range(0, SEARCH_STEPS):
                middle = (lower_bounds + upper_bounds) // 2
                safe_middle = tl.minimum(middle, CANDIDATE_COUNT - 1)
                middle_ids = tl.gather(sorted_draft_ids, safe_middle, axis=0)
                search_right = (middle < CANDIDATE_COUNT) & (middle_ids < target_ids)
                lower_bounds = tl.where(search_right, middle + 1, lower_bounds)
                upper_bounds = tl.where(search_right, upper_bounds, middle)
            else:
                pass
            matched_ranks = tl.minimum(lower_bounds, CANDIDATE_COUNT - 1)
            matched_ids = tl.gather(sorted_draft_ids, matched_ranks, axis=0)
            matched_values = tl.gather(sorted_draft_values, matched_ranks, axis=0)
            draft_values_for_target = tl.where(
                matched_ids == target_ids, matched_values, 0.0
            )
            selected_values = tl.maximum(target_values - draft_values_for_target, 0.0)
            residual_mass = tl.sum(selected_values.to(tl.float64), axis=0)
            if residual_mass <= 0.0:
                invalid_distribution = invalid_distribution | is_active
            else:
                pass
        else:
            selected_values = target_values
        selected_token_ids = target_ids
        selected_scale = tl.max(selected_values, axis=0).to(tl.float64)
        safe_selected_scale = tl.where(selected_scale > 0.0, selected_scale, 1.0)
        scaled_values = selected_values.to(tl.float64) / safe_selected_scale
        scaled_mass = tl.sum(scaled_values, axis=0)
        safe_scaled_mass = tl.where(scaled_mass > 0.0, scaled_mass, 1.0)
        normalized_values = (scaled_values / safe_scaled_mass).to(tl.float32)
        cumulative_values = tl.cumsum(normalized_values, axis=0)
        selected_rank = tl.sum(
            (cumulative_values <= selected_uniform).to(tl.int32), axis=0
        )
        last_positive_rank = tl.max(
            tl.where(normalized_values > 0.0, candidate_offsets, 0), axis=0
        )
        selected_rank = tl.minimum(selected_rank, last_positive_rank)
        final_token = tl.sum(
            tl.where(
                candidate_offsets == selected_rank,
                selected_token_ids,
                0,
            ),
            axis=0,
        )
        final_token = tl.where(is_active, final_token, 0)
        row_is_invalid = invalid_distribution & is_active
        final_token = tl.where(row_is_invalid, 0, final_token)

        output_offsets = tl.arange(0, BLOCK_WIDTH)
        draft_output_mask = output_offsets < DRAFT_WIDTH
        if DRAFT_WIDTH > 0:
            committed_drafts = tl.load(
                draft_tokens
                + row * DRAFT_WIDTH
                + tl.minimum(output_offsets, DRAFT_WIDTH - 1),
                mask=draft_output_mask,
                other=0,
            ).to(tl.int64)
            tl.store(
                committed_tokens + row * VERIFY_WIDTH + output_offsets,
                tl.where(output_offsets < accepted_count, committed_drafts, 0),
                mask=output_offsets < VERIFY_WIDTH,
            )
        else:
            pass
        tl.store(
            committed_tokens + row * VERIFY_WIDTH + accepted_count,
            final_token.to(tl.int64),
            mask=is_active,
        )
        tl.store(
            committed_token_counts + row,
            tl.where(is_active & (~row_is_invalid), accepted_count + 1, 0),
        )
        accepted_count = tl.where(row_is_invalid, 0, accepted_count)
        tl.store(accepted_draft_counts + row, accepted_count)
        tl.store(rejected_rows + row, has_rejection & is_active)
        tl.store(
            bonus_rows + row,
            (~has_rejection) & is_active & (~row_is_invalid),
        )
        tl.store(invalid_rows + row, row_is_invalid)

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
    invalid_rows: torch.Tensor


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
    outputs: SpeculativeAcceptanceBatch | None = None,
) -> SpeculativeAcceptanceBatch:
    """Run compact first-rejection and residual sampling in one Triton launch."""
    if draft_tokens.ndim != 2 or draft_probabilities.ndim != 2:
        raise ValueError("compact speculative draft tensors have invalid ranks")
    else:
        pass
    batch_size, draft_width = draft_tokens.shape
    if draft_probabilities.shape[0] != batch_size * draft_width:
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
    elif not draft_probabilities.is_contiguous() or not draft_token_ids.is_contiguous():
        raise ValueError("compact draft probability rows must be contiguous")
    elif (
        target_probabilities.ndim != 2
        or target_probabilities.shape[0] != batch_size * verify_width
    ):
        raise ValueError("compact target rows must include a trailing bonus row")
    elif target_token_ids.shape != target_probabilities.shape:
        raise ValueError("target token IDs do not match compact probabilities")
    elif (
        not target_probabilities.is_contiguous() or not target_token_ids.is_contiguous()
    ):
        raise ValueError("compact target probability rows must be contiguous")
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
    elif (
        draft_probabilities.dtype is not torch.float32
        or target_probabilities.dtype is not torch.float32
    ):
        raise ValueError("speculative compact probabilities must use float32")
    elif (
        acceptance_uniforms.dtype is not torch.float32
        or correction_uniforms.dtype is not torch.float32
        or bonus_uniforms.dtype is not torch.float32
    ):
        raise ValueError("speculative uniforms must use float32")
    elif draft_token_ids.dtype not in (
        torch.int32,
        torch.int64,
    ) or target_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compact token IDs must use an integer dtype")
    elif not draft_tokens.is_contiguous():
        raise ValueError("draft token rows must be contiguous")
    elif not proposal_counts.is_contiguous() or not active_rows.is_contiguous():
        raise ValueError("proposal metadata must be contiguous")
    elif not draft_tokens.is_cuda or compact_speculative_acceptance_kernel is None:
        raise RuntimeError("compact speculative acceptance requires CUDA and Triton")
    else:
        pass

    if outputs is None:
        outputs = SpeculativeAcceptanceBatch(
            committed_tokens=torch.empty(
                (batch_size, verify_width),
                device=draft_tokens.device,
                dtype=torch.long,
            ),
            accepted_draft_counts=torch.empty_like(proposal_counts),
            committed_token_counts=torch.empty_like(proposal_counts),
            rejected_rows=torch.empty(
                (batch_size,), device=draft_tokens.device, dtype=torch.bool
            ),
            bonus_rows=torch.empty(
                (batch_size,), device=draft_tokens.device, dtype=torch.bool
            ),
            invalid_rows=torch.empty(
                (batch_size,), device=draft_tokens.device, dtype=torch.bool
            ),
        )
    elif (
        outputs.committed_tokens.shape != (batch_size, verify_width)
        or outputs.accepted_draft_counts.shape != (batch_size,)
        or outputs.committed_token_counts.shape != (batch_size,)
        or outputs.rejected_rows.shape != (batch_size,)
        or outputs.bonus_rows.shape != (batch_size,)
        or outputs.invalid_rows.shape != (batch_size,)
        or outputs.committed_tokens.device != draft_tokens.device
        or outputs.accepted_draft_counts.device != draft_tokens.device
        or outputs.committed_token_counts.device != draft_tokens.device
        or outputs.rejected_rows.device != draft_tokens.device
        or outputs.bonus_rows.device != draft_tokens.device
        or outputs.invalid_rows.device != draft_tokens.device
        or outputs.committed_tokens.dtype is not torch.long
        or outputs.accepted_draft_counts.dtype is not proposal_counts.dtype
        or outputs.committed_token_counts.dtype is not proposal_counts.dtype
        or outputs.rejected_rows.dtype is not torch.bool
        or outputs.bonus_rows.dtype is not torch.bool
        or outputs.invalid_rows.dtype is not torch.bool
        or not outputs.committed_tokens.is_contiguous()
        or not outputs.accepted_draft_counts.is_contiguous()
        or not outputs.committed_token_counts.is_contiguous()
        or not outputs.rejected_rows.is_contiguous()
        or not outputs.bonus_rows.is_contiguous()
        or not outputs.invalid_rows.is_contiguous()
    ):
        raise ValueError("persistent acceptance outputs do not match the round")
    else:
        pass
    compact_speculative_acceptance_kernel[(batch_size,)](
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
        outputs.committed_tokens,
        outputs.accepted_draft_counts,
        outputs.committed_token_counts,
        outputs.rejected_rows,
        outputs.bonus_rows,
        outputs.invalid_rows,
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
    return outputs


def update_speculative_dense_probability_table(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    dense_probabilities: torch.Tensor,
    previous_token_ids: torch.Tensor,
    current_token_ids: torch.Tensor,
) -> None:
    """Replace each row's sparse support in its persistent dense table."""
    row_count, candidate_count = probabilities.shape
    if token_ids.shape != (row_count, candidate_count):
        raise ValueError("compact token IDs must match probability rows")
    elif dense_probabilities.ndim != 2 or dense_probabilities.shape[0] < row_count:
        raise ValueError("dense probability storage has insufficient rows")
    elif previous_token_ids.shape[0] < row_count:
        raise ValueError("previous support storage has insufficient rows")
    elif current_token_ids.shape[0] < row_count:
        raise ValueError("current support storage has insufficient rows")
    elif (
        previous_token_ids.shape[1] < candidate_count
        or current_token_ids.shape[1] < candidate_count
    ):
        raise ValueError("dense support storage has insufficient candidate width")
    elif dense_probabilities.dtype is not torch.float32:
        raise ValueError("dense probability tables must use float32")
    elif token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compact token IDs must use an integer dtype")
    elif (
        previous_token_ids.dtype is not torch.long
        or current_token_ids.dtype is not torch.long
    ):
        raise ValueError("dense support IDs must use int64")
    elif any(
        tensor.device != probabilities.device
        for tensor in (
            token_ids,
            dense_probabilities,
            previous_token_ids,
            current_token_ids,
        )
    ):
        raise ValueError("dense probability inputs must share one device")
    elif probabilities.dtype is not torch.float32:
        raise ValueError("compact probabilities must use float32")
    else:
        pass
    if row_count == 0:
        return
    else:
        pass

    active_previous_ids = previous_token_ids[:row_count]
    active_current_ids = current_token_ids[:row_count, :candidate_count]
    active_dense_probabilities = dense_probabilities[:row_count]
    active_current_ids.copy_(token_ids)
    active_current_ids.clamp_(min=0, max=dense_probabilities.shape[1] - 1)
    active_dense_probabilities.scatter_(1, active_previous_ids, 0.0)
    active_dense_probabilities.scatter_add_(1, active_current_ids, probabilities)
    active_previous_ids.zero_()
    active_previous_ids[:, :candidate_count].copy_(active_current_ids)


def accept_speculative_candidates_dense_indexed(
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
    dense_draft_probabilities: torch.Tensor,
    dense_target_probabilities: torch.Tensor,
    candidate_offsets: torch.Tensor,
    outputs: SpeculativeAcceptanceBatch | None = None,
) -> SpeculativeAcceptanceBatch:
    """Use token-indexed tables for acceptance and compact-order residual draws."""
    if draft_tokens.ndim != 2 or draft_probabilities.ndim != 2:
        raise ValueError("dense-indexed speculative draft tensors have invalid ranks")
    else:
        pass
    batch_size, draft_width = draft_tokens.shape
    verify_width = draft_width + 1
    candidate_count = draft_probabilities.shape[1]
    if candidate_count <= 0:
        raise ValueError("dense-indexed supports must contain candidates")
    elif draft_probabilities.shape[0] != batch_size * draft_width:
        raise ValueError("dense-indexed draft rows do not match draft tokens")
    elif draft_token_ids.shape != draft_probabilities.shape:
        raise ValueError("dense-indexed draft IDs do not match probabilities")
    elif target_probabilities.shape != (batch_size * verify_width, candidate_count):
        raise ValueError("dense-indexed target rows must include a bonus row")
    elif target_token_ids.shape != target_probabilities.shape:
        raise ValueError("dense-indexed target IDs do not match probabilities")
    elif dense_draft_probabilities.shape[:2] != (batch_size, draft_width):
        raise ValueError("dense draft tables do not match the proposal window")
    elif dense_target_probabilities.shape[:2] != (batch_size, verify_width):
        raise ValueError("dense target tables do not match the verification window")
    elif dense_draft_probabilities.shape[2] != dense_target_probabilities.shape[2]:
        raise ValueError("dense draft and target tables must share a vocabulary")
    elif candidate_offsets.shape != (candidate_count,):
        raise ValueError("candidate offsets do not match compact support width")
    elif proposal_counts.shape != (batch_size,) or active_rows.shape != (batch_size,):
        raise ValueError("dense-indexed metadata must contain one value per row")
    elif acceptance_uniforms.shape != (batch_size, draft_width):
        raise ValueError("acceptance uniforms must match the proposal window")
    elif correction_uniforms.shape != (batch_size, verify_width):
        raise ValueError("correction uniforms must match the verification window")
    elif bonus_uniforms.shape != (batch_size, verify_width):
        raise ValueError("bonus uniforms must match the verification window")
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
        dense_draft_probabilities,
        dense_target_probabilities,
        candidate_offsets,
    )
    if any(tensor.device != draft_tokens.device for tensor in round_tensors):
        raise ValueError("dense-indexed speculative tensors must share one device")
    elif draft_probabilities.dtype is not torch.float32:
        raise ValueError("dense-indexed compact probabilities must use float32")
    elif target_probabilities.dtype is not torch.float32:
        raise ValueError("dense-indexed compact probabilities must use float32")
    elif (
        dense_draft_probabilities.dtype is not torch.float32
        or dense_target_probabilities.dtype is not torch.float32
    ):
        raise ValueError("dense-indexed tables must use float32")
    elif candidate_offsets.dtype is not torch.long:
        raise ValueError("candidate offsets must use int64")
    elif active_rows.dtype is not torch.bool:
        raise ValueError("active-row metadata must use booleans")
    elif proposal_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("proposal counts must use integers")
    elif any(
        tensor.dtype is not torch.float32
        for tensor in (acceptance_uniforms, correction_uniforms, bonus_uniforms)
    ):
        raise ValueError("dense-indexed uniforms must use float32")
    else:
        pass

    if outputs is None:
        outputs = SpeculativeAcceptanceBatch(
            committed_tokens=torch.empty(
                (batch_size, verify_width), device=draft_tokens.device, dtype=torch.long
            ),
            accepted_draft_counts=torch.empty_like(proposal_counts),
            committed_token_counts=torch.empty_like(proposal_counts),
            rejected_rows=torch.empty(
                (batch_size,), device=draft_tokens.device, dtype=torch.bool
            ),
            bonus_rows=torch.empty(
                (batch_size,), device=draft_tokens.device, dtype=torch.bool
            ),
            invalid_rows=torch.empty(
                (batch_size,), device=draft_tokens.device, dtype=torch.bool
            ),
        )
    elif (
        outputs.committed_tokens.shape != (batch_size, verify_width)
        or outputs.accepted_draft_counts.shape != (batch_size,)
        or outputs.committed_token_counts.shape != (batch_size,)
        or outputs.rejected_rows.shape != (batch_size,)
        or outputs.bonus_rows.shape != (batch_size,)
        or outputs.invalid_rows.shape != (batch_size,)
        or outputs.committed_tokens.device != draft_tokens.device
        or outputs.accepted_draft_counts.device != draft_tokens.device
        or outputs.committed_token_counts.device != draft_tokens.device
        or outputs.rejected_rows.device != draft_tokens.device
        or outputs.bonus_rows.device != draft_tokens.device
        or outputs.invalid_rows.device != draft_tokens.device
    ):
        raise ValueError("persistent dense-indexed outputs do not match the round")
    else:
        pass

    dense_draft_rows = dense_draft_probabilities.view(
        batch_size, draft_width, dense_draft_probabilities.shape[-1]
    )
    dense_target_rows = dense_target_probabilities.view(
        batch_size, verify_width, dense_target_probabilities.shape[-1]
    )
    compact_target_probabilities = target_probabilities.view(
        batch_size, verify_width, candidate_count
    )
    compact_target_ids = (
        target_token_ids.view(batch_size, verify_width, candidate_count)
        .to(torch.long)
        .clamp_(min=0, max=dense_target_probabilities.shape[-1] - 1)
    )
    active_draft_tokens = draft_tokens.to(torch.long)
    q_values = torch.empty(
        (batch_size, draft_width), device=draft_tokens.device, dtype=torch.float64
    )
    p_values = torch.empty_like(q_values)
    for draft_index in range(draft_width):
        token_index = active_draft_tokens[:, draft_index : draft_index + 1]
        q_values[:, draft_index] = (
            dense_draft_rows[:, draft_index, :]
            .gather(1, token_index)[:, 0]
            .to(torch.float64)
        )
        p_values[:, draft_index] = (
            dense_target_rows[:, draft_index, :]
            .gather(1, token_index)[:, 0]
            .to(torch.float64)
        )
    else:
        pass

    safe_q_values = torch.where(q_values > 0.0, q_values, 1.0)
    acceptance_ratios = torch.where(
        q_values > 0.0,
        p_values / safe_q_values,
        0.0,
    ).clamp_(0.0, 1.0)
    proposal_positions = candidate_offsets[:draft_width]
    valid_drafts = proposal_positions[None, :] < proposal_counts[:, None]
    valid_drafts &= active_rows[:, None]
    rejected_drafts = valid_drafts & (
        acceptance_uniforms.to(torch.float64) >= acceptance_ratios
    )
    prior_rejection = (
        rejected_drafts.to(torch.int32).cumsum(dim=1) - rejected_drafts.to(torch.int32)
        > 0
    )
    accepted_drafts = valid_drafts & ~rejected_drafts & ~prior_rejection
    accepted_counts = accepted_drafts.sum(dim=1).to(proposal_counts.dtype)
    rejected_rows = rejected_drafts.any(dim=1) & active_rows
    selected_target_rows = accepted_counts.clamp(max=draft_width).to(torch.long)
    selected_target_indices = selected_target_rows[:, None, None].expand(
        batch_size, 1, candidate_count
    )
    selected_target_probabilities = compact_target_probabilities.gather(
        1, selected_target_indices
    )[:, 0, :].to(torch.float64)
    selected_target_ids = compact_target_ids.gather(1, selected_target_indices)[:, 0, :]
    selected_draft_probabilities = torch.zeros_like(selected_target_probabilities)
    if draft_width > 0:
        for draft_index in range(draft_width):
            selected_ids = selected_target_ids
            proposal_row = dense_draft_rows[:, draft_index, :]
            selected_q = proposal_row.gather(1, selected_ids)
            selected_draft_probabilities = torch.where(
                (rejected_rows & (accepted_counts == draft_index))[:, None],
                selected_q.to(torch.float64),
                selected_draft_probabilities,
            )
        else:
            pass
    else:
        pass
    residual_values = torch.where(
        rejected_rows[:, None],
        (selected_target_probabilities - selected_draft_probabilities).clamp_min(0.0),
        selected_target_probabilities,
    )
    residual_scales = residual_values.max(dim=1).values
    invalid_rows = rejected_rows & (residual_scales <= 0.0)
    safe_residual_scales = torch.where(residual_scales > 0.0, residual_scales, 1.0)
    scaled_residuals = residual_values / safe_residual_scales[:, None]
    residual_masses = scaled_residuals.sum(dim=1)
    safe_residual_masses = torch.where(residual_masses > 0.0, residual_masses, 1.0)
    normalized_residuals = scaled_residuals / safe_residual_masses[:, None]
    selected_uniforms = torch.where(
        rejected_rows,
        correction_uniforms.gather(1, selected_target_rows[:, None])[:, 0],
        bonus_uniforms.gather(1, selected_target_rows[:, None])[:, 0],
    ).to(torch.float64)
    cumulative_residuals = normalized_residuals.cumsum(dim=1)
    selected_ranks = (cumulative_residuals <= selected_uniforms[:, None]).sum(dim=1)
    last_positive_ranks = (
        torch.where(
            normalized_residuals > 0.0,
            candidate_offsets[None, :],
            0,
        )
        .max(dim=1)
        .values
    )
    selected_ranks = torch.minimum(selected_ranks, last_positive_ranks)
    final_tokens = selected_target_ids.gather(1, selected_ranks[:, None])[:, 0]
    final_tokens = torch.where(active_rows & ~invalid_rows, final_tokens, 0)
    accepted_counts = torch.where(invalid_rows, 0, accepted_counts)
    outputs.committed_tokens.zero_()
    for draft_index in range(draft_width):
        outputs.committed_tokens[:, draft_index].copy_(
            torch.where(
                accepted_counts > draft_index,
                draft_tokens[:, draft_index].to(torch.long),
                0,
            )
        )
    else:
        pass
    outputs.committed_tokens.scatter_(
        1, accepted_counts[:, None].to(torch.long), final_tokens[:, None]
    )
    outputs.accepted_draft_counts.copy_(accepted_counts)
    outputs.committed_token_counts.copy_(
        torch.where(active_rows & ~invalid_rows, accepted_counts + 1, 0)
    )
    outputs.rejected_rows.copy_(rejected_rows)
    outputs.bonus_rows.copy_(active_rows & ~rejected_rows & ~invalid_rows)
    outputs.invalid_rows.copy_(invalid_rows)
    return outputs


def sample_compact_probability_rows_fused(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    uniforms: torch.Tensor,
    active_rows: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample compact proposal rows with a single inverse-CDF launch."""
    if probabilities.ndim != 2 or token_ids.shape != probabilities.shape:
        raise ValueError(
            "compact probabilities and token IDs must have equal 2D shapes"
        )
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
    if output is None:
        sampled_tokens = torch.empty(
            (batch_size, proposal_width),
            device=probabilities.device,
            dtype=torch.long,
        )
    elif (
        output.shape != (batch_size, proposal_width)
        or output.device != probabilities.device
        or output.dtype is not torch.long
        or not output.is_contiguous()
    ):
        raise ValueError("persistent proposal output does not match the rows")
    else:
        sampled_tokens = output
    if row_count == 0:
        return sampled_tokens
    else:
        pass
    flat_sampled_tokens = sampled_tokens.view(row_count)
    sample_compact_probability_rows_kernel[(row_count,)](
        probabilities,
        token_ids,
        uniforms,
        active_rows.view(-1),
        flat_sampled_tokens,
        PROPOSAL_WIDTH=proposal_width,
        CANDIDATE_COUNT=candidate_count,
        UNIFORM_STRIDE_ROW=uniforms.stride(0),
        UNIFORM_STRIDE_COLUMN=uniforms.stride(1),
        BLOCK_CANDIDATES=triton.next_power_of_2(candidate_count),
        num_warps=4,
    )
    return sampled_tokens

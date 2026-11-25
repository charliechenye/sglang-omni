# SPDX-License-Identifier: Apache-2.0
"""Packed verification input preparation for Qwen3-TTS speculative rounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


@dataclass(kw_only=True)
class SpeculativeVerifierInputs:
    token_embeddings: torch.Tensor
    positions: torch.Tensor
    target_head_indices: torch.Tensor
    valid_queries: torch.Tensor


if triton is not None:

    @triton.jit
    def prepare_speculative_round_metadata_kernel(
        committed_depths,
        active_rows,
        temperatures,
        top_ks,
        top_ps,
        proposal_head_offsets,
        proposal_head_indices,
        proposal_active_rows,
        proposal_counts,
        valid_query_counts,
        proposal_temperatures,
        proposal_top_ks,
        proposal_top_ps,
        target_temperatures,
        target_top_ks,
        target_top_ps,
        DRAFT_WIDTH: tl.constexpr,
        VERIFY_WIDTH: tl.constexpr,
        NUM_CODE_GROUPS: tl.constexpr,
        DRAFT_BLOCK: tl.constexpr,
        VERIFY_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        committed_depth = tl.load(committed_depths + row).to(tl.int32)
        row_is_active = tl.load(active_rows + row).to(tl.int1)
        remaining_count = tl.maximum(NUM_CODE_GROUPS - 2 - committed_depth, 0)
        proposal_count = tl.minimum(remaining_count, DRAFT_WIDTH)
        proposal_count = tl.where(row_is_active, proposal_count, 0)
        tl.store(proposal_counts + row, proposal_count)
        tl.store(
            valid_query_counts + row,
            tl.where(row_is_active, proposal_count + 1, 0),
        )
        if DRAFT_WIDTH > 0:
            proposal_offsets = tl.arange(0, DRAFT_BLOCK)
            proposal_mask = proposal_offsets < DRAFT_WIDTH
            proposal_offset = tl.load(
                proposal_head_offsets + row * DRAFT_WIDTH + proposal_offsets,
                mask=proposal_mask,
                other=0,
            ).to(tl.int32)
            proposal_row_is_active = row_is_active & (proposal_offsets < proposal_count)
            proposal_flat_row = row * DRAFT_WIDTH + proposal_offsets
            tl.store(
                proposal_head_indices + proposal_flat_row,
                committed_depth + proposal_offset,
                mask=proposal_mask,
            )
            tl.store(
                proposal_active_rows + proposal_flat_row,
                proposal_row_is_active,
                mask=proposal_mask,
            )
            temperature = tl.load(temperatures + row)
            top_k = tl.load(top_ks + row)
            top_p = tl.load(top_ps + row)
            tl.store(
                proposal_temperatures + proposal_flat_row,
                temperature,
                mask=proposal_mask,
            )
            tl.store(
                proposal_top_ks + proposal_flat_row,
                top_k,
                mask=proposal_mask,
            )
            tl.store(
                proposal_top_ps + proposal_flat_row,
                top_p,
                mask=proposal_mask,
            )
        else:
            pass
        query_offsets = tl.arange(0, VERIFY_BLOCK)
        query_mask = query_offsets < VERIFY_WIDTH
        target_flat_row = row * VERIFY_WIDTH + query_offsets
        temperature = tl.load(temperatures + row)
        top_k = tl.load(top_ks + row)
        top_p = tl.load(top_ps + row)
        tl.store(target_temperatures + target_flat_row, temperature, mask=query_mask)
        tl.store(target_top_ks + target_flat_row, top_k, mask=query_mask)
        tl.store(target_top_ps + target_flat_row, top_p, mask=query_mask)

    @triton.jit
    def prepare_speculative_verification_inputs_kernel(
        pending_tokens,
        draft_tokens,
        committed_depths,
        logical_kv_lengths,
        active_rows,
        proposal_counts,
        projected_tables,
        token_embeddings,
        positions,
        target_head_indices,
        valid_queries,
        DRAFT_WIDTH: tl.constexpr,
        VERIFY_WIDTH: tl.constexpr,
        TABLE_COUNT: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        INPUT_STRIDE_ROW: tl.constexpr,
        INPUT_STRIDE_QUERY: tl.constexpr,
        INPUT_STRIDE_HIDDEN: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        row = tl.program_id(0)
        query_offset = tl.program_id(1)
        hidden_offsets = tl.program_id(2) * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        committed_depth = tl.load(committed_depths + row).to(tl.int32)
        proposal_count = tl.load(proposal_counts + row).to(tl.int32)
        is_active = tl.load(active_rows + row).to(tl.int1)
        is_valid = is_active & (query_offset < proposal_count + 1)
        query_depth = committed_depth + query_offset
        if query_offset == 0:
            token_id = tl.load(pending_tokens + row).to(tl.int32)
        else:
            token_id = tl.load(
                draft_tokens + row * DRAFT_WIDTH + query_offset - 1,
                mask=is_valid,
                other=0,
            ).to(tl.int32)
        safe_table_depth = tl.minimum(tl.maximum(query_depth - 1, 0), TABLE_COUNT - 1)
        safe_token_id = tl.minimum(tl.maximum(token_id, 0), VOCAB_SIZE - 1)
        table_offsets = (
            safe_table_depth * VOCAB_SIZE * HIDDEN_SIZE
            + safe_token_id * HIDDEN_SIZE
            + hidden_offsets
        )
        embedding_values = tl.load(
            projected_tables + table_offsets,
            mask=is_valid & hidden_mask,
            other=0.0,
        )
        tl.store(
            token_embeddings
            + row * INPUT_STRIDE_ROW
            + query_offset * INPUT_STRIDE_QUERY
            + hidden_offsets * INPUT_STRIDE_HIDDEN,
            embedding_values,
            mask=hidden_mask,
        )
        if tl.program_id(2) == 0:
            flat_query = row * VERIFY_WIDTH + query_offset
            logical_kv_length = tl.load(logical_kv_lengths + row).to(tl.int32)
            tl.store(
                positions + flat_query,
                tl.where(is_valid, logical_kv_length + query_offset, -1),
            )
            tl.store(
                target_head_indices + flat_query,
                tl.where(is_valid, query_depth, 0),
            )
            tl.store(valid_queries + flat_query, is_valid)
        else:
            pass

else:
    prepare_speculative_round_metadata_kernel = None
    prepare_speculative_verification_inputs_kernel = None


def prepare_speculative_round_metadata(
    committed_depths: torch.Tensor,
    active_rows: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    proposal_head_offsets: torch.Tensor,
    proposal_head_indices: torch.Tensor,
    proposal_active_rows: torch.Tensor,
    proposal_counts: torch.Tensor,
    valid_query_counts: torch.Tensor,
    proposal_temperatures: torch.Tensor,
    proposal_top_ks: torch.Tensor,
    proposal_top_ps: torch.Tensor,
    target_temperatures: torch.Tensor,
    target_top_ks: torch.Tensor,
    target_top_ps: torch.Tensor,
    verify_width: int,
    num_code_groups: int,
) -> bool:
    """Prepare dynamic selectors and per-query sampling parameters on device."""
    batch_size = int(committed_depths.shape[0])
    draft_width = verify_width - 1
    if committed_depths.ndim != 1 or active_rows.shape != (batch_size,):
        raise ValueError("speculative round state must contain aligned batch rows")
    elif any(
        tensor.shape != (batch_size,)
        for tensor in (
            temperatures,
            top_ks,
            top_ps,
            proposal_counts,
            valid_query_counts,
        )
    ):
        raise ValueError("speculative round parameters must contain aligned rows")
    elif proposal_head_offsets.shape != (batch_size * draft_width,):
        raise ValueError("static proposal offsets do not match the verification width")
    elif proposal_head_indices.shape != (batch_size * draft_width,):
        raise ValueError("proposal head output does not match the verification width")
    elif proposal_active_rows.shape != (batch_size * draft_width,):
        raise ValueError(
            "proposal activity output does not match the verification width"
        )
    elif any(
        tensor.shape != (batch_size * draft_width,)
        for tensor in (proposal_temperatures, proposal_top_ks, proposal_top_ps)
    ):
        raise ValueError(
            "proposal sampling outputs do not match the verification width"
        )
    elif any(
        tensor.shape != (batch_size * verify_width,)
        for tensor in (target_temperatures, target_top_ks, target_top_ps)
    ):
        raise ValueError("target sampling outputs do not match the verification width")
    else:
        pass
    tensors = (
        committed_depths,
        active_rows,
        temperatures,
        top_ks,
        top_ps,
        proposal_head_offsets,
        proposal_head_indices,
        proposal_active_rows,
        proposal_counts,
        valid_query_counts,
        proposal_temperatures,
        proposal_top_ks,
        proposal_top_ps,
        target_temperatures,
        target_top_ks,
        target_top_ps,
    )
    if any(tensor.device != committed_depths.device for tensor in tensors):
        raise ValueError("speculative round metadata must share one device")
    elif (
        not committed_depths.is_cuda
        or prepare_speculative_round_metadata_kernel is None
    ):
        return False
    elif any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("speculative round metadata must be contiguous")
    elif temperatures.dtype is not torch.float32 or top_ps.dtype is not torch.float32:
        raise ValueError("speculative temperatures and top-p values must use FP32")
    elif top_ks.dtype is not torch.long or committed_depths.dtype is not torch.long:
        raise ValueError("speculative selectors must use int64")
    elif (
        active_rows.dtype is not torch.bool
        or proposal_active_rows.dtype is not torch.bool
    ):
        raise ValueError("speculative activity selectors must use booleans")
    else:
        pass
    prepare_speculative_round_metadata_kernel[(batch_size,)](
        committed_depths,
        active_rows,
        temperatures,
        top_ks,
        top_ps,
        proposal_head_offsets,
        proposal_head_indices,
        proposal_active_rows,
        proposal_counts,
        valid_query_counts,
        proposal_temperatures,
        proposal_top_ks,
        proposal_top_ps,
        target_temperatures,
        target_top_ks,
        target_top_ps,
        DRAFT_WIDTH=draft_width,
        VERIFY_WIDTH=verify_width,
        NUM_CODE_GROUPS=num_code_groups,
        DRAFT_BLOCK=triton.next_power_of_2(max(draft_width, 1)),
        VERIFY_BLOCK=triton.next_power_of_2(verify_width),
        num_warps=4,
    )
    return True


def prepare_speculative_verification_inputs(
    pending_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    committed_depths: torch.Tensor,
    logical_kv_lengths: torch.Tensor,
    active_rows: torch.Tensor,
    proposal_counts: torch.Tensor,
    projected_tables: torch.Tensor,
    outputs: SpeculativeVerifierInputs,
) -> bool:
    """Fill persistent packed embeddings and verifier selectors in one launch."""
    if pending_tokens.ndim != 1 or draft_tokens.ndim != 2:
        raise ValueError("pending and draft token tensors have invalid ranks")
    else:
        pass
    batch_size, draft_width = draft_tokens.shape
    verify_width = draft_width + 1
    if pending_tokens.shape != (batch_size,):
        raise ValueError("pending tokens must align with draft batch rows")
    elif any(
        tensor.shape != (batch_size,)
        for tensor in (
            committed_depths,
            logical_kv_lengths,
            active_rows,
            proposal_counts,
        )
    ):
        raise ValueError("verifier preparation metadata must contain aligned rows")
    elif projected_tables.ndim != 3:
        raise ValueError("projected codec tables must have three dimensions")
    elif outputs.token_embeddings.shape[0] < batch_size:
        raise ValueError("persistent verifier inputs do not cover the batch")
    elif outputs.token_embeddings.shape[1:] != (
        verify_width,
        projected_tables.shape[2],
    ):
        raise ValueError("persistent verifier embedding shape does not match the round")
    elif outputs.positions.numel() < batch_size * verify_width:
        raise ValueError("persistent verifier positions do not cover the batch")
    elif outputs.target_head_indices.shape[0] < batch_size:
        raise ValueError("persistent target selectors do not cover the batch")
    elif outputs.valid_queries.shape[0] < batch_size:
        raise ValueError("persistent query masks do not cover the batch")
    elif outputs.target_head_indices.shape[1] != verify_width:
        raise ValueError("persistent target selectors do not match the round width")
    elif outputs.valid_queries.shape[1] != verify_width:
        raise ValueError("persistent query masks do not match the round width")
    else:
        pass
    input_tensors = (
        pending_tokens,
        draft_tokens,
        committed_depths,
        logical_kv_lengths,
        active_rows,
        proposal_counts,
        projected_tables,
        outputs.token_embeddings,
        outputs.positions,
        outputs.target_head_indices,
        outputs.valid_queries,
    )
    if any(tensor.device != pending_tokens.device for tensor in input_tensors):
        raise ValueError("verifier preparation tensors must share one device")
    elif (
        not pending_tokens.is_cuda
        or prepare_speculative_verification_inputs_kernel is None
    ):
        return False
    elif projected_tables.dtype is not torch.bfloat16:
        raise ValueError("projected codec tables must use BF16")
    elif outputs.token_embeddings.dtype is not projected_tables.dtype:
        raise ValueError("verifier embedding output dtype must match its table")
    elif any(not tensor.is_contiguous() for tensor in input_tensors):
        raise ValueError("verifier preparation tensors must be contiguous")
    else:
        pass
    hidden_size = int(projected_tables.shape[2])
    hidden_block = 256
    prepare_speculative_verification_inputs_kernel[
        (batch_size, verify_width, triton.cdiv(hidden_size, hidden_block))
    ](
        pending_tokens,
        draft_tokens,
        committed_depths,
        logical_kv_lengths,
        active_rows,
        proposal_counts,
        projected_tables,
        outputs.token_embeddings,
        outputs.positions,
        outputs.target_head_indices,
        outputs.valid_queries,
        DRAFT_WIDTH=draft_width,
        VERIFY_WIDTH=verify_width,
        TABLE_COUNT=projected_tables.shape[0],
        VOCAB_SIZE=projected_tables.shape[1],
        HIDDEN_SIZE=hidden_size,
        INPUT_STRIDE_ROW=outputs.token_embeddings.stride(0),
        INPUT_STRIDE_QUERY=outputs.token_embeddings.stride(1),
        INPUT_STRIDE_HIDDEN=outputs.token_embeddings.stride(2),
        BLOCK_HIDDEN=hidden_block,
        num_warps=4,
    )
    return True

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
    prepare_speculative_verification_inputs_kernel = None


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

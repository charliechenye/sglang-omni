# SPDX-License-Identifier: Apache-2.0
"""Batched codec commitment for Qwen3-TTS speculative rounds."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def commit_speculative_codes_kernel(
        committed_tokens,
        committed_token_counts,
        committed_depths,
        codec_tables,
        output_codes,
        summed_embeddings,
        CODE_TABLES: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        VERIFY_WIDTH: tl.constexpr,
        OUTPUT_GROUPS: tl.constexpr,
        CODE_STRIDE_ROW: tl.constexpr,
        CODE_STRIDE_DEPTH: tl.constexpr,
        SUM_STRIDE_ROW: tl.constexpr,
        SUM_STRIDE_HIDDEN: tl.constexpr,
        HIDDEN_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        hidden_offsets = tl.program_id(1) * HIDDEN_BLOCK + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        commit_count = tl.load(committed_token_counts + row).to(tl.int32)
        starting_depth = tl.load(committed_depths + row).to(tl.int32)
        accumulator = tl.load(
            summed_embeddings
            + row * SUM_STRIDE_ROW
            + hidden_offsets * SUM_STRIDE_HIDDEN,
            mask=hidden_mask,
            other=0,
        )
        for commit_offset in tl.static_range(0, VERIFY_WIDTH):
            output_depth = starting_depth + 1 + commit_offset
            table_depth = output_depth - 1
            is_valid = (
                (commit_offset < commit_count)
                & (output_depth >= 0)
                & (output_depth < OUTPUT_GROUPS)
                & (table_depth >= 0)
                & (table_depth < CODE_TABLES)
            )
            token_id = tl.load(
                committed_tokens + row * VERIFY_WIDTH + commit_offset,
                mask=is_valid,
                other=0,
            ).to(tl.int32)
            token_is_valid = (token_id >= 0) & (token_id < VOCAB_SIZE)
            table_offset = (
                table_depth * VOCAB_SIZE * HIDDEN_SIZE
                + token_id * HIDDEN_SIZE
                + hidden_offsets
            )
            embedding = tl.load(
                codec_tables + table_offset,
                mask=is_valid
                & token_is_valid
                & hidden_mask
                & (table_depth >= 0)
                & (table_depth < CODE_TABLES),
                other=0,
            ).to(tl.float32)
            accumulator = (accumulator.to(tl.float32) + embedding).to(tl.bfloat16)
            tl.store(
                output_codes + row * CODE_STRIDE_ROW + output_depth * CODE_STRIDE_DEPTH,
                token_id,
                mask=is_valid & token_is_valid & (tl.program_id(1) == 0),
            )
        tl.store(
            summed_embeddings
            + row * SUM_STRIDE_ROW
            + hidden_offsets * SUM_STRIDE_HIDDEN,
            accumulator,
            mask=hidden_mask,
        )

else:
    commit_speculative_codes_kernel = None


if triton is not None:

    @triton.jit
    def commit_speculative_state_kernel(
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
        VERIFY_WIDTH: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        NUM_CODE_GROUPS: tl.constexpr,
        VERIFIED_STRIDE_ROW: tl.constexpr,
        VERIFIED_STRIDE_QUERY: tl.constexpr,
        VERIFIED_STRIDE_HIDDEN: tl.constexpr,
        PRECEDING_STRIDE_ROW: tl.constexpr,
        PRECEDING_STRIDE_HIDDEN: tl.constexpr,
        METADATA_STRIDE_ROW: tl.constexpr,
        HIDDEN_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        hidden_offsets = tl.program_id(1) * HIDDEN_BLOCK + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        accepted_count = tl.load(accepted_draft_counts + row).to(tl.int32)
        committed_count = tl.load(committed_token_counts + row).to(tl.int32)
        is_invalid = tl.load(invalid_rows + row).to(tl.int1)
        has_commit = (committed_count > 0) & (~is_invalid)
        hidden_query = tl.minimum(accepted_count, VERIFY_WIDTH - 1)
        carried_hidden = tl.load(
            verified_hiddens
            + row * VERIFIED_STRIDE_ROW
            + hidden_query * VERIFIED_STRIDE_QUERY
            + hidden_offsets * VERIFIED_STRIDE_HIDDEN,
            mask=hidden_mask & has_commit,
            other=0,
        )
        tl.store(
            preceding_hidden
            + row * PRECEDING_STRIDE_ROW
            + hidden_offsets * PRECEDING_STRIDE_HIDDEN,
            carried_hidden,
            mask=hidden_mask & has_commit,
        )
        if tl.program_id(1) == 0:
            starting_depth = tl.load(committed_depths + row).to(tl.int32)
            starting_kv_length = tl.load(logical_kv_lengths + row).to(tl.int32)
            is_active = tl.load(active_rows + row).to(tl.int1)
            next_depth = starting_depth + committed_count
            next_kv_length = starting_kv_length + committed_count
            next_active = tl.where(
                is_invalid, is_active, next_depth < NUM_CODE_GROUPS - 1
            )
            final_token = tl.load(
                committed_tokens
                + row * VERIFY_WIDTH
                + tl.maximum(committed_count - 1, 0),
                mask=has_commit,
                other=0,
            )
            tl.store(pending_tokens + row, final_token, mask=has_commit)
            tl.store(committed_depths + row, next_depth)
            tl.store(logical_kv_lengths + row, next_kv_length)
            tl.store(active_rows + row, next_active)
            row_metadata = metadata + row * METADATA_STRIDE_ROW
            tl.store(row_metadata + 0, accepted_count.to(tl.int64))
            tl.store(row_metadata + 1, next_active.to(tl.int64))
            tl.store(row_metadata + 2, tl.load(proposal_counts + row).to(tl.int64))
            tl.store(row_metadata + 3, committed_count.to(tl.int64))
            tl.store(row_metadata + 4, tl.load(bonus_rows + row).to(tl.int64))
            tl.store(row_metadata + 5, tl.load(rejected_rows + row).to(tl.int64))
            tl.store(row_metadata + 6, tl.load(valid_query_counts + row).to(tl.int64))
            tl.store(row_metadata + 7, next_depth.to(tl.int64))
            tl.store(row_metadata + 8, is_invalid.to(tl.int64))
        else:
            pass

else:
    commit_speculative_state_kernel = None


def commit_speculative_codes(
    committed_tokens: torch.Tensor,
    committed_token_counts: torch.Tensor,
    committed_depths: torch.Tensor,
    codec_tables: torch.Tensor,
    output_codes: torch.Tensor,
    summed_embeddings: torch.Tensor,
) -> None:
    """Write valid speculative codes and accumulate BF16 embeddings in order."""
    if committed_tokens.ndim != 2 or codec_tables.ndim != 3:
        raise ValueError("speculative commitment tensors have invalid ranks")
    else:
        pass
    batch_size, verify_width = committed_tokens.shape
    if output_codes.ndim != 2 or output_codes.shape[0] != batch_size:
        raise ValueError("output codes must have shape [batch, code groups]")
    elif batch_size <= 0 or verify_width <= 0:
        raise ValueError("speculative commitment requires nonempty rows and width")
    elif codec_tables.shape[0] != output_codes.shape[1] - 1:
        raise ValueError("codec table depth must match the output code groups")
    elif codec_tables.shape[1] <= 0 or codec_tables.shape[2] <= 0:
        raise ValueError("codec tables must have positive vocabulary and hidden sizes")
    elif committed_token_counts.shape != (batch_size,):
        raise ValueError("commit counts must contain one value per batch row")
    elif committed_depths.shape != (batch_size,):
        raise ValueError("committed depths must contain one value per batch row")
    elif summed_embeddings.ndim != 2 or summed_embeddings.shape[0] != batch_size:
        raise ValueError("summed embeddings must have shape [batch, hidden]")
    elif codec_tables.shape[2] != summed_embeddings.shape[1]:
        raise ValueError("codec table hidden size does not match output embeddings")
    elif committed_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("committed tokens must use an integer dtype")
    elif committed_token_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("commit counts must use an integer dtype")
    elif committed_depths.dtype not in (torch.int32, torch.int64):
        raise ValueError("committed depths must use an integer dtype")
    elif output_codes.dtype not in (torch.int32, torch.int64):
        raise ValueError("output codes must use an integer dtype")
    elif (
        codec_tables.dtype is not torch.bfloat16
        or summed_embeddings.dtype is not torch.bfloat16
    ):
        raise ValueError("codec commitment requires BF16 tables and sums")
    elif not codec_tables.is_contiguous():
        raise ValueError("codec tables must be contiguous")
    elif not committed_tokens.is_contiguous():
        raise ValueError("committed token rows must be contiguous")
    elif (
        not committed_token_counts.is_contiguous()
        or not committed_depths.is_contiguous()
    ):
        raise ValueError("commit metadata must be contiguous")
    elif any(
        tensor.device != committed_tokens.device
        for tensor in (
            committed_token_counts,
            committed_depths,
            codec_tables,
            output_codes,
            summed_embeddings,
        )
    ):
        raise ValueError("speculative commitment tensors must share one device")
    elif not committed_tokens.is_cuda or commit_speculative_codes_kernel is None:
        raise RuntimeError("speculative commitment requires CUDA and Triton")
    else:
        pass
    hidden_size = summed_embeddings.shape[1]
    hidden_block = 256
    commit_speculative_codes_kernel[
        (batch_size, triton.cdiv(hidden_size, hidden_block))
    ](
        committed_tokens,
        committed_token_counts,
        committed_depths,
        codec_tables,
        output_codes,
        summed_embeddings,
        CODE_TABLES=codec_tables.shape[0],
        VOCAB_SIZE=codec_tables.shape[1],
        HIDDEN_SIZE=hidden_size,
        VERIFY_WIDTH=verify_width,
        OUTPUT_GROUPS=output_codes.shape[1],
        CODE_STRIDE_ROW=output_codes.stride(0),
        CODE_STRIDE_DEPTH=output_codes.stride(1),
        SUM_STRIDE_ROW=summed_embeddings.stride(0),
        SUM_STRIDE_HIDDEN=summed_embeddings.stride(1),
        HIDDEN_BLOCK=hidden_block,
        num_warps=4,
    )


def commit_speculative_state(
    committed_tokens: torch.Tensor,
    accepted_draft_counts: torch.Tensor,
    committed_token_counts: torch.Tensor,
    proposal_counts: torch.Tensor,
    bonus_rows: torch.Tensor,
    rejected_rows: torch.Tensor,
    invalid_rows: torch.Tensor,
    valid_query_counts: torch.Tensor,
    verified_hiddens: torch.Tensor,
    pending_tokens: torch.Tensor,
    preceding_hidden: torch.Tensor,
    committed_depths: torch.Tensor,
    logical_kv_lengths: torch.Tensor,
    active_rows: torch.Tensor,
    metadata: torch.Tensor,
    num_code_groups: int,
) -> None:
    """Advance carry-forward state and stage one row of round metadata per request."""
    if committed_tokens.ndim != 2 or verified_hiddens.ndim != 3:
        raise ValueError("speculative state tensors have invalid ranks")
    else:
        pass
    batch_size, verify_width = committed_tokens.shape
    hidden_size = int(verified_hiddens.shape[2])
    if verified_hiddens.shape[:2] != (batch_size, verify_width):
        raise ValueError("verified hidden rows must match committed query positions")
    elif preceding_hidden.shape != (batch_size, hidden_size):
        raise ValueError("preceding hidden state must match verified hidden rows")
    elif pending_tokens.shape != (batch_size,):
        raise ValueError("pending tokens must contain one token per batch row")
    elif any(
        tensor.shape != (batch_size,)
        for tensor in (
            accepted_draft_counts,
            committed_token_counts,
            proposal_counts,
            bonus_rows,
            rejected_rows,
            invalid_rows,
            valid_query_counts,
            committed_depths,
            logical_kv_lengths,
            active_rows,
        )
    ):
        raise ValueError("speculative state metadata must contain aligned rows")
    elif metadata.ndim != 2 or metadata.shape[0] < batch_size or metadata.shape[1] < 9:
        raise ValueError("speculative round metadata must provide nine columns")
    elif not verified_hiddens.is_contiguous() or not metadata.is_contiguous():
        raise ValueError("verified hidden rows and metadata must be contiguous")
    elif any(
        tensor.device != committed_tokens.device
        for tensor in (
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
        )
    ):
        raise ValueError("speculative state tensors must share one device")
    elif not committed_tokens.is_cuda or commit_speculative_state_kernel is None:
        raise RuntimeError("speculative state transition requires CUDA and Triton")
    else:
        pass
    hidden_block = 256
    commit_speculative_state_kernel[
        (batch_size, triton.cdiv(hidden_size, hidden_block))
    ](
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
        VERIFY_WIDTH=verify_width,
        HIDDEN_SIZE=hidden_size,
        NUM_CODE_GROUPS=num_code_groups,
        VERIFIED_STRIDE_ROW=verified_hiddens.stride(0),
        VERIFIED_STRIDE_QUERY=verified_hiddens.stride(1),
        VERIFIED_STRIDE_HIDDEN=verified_hiddens.stride(2),
        PRECEDING_STRIDE_ROW=preceding_hidden.stride(0),
        PRECEDING_STRIDE_HIDDEN=preceding_hidden.stride(1),
        METADATA_STRIDE_ROW=metadata.stride(0),
        HIDDEN_BLOCK=hidden_block,
        num_warps=4,
    )

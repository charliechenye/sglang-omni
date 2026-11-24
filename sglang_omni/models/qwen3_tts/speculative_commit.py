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
            summed_embeddings + row * SUM_STRIDE_ROW + hidden_offsets * SUM_STRIDE_HIDDEN,
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
            summed_embeddings + row * SUM_STRIDE_ROW + hidden_offsets * SUM_STRIDE_HIDDEN,
            accumulator,
            mask=hidden_mask,
        )

else:
    commit_speculative_codes_kernel = None


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
    elif codec_tables.dtype is not torch.bfloat16 or summed_embeddings.dtype is not torch.bfloat16:
        raise ValueError("codec commitment requires BF16 tables and sums")
    elif not codec_tables.is_contiguous():
        raise ValueError("codec tables must be contiguous")
    elif not committed_tokens.is_contiguous():
        raise ValueError("committed token rows must be contiguous")
    elif not committed_token_counts.is_contiguous() or not committed_depths.is_contiguous():
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
    commit_speculative_codes_kernel[(batch_size, triton.cdiv(hidden_size, hidden_block))](
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

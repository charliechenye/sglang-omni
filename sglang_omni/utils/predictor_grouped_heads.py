# SPDX-License-Identifier: Apache-2.0
"""Indexed grouped LM-head projections for packed Predictor rows."""

from __future__ import annotations

from typing import Literal

import torch

GROUPED_HEAD_BLOCK_VOCAB = 64
GROUPED_HEAD_BLOCK_HIDDEN = 32
WARP_GEMV_BLOCK_VOCAB = 8
WARP_ACCUMULATE_OUTPUTS_PER_CTA = 4
WARP_ACCUMULATE_BLOCK_HIDDEN = 64

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None
else:
    pass


if triton is not None:

    @triton.jit
    def grouped_lm_head_kernel(
        hidden_states,
        packed_weights,
        hidden_indices,
        head_indices,
        active_rows,
        logits,
        source_rows: tl.constexpr,
        head_count: tl.constexpr,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        block_vocab: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        row = tl.program_id(0)
        vocab_start = tl.program_id(1) * block_vocab
        vocab_offsets = vocab_start + tl.arange(0, block_vocab)
        hidden_offsets = tl.arange(0, block_hidden)
        is_active = tl.load(active_rows + row)
        hidden_index = tl.load(hidden_indices + row)
        head_index = tl.load(head_indices + row)
        is_valid = (
            is_active
            & (hidden_index >= 0)
            & (hidden_index < source_rows)
            & (head_index >= 0)
            & (head_index < head_count)
        )
        safe_hidden_index = tl.minimum(tl.maximum(hidden_index, 0), source_rows - 1)
        safe_head_index = tl.minimum(tl.maximum(head_index, 0), head_count - 1)
        accumulator = tl.zeros((block_vocab,), dtype=tl.float32)
        for hidden_start in range(0, hidden_size, block_hidden):
            hidden_offsets_at_block = hidden_start + hidden_offsets
            hidden_tile = tl.load(
                hidden_states
                + safe_hidden_index * hidden_size
                + hidden_offsets_at_block,
                mask=is_valid & (hidden_offsets_at_block < hidden_size),
                other=0.0,
            )
            weight_tile = tl.load(
                packed_weights
                + safe_head_index * vocab_size * hidden_size
                + vocab_offsets[:, None] * hidden_size
                + hidden_offsets_at_block[None, :],
                mask=is_valid
                & (vocab_offsets[:, None] < vocab_size)
                & (hidden_offsets_at_block[None, :] < hidden_size),
                other=0.0,
            )
            accumulator += tl.sum(
                weight_tile.to(tl.float32) * hidden_tile.to(tl.float32)[None, :],
                axis=1,
            )
        output = accumulator.to(tl.bfloat16)
        tl.store(
            logits + row * vocab_size + vocab_offsets,
            tl.where(is_valid, output, 0.0),
            mask=vocab_offsets < vocab_size,
        )

    @triton.jit
    def grouped_lm_head_warp_gemv_kernel(
        hidden_states,
        packed_weights,
        hidden_indices,
        head_indices,
        active_rows,
        logits,
        source_rows: tl.constexpr,
        head_count: tl.constexpr,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        block_vocab: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        row = tl.program_id(0)
        vocab_start = tl.program_id(1) * block_vocab
        vocab_offsets = vocab_start + tl.arange(0, block_vocab)
        hidden_offsets = tl.arange(0, block_hidden)
        is_active = tl.load(active_rows + row)
        hidden_index = tl.load(hidden_indices + row)
        head_index = tl.load(head_indices + row)
        is_valid = (
            is_active
            & (hidden_index >= 0)
            & (hidden_index < source_rows)
            & (head_index >= 0)
            & (head_index < head_count)
        )
        safe_hidden_index = tl.minimum(tl.maximum(hidden_index, 0), source_rows - 1)
        safe_head_index = tl.minimum(tl.maximum(head_index, 0), head_count - 1)
        accumulator = tl.zeros((block_vocab,), dtype=tl.float32)
        for hidden_start in range(0, hidden_size, block_hidden):
            hidden_offsets_at_block = hidden_start + hidden_offsets
            hidden_tile = tl.load(
                hidden_states
                + safe_hidden_index * hidden_size
                + hidden_offsets_at_block,
                mask=is_valid & (hidden_offsets_at_block < hidden_size),
                other=0.0,
            )
            weight_tile = tl.load(
                packed_weights
                + safe_head_index * vocab_size * hidden_size
                + vocab_offsets[:, None] * hidden_size
                + hidden_offsets_at_block[None, :],
                mask=is_valid
                & (vocab_offsets[:, None] < vocab_size)
                & (hidden_offsets_at_block[None, :] < hidden_size),
                other=0.0,
            )
            accumulator += tl.sum(
                weight_tile.to(tl.float32) * hidden_tile.to(tl.float32)[None, :],
                axis=1,
            )
        output = accumulator.to(tl.bfloat16)
        tl.store(
            logits + row * vocab_size + vocab_offsets,
            tl.where(is_valid, output, 0.0),
            mask=vocab_offsets < vocab_size,
        )

    @triton.jit
    def grouped_lm_head_warp_accumulate_kernel(
        hidden_states,
        packed_weights,
        hidden_indices,
        head_indices,
        active_rows,
        logits,
        source_rows: tl.constexpr,
        head_count: tl.constexpr,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        outputs_per_cta: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        row = tl.program_id(0)
        output_offsets = tl.program_id(1) * outputs_per_cta + tl.arange(
            0, outputs_per_cta
        )
        hidden_offsets = tl.arange(0, block_hidden)
        is_active = tl.load(active_rows + row)
        hidden_index = tl.load(hidden_indices + row)
        head_index = tl.load(head_indices + row)
        is_valid = (
            is_active
            & (hidden_index >= 0)
            & (hidden_index < source_rows)
            & (head_index >= 0)
            & (head_index < head_count)
        )
        safe_hidden_index = tl.minimum(tl.maximum(hidden_index, 0), source_rows - 1)
        safe_head_index = tl.minimum(tl.maximum(head_index, 0), head_count - 1)
        lane_accumulators = tl.zeros(
            (outputs_per_cta, block_hidden), dtype=tl.float32
        )
        for hidden_start in range(0, hidden_size, block_hidden):
            hidden_offsets_at_block = hidden_start + hidden_offsets
            hidden_tile = tl.load(
                hidden_states
                + safe_hidden_index * hidden_size
                + hidden_offsets_at_block,
                mask=is_valid & (hidden_offsets_at_block < hidden_size),
                other=0.0,
            )
            weight_tile = tl.load(
                packed_weights
                + safe_head_index * vocab_size * hidden_size
                + output_offsets[:, None] * hidden_size
                + hidden_offsets_at_block[None, :],
                mask=is_valid
                & (output_offsets[:, None] < vocab_size)
                & (hidden_offsets_at_block[None, :] < hidden_size),
                other=0.0,
            )
            lane_accumulators += (
                weight_tile.to(tl.float32) * hidden_tile.to(tl.float32)[None, :]
            )
        output = tl.sum(lane_accumulators, axis=1).to(tl.bfloat16)
        tl.store(
            logits + row * vocab_size + output_offsets,
            tl.where(is_valid, output, 0.0),
            mask=output_offsets < vocab_size,
        )

else:
    grouped_lm_head_kernel = None
    grouped_lm_head_warp_gemv_kernel = None
    grouped_lm_head_warp_accumulate_kernel = None


def grouped_lm_head_cuda_available() -> bool:
    """Return whether the row-selected CUDA projection kernel is importable."""
    return triton is not None and grouped_lm_head_kernel is not None


def grouped_lm_head_warp_gemv_available() -> bool:
    """Return whether the eight-output CUDA GEMV candidate is importable."""
    return triton is not None and grouped_lm_head_warp_gemv_kernel is not None


def grouped_lm_head_warp_accumulate_available() -> bool:
    """Return whether the one-output-per-warp BF16 projection is importable."""
    return triton is not None and grouped_lm_head_warp_accumulate_kernel is not None


def grouped_lm_head_linear(
    hidden_states: torch.Tensor,
    packed_weights: torch.Tensor,
    hidden_indices: torch.Tensor,
    head_indices: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    backend: Literal["grouped_triton", "warp_gemv", "warp_accumulate"] = (
        "grouped_triton"
    ),
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project selected hidden rows through selected heads, returning model-dtype logits."""
    if hidden_states.ndim != 2 or packed_weights.ndim != 3:
        raise ValueError("grouped LM-head inputs have invalid rank")
    else:
        pass
    if hidden_indices.ndim != 1 or head_indices.shape != hidden_indices.shape:
        raise ValueError(
            "grouped LM-head selectors must be one-dimensional and aligned"
        )
    else:
        pass
    if active_rows.shape != hidden_indices.shape or active_rows.dtype is not torch.bool:
        raise ValueError("grouped LM-head active mask must align with its selectors")
    else:
        pass
    if hidden_states.shape[1] != packed_weights.shape[2]:
        raise ValueError("grouped LM-head hidden width does not match its weights")
    elif backend not in ("grouped_triton", "warp_gemv", "warp_accumulate"):
        raise ValueError("unsupported grouped LM-head backend")
    else:
        pass
    if any(
        tensor.device != hidden_states.device
        for tensor in (packed_weights, hidden_indices, head_indices, active_rows)
    ):
        raise ValueError("grouped LM-head tensors must share one device")
    else:
        pass

    rows = hidden_indices.shape[0]
    if output is None:
        logits = torch.empty(
            (rows, packed_weights.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif (
        output.shape != (rows, packed_weights.shape[1])
        or output.device != hidden_states.device
        or output.dtype != hidden_states.dtype
        or not output.is_contiguous()
    ):
        raise ValueError("grouped LM-head output does not match its rows")
    else:
        logits = output
    if rows == 0:
        return logits
    else:
        pass
    if hidden_states.is_cuda:
        if backend == "grouped_triton" and not grouped_lm_head_cuda_available():
            raise RuntimeError("grouped CUDA LM-head kernel is unavailable")
        elif backend == "warp_gemv" and not grouped_lm_head_warp_gemv_available():
            raise RuntimeError("warp-oriented CUDA LM-head kernel is unavailable")
        elif backend == "warp_accumulate" and not grouped_lm_head_warp_accumulate_available():
            raise RuntimeError("warp-accumulate CUDA LM-head kernel is unavailable")
        else:
            pass
        if (
            hidden_states.dtype is not torch.bfloat16
            or packed_weights.dtype is not torch.bfloat16
            or not hidden_states.is_contiguous()
            or not packed_weights.is_contiguous()
            or not hidden_indices.is_contiguous()
            or not head_indices.is_contiguous()
            or not active_rows.is_contiguous()
        ):
            raise RuntimeError("grouped CUDA LM-head inputs are unsupported")
        else:
            pass
        if backend == "grouped_triton":
            kernel = grouped_lm_head_kernel
            block_vocab = GROUPED_HEAD_BLOCK_VOCAB
            block_hidden = GROUPED_HEAD_BLOCK_HIDDEN
            num_warps = 4
        elif backend == "warp_gemv":
            kernel = grouped_lm_head_warp_gemv_kernel
            block_vocab = WARP_GEMV_BLOCK_VOCAB
            block_hidden = GROUPED_HEAD_BLOCK_HIDDEN
            num_warps = 8
        else:
            kernel = grouped_lm_head_warp_accumulate_kernel
            block_vocab = WARP_ACCUMULATE_OUTPUTS_PER_CTA
            block_hidden = WARP_ACCUMULATE_BLOCK_HIDDEN
            num_warps = 4
        kernel[(rows, triton.cdiv(packed_weights.shape[1], block_vocab))](
            hidden_states,
            packed_weights,
            hidden_indices,
            head_indices,
            active_rows,
            logits,
            hidden_states.shape[0],
            packed_weights.shape[0],
            packed_weights.shape[1],
            packed_weights.shape[2],
            block_vocab,
            block_hidden,
            num_warps=num_warps,
        )
    else:
        valid_rows = (
            active_rows
            & (hidden_indices >= 0)
            & (hidden_indices < hidden_states.shape[0])
            & (head_indices >= 0)
            & (head_indices < packed_weights.shape[0])
        )
        safe_hidden_indices = hidden_indices.clamp(0, hidden_states.shape[0] - 1)
        safe_head_indices = head_indices.clamp(0, packed_weights.shape[0] - 1)
        for row in range(rows):
            logits[row].copy_(
                torch.nn.functional.linear(
                    hidden_states[safe_hidden_indices[row]],
                    packed_weights[safe_head_indices[row]],
                )
            )
        logits.masked_fill_(~valid_rows[:, None], 0.0)
    return logits

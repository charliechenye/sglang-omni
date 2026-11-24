# SPDX-License-Identifier: Apache-2.0
"""Indexed grouped LM-head projections for packed Predictor rows."""

from __future__ import annotations

import torch

GROUPED_HEAD_BLOCK_VOCAB = 64
GROUPED_HEAD_BLOCK_HIDDEN = 32

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

else:
    grouped_lm_head_kernel = None


def grouped_lm_head_cuda_available() -> bool:
    """Return whether the row-selected CUDA projection kernel is importable."""
    return triton is not None and grouped_lm_head_kernel is not None


def grouped_lm_head_linear(
    hidden_states: torch.Tensor,
    packed_weights: torch.Tensor,
    hidden_indices: torch.Tensor,
    head_indices: torch.Tensor,
    active_rows: torch.Tensor,
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
    logits = torch.empty(
        (rows, packed_weights.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if rows == 0:
        return logits
    else:
        pass
    if hidden_states.is_cuda:
        if not grouped_lm_head_cuda_available():
            raise RuntimeError("grouped CUDA LM-head kernel is unavailable")
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
        grouped_lm_head_kernel[
            (
                rows,
                triton.cdiv(packed_weights.shape[1], GROUPED_HEAD_BLOCK_VOCAB),
            )
        ](
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
            GROUPED_HEAD_BLOCK_VOCAB,
            GROUPED_HEAD_BLOCK_HIDDEN,
            num_warps=4,
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

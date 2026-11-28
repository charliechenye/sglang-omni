# SPDX-License-Identifier: Apache-2.0
"""Seed-isolated draws for speculative CodePredictor sampling.

The seed is the request's subtalker sampling seed. Transport request IDs are not
part of the stream so explicit seeds remain reproducible across submissions.
"""

from __future__ import annotations

from typing import Literal

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    murmur_hash32_pytorch,
    speculative_uniforms_frame_pool_kernel,
    triton,
)

SpeculativeDrawPurpose = Literal["proposal", "acceptance", "correction", "bonus"]

SPECULATIVE_DRAW_SALTS: dict[SpeculativeDrawPurpose, int] = {
    "proposal": 0x68E31DA4,
    "acceptance": 0x1D8E4E27,
    "correction": 0x7A4C2D61,
    "bonus": 0x3C6EF372,
}


def speculative_uniforms(
    seeds: torch.Tensor,
    semantic_positions: torch.Tensor,
    round_index: int,
    num_code_groups: int,
    column_count: int,
    purpose: SpeculativeDrawPurpose,
) -> torch.Tensor:
    """Build graph-safe draws from seed, position, round, purpose, and index."""
    if seeds.ndim != 1 or semantic_positions.shape != seeds.shape:
        raise ValueError("speculative RNG seeds and positions must be aligned rows")
    else:
        pass
    if seeds.device != semantic_positions.device:
        raise ValueError("speculative RNG tensors must share one device")
    else:
        pass
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
        or round_index >= num_code_groups
    ):
        raise ValueError("speculative RNG round index is outside the frame")
    else:
        pass
    if column_count <= 0 or num_code_groups <= 0:
        raise ValueError("speculative RNG dimensions must be positive")
    else:
        pass
    try:
        salt = SPECULATIVE_DRAW_SALTS[purpose]
    except KeyError as exc:
        raise ValueError("unsupported speculative draw purpose") from exc
    position = semantic_positions.to(torch.long) * (2 * num_code_groups) + round_index
    hashes = murmur_hash32_pytorch(seeds ^ salt, position, column_count)
    uniforms = (hashes.to(torch.float64) + 0.5) / float(2**32)
    return uniforms.to(torch.float32).clamp_(max=1.0 - 2.0**-24)


def speculative_uniforms_frame_pool(
    seeds: torch.Tensor,
    semantic_positions: torch.Tensor,
    num_code_groups: int,
    max_rounds: int,
    max_verify_width: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Precompute four request-isolated draw domains for every legal frame round."""
    if seeds.ndim != 1 or semantic_positions.shape != seeds.shape:
        raise ValueError("speculative RNG seeds and positions must be aligned rows")
    elif seeds.device != semantic_positions.device:
        raise ValueError("speculative RNG tensors must share one device")
    elif num_code_groups <= 2 or max_rounds != num_code_groups - 2:
        raise ValueError("frame RNG pool must cover every speculative frame round")
    elif max_verify_width < 2:
        raise ValueError("frame RNG pool must cover a proposal and target row")
    elif (
        not seeds.is_cuda
        or triton is None
        or speculative_uniforms_frame_pool_kernel is None
    ):
        raise RuntimeError("frame speculative RNG pool requires CUDA and Triton")
    elif output is not None and (
        output.ndim != 4
        or output.shape[0] < seeds.shape[0]
        or output.shape[1] != 4
        or output.shape[2] < max_rounds
        or output.shape[3] < max_verify_width
        or output.device != seeds.device
        or output.dtype is not torch.float32
    ):
        raise ValueError("persistent speculative RNG pool has an invalid shape")
    else:
        pass
    if output is None:
        output = torch.empty(
            (seeds.shape[0], 4, max_rounds, max_verify_width),
            device=seeds.device,
            dtype=torch.float32,
        )
    else:
        pass
    if (
        not output.is_contiguous()
        or not seeds.is_contiguous()
        or not semantic_positions.is_contiguous()
    ):
        raise ValueError("speculative RNG pool inputs must be contiguous")
    else:
        pass
    speculative_uniforms_frame_pool_kernel[(seeds.shape[0], max_rounds)](
        seeds,
        semantic_positions,
        output,
        OUTPUT_STRIDE_ROW=output.stride(0),
        OUTPUT_STRIDE_PURPOSE=output.stride(1),
        OUTPUT_STRIDE_ROUND=output.stride(2),
        OUTPUT_STRIDE_COLUMN=output.stride(3),
        NUM_CODE_GROUPS=num_code_groups,
        MAX_COLUMNS=max_verify_width,
        BLOCK_COLUMNS=triton.next_power_of_2(max_verify_width),
        PROPOSAL_SALT=SPECULATIVE_DRAW_SALTS["proposal"],
        ACCEPTANCE_SALT=SPECULATIVE_DRAW_SALTS["acceptance"],
        CORRECTION_SALT=SPECULATIVE_DRAW_SALTS["correction"],
        BONUS_SALT=SPECULATIVE_DRAW_SALTS["bonus"],
        num_warps=4,
    )
    return output[: seeds.shape[0], :, :max_rounds, :max_verify_width]

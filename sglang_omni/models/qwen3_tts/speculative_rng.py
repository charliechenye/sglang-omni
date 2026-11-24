# SPDX-License-Identifier: Apache-2.0
"""Request-isolated random draws for unseeded CodePredictor speculation."""

from __future__ import annotations

from typing import Literal

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    murmur_hash32_pytorch,
    speculative_uniforms_four_purpose_kernel,
    triton,
)
from sglang_omni.sampling.seed import derive_sampling_seed

SpeculativeDrawPurpose = Literal["proposal", "acceptance", "correction", "bonus"]

SPECULATIVE_DRAW_SALTS: dict[SpeculativeDrawPurpose, int] = {
    "proposal": 0x68E31DA4,
    "acceptance": 0x1D8E4E27,
    "correction": 0x7A4C2D61,
    "bonus": 0x3C6EF372,
}


def derive_speculative_request_seed(internal_seed: int, request_id: str) -> int:
    """Add request identity to the internally generated per-request seed."""
    if not request_id:
        raise ValueError("speculative random draws require a request ID")
    else:
        pass
    if isinstance(internal_seed, bool) or not isinstance(internal_seed, int):
        raise ValueError("internal sampling seed must be an integer")
    else:
        pass
    return derive_sampling_seed(
        "qwen3-tts-codepredictor-speculative", internal_seed, request_id
    )


def speculative_uniforms(
    seeds: torch.Tensor,
    semantic_positions: torch.Tensor,
    round_index: int,
    num_code_groups: int,
    column_count: int,
    purpose: SpeculativeDrawPurpose,
) -> torch.Tensor:
    """Build graph-safe row-isolated uniforms for one speculative draw purpose."""
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


def speculative_uniforms_all_purposes(
    seeds: torch.Tensor,
    semantic_positions: torch.Tensor,
    round_index: torch.Tensor,
    num_code_groups: int,
    proposal_width: int,
    verify_width: int,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all four draw domains in one graph-safe CUDA launch."""
    if seeds.ndim != 1 or semantic_positions.shape != seeds.shape:
        raise ValueError("speculative RNG seeds and positions must be aligned rows")
    elif seeds.device != semantic_positions.device:
        raise ValueError("speculative RNG tensors must share one device")
    elif round_index.ndim != 0 or round_index.device != seeds.device:
        raise ValueError("runtime speculative round index must be a device scalar")
    elif num_code_groups <= 0 or proposal_width <= 0:
        raise ValueError("speculative RNG dimensions must be positive")
    elif verify_width != proposal_width + 1:
        raise ValueError("verify width must include one trailing target row")
    elif (
        not seeds.is_cuda
        or triton is None
        or speculative_uniforms_four_purpose_kernel is None
    ):
        raise RuntimeError("fused speculative RNG requires CUDA and Triton")
    elif output is not None and (
        output.ndim != 3
        or output.shape[0] < seeds.shape[0]
        or output.shape[1] != 4
        or output.shape[2] < verify_width
        or output.device != seeds.device
        or output.dtype is not torch.float32
    ):
        raise ValueError("persistent speculative RNG output has an invalid shape")
    else:
        pass
    max_columns = verify_width
    block_columns = triton.next_power_of_2(max_columns)
    if output is None:
        output = torch.empty(
            (seeds.shape[0], 4, max_columns),
            device=seeds.device,
            dtype=torch.float32,
        )
    else:
        pass
    speculative_uniforms_four_purpose_kernel[(seeds.shape[0],)](
        seeds,
        semantic_positions,
        round_index,
        output,
        OUTPUT_STRIDE_ROW=output.stride(0),
        OUTPUT_STRIDE_PURPOSE=output.stride(1),
        OUTPUT_STRIDE_COLUMN=output.stride(2),
        NUM_CODE_GROUPS=num_code_groups,
        PROPOSAL_WIDTH=proposal_width,
        VERIFY_WIDTH=verify_width,
        MAX_COLUMNS=max_columns,
        BLOCK_COLUMNS=block_columns,
        PROPOSAL_SALT=SPECULATIVE_DRAW_SALTS["proposal"],
        ACCEPTANCE_SALT=SPECULATIVE_DRAW_SALTS["acceptance"],
        CORRECTION_SALT=SPECULATIVE_DRAW_SALTS["correction"],
        BONUS_SALT=SPECULATIVE_DRAW_SALTS["bonus"],
        num_warps=4,
    )
    return (
        output[: seeds.shape[0], 0, :proposal_width],
        output[: seeds.shape[0], 1, :proposal_width],
        output[: seeds.shape[0], 2, :verify_width],
        output[: seeds.shape[0], 3, :verify_width],
    )

# SPDX-License-Identifier: Apache-2.0
"""Request-isolated random draws for unseeded CodePredictor speculation."""

from __future__ import annotations

from typing import Literal

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import murmur_hash32_pytorch
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

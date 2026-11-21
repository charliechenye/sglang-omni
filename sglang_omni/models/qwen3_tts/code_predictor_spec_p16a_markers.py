# SPDX-License-Identifier: Apache-2.0
"""P16A only: distinct GPU kernel markers for actual graph-node traces.

The round marker executes once for each causal round in both graph arms.
The bonus marker executes ONLY immediately before the bonus LM head,
inside the true conditional branch (or always in the unconditional baseline).
Their presence is physical branch evidence, not by itself proof that a
specific model GEMM was skipped. Never enable in normal serving.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def p16a_round_start_marker_kernel(IN, OUT):
    value = tl.load(IN)
    tl.store(OUT, value)


@triton.jit
def p16a_bonus_head_marker_kernel(IN, OUT):
    value = tl.load(IN)
    tl.store(OUT, value)


def mark_round(hidden: torch.Tensor) -> None:
    # No host synchronization, explicit GPU activity with identifiable name.
    scratch = torch.empty((1,), device=hidden.device, dtype=hidden.dtype)
    p16a_round_start_marker_kernel[(1,)](
        hidden, scratch, num_warps=4
    )


def mark_bonus(after_hidden: torch.Tensor) -> None:
    scratch = torch.empty((1,), device=after_hidden.device,
                          dtype=after_hidden.dtype)
    p16a_bonus_head_marker_kernel[(1,)](
        after_hidden, scratch, num_warps=4
    )

# SPDX-License-Identifier: Apache-2.0
"""P11B: opt-in AR prefix, followed by the exact existing sampled K2 loop.

For Qwen3-TTS's 16 codec groups, the opening codebook (depth 1) stays
unchanged. Direct target AR commits depth 2 and optionally 3, then the
existing K2 branch resumes at the next depth on the SAME KV cache.
Each direct-AR code uses the real target head AFTER consuming the most
recent committed code embedding; the future K2 q head sees that same
new hidden state. This is **not** a production optimized GPU K2 path.

Activation is request-scoped: current K2 enable/ARM must be active AND a
separate P11B hybrid marker must exist. Without the marker, original K2
is untouched in the same running server. The AR sampling uses the
debug K2's already initialized torch.Generator, so a hybrid run draws a
different RNG sequence than pure K2; matching WAV hashes is NOT parity.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch

HYBRID_ARM = "SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE"
HYBRID_THROUGH = "SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH"
AUDIT = "SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT"
VALID_THROUGH = (2, 3)


def active_ar_through() -> int:
    """One-shot policy decision at start of each K2 frame (no silent fallback).

    1 means original K2: no AR-prefix steps. The user may keep the
    marker PATH configured throughout the server lifetime and create
    or delete only the marker file between HTTP requests.
    """
    marker = os.environ.get(HYBRID_ARM)
    if not marker:
        return 1
    target = os.environ.get(HYBRID_THROUGH)
    if target is None or not target.isdecimal() or int(target) not in VALID_THROUGH:
        raise ValueError("P11B needs AR_THROUGH_DEPTH=2 or 3 with HYBRID_ARM_FILE")
    if not Path(marker).is_file():
        return 1
    return int(target)


def direct_target_ar_step(
    *,
    model,
    depth: int,
    cache_len: int,
    last_embedding: torch.Tensor,
    commit: Callable[[int, torch.Tensor], torch.Tensor],
    generator: torch.Generator,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Commit exactly one target AR code without a speculative draft.

    The next K2 q head MUST read target_hidden from before consuming
    the newly generated code (same state convention as rejected K2).
    Each forward advances valid KV by one; no draft suffix exists.
    """
    if not 1 <= depth <= 2 or cache_len != depth + 1:
        raise ValueError(
            f"P11B unexpected AR prefix cache depth={depth}, cache_len={cache_len}"
        )
    from .code_predictor_spec import sampling_probs

    h_after_committed = model.predictor_forward_tokens(
        token_embeds=last_embedding.clone(),
        batch_size=1,
        cache_len=cache_len,
    ).clone()
    logits, _ = model.code_predictor.lm_head[depth](h_after_committed)
    p = sampling_probs(
        logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
    )
    new_code = torch.multinomial(p, 1, generator=generator).reshape(1)
    new_depth = depth + 1
    new_embedding = commit(new_depth, new_code)
    return new_depth, cache_len + 1, h_after_committed, new_embedding


def validate_counts(
    *, ar_steps: int, rounds: int, proposals: int,
    accepted: int, rejected: int, bonus: int,
    residual_codes: int = 14,
) -> None:
    """Conservation of all 14 residual code slots, including AR prefix."""
    if (
        not all(
            type(x) is int and x >= 0
            for x in (
                ar_steps, rounds, proposals,
                accepted, rejected, bonus,
            )
        )
        or ar_steps not in (0, 1, 2)
        or rounds != proposals
        or proposals != accepted + rejected
        or ar_steps + proposals + bonus != residual_codes
    ):
        raise RuntimeError(
            "P11B residual code accounting invalid "
            f"ar={ar_steps} rounds={rounds} proposed={proposals} "
            f"accepted={accepted} rejected={rejected} bonus={bonus}"
        )


def stats_line(
    *,
    frame: int,
    ar_through: int,
    ar: list[int],
    proposed: list[int],
    accepted: list[int],
    bonus: list[int],
) -> str:
    """Fixed parseable log; all lists have sorted unique residual depths."""
    if not 1 <= ar_through <= 3:
        raise ValueError("P11B ar_through out of range")
    for lst in (ar, proposed, accepted, bonus):
        if lst != sorted(set(lst)) or any(not 2 <= d <= 15 for d in lst):
            raise ValueError("P11B audit depths invalid")
    if len(set(ar) & set(proposed)) > 0:
        raise ValueError("P11B cannot AR and propose the same residual depth")
    if any(v not in proposed for v in accepted):
        raise ValueError("P11B accepted must be a proposed depth")
    def fmt(values: list[int]) -> str:
        return ",".join(str(x) for x in values) if values else "-"
    return (
        f"QTTS_CP_K2_DEPTHS frame={frame} ar_through={ar_through} "
        f"ar={fmt(ar)} proposed={fmt(proposed)} "
        f"accepted={fmt(accepted)} bonus={fmt(bonus)}"
    )

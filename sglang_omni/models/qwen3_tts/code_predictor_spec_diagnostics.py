# SPDX-License-Identifier: Apache-2.0
"""Opt-in correctness diagnostics for Qwen3-TTS CP SpecDec K2.

NO production performance claim. Re-executes the real model's two-token
verifier as two serial target calls, compares hidden/logits/KV using the *same*
checkpoint and cache prefix, and restores the packed cache after observing it.
Only invoked inside already ARM-gated K2 C1 sampled execution.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

SCHEMA = "qwen3tts-cpspec-k2-per-depth-parity-v1"
ATOL = 2**-6
RTOL = 2**-7


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    if a.shape != b.shape:
        raise ValueError(f"parity shapes mismatch: {a.shape} vs {b.shape}")
    if a.numel() == 0:
        raise ValueError("empty parity tensors")
    x, y = a.detach().float(), b.detach().float()
    delta = (x - y).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "close": bool(torch.allclose(x, y, atol=ATOL, rtol=RTOL)),
    }


def check_packed_verifier(
    model: "Qwen3TTSTalker",
    *,
    cache_len: int,
    depth: int,
    last_embedding: torch.Tensor,
    proposed_embedding: torch.Tensor,
    packed_hidden: torch.Tensor,
    prefix_k_before: torch.Tensor,
    prefix_v_before: torch.Tensor,
) -> dict:
    """Real-model packed-vs-two-serial target parity on identical prefix.

    The original packed KV (including its valid cache prefix) is restored in
    a finally block. Serial replay does not sample tokens, consume the
    generator or advance model counters. Any failed parity must stop the
    experiment; there is no silent fallback.
    """
    if cache_len < 2 or depth < 1 or depth >= model.config.num_code_groups - 1:
        raise ValueError("invalid target verifier depth/cache offset")
    if packed_hidden.shape[1] != 2 or last_embedding.shape[1] != 1 or proposed_embedding.shape[1] != 1:
        raise ValueError("P1 checks exactly a two-token target append")
    end = cache_len + 2
    if prefix_k_before.shape[2] != cache_len or prefix_v_before.shape[2] != cache_len:
        raise ValueError("must snapshot exact logical prefix before packed verification")
    k_cache = model.predictor_k_cache
    v_cache = model.predictor_v_cache
    if end > k_cache.shape[2]:
        raise ValueError("2-token verification exceeds allocated KV")
    # Snapshot entire *logical* prefix + appended suffix. Repeated serial
    # forward calls may mutate the same GPU workspace, so never trust views.
    saved_k = k_cache[:, :1, :end].clone()
    saved_v = v_cache[:, :1, :end].clone()
    before_checks = {
        "packed_prefix_k_preserved": _compare(
            prefix_k_before, saved_k[:, :, :cache_len]
        ),
        "packed_prefix_v_preserved": _compare(
            prefix_v_before, saved_v[:, :, :cache_len]
        ),
    }
    packed = packed_hidden.detach().clone()
    try:
        first = model.predictor_forward_tokens(
            token_embeds=last_embedding.clone(), batch_size=1, cache_len=cache_len
        ).clone()
        second = model.predictor_forward_tokens(
            token_embeds=proposed_embedding.clone(), batch_size=1,
            cache_len=cache_len + 1
        ).clone()
        logits_packed, _ = model.code_predictor.lm_head[depth](packed[:, :1])
        logits_serial, _ = model.code_predictor.lm_head[depth](first)
        checks = {
            **before_checks,
            "hidden_first": _compare(packed[:, :1], first),
            "hidden_second": _compare(packed[:, 1:], second),
            "target_logits": _compare(logits_packed, logits_serial),
            "kv_prefix_k": _compare(saved_k[:, :, :cache_len], k_cache[:, :1, :cache_len]),
            "kv_prefix_v": _compare(saved_v[:, :, :cache_len], v_cache[:, :1, :cache_len]),
            "kv_suffix_k": _compare(saved_k[:, :, cache_len:end], k_cache[:, :1, cache_len:end]),
            "kv_suffix_v": _compare(saved_v[:, :, cache_len:end], v_cache[:, :1, cache_len:end]),
        }
        return {
            "depth": depth + 1,
            "cache_len": cache_len,
            "checks": checks,
            "pass": all(x["close"] for x in checks.values()),
            "atol": ATOL,
            "rtol": RTOL,
        }
    finally:
        # The diagnostic must not alter the candidate's KV trajectory.
        k_cache[:, :1, :end].copy_(saved_k)
        v_cache[:, :1, :end].copy_(saved_v)


def check_embedding_sum(
    model: "Qwen3TTSTalker",
    layer0_embedding: torch.Tensor,
    pos_codes: torch.Tensor,
    pos_summed: torch.Tensor,
) -> dict:
    """Compare residual embedding sum with target code indices, same BF16 order."""
    total = model.config.num_code_groups
    if pos_codes.shape != (1, total):
        raise ValueError("unexpected output codec group count")
    reconstructed = layer0_embedding[:, 0, :].clone().to(pos_summed.dtype)
    tables = model.code_predictor.model.codec_embedding.weight
    for depth in range(1, total):
        reconstructed.add_(
            F.embedding(pos_codes[:, depth], tables[depth - 1]).to(pos_summed.dtype)
        )
    result = _compare(pos_summed, reconstructed)
    result["code_groups"] = total
    return result


def frame_enabled(frame_index: int) -> tuple[Path | None, bool]:
    """Only a bounded number of already-arm-gated frames create debug traces."""
    path = os.environ.get("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR")
    if not path:
        return None, False
    max_frames = int(os.environ.get("SGLANG_OMNI_QTTS_CP_K2_DIAG_MAX_FRAMES", "16"))
    max_parity = int(os.environ.get("SGLANG_OMNI_QTTS_CP_K2_PARITY_MAX_FRAMES", "4"))
    if not 0 <= max_parity <= max_frames <= 4096:
        raise ValueError("CP K2 diagnostic frame limits are invalid")
    if frame_index > max_frames:
        return None, False
    root = Path(path).expanduser().resolve()
    return root, frame_index <= max_parity


def write_trace(root: Path, frame_index: int, payload: dict, *, failed: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = f"frame_pid{os.getpid()}_{frame_index:05d}"
    if failed:
        name += "_FAIL"
    dest = root / (name + ".json")
    with dest.open("x", encoding="utf-8") as output:
        json.dump({"schema": SCHEMA, "pid": os.getpid(), **payload}, output, indent=2)
        output.write("\n")
    return dest


def assert_diagnostic_parity(row: dict) -> None:
    if not row.get("pass"):
        failed = [k for k, value in row["checks"].items() if not value["close"]]
        raise RuntimeError(
            f"CP_SPEC_K2_REAL_TARGET_PARITY_FAIL depth={row['depth']} "
            f"cache_len={row['cache_len']} checks={failed}"
        )


def validate_depth_events(events: list[dict], *, expected_residual: int = 14) -> None:
    """Independent, CPU-testable accounting for a 16-group K2 frame."""
    if not events:
        raise ValueError("no speculative events")
    used = 1
    for event in events:
        depth = event["proposal_depth"]
        if depth != used + 1:
            raise ValueError(f"wrong proposal depth: expected {used + 1}, got {depth}")
        overlap = event["expected_overlap"]
        if not isinstance(overlap, (int, float)) or not math.isfinite(overlap) or not 0 <= overlap <= 1:
            raise ValueError("invalid per-depth p/q overlap")
        if type(event["accepted"]) is not bool:
            raise ValueError("accepted must be bool")
        used = depth
        if event["accepted"] and depth < expected_residual + 1:
            if event.get("bonus_depth") != depth + 1:
                raise ValueError("missing bonus depth on accepted nonterminal proposal")
            used += 1
        elif event.get("bonus_depth") is not None:
            raise ValueError("spurious bonus depth")
    if used != expected_residual + 1:
        raise ValueError(f"frame incomplete: last depth={used}")

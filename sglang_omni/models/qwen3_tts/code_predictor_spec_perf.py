# SPDX-License-Identifier: Apache-2.0
"""Opt-in H200 fixed-on-policy-state K2 cost probe.

CUDA-event intervals are launch envelopes (potential host launch gaps), NOT
pure GPU kernel busy time, CUDA Graph replay, or a serving speedup.
Only runs under manually ARMed C1 K2. Exact K/V prefix and verifier suffix
are restored; a separate generator prevents altering the real sampler RNG.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .code_predictor_spec import accept_or_residual, sampling_probs

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

SCHEMA = "qwen3tts-k2-p4-fixed-state-forward-cost-v1"
ROOT_ENV = "SGLANG_OMNI_QTTS_CP_P4_DIR"


def parse_depths(raw: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(x.strip()) for x in raw.split(","))
    except ValueError as exc:
        raise ValueError("P4 depths must be integers") from exc
    if not depths or len(set(depths)) != len(depths) or any(d < 2 or d > 15 for d in depths):
        raise ValueError("P4 depths must be distinct indices 2..15")
    return depths


def read_config() -> dict | None:
    root = os.environ.get(ROOT_ENV)
    if not root:
        return None
    if os.environ.get("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR"):
        raise RuntimeError("P4 cost probes and P1 parity instrumentation must use separate runs")
    max_states = int(os.environ.get("SGLANG_OMNI_QTTS_CP_P4_MAX_STATES", "4"))
    repeats = int(os.environ.get("SGLANG_OMNI_QTTS_CP_P4_REPEATS", "12"))
    warmup = int(os.environ.get("SGLANG_OMNI_QTTS_CP_P4_WARMUP", "3"))
    if not 1 <= max_states <= 16 or not 3 <= repeats <= 128 or not 1 <= warmup <= 16:
        raise ValueError("P4 max_states 1..16, repeats 3..128, warmup 1..16")
    return {
        "root": Path(root).expanduser().resolve(),
        "max_states": max_states,
        "depths": parse_depths(os.environ.get("SGLANG_OMNI_QTTS_CP_P4_DEPTHS", "2,6,10,14")),
        "repeats": repeats,
        "warmup": warmup,
    }


def time_repeated(fn: Callable[[], object], *, repeats: int, warmup: int) -> dict:
    """CUDA-event launch envelope and host wall ms per fixed-state invocation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    stop.synchronize()
    host_ms = 1000 * (time.perf_counter() - t0) / repeats
    event_ms = start.elapsed_time(stop) / repeats
    if not 0 < event_ms < max(1.0, 1.5 * host_ms):
        raise RuntimeError("invalid P4 CUDA-event measurement")
    return {"cuda_event_ms": event_ms, "host_wall_ms": host_ms}


def profile_state(
    model: "Qwen3TTSTalker",
    *,
    proposal_depth: int,
    cache_len: int,
    last_embedding: torch.Tensor,
    proposed_embedding: torch.Tensor,
    draft_hidden: torch.Tensor,
    target_head: torch.nn.Module,
    p: torch.Tensor,
    q: torch.Tensor,
    proposal: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repeats: int,
    warmup: int,
) -> dict:
    if (
        proposal_depth < 2 or proposal_depth > 15 or cache_len < 2
        or last_embedding.shape[0:2] != (1, 1)
        or proposed_embedding.shape[0:2] != (1, 1)
        or p.shape != q.shape or p.ndim != 2 or proposal.numel() != 1
        or not last_embedding.is_cuda
    ):
        raise ValueError("invalid fixed on-policy P4 state")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("P4 cannot profile during CUDA Graph capture")
    end = cache_len + 2
    if end > model.predictor_k_cache.shape[2]:
        raise ValueError("P4 KV capacity exceeded")
    k_cache = model.predictor_k_cache
    v_cache = model.predictor_v_cache
    original_k = k_cache[:, :1, :end].clone()
    original_v = v_cache[:, :1, :end].clone()
    packed_input = torch.cat((last_embedding, proposed_embedding), dim=1)
    test_rng = torch.Generator(device=p.device)
    test_rng.manual_seed(20261121)

    def serial_one():
        return model.predictor_forward_tokens(
            token_embeds=last_embedding, batch_size=1, cache_len=cache_len
        )

    def serial_two():
        serial_one()
        return model.predictor_forward_tokens(
            token_embeds=proposed_embedding, batch_size=1, cache_len=cache_len+1
        )

    def packed_two():
        return model.predictor_forward_tokens(
            token_embeds=packed_input, batch_size=1, cache_len=cache_len
        )

    def draft_head():
        return target_head(draft_hidden)

    def draft_select_debug():
        logits, _ = target_head(draft_hidden)
        probs = sampling_probs(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        return torch.multinomial(probs, 1, generator=test_rng)

    def accept_debug():
        return accept_or_residual(proposal, p, q, test_rng)

    cells = {
        "serial_one_stack": serial_one,
        "serial_two_stack": serial_two,
        "packed_two_stack": packed_two,
        "draft_head": draft_head,
        "draft_select_debug": draft_select_debug,
        "accept_debug": accept_debug,
    }
    schedule = (
        "serial_one_stack", "packed_two_stack", "serial_two_stack",
        "draft_head", "draft_select_debug", "accept_debug",
        "accept_debug", "draft_select_debug", "draft_head",
        "serial_two_stack", "packed_two_stack", "serial_one_stack",
    )
    samples: dict[str, list[dict]] = {name: [] for name in cells}
    try:
        with torch.inference_mode():
            for name in schedule:
                samples[name].append(
                    time_repeated(cells[name], repeats=repeats, warmup=warmup)
                )
    finally:
        k_cache[:, :1, :end].copy_(original_k)
        v_cache[:, :1, :end].copy_(original_v)
        torch.cuda.synchronize()
    measurements = {
        name: {
            "cuda_event_ms": statistics.median(x["cuda_event_ms"] for x in rows),
            "host_wall_ms": statistics.median(x["host_wall_ms"] for x in rows),
            "blocks": len(rows),
        }
        for name, rows in samples.items()
    }
    return {
        "status": "P4_FIXED_STATE_LAUNCH_ENVELOPE_ONLY_NOT_GPU_BUSY_OR_SPEEDUP",
        "proposal_depth": proposal_depth,
        "cache_len": cache_len,
        "gpu": torch.cuda.get_device_name(last_embedding.device),
        "dtype": str(last_embedding.dtype),
        "repeats": repeats,
        "warmup": warmup,
        "measurements": measurements,
        "limitations": [
            "Identical fixed token inputs and existing real-model KV prefix per cell",
            "CUDA event envelopes include Python host gaps and are not GPU kernel busy times",
            "Debug sampler and accept control include per-round host synchronization",
            "Forward stack timings exclude normal AR head/sample/embedding overhead",
            "No live fused-K2 graph, no optimized Graph-ON end-to-end speedup proof",
        ],
    }


def maybe_profile(
    model: "Qwen3TTSTalker",
    *,
    frame_index: int,
    depth: int,
    cache_len: int,
    last_embedding: torch.Tensor,
    proposed_embedding: torch.Tensor,
    draft_hidden: torch.Tensor,
    target_head: torch.nn.Module,
    p: torch.Tensor,
    q: torch.Tensor,
    proposal: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    config = read_config()
    if config is None:
        return
    proposal_depth = depth + 1
    seen = getattr(model, "_qt_cp_p4_seen_depths", None)
    if seen is None:
        seen = set()
        model._qt_cp_p4_seen_depths = seen
    if (
        proposal_depth not in config["depths"]
        or proposal_depth in seen
        or len(seen) >= config["max_states"]
    ):
        return
    record = profile_state(
        model, proposal_depth=proposal_depth, cache_len=cache_len,
        last_embedding=last_embedding, proposed_embedding=proposed_embedding,
        draft_hidden=draft_hidden, target_head=target_head,
        p=p, q=q, proposal=proposal, temperature=temperature,
        top_k=top_k, top_p=top_p, repeats=config["repeats"],
        warmup=config["warmup"],
    )
    seen.add(proposal_depth)
    record.update({
        "pid": os.getpid(),
        "frame": frame_index,
        "source_sha": os.environ.get("SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"),
        "num_code_groups": model.config.num_code_groups,
        "num_predictor_layers": len(model.code_predictor.model.layers),
    })
    root = config["root"]
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    dest = root / (
        f"perf_pid{os.getpid()}_frame{frame_index:05d}_depth{proposal_depth:02d}.json"
    )
    with dest.open("x", encoding="utf-8") as fp:
        json.dump({"schema": SCHEMA, **record}, fp, indent=2)
        fp.write("\n")

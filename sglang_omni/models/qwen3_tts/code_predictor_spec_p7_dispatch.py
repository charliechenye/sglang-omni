# SPDX-License-Identifier: Apache-2.0
"""P7 H200: one captured CUDA IF graph selects real Predictor KV offsets 2/6.

Bounded, ARM-gated source-locked experiment only. We intentionally use the
PyTorch 2.14 internal cond_op and CUDAGraphCaptureControlFlowOpDispatchMode,
NOT torch.switch (the 2.14 Inductor implementation disables CUDA graphs for
switch). This verifies an actual model's packed-two kernel branch dispatch
on the SAME living model, with full K/V restoration.

This is NOT a dynamic GPU WHILE+real model integration, RNG/accept/bonus
implementation, production-safe graph-cache policy, or serving speedup.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

ROOT_ENV = "SGLANG_OMNI_QTTS_CP_P7_DIR"
SCHEMA = "qwen3tts-cpspec-p7-two-branch-real-verifier-v1"
OFFSETS = (2, 6)
ATOL = 2**-6
RTOL = 2**-7


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    if a.shape != b.shape or a.numel() == 0:
        raise ValueError("P7 comparison requires identical nonempty tensor shapes")
    x = a.detach().float()
    y = b.detach().float()
    abs_diff = (x - y).abs()
    return {
        "max_abs": float(abs_diff.max().item()),
        "close": bool(torch.allclose(x, y, atol=ATOL, rtol=RTOL)),
    }


def capture_two_offsets(
    model: "Qwen3TTSTalker",
    *,
    fixed_embeds: torch.Tensor,
    repeats: int,
) -> dict:
    """Capture conditional true(offset2)/false(offset6) real target stack.

    Both branch calls execute on exactly the SAME saved model prefix and
    identical fixed token embeddings. Choice is a GPU scalar, changed only
    between graph replays. Oracle eager runs and captures are individually
    restored to the same initial K/V before comparing result and suffix.
    """
    from torch._higher_order_ops.cond import cond_op
    from torch._higher_order_ops.cudagraph_conditional_nodes import (
        CUDAGraphCaptureControlFlowOpDispatchMode,
    )

    if (
        fixed_embeds.shape[:2] != (1, 2)
        or not fixed_embeds.is_cuda
        or not 8 <= repeats <= 256
        or model.predictor_k_cache.shape[2] < 8
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("P7 requires C1 real CUDA 2-token inputs with 8 KV slots")
    device = fixed_embeds.device
    cache_k = model.predictor_k_cache
    cache_v = model.predictor_v_cache
    original_k = cache_k[:, :1, :8].clone()
    original_v = cache_v[:, :1, :8].clone()
    pair = fixed_embeds.detach().contiguous().clone()
    selector = torch.ones((), device=device, dtype=torch.bool)
    result = {
        "status": "P7_FAILED_NOT_PROOF",
        "gpu": torch.cuda.get_device_name(device),
        "source_sha": os.environ.get("SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"),
        "offsets": list(OFFSETS),
        "repeats": repeats,
        "cases": [],
    }

    def branch2(embeds):
        return model.predictor_forward_tokens(
            token_embeds=embeds, batch_size=1, cache_len=2,
        ).clone()

    def branch6(embeds):
        return model.predictor_forward_tokens(
            token_embeds=embeds, batch_size=1, cache_len=6,
        ).clone()

    def operation():
        return cond_op(selector, branch2, branch6, (pair,))

    def restore():
        cache_k[:, :1, :8].copy_(original_k)
        cache_v[:, :1, :8].copy_(original_v)

    def gpu_snapshot():
        return (
            cache_k[:, :1, :8].clone(),
            cache_v[:, :1, :8].clone(),
        )

    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    try:
        # Explicitly initialize each fixed-offset variant before capture.
        with torch.inference_mode():
            for fn in (branch2, branch6):
                restore()
                for _ in range(2):
                    fn(pair)
        torch.cuda.synchronize(device)
        restore()
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(
                graph, stream=side, capture_error_mode="thread_local"
            ),
            CUDAGraphCaptureControlFlowOpDispatchMode(),
            torch.inference_mode(),
        ):
            output = operation()
        torch.cuda.synchronize(device)

        # The test consumes the one capture on both GPU choices; all K/V and
        # hidden comparisons include inactive-branch suffix ownership.
        for flag, depth in ((True, 2), (False, 6), (True, 2)):
            restore()
            selector.fill_(flag)
            graph.replay()
            torch.cuda.synchronize(device)
            graph_hidden = output.detach().clone()
            graph_k, graph_v = gpu_snapshot()

            restore()
            with torch.inference_mode():
                serial_hidden = model.predictor_forward_tokens(
                    token_embeds=pair, batch_size=1, cache_len=depth
                ).clone()
            serial_k, serial_v = gpu_snapshot()
            checks = {
                "hidden": compare(graph_hidden, serial_hidden),
                "kv_prefix_k": compare(
                    graph_k[:, :, :depth], serial_k[:, :, :depth]
                ),
                "kv_prefix_v": compare(
                    graph_v[:, :, :depth], serial_v[:, :, :depth]
                ),
                "kv_suffix_k": compare(
                    graph_k[:, :, depth:depth + 2], serial_k[:, :, depth:depth + 2]
                ),
                "kv_suffix_v": compare(
                    graph_v[:, :, depth:depth + 2], serial_v[:, :, depth:depth + 2]
                ),
                "inactive_kv_k": compare(
                    graph_k[:, :, depth + 2:8], serial_k[:, :, depth + 2:8]
                ) if depth < 6 else {"max_abs": 0.0, "close": True},
                "inactive_kv_v": compare(
                    graph_v[:, :, depth + 2:8], serial_v[:, :, depth + 2:8]
                ) if depth < 6 else {"max_abs": 0.0, "close": True},
            }
            passed = all(row["close"] for row in checks.values())
            result["cases"].append({
                "offset": depth, "selector": flag, "checks": checks,
                "pass": passed,
            })
            if not passed:
                raise RuntimeError(
                    f"P7_REAL_PREDICTOR_BRANCH_PARITY_FAIL at offset={depth}"
                )
        timings = {}
        for flag, depth in ((True, 2), (False, 6)):
            restore()
            selector.fill_(flag)
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize(device)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            t = begin.elapsed_time(end) / repeats
            if not math.isfinite(t) or t <= 0:
                raise RuntimeError("invalid P7 captured conditional time")
            timings[str(depth)] = t
        result["conditional_cuda_event_ms_by_offset"] = timings
        result["status"] = "P7_REAL_MODEL_TWO_FIXED_DEPTH_CUDA_COND_PASS_NOT_WHOLE_K2_OR_SPEEDUP"
        result["limitations"] = [
            "Only two fixed offsets (2 and 6), not 14-way routing or nested CUDA WHILE",
            "One real target stack at fixed embeddings; no GPU sampled tokens or rejection correction",
            "P7 measures CUDA Graph conditional replay launch envelope, not GPU kernel busy time",
            "All C1 model K/V is restored; no inference request performance claim",
        ]
    finally:
        torch.cuda.synchronize(device)
        restore()
        torch.cuda.synchronize(device)
    return result


def maybe_profile(
    model: "Qwen3TTSTalker",
    *,
    frame_index: int,
    proposal_depth: int,
    token_embeds: torch.Tensor,
) -> None:
    root_path = os.environ.get(ROOT_ENV)
    if not root_path:
        return
    if os.environ.get("SGLANG_OMNI_QTTS_CP_P4_DIR") or os.environ.get(
        "SGLANG_OMNI_QTTS_CP_P5_DIR"
    ) or os.environ.get("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR"):
        raise RuntimeError("P7 must run without P1/P4/P5 profiling")
    if proposal_depth != 6 or getattr(model, "_qt_cp_p7_profiled", False):
        return
    result = capture_two_offsets(model, fixed_embeds=token_embeds, repeats=32)
    output_dir = Path(root_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result.update({
        "pid": os.getpid(),
        "frame": frame_index,
        "real_proposal_depth": proposal_depth,
        "num_predictor_layers": len(model.code_predictor.model.layers),
    })
    file = output_dir / f"real_cond_pid{os.getpid()}_frame{frame_index:04d}.json"
    with file.open("x", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, **result}, handle, indent=2)
        handle.write("\n")
    model._qt_cp_p7_profiled = True

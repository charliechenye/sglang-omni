# SPDX-License-Identifier: Apache-2.0
"""P9A: four real Qwen3-TTS Predictor KV offsets in ONE GPU WHILE + IF graph.

Capture a five-layer real-checkpoint Predictor at static cache offsets
2/6/10/14; a GPU tensor chooses the branch and skips subsequent test stages.
Every captured replay is compared with a serial oracle from identical model
K/V and immutable token embeddings. Experimental C1 / post-health ARM only.

This gate DOES NOT implement full K2 sampling or contiguous residual-depth
transitions: the 4 offsets are spaced, the test inputs are frozen from an
actual on-policy proposal near depth 14, and the GPU Boolean skip mask is
synthetic. It measures the real model's 4-depth branch/control overhead,
not a production serving speedup. All original K/V is restored.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .code_predictor_spec_p7_dispatch import compare

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

ROOT_ENV = "SGLANG_OMNI_QTTS_CP_P9A_DIR"
SCHEMA = "qwen3tts-cpspec-p9a-four-real-offsets-v1"
OFFSETS = (2, 6, 10, 14)
# These are *stage skip* decisions, not accepted-token labels.
CASE_MASKS = (
    ("none", (False, False, False, False)),
    ("skip_second_and_fourth", (True, False, True, False)),
    ("skip_third", (False, True, False, False)),
    ("skip_fourth", (False, False, True, False)),
    ("skip_second_only", (True, False, False, False)),
    ("skip_third_after_second", (False, True, True, False)),
)


def reference(mask: tuple[bool, ...]) -> dict:
    if len(mask) != 4 or any(type(x) is not bool for x in mask):
        raise ValueError("P9A requires four binary GPU skip decisions")
    index = 0
    visited: list[int] = []
    skips = 0
    while index < len(OFFSETS):
        visited.append(OFFSETS[index])
        skip = bool(mask[index] and index < 3)
        skips += int(skip)
        index += 1 + int(skip)
    return {
        "offsets": visited,
        "rounds": len(visited),
        "stage_skips": skips,
        "final_index": index,
    }


def capture_four_real_offsets(
    model: "Qwen3TTSTalker",
    *,
    token_embeds: torch.Tensor,
    repeats: int = 16,
) -> dict:
    from torch._higher_order_ops.cond import cond_op
    from torch._higher_order_ops.while_loop import while_loop_op
    from torch._higher_order_ops.cudagraph_conditional_nodes import (
        CUDAGraphCaptureControlFlowOpDispatchMode,
    )

    if (
        token_embeds.device.type != "cuda"
        or token_embeds.shape[:2] != (1, 2)
        or token_embeds.dtype != model.predictor_k_cache.dtype
        or model.predictor_k_cache.shape[2] < 16
        or model.predictor_v_cache.shape[2] < 16
        or not 8 <= repeats <= 128
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("P9A needs real C1 CUDA 2-token, 16 KV slots and 8..128 repeats")
    dev = token_embeds.device
    cache_k, cache_v = model.predictor_k_cache, model.predictor_v_cache
    old_k = cache_k[:, :1, :16].clone()
    old_v = cache_v[:, :1, :16].clone()
    pair = token_embeds.detach().clone().contiguous()
    original_pair = pair.clone()

    offset_tensor = torch.tensor(OFFSETS, dtype=torch.long, device=dev)
    skips = torch.zeros(4, dtype=torch.bool, device=dev)
    index0 = torch.zeros((), dtype=torch.long, device=dev)
    count0 = torch.zeros((), dtype=torch.long, device=dev)
    skip0 = torch.zeros((), dtype=torch.long, device=dev)
    summed0 = torch.zeros_like(pair)

    def at2(x):
        return model.predictor_forward_tokens(
            token_embeds=x.clone(), batch_size=1, cache_len=2
        ).clone()

    def at6(x):
        return model.predictor_forward_tokens(
            token_embeds=x.clone(), batch_size=1, cache_len=6
        ).clone()

    def at10(x):
        return model.predictor_forward_tokens(
            token_embeds=x.clone(), batch_size=1, cache_len=10
        ).clone()

    def at14(x):
        return model.predictor_forward_tokens(
            token_embeds=x.clone(), batch_size=1, cache_len=14
        ).clone()

    def low(x, selected):
        return cond_op(selected < 4, at2, at6, (x,))

    def high(x, selected):
        return cond_op(selected < 12, at10, at14, (x,))

    def condition(
        index, rounds, accepted, total, decisions, offsets,
    ):
        return index < 4

    def body(
        index, rounds, accepted, total, decisions, offsets,
    ):
        off = offsets.gather(0, index.reshape(1)).reshape(())
        hidden = cond_op(
            off < 8, low, high, (pair, off)
        )
        decision = decisions.gather(0, index.reshape(1)).reshape(())
        can_skip = decision & (index < 3)
        skip = can_skip.to(torch.long)
        return index + 1 + skip, rounds + 1, accepted + skip, total + hidden

    def operation():
        return while_loop_op(
            condition,
            body,
            (index0, count0, skip0, summed0),
            (skips, offset_tensor),
        )

    def restore():
        cache_k[:, :1, :16].copy_(old_k)
        cache_v[:, :1, :16].copy_(old_v)

    def snapshot():
        return (
            cache_k[:, :1, :16].clone(),
            cache_v[:, :1, :16].clone(),
        )

    def oracle(offsets: list[int]):
        restore()
        total = torch.zeros_like(pair)
        with torch.inference_mode():
            for off in offsets:
                hid = model.predictor_forward_tokens(
                    token_embeds=pair.clone(),
                    batch_size=1,
                    cache_len=off,
                ).clone()
                total = total + hid
        k, v = snapshot()
        return total, k, v

    report = {
        "status": "P9A_NOT_VALIDATED",
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "gpu": torch.cuda.get_device_name(dev),
        "predictor_layers": len(model.code_predictor.model.layers),
        "sampled_offsets": list(OFFSETS),
        "cases": [],
        "repeats_per_cell": repeats,
    }
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    captured = False
    try:
        with torch.inference_mode():
            for fn in (at2, at6, at10, at14):
                restore()
                fn(pair)
        torch.cuda.synchronize(dev)
        restore()
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(
                graph, stream=side, capture_error_mode="thread_local"
            ),
            CUDAGraphCaptureControlFlowOpDispatchMode(),
            torch.inference_mode(),
        ):
            actual_index, actual_count, actual_skips, actual_total = operation()
        torch.cuda.synchronize(dev)
        captured = True

        for label, mask in CASE_MASKS:
            expected = reference(mask)
            restore()
            skips.copy_(torch.tensor(mask, dtype=torch.bool, device=dev))
            graph.replay()
            torch.cuda.synchronize(dev)
            actual = {
                "rounds": int(actual_count.item()),
                "stage_skips": int(actual_skips.item()),
                "final_index": int(actual_index.item()),
            }
            graph_total = actual_total.clone()
            graph_k, graph_v = snapshot()
            serial_total, serial_k, serial_v = oracle(expected["offsets"])
            checks = {
                "hidden": compare(graph_total, serial_total),
                "whole_kv_k": compare(graph_k, serial_k),
                "whole_kv_v": compare(graph_v, serial_v),
                "persistent_input_unchanged": compare(pair, original_pair),
            }
            passed = (
                actual["rounds"] == expected["rounds"]
                and actual["stage_skips"] == expected["stage_skips"]
                and actual["final_index"] == expected["final_index"]
                and all(row["close"] for row in checks.values())
            )
            report["cases"].append({
                "label": label,
                "mask": list(mask),
                "expected": expected,
                "actual": actual,
                "checks": checks,
                "pass": passed,
            })
            if not passed:
                failed = {
                    k: v["max_abs"] for k, v in checks.items() if not v["close"]
                }
                raise RuntimeError(
                    f"P9A_REAL_FOUR_OFFSET_PARITY_FAIL case={label} "
                    f"counts={actual} expected={expected} tensors={failed}"
                )

        timing = []
        for label, mask in CASE_MASKS:
            restore()
            skips.copy_(torch.tensor(mask, dtype=torch.bool, device=dev))
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize(dev)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / repeats
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError(f"P9A bad graph replay time for {label}")
            timing.append({
                "case": label,
                "model_calls": reference(mask)["rounds"],
                "cuda_event_ms_per_graph": ms,
            })
        report["cost_by_dynamic_stage_sequence"] = timing
        report["status"] = (
            "P9A_FOUR_REAL_PREDICTOR_OFFSET_GRAPH_PARITY_PASS_"
            "NOT_FULL_K2_SPEEDUP"
        )
        report["limitations"] = [
            "Four spatially separated offsets, not contiguous K2 depth transitions",
            "On-policy input embeddings frozen at depth 14, so earlier branches are off-policy",
            "Device Boolean skip mask is not a real p/q acceptance outcome",
            "Measurements contain only nested conditional real verifier stack",
            "No GPU-resident draft/target sampling, residual correction, bonus, or real serving speedup",
        ]
    finally:
        # A failed *capture* poisons its stream. After successful capture,
        # logical parity errors must still restore live K/V.
        if captured:
            torch.cuda.synchronize(dev)
            restore()
            torch.cuda.synchronize(dev)
    return report


def maybe_probe(
    model: "Qwen3TTSTalker",
    *,
    frame_index: int,
    proposal_depth: int,
    token_embeds: torch.Tensor,
) -> None:
    root = os.environ.get(ROOT_ENV)
    if not root:
        return
    if any(os.environ.get(name) for name in (
        "SGLANG_OMNI_QTTS_CP_P4_DIR",
        "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_P7_DIR",
        "SGLANG_OMNI_QTTS_CP_P8_DIR",
        "SGLANG_OMNI_QTTS_CP_P9B_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    )):
        raise RuntimeError("P9A cannot coexist with other predictor graph probes")
    if proposal_depth != 14 or getattr(model, "_qt_cp_p9a_done", False):
        return
    # At proposal depth 14, 14 positions are already committed in the
    # actual K2 cache. This is the only allowed insertion point for P9A.
    root = Path(root).expanduser().resolve()
    try:
        report = capture_four_real_offsets(
            model, token_embeds=token_embeds, repeats=24
        )
    except BaseException:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"failure_pid{os.getpid()}.txt").write_text(
            traceback.format_exc()
        )
        raise
    root.mkdir(parents=True, exist_ok=False)
    report.update({"frame": frame_index, "pid": os.getpid()})
    file = root / f"four_depth_pid{os.getpid()}_frame{frame_index:04d}.json"
    with file.open("x", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, **report}, handle, indent=2)
        handle.write("\n")
    model._qt_cp_p9a_done = True

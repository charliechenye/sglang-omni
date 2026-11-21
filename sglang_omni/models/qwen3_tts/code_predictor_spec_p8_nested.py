# SPDX-License-Identifier: Apache-2.0
"""P8 opt-in H200 nested real-model CUDA WHILE + IF conditional graph gate.

This tests a SINGLE captured graph whose GPU-driven WHILE body invokes
the actual loaded Predictor via a GPU-selected IF branch at two static KV
offsets, 2 and 6. Replay updates a device-side branch mask and loop count;
a separate eager reference runs exactly the same ordered forwards.

No model modification, no sampling or live RNG consumption, no actual K2
accept/reject correction, no whole-model speedup claim. K/V is snapshotted
and restored after success. A failed capture is fail-closed; preserve its
first exception and restart the worker before additional experiments.
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

ENV = "SGLANG_OMNI_QTTS_CP_P8_DIR"
SCHEMA = "qwen3tts-cpspec-p8-nested-real-while-if-v1"
STATES = (
    ("offset2_once", (True, False), 1),
    ("offset6_once", (False, False), 1),
    ("offset2_offset6", (True, False), 2),
    ("offset6_offset2", (False, True), 2),
    ("offset2_twice", (True, True), 2),
    ("offset6_twice", (False, False), 2),
)


def expected_offsets(choices: tuple[bool, bool], limit: int) -> list[int]:
    if len(choices) != 2 or not all(type(v) is bool for v in choices):
        raise ValueError("P8 expects two Boolean GPU branch selections")
    if type(limit) is not int or limit not in (1, 2):
        raise ValueError("P8 supports one or two device-driven iterations")
    return [2 if decision else 6 for decision in choices[:limit]]


def check_layout(model: "Qwen3TTSTalker", fixed_embeds: torch.Tensor) -> None:
    if (
        fixed_embeds.shape[:2] != (1, 2)
        or fixed_embeds.device.type != "cuda"
        or fixed_embeds.dtype != model.predictor_k_cache.dtype
        or model.predictor_k_cache.shape[2] < 8
        or model.predictor_v_cache.shape[2] < 8
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("P8 requires real CUDA C1 2-token embedding and 8-slot KV")


def capture_nested_real_predictor(
    model: "Qwen3TTSTalker",
    *,
    fixed_embeds: torch.Tensor,
    repeats: int = 24,
) -> dict:
    from torch._higher_order_ops.cond import cond_op
    from torch._higher_order_ops.while_loop import while_loop_op
    from torch._higher_order_ops.cudagraph_conditional_nodes import (
        CUDAGraphCaptureControlFlowOpDispatchMode,
    )

    check_layout(model, fixed_embeds)
    if not 8 <= repeats <= 128:
        raise ValueError("P8 repeats must be 8..128")
    dev = fixed_embeds.device
    k_cache = model.predictor_k_cache
    v_cache = model.predictor_v_cache
    old_k = k_cache[:, :1, :8].clone()
    old_v = v_cache[:, :1, :8].clone()
    pair = fixed_embeds.detach().clone().contiguous()
    original_pair = pair.clone()

    # Device-controlled inputs to the SAME graph across all test cases.
    mask = torch.tensor((True, False), device=dev, dtype=torch.bool)
    bound = torch.ones((), device=dev, dtype=torch.int64)
    initial = torch.zeros((), device=dev, dtype=torch.int64)
    total_hidden0 = torch.zeros_like(pair)

    def branch2(embeds: torch.Tensor):
        return model.predictor_forward_tokens(
            token_embeds=embeds.clone(), batch_size=1, cache_len=2
        ).clone()

    def branch6(embeds: torch.Tensor):
        return model.predictor_forward_tokens(
            token_embeds=embeds.clone(), batch_size=1, cache_len=6
        ).clone()

    def condition(
        round_idx: torch.Tensor,
        summed: torch.Tensor,
        selectors: torch.Tensor,
        limit: torch.Tensor,
    ) -> torch.Tensor:
        return round_idx < limit

    def body(
        round_idx: torch.Tensor,
        summed: torch.Tensor,
        selectors: torch.Tensor,
        limit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        choice = selectors.gather(0, round_idx.reshape(1)).reshape(())
        hidden = cond_op(choice, branch2, branch6, (pair,))
        return round_idx + 1, summed + hidden

    def operation():
        return while_loop_op(
            condition, body, (initial, total_hidden0), (mask, bound)
        )

    def restore():
        k_cache[:, :1, :8].copy_(old_k)
        v_cache[:, :1, :8].copy_(old_v)

    def snapshot():
        return (
            k_cache[:, :1, :8].clone(),
            v_cache[:, :1, :8].clone(),
        )

    def oracle(offsets: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        restore()
        acc = torch.zeros_like(pair)
        with torch.inference_mode():
            for depth in offsets:
                # BF16 fused kernels can overwrite residual inputs in-place.
                hidden = model.predictor_forward_tokens(
                    token_embeds=pair.clone(), batch_size=1, cache_len=depth
                ).clone()
                acc = acc + hidden
        k, v = snapshot()
        return acc, k, v

    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    captured_ok = False
    report = {
        "status": "P8_NESTED_REAL_MODEL_NOT_YET_PASS",
        "gpu": torch.cuda.get_device_name(dev),
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "predictor_layers": len(model.code_predictor.model.layers),
        "cases": [],
        "repeats_per_case": repeats,
    }
    try:
        # Host-driven warmup of the two static branch kernels, not a simulated
        # conditional gate. Test in a separate H200 process before serving.
        with torch.inference_mode():
            restore()
            branch2(pair)
            restore()
            branch6(pair)
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
            result_rounds, result_sum = operation()
        torch.cuda.synchronize(dev)
        captured_ok = True

        for label, choices, rounds in STATES:
            offsets = expected_offsets(choices, rounds)
            restore()
            mask.copy_(torch.tensor(choices, device=dev, dtype=torch.bool))
            bound.fill_(rounds)
            graph.replay()
            torch.cuda.synchronize(dev)
            actual_rounds = int(result_rounds.item())
            graph_sum = result_sum.clone()
            graph_k, graph_v = snapshot()
            reference_sum, reference_k, reference_v = oracle(offsets)
            checks = {
                "hidden_sum": compare(graph_sum, reference_sum),
                "k_entire_8slot": compare(graph_k, reference_k),
                "v_entire_8slot": compare(graph_v, reference_v),
                "persistent_pair_unchanged": compare(pair, original_pair),
            }
            passed = actual_rounds == rounds and all(
                check["close"] for check in checks.values()
            )
            record = {
                "label": label,
                "gpu_choices": list(choices),
                "actual_offsets": offsets,
                "expected_rounds": rounds,
                "actual_rounds": actual_rounds,
                "checks": checks,
                "pass": passed,
            }
            report["cases"].append(record)
            if not passed:
                failed = {
                    name: check["max_abs"]
                    for name, check in checks.items() if not check["close"]
                }
                raise RuntimeError(
                    f"P8_NESTED_WHILE_IF_REAL_MODEL_PARITY_FAIL "
                    f"case={label} expected_rounds={rounds} "
                    f"actual_rounds={actual_rounds} failed_checks={failed}"
                )

        # Timing is included for coarse graph-control feasibility only.
        # Replays use static input, and the old K/V is restored for every cell.
        times = {}
        for label, choices, rounds in STATES:
            restore()
            mask.copy_(torch.tensor(choices, device=dev, dtype=torch.bool))
            bound.fill_(rounds)
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize(dev)
            first = torch.cuda.Event(enable_timing=True)
            last = torch.cuda.Event(enable_timing=True)
            first.record()
            for _ in range(repeats):
                graph.replay()
            last.record()
            last.synchronize()
            val = first.elapsed_time(last) / repeats
            if not math.isfinite(val) or val <= 0:
                raise RuntimeError("invalid P8 captured graph event time")
            times[label] = val
        report["cuda_graph_event_ms_per_case"] = times
        report["status"] = (
            "P8_NESTED_WHILE_IF_REAL_5_LAYER_MODEL_PASS_"
            "NOT_SAMPLER_OR_SERVING_SPEEDUP"
        )
        report["limitations"] = [
            "Two fixed-offset branches only, both run on frozen verifier embedding",
            "Loop bound and GPU branch sequence are supplied on device, not actual sampled acceptance",
            "KV slot 2/6 are not the contiguous dynamic path of genuine K2",
            "CUDA-event graph time is a launch envelope, not pure GPU kernel busy time",
            "No p/q sampling, rejection correction, bonus encoding or end-to-end acceleration",
        ]
    finally:
        # Do not hide the original failure behind a second CUDA
        # StreamCaptureInvalidated from cleanup after failed capture.
        if captured_ok:
            torch.cuda.synchronize(dev)
            restore()
            torch.cuda.synchronize(dev)
    return report


def maybe_capture(
    model: "Qwen3TTSTalker",
    *,
    frame_index: int,
    proposal_depth: int,
    token_embeds: torch.Tensor,
) -> None:
    root = os.environ.get(ENV)
    if not root:
        return
    if (
        os.environ.get("SGLANG_OMNI_QTTS_CP_P4_DIR")
        or os.environ.get("SGLANG_OMNI_QTTS_CP_P5_DIR")
        or os.environ.get("SGLANG_OMNI_QTTS_CP_P7_DIR")
        or os.environ.get("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR")
    ):
        raise RuntimeError("P8 must be isolated from all P1/P4/P5/P7 probes")
    if proposal_depth != 6 or getattr(model, "_qt_cp_p8_done", False):
        return
    folder = Path(root).expanduser().resolve()
    try:
        evidence = capture_nested_real_predictor(
            model, fixed_embeds=token_embeds, repeats=24
        )
    except BaseException:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        error = folder / f"capture_fail_pid{os.getpid()}.txt"
        with error.open("x", encoding="utf-8") as fp:
            fp.write(traceback.format_exc())
        raise
    folder.mkdir(parents=True, exist_ok=False)
    evidence.update({"pid": os.getpid(), "frame": frame_index})
    path = folder / f"nested_pid{os.getpid()}_frame{frame_index:04d}.json"
    with path.open("x", encoding="utf-8") as fp:
        json.dump({"schema": SCHEMA, **evidence}, fp, indent=2)
        fp.write("\n")
    model._qt_cp_p8_done = True

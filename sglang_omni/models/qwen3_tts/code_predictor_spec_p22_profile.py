# SPDX-License-Identifier: Apache-2.0
"""P22 opt-in NVTX + host enqueue/wait attribution for REAL P21 serving K2.

IMPORTANT: wall_ns measures CPU time in a Python NVTX region; asynchronous
CUDA launches mean this is NOT GPU kernel execution time. Inspect Nsight
Systems' CUDA timeline for GPU times. Never enable in performance ABBA.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

PROFILE_ENV = "SGLANG_OMNI_QTTS_CP_P22_PROFILE"
CAPTURE_ARM_ENV = "SGLANG_OMNI_QTTS_CP_P22_CAPTURE_ARM"
SCHEMA = "qwen3tts-p22-real-serving-k2-nvtx-attribution-v1"
STAGES = frozenset((
    "INIT_CPU_PARAMS",
    "INIT_PAIR_FORWARD",
    "INIT_CODE1_SAMPLER",
    "ROUND",
    "ROUND_UNIFORMS",
    "Q_PROPOSALS",
    "WIDE_VERIFIER",
    "TARGET_SUPPORT_ACCEPT_BONUS",
    "ROUND_CPU_ACCEPT_LENGTH",
    "ROUND_CPU_CODE_VALIDITY",
    "ROUND_GPU_COMMIT",
    "NATIVE_FINAL_AR",
    "FULL_CODE_PREDICTOR",
))


def enabled() -> bool:
    value = os.environ.get(PROFILE_ENV, "0")
    if value not in ("0", "1"):
        raise RuntimeError("P22_PROFILE_FLAG_MUST_BE_0_OR_1")
    return value == "1"


def validate_environment() -> None:
    if not enabled():
        return
    if os.environ.get("SGLANG_OMNI_QTTS_CP_P21_SERVING_K2") != "1":
        raise RuntimeError("P22_REQUIRES_REAL_P21_SERVING_K2")
    if not os.environ.get("SGLANG_OMNI_QTTS_CP_P21_ARM_FILE"):
        raise RuntimeError("P22_REQUIRES_P21_ARM_PATH")
    for name in (
        "SGLANG_OMNI_QTTS_CP_P18_DIR",
        "SGLANG_OMNI_QTTS_CP_P19_COST_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    ):
        if os.environ.get(name):
            raise RuntimeError("P22_REFUSES_COUNTERFACTUAL_OR_LEGACY_HARNESS")
    path = os.environ.get(CAPTURE_ARM_ENV, "")
    if not path or not os.path.isabs(path):
        raise RuntimeError("P22_REQUIRES_ABSOLUTE_CAPTURE_ARM_PATH")


@contextmanager
def span(model, stage: str):
    """CPU NVTX range on actual worker; no CUDA synchronization is inserted."""
    if not enabled():
        yield
        return
    if stage not in STAGES:
        raise RuntimeError(f"P22_UNKNOWN_STAGE {stage}")
    # Import only on the explicitly instrumented disposable H200 worker.
    import torch

    begin = time.perf_counter_ns()
    torch.cuda.nvtx.range_push("P22_" + stage)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        elapsed = time.perf_counter_ns() - begin
        bucket = model.__dict__.setdefault("_p22_stage_totals", {})
        counter = bucket.setdefault(stage, {"calls": 0, "cpu_wall_ns": 0})
        counter["calls"] += 1
        counter["cpu_wall_ns"] += elapsed


def begin_capture_if_armed(model) -> bool:
    """Profiling range begins on first B frame AFTER explicit client ARM."""
    if not enabled():
        return False
    validate_environment()
    arm = os.environ[CAPTURE_ARM_ENV]
    if not os.path.isfile(arm):
        return False
    if model.__dict__.get("_p22_capture_active", False):
        return True
    if model.__dict__.get("_p22_capture_already_stopped", False):
        raise RuntimeError("P22_ONE_PROFILER_CAPTURE_PER_WORKER")
    import torch

    ret = torch.cuda.cudart().cudaProfilerStart()
    if ret not in (None, 0):
        raise RuntimeError(f"P22_CUDA_PROFILER_START_FAILED {ret!r}")
    model._p22_capture_active = True
    model._p22_capture_started_ns = time.perf_counter_ns()
    return True


def finish_request(model, *, request_hash: str, candidate_frames: int) -> dict | None:
    if not enabled() or candidate_frames <= 0:
        return None
    validate_environment()
    import torch

    capturing = bool(model.__dict__.get("_p22_capture_active", False))
    if capturing:
        # Profiler-only: include all enqueued kernels, and never contaminate
        # the unprofiled P21 performance benchmark with this fence.
        torch.cuda.synchronize(model.predictor_device)
        ret = torch.cuda.cudart().cudaProfilerStop()
        if ret not in (None, 0):
            raise RuntimeError(f"P22_CUDA_PROFILER_STOP_FAILED {ret!r}")
        model._p22_capture_active = False
        model._p22_capture_already_stopped = True
    totals = model.__dict__.pop("_p22_stage_totals", {})
    expected = {"FULL_CODE_PREDICTOR", "ROUND", "Q_PROPOSALS",
                "WIDE_VERIFIER", "ROUND_CPU_ACCEPT_LENGTH"}
    if not expected.issubset(totals):
        raise RuntimeError(
            "P22_INCOMPLETE_REAL_K2_STAGE_COVERAGE "
            + ",".join(sorted(expected - totals.keys()))
        )
    info = {
        "schema": SCHEMA,
        "request_hash": request_hash,
        "candidate_frames": candidate_frames,
        "cuda_profiler_capture": capturing,
        "timer_semantics": "CPU_WALL_NVTX_REGION_NOT_GPU_KERNEL_TIME",
        "stage_totals": totals,
    }
    # Synchronous one-line durable stdout for worker-side activity evidence.
    print("P22_REAL_REQUEST_PROFILE "
          + json.dumps(info, sort_keys=True), flush=True)
    return info

# SPDX-License-Identifier: Apache-2.0
"""P22B disposable native CodePredictor graph-only CUDA capture.

Only captures REAL native serving A, never a P21 speculative candidate.
A mandatory post-health ARM starts ONE Nsight CUDA profiler range on the
first code predictor invocation. A scheduler request-finish hook closes it.
The enclosing NVTX scope measures host CPU time, not GPU kernel elapsed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

ENABLE = "SGLANG_OMNI_QTTS_CP_P22B_NATIVE_PROFILE"
ARM = "SGLANG_OMNI_QTTS_CP_P22B_NATIVE_ARM_FILE"
DIR = "SGLANG_OMNI_QTTS_CP_P22B_NATIVE_PROFILE_DIR"
SOURCE = "SGLANG_OMNI_QTTS_CP_P22B_NATIVE_SOURCE_SHA"
SCHEMA = "qwen3tts-p22b-native-real-serving-capture-v1"


def enabled() -> bool:
    setting = os.environ.get(ENABLE, "0")
    if setting not in ("0", "1"):
        raise RuntimeError("P22B_NATIVE_FLAG_0_OR_1_REQUIRED")
    return setting == "1"


def _check_config() -> None:
    for name in (ARM, DIR):
        path = os.environ.get(name, "")
        if not path or not os.path.isabs(path):
            raise RuntimeError("P22B_NATIVE_ABSOLUTE_ARM_AND_DIR_REQUIRED")
    sha = os.environ.get(SOURCE, "")
    if len(sha) != 40:
        raise RuntimeError("P22B_NATIVE_SOURCE_SHA_MISSING")
    if os.environ.get("SGLANG_OMNI_QTTS_CP_P21_SERVING_K2") != "1":
        raise RuntimeError("P22B_NATIVE_REQUIRES_P21_ACTIVITY_COUNTERS")
    if os.environ.get("SGLANG_OMNI_QTTS_CP_P22_PROFILE") == "1":
        raise RuntimeError("P22B_NATIVE_REFUSES_K2_PROFILE")
    other_arm = os.environ.get("SGLANG_OMNI_QTTS_CP_P21_ARM_FILE", "")
    if other_arm and os.path.isfile(other_arm):
        raise RuntimeError("P22B_NATIVE_REFUSES_K2_CANDIDATE_ARM")


def start_if_armed(model, request_id: str) -> bool:
    if not enabled():
        return False
    _check_config()
    if not os.path.isfile(os.environ[ARM]):
        return False
    if model.__dict__.get("_p22b_native_stopped", False):
        raise RuntimeError("P22B_NATIVE_ONLY_ONE_CAPTURE_PER_WORKER")
    active = getattr(model, "_p22b_native_request_id", None)
    if active is not None:
        if active != request_id:
            raise RuntimeError("P22B_NATIVE_INTERLEAVED_REQUESTS_REFUSED")
        return True
    import torch

    ret = torch.cuda.cudart().cudaProfilerStart()
    if ret not in (None, 0):
        raise RuntimeError(f"P22B_NATIVE_CUDA_PROFILER_START_FAIL_{ret}")
    model._p22b_native_request_id = request_id
    model._p22b_native_host_pred_calls = 0
    model._p22b_native_host_predictor_ns = 0
    return True


@contextmanager
def predictor_span(model, request_id: str):
    active = bool(enabled() and
                  getattr(model, "_p22b_native_request_id", None) == request_id)
    if not active:
        yield
        return
    import torch

    torch.cuda.nvtx.range_push("P22B_NATIVE_CODE_PREDICTOR")
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed = time.perf_counter_ns() - started
        torch.cuda.nvtx.range_pop()
        model._p22b_native_host_pred_calls += 1
        model._p22b_native_host_predictor_ns += elapsed


def stop_if_captured(model, *, request_id: str, stats: dict) -> dict | None:
    if not enabled():
        return None
    _check_config()
    active = getattr(model, "_p22b_native_request_id", None)
    if active is None:
        return None
    if active != request_id:
        raise RuntimeError("P22B_NATIVE_REQUEST_END_ID_DRIFT")
    if (stats.get("candidate_frames", -1) != 0
            or stats.get("native_graph_replays", 0) < 2
            or stats.get("predictor_calls") != stats.get("native_graph_replays")
            or model._p22b_native_host_pred_calls != stats["predictor_calls"]):
        raise RuntimeError("P22B_NATIVE_GRAPH_AND_FRAME_COVERAGE_FAIL")
    import torch

    torch.cuda.synchronize(model.predictor_device)
    rc = torch.cuda.cudart().cudaProfilerStop()
    if rc not in (None, 0):
        raise RuntimeError(f"P22B_NATIVE_CUDA_PROFILER_STOP_FAIL_{rc}")
    model._p22b_native_stopped = True
    model._p22b_native_request_id = None
    rid_hash = hashlib.sha256(request_id.encode()).hexdigest()[:12]
    data = {
        "schema": SCHEMA,
        "source_sha": os.environ[SOURCE],
        "request_hash": rid_hash,
        "native_predictor_graph_replays": stats["native_graph_replays"],
        "actual_predictor_calls": stats["predictor_calls"],
        "candidate_frames": 0,
        "cpu_predictor_enclosure_ms": model._p22b_native_host_predictor_ns / 1e6,
        "cpu_predictor_per_frame_ms": (
            model._p22b_native_host_predictor_ns / 1e6
            / stats["predictor_calls"]
        ),
        "cuda_profiler_capture": True,
        "scope": "REAL_NATIVE_GRAPH_A_ONLY_NO_B_NO_PRODUCTION_SPEEDUP",
        "timer_semantics": "CPU_ENQUEUE_NVTX_NOT_GPU_KERNEL_TIME",
    }
    directory = Path(os.environ[DIR])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    out = directory / f"request_{rid_hash}.json"
    with out.open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"P22B_NATIVE_CAPTURE_FILE={out}", flush=True)
    return data

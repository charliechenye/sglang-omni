# SPDX-License-Identifier: Apache-2.0
"""P19 CUDA700 single-arm diagnostics: durable CPU-only phase breadcrumbs.

Events deliberately NEVER read GPU tensors or synchronize CUDA. A sticky
CUDA700 can kill the worker before Python raises a catchable exception.
A successful request is diagnostic ONLY, not P19A correctness/cost proof.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SCHEMA = "qwen3tts-p19-cuda700-phase-trace-v1"
ARM_ENV = "SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM"
DEPTH_ENV = "SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH"
DIR_ENV = "SGLANG_OMNI_QTTS_CP_P19_COST_DIR"
MODES = ("OFF", "AR_EAGER", "AR_GRAPH", "K2_EAGER", "K2_GRAPH",
         "K3_EAGER", "K3_GRAPH", "K4_EAGER", "K4_GRAPH",
         "ALL_GRAPH")


def diagnostic_config() -> tuple[str, int] | None:
    arm = os.environ.get(ARM_ENV)
    if arm is None:
        return None
    if arm not in MODES:
        raise RuntimeError(f"P19_UNKNOWN_DIAGNOSTIC_ARM {arm!r}")
    raw_depth = os.environ.get(DEPTH_ENV, "")
    if not raw_depth.isdecimal() or not 1 <= int(raw_depth) <= 14:
        raise RuntimeError("P19_DIAGNOSTIC_DEPTH_MUST_BE_1_TO_14")
    depth = int(raw_depth)
    if (arm.startswith("K4") and depth > 11
            or arm.startswith("K3") and depth > 12
            or arm.startswith("K2") and depth > 13):
        raise RuntimeError("P19_DIAGNOSTIC_ILLEGAL_K_AT_DEPTH")
    return arm, depth


def event(stage: str, *, depth: int | None = None, arm: str | None = None,
          frame: int | None = None, case_id: str | None = None) -> None:
    """Append+fsync one CPU-only record, even if next CUDA call aborts."""
    if diagnostic_config() is None:
        return
    root_name = os.environ.get(DIR_ENV)
    if not root_name:
        raise RuntimeError("P19_DIAGNOSTIC_COST_ROOT_MISSING")
    root = Path(root_name)
    if not root.is_absolute():
        raise RuntimeError("P19_DIAGNOSTIC_COST_ROOT_NOT_ABSOLUTE")
    directory = root / "diagnostic"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"worker_pid{os.getpid()}.jsonl"
    record = {
        "schema": SCHEMA, "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        "stage": stage, "depth": depth, "arm": arm,
        "frame": frame, "case_id": case_id,
    }
    blob = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        if os.write(fd, blob) != len(blob):
            raise RuntimeError("P19_DIAGNOSTIC_SHORT_APPEND")
        os.fsync(fd)
    finally:
        os.close(fd)

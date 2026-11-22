# SPDX-License-Identifier: Apache-2.0
"""P17 opt-in bounded worker-side dispatch/ARM observability.

Executed only when SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR is set. One
JSONL per PID. No GPU synchronization or tensor content access. This is
diagnostic evidence, NOT a substitute for native K2/probe activity.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

MAX_PER_STATE = 3
_COUNTS: dict[tuple[str, bool, bool], int] = {}


def record(event: str, *, caller_module: str) -> None:
    dest = os.environ.get("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR")
    if not dest:
        return
    root = Path(dest)
    if not root.is_absolute():
        raise RuntimeError("P17_ACTIVATION_DIR_MUST_BE_ABSOLUTE")
    spec_path = os.environ.get("SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE", "")
    base_path = os.environ.get("SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE", "")
    # Observe on EVERY invocation, including the transition when Terminal B
    # creates/removes ARM files between HTTP requests.
    spec_exists = bool(spec_path and os.path.isfile(spec_path))
    base_exists = bool(base_path and os.path.isfile(base_path))
    key = (event, spec_exists, base_exists)
    count = _COUNTS.get(key, 0)
    if count >= MAX_PER_STATE:
        return
    _COUNTS[key] = count + 1
    record_data = {
        "schema": "qwen3tts-p17-activation-v1",
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "caller_module": caller_module,
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "spec_k": os.environ.get("SGLANG_OMNI_QTTS_CP_SPEC_K"),
        "spec_arm_path": spec_path,
        "spec_arm_exists": spec_exists,
        "base_arm_path": base_path,
        "base_arm_exists": base_exists,
        "probe_dir": os.environ.get("SGLANG_OMNI_QTTS_CP_P17_MULTIWIDTH_DIR"),
        "activation_only": (
            os.environ.get("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_ONLY") == "1"
        ),
    }
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = root / f"dispatch_pid{os.getpid()}.jsonl"
    # Single short write per record per worker, append-only, no overwrite.
    fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        payload = (json.dumps(record_data, sort_keys=True) + "\n").encode()
        if os.write(fd, payload) != len(payload):
            raise RuntimeError("P17_ACTIVATION_SHORT_WRITE")
    finally:
        os.close(fd)

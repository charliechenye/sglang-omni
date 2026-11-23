# SPDX-License-Identifier: Apache-2.0
"""P22B Native A Graph-ON Nsight API + trace-wide GPU-work context.

Graph node GPU kernel work is NOT identified as exclusively belonging
to CodePredictor using a CPU NVTX range. Only CPU CUDA API launch calls
are directly attributed to P22B_NATIVE_CODE_PREDICTOR. All captured
GPU work includes other Serving stages and must not be compared with
P22 B CodePredictor-only GPU kernel work.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .p22b_cuda_api_nvtx import analyze_sqlite
from .p22_nsys_sqlite import KERNEL_TABLE

NATIVE_STAGE = "P22B_NATIVE_CODE_PREDICTOR"
STATUS = "P22B_NATIVE_GRAPH_LAUNCH_API_CAPTURED_GPU_WORK_NOT_CODEPREDICTOR_ISOLATED"


def analyze(root: Path, sqlite_file: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    sqlite_file = sqlite_file.expanduser().resolve(strict=True)
    if sqlite_file.parent != root:
        raise RuntimeError("P22B_NATIVE_SQLITE_MUST_BE_WITHIN_ROOT")
    cfg = json.loads((root / "run.json").read_text())
    client = json.loads((root / "p22b_native_client.json").read_text())
    sha = (root / "source.sha").read_text().strip()
    if (len(sha) != 40 or cfg.get("source_sha") != sha
            or client.get("source_sha") != sha
            or cfg.get("schema") != "qwen3tts-p22b-native-a-profile-v1"
            or client.get("status") !=
               "P22B_NATIVE_GRAPH_ON_REAL_REQUEST_CAPTURED_NOT_E2E_ABBA"):
        raise RuntimeError("P22B_NATIVE_MISSING_SOURCE_OR_REQUEST_PROOF")

    api = analyze_sqlite(
        sqlite_file, required_stages=(NATIVE_STAGE,)
    )
    actual = client["native_graph_cpu_enqueue"]
    frames = actual["native_predictor_graph_replays"]
    if (frames < 2 or frames != client["measured_worker"]["predictor_calls"]
            or client["measured_worker"]["candidate_frames"] != 0):
        raise RuntimeError("P22B_NATIVE_GRAPH_PROOF_MISSING")
    stage = api["stages"].get(NATIVE_STAGE, {})
    graphs = stage.get("graph_launch", {"calls": 0, "summed_cpu_api_ms": 0.0})
    if graphs["calls"] < 1:
        raise RuntimeError("P22B_NATIVE_NO_GRAPH_LAUNCH_WITHIN_NVTX")
    total = stage.get("kernel_launch", {"calls": 0, "summed_cpu_api_ms": 0.0})
    db = sqlite3.connect(sqlite_file.as_uri() + "?mode=ro", uri=True)
    try:
        count, gpu_ns = db.execute(
            f"SELECT COUNT(*), SUM(end-start) FROM {KERNEL_TABLE} "
            "WHERE end >= start"
        ).fetchone()
    finally:
        db.close()
    if count < 1 or gpu_ns is None:
        raise RuntimeError("P22B_NATIVE_EMPTY_GPU_TRACE")
    api.update({
        "status": STATUS,
        "source_sha": sha,
        "native_graph_frames": frames,
        "native_graph_cpu_predictor_per_frame_ms":
            actual["cpu_predictor_per_frame_ms"],
        "native_graph_launch_api_calls": graphs["calls"],
        "native_graph_launch_api_cpu_total_ms": graphs["summed_cpu_api_ms"],
        "native_graph_launch_api_cpu_per_frame_ms":
            graphs["summed_cpu_api_ms"] / frames,
        "other_direct_kernel_launch_api_calls_within_native_stage":
            total["calls"],
        "all_captured_gpu_kernels_any_serving_stage": count,
        "all_captured_gpu_kernel_work_ms_any_serving_stage": gpu_ns / 1e6,
        "GPU_STAGE_SCOPE": "WHOLE_CAPTURE_TALKER_AND_CODEPREDICTOR_NOT_ISOLATED",
        "note": (
            "CPU NVTX proves native CodePredictor graph launches per request. "
            "GPU graph child kernels cannot be safely assigned to the CP CPU "
            "range by timestamp; full captured GPU work includes Talker."
        ),
    })
    return api


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sqlite", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError("P22B_NATIVE_REFUSES_REPORT_OVERWRITE")
    report = analyze(a.root, a.sqlite)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print("P22B_NATIVE_API_STATUS=" + report["status"])
    print("P22B_NATIVE_CODEPREDICTOR_FRAMES="
          + str(report["native_graph_frames"]))
    print("P22B_NATIVE_GRAPH_LAUNCH_CPU_MS_PER_FRAME="
          + f"{report['native_graph_launch_api_cpu_per_frame_ms']:.4f}")
    print("P22B_NATIVE_JSON=" + str(out))


if __name__ == "__main__":
    main()

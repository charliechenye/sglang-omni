# SPDX-License-Identifier: Apache-2.0
"""P22B single native Graph-ON real HTTP request, matched to P22 B input.

Uses exactly one native warmup and one captured native request. Same model,
seed, text, H200 and source on THIS native run; different source/run vs P22
K2, so cannot claim a same-SHA ABBA or a NVIDIA-vs-Omni speedup.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path
from urllib.request import urlopen

from .p18_run import PROMPTS
from .p21_serving_abba import _http_with_proof
from sglang_omni.models.qwen3_tts.code_predictor_spec_p22b_native import (
    SCHEMA,
)

SUCCESS = "P22B_NATIVE_GRAPH_ON_REAL_REQUEST_CAPTURED_NOT_E2E_ABBA"


def worker_files(directory: Path) -> list[dict]:
    rows = []
    for p in sorted(directory.glob("request_*.json")):
        data = json.loads(p.read_text())
        if (data.get("schema") != SCHEMA or
                p.stem != "request_" + str(data.get("request_hash"))):
            raise RuntimeError("P22B_NATIVE_WORKER_EVIDENCE_DRIFT")
        rows.append(data)
    return rows


def validate_native_evidence(row: dict, stats: dict, sha: str) -> None:
    if (row.get("source_sha") != sha
            or row.get("cuda_profiler_capture") is not True
            or row.get("candidate_frames") != 0
            or row.get("native_predictor_graph_replays") !=
               stats["native_graph_replays"]
            or row.get("actual_predictor_calls") !=
               stats["predictor_calls"]
            or row.get("native_predictor_graph_replays", 0) < 2
            or type(row.get("cpu_predictor_per_frame_ms")) is not float
            or row["cpu_predictor_per_frame_ms"] <= 0):
        raise RuntimeError("P22B_NATIVE_REAL_GRAPH_COVERAGE_FAILED")


def run(root: Path, *, base_url: str, max_tokens: int) -> dict:
    root = root.expanduser().resolve(strict=True)
    sha = (root / "source.sha").read_text().strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    cfg = json.loads((root / "run.json").read_text())
    if (len(sha) != 40 or sha != current
            or cfg.get("schema") != "qwen3tts-p22b-native-a-profile-v1"
            or cfg.get("source_sha") != sha
            or cfg.get("k2_speculative") != "DISABLED"
            or cfg.get("native_code_predictor_graph") !=
               "REQUIRED_ALL_SAMPLE_FRAMES"):
        raise RuntimeError("P22B_NATIVE_SOURCE_OR_MODE_MISMATCH")
    if max_tokens < 512:
        raise RuntimeError("P22B_NATIVE_REQUIRES_NONTRUNCATED_OUTPUT")
    arm = root / "P22B_NATIVE_ARM"
    k2_arm = root / "P21_ARM_STAYS_ABSENT"
    log = root / "server.log"
    report = root / "p22b_native_client.json"
    directory = root / "native_profiles"
    if (arm.exists() or k2_arm.exists()
            or report.exists() or worker_files(directory)
            or not log.is_file()):
        raise RuntimeError("P22B_NATIVE_REQUIRES_FRESH_WORKER_AND_ARM")
    with urlopen(base_url.rstrip("/") + "/health", timeout=30) as res:
        if res.status != 200:
            raise RuntimeError("P22B_NATIVE_SERVER_UNHEALTHY")
    name, prompt = PROMPTS[0]
    case = {"id": name, "text": prompt, "seed": 20261123}
    warmup = {
        "id": "_native_warmup", "text": "A short real speech warmup sentence.",
        "seed": 20261124,
    }
    outcome = {
        "schema": "qwen3tts-p22b-native-client-v1",
        "status": "P22B_NATIVE_INCOMPLETE_NO_GPU_OR_SPEEDUP_CLAIM",
        "source_sha": sha,
        "case": case, "nsys_capture_range": cfg["nsys_capture_range"],
    }
    try:
        a = _http_with_proof(
            base_url=base_url, server_log=log, phase="A",
            case=warmup, max_tokens=max_tokens,
        )
        if worker_files(directory):
            raise RuntimeError("P22B_NATIVE_EARLY_PROFILER_CAPTURE")
        print("P22B_NATIVE_GRAPH_WARMUP_PASS", flush=True)
        arm.touch(exist_ok=False)
        observed = _http_with_proof(
            base_url=base_url, server_log=log, phase="A",
            case=case, max_tokens=max_tokens,
        )
        evidence = []
        for _ in range(50):
            evidence = worker_files(directory)
            if len(evidence) >= 1:
                break
            time.sleep(0.1)
        if len(evidence) != 1:
            raise RuntimeError("P22B_NATIVE_EXPECTED_ONE_WORKER_RECORD")
        validate_native_evidence(evidence[0], observed["worker"], sha)
        outcome.update({
            "status": SUCCESS,
            "warmup_worker": a["worker"],
            "measured_worker": observed["worker"],
            "measured_wav_sha256": observed["wav_sha256"],
            "measured_finish_reason": observed["finish_reason"],
            "measured_audio_duration_s": observed["audio_duration_s"],
            "native_graph_cpu_enqueue": evidence[0],
            "scope": "ONE_NATIVE_A_SOURCE_PINNED_NSYS_PROFILER_NOT_ABBA",
        })
        print("P22B_NATIVE_REAL_FRAMES="
              + str(observed["worker"]["native_graph_replays"]), flush=True)
        print("P22B_NATIVE_CAPTURE_FINISHED=True", flush=True)
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        outcome["traceback"] = traceback.format_exc()
        raise
    finally:
        if arm.exists():
            arm.unlink()
        with report.open("x") as fh:
            json.dump(outcome, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("P22B_NATIVE_STATUS=" + outcome["status"], flush=True)
        print("P22B_NATIVE_REPORT=" + str(report), flush=True)
    return outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-new-tokens", default=768, type=int)
    x = ap.parse_args()
    run(x.root, base_url=x.base_url, max_tokens=x.max_new_tokens)


if __name__ == "__main__":
    main()

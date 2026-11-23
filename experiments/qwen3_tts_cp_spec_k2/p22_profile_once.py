# SPDX-License-Identifier: Apache-2.0
"""P22: one real native-Graph A warmup, B warmup, then one profiled B request.

No performance comparison or QPS claim: the CUDA profiler contaminates
wall-clock numbers. Nsight CUDA timeline + NVTX are the GPU evidence;
per-region perf_counter_ns captures CPU enqueue/blocking work only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from urllib.request import urlopen

from .p18_run import PROMPTS
from .p21_serving_abba import _http_with_proof
from sglang_omni.models.qwen3_tts.code_predictor_spec_p22_profile import (
    SCHEMA as PROFILE_SCHEMA, STAGES,
)

STATUS = "P22_ONE_REAL_SERVING_B_REQUEST_PROFILE_CAPTURED_NOT_SPEEDUP"
STATUS_NO_NSYS = "P22_ONE_REAL_SERVING_B_CPU_ATTRIBUTION_NOT_GPU_TIME"
STATUS_FAIL = "P22_PROFILE_FAIL_CLOSED_NO_PERFORMANCE_CLAIM"
def worker_profiles(directory: Path) -> list[dict]:
    """Read real worker JSON files, NOT potentially wrapped server logs."""
    found = []
    for file in directory.glob("request_*.json"):
        item = json.loads(file.read_text())
        if (item.get("schema") != PROFILE_SCHEMA
                or file.stem != "request_" + str(item.get("request_hash"))):
            raise RuntimeError("P22_WORKER_PROFILE_SCHEMA_OR_FILENAME_MISMATCH")
        if type(item.get("monotonic_ns")) is not int:
            raise RuntimeError("P22_WORKER_PROFILE_TIMESTAMP_MISSING")
        found.append(item)
    found.sort(key=lambda e: e["monotonic_ns"])
    if len({e["monotonic_ns"] for e in found}) != len(found):
        raise RuntimeError("P22_DUPLICATE_WORKER_PROFILE_TIMESTAMP")
    return found


def validate_profile(profile: dict, *, expected_capture: bool,
                     worker: dict) -> dict:
    if profile.get("cuda_profiler_capture") is not expected_capture:
        raise RuntimeError("P22_PROFILE_CUDA_CAPTURE_STATE_MISMATCH")
    if profile.get("candidate_frames") != worker["candidate_frames"]:
        raise RuntimeError("P22_PROFILE_REAL_FRAME_COUNT_DRIFT")
    stages = profile.get("stage_totals")
    if not isinstance(stages, dict):
        raise RuntimeError("P22_PROFILE_STAGE_TOTALS_MISSING")
    expected = {
        "FULL_CODE_PREDICTOR": worker["candidate_frames"],
        "INIT_CPU_PARAMS": worker["candidate_frames"],
        "INIT_PAIR_FORWARD": worker["candidate_frames"],
        "INIT_CODE1_SAMPLER": worker["candidate_frames"],
        "ROUND": worker["k2_rounds"],
        "ROUND_UNIFORMS": worker["k2_rounds"],
        "Q_PROPOSALS": worker["k2_rounds"],
        "WIDE_VERIFIER": worker["k2_rounds"],
        "TARGET_SUPPORT_ACCEPT_BONUS": worker["k2_rounds"],
        "ROUND_CPU_ACCEPT_LENGTH": worker["k2_rounds"],
        "ROUND_CPU_CODE_VALIDITY": worker["k2_rounds"],
        "ROUND_GPU_COMMIT": worker["k2_rounds"],
    }
    for stage, wanted in expected.items():
        item = stages.get(stage)
        if (not isinstance(item, dict)
                or item.get("calls") != wanted
                or type(item.get("cpu_wall_ns")) is not int
                or item["cpu_wall_ns"] <= 0):
            raise RuntimeError(
                f"P22_STAGE_CALL_COVERAGE_MISMATCH {stage} "
                f"expected={wanted} actual={item}"
            )
    if not set(stages).issubset(STAGES):
        raise RuntimeError("P22_UNKNOWN_WORKER_PROFILE_STAGE")
    frames = worker["candidate_frames"]
    rounds = worker["k2_rounds"]
    output = {}
    for name, item in sorted(stages.items()):
        total_ms = item["cpu_wall_ns"] / 1_000_000
        output[name] = {
            "calls": item["calls"],
            "cpu_wall_total_ms": total_ms,
            "cpu_wall_per_call_ms": total_ms / item["calls"],
            "cpu_wall_per_frame_ms": total_ms / frames,
            "cpu_wall_per_k2_round_ms": total_ms / rounds,
        }
    return output


def run(root: Path, *, base_url: str, max_new_tokens: int) -> dict:
    root = root.expanduser().resolve(strict=True)
    source = (root / "source.sha").read_text().strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    cfg = json.loads((root / "run.json").read_text())
    if (source != actual or len(source) != 40
            or cfg.get("source_sha") != source
            or cfg.get("schema") != "qwen3tts-p22-serving-profile-v1"
            or cfg.get("P18_P19") != "DISABLED"
            or cfg.get("profiling_enabled") is not True):
        raise RuntimeError("P22_WORKER_CLIENT_SOURCE_OR_CONFIG_MISMATCH")
    if max_new_tokens < 512:
        raise RuntimeError("P22_REQUIRES_UNTRUNCATED_SPEECH_MAX_TOKENS")
    p21_arm = root / "P21_ARM"
    capture_arm = root / "P22_CAPTURE_ARM"
    report_path = root / "p22_client.json"
    if (p21_arm.exists() or capture_arm.exists()
            or report_path.exists() or not (root / "server.log").exists()):
        raise RuntimeError("P22_REQUIRES_FRESH_WORKER_ARM_AND_REPORT")
    with urlopen(base_url.rstrip("/") + "/health", timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError("P22_SERVER_HEALTH_FAILED")

    want_nsys = cfg["nsys_capture_range"]
    if type(want_nsys) is not bool:
        raise RuntimeError("P22_NSYS_CAPTURE_CONFIG_MUST_BE_BOOL")
    name, prompt = PROMPTS[0]
    case = {"id": name, "text": prompt, "seed": 20261123}
    warm = {
        "id": "p22_warmup", "text": "A short real speech warmup sentence.",
        "seed": 20261124,
    }
    log = root / "server.log"
    profiles_dir = root / "p22_profiles"
    if profiles_dir.exists() and list(profiles_dir.glob("request_*.json")):
        raise RuntimeError("P22_PROFILE_REQUIRES_ABSENT_WORKER_EVIDENCE")
    result = {
        "schema": "qwen3tts-p22-one-serving-request-profile-v1",
        "status": STATUS_FAIL, "source_sha": source,
        "case": case, "nsys_capture_range": want_nsys,
        "no_perf_or_quality_claim": True, "visits": [],
    }
    started_profiles = len(worker_profiles(profiles_dir))
    try:
        native = _http_with_proof(
            base_url=base_url, server_log=log, phase="A",
            case=warm, max_tokens=max_new_tokens,
        )
        result["visits"].append({"name": "native_graph_warmup",
                                 "worker": native["worker"]})
        print("P22_NATIVE_GRAPH_WARMUP_PASS", flush=True)

        p21_arm.touch(exist_ok=False)
        b_warm = _http_with_proof(
            base_url=base_url, server_log=log, phase="B",
            case=warm, max_tokens=max_new_tokens,
        )
        result["visits"].append({"name": "B_unprofiled_kernel_warmup",
                                 "worker": b_warm["worker"]})
        warms = worker_profiles(profiles_dir)
        if len(warms) != started_profiles + 1:
            raise RuntimeError("P22_WARMUP_PROFILE_COUNT_DRIFT")
        validate_profile(
            warms[-1], expected_capture=False, worker=b_warm["worker"]
        )
        print("P22_K2_WARMUP_PASS", flush=True)

        # The only profiled request: capture begins at first real K2
        # frame, ends after the request's real scheduler termination.
        if want_nsys:
            capture_arm.touch(exist_ok=False)
        measured = _http_with_proof(
            base_url=base_url, server_log=log, phase="B",
            case=case, max_tokens=max_new_tokens,
        )
        result["visits"].append({"name": "B_measured_real_request",
                                 "worker": measured["worker"],
                                 "finish_reason": measured["finish_reason"],
                                 "wav_sha256": measured["wav_sha256"],
                                 "audio_duration_s": measured["audio_duration_s"]})
        profiles = worker_profiles(profiles_dir)
        if len(profiles) != started_profiles + 2:
            raise RuntimeError("P22_MEASURED_PROFILE_COUNT_DRIFT")
        report_stages = validate_profile(
            profiles[-1], expected_capture=want_nsys,
            worker=measured["worker"],
        )
        result["stage_cpu_totals"] = report_stages
        result["profiled_candidate_frames"] = measured["worker"]["candidate_frames"]
        result["profiled_k2_rounds"] = measured["worker"]["k2_rounds"]
        result["capture_completed_by_cuda_api"] = bool(want_nsys)
        result["status"] = STATUS if want_nsys else STATUS_NO_NSYS
        print("P22_PROFILE_REAL_K2_FRAMES="
              + str(result["profiled_candidate_frames"]), flush=True)
        print("P22_PROFILE_REAL_K2_ROUNDS="
              + str(result["profiled_k2_rounds"]), flush=True)
        print("P22_PROFILE_CAPTURE_FINISHED=" + str(want_nsys), flush=True)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if capture_arm.exists():
            capture_arm.unlink()
        if p21_arm.exists():
            p21_arm.unlink()
        with report_path.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        print("P22_PROFILE_STATUS=" + result["status"], flush=True)
        print("P22_PROFILE_REPORT=" + str(report_path), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()
    report = run(args.root, base_url=args.base_url,
                 max_new_tokens=args.max_new_tokens)
    for stage, item in sorted(report["stage_cpu_totals"].items()):
        print("P22_HOST_STAGE "
              f"{stage} calls={item['calls']} "
              f"per_frame_ms={item['cpu_wall_per_frame_ms']:.4f} "
              f"per_call_ms={item['cpu_wall_per_call_ms']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Trigger P5 Graph ON baseline and real-state K2 captured-graph probes.

Runs two sequential complete C1 speech requests on one loaded model:
existing Graph ON baseline first, K2 with isolated verifier-graph captures
second. These HTTP times are NOT a performance comparison (P5 intentionally
profiles/warms captured subgraphs inside inference).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

PHASES = ("base", "spec")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--baseline-arm-file", required=True, type=Path)
    p.add_argument("--spec-arm-file", required=True, type=Path)
    p.add_argument("--server-log", required=True, type=Path)
    p.add_argument("--profiles-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()
    arm_base = args.baseline_arm_file.expanduser().resolve()
    arm_spec = args.spec_arm_file.expanduser().resolve()
    server_log = args.server_log.expanduser().resolve()
    profiles = args.profiles_dir.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if arm_base == arm_spec:
        p.error("base/spec ARM files must be separate")
    if output.exists() or profiles.exists():
        p.error("output and profile directory must not preexist")
    if arm_base.exists() or arm_spec.exists():
        p.error("remove stale ARM markers before starting")
    if not server_log.is_file() or not arm_base.parent.is_dir() or not arm_spec.parent.is_dir():
        p.error("server log and both ARM parent directories must exist")
    if not 256 <= args.max_new_tokens <= 2048:
        p.error("max_new_tokens 256..2048")
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as res:
        if res.status != 200:
            raise RuntimeError("server unhealthy")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": MODEL,
        "input": TEXT,
        "voice": "Vivian",
        "response_format": "wav",
        "seed": 20261120,
        "max_new_tokens": args.max_new_tokens,
        "stage_params": {"tts_engine": {
            "subtalker_dosample": True,
            "subtalker_temperature": 0.9,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
        }},
    }
    data = {
        "status": "P5_GRAPH_COST_INCOMPLETE_NO_SPEEDUP_CLAIM",
        "request": payload,
        "visits": [],
        "limitations": [
            "Instrumented HTTP timings are deliberately incomparable to live production",
            "Both modes use a shared loaded model, not fresh-server performance ABBA",
            "Graph ON baseline profiler must observe real full Predictor graph replay",
            "Candidate profiler captures separate fixed-input stack graphs, not a full K2 graph",
        ],
    }
    (output / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    try:
        for phase in PHASES:
            arm = arm_base if phase == "base" else arm_spec
            prior = len(parse_k2_log_entries(server_log.read_text(errors="replace")))
            print(f"P5_GRAPH_BEGIN={phase}", flush=True)
            try:
                arm.touch(exist_ok=False)
                audio, finish, elapsed = _http_call(args.base_url, payload)
            finally:
                arm.unlink(missing_ok=True)
            after = len(parse_k2_log_entries(server_log.read_text(errors="replace")))
            k2_frames = after - prior
            if finish != "stop" or (phase == "base" and k2_frames != 0) or (
                phase == "spec" and k2_frames <= 0
            ):
                raise RuntimeError(
                    f"P5 {phase} not naturally stopped or K2 was misrouted: "
                    f"finish={finish!r} k2_frames={k2_frames}"
                )
            (output / f"{phase}.wav").write_bytes(audio)
            data["visits"].append({
                "phase": phase,
                "finish_reason": finish,
                "k2_frames": k2_frames,
                "wav_sha256": hashlib.sha256(audio).hexdigest(),
                "http_seconds_instrumented_not_perf": elapsed,
                **wav_info(audio),
            })
            (output / "report.json").write_text(json.dumps(data, indent=2) + "\n")
            print(
                f"P5_GRAPH_DONE={phase} stop audio_s={wav_info(audio)['duration_s']:.2f} "
                f"k2_frames={k2_frames}", flush=True,
            )
        base_files = list(profiles.glob("baseline_graph_pid*.json"))
        packed_files = list(profiles.glob("packed_graph_pid*_frame*_depth*.json"))
        if len(base_files) != 1 or len(packed_files) < 4:
            raise RuntimeError(
                f"missing P5 graph evidence: baseline={len(base_files)}, "
                f"packed={len(packed_files)}; see server log"
            )
        data["status"] = "P5_GRAPH_BASE_AND_FIXED_VERIFIER_CAPTURED_NOT_SERVING_SPEEDUP"
    except Exception as exc:
        data["error"] = str(exc)
        (output / "report.json").write_text(json.dumps(data, indent=2) + "\n")
        raise
    (output / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"{data['status']}: {output / 'report.json'}")


if __name__ == "__main__":
    main()

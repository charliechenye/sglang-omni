# SPDX-License-Identifier: Apache-2.0
"""P9 true H200 checkpoint A/B component evidence collector.

A single healthy Omni server, same model/weights/GPU/source, first measures
the EXISTING complete optimized sampled Predictor Graph-ON replay with
a dedicated BASE_ARM. It then enters the real K2 path under SPEC_ARM
and captures EITHER P9A four-offset model routing OR P9B GPU sample/accept
from the actual request. It verifies natural EOS and candidate-only activity.

HTTP times are DEBUG INSTRUMENTED and must NEVER be compared as QPS.
This script does not activate any GPU-resident speculative decoding output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

PHASES = ("base", "spec", "spec", "base")
STATUS = "P9_BASELINE_GRAPH_ON_AND_REAL_K2_COMPONENT_EVIDENCE_CAPTURED_NOT_SPEEDUP"


def run(args: argparse.Namespace) -> dict:
    root = args.out.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"output exists: {root}")
    arms = {
        "base": args.baseline_arm_file.expanduser().resolve(),
        "spec": args.spec_arm_file.expanduser().resolve(),
    }
    if len(set(arms.values())) != 2 or any(x.exists() for x in arms.values()):
        raise RuntimeError("P9 BASE/SPEC ARM must be distinct and absent before start")
    if any(not x.parent.is_dir() for x in arms.values()):
        raise FileNotFoundError("P9 ARM parents must exist")
    log = args.server_log.expanduser().resolve(strict=True)
    baseline_dir = args.baseline_graph_dir.expanduser().resolve()
    probe_dir = args.probe_dir.expanduser().resolve()
    if baseline_dir.exists() or probe_dir.exists():
        raise FileExistsError("P9 baseline and probe dirs must be fresh")
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("Omni P9 server is not healthy")

    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": "wav",
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "stage_params": {
            "tts_engine": {
                "subtalker_dosample": True,
                "subtalker_temperature": args.temperature,
                "subtalker_top_k": args.top_k,
                "subtalker_top_p": args.top_p,
            }
        },
    }
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "P9_INSTRUMENTED_ABBA_INCOMPLETE",
        "mode": args.mode,
        "visits": [],
        "request": payload,
        "limitations": [
            "Shared-server ARM routing and EOS only; debug time is NOT a speedup",
            "P9 probes operate on real model states but do not replace CPU K2 sampling/output",
            "Baseline graph event and sampled component graphs are different code paths",
            "Final whole-K2 Graph-ON profiler-OFF performance ABBA remains unimplemented",
        ],
    }
    (root / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    try:
        for idx, phase in enumerate(PHASES):
            tag = f"{idx:02d}_{phase}"
            arm = arms[phase]
            prior = len(parse_k2_log_entries(log.read_text(errors="replace")))
            print(f"P9_{args.mode.upper()}_BEGIN={tag}", flush=True)
            try:
                arm.touch(exist_ok=False)
                wav, finish, elapsed = _http_call(args.base_url, payload)
            finally:
                arm.unlink(missing_ok=True)
            after = len(parse_k2_log_entries(log.read_text(errors="replace")))
            k2_frames = after - prior
            if finish != "stop" or (phase == "base" and k2_frames != 0) or (
                phase == "spec" and k2_frames == 0
            ):
                raise RuntimeError(
                    f"P9 C1 natural stop or routing failed: {tag}, "
                    f"finish={finish}, k2_frames={k2_frames}"
                )
            (root / f"{tag}.wav").write_bytes(wav)
            record = {
                "phase": phase,
                "tag": tag,
                "finish": finish,
                "k2_frames": k2_frames,
                "sha256": hashlib.sha256(wav).hexdigest(),
                "debug_http_s_not_perf": elapsed,
                **wav_info(wav),
            }
            report["visits"].append(record)
            (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"P9_{args.mode.upper()}_DONE={tag} finish=stop k2_frames={k2_frames}", flush=True)
        bases = list(baseline_dir.glob("baseline_graph_pid*.json"))
        if args.mode == "a":
            candidates = list(probe_dir.glob("four_depth_pid*_frame*.json"))
            required = 1
        else:
            candidates = list(probe_dir.glob("sample_pid*_frame*_depth*.json"))
            required = 2
        if len(bases) != 1 or len(candidates) != required:
            raise RuntimeError(
                f"P9 evidence incomplete: baseline={len(bases)} "
                f"P9{args.mode.upper()}={len(candidates)} expected={required}"
            )
        source = json.loads(bases[0].read_text())
        if source.get("graph_replays") != source.get("frames") or source.get("frames", 0) < 8:
            raise RuntimeError("P9 did not capture real full Predictor Graph ON baseline")
        for file in candidates:
            row = json.loads(file.read_text())
            if row.get("source_sha") != source.get("source_sha") or row.get("pid") != source.get("pid"):
                raise RuntimeError("P9 baseline and candidate trace are not SAME SHA/PID")
        report["source_sha"] = source.get("source_sha")
        report["status"] = STATUS
    except Exception as exc:
        report["status"] = "P9_FAILED_NO_ACCELERATION_SELECTION"
        report["error"] = str(exc)
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{STATUS} report={root / 'report.json'}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("a", "b"), required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--baseline-arm-file", type=Path, required=True)
    p.add_argument("--spec-arm-file", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--baseline-graph-dir", type=Path, required=True)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--text", default=TEXT)
    p.add_argument("--voice", default="Vivian")
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.)
    args = p.parse_args()
    if not 256 <= args.max_new_tokens <= 2048:
        p.error("natural EOS check requires max_new_tokens 256..2048")
    if not (0 < args.temperature <= 5 and 0 < args.top_p <= 1 and args.top_k >= 1):
        p.error("invalid sampled configuration")
    run(args)


if __name__ == "__main__":
    main()

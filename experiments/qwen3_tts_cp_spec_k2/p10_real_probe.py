# SPDX-License-Identifier: Apache-2.0
"""P10 same-checkpoint, same-SHA real H200 dynamic-KV / sparse sampler probe.

Capture the ACTUAL full optimized Predictor Graph-ON baseline and P10A or
P10B from the SAME running C1 Omni worker (fresh worker per arm).
Natural EOS and K2 activity, source SHA, PID and GPU match are mandatory.

Four HTTP visits (baseline / spec / spec / baseline) are for routing and
completion evidence, NOT a performance comparison: speculative K2 still
does host sync and probes. Do not use debug HTTP latency to claim speedup.
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

PHASES = ("base", "spec", "spec", "base")
STATUS = "P10_REAL_MODEL_AND_GRAPH_ON_COMPONENT_EVIDENCE_PASS_NOT_E2E"


def run(args: argparse.Namespace) -> dict:
    output = args.out.expanduser().resolve()
    arms = {
        "base": args.baseline_arm_file.expanduser().resolve(),
        "spec": args.spec_arm_file.expanduser().resolve(),
    }
    graph_dir = args.baseline_graph_dir.expanduser().resolve()
    probe_dir = args.probe_dir.expanduser().resolve()
    if output.exists() or graph_dir.exists() or probe_dir.exists():
        raise FileExistsError(
            "P10 output/evidence must be new; preserve prior GPU runs"
        )
    if len(set(arms.values())) != 2 or any(p.exists() for p in arms.values()):
        raise RuntimeError("Distinct P10 BASE/SPEC ARM markers must be absent")
    if any(not p.parent.is_dir() for p in arms.values()):
        raise FileNotFoundError("Create P10 ARM parent directories first")
    log = args.server_log.expanduser().resolve(strict=True)
    with urllib.request.urlopen(
        args.base_url.rstrip("/") + "/health", timeout=30
    ) as res:
        if res.status != 200:
            raise RuntimeError("P10 Omni worker not healthy")
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
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(payload, indent=2)+"\n")
    report = {
        "status": "P10_EVIDENCE_INCOMPLETE",
        "mode": args.mode,
        "visits": [],
        "request": payload,
        "limits": [
            "Client HTTP wall time includes deliberate probes and Python sync; NOT a perf ABBA.",
            "P10A/B are independent component graph captures; live output remains the debug K2 path.",
            "Native GPU sampler RNG and full GPU K2 bonus/cache integration are NOT yet validated.",
        ],
    }
    try:
        for idx, phase in enumerate(PHASES):
            arm = arms[phase]
            tag = f"{idx:02d}_{phase}"
            prior = len(parse_k2_log_entries(log.read_text(errors="replace")))
            print(f"P10_{args.mode.upper()}_BEGIN={tag}", flush=True)
            try:
                arm.touch(exist_ok=False)
                audio, finish, elapsed = _http_call(args.base_url, payload)
            finally:
                arm.unlink(missing_ok=True)
            after = len(parse_k2_log_entries(log.read_text(errors="replace")))
            frames = after - prior
            if finish != "stop" or (phase == "base" and frames != 0) or (
                phase == "spec" and frames == 0
            ):
                raise RuntimeError(
                    f"P10_STOP_OR_K2_ACTIVITY_FAIL {tag} finish={finish} frames={frames}"
                )
            (output / f"{tag}.wav").write_bytes(audio)
            report["visits"].append({
                "phase": phase,
                "finish_reason": finish,
                "k2_frames": frames,
                "wav_sha256": hashlib.sha256(audio).hexdigest(),
                "debug_http_wall_s_NOT_PERF": elapsed,
                **wav_info(audio),
            })
            (output / "report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(
                f"P10_{args.mode.upper()}_DONE={tag} stop={finish} "
                f"k2_frames={frames}", flush=True
            )
        graph_files = list(graph_dir.glob("baseline_graph_pid*.json"))
        if args.mode == "a":
            probe_files = list(probe_dir.glob("dynamic_kv_pid*_frame*.json"))
            required = 1
        else:
            probe_files = list(probe_dir.glob("sparse_pid*_frame*_depth*.json"))
            required = 2
        if len(graph_files) != 1 or len(probe_files) != required:
            raise RuntimeError(
                f"P10 EVIDENCE MISSING baseline={len(graph_files)} "
                f"P10{args.mode.upper()}={len(probe_files)} require={required}"
            )
        base = json.loads(graph_files[0].read_text())
        if (
            base.get("kind") != "baseline_full_predictor_graph_on"
            or base.get("frames", 0) < 8
            or base.get("graph_replays") != base.get("frames")
            or not isinstance(base.get("source_sha"), str)
            or len(base["source_sha"]) != 40
        ):
            raise RuntimeError("P10 baseline was not real optimized full Graph ON")
        seen = set()
        for file in probe_files:
            row = json.loads(file.read_text())
            if (
                row.get("source_sha") != base["source_sha"]
                or row.get("pid") != base.get("pid")
                or row.get("gpu") != base.get("device")
            ):
                raise RuntimeError(
                    f"P10 {file.name} mismatch source SHA, PID or GPU vs baseline"
                )
            if args.mode == "a":
                if (
                    not row.get("status", "").startswith(
                        "P10A_ONE_GRAPH_GPU_DYNAMIC_KV_FUSED_PREDICTOR_PARITY_PASS"
                    )
                    or row.get("predictor_layers") != 5
                    or len(row.get("cases", [])) != 4
                    or not all(case.get("pass") is True for case in row["cases"])
                ):
                    raise RuntimeError("P10A real five-layer KV parity failed")
            else:
                seen.add(row.get("proposal_depth"))
                if (
                    not row.get("status", "").startswith(
                        "P10B_REAL_PQ_SPARSE_TOPK_SPEC_SAMPLER_PARITY_PASS"
                    )
                    or row.get("private_rng_replay_unique") is not True
                    or row.get("input_logits_immutable") is not True
                    or len(row.get("checks", [])) != 4
                    or not all(case.get("pass") is True for case in row["checks"])
                ):
                    raise RuntimeError("P10B sparse real-logit parity/RNG failed")
        if args.mode == "b" and seen != {2, 10}:
            raise RuntimeError(f"P10B missing required on-policy depths 2/10: {seen}")
        report["status"] = STATUS
        report["source_sha"] = base["source_sha"]
        report["gpu"] = base["device"]
        report["pid"] = base["pid"]
    except Exception as exc:
        report["status"] = "P10_STOP_NO_PERFORMANCE_SELECTION"
        report["error"] = str(exc)
        (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        raise
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"P10_{args.mode.upper()}_RESULT={STATUS} REPORT={output/'report.json'}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("a", "b"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline-arm-file", type=Path, required=True)
    p.add_argument("--spec-arm-file", type=Path, required=True)
    p.add_argument("--baseline-graph-dir", type=Path, required=True)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
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
        p.error("P10 natural EOS requires max-new-tokens 256..2048")
    if not (0 < args.temperature <= 5 and 0 < args.top_p <= 1):
        p.error("P10 invalid temperature or top-p")
    if args.top_k != 50:
        p.error("P10B fused sparse top-k gate currently requires top_k=50")
    run(args)


if __name__ == "__main__":
    main()

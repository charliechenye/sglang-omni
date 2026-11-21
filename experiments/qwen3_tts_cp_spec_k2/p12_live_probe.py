# SPDX-License-Identifier: Apache-2.0
"""P12 strict source-locked native Graph ON / real K2 GPU component probe.

Runs baseline/spec/spec/baseline on one fresh H200 Qwen3-TTS checkpoint.
Mode selector: q stage observed before actual proposal and matching p stage
after the ACTUAL two-token verifier, then independent selector graphs.
Mode ar: one-token AR and packed verifier Graphs captured from identical
real committed KV and current model weights, with full KV restoration.
HTTP debug wall is NOT serving speed, and mode selector is NOT one
causal integrated Graph even though q/p are sampled at causal call sites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from urllib.request import urlopen

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

STATUS = "P12_REAL_CHECKPOINT_COMPONENT_PROBE_PASS_NOT_FULL_K2_OR_E2E"
PHASES = ("base", "spec", "spec", "base")


def validate_probe(row: dict, *, mode: str) -> None:
    if mode == "selector":
        if (
            row.get("schema") != "qwen3tts-cpspec-p12a-causal-stage-single-distribution-top50-v1"
            or not row.get("status", "").startswith(
                "P12A_CAUSAL_REAL_Q_PRE_AND_P_POST_COMPONENT_GATE_PASS"
            )
            or set(row.get("stages", {})) != {
                "q_before_verifier", "p_after_verifier",
            }
        ):
            raise ValueError("P12A real q/pre p/post evidence missing")
        for stage in row["stages"].values():
            if (
                not stage.get("status", "").startswith(
                    "P12A_SINGLE_DISTRIBUTION_TOP50_EXACT_MATCH_PASS"
                )
                or stage.get("selected_ids_match") is not True
                or stage.get("probability_match") is not True
                or len(stage.get("fixed_uniform_draws", [])) != 4
                or any(
                    len(series) != 4 or any(
                        type(v) not in (float, int)
                        or not math.isfinite(v) or v <= 0 for v in series
                    )
                    for series in stage.get("times_ms", {}).values()
                )
            ):
                raise ValueError("P12A tie, probability, deterministic draw or timer gate failed")
    else:
        if (
            row.get("schema") != "qwen3tts-cpspec-p12b-real-target-ar-vs-packed-verifier-cost-v1"
            or not row.get("status", "").startswith(
                "P12B_REAL_TARGET_AR_ONE_VS_PACKED_TWO_KV_PARITY_PASS"
            )
            or row.get("predictor_layers") != 5
            or row.get("parity_pass") is not True
            or len(row.get("checks", {})) < 8
            or not all(x.get("close") is True for x in row["checks"].values())
            or set(row.get("median_cuda_event_ms", {})) != {"ar_one", "packed_two"}
        ):
            raise ValueError("P12B real-model AR/packed KV parity evidence invalid")


def run(args):
    out = args.out.expanduser().resolve()
    base_arm = args.base_arm.expanduser().resolve()
    spec_arm = args.spec_arm.expanduser().resolve()
    root = args.evidence_dir.expanduser().resolve()
    base_dir = args.baseline_dir.expanduser().resolve()
    log = args.server_log.expanduser().resolve(strict=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != args.expect_sha or len(sha) != 40:
        raise RuntimeError("P12 SHA mismatch: restart on exact same commit")
    if (
        out.exists() or root.exists() or base_dir.exists()
        or base_arm == spec_arm or base_arm.exists() or spec_arm.exists()
        or not base_arm.parent.is_dir() or not spec_arm.parent.is_dir()
    ):
        raise RuntimeError("P12 all evidence/ARM/output paths must be fresh and distinct")
    with urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError("P12 Omni server not healthy")
    payload = {
        "model": args.model, "input": args.text, "voice": args.voice,
        "response_format": "wav", "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "stage_params": {"tts_engine": {
            "subtalker_dosample": True,
            "subtalker_temperature": args.temperature,
            "subtalker_top_k": args.top_k,
            "subtalker_top_p": args.top_p,
        }},
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "request.json").write_text(json.dumps(payload, indent=2)+"\n")
    report = {
        "status": "P12_INCOMPLETE",
        "mode": args.mode, "sha": sha,
        "visits": [], "request": payload,
    }
    try:
        for i, phase in enumerate(PHASES):
            if base_arm.exists() or spec_arm.exists():
                raise RuntimeError("P12 ARM leak between visits")
            arm = base_arm if phase == "base" else spec_arm
            prior = len(parse_k2_log_entries(log.read_text(errors="replace")))
            tag = f"{i:02d}_{phase}"
            print(f"P12_{args.mode.upper()}_BEGIN={tag}", flush=True)
            try:
                arm.touch(exist_ok=False)
                wav, finish, elapsed = _http_call(args.base_url, payload)
            finally:
                base_arm.unlink(missing_ok=True)
                spec_arm.unlink(missing_ok=True)
            observed = len(parse_k2_log_entries(log.read_text(errors="replace"))) - prior
            if finish != "stop" or (
                phase == "base" and observed != 0
            ) or (phase == "spec" and observed < 1):
                raise RuntimeError(
                    f"P12 {tag} STOP / K2 routing failed "
                    f"finish={finish} k2frames={observed}"
                )
            (out / f"{tag}.wav").write_bytes(wav)
            report["visits"].append({
                "tag": tag, "finish": finish,
                "k2_frames": observed,
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                "debug_http_wall_s_NOT_PERF": elapsed,
                **wav_info(wav),
            })
            (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"P12_{args.mode.upper()}_DONE={tag} stop k2frames={observed}", flush=True)

        baseline_files = list(base_dir.glob("baseline_graph_pid*.json"))
        evidence_files = sorted(root.glob(
            "selector_pid*_frame*_depth*.json" if args.mode == "selector"
            else "ar_vs_pair_pid*_frame*_depth*.json"
        ))
        if len(baseline_files) != 1 or len(evidence_files) != 2:
            raise RuntimeError(
                f"P12 missing same-worker baseline={len(baseline_files)} "
                f"probes={len(evidence_files)} expected 2"
            )
        baseline = json.loads(baseline_files[0].read_text())
        if (
            baseline.get("source_sha") != sha
            or baseline.get("kind") != "baseline_full_predictor_graph_on"
            or baseline.get("frames", 0) < 8
            or baseline.get("graph_replays") != baseline.get("frames")
        ):
            raise RuntimeError("P12 native full Predictor Graph ON baseline invalid")
        rows = [json.loads(p.read_text()) for p in evidence_files]
        depths = {row.get("depth") for row in rows}
        if depths != ({2, 10} if args.mode == "selector" else {4, 10}):
            raise RuntimeError(f"P12 expected real depths missing: {depths}")
        for row in rows:
            validate_probe(row, mode=args.mode)
            if (
                row.get("source_sha") != sha
                or row.get("pid") != baseline.get("pid")
                or row.get("gpu") != baseline.get("device")
            ):
                raise RuntimeError("P12 probe source SHA, worker PID or GPU mismatch")
        report["status"] = STATUS
        report["gpu"] = baseline["device"]
        report["baseline_graph_on_ms"] = baseline["median_cuda_event_ms"]
        report["probes"] = rows
        report["limits"] = [
            "CPU-sync debug K2 HTTP latency is NOT QPS evidence",
            "P12A q/p observed in causal order, but selection Graphs replay independently",
            "P12A exact torch topk ties and fixed-uniform results required before cost selection",
            "P12B AR/pair Graphs share model weights and KV prefix, no heads or sampler cost",
            "No full GPU K2, native seeded RNG parity or hybrid quality proof",
        ]
    except Exception as exc:
        report["status"] = "P12_FAILED_NO_PERFORMANCE_SELECTION"
        report["error"] = str(exc)
        (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        raise
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = [
        f"# P12 {args.mode} on-policy H200 real component screen",
        "", f"**{STATUS}**", "",
        f"Source SHA: {sha}",
        f"GPU: {report['gpu']}",
        f"Native full Graph ON: {report['baseline_graph_on_ms']:.5f} ms/frame", "",
    ]
    if args.mode == "selector":
        lines += [
            "| Real proposal depth | Stage | Torch topk+draw ms | Triton top50+draw ms | Ratio |",
            "|---:|---|---:|---:|---:|",
        ]
        for row in rows:
            for stage, data in row["stages"].items():
                med = data["median_ms"]
                lines.append(
                    f"| {row['depth']} | {stage} | "
                    f"{med['torch_topk_reference']:.5f} | "
                    f"{med['triton_single_top50']:.5f} | "
                    f"{data['fast_over_reference_ratio']:.3f}x |"
                )
    else:
        lines += [
            "| Real proposal depth | 1-token target AR ms | Packed 2-token verifier ms | AR / packed | Hidden/KV parity |",
            "|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            timing = row["median_cuda_event_ms"]
            lines.append(
                f"| {row['depth']} | {timing['ar_one']:.5f} | "
                f"{timing['packed_two']:.5f} | "
                f"{row['one_over_pair_ratio']:.3f}x | {row['parity_pass']} |"
            )
    lines += ["", "## Non-claims", ""] + ["- "+x for x in report["limits"]]
    (out / "summary.md").write_text("\n".join(lines)+"\n")
    print(f"P12_{args.mode.upper()}_STATUS={STATUS} REPORT={out/'summary.md'}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("selector","ar"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-arm", type=Path, required=True)
    p.add_argument("--spec-arm", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--expect-sha", required=True)
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
    if (
        not 256 <= args.max_new_tokens <= 2048
        or not 0 < args.temperature <= 5
        or not 0 < args.top_p <= 1
        or args.top_k != 50
    ):
        p.error("P12 only supports C1 sampled topk50 natural-stop path")
    run(args)


if __name__ == "__main__":
    main()

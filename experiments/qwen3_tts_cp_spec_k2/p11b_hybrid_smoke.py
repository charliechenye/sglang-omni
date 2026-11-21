# SPDX-License-Identifier: Apache-2.0
"""P11B C1 on-policy native / K2 / hybrid AR(2,3)+K2 correctness smoke.

One real Qwen3-TTS server, same weights and source SHA, with request-scoped
file markers. The native path is the real optimized Graph-ON decoder;
K2/hybrid run the existing CPU-sync debug path. Sequence for every seed:
native, K2, hybrid, hybrid, K2, native. All visits require natural EOS.
The hybrid arm must be created only during a hybrid request.

This is NOT a serving performance ABBA: debug K2 and hybrid have Python
sync, extra logs, and different sampling RNG consumption. Different WAV
hashes are expected. Use it to check actual target AR prefix, K2
proposal/bonus accounting, EOS and repeatability, not speed or WER.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

MODES = ("native", "k2", "hybrid", "hybrid", "k2", "native")
STATUS = "P11B_REAL_ON_POLICY_HYBRID_AR_K2_ALL_STOP_ACCOUNTING_PASS_NOT_PARITY_OR_SPEED"
DEPTH_RE = re.compile(
    r"QTTS_CP_K2_DEPTHS frame=(\d+) ar_through=(\d+) "
    r"ar=([0-9,-]+) proposed=([0-9,-]+) "
    r"accepted=([0-9,-]+) bonus=([0-9,-]+) source_sha=([0-9a-f]{40})"
)


def decode_depths(field: str) -> list[int]:
    if field == "-":
        return []
    values = [int(v) for v in field.split(",")]
    if values != sorted(set(values)) or any(v < 2 or v > 15 for v in values):
        raise ValueError(f"P11B invalid or duplicate codebook depth list: {field}")
    return values


def parse_depth_stats(text: str) -> list[dict]:
    output = []
    for match in DEPTH_RE.finditer(text):
        frame, through, ar, proposed, accepted, bonus, sha = match.groups()
        row = {
            "frame": int(frame),
            "ar_through": int(through),
            "ar": decode_depths(ar),
            "proposed": decode_depths(proposed),
            "accepted": decode_depths(accepted),
            "bonus": decode_depths(bonus),
            "source_sha": sha,
        }
        depths = row["ar"] + row["proposed"] + row["bonus"]
        if (
            row["ar_through"] not in (1, 2, 3)
            or sorted(depths) != list(range(2, 16))
            or any(x not in row["proposed"] for x in row["accepted"])
            or len(row["ar"]) not in (0, row["ar_through"] - 1)
        ):
            raise ValueError(f"P11B per-depth trace failed conservation: {row}")
        output.append(row)
    return output


def check_visit(
    mode: str,
    rows: list[dict],
    depths: list[dict],
    expected_sha: str,
    expected_through: int,
) -> dict:
    if mode == "native":
        if rows or depths:
            raise RuntimeError("P11B native Graph-ON reference executed K2")
        return {"frames": 0, "target_calls": 0, "spec_rounds": 0, "ar_steps": 0}

    if len(rows) < 1 or len(rows) != len(depths):
        raise RuntimeError(
            f"P11B {mode} trace incomplete: K2 frames {len(rows)}, depth rows {len(depths)}"
        )
    for x, d in zip(rows, depths):
        desired_ar = [2, 3] if mode == "hybrid" and expected_through == 3 else (
            [2] if mode == "hybrid" and expected_through == 2 else []
        )
        if (
            x["frame"] != d["frame"]
            or d["source_sha"] != expected_sha
            or x["ar_steps"] != len(desired_ar)
            or x["ar_through"] != (expected_through if mode == "hybrid" else 1)
            or d["ar_through"] != x["ar_through"]
            or d["ar"] != desired_ar
            or x["rounds"] != len(d["proposed"])
            or x["accepted"] != len(d["accepted"])
            or x["bonus"] != len(d["bonus"])
        ):
            raise RuntimeError(
                f"P11B {mode} source/counters/hybrid mask mismatch "
                f"K2={x} depth={d}"
            )
    rounds = sum(x["rounds"] for x in rows)
    direct = sum(x["ar_steps"] for x in rows)
    accepted = sum(x["accepted"] for x in rows)
    bonuses = sum(x["bonus"] for x in rows)
    proposed = sum(x["proposed"] for x in rows)
    depth_activity = {}
    for depth in range(2, 16):
        depth_activity[str(depth)] = {
            "ar": sum(depth in x["ar"] for x in depths),
            "proposed": sum(depth in x["proposed"] for x in depths),
            "accepted": sum(depth in x["accepted"] for x in depths),
            "bonus": sum(depth in x["bonus"] for x in depths),
        }
    return {
        "frames": len(rows),
        "spec_rounds": rounds,
        "ar_steps": direct,
        "target_calls": rounds + direct,
        "spec_rounds_per_frame": rounds / len(rows),
        "all_target_calls_per_frame": (rounds + direct) / len(rows),
        "accept_fraction_on_proposals": accepted / proposed,
        "bonus_codes": bonuses,
        "depth_activity": depth_activity,
    }


def summarize(output: dict) -> str:
    lines = [
        "# P11B — Native vs K2 vs target AR prefix + K2, same real C1 worker",
        "",
        f"**{output['status']}**",
        "",
        f"Source SHA: {output['source_sha']}",
        f"Seed(s): {output['seeds']}",
        f"Hybrid AR through: residual depth {output['ar_through']}",
        "",
        "| Seed | Mode | finish | TTS audio s | frames | Spec rounds/frame | Direct AR/frame | Combined target calls/frame | Accept fraction |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output["visits"]:
        m = row["metrics"]
        frames = m["frames"]
        lines.append(
            f"| {row['seed']} | {row['mode']} | {row['finish_reason']} | "
            f"{row['duration_s']:.3f} | {frames} | "
            f"{(m['spec_rounds']/frames if frames else 0):.3f} | "
            f"{(m['ar_steps']/frames if frames else 0):.3f} | "
            f"{(m['target_calls']/frames if frames else 0):.3f} | "
            f"{(m.get('accept_fraction_on_proposals') or 0):.2%} |"
        )
    lines.extend([
        "", "## On-policy depth usage aggregated over the seeded visits", "",
        "| Residual depth | Pure K2 proposals | Pure K2 acceptance | Pure K2 bonus | Hybrid direct AR | Hybrid proposals | Hybrid acceptance | Hybrid bonus |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for depth in range(2, 16):
        def stats(mode: str, field: str) -> int:
            return sum(
                item["metrics"]["depth_activity"][str(depth)][field]
                for item in output["visits"] if item["mode"] == mode
            )
        k2_proposed = stats("k2", "proposed")
        k2_accept = stats("k2", "accepted")
        h_proposed = stats("hybrid", "proposed")
        h_accept = stats("hybrid", "accepted")
        k2_rate = (f"{k2_accept/k2_proposed:.1%}"
                   if k2_proposed else "—")
        hybrid_rate = (f"{h_accept/h_proposed:.1%}"
                       if h_proposed else "—")
        lines.append(
            f"| {depth} | {k2_proposed} | {k2_rate} | "
            f"{stats('k2', 'bonus')} | {stats('hybrid', 'ar')} | "
            f"{h_proposed} | {hybrid_rate} | {stats('hybrid', 'bonus')} |"
        )
    lines.extend([
        "", "## Gate limitations", "",
        "- Natural EOS, correct depth accounting and fixed-seed source attribution are tested.",
        "- Native Graph-ON, K2 debug, and hybrid debug must NOT be ranked by HTTP wall time.",
        "- The direct AR steps use the debug K2 generator, not native fused sampler RNG.",
        "- Hybrid changes RNG consumption and state visitation; WAV hash inequality is expected.",
        "- No WER, speaker similarity, native sampler parity or E2E GPU K2 speedup claim.",
        "- A hybrid with fewer speculative rounds may still require more/longer target GPU work.",
    ])
    return "\n".join(lines) + "\n"


def run(args) -> dict:
    out = args.out.expanduser().resolve()
    spec_arm = args.spec_arm.expanduser().resolve()
    hybrid_arm = args.hybrid_arm.expanduser().resolve()
    server_log = args.server_log.expanduser().resolve(strict=True)
    if (
        out.exists() or spec_arm == hybrid_arm
        or spec_arm.exists() or hybrid_arm.exists()
        or not spec_arm.parent.is_dir()
        or not hybrid_arm.parent.is_dir()
    ):
        raise RuntimeError("P11B requires fresh output and distinct absent ARM markers")
    if not 1 <= len(args.seeds) <= 8 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("P11B seeds must be 1..8 distinct integers")
    if args.ar_through not in (2, 3):
        raise ValueError("P11B hybrid through depth must be 2 or 3")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if len(sha) != 40 or sha != args.expect_source_sha:
        raise RuntimeError("P11B Git SHA mismatch: stop and restart the server on one revision")
    with urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as health:
        if health.status != 200:
            raise RuntimeError("P11B Omni server health failed")
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "P11B_INCOMPLETE",
        "source_sha": sha,
        "ar_through": args.ar_through,
        "seeds": args.seeds,
        "visits": [],
        "limits": [
            "C1 debug serving traces only, not profiler-OFF HTTP speedup",
            "Never infer lossless sampler/native RNG equivalence from WAV hashes",
        ],
    }
    try:
        for seed in args.seeds:
            payload = {
                "model": args.model,
                "input": args.text,
                "voice": args.voice,
                "seed": seed,
                "response_format": "wav",
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
            for idx, mode in enumerate(MODES):
                if spec_arm.exists() or hybrid_arm.exists():
                    raise RuntimeError("P11B leaked ARM marker from previous visit")
                content_before = server_log.read_text(errors="replace")
                old_rows = len(parse_k2_log_entries(content_before))
                old_depths = len(parse_depth_stats(content_before))
                name = f"seed{seed}_{idx:02d}_{mode}"
                print(f"P11B_BEGIN={name}", flush=True)
                try:
                    if mode in ("k2", "hybrid"):
                        spec_arm.touch(exist_ok=False)
                    if mode == "hybrid":
                        hybrid_arm.touch(exist_ok=False)
                    wav, finish, elapsed = _http_call(args.base_url, payload)
                finally:
                    hybrid_arm.unlink(missing_ok=True)
                    spec_arm.unlink(missing_ok=True)
                if finish != "stop":
                    raise RuntimeError(
                        f"P11B {name}: finish={finish}; natural-stop required"
                    )
                content_after = server_log.read_text(errors="replace")
                rows = parse_k2_log_entries(content_after)[old_rows:]
                depths = parse_depth_stats(content_after)[old_depths:]
                metrics = check_visit(
                    mode, rows, depths, sha, args.ar_through
                )
                (out / f"{name}.wav").write_bytes(wav)
                record = {
                    "seed": seed,
                    "tag": name,
                    "mode": mode,
                    "finish_reason": finish,
                    "debug_http_wall_s_NOT_PERF": elapsed,
                    "wav_sha256": hashlib.sha256(wav).hexdigest(),
                    **wav_info(wav),
                    "metrics": metrics,
                }
                report["visits"].append(record)
                (out / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(
                    f"P11B_DONE={name} stop={finish} "
                    f"frames={metrics['frames']} ar={metrics['ar_steps']} "
                    f"spec={metrics['spec_rounds']}",
                    flush=True,
                )
        report["status"] = STATUS
    except Exception as exc:
        report["status"] = "P11B_FAIL_NO_HYBRID_SELECTION"
        report["error"] = str(exc)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.md").write_text(summarize(report))
    print(f"P11B_RESULT={report['status']} REPORT={out/'summary.md'}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--server-log", required=True, type=Path)
    p.add_argument("--spec-arm", required=True, type=Path)
    p.add_argument("--hybrid-arm", required=True, type=Path)
    p.add_argument("--expect-source-sha", required=True)
    p.add_argument("--ar-through", type=int, default=3)
    p.add_argument("--seeds", default="20261120,20261121")
    p.add_argument("--text", default=TEXT)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--voice", default="Vivian")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    args = p.parse_args()
    args.seeds = [int(x) for x in args.seeds.split(",")]
    if not 256 <= args.max_new_tokens <= 2048:
        p.error("P11B natural-stop requires max-new-tokens 256..2048")
    if (
        not 0 < args.temperature <= 5
        or not 0 < args.top_p <= 1
        or args.top_k != 50
    ):
        p.error("P11B requires current sampled TTS top-k50 configuration")
    run(args)


if __name__ == "__main__":
    main()

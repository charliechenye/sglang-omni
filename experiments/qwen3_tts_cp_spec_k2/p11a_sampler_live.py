# SPDX-License-Identifier: Apache-2.0
"""P11A same-worker real p/q sampler attribution, source-locked Graph ON.

Run baseline/spec/spec/baseline in a single fresh H200 Omni worker,
capturing native complete Predictor Graph ON baseline and *actual K2*
p/q states at residual depths 2,10. The post-topk Triton kernel is
validated independently against P10B before timing topk/post/RNG/full.

Output is a COMPONENT profiling report, never production QPS/latency.
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
from sglang_omni.models.qwen3_tts.code_predictor_spec_p11a_attribution import (
    CELLS, SCHEMA,
)

PHASES = ("base", "spec", "spec", "base")
STATUS = "P11A_REAL_K2_PQ_SAMPLER_ATTRIBUTION_PASS_NOT_E2E"


def collect(args):
    out = args.out.expanduser().resolve()
    base_arm = args.base_arm.expanduser().resolve()
    spec_arm = args.spec_arm.expanduser().resolve()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    evidence_dir = args.attribution_dir.expanduser().resolve()
    server_log = args.server_log.expanduser().resolve(strict=True)
    if (
        out.exists() or baseline_dir.exists() or evidence_dir.exists()
        or base_arm == spec_arm
        or base_arm.exists() or spec_arm.exists()
        or not base_arm.parent.is_dir() or not spec_arm.parent.is_dir()
    ):
        raise RuntimeError("P11A requires new output, unique absent ARM paths and evidence")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != args.expect_source_sha:
        raise RuntimeError("P11A source SHA drift; restart worker with pinned source")
    with urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as health:
        if health.status != 200:
            raise RuntimeError("P11A Omni worker not ready")
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
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "P11A_INCOMPLETE",
        "source_sha": sha,
        "request": payload,
        "visits": [],
    }
    try:
        for idx, phase in enumerate(PHASES):
            arm = base_arm if phase == "base" else spec_arm
            if base_arm.exists() or spec_arm.exists():
                raise RuntimeError("P11A leaked ARM from previous visit")
            tag = f"{idx:02d}_{phase}"
            prior = len(parse_k2_log_entries(server_log.read_text(errors="replace")))
            print(f"P11A_BEGIN={tag}", flush=True)
            try:
                arm.touch(exist_ok=False)
                wav, finish, elapsed = _http_call(args.base_url, payload)
            finally:
                arm.unlink(missing_ok=True)
            after = len(parse_k2_log_entries(server_log.read_text(errors="replace")))
            frames = after - prior
            if finish != "stop" or (phase == "base" and frames != 0) or (
                phase == "spec" and frames == 0
            ):
                raise RuntimeError(
                    f"P11A wrong finish/activation {tag} finish={finish} frames={frames}"
                )
            (out / f"{tag}.wav").write_bytes(wav)
            report["visits"].append({
                "tag": tag,
                "finish_reason": finish,
                "k2_frames": frames,
                "debug_http_wall_NOT_PERF": elapsed,
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                **wav_info(wav),
            })
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"P11A_DONE={tag} finish=stop k2_frames={frames}", flush=True)

        bfiles = list(baseline_dir.glob("baseline_graph_pid*.json"))
        cfiles = sorted(evidence_dir.glob("attr_pid*_frame*_depth*.json"))
        if len(bfiles) != 1 or len(cfiles) != 2:
            raise RuntimeError(
                f"P11A incomplete baseline={len(bfiles)} sampled states={len(cfiles)}"
            )
        base = json.loads(bfiles[0].read_text())
        if (
            base.get("kind") != "baseline_full_predictor_graph_on"
            or base.get("source_sha") != sha
            or base.get("frames", 0) < 8
            or base.get("graph_replays") != base.get("frames")
        ):
            raise RuntimeError("P11A not an actual source-matched full Graph-ON baseline")
        rows = [json.loads(path.read_text()) for path in cfiles]
        if {x.get("proposal_depth") for x in rows} != {2, 10}:
            raise RuntimeError("P11A requires separate on-policy depth 2 and 10 logits")
        for row in rows:
            metrics = row.get("measurements", {})
            if (
                row.get("schema") != SCHEMA
                or not row.get("status", "").startswith(
                    "P11A_REAL_PQ_TOPK_POST_RNG_ATTRIBUTION_PASS"
                )
                or row.get("source_sha") != sha
                or row.get("pid") != base.get("pid")
                or row.get("gpu") != base.get("device")
                or row.get("torch_reference_distribution_parity") is not True
                or row.get("original_pq_unchanged") is not True
                or row.get("private_rng_independent_unique_replays") is not True
                or set(metrics) != set(CELLS)
                or any(
                    len(metrics[name].get("block_cuda_event_ms", [])) != 4
                    or not math.isfinite(metrics[name].get("median_cuda_event_ms", math.nan))
                    or metrics[name]["median_cuda_event_ms"] <= 0
                    for name in CELLS
                )
            ):
                raise RuntimeError("P11A numerical/RNG or stage capture gate failed")

        report.update({
            "status": STATUS,
            "native_full_graph_on_ms_per_frame": base["median_cuda_event_ms"],
            "gpu": base["device"],
            "real_model_pid": base["pid"],
            "depth_profiles": rows,
            "non_claims": [
                "Four stage graph envelopes are NOT additive.",
                "No native seeded sampler RNG equivalence.",
                "Historical P10B source differed; same-worker P10B included in each P11A depth.",
                "Not integrated full GPU K2 or serving HTTP speedup.",
            ],
        })
    except Exception as exc:
        report["status"] = "P11A_FAIL_NO_SAMPLER_SELECTION"
        report["error"] = str(exc)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# P11A real p/q H200 sparse sampler cost attribution",
        "", f"**{STATUS}**", "",
        f"Source SHA: {sha}",
        f"GPU: {base['device']}",
        f"Native complete Graph ON baseline: {base['median_cuda_event_ms']:.5f} ms/frame",
        "",
        "| On-policy depth | Two topk ms | Triton post-topk ms | Private RNG ms | Complete sparse graph ms | Same-worker P10B sparse ms | P10B dense ms | Overlap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x:x["proposal_depth"]):
        m=row["measurements"]
        lines.append(
            f"| {row['proposal_depth']} | "
            f"{m['topk_pair']['median_cuda_event_ms']:.5f} | "
            f"{m['post_topk_triton']['median_cuda_event_ms']:.5f} | "
            f"{m['private_rng']['median_cuda_event_ms']:.5f} | "
            f"{m['full_sparse']['median_cuda_event_ms']:.5f} | "
            f"{row['p10b_same_worker_sparse_graph_ms']:.5f} | "
            f"{row['p10b_same_worker_dense_graph_ms']:.5f} | "
            f"{row['overlap']:.2%} |"
        )
    lines += [
        "", "## Interpretation constraints", "",
        "- Comparing stage CUDA Graph envelopes does NOT yield causal kernel busy-time fractions.",
        "- Do not sum stage times as a substitute for a composite Graph measurement.",
        "- Real top-50 source and output correctness are validated; no native RNG parity.",
        "- The next fused dual-topk kernel should be considered only if topk is material.",
        "- No production full-K2 serving speedup follows from this report.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"P11A_RESULT={STATUS} SUMMARY={out/'summary.md'}")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--base-arm", type=Path, required=True)
    p.add_argument("--spec-arm", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--attribution-dir", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expect-source-sha", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--text", default=TEXT)
    p.add_argument("--voice", default="Vivian")
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.)
    args=p.parse_args()
    if (
        not 256<=args.max_new_tokens<=2048
        or not 0<args.temperature<=5
        or not 0<args.top_p<=1
        or args.top_k!=50
    ):
        p.error("P11A requires current natural-stop sampled topk50 path")
    collect(args)


if __name__=="__main__":
    main()

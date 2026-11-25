# SPDX-License-Identifier: Apache-2.0
"""P5 native Graph-ON whole-AR baseline vs isolated K2 verifier graphs.

A deliberately optimistic *feasibility* screen, NOT real serving speedup.
Baseline is the actual full Predictor CUDA graph (15 codebook chain, sampled)
and candidate is only repeated one-/two-token captured verifier stacks,
excluding new K2 sampling, bonus, embedding/scheduler and control flow.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sglang_omni.models.qwen3_tts.code_predictor_spec_graph_cost import SCHEMA
from experiments.qwen3_tts_cp_spec_k2.p4_cost_screen import validate_acceptance

STATUS = "P5_GRAPH_ON_VS_K2_CAPTURED_STACK_LOWER_BOUND_ONLY_NOT_SERVING_SPEEDUP"


def valid_real(x: object) -> bool:
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def analyze(perf_dir: Path, acceptance_report: Path, *, min_probes: int = 4) -> dict:
    perf_dir = perf_dir.expanduser().resolve(strict=True)
    baselines = sorted(perf_dir.glob("baseline_graph_pid*.json"))
    if len(baselines) != 1:
        raise RuntimeError(
            f"P5 requires exactly one actual Graph ON baseline report, got {len(baselines)}"
        )
    base = json.loads(baselines[0].read_text())
    if base.get("schema") != SCHEMA or base.get("kind") != "baseline_full_predictor_graph_on":
        raise ValueError("P5 baseline missing actual existing Graph ON evidence")
    if base.get("graph_replays") != base.get("frames") or base.get("frames", 0) < 8:
        raise ValueError("P5 baseline missing replay coverage")
    base_ms = base.get("median_cuda_event_ms")
    if not valid_real(base_ms):
        raise ValueError("bad P5 baseline CUDA event timing")
    selected = sorted(perf_dir.glob("packed_graph_pid*_frame*_depth*.json"))
    if len(selected) < min_probes:
        raise RuntimeError(
            f"P5 needs {min_probes} K2 captured-graph probes; got {len(selected)}"
        )
    rows = [json.loads(p.read_text()) for p in selected]
    if any(
        r.get("schema") != SCHEMA
        or r.get("kind") != "single_vs_packed_verifier_cuda_graph"
        or not valid_real(r.get("serial_one_graph_ms"))
        or not valid_real(r.get("packed_two_graph_ms"))
        or r.get("proposal_depth") not in range(2, 16)
        for r in rows
    ):
        raise ValueError("invalid/mixed P5 captured graph samples")
    if len({r["proposal_depth"] for r in rows}) != len(rows):
        raise ValueError("P5 repeated measured depths; use one server run")
    source_shas = {base.get("source_sha")} | {r.get("source_sha") for r in rows}
    gpu_names = {base.get("device")} | {r.get("device") for r in rows}
    pids = {base.get("pid")} | {r.get("pid") for r in rows}
    if (
        len(source_shas) != 1 or "UNATTESTED" in source_shas or None in source_shas
        or len(gpu_names) != 1 or None in gpu_names
        or len(pids) != 1
    ):
        raise ValueError("P5 baseline and K2 graphs must be captured in ONE source/GPU/worker")
    acceptance = validate_acceptance(
        json.loads(acceptance_report.expanduser().resolve(strict=True).read_text())
    )
    by_depth = {r["proposal_depth"]: r for r in rows}
    measured_depths = sorted(by_depth)

    def nearest(depth: int) -> dict:
        return by_depth[min(measured_depths, key=lambda d: (abs(d - depth), d))]

    predicted_packed_ms = 0.0
    predicted_graph1_ms = 0.0
    mapping = []
    for cell in acceptance["depths"]:
        depth = cell["residual_depth"]
        n = cell["proposals"]
        other = nearest(depth)
        predicted_packed_ms += n * other["packed_two_graph_ms"]
        predicted_graph1_ms += n * other["serial_one_graph_ms"]
        mapping.append({
            "depth": depth,
            "n_proposals": n,
            "profiled_state_depth": other["proposal_depth"],
        })
    # These costs are normalized per independent sampled predictor frame.
    candidate_stack_per_frame_ms = predicted_packed_ms / acceptance["frames"]
    baseline_per_frame_ms = base_ms
    screen_ratio = baseline_per_frame_ms / candidate_stack_per_frame_ms
    if any(not valid_real(x) for x in (screen_ratio, candidate_stack_per_frame_ms, predicted_graph1_ms)):
        raise ValueError("P5 invalid computed cost")
    return {
        "status": STATUS,
        "candidate_source_sha": next(iter(source_shas)),
        "acceptance_cohort_sha": acceptance.get("source_sha"),
        "source_sha_same_as_prior_cohort": next(iter(source_shas)) == acceptance.get("source_sha"),
        "gpu": next(iter(gpu_names)),
        "baseline_graph_frames": base["frames"],
        "baseline_frame_graph_event_median_ms": baseline_per_frame_ms,
        "k2_proposals_per_frame_from_prior_cohort": (
            acceptance["proposals"] / acceptance["frames"]
        ),
        "measured_packed_graph_depths": measured_depths,
        "k2_verifier_graph_only_extrapolated_ms_per_frame": candidate_stack_per_frame_ms,
        "optimistic_baseline_over_k2_verifier_stack_quotient": screen_ratio,
        "optimistic_k2_verifier_below_baseline_graph": (
            candidate_stack_per_frame_ms < baseline_per_frame_ms
        ),
        "actual_p5_graph_samples": [{
            "depth": r["proposal_depth"],
            "cache_len": r["cache_len"],
            "serial_one_graph_ms": r["serial_one_graph_ms"],
            "packed_two_graph_ms": r["packed_two_graph_ms"],
            "packed_to_one_ratio": r["packed_to_one_ratio"],
        } for r in sorted(rows, key=lambda r: r["proposal_depth"])],
        "nearest_depth_approximations": mapping,
        "limitations": [
            "This is an OPTIMISTIC feasibility lower-bound cost for K2, not serving throughput.",
            "Graph-ON baseline is full target predictor chain including normal sampling; K2 captured graphs contain ONLY verify stack.",
            "Missing K2 initial pair, draft heads, sample/accept, bonus sampler, rejected-suffix management, remaining host graph launches, and vocoder/talker.",
            "Previously observed 720-frame K2 proposals are imported from prior source SHA; if different, on-policy acceptance may drift.",
            "Four KV states are extrapolated by nearest depth and may differ from later states/other prompts.",
            "Independently graph-replayed 2-row stack is not a CUDA Graph of the entire conditional K2 algorithm.",
            "CUDA event includes graph launch envelope; not CUPTI or Nsight pure GPU kernel busy time.",
            "If lower-bound K2 already costs more than full baseline, this independent-graph-per-round implementation is not promising; whole-chain fusion could still differ.",
            "If optimistic screen is positive, a real GPU-resident implementation and fresh-process equal-workload ABBA remain mandatory.",
        ],
    }


def write_report(out: Path, data: dict) -> Path:
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Qwen3-TTS P5 — real Graph ON baseline vs captured K2 verifier lower bound",
        "",
        "**" + data["status"] + "**", "",
        f"GPU: {data['gpu']}",
        f"Source SHA: {data['candidate_source_sha']}",
        f"P2 cohort SHA: {data['acceptance_cohort_sha']}",
        f"Same source: {data['source_sha_same_as_prior_cohort']}",
        f"Baseline sampled full-Predictor Graph ON frames: {data['baseline_graph_frames']}",
        f"Prior K2 proposals/frame: {data['k2_proposals_per_frame_from_prior_cohort']:.3f}",
        "",
        "| Cost metric | CUDA event ms |",
        "|---|---:|",
        f"| Existing full Predictor Graph ON per frame (median) | {data['baseline_frame_graph_event_median_ms']:.4f} |",
        f"| K2 verifier graph-only lower bound per frame | {data['k2_verifier_graph_only_extrapolated_ms_per_frame']:.4f} |",
        "",
        "**Optimistic baseline / K2 verifier stack quotient = "
        f"{data['optimistic_baseline_over_k2_verifier_stack_quotient']:.3f}x**",
        "",
        "## Measured static on-policy captured graph replays",
        "",
        "| Depth | Cache len | One-token graph ms | Packed-two graph ms | Ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in data["actual_p5_graph_samples"]:
        lines.append(
            f"| {row['depth']} | {row['cache_len']} | "
            f"{row['serial_one_graph_ms']:.4f} | {row['packed_two_graph_ms']:.4f} | "
            f"{row['packed_to_one_ratio']:.3f} |"
        )
    lines.extend(["", "## Mandatory interpretation", ""] + [
        "- " + item for item in data["limitations"]
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return out / "summary.md"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perf-dir", required=True, type=Path)
    p.add_argument("--acceptance-report", required=True, type=Path)
    p.add_argument("--min-probes", type=int, default=4)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if not 1 <= args.min_probes <= 14:
        p.error("min-probes must be 1..14")
    result = analyze(
        args.perf_dir, args.acceptance_report, min_probes=args.min_probes
    )
    print(f"{result['status']}: {write_report(args.out, result)}")


if __name__ == "__main__":
    main()

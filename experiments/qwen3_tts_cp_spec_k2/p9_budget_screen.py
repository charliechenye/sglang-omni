# SPDX-License-Identifier: Apache-2.0
"""P9 H200 composite acceleration *screen*: real multi-depth GPU graph + p/q GPU sampler.

Unlike P5's verifier-only screen, P9 combines the measured nested 4-way
real Predictor graph with the actual CUDA-graph top-k/top-p/sample/accept/
positive-residual kernel on real checkpoint logits. It still CANNOT
predict the full GPU K2 runtime: 4 nonadjacent offsets and separate graph
launches do not implement a 14-depth on-policy algorithm, and no opening
pair, heads/bonus/embedding/KV correction are included.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from sglang_omni.models.qwen3_tts.code_predictor_spec_p9a import (
    SCHEMA as A_SCHEMA,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p9b import (
    SCHEMA as B_SCHEMA,
)
from experiments.qwen3_tts_cp_spec_k2.p4_cost_screen import validate_acceptance

NO_GAIN = "P9_COMPONENT_COST_SCREEN_UNFAVORABLE_NO_SPEEDUP_CLAIM"
POSSIBLE = "P9_COMPONENT_COST_SCREEN_POTENTIALLY_FAVORABLE_NOT_E2E"
SCHEMA = "qwen3tts-cpspec-p9-component-budget-screen-v1"


def check_positive(x) -> float:
    if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"expected finite positive measurement, got {x!r}")
    return float(x)


def linear_cost(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares fit of 2/3/4-call nested control graph cost in ms."""
    if len(points) < 3 or {x for x, _ in points} != {2, 3, 4}:
        raise ValueError("P9A needs 2, 3 AND 4 rounds for extrapolation")
    avg_x = sum(x for x, _ in points) / len(points)
    avg_y = sum(y for _, y in points) / len(points)
    denom = sum((x-avg_x)**2 for x, _ in points)
    slope = sum((x-avg_x)*(y-avg_y) for x, y in points) / denom
    intercept = avg_y - slope * avg_x
    if slope <= 0:
        raise ValueError("P9A graph time did not increase with actual rounds")
    return intercept, slope


def analyze(
    four_offset_dir: Path,
    sampling_dir: Path,
    baseline_graph_dir: Path,
    acceptance_report: Path,
    p9b_baseline_dir: Path | None = None,
) -> dict:
    four_offset_dir = four_offset_dir.expanduser().resolve(strict=True)
    sampling_dir = sampling_dir.expanduser().resolve(strict=True)
    baseline_graph_dir = baseline_graph_dir.expanduser().resolve(strict=True)
    a_paths = sorted(four_offset_dir.glob("four_depth_pid*_frame*.json"))
    b_paths = sorted(sampling_dir.glob("sample_pid*_depth*.json"))
    # P9B filenames include frame before depth; use explicit per-file validation.
    if not b_paths:
        b_paths = sorted(sampling_dir.glob("sample_pid*_frame*_depth*.json"))
    baseline_paths = sorted(baseline_graph_dir.glob("baseline_graph_pid*.json"))
    if len(a_paths) != 1 or len(b_paths) != 2 or len(baseline_paths) != 1:
        raise RuntimeError(
            f"P9 evidence missing/mixed: fourdepth={len(a_paths)} "
            f"sampler={len(b_paths)} GraphON={len(baseline_paths)}"
        )
    a = json.loads(a_paths[0].read_text())
    bs = [json.loads(x.read_text()) for x in b_paths]
    baseline = json.loads(baseline_paths[0].read_text())
    if a.get("schema") != A_SCHEMA or not a.get("status", "").startswith(
        "P9A_FOUR_REAL_PREDICTOR_OFFSET_GRAPH_PARITY_PASS"
    ):
        raise ValueError("P9A real-model four-offset parity has not passed")
    if len(a.get("cases", [])) != 6 or not all(
        x.get("pass") is True for x in a["cases"]
    ):
        raise ValueError("P9A branch/KV parity cases missing")
    if a.get("predictor_layers") != 5:
        raise ValueError("P9A must use the real five-layer checkpoint")
    if any(
        b.get("schema") != B_SCHEMA or not b.get("status", "").startswith(
            "P9B_REAL_LOGITS_GPU_PROPOSAL_ACCEPT_RESIDUAL_AND_RNG_PASS"
        )
        or b.get("independent_graph_rng_replay_unique") is not True
        or b.get("output_input_parity") is not True
        or len(b.get("checks", [])) != 4
        or not all(x.get("pass") is True for x in b["checks"])
        for b in bs
    ):
        raise ValueError("P9B real checkpoint GPU sample/accept/residual parity incomplete")
    if {b.get("proposal_depth") for b in bs} != {2, 10}:
        raise ValueError("P9B requires sampled logits from BOTH depths 2 and 10")
    if (
        baseline.get("kind") != "baseline_full_predictor_graph_on"
        or baseline.get("frames", 0) < 8
        or baseline.get("graph_replays") != baseline.get("frames")
    ):
        raise ValueError("P5 baseline was not true Graph ON replay")
    sha_values = {a.get("source_sha")} | {b.get("source_sha") for b in bs}
    if len(sha_values) != 1 or None in sha_values or "UNATTESTED" in sha_values:
        raise ValueError("P9A/P9B probes must share one fixed source revision")
    if len({a.get("gpu"), *[b.get("gpu") for b in bs], baseline.get("device")}) != 1:
        raise ValueError("P9A/B and full Graph ON baseline GPU name mismatch")
    if baseline.get("source_sha") != next(iter(sha_values)):
        raise ValueError(
            "P9 requires SAME source SHA for full Graph ON and P9A/P9B; "
            "use the fresh P9 baseline, not prior P5 reports"
        )

    costs = a.get("cost_by_dynamic_stage_sequence", [])
    if len(costs) != 6:
        raise ValueError("P9A missing all six nested dynamic replay cost cells")
    points = [
        (int(x["model_calls"]), check_positive(x["cuda_event_ms_per_graph"]))
        for x in costs
    ]
    overhead, per_round = linear_cost(points)
    sampler_ms = median(
        check_positive(b.get("gpu_sampler_with_independent_rng_graph_ms"))
        for b in bs
    )
    full_ms = check_positive(baseline.get("median_cuda_event_ms"))
    p9b_baseline_ms = None
    baseline_drift_fraction = None
    if p9b_baseline_dir is not None:
        other_dir = p9b_baseline_dir.expanduser().resolve(strict=True)
        other_files = list(other_dir.glob("baseline_graph_pid*.json"))
        if len(other_files) != 1:
            raise RuntimeError("P9B fresh server must have exactly one Graph ON baseline")
        other = json.loads(other_files[0].read_text())
        if (
            other.get("source_sha") != next(iter(sha_values))
            or other.get("device") != baseline.get("device")
            or other.get("kind") != "baseline_full_predictor_graph_on"
            or other.get("frames", 0) < 8
            or other.get("graph_replays") != other.get("frames")
        ):
            raise ValueError("P9B Graph ON baseline source/GPU/replay mismatch")
        p9b_baseline_ms = check_positive(other.get("median_cuda_event_ms"))
        baseline_drift_fraction = abs(p9b_baseline_ms - full_ms) / full_ms
        if baseline_drift_fraction > .10:
            raise RuntimeError(
                f"P9A vs P9B fresh-process Graph ON baseline drift "
                f"{baseline_drift_fraction:.1%} exceeds 10%; no cost selection"
            )
    try:
        cohort_path = acceptance_report.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"P2 acceptance report missing: {acceptance_report}. "
            "If the old p2-report-004 directory was lost, explicitly use "
            "--p2-acceptance-report "
            "experiments/qwen3_tts_cp_spec_k2/data/"
            "p2_720_historical_aggregate_reconstructed.json "
            "(prior reported counts only, not original per-frame evidence)."
        ) from exc
    cohort_raw = json.loads(cohort_path.read_text())
    acceptance = validate_acceptance(cohort_raw)

    # The original P2 -004 report may be lost after node cleanup. A checked-in,
    # explicitly attributed historical count aggregate can support this cost
    # *screen* only; it is NOT the lost per-frame trace or a new parity proof.
    provenance = cohort_raw.get("provenance")
    reconstructed = cohort_raw.get("schema") == (
        "qwen3tts-cpspec-p2-conversation-reconstructed-aggregate-v1"
    )
    if reconstructed:
        if (
            cohort_raw.get("status") !=
            "P2_RECONSTRUCTED_HISTORICAL_AGGREGATE_NOT_ORIGINAL_TRACE_OR_PARITY_EVIDENCE"
            or not isinstance(provenance, dict)
            or provenance.get("kind") !=
            "reconstructed_from_user_shared_prior_p2_720_frame_aggregate_in_chat"
            or provenance.get("is_original_report") is not False
            or provenance.get("is_raw_trace") is not False
            or sum(row.get("accepted", -1) for row in acceptance["depths"])
            != acceptance["accepted"]
            or sum(row.get("rejected", -1) for row in acceptance["depths"])
            != acceptance["rejected"]
            or sum(row.get("bonus_skipped_proposals", -1)
                   for row in acceptance["depths"]) != acceptance["bonus"]
            or any(
                row["proposals"] + row["bonus_skipped_proposals"] != acceptance["frames"]
                for row in acceptance["depths"]
            )
        ):
            raise ValueError(
                "Historical P2 reconstruction lacks provenance or fails per-depth census"
            )
    provenance_label = (
        "RECONSTRUCTED_PRIOR_USER_REPORTED_AGGREGATE_NOT_RAW_FRAMES"
        if reconstructed else
        "SUPPLIED_P2_REPORT_SOURCE_NOT_RECONSTRUCTED"
    )
    avg_rounds = acceptance["proposals"] / acceptance["frames"]
    model_screen = overhead + avg_rounds * per_round
    sample_screen = avg_rounds * sampler_ms
    combined_screen = model_screen + sample_screen
    if model_screen <= 0 or sample_screen <= 0:
        raise ValueError("invalid component budget after extrapolation")
    remaining = full_ms - combined_screen
    result = {
        "schema": SCHEMA,
        "status": POSSIBLE if remaining > 0 else NO_GAIN,
        "p9_source_sha": next(iter(sha_values)),
        "p5_baseline_sha": baseline.get("source_sha"),
        "p2_acceptance_sha": acceptance.get("source_sha"),
        "p2_acceptance_provenance": provenance_label,
        "p2_acceptance_reconstructed": reconstructed,
        "p2_aggregate_input": str(cohort_path),
        "source_locked_p9_components": True,
        "gpu": a["gpu"],
        "real_predictor_layers": a["predictor_layers"],
        "baseline_full_predictor_graph_on_ms_per_frame": full_ms,
        "p9b_second_process_graph_on_baseline_ms_per_frame": p9b_baseline_ms,
        "p9a_vs_p9b_baseline_drift_fraction": baseline_drift_fraction,
        "historical_accepted_proposals_per_frame": avg_rounds,
        "p9a_nested_control_intercept_ms": overhead,
        "p9a_model_round_slope_ms": per_round,
        "p9a_four_depth_model_only_extrapolated_ms_per_frame": model_screen,
        "p9b_private_rng_gpu_sampling_ms_per_round": sampler_ms,
        "p9b_gpu_sample_extrapolated_ms_per_frame": sample_screen,
        "separate_graph_component_sum_ms_per_frame": combined_screen,
        "remaining_baseline_budget_ms_per_frame": remaining,
        "optimistic_baseline_over_components_ratio": full_ms / combined_screen,
        "p9a_measured_costs": costs,
        "p9b_real_logits_sample_costs": [
            {
                "depth": b["proposal_depth"],
                "gpu_sample_graph_ms": b["gpu_sampler_with_independent_rng_graph_ms"],
                "theoretical_overlap": b["overlap"],
                "p_distribution_max_abs": b["distribution_p_max_abs"],
                "q_distribution_max_abs": b["distribution_q_max_abs"],
            }
            for b in bs
        ],
        "limits": [
            "THIS IS NOT AN ACTUAL K2 SPEEDUP TEST: the candidate has not replaced serving.",
            "P9A four off-policy spaced offsets (2/6/10/14) extrapolate to ~9.7 full rounds.",
            "P9B is an independent graph using real p/q logits, NOT linked to model output or KV.",
            "Summing two graph launch envelopes can double-count or misrepresent shared control.",
            "The K2 initial pair, codebook heads, bonus, projected embedding and dynamic KV ownership are not included.",
            "P9A/P9B/P9 Graph ON baseline sources are matched; historical P2 acceptance cohort differs.",
            "If P2 source is reconstructed, only reported aggregate counts are restored; original per-frame traces/parity proof do NOT exist in this input.",
            "Actual device RNG distribution is compared only to PyTorch p/q, not native Omni fused sampler.",
            "A positive screen justifies integration; only profiler-OFF full-HTTP Graph-ON ABBA can demonstrate speedup.",
            "A negative component screen is a stop signal for naive four-way branching, not a proof all kernels are hopeless.",
        ],
    }
    return result


def write_report(out: Path, report: dict) -> Path:
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Omni Qwen3-TTS P9 — H200 real model + real p/q GPU sampler cost screen",
        "", "**" + report["status"] + "**", "",
        f"Source SHA: {report['p9_source_sha']}",
        f"Baseline SHA: {report['p5_baseline_sha']}",
        f"P2 acceptance SHA: {report['p2_acceptance_sha']}",
        f"P2 acceptance provenance: {report['p2_acceptance_provenance']}",
        f"P2 cohort reconstructed (not raw trace): {report['p2_acceptance_reconstructed']}",
        f"Real predictor layers: {report['real_predictor_layers']}",
        f"GPU: {report['gpu']}",
        "",
        "| Cost comparison | CUDA Graph event ms/frame |",
        "|---|---:|",
        f"| Existing entire Predictor Graph ON baseline | {report['baseline_full_predictor_graph_on_ms_per_frame']:.5f} |",
        f"| P9B fresh-process Graph ON baseline | {report['p9b_second_process_graph_on_baseline_ms_per_frame']:.5f} |" if report["p9b_second_process_graph_on_baseline_ms_per_frame"] is not None else "| P9B fresh-process Graph ON baseline | — |",
        f"| Four-offset nested verifier extrapolation | {report['p9a_four_depth_model_only_extrapolated_ms_per_frame']:.5f} |",
        f"| Device proposal+accept+residual RNG extrapolation | {report['p9b_gpu_sample_extrapolated_ms_per_frame']:.5f} |",
        f"| Component sum (not full K2) | {report['separate_graph_component_sum_ms_per_frame']:.5f} |",
        f"| Remaining budget (can be negative) | {report['remaining_baseline_budget_ms_per_frame']:+.5f} |",
        "",
        "Optimistic baseline/component quotient: "
        f"{report['optimistic_baseline_over_components_ratio']:.3f}x",
        "",
        "## Real H200 device graph costs", "",
        "| P9A case | Real model calls | CUDA event ms / graph |",
        "|---|---:|---:|",
    ]
    for item in report["p9a_measured_costs"]:
        lines.append(
            f"| {item['case']} | {item['model_calls']} | "
            f"{item['cuda_event_ms_per_graph']:.5f} |"
        )
    lines += [
        "", "| P9B on-policy depth | Full GPU sample/accept/residual graph ms | Theoretical overlap |",
        "|---:|---:|---:|",
    ]
    for item in report["p9b_real_logits_sample_costs"]:
        lines.append(
            f"| {item['depth']} | {item['gpu_sample_graph_ms']:.5f} | "
            f"{item['theoretical_overlap']:.3%} |"
        )
    lines += ["", "## Required limitations", ""] + [
        "- " + limitation for limitation in report["limits"]
    ]
    file = out / "summary.md"
    file.write_text("\n".join(lines) + "\n")
    return file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p9a-dir", type=Path, required=True)
    p.add_argument("--p9b-dir", type=Path, required=True)
    p.add_argument(
        "--baseline-graph-dir", "--p5-baseline-graph-dir",
        dest="p5_baseline_graph_dir", type=Path, required=True,
        help="Fresh same-SHA optimized Graph-ON baseline captured by P9 runner",
    )
    p.add_argument("--p2-acceptance-report", type=Path, required=True)
    p.add_argument("--p9b-baseline-graph-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = analyze(
        args.p9a_dir, args.p9b_dir,
        args.p5_baseline_graph_dir, args.p2_acceptance_report,
        p9b_baseline_dir=args.p9b_baseline_graph_dir,
    )
    file = write_report(args.out, result)
    print(f"P9_COMPONENT_SCREEN={result['status']} RESULT={file}")
    print(
        f"BASELINE_GRAPH_ON_MS={result['baseline_full_predictor_graph_on_ms_per_frame']:.5f} "
        f"COMPONENT_ESTIMATE_MS={result['separate_graph_component_sum_ms_per_frame']:.5f} "
        f"REMAINING_BUDGET_MS={result['remaining_baseline_budget_ms_per_frame']:.5f}"
    )


if __name__ == "__main__":
    main()

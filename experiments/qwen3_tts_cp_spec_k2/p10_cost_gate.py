# SPDX-License-Identifier: Apache-2.0
"""P10 real checkpoint dynamic-KV + sparse-GPU-sampling feasibility cost gate.

Uses SAME SOURCE SHA and device from two *fresh* C1 workers:
 - P10A real 5-layer Predictor single device-offset graph, compared to
   four separately captured fixed-depth verifier graphs from same worker;
 - P10B GPU two-topk + fused top-50 accept/residual vs P9B dense reference
   graphs captured on real p/q logits, with private independent RNG;
 - real optimized Graph-ON baseline of each fresh C1 worker.
It weights P10A/B per-round costs by the old source-attributed P2 aggregate.
This is an **optimistic component feasibility screen, NEVER serving QPS**.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from experiments.qwen3_tts_cp_spec_k2.p4_cost_screen import validate_acceptance
from sglang_omni.models.qwen3_tts.code_predictor_spec_p10a_dynamic_kv import (
    SCHEMA as A_SCHEMA,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p10b_sparse_sampler import (
    SCHEMA as B_SCHEMA,
)

SCHEMA = "qwen3tts-cpspec-p10-source-locked-component-cost-gate-v1"
PASS = "P10_OPTIMISTIC_COMPONENT_BUDGET_HAS_HEADROOM_NOT_SERVING_SPEEDUP"
STOP = "P10_COMPONENT_BUDGET_TOO_TIGHT_NO_FULL_K2_INTEGRATION_SELECTION"


def positive(x):
    if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"expected finite positive H200 event ms, got {x!r}")
    return float(x)


def only_file(folder: Path, pattern: str) -> dict:
    folder = folder.expanduser().resolve(strict=True)
    items = sorted(folder.glob(pattern))
    if len(items) != 1:
        raise ValueError(f"expected exactly 1 evidence {pattern} in {folder}, got {len(items)}")
    return json.loads(items[0].read_text())


def analyze(
    a_dir: Path, b_dir: Path,
    a_baseline_dir: Path, b_baseline_dir: Path,
    acceptance_file: Path,
) -> dict:
    a = only_file(a_dir, "dynamic_kv_pid*_frame*.json")
    b_dir = b_dir.expanduser().resolve(strict=True)
    b_paths = sorted(b_dir.glob("sparse_pid*_frame*_depth*.json"))
    if len(b_paths) != 2:
        raise ValueError(f"P10B requires two real-logit depths 2 and 10, got {len(b_paths)}")
    b = [json.loads(path.read_text()) for path in b_paths]
    base_a = only_file(a_baseline_dir, "baseline_graph_pid*.json")
    base_b = only_file(b_baseline_dir, "baseline_graph_pid*.json")

    if (
        a.get("schema") != A_SCHEMA
        or not a.get("status", "").startswith(
            "P10A_ONE_GRAPH_GPU_DYNAMIC_KV_FUSED_PREDICTOR_PARITY_PASS"
        )
        or a.get("predictor_layers") != 5
        or a.get("backend") != "same_production_fused_predictor_layers"
        or [row.get("depth") for row in a.get("cases", [])] != [2, 6, 10, 14]
        or not all(row.get("pass") is True for row in a["cases"])
    ):
        raise ValueError("P10A five-layer single-graph dynamic KV parity evidence invalid")
    if any(
        item.get("schema") != B_SCHEMA
        or not item.get("status", "").startswith(
            "P10B_REAL_PQ_SPARSE_TOPK_SPEC_SAMPLER_PARITY_PASS"
        )
        or item.get("top_k") != 50
        or item.get("vocab") != 2048
        or item.get("private_rng_replay_unique") is not True
        or item.get("input_logits_immutable") is not True
        or len(item.get("checks", [])) != 4
        or not all(cell.get("pass") is True for cell in item["checks"])
        or item.get("p_max_abs", math.inf) > 1e-6
        or item.get("q_max_abs", math.inf) > 1e-6
        for item in b
    ):
        raise ValueError("P10B real logits sparse sampler / RNG / distribution parity invalid")
    if {item.get("proposal_depth") for item in b} != {2, 10}:
        raise ValueError("P10B must have one on-policy p/q profile each for depths 2 and 10")
    sha = a.get("source_sha")
    if not isinstance(sha, str) or len(sha) != 40 or sha == "UNATTESTED":
        raise ValueError("P10A source SHA not pinned")
    if any(item.get("source_sha") != sha for item in b):
        raise ValueError("P10A/B source SHA mismatch; rerun from same commit")
    gpu = a.get("gpu")
    if not gpu or any(item.get("gpu") != gpu for item in b):
        raise ValueError("P10A/B measured on different GPUs")
    baselines = (base_a, base_b)
    for row in baselines:
        if (
            row.get("kind") != "baseline_full_predictor_graph_on"
            or row.get("frames", 0) < 8
            or row.get("graph_replays") != row.get("frames")
            or row.get("source_sha") != sha
            or row.get("device") != gpu
        ):
            raise ValueError("P10 requires fresh SAME-SHA real Graph-ON baseline")
    baseline_ms = positive(base_a.get("median_cuda_event_ms"))
    other_base_ms = positive(base_b.get("median_cuda_event_ms"))
    baseline_drift = abs(baseline_ms - other_base_ms) / baseline_ms
    if baseline_drift > .10:
        raise RuntimeError(
            f"P10 fresh-process native Graph ON baseline drift {baseline_drift:.2%} >10%"
        )

    raw = json.loads(acceptance_file.expanduser().resolve(strict=True).read_text())
    acceptance = validate_acceptance(raw)
    if (
        raw.get("schema") != "qwen3tts-cpspec-p2-conversation-reconstructed-aggregate-v1"
        or raw.get("provenance", {}).get("is_raw_trace") is not False
        or raw.get("provenance", {}).get("is_original_report") is not False
        or raw.get("source_sha") != "398297b1d4d2a44d10d1056da5607a7d4ac08e41"
        or sum(row.get("accepted", 0) for row in acceptance["depths"]) != acceptance["accepted"]
        or sum(row.get("rejected", 0) for row in acceptance["depths"]) != acceptance["rejected"]
        or sum(row.get("bonus_skipped_proposals", 0) for row in acceptance["depths"]) != acceptance["bonus"]
        or any(
            row["proposals"] + row.get("bonus_skipped_proposals", -1) != acceptance["frames"]
            for row in acceptance["depths"]
        )
    ):
        raise ValueError("P10 requires attributed historical 720-frame P2 aggregate census")
    avg_rounds = acceptance["proposals"] / acceptance["frames"]

    a_costs = [positive(x["dynamic_cuda_graph_ms"]) for x in a["cases"]]
    static_costs = [positive(x["fixed_cuda_graph_ms"]) for x in a["cases"]]
    sparse_costs = [positive(x["sparse_fused_cuda_graph_ms"]) for x in b]
    dense_costs = [positive(x["p9b_dense_reference_cuda_graph_ms"]) for x in b]
    dynamic_median, static_median = median(a_costs), median(static_costs)
    sparse_median, dense_median = median(sparse_costs), median(dense_costs)
    model_ms = avg_rounds * dynamic_median
    sampler_ms = avg_rounds * sparse_median
    total_components = model_ms + sampler_ms
    remaining = baseline_ms - total_components
    # 0.30ms/frame is an engineering *budget reserve*, not a measured
    # guarantee that remaining target heads, first pair and bonus fit.
    reserve_ms = .30
    result = {
        "schema": SCHEMA,
        "status": PASS if remaining >= reserve_ms else STOP,
        "source_sha": sha,
        "gpu": gpu,
        "predictor_layers": 5,
        "historical_p2_aggregate_reconstructed": True,
        "historical_p2_source_sha": acceptance["source_sha"],
        "historical_rounds_per_frame": avg_rounds,
        "real_full_graph_on_baseline_ms": baseline_ms,
        "second_real_full_graph_on_baseline_ms": other_base_ms,
        "baseline_cross_process_drift_fraction": baseline_drift,
        "four_depths": [x["depth"] for x in a["cases"]],
        "p10a_per_depth_dynamic_vs_fixed": a["cases"],
        "p10a_dynamic_median_ms_per_round": dynamic_median,
        "p10a_fixed_median_ms_per_round": static_median,
        "p10a_dynamic_over_fixed_ratio": dynamic_median / static_median,
        "p10b_real_depths": [x["proposal_depth"] for x in b],
        "p10b_sparse_sampler_median_ms_per_round": sparse_median,
        "p10b_dense_reference_median_ms_per_round": dense_median,
        "p10b_sparse_over_dense_ratio": sparse_median / dense_median,
        "p10b_state_specific": [
            {
                "depth": row["proposal_depth"],
                "overlap": row["overlap"],
                "sparse_ms": row["sparse_fused_cuda_graph_ms"],
                "dense_ms": row["p9b_dense_reference_cuda_graph_ms"],
                "p_max_abs": row["p_max_abs"],
                "q_max_abs": row["q_max_abs"],
            }
            for row in b
        ],
        "estimated_real_dynamic_predictor_ms_per_frame": model_ms,
        "estimated_sparse_gpu_sampling_ms_per_frame": sampler_ms,
        "optimistic_component_sum_ms_per_frame": total_components,
        "remaining_graph_on_budget_ms_per_frame": remaining,
        "reserve_required_ms_per_frame": reserve_ms,
        "optimistic_baseline_over_components_ratio": baseline_ms / total_components,
        "limits": [
            "A positive reserve DOES NOT prove full GPU-resident K2 or serving speedup.",
            "P10A tests on-policy inputs at depth14, earlier offsets off-policy.",
            "P10A exactly matches optimized fused backend; CuTeDSL backend needs separate dynamic kernel.",
            "P10B topk remains PyTorch; post-topk accept/correction is Triton fused.",
            "P10B private RNG does NOT prove native fused sampler RNG equivalence.",
            "Independent graph launch envelopes may not add in the real graph.",
            "No initial pair, next-head, bonus, embedding, real depth-KV transitions or HTTP speedup.",
            "Historical 720-frame acceptance reconstruction, not original per-frame trace; prior source SHA differs.",
            "Graph-ON baseline is the real production performance target; profiler-OFF E2E ABBA still required.",
        ],
    }
    return result


def write(out: Path, report: dict):
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# P10 H200 — Real dynamic KV + sparse fused GPU SpecDec cost gate",
        "", "**" + report["status"] + "**", "",
        f"Source: {report['source_sha']}",
        f"GPU: {report['gpu']}",
        f"Historical P2 SHA (RECONSTRUCTED AGGREGATE, NOT RAW): {report['historical_p2_source_sha']}",
        f"Historical rounds/frame: {report['historical_rounds_per_frame']:.5f}",
        "",
        "| Cell | CUDA Event ms |",
        "|---|---:|",
        f"| Native full Graph-ON Predictor per frame | {report['real_full_graph_on_baseline_ms']:.5f} |",
        f"| Native baseline in second fresh worker | {report['second_real_full_graph_on_baseline_ms']:.5f} |",
        f"| P10A dynamic KV real stack per round (median) | {report['p10a_dynamic_median_ms_per_round']:.5f} |",
        f"| P10A fixed KV real stack per round (median) | {report['p10a_fixed_median_ms_per_round']:.5f} |",
        f"| P10B sparse topk+fused SpecDec sample per round | {report['p10b_sparse_sampler_median_ms_per_round']:.5f} |",
        f"| P10B original dense sample per round | {report['p10b_dense_reference_median_ms_per_round']:.5f} |",
        f"| P10A dynamic predictor extrapolated per frame | {report['estimated_real_dynamic_predictor_ms_per_frame']:.5f} |",
        f"| P10B sparse sampler extrapolated per frame | {report['estimated_sparse_gpu_sampling_ms_per_frame']:.5f} |",
        f"| Component sum (NOT E2E) | {report['optimistic_component_sum_ms_per_frame']:.5f} |",
        f"| Remaining budget | {report['remaining_graph_on_budget_ms_per_frame']:+.5f} |",
        f"| Engineering reserve criterion | {report['reserve_required_ms_per_frame']:.5f} |",
        "",
        f"Dynamic-vs-fixed verifier ratio: {report['p10a_dynamic_over_fixed_ratio']:.3f}x",
        f"Sparse-vs-dense sampler ratio: {report['p10b_sparse_over_dense_ratio']:.3f}x",
        "",
        "## Per-offset real 5-layer dynamic verifier", "",
        "| GPU KV offset | Dynamic ms | Same-worker fixed ms | Parity |",
        "|---:|---:|---:|---|",
    ]
    for row in report["p10a_per_depth_dynamic_vs_fixed"]:
        lines.append(
            f"| {row['depth']} | {row['dynamic_cuda_graph_ms']:.5f} | "
            f"{row['fixed_cuda_graph_ms']:.5f} | {row['pass']} |"
        )
    lines += ["", "## Real on-policy p/q sparse sampler", "",
              "| Depth | Sparse ms | Dense P9B ms | p/q overlap |",
              "|---:|---:|---:|---:|"]
    for row in report["p10b_state_specific"]:
        lines.append(
            f"| {row['depth']} | {row['sparse_ms']:.5f} | "
            f"{row['dense_ms']:.5f} | {row['overlap']:.2%} |"
        )
    lines += ["", "## Non-claims / caveats", ""] + [
        "- " + x for x in report["limits"]
    ]
    summary = out / "summary.md"
    summary.write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p10a-dir", type=Path, required=True)
    parser.add_argument("--p10b-dir", type=Path, required=True)
    parser.add_argument("--p10a-baseline", type=Path, required=True)
    parser.add_argument("--p10b-baseline", type=Path, required=True)
    parser.add_argument("--p2-aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.p10a_dir, args.p10b_dir, args.p10a_baseline,
        args.p10b_baseline, args.p2_aggregate,
    )
    out = write(args.out, result)
    print(f"P10_COST_GATE={result['status']} REPORT={out}")
    print(
        f"FULL_GRAPH_ON_MS={result['real_full_graph_on_baseline_ms']:.5f} "
        f"DYNAMIC_KV_PLUS_SPARSE_MS={result['optimistic_component_sum_ms_per_frame']:.5f} "
        f"REMAINING_MS={result['remaining_graph_on_budget_ms_per_frame']:.5f}"
    )


if __name__ == "__main__":
    main()

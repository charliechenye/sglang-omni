# SPDX-License-Identifier: Apache-2.0
"""P23 pre-implementation gate: evidence, required speedup and GPU loop blockers.

This is NOT a full-frame GPU K2 implementation. Reject unproven
optimization promises before committing engineering time or shipping
an unsafe CUDA Graph. Run only after P22 profiling and P21 real ABBA.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from pathlib import Path

GATE = "P23_FULL_GPU_K2_IMPLEMENTATION_REQUIRED_NOT_PRODUCTION_GO"


def required_latency_reduction(base_s: float, candidate_s: float,
                               target_win: float = 0.05) -> float:
    """What fraction of current candidate latency must disappear?"""
    if (not all(math.isfinite(x) and x > 0 for x in (base_s, candidate_s))
            or not 0 < target_win < 1):
        raise ValueError("P23_INVALID_LATENCY_OR_TARGET")
    return max(0.0, 1.0 - (1.0 - target_win) * base_s / candidate_s)


def source_blockers(file_text: str) -> list[str]:
    """Reject host-driven K2 loop until a separate device-only path exists."""
    ast.parse(file_text)
    contract = {
        "HOST_DYNAMIC_DEPTH_READ": 'step["committed_count"].item()',
        "HOST_CODE_VALIDITY_SYNC": "(candidate_codes < 0).any().item()",
        "PYTHON_DYNAMIC_DEPTH_WHILE": "while depth < 15:",
        "PYTHON_HEAD_BANK_INDEX_BY_DEPTH": "head_bank=banks[depth]",
        "PYTHON_TERMINAL_BRANCH": "if depth == 14:",
        "SYNCHRONOUS_HOST_VALIDATION": "torch.equal(codes[:, 0], layer0_codes[:, 0])",
        "PRIVATE_DRAFT_RNG_NOT_NATIVE_PARITY": "_request_seed(native_seed, semantic_pos)",
        "MISSING_GRAPH_CAPTURE_FOR_FULL_14_DEPTH": "p22_capture_begin(model)",
    }
    return [
        issue for issue, needle in contract.items() if needle in file_text
    ]


def analyze(p21: dict, p22: dict, serving_source: str) -> dict:
    if (p21.get("status") !=
            "P21_REAL_SERVING_C1_ABBA_SCREEN_NOT_QUALITY_OR_PRODUCTION_GO"
            or p21.get("quality_certificate") !=
            "NONE_NOT_ASR_OR_SPEAKER_EQUIVALENCE"
            or p22.get("status") !=
            "P22_ATTRIBUTION_COMPLETE_NOT_PERFORMANCE_BENCHMARK"
            or not p21.get("within_arm_wav_hash_repeatable")
            or not p21.get("paired")):
        raise RuntimeError("P23_MISSING_P21_P22_CAUSAL_SCOPE_GATES")
    if not p21.get("source_sha") or not p22.get("source_sha"):
        raise RuntimeError("P23_MISSING_SOURCE_SHA")
    if not p22.get("candidate_frames", 0) or not p22.get("k2_rounds", 0):
        raise RuntimeError("P23_MISSING_REAL_K2_EVIDENCE")
    per_case = []
    for row in p21["paired"]:
        a = row.get("a_mean_s")
        b = row.get("b_mean_s")
        if (not row.get("a_hash_repeatable")
                or not row.get("b_hash_repeatable")):
            raise RuntimeError("P23_P21_WITHIN_ARM_WAV_HASH_DRIFT")
        cut = required_latency_reduction(a, b)
        per_case.append({
            "case_id": row["case_id"],
            "native_a_mean_s": a,
            "current_b_mean_s": b,
            "target_b_s_for_5pct_win": 0.95 * a,
            "required_b_reduction_fraction": cut,
        })
    median = statistics.median(
        item["required_b_reduction_fraction"] for item in per_case
    )
    blockers = source_blockers(serving_source)
    if not blockers:
        raise RuntimeError("P23_EXPECTED_HOST_DRIVEN_SOURCE_NOT_FOUND")
    return {
        "status": GATE,
        "p21_source_sha": p21["source_sha"],
        "p22_source_sha": p22["source_sha"],
        "sources_are_distinct": p21["source_sha"] != p22["source_sha"],
        "note": (
            "P21 unprofiled latency sets the improvement hurdle. "
            "P22 profiled host timers cannot be subtracted from P21 latency "
            "or treated as GPU kernel time. This is not a same-SHA ABBA."
        ),
        "paired_required_reductions": per_case,
        "median_required_b_reduction_fraction_for_5pct_win": median,
        "p22_measured_k2_rounds_per_frame": p22["rounds_per_frame"],
        "p22_cpu_host_sync_items_per_frame_ms": (
            p22["cpu_host_sync_items_per_frame_ms"]
        ),
        "p22_nsys_report_finalized": p22["nsys_report_finalized"],
        "host_driven_current_source_blockers": blockers,
        "graph_correctness_gates_needed": [
            "All 14 residual-depth transitions stay on GPU, with valid next depth",
            "GPU-indexed distinct LM heads; no Python-depth-specific capture",
            "Device KV END and physical prefix/suffix ownership parity",
            "Conditional commit and native target distribution under sampled RNG",
            "Capture-safe fixed-address buffers and no per-round .item()",
            "E2E real HTTP A-B-B-A and independent ASR/audio quality check",
            "C8/C16, memory, race and startup regression only after C1 GO",
        ],
        "decision": (
            "NO_P23_FULL_GRAPH_CAPTURE_OR_PRODUCTION_MERGE_WITHOUT_DEVICE_LOOP "
            "AND_NATIVE_SAMPLER_DISTRIBUTION_PARITY"
        ),
    }


def markdown(report: dict) -> str:
    lines = [
        "# P23 full-frame GPU K2: feasibility gate",
        "",
        "**This is a gate, not an implementation or production speedup.**",
        "",
        f"P21 unprofiled source: `{report['p21_source_sha']}`",
        f"P22 profiled source: `{report['p22_source_sha']}`",
        "",
        "| Real ABBA case | Native A mean | Host-driven B mean | B target (5% faster) | Required B reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["paired_required_reductions"]:
        lines.append(
            f"| {row['case_id']} | {row['native_a_mean_s']:.3f}s "
            f"| {row['current_b_mean_s']:.3f}s "
            f"| {row['target_b_s_for_5pct_win']:.3f}s "
            f"| {row['required_b_reduction_fraction']:.1%} |"
        )
    lines += [
        "",
        "Median required B latency reduction to win 5%: **"
        + f"{report['median_required_b_reduction_fraction_for_5pct_win']:.1%}**.",
        "",
        "P22 real K2 rounds/frame: **"
        + f"{report['p22_measured_k2_rounds_per_frame']:.3f}**.",
        "P22 HOST synchronization region: "
        + f"{report['p22_cpu_host_sync_items_per_frame_ms']:.3f} ms/frame "
        + "(profiler ON, CPU wall only).",
        "Nsight report verified: "
        + str(report["p22_nsys_report_finalized"]) + ".",
        "",
        "## Current source-level device-loop blockers",
    ]
    lines += [f"- {block}" for block in report["host_driven_current_source_blockers"]]
    lines += [
        "",
        "## Required before implementing a deployable full-graph K2",
    ]
    lines += [f"- {item}" for item in report["graph_correctness_gates_needed"]]
    lines += ["", "**Decision: " + report["decision"] + "**", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p21-report", type=Path, required=True)
    p.add_argument("--p22-report", type=Path, required=True)
    p.add_argument("--p21-candidate-source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError("P23_REFUSE_OVERWRITE")
    p21 = json.loads(a.p21_report.read_text())
    p22 = json.loads(a.p22_report.read_text())
    report = analyze(
        p21, p22, a.p21_candidate_source.read_text()
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "summary.md").write_text(markdown(report))
    print("P23_GATE=" + report["status"])
    print("P23_REPORT=" + str(out / "summary.md"))


if __name__ == "__main__":
    main()

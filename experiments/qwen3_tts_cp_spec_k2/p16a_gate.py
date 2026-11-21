# SPDX-License-Identifier: Apache-2.0
"""P16A source-locked native-Graph ABBA + 4-branch event-census gate."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from experiments.qwen3_tts_cp_spec_k2 import p15_two_round_gate as p15
from sglang_omni.models.qwen3_tts.code_predictor_spec_p16a_profile import (
    SCHEMA, STATUS, ARMS, BRANCHES, expected_bonus_markers,
)

PENDING = "P16A_FOUR_BRANCH_PAIRED_EVENT_TIMING_PASS_NSYS_NODE_TRACE_PENDING"
FAIL = "P16A_PAIRED_EVENT_OR_SOURCE_VALIDATION_FAIL"


def validate_profile(row: dict, profile: dict) -> None:
    """Fail closed on missing matched-graph, branch or source evidence."""
    if (
        row.get("status") != p15.PASS
        or profile.get("schema") != SCHEMA
        or profile.get("status") != STATUS
        or profile.get("all_16_matched_against_independent_P15_oracle") is not True
        or profile.get("graph_count_total") != 4
        or profile.get("timed_graphs_unmarked") is not True
        or profile.get("trace_graphs_marked_not_timed") is not True
        or profile.get("nsys_node_trace_verified") is not False
        or profile.get("bonus_head_kernel_skip_proven") is not False
        or profile.get("native_seeded_rng_verified") is not False
        or set(profile.get("cases", {})) != set(BRANCHES)
        or set(profile.get("branch_case_indices", {})) != set(BRANCHES)
        or len(profile.get("trace_ranges", [])) != 8
    ):
        raise ValueError("P16A incomplete or fabricated matched Graph evidence")
    expected_index = {}
    for case in row["cases"]:
        expected_index.setdefault(case["transition"], case["case_index"])
    if expected_index != profile["branch_case_indices"]:
        raise ValueError("P16A branch indices do not match P15 oracle")
    expected_ranges = set()
    for label in BRANCHES:
        cell = profile["cases"][label]
        if (
            cell.get("case_index") != expected_index[label]
            or cell.get("uniforms") != row["cases"][expected_index[label]]["uniforms"]
            or cell.get("accept_count") != label.count("A")
            or cell.get("rejection_count") != label.count("R")
            or set(cell.get("cuda_event_ms_blocks", {})) != set(ARMS)
            or set(cell.get("median_cuda_event_ms", {})) != set(ARMS)
        ):
            raise ValueError(f"P16A incorrect branch metric metadata: {label}")
        for arm in ARMS:
            blocks = cell["cuda_event_ms_blocks"][arm]
            med = cell["median_cuda_event_ms"][arm]
            if (
                not isinstance(blocks, list)
                or len(blocks) != 4
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v) or v <= 0
                    for v in blocks
                )
                or type(med) not in (int, float)
                or not math.isfinite(med) or med <= 0
                or med != statistics.median(blocks)
            ):
                raise ValueError(f"P16A invalid per-arm timing: {label}/{arm}")
            expected_ranges.add((f"P16A_CASE_{label}_{arm.upper()}", label, arm))
        actual = cell["unconditional_minus_conditional_ms"]
        expected = (
            cell["median_cuda_event_ms"]["unconditional"]
            - cell["median_cuda_event_ms"]["conditional"]
        )
        if not math.isfinite(actual) or actual != expected:
            raise ValueError("P16A unpaired or rounded performance delta")
    found_ranges = set()
    for trace in profile["trace_ranges"]:
        key = (trace.get("nvtx_range"),
               trace.get("transition"), trace.get("arm"))
        if (
            key not in expected_ranges or key in found_ranges
            or trace.get("marked_replays") != 1
            or trace.get("expected_round_start_marker_nodes") != 2
            or trace.get("expected_bonus_head_marker_nodes")
            != expected_bonus_markers(trace["transition"], trace["arm"])
        ):
            raise ValueError(f"P16A invalid/incomplete NVTX range {key}")
        found_ranges.add(key)
    if found_ranges != expected_ranges:
        raise ValueError("P16A missing NVTX replay ranges")


def summarize(result: dict) -> str:
    row = result["two_round_probe"]
    p = row["p16a"]
    lines = [
        "# P16A — Real two-round conditional bonus vs always-compute baseline",
        "",
        f"**{result['status']}**",
        "",
        f"Source SHA: {result['source_sha']}",
        f"GPU: {result['gpu']}; PID: {result['pid']}",
        "Native full Predictor Graph-ON baseline (context only): "
        f"{result['native_graph_ms_per_frame']:.5f} ms/frame",
        "",
        "| Transition | Case index | Conditional unmarked (ms) | "
        "Always-compute unmarked (ms) | Unconditional - Conditional (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("AA", "AR", "RA", "RR"):
        cell = p["cases"][label]
        med = cell["median_cuda_event_ms"]
        lines.append(
            f"| {label} | {cell['case_index']} | "
            f"{med['conditional']:.5f} | {med['unconditional']:.5f} | "
            f"{cell['unconditional_minus_conditional_ms']:+.5f} |"
        )
    lines += [
        "",
        "- All 16 cases: P15 independent CPU/eager correctness, matched "
        "unconditional/marked graph outputs, and full physical K/V.",
        "- Timing: four alternating-order paired event blocks per branch, "
        "16 replays/block; **unmarked graphs only**, not HTTP latency.",
        "- Nsight capture range: P16A_TRACE; eight named marked replays. "
        "Marker counts/head-kernel skip are **not** asserted until trace audit.",
        "- CUDA graph node tracing can be overhead-heavy; trace timings "
        "are never used as event-cost speedup estimates.",
        "- No native seeded RNG, 14-depth completion, audio quality "
        "or serving throughput equivalence.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    # Reuse P15's real ABBA HTTP+PID/GPU/source/correctness checks unchanged.
    result = p15.run(args)
    try:
        if result["status"] != p15.GO:
            raise ValueError(
                "P16A requires full P15 4/4 branch coverage and parity"
            )
        profile = result["two_round_probe"].get("p16a")
        if not isinstance(profile, dict):
            raise ValueError("P16A instrumented graph profile is missing")
        validate_profile(result["two_round_probe"], profile)
        result["status"] = PENDING
        summary = summarize(result)
        (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out / "p16a-summary.md").write_text(summary + "\n")
        print(f"P16A_STATUS={PENDING} REPORT={args.out/'p16a-summary.md'}")
    except BaseException as exc:
        result["status"] = FAIL
        result["p16a_error"] = str(exc)
        (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--base-arm", type=Path, required=True)
    parser.add_argument("--spec-arm", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--expect-source-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--model", default=p15.MODEL)
    parser.add_argument("--text", default=p15.TEXT)
    parser.add_argument("--voice", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.)
    args = parser.parse_args()
    if (
        not 256 <= args.max_new_tokens <= 2048
        or args.top_k != 50
        or not 0 < args.temperature <= 5
        or not 0 < args.top_p <= 1
    ):
        parser.error("P16A requires C1 sampled top-k50 and valid t/top-p")
    run(args)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""P16A Nsight node trace audit. Never infer kernel skipping from GPU booleans.

Run AFTER the nsys-profiled worker has terminated and the .nsys-rep exists.
Reads Nsight's nvtx_kern_sum CSV. Requires per-range *physical* Triton
round/bonus marker nodes from actual replay, verifies positive/negative
controls, and optionally checks a user-identified true LM-head kernel
signature. No regex -> only branch-marker proof, NOT bonus-head proof.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from experiments.qwen3_tts_cp_spec_k2.p16a_gate import (
    PENDING, validate_profile,
)
from experiments.qwen3_tts_cp_spec_k2.p15_two_round_gate import (
    PASS as P15_PASS,
)

MARKER_PASS = "P16A_NODE_TRACE_BRANCH_MARKERS_PASS_BONUS_HEAD_KERNEL_UNATTRIBUTED"
HEAD_PASS = "P16A_NODE_TRACE_BONUS_HEAD_KERNEL_SKIP_PROVEN_NOT_E2E_OR_SEEDED_RNG"
FAILED = "P16A_NSYS_NODE_KERNEL_EVIDENCE_INCOMPLETE"
ROUND_MARKER = "p16a_round_start_marker_kernel"
BONUS_MARKER = "p16a_bonus_head_marker_kernel"


def parse_nvtx_kernel_csv(data: str) -> dict[str, Counter]:
    """Return physical kernel counts per exact NVTX range, fail on ambiguity."""
    raw_rows = list(csv.reader(io.StringIO(data)))
    header = None
    index = None
    grouped: dict[str, Counter] = {}
    seen = 0
    for values in raw_rows:
        keys = [" ".join(v.strip().lower().split()) for v in values]
        if {"nvtx range", "kernel name", "kern inst"}.issubset(keys):
            if header is not None:
                raise ValueError("P16A multiple ambiguous nsys report headers")
            header = keys
            index = {name: header.index(name) for name in (
                "nvtx range", "kernel name", "kern inst",
            )}
            continue
        if index is None or len(values) != len(header):
            continue
        label = values[index["nvtx range"]].strip()
        if not label.startswith("P16A_CASE_"):
            continue
        name = values[index["kernel name"]].strip()
        if not name:
            raise ValueError("P16A empty kernel name")
        raw_count = values[index["kern inst"]].strip().replace(",", "")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError("P16A nsys malformed kernel instance count") from exc
        if count <= 0:
            raise ValueError("P16A nsys non-positive kernel count")
        grouped.setdefault(label, Counter())[name] += count
        seen += 1
    if index is None or seen == 0:
        raise ValueError(
            "P16A nsys NVTX-to-GPU-kernel mapping unavailable; "
            "do not substitute whole-graph or NVTX CPU-only counts"
        )
    return grouped


def _count_matching(kernels: Counter, pattern: str) -> int:
    return sum(n for kernel, n in kernels.items() if pattern in kernel)


def audit_markers(grouped: dict[str, Counter], traces: list[dict]) -> dict:
    expected = {t["nvtx_range"]: t for t in traces}
    if len(expected) != 8 or set(grouped) != set(expected):
        raise ValueError(
            f"P16A incomplete NVTX case ranges: "
            f"expected={sorted(expected)} observed={sorted(grouped)}"
        )
    results = {}
    for range_name, entry in expected.items():
        counts = grouped[range_name]
        round_count = _count_matching(counts, ROUND_MARKER)
        bonus_count = _count_matching(counts, BONUS_MARKER)
        expected_round = entry["expected_round_start_marker_nodes"]
        expected_bonus = entry["expected_bonus_head_marker_nodes"]
        if round_count != expected_round or bonus_count != expected_bonus:
            raise ValueError(
                f"P16A physical marker mismatch {range_name}: "
                f"round={round_count}/{expected_round} "
                f"bonus={bonus_count}/{expected_bonus}; "
                "GPU conditional node trace could be incomplete"
            )
        results[range_name] = {
            "physical_round_marker_kernels": round_count,
            "physical_bonus_marker_kernels": bonus_count,
            "total_kernel_nodes": sum(counts.values()),
            "distinct_kernel_names": len(counts),
        }
    if not (
        results["P16A_CASE_AA_CONDITIONAL"]["physical_bonus_marker_kernels"] == 2
        and results["P16A_CASE_RR_CONDITIONAL"]["physical_bonus_marker_kernels"] == 0
        and results["P16A_CASE_RR_UNCONDITIONAL"]["physical_bonus_marker_kernels"] == 2
    ):
        raise ValueError("P16A missing positive/negative bonus controls")
    return results


def candidate_signatures(grouped: dict[str, Counter]) -> list[dict]:
    """Find name patterns matching 0/1/1/2 skipped kernels across branches.

    These are ONLY candidates; some non-LM-head kernels can have the same
    delta. Manual identification of the actual LM head is still mandatory.
    """
    names = set().union(*(set(x) for x in grouped.values()))
    found = []
    for name in sorted(names):
        if "p16a_" in name.lower():
            continue
        delta = {}
        for label in ("AA", "AR", "RA", "RR"):
            a = grouped[f"P16A_CASE_{label}_CONDITIONAL"].get(name, 0)
            b = grouped[f"P16A_CASE_{label}_UNCONDITIONAL"].get(name, 0)
            delta[label] = b - a
        if delta["AA"] == 0 and delta["AR"] > 0 and (
            delta["RA"] == delta["AR"]
            and delta["RR"] == 2 * delta["AR"]
        ):
            found.append({
                "kernel_name": name,
                "extra_nodes_per_rejected_bonus": delta["AR"],
                "observed_deltas": delta,
            })
    return found


def verify_head_kernel_signature(
    grouped: dict[str, Counter], regex: str,
) -> dict:
    if not regex.strip():
        raise ValueError("P16A head kernel regex must be nonempty")
    pattern = re.compile(regex)
    matches = {
        name for count in grouped.values() for name in count
        if pattern.search(name)
    }
    if not matches:
        raise ValueError("P16A head kernel pattern matches no physical kernel")
    if any("p16a_" in name.lower() or "marker" in name.lower()
           for name in matches):
        raise ValueError("P16A head kernel pattern matched instrument markers")
    delta = {}
    counts = {}
    for label in ("AA", "AR", "RA", "RR"):
        conditional = sum(
            n for name, n in
            grouped[f"P16A_CASE_{label}_CONDITIONAL"].items()
            if pattern.search(name)
        )
        unconditional = sum(
            n for name, n in
            grouped[f"P16A_CASE_{label}_UNCONDITIONAL"].items()
            if pattern.search(name)
        )
        delta[label] = unconditional - conditional
        counts[label] = {
            "conditional": conditional,
            "unconditional": unconditional,
        }
    k = delta["AR"]
    if not (
        k >= 1
        and delta["AA"] == 0
        and delta["RA"] == k
        and delta["RR"] == 2 * k
    ):
        raise ValueError(
            "P16A bonus head CUDA kernel counts fail "
            f"AA=0 AR=1 RA=1 RR=2 rejection delta: {delta}"
        )
    return {
        "head_kernel_regex": regex,
        "matched_kernel_names": sorted(matches),
        "extra_kernel_nodes_per_rejected_bonus": k,
        "kernel_counts": counts,
        "delta_unconditional_minus_conditional": delta,
    }


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def audit_report(
    result: dict, grouped: dict[str, Counter],
    *, head_regex: str | None = None,
) -> dict:
    if result.get("status") != PENDING:
        raise ValueError("P16A source-locked event timing/ABBA gate must pass")
    probe = result.get("two_round_probe", {})
    if probe.get("status") != P15_PASS:
        raise ValueError("P16A independent P15 reference did not pass")
    p16 = probe.get("p16a")
    if not isinstance(p16, dict):
        raise ValueError("P16A profile absent")
    validate_profile(probe, p16)
    physical = audit_markers(grouped, p16["trace_ranges"])
    candidates = candidate_signatures(grouped)
    head = (
        verify_head_kernel_signature(grouped, head_regex)
        if head_regex is not None else None
    )
    return {
        "status": HEAD_PASS if head else MARKER_PASS,
        "source_sha": result["source_sha"],
        "pid": result["pid"],
        "gpu": result["gpu"],
        "nsys_node_trace_markers_pass": True,
        "bonus_head_kernel_skip_proven": head is not None,
        "physical_marker_counts": physical,
        "candidate_kernel_signatures_not_proof": candidates,
        "validated_real_bonus_head_kernel": head,
        "limits": [
            "Physical markers + independent P15 parity are not enough to "
            "identify a specific LM-head CUDA kernel",
            "Head regex requires manual source/kernel attribution; "
            "positive rejected-branch differences are not E2E speedup",
            "Only two-round fixed uniforms, not request seeded sampling",
            "No complete 14-depth decoder, audio quality, serving p95 or QPS",
        ],
    }


def summarize(row: dict) -> str:
    lines = [
        "# P16A — Nsight physical CUDA graph-node branch audit",
        "",
        f"**{row['status']}**",
        "",
        f"Source SHA: {row['source_sha']}",
        f"Worker PID: {row['pid']}; GPU: {row['gpu']}",
        f"Nsight report SHA256: {row.get('nsys_report_sha256', 'UNKNOWN')}",
        "",
        "| Branch | Conditional bonus marker nodes | "
        "Unconditional bonus marker nodes |",
        "|---|---:|---:|",
    ]
    data = row["physical_marker_counts"]
    for label in ("AA", "AR", "RA", "RR"):
        a = data[f"P16A_CASE_{label}_CONDITIONAL"]
        b = data[f"P16A_CASE_{label}_UNCONDITIONAL"]
        lines.append(
            f"| {label} | {a['physical_bonus_marker_kernels']} | "
            f"{b['physical_bonus_marker_kernels']} |"
        )
    head = row["validated_real_bonus_head_kernel"]
    lines += [
        "",
        "- Actual marked CUDA kernels traced with --cuda-graph-trace=node, "
        "not inferred from GPU output booleans.",
        "- Timed performance is from separate unmarked CUDA graphs, "
        "not Nsight trace overhead.",
    ]
    if head:
        lines.append(
            "- A separately identified LM-head kernel signature exhibited "
            "the expected 0/1/1/2 extra executions on the unconditional arm."
        )
    else:
        lines.append(
            "- **Actual LM-head kernel skip NOT yet proven.** "
            "Inspect candidate_kernel_signatures_not_proof, verify the "
            "true head kernel, then rerun with --bonus-head-regex."
        )
    lines += [
        "- No native seeded RNG parity, full 14-depth loop, HTTP QPS/p95 "
        "or audio quality equivalence.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--nsys-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bonus-head-regex", default=None,
                        help="Only after manually identifying real LM-head kernel")
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    if args.out.exists():
        parser.error("P16A trace audit output must not already exist")
    report_file = args.nsys_report.expanduser().resolve(strict=True)
    result = json.loads(args.smoke_report.expanduser().resolve(
        strict=True
    ).read_text())
    # Run Nsight stats only on the native .nsys-rep and preserve raw CSV.
    try:
        proc = subprocess.run([
            "nsys", "stats", "--report=nvtx_kern_sum",
            "--format=csv", str(report_file),
        ], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "P16A nsys NVTX kernel summary unavailable; "
            "no CPU-only or graph-level fallback"
        ) from exc
    grouped = parse_nvtx_kernel_csv(proc.stdout)
    audit = audit_report(
        result, grouped, head_regex=args.bonus_head_regex,
    )
    audit["nsys_report_sha256"] = _sha256(report_file)
    audit["nsys_version"] = subprocess.check_output(
        ["nsys", "--version"], text=True,
    ).strip()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "nvtx_kern_sum.csv").write_text(proc.stdout)
    (args.out / "trace-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
    )
    (args.out / "trace-audit.md").write_text(summarize(audit) + "\n")
    print(
        f"P16A_TRACE_STATUS={audit['status']} "
        f"REPORT={args.out/'trace-audit.md'}"
    )


if __name__ == "__main__":
    main()

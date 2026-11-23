# SPDX-License-Identifier: Apache-2.0
"""Inspect source-matched P19 CUDA700 diagnostic controls (never performance).

Read-only, no server/GPU calls, no new evidence roots. Compares only cases
with identical source SHA, real request, depth, single-frame workload gates.
Prints trace coverage, expected call counts and confirmed CUDA700 log evidence.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_profile import (
    profile_counts, validate_profile,
)

CUDA700 = re.compile(r"CUDA_ERROR_ILLEGAL_ADDRESS|illegal memory access")
PASS_STATUSES = {
    ("ALL_DEPTHS_EAGER", "DEFAULT"):
        "P19_ALL_DEPTHS_EAGER_SINGLE_FRAME_CONTROL_PASS_NOT_GRAPH_OR_SERVING",
    ("ALL_DEPTHS_GRAPH", "DEFAULT"):
        "P19_ALL_DEPTHS_GRAPH_SINGLE_FRAME_LIFECYCLE_PASS_NOT_COST_OR_SERVING",
    ("ALL_DEPTHS_EAGER", "MATCHED_EAGER"):
        "P19_MATCHED_EAGER_65_OPS_ONE_FRAME_NEXT_NATIVE_PASS",
    ("ALL_DEPTHS_GRAPH", "MIN_REPLAY"):
        "P19_MIN_REPLAY_ONE_FRAME_NEXT_NATIVE_PASS_NOT_COST",
    ("ALL_DEPTHS_GRAPH", "GC_GUARD"):
        "P19_GC_CAPTURE_GUARD_ONE_FRAME_NEXT_NATIVE_PASS_NOT_COST",
    ("ALL_DEPTHS_GRAPH", "RETAIN_GRAPH"):
        "P19_RETAINED_GRAPH_ONE_FRAME_NEXT_NATIVE_PASS_NOT_COST",
}


def inspect(root: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    cfg = json.loads((root / "matched_cost/config.json").read_text())
    d = cfg.get("diagnostic")
    if not isinstance(d, dict):
        raise RuntimeError("P19_COMPARISON_NO_DIAGNOSTIC_MODE")
    mode = d.get("arm")
    profile = validate_profile(
        mode, d.get("depth"), d.get("profile", "DEFAULT")
    )
    if mode not in ("ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER"):
        raise RuntimeError("P19_COMPARISON_REQUIRES_FULL_DEPTHS_DIAGNOSTIC")
    if (d.get("depth") != 14
            or cfg.get("p18_frames_per_request") != 2
            or cfg.get("p19_cost_frames_per_request") != 1
            or d.get("fence") != 0
            or d.get("native_fence") != 0
            or d.get("tail_fence") != 1):
        raise RuntimeError("P19_COMPARISON_UNPAIRED_CONFIG")
    sha = (root / "source.sha").read_text().strip()
    if not len(sha) == 40 or cfg.get("source_sha") != sha:
        raise RuntimeError("P19_COMPARISON_SOURCE_SHA_MISMATCH")
    cli = root / "matched_cost/diagnostic/client.json"
    result = json.loads(cli.read_text()) if cli.exists() else {}
    if result and (
        result.get("source_sha") != sha
        or result.get("diagnostic") != d
        or result.get("P19_full_cost_rows_should_be_zero") != 0
        or result.get("no_cost_or_dp_selection") is not True
    ):
        raise RuntimeError("P19_COMPARISON_CLIENT_PROVENANCE_FAIL")
    cov = result.get("trace_coverage", {})
    if cov and (
        cov.get("profile", "DEFAULT") != profile
        or cov.get("mode") != mode
        or cov.get("depths_verified") != list(range(1, 15))
        or cov.get("graph_arms_verified", 0)
            + cov.get("eager_arms_verified", 0) != 50
    ):
        raise RuntimeError("P19_COMPARISON_ARM_COVERAGE_DRIFT")

    server_log = root / "server.log"
    match = CUDA700.search(
        server_log.read_text(errors="replace")
    ) if server_log.exists() else None
    plan = profile_counts(mode, profile)
    expected = PASS_STATUSES[(mode, profile)]
    passed = (
        result.get("status") == expected
        and result.get("finish_reason") == "stop"
        and result.get("census_rows", 0) >= 28
        and "P19_NATIVE_NEXT_NATIVE_PREDICTOR_RETURNED"
            in result.get("native_next_frame_phases", [])
        and bool(cov)
        and match is None
    )
    status = (
        "ONE_REQUEST_PASS_NOT_SAFETY_PROOF" if passed
        else ("CUDA700_CONFIRMED" if match else
              "INCOMPLETE_OR_OTHER_FAILURE_NO_CUDA700_PROOF")
    )
    return {
        "root": str(root), "source_sha": sha,
        "mode": mode, "profile": profile,
        "prompt_id": result.get("case", {}).get("prompt_id"),
        "seed": result.get("case", {}).get("seed"),
        "input": result.get("request", {}).get("input"),
        "selected_frame": cov.get("frame"),
        "case_id": cov.get("case_id"),
        "status": status, "client_status": result.get("status"),
        "census_rows": result.get("census_rows", 0),
        "arms_verified": cov.get("graph_arms_verified", 0)
            + cov.get("eager_arms_verified", 0),
        "expected_eager_ops": 50 * plan["eager_operations"],
        "expected_capture_ops": 50 * plan["captures"],
        "expected_replays": 50 * plan["graph_replays"],
        "expected_operations": 50 * plan["operation_equivalents"],
        "last_tail": result.get("last_native_tail_stage"),
        "last_native": result.get("last_native_next_frame_stage"),
        "cuda700_in_server_log": match is not None,
    }


def compare(roots: list[Path]) -> dict:
    if len(roots) < 2 or len(roots) > 6:
        raise RuntimeError("P19_COMPARISON_REQUIRES_2_TO_6_ROOTS")
    runs = [inspect(root) for root in roots]
    if len({run["root"] for run in runs}) != len(runs):
        raise RuntimeError("P19_COMPARISON_DUPLICATE_ROOT")
    if len({(run["mode"], run["profile"]) for run in runs}) != len(runs):
        raise RuntimeError("P19_COMPARISON_DUPLICATE_MODE_AND_PROFILE")
    if len({run["source_sha"] for run in runs}) != 1:
        raise RuntimeError("P19_COMPARISON_CROSS_SHA_FORBIDDEN")
    requests = {
        (run["prompt_id"], run["seed"], run["input"]) for run in runs
    }
    if len(requests) != 1 or any(None in req for req in requests):
        raise RuntimeError("P19_COMPARISON_REQUEST_IDENTITY_DRIFT")
    frames = {(run["selected_frame"], run["case_id"]) for run in runs}
    if len(frames) != 1 or any(None in identity for identity in frames):
        raise RuntimeError("P19_COMPARISON_SELECTED_FRAME_OR_CASE_DRIFT")
    return {
        "status": "P19_SOURCE_MATCHED_DIAGNOSTIC_COMPARISON_NOT_CAUSAL_PROOF",
        "source_sha": runs[0]["source_sha"],
        "runs": runs,
        "decision": (
            "NO_P19A_COST_SELECTION_NO_P19B_DP_NO_PRODUCTION_MERGE"
        ),
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# P19 CUDA700 source-matched single-request controls",
        "",
        "**Not a production benchmark, memory safety proof, or cost selection.**",
        "",
        "Source SHA: `" + report["source_sha"] + "`",
        "",
        "| Profile | Arms | Expected calls (eager/capture/replay) | HTTP/CUDA outcome | Last tail stage |",
        "|---|---:|---:|---|---|",
    ]
    for run in report["runs"]:
        calls = (
            f'{run["expected_eager_ops"]}/'
            f'{run["expected_capture_ops"]}/'
            f'{run["expected_replays"]}'
        )
        lines.append(
            "| " + " | ".join((
                run["profile"] + " (" + run["mode"] + ")",
                str(run["arms_verified"]),
                calls,
                run["status"],
                str(run["last_tail"] or "NONE"),
            )) + " |"
        )
    lines += [
        "",
        "Calls are operation-equivalents, not matched kernels, addresses, timing,",
        "memory pools, or production QPS. One pass cannot prove race freedom.",
        "",
        "Decision: **" + report["decision"] + "**",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument(
        "--out", type=Path, help="New absent output prefix (.json/.md)"
    )
    a = parser.parse_args()
    report = compare(a.roots)
    print(to_markdown(report))
    if a.out:
        prefix = a.out.expanduser().resolve()
        j = prefix.with_suffix(".json")
        m = prefix.with_suffix(".md")
        if j.exists() or m.exists():
            raise RuntimeError("P19_COMPARISON_REFUSES_OVERWRITE")
        j.parent.mkdir(parents=True, exist_ok=True)
        with j.open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        with m.open("x") as f:
            f.write(to_markdown(report))
        print("P19_COMPARISON_JSON=" + str(j))
        print("P19_COMPARISON_MD=" + str(m))


if __name__ == "__main__":
    main()

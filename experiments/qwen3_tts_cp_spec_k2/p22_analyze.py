# SPDX-License-Identifier: Apache-2.0
"""Read-only P22 real Serving K2 CPU/NVTX attribution analyzer.

No cross-SHA ABBA performance comparisons. CPU regions are inclusive and
may nest; do not sum categories except the two non-overlapping .item() spans.
Nsight Systems CUDA timeline is needed for GPU kernel execution time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .p22_profile_once import (
    STATUS, STATUS_NO_NSYS, validate_profile,
)


def analyze(root: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    cfg = json.loads((root / "run.json").read_text())
    client = json.loads((root / "p22_client.json").read_text())
    sha = (root / "source.sha").read_text().strip()
    if (cfg.get("source_sha") != sha
            or client.get("source_sha") != sha
            or cfg.get("schema") != "qwen3tts-p22-serving-profile-v1"
            or client.get("status") not in (STATUS, STATUS_NO_NSYS)
            or not client.get("no_perf_or_quality_claim")):
        raise RuntimeError("P22_ANALYZER_UNQUALIFIED_OR_UNPINNED_RESULT")
    profile = client.get("stage_cpu_totals")
    if not isinstance(profile, dict):
        raise RuntimeError("P22_ANALYZER_MISSING_STAGE_SUMMARY")
    frames = client.get("profiled_candidate_frames")
    rounds = client.get("profiled_k2_rounds")
    if (type(frames) is not int or type(rounds) is not int
            or not frames > 0 or not rounds > 0):
        raise RuntimeError("P22_ANALYZER_INVALID_FRAME_OR_ROUND_COUNT")
    for name in ("FULL_CODE_PREDICTOR", "ROUND",
                 "ROUND_CPU_ACCEPT_LENGTH", "ROUND_CPU_CODE_VALIDITY",
                 "Q_PROPOSALS", "WIDE_VERIFIER",
                 "TARGET_SUPPORT_ACCEPT_BONUS"):
        if name not in profile:
            raise RuntimeError(f"P22_ANALYZER_MISSING_STAGE_{name}")
    full_ms = profile["FULL_CODE_PREDICTOR"]["cpu_wall_total_ms"]
    round_ms = profile["ROUND"]["cpu_wall_total_ms"]
    first_sync_ms = profile["ROUND_CPU_ACCEPT_LENGTH"]["cpu_wall_total_ms"]
    second_sync_ms = profile["ROUND_CPU_CODE_VALIDITY"]["cpu_wall_total_ms"]
    if (full_ms < round_ms or round_ms < first_sync_ms + second_sync_ms):
        # Nested timer overlap and measurement noise can create small
        # differences; >1% is a real broken trace/data contract.
        if (full_ms * 1.01 < round_ms
                or round_ms * 1.01 < first_sync_ms + second_sync_ms):
            raise RuntimeError("P22_ANALYZER_TIMER_INCLUSIVE_CONTRACT_DRIFT")
    measured = client.get("visits", [])
    if len(measured) != 3 or measured[-1].get("name") != "B_measured_real_request":
        raise RuntimeError("P22_ANALYZER_REAL_REQUEST_EVIDENCE_MISSING")
    worker = measured[-1]["worker"]
    if worker["candidate_frames"] != frames or worker["k2_rounds"] != rounds:
        raise RuntimeError("P22_ANALYZER_WORKER_FRAME_COUNT_DRIFT")

    want_nsys = cfg.get("nsys_capture_range")
    rep = root / "p22_measured_b.nsys-rep"
    # nsys may not write its final report until the server exits. An
    # instrumentation-only record is still valid without the file, but
    # there is NO verified GPU kernel timeline evidence in that case.
    has_nsys = want_nsys is True and rep.is_file() and rep.stat().st_size > 0
    return {
        "schema": "qwen3tts-p22-real-serving-profile-summary-v1",
        "status": "P22_ATTRIBUTION_COMPLETE_NOT_PERFORMANCE_BENCHMARK",
        "source_sha": sha,
        "candidate_frames": frames, "k2_rounds": rounds,
        "rounds_per_frame": rounds / frames,
        "accepted_drafts": worker["accepted_drafts"],
        "rejected_drafts": worker["rejected_drafts"],
        "acceptance_rate": worker["accepted_drafts"] / rounds,
        "cpu_full_predictor_per_frame_ms": full_ms / frames,
        "cpu_round_regions_per_frame_ms": round_ms / frames,
        "cpu_accept_length_item_per_frame_ms": first_sync_ms / frames,
        "cpu_code_validity_item_per_frame_ms": second_sync_ms / frames,
        "cpu_host_sync_items_per_frame_ms": (
            first_sync_ms + second_sync_ms
        ) / frames,
        "cpu_sync_share_of_inclusive_round_regions": (
            first_sync_ms + second_sync_ms
        ) / round_ms if round_ms else None,
        "nvtx_stages": profile,
        "nsys_requested": want_nsys,
        "nsys_report_finalized": has_nsys,
        "nsys_report": str(rep) if has_nsys else None,
        "kernel_gpu_execution_ms": "NOT_MEASURED_BY_CPU_STAGE_TIMERS",
        "limitations": [
            "NVTX region host time is not CUDA kernel time",
            "ROUND and sub-stages are nested; do not sum them",
            "Trace collection changes dispatch/scheduling; not ABBA QPS",
            "No C8/C16, native seeded parity, audio quality or merge GO",
        ],
    }


def markdown(report: dict) -> str:
    lines = [
        "# P22 real Serving K2 — CPU/NVTX attribution",
        "",
        "**Profiling only. No E2E speedup or GPU kernel time claim.**",
        "",
        f"Source SHA: `{report['source_sha']}`",
        f"Real candidate frames: **{report['candidate_frames']}**",
        f"Real K2 rounds: **{report['k2_rounds']}**",
        f"Round/frame: **{report['rounds_per_frame']:.3f}**",
        f"Acceptance rate: **{report['acceptance_rate']:.1%}**",
        "",
        "| Host stage (inclusive when nested) | CPU wall ms / frame |",
        "|---|---:|",
    ]
    for name in ("FULL_CODE_PREDICTOR", "ROUND", "Q_PROPOSALS",
                 "WIDE_VERIFIER", "TARGET_SUPPORT_ACCEPT_BONUS",
                 "ROUND_CPU_ACCEPT_LENGTH", "ROUND_CPU_CODE_VALIDITY",
                 "ROUND_GPU_COMMIT"):
        v = report["nvtx_stages"].get(name)
        if v:
            lines.append(f"| {name} | {v['cpu_wall_per_frame_ms']:.3f} |")
    lines += [
        "",
        "Two nonoverlapping host synchronization spans combined: "
        f"**{report['cpu_host_sync_items_per_frame_ms']:.3f} CPU ms/frame**.",
        "Share of inclusive ROUND CPU spans: "
        f"**{report['cpu_sync_share_of_inclusive_round_regions']:.1%}**.",
        "",
        "Nsight CUDA report: **"
        + ("FINALIZED" if report["nsys_report_finalized"] else "NOT YET VERIFIED")
        + "**. Host timers do NOT represent GPU kernel busy time.",
        "",
        "To inspect actual CUDA kernels, after the Nsight server exits run:",
        "```bash",
        "nsys stats --report cuda_gpu_kern_sum --report nvtx_sum "
        + str(Path(report['nsys_report']) if report['nsys_report']
              else 'ROOT/p22_measured_b.nsys-rep'),
        "```",
        "",
        "**P23 production GPU K2: BLOCKED pending GPU timeline and "
        "14-depth device-only correctness implementation.**",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError("P22_ANALYZER_REFUSE_OVERWRITE")
    report = analyze(a.root)
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "summary.md").write_text(markdown(report))
    print("P22_ANALYZER_STATUS=" + report["status"])
    print("P22_ANALYZER_REPORT=" + str(out / "summary.md"))


if __name__ == "__main__":
    main()

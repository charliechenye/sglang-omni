# SPDX-License-Identifier: Apache-2.0
"""P4 K2 fixed-state cost screen against a saved on-policy acceptance cohort.

This is NOT a measured serving speedup and CUDA-event envelopes are NOT
kernel busy time. The candidate must still eliminate Python host sync, use
correct RNG/rollback, and beat a real optimized Graph-ON no-spec baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from sglang_omni.models.qwen3_tts.code_predictor_spec_perf import SCHEMA

STATUS = "P4_FIXED_STATE_FORWARD_BREAK_EVEN_SCREEN_NOT_GRAPH_ON_SERVING_SPEEDUP"
CELLS = (
    "serial_one_stack", "serial_two_stack", "packed_two_stack",
    "draft_head", "draft_select_debug", "accept_debug",
)


def validate_acceptance(data: dict) -> dict:
    depths = data.get("depths")
    if not isinstance(depths, list) or len(depths) != 14:
        raise ValueError("expected 14 P2 per-depth proposal rows")
    frames = data.get("frames")
    proposals = data.get("total_proposals")
    accepted = data.get("total_accepted")
    rejected = data.get("total_rejected")
    bonus = data.get("total_bonus")
    if (
        type(frames) is not int or frames < 1
        or any(type(x) is not int or x < 0 for x in (proposals, accepted, rejected, bonus))
        or proposals != accepted + rejected
        or accepted + rejected + bonus != 14 * frames
        or sorted(row.get("residual_depth") for row in depths) != list(range(2, 16))
        or sum(row.get("proposals", -1) for row in depths) != proposals
    ):
        raise ValueError("P2 counters do not cover all 14 residual depths")
    if proposals <= 0:
        raise ValueError("zero proposals cannot estimate K2")
    return {
        "frames": frames,
        "proposals": proposals,
        "accepted": accepted,
        "rejected": rejected,
        "bonus": bonus,
        "codes_per_round": 14 * frames / proposals,
        "depths": depths,
        "source_sha": data.get("source_sha"),
    }


def validate_probe(row: dict) -> dict:
    if row.get("schema") != SCHEMA:
        raise ValueError("wrong P4 probe schema")
    depth = row.get("proposal_depth")
    if type(depth) is not int or not 2 <= depth <= 15:
        raise ValueError("invalid P4 proposal depth")
    measures = row.get("measurements")
    if not isinstance(measures, dict) or any(name not in measures for name in CELLS):
        raise ValueError("missing P4 hotpath cells")
    for name in CELLS:
        v = measures[name]
        if v.get("blocks") != 2 or any(
            type(v.get(k)) not in (int, float)
            or not math.isfinite(v[k]) or v[k] <= 0
            for k in ("cuda_event_ms", "host_wall_ms")
        ):
            raise ValueError(f"invalid P4 samples for {name}")
    return row


def analyze(
    perf_dir: Path, acceptance_report: Path, *, min_probes: int = 3
) -> dict:
    perf_dir = perf_dir.expanduser().resolve(strict=True)
    paths = sorted(perf_dir.glob("perf_pid*_frame*_depth*.json"))
    if len(paths) < min_probes:
        raise RuntimeError(f"only {len(paths)} P4 profiles; require >= {min_probes}")
    profiles = [validate_probe(json.loads(p.read_text())) for p in paths]
    if len({r.get("pid") for r in profiles}) > 1:
        raise RuntimeError("mixed P4 workers; use one GPU worker")
    if len({r["proposal_depth"] for r in profiles}) != len(profiles):
        raise RuntimeError("repeated P4 depths from multiple runs")
    sha_values = {row.get("source_sha") for row in profiles}
    gpu_values = {row.get("gpu") for row in profiles}
    if len(sha_values) != 1 or "UNATTESTED" in sha_values or len(gpu_values) != 1:
        raise RuntimeError("missing/mixed P4 source SHA or GPU name")
    acceptance = validate_acceptance(
        json.loads(acceptance_report.expanduser().resolve(strict=True).read_text())
    )
    by_depth = {row["proposal_depth"]: row for row in profiles}
    sampled = sorted(by_depth)
    # Nearest profiled depth, as an explicitly extrapolated, not measured,
    # approximation for unprofiled depths. Never interpret as a kernel benchmark.
    def nearest(depth: int) -> dict:
        return by_depth[min(sampled, key=lambda x: (abs(x - depth), x))]

    stage = "cuda_event_ms"
    baseline_stack_ms = 0.0
    candidate_stack_ms = 0.0
    candidate_with_head_ms = 0.0
    debug_candidate_ms = 0.0
    approximations = []
    for info in acceptance["depths"]:
        depth = info["residual_depth"]
        n = info["proposals"]
        prof = nearest(depth)
        measurements = prof["measurements"]
        t1 = measurements["serial_one_stack"][stage]
        t2 = measurements["packed_two_stack"][stage]
        th = measurements["draft_head"][stage]
        td = measurements["draft_select_debug"][stage]
        ta = measurements["accept_debug"][stage]
        # Every frame produces exactly one code at each residual depth, but
        # proposals only occur where the speculative path did not bonus-skip.
        baseline_stack_ms += acceptance["frames"] * t1
        candidate_stack_ms += n * t2
        candidate_with_head_ms += n * (t2 + th)
        debug_candidate_ms += n * (t2 + td + ta)
        approximations.append({
            "depth": depth, "profiled_at_depth": prof["proposal_depth"],
            "proposed": n, "serial_one_cuda_event_ms": t1,
            "packed_two_cuda_event_ms": t2,
            "packed_vs_one_ratio": t2 / t1,
        })
    if min(
        baseline_stack_ms, candidate_stack_ms, candidate_with_head_ms,
        debug_candidate_ms
    ) <= 0:
        raise ValueError("invalid total cost model")
    selected = [{
        "depth": row["proposal_depth"],
        "cache_len": row["cache_len"],
        "serial_one_cuda_event_ms": row["measurements"]["serial_one_stack"][stage],
        "serial_two_cuda_event_ms": row["measurements"]["serial_two_stack"][stage],
        "packed_two_cuda_event_ms": row["measurements"]["packed_two_stack"][stage],
        "draft_head_cuda_event_ms": row["measurements"]["draft_head"][stage],
        "draft_select_debug_cuda_event_ms": row["measurements"]["draft_select_debug"][stage],
        "accept_debug_cuda_event_ms": row["measurements"]["accept_debug"][stage],
    } for row in sorted(profiles, key=lambda r: r["proposal_depth"])]
    ratio = acceptance["codes_per_round"]
    optimistic = baseline_stack_ms / candidate_stack_ms
    return {
        "status": STATUS,
        "source_sha": next(iter(sha_values)),
        "acceptance_source_sha": acceptance["source_sha"],
        "same_source": next(iter(sha_values)) == acceptance["source_sha"],
        "gpu": next(iter(gpu_values)),
        "measured_state_count": len(profiles),
        "on_policy_accounting": {
            k: acceptance[k] for k in (
                "frames", "proposals", "accepted", "rejected", "bonus",
                "codes_per_round",
            )
        },
        "weighted_fixed_state_screen": {
            "baseline_serial_stack_ms": baseline_stack_ms,
            "candidate_packed_stack_ms": candidate_stack_ms,
            "candidate_packed_plus_draft_head_ms": candidate_with_head_ms,
            "candidate_packed_plus_current_debug_draft_accept_ms": debug_candidate_ms,
            "serial_stack_over_packed_stack_quotient": optimistic,
            "serial_stack_over_packed_plus_head_quotient": (
                baseline_stack_ms / candidate_with_head_ms
            ),
            "serial_stack_over_debug_quotient": (
                baseline_stack_ms / debug_candidate_ms
            ),
            "optimistic_packed_stack_below_serial_stack": (
                candidate_stack_ms < baseline_stack_ms
            ),
        },
        "selected_real_state_profiles": selected,
        "nearest_depth_extrapolations": approximations,
        "limitations": [
            "This is an extrapolated fixed-state launch-envelope screen, NOT a serving speedup.",
            "CUDA event duration includes host launch gaps, NOT a CUPTI GPU busy measure.",
            "Accept/reject, draft and target stages are not yet a GPU-only fused implementation.",
            "Serial baseline here is Predictor stack ONLY and not the actual optimized Graph-ON AR loop.",
            "Unprofiled residual depths use nearest profiled depth; this is approximation not measurement.",
            "Different source_sha for P2/P4 means accepted-token counts are a prior-cohort reference.",
            "A positive screen is permission to prototype a fused GPU path, not proof of performance.",
            "A negative screen may be caused by launch overhead; confirm with GPU kernel timeline before stopping.",
        ],
    }


def write_report(out: Path, data: dict) -> None:
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    s = data["weighted_fixed_state_screen"]
    lines = [
        "# Qwen3-TTS K2 P4 — Fixed-state forward break-even screen", "",
        "**" + data["status"] + "**", "",
        "Candidate source: " + str(data["source_sha"]),
        "Acceptance cohort source: " + str(data["acceptance_source_sha"]),
        "Same source: " + str(data["same_source"]),
        "GPU: " + data["gpu"],
        "Measured on-policy KV states: " + str(data["measured_state_count"]),
        "Post-first residual codes per K2 round: "
        + f"{data['on_policy_accounting']['codes_per_round']:.4f}",
        "",
        "| Launch-envelope estimate | ms (weighted fixed-state) |",
        "|---|---:|",
        f"| Serial stack | {s['baseline_serial_stack_ms']:.2f} |",
        f"| K2 packed stack only | {s['candidate_packed_stack_ms']:.2f} |",
        f"| K2 packed + head | {s['candidate_packed_plus_draft_head_ms']:.2f} |",
        f"| K2 packed + current debug draft/accept | {s['candidate_packed_plus_current_debug_draft_accept_ms']:.2f} |",
        "",
        f"Stack-only screening quotient serial/packed = "
        f"**{s['serial_stack_over_packed_stack_quotient']:.3f}x**",
        "",
        "## Actual fixed-KV sample costs (CUDA-event launch envelopes, NOT GPU busy)", "",
        "| Depth | Cache len | Serial 1 | Serial 2 | Packed 2 | Draft head | Debug draft | Debug accept |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["selected_real_state_profiles"]:
        lines.append(
            "| {depth} | {cache_len} | {serial_one_cuda_event_ms:.4f} | "
            "{serial_two_cuda_event_ms:.4f} | {packed_two_cuda_event_ms:.4f} | "
            "{draft_head_cuda_event_ms:.4f} | "
            "{draft_select_debug_cuda_event_ms:.4f} | "
            "{accept_debug_cuda_event_ms:.4f} |".format(**row)
        )
    lines += ["", "## Strict interpretation", ""] + [
        "- " + item for item in data["limitations"]
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perf-dir", type=Path, required=True)
    ap.add_argument("--acceptance-report", type=Path, required=True)
    ap.add_argument("--min-probes", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.min_probes <= 16:
        ap.error("min-probes must be in 1..16")
    data = analyze(args.perf_dir, args.acceptance_report, min_probes=args.min_probes)
    write_report(args.out, data)
    print(data["status"] + ": " + str(args.out / "summary.md"))


if __name__ == "__main__":
    main()

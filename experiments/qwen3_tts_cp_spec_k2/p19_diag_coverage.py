# SPDX-License-Identifier: Apache-2.0
"""Fail-closed P19 diagnostic phase coverage validation (CPU only).

A completed HTTP request is NOT evidence a selected diagnostic arm ran.
This validates exactly ONE real selected native frame and EVERY requested
depth/arm stage, using fsynced worker traces; never generates GPU cost
or production-performance claims.
"""
from __future__ import annotations

from collections import Counter

from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_profile import (
    DEFAULT_OPS, profile_counts, validate_profile,
)

GRAPH_STAGES = (
    "P19_ARM_SELECTED",
    "P19_ARM_EAGER_AND_WARMUP_SYNC_PASS",
    "P19_ARM_GRAPH_CAPTURE_SYNC_PASS",
    "P19_ARM_FIRST_REPLAY_SYNC_PASS",
    "P19_ARM_REPLAY_EAGER_PARITY_PASS",
    "P19_ARM_TIMING_BLOCK_0_SYNC_PASS",
    "P19_ARM_TIMING_BLOCK_1_SYNC_PASS",
    "P19_ARM_TIMING_BLOCK_2_SYNC_PASS",
    "P19_ARM_TIMING_BLOCK_3_SYNC_PASS",
    "P19_ARM_GRAPH_RELEASE_AND_SYNC_PASS",
)
EAGER_STAGES = (
    "P19_ARM_SELECTED",
    "P19_EAGER_ONLY_DEVICE_SYNC_PASS",
)
GC_GRAPH_STAGES = (
    GRAPH_STAGES[:2]
    + ("P19_ARM_GC_CAPTURE_GUARD_ENTER",
       "P19_ARM_GC_CAPTURE_GUARD_EXIT")
    + GRAPH_STAGES[2:]
)
MIN_GRAPH_STAGES = (
    GRAPH_STAGES[:5]
    + ("P19_ARM_MIN_REPLAY_NO_TIMING_PASS",)
    + GRAPH_STAGES[-1:]
)
RETAIN_GRAPH_STAGES = (
    GRAPH_STAGES[:-1] + ("P19_ARM_GRAPH_RETAINED_AND_SYNC_PASS",)
)
MATCHED_EAGER_STAGES = (
    "P19_ARM_SELECTED",
    "P19_ARM_MATCHED_EAGER_REFERENCE_SYNC_PASS",
    "P19_ARM_MATCHED_EAGER_WARMUPS_SYNC_PASS",
    "P19_ARM_MATCHED_EAGER_CAPTURE_SURROGATE_SYNC_PASS",
    "P19_ARM_MATCHED_EAGER_FIRST_SURROGATE_SYNC_PASS",
    *(f"P19_ARM_MATCHED_EAGER_BLOCK_{i}_SYNC_PASS" for i in range(4)),
    "P19_ARM_MATCHED_EAGER_TOTAL_65_SYNC_PASS",
    "P19_EAGER_ONLY_DEVICE_SYNC_PASS",
)


def stages_for_profile(mode: str, profile: str) -> tuple[str, ...]:
    if profile == "MATCHED_EAGER":
        return MATCHED_EAGER_STAGES
    if mode.endswith("_EAGER"):
        return EAGER_STAGES
    return {
        "MIN_REPLAY": MIN_GRAPH_STAGES,
        "GC_GUARD": GC_GRAPH_STAGES,
        "RETAIN_GRAPH": RETAIN_GRAPH_STAGES,
    }.get(profile, GRAPH_STAGES)


DEPTH_STAGES = (
    "P19_COST_TARGET_FRAME_SELECTED",
    "P19_DIAGNOSTIC_ARM_RETURNED_NO_COST_SELECTION",
    "P19_DIAGNOSTIC_FINAL_RESTORE_SYNC_PASS",
)


def arms_at_depth(depth: int) -> tuple[str, ...]:
    if not 1 <= depth <= 14:
        raise ValueError("P19_COVERAGE_INVALID_DEPTH")
    return ("AR", *(str(k) for k in (2, 3, 4)
                    if depth + k - 1 <= 14))


def _require_single(
    counts: Counter, depth: int, arm: str | None, stage: str,
) -> None:
    count = counts[(depth, arm, stage)]
    if count != 1:
        raise RuntimeError(
            "P19_DIAGNOSTIC_PHASE_COVERAGE_INCOMPLETE "
            f"depth={depth} arm={arm} stage={stage} count={count}"
        )


def validate_diagnostic_trace(events: list[dict], diagnostic: dict) -> dict:
    """Validates exactly requested depth/arms in one native frame and worker.

    Raises instead of claiming a pass when any expected phase is missing,
    duplicate, mixed between workers/frames, or no arm was actually executed.
    """
    if not isinstance(diagnostic, dict):
        raise RuntimeError("P19_DIAG_INVALID_LAUNCH_CONFIG")
    arm = diagnostic.get("arm")
    depth = diagnostic.get("depth")
    profile = diagnostic.get("profile", "DEFAULT")
    if arm not in (
        "OFF", "AR_EAGER", "AR_GRAPH",
        "K2_EAGER", "K2_GRAPH", "K3_EAGER", "K3_GRAPH",
        "K4_EAGER", "K4_GRAPH", "ALL_GRAPH",
        "ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER",
    ) or type(depth) is not int or not 1 <= depth <= 14:
        raise RuntimeError("P19_DIAG_INVALID_MODE_OR_DEPTH")
    validate_profile(arm, depth, profile)
    if not events:
        raise RuntimeError("P19_DIAG_NO_WORKER_TRACE")
    # Depth/arm boundaries never depend on CUDA tensor reads.
    if arm == "OFF":
        skips = [
            ev for ev in events
            if ev.get("stage") == "P19_COST_PROBE_SKIPPED_OFF"
            and ev.get("depth") == depth and ev.get("arm") == "OFF"
        ]
        any_arms = any(
            ev.get("stage") == "P19_ARM_SELECTED" for ev in events
        )
        if not skips or any_arms:
            raise RuntimeError("P19_DIAG_OFF_ARM_NOT_PROVEN")
        return {
            "mode": arm, "terminal_depth": depth,
            "diagnostic_frame": None, "depths_verified": [depth],
            "graph_arms_verified": 0, "off_events": len(skips),
            "result": "P19_OFF_P18_ONLY_CHECKPOINT_OBSERVED_NOT_COST",
        }

    if arm in ("ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER"):
        expected_depths = tuple(range(1, depth + 1))
        if diagnostic.get("policy") != "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION":
            raise RuntimeError("P19_DIAG_ALL_DEPTHS_POLICY_GATE")
    else:
        expected_depths = (depth,)

    targets = [
        ev for ev in events
        if ev.get("stage") == "P19_COST_TARGET_FRAME_SELECTED"
    ]
    if len(targets) != len(expected_depths):
        raise RuntimeError(
            "P19_DIAG_SELECTED_DEPTH_COUNT_MISMATCH "
            f"actual={len(targets)} wanted={len(expected_depths)}"
        )
    if set(ev.get("depth") for ev in targets) != set(expected_depths):
        raise RuntimeError("P19_DIAG_SELECTED_DEPTH_SET_DRIFT")
    frames = {(ev.get("case_id"), ev.get("frame"), ev.get("pid"))
              for ev in targets}
    if len(frames) != 1:
        raise RuntimeError("P19_DIAG_MIXED_NATIVE_FRAME_OR_WORKER")
    case_id, frame, pid = next(iter(frames))
    if (not isinstance(case_id, str) or not case_id
            or type(frame) is not int or type(pid) is not int):
        raise RuntimeError("P19_DIAG_INVALID_NATIVE_FRAME_IDENTITY")
    scoped = [
        ev for ev in events if ev.get("stage", "").startswith("P19_")
    ]
    if any(ev.get("case_id") != case_id
           or ev.get("frame") != frame
           or ev.get("pid") != pid
           for ev in scoped):
        raise RuntimeError("P19_DIAG_UNEXPECTED_FRAME_OR_WORKER_PHASE")
    counts = Counter(
        (ev.get("depth"), ev.get("arm"), ev.get("stage"))
        for ev in scoped
    )
    if any(type(ev.get("monotonic_ns")) is not int
           for ev in scoped):
        raise RuntimeError("P19_DIAG_MISSING_CPU_MONOTONIC_EVENT_TIME")
    timestamps = {
        (ev.get("depth"), ev.get("arm"), ev.get("stage")):
            ev["monotonic_ns"] for ev in scoped
    }
    verified_arms = 0
    prior_final = -1
    for d in expected_depths:
        _require_single(counts, d, arm, DEPTH_STAGES[0])
        _require_single(counts, d, arm, DEPTH_STAGES[1])
        _require_single(counts, d, arm, DEPTH_STAGES[2])
        if arm in ("ALL_GRAPH", "ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER"):
            arms = arms_at_depth(d)
        else:
            chosen = arm.split("_", 1)[0]
            wanted = "AR" if chosen == "AR" else chosen[1:]
            if wanted not in arms_at_depth(d):
                raise RuntimeError("P19_DIAG_ILLEGAL_ARM_AT_TARGET_DEPTH")
            arms = (wanted,)
        phases = stages_for_profile(arm, profile)
        selected_time = timestamps[(d, arm, DEPTH_STAGES[0])]
        returned_time = timestamps[(d, arm, DEPTH_STAGES[1])]
        final_time = timestamps[(d, arm, DEPTH_STAGES[2])]
        if not prior_final < selected_time < returned_time < final_time:
            raise RuntimeError(
                f"P19_DIAG_NONMONOTONIC_DEPTH_BOUNDARY_DEPTH_{d}"
            )
        last_arm_finish = selected_time
        for selected_arm in arms:
            for stage in phases:
                _require_single(counts, d, selected_arm, stage)
            arm_times = [
                timestamps[(d, selected_arm, stage)] for stage in phases
            ]
            if (arm_times != sorted(arm_times)
                    or len(set(arm_times)) != len(arm_times)
                    or not selected_time < arm_times[0]
                    or not last_arm_finish < arm_times[0]
                    or not arm_times[-1] < returned_time):
                raise RuntimeError(
                    f"P19_DIAG_ARM_PHASE_ORDER_INVALID_D{d}_{selected_arm}"
                )
            last_arm_finish = arm_times[-1]
            verified_arms += 1
        prior_final = final_time

    if profile == "RETAIN_GRAPH":
        _require_single(
            counts, depth, arm,
            "P19_RETAINED_50_GRAPHS_LIVE_UNTIL_WORKER_EXIT",
        )
    expected_selected = verified_arms
    selected_count = sum(
        count for (_, _, stage), count in counts.items()
        if stage == "P19_ARM_SELECTED"
    )
    if selected_count != expected_selected:
        raise RuntimeError(
            "P19_DIAG_UNEXPECTED_EXTRA_ARM_SELECTED "
            f"actual={selected_count} expected={expected_selected}"
        )
    work = profile_counts(arm, profile)
    return {
        "mode": arm,
        "profile": profile,
        "expected_operations_per_arm": work["operation_equivalents"],
        "expected_total_arm_operations": (
            work["operation_equivalents"] * verified_arms
        ),
        "expected_graph_replays": work["graph_replays"] * verified_arms,
        "expected_eager_operations": work["eager_operations"] * verified_arms,
        "expected_captures": work["captures"] * verified_arms,
        "terminal_depth": depth,
        "case_id": case_id,
        "frame": frame,
        "worker_pid": pid,
        "depths_verified": list(expected_depths),
        "graph_arms_verified": (
            verified_arms if not arm.endswith("_EAGER") else 0
        ),
        "eager_arms_verified": (
            verified_arms if arm.endswith("_EAGER") else 0
        ),
        "result": "P19_DIAGNOSTIC_TRACE_COVERAGE_PASS_NO_COST_SELECTION",
    }

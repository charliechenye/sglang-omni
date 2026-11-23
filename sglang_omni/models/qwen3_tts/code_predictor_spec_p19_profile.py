# SPDX-License-Identifier: Apache-2.0
"""P19 diagnostic-only controlled CUDA Graph lifetime/pressure experiments.

Pure CPU config and expected call-count contract. Never change formal P19A
matched-cost measurement or production serving.
"""
from __future__ import annotations

import os

PROFILE_ENV = "SGLANG_OMNI_QTTS_CP_P19_DIAG_PROFILE"
PROFILES = ("DEFAULT", "MIN_REPLAY", "MATCHED_EAGER",
            "GC_GUARD", "RETAIN_GRAPH")

# The original P19 cost probe executes three eager operations (reference
# and two warmups), one capture, and 1 + 4*(3+12) graph replays.
EAGER_REFERENCE = 1
EAGER_WARMUPS = 2
CAPTURES = 1
FIRST_GRAPH_REPLAYS = 1
TIMING_BLOCKS = 4
TIMING_WARM_REPLAYS = 3
TIMING_REPEATS = 12
DEFAULT_REPLAYS = (
    FIRST_GRAPH_REPLAYS
    + TIMING_BLOCKS * (TIMING_WARM_REPLAYS + TIMING_REPEATS)
)
DEFAULT_OPS = EAGER_REFERENCE + EAGER_WARMUPS + CAPTURES + DEFAULT_REPLAYS
MIN_REPLAY_OPS = EAGER_REFERENCE + EAGER_WARMUPS + CAPTURES + FIRST_GRAPH_REPLAYS
MAX_RETAINED_GRAPHS = 50


def validate_profile(mode: str | None, depth: int | None, profile: str) -> str:
    if profile not in PROFILES:
        raise RuntimeError(f"P19_DIAG_UNKNOWN_PROFILE {profile!r}")
    if profile == "DEFAULT":
        return profile
    if depth != 14:
        raise RuntimeError("P19_PROFILE_REQUIRES_TERMINAL_DEPTH14")
    graph = {"MIN_REPLAY", "GC_GUARD", "RETAIN_GRAPH"}
    if (profile in graph and mode != "ALL_DEPTHS_GRAPH"
            or profile == "MATCHED_EAGER" and mode != "ALL_DEPTHS_EAGER"):
        raise RuntimeError("P19_DIAG_PROFILE_MODE_MISMATCH")
    return profile


def current_profile(diagnostic: tuple[str, int] | None) -> str:
    raw = os.environ.get(PROFILE_ENV, "DEFAULT")
    if diagnostic is None:
        if PROFILE_ENV in os.environ:
            raise RuntimeError("P19_DIAG_PROFILE_REQUIRES_DIAGNOSTIC_MODE")
        return "DEFAULT"
    return validate_profile(diagnostic[0], diagnostic[1], raw)


def profile_counts(mode: str, profile: str) -> dict[str, int]:
    """Operation-equivalent count, not kernel/time equivalence."""
    validate_profile(mode, 14, profile)
    if mode.endswith("_EAGER"):
        return {
            "eager_operations": DEFAULT_OPS if profile == "MATCHED_EAGER" else 1,
            "captures": 0, "graph_replays": 0,
            "operation_equivalents": (
                DEFAULT_OPS if profile == "MATCHED_EAGER" else 1
            ),
        }
    replays = FIRST_GRAPH_REPLAYS if profile == "MIN_REPLAY" else DEFAULT_REPLAYS
    return {
        "eager_operations": EAGER_REFERENCE + EAGER_WARMUPS,
        "captures": CAPTURES, "graph_replays": replays,
        "operation_equivalents": (
            EAGER_REFERENCE + EAGER_WARMUPS + CAPTURES + replays
        ),
    }

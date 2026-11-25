# SPDX-License-Identifier: Apache-2.0
"""P11 real on-policy C1 evidence parsers, anti-overclaim and depth ledgers."""
from __future__ import annotations

import pytest

from experiments.qwen3_tts_cp_spec_k2.p11b_hybrid_smoke import (
    check_visit, decode_depths, parse_depth_stats,
)
from experiments.qwen3_tts_cp_spec_k2.quality_smoke import parse_k2_log_entries
from sglang_omni.models.qwen3_tts.code_predictor_spec_p11b_hybrid import (
    stats_line,
)

SHA = "a" * 40


def make_pair(*, mode: str):
    if mode == "k2":
        ar = []
        proposed = [2, 4, 6, 8, 10, 12, 14]
        bonus = [3, 5, 7, 9, 11, 13, 15]
        accepted = proposed
        ar_through = 1
        ar_steps = 0
    else:
        ar = [2, 3]
        proposed = [4, 6, 8, 10, 12, 14]
        bonus = [5, 7, 9, 11, 13, 15]
        accepted = proposed
        ar_through = 3
        ar_steps = 2
    raw_depth = stats_line(
        frame=4, ar_through=ar_through,
        ar=ar, proposed=proposed, accepted=accepted,
        bonus=bonus, source_sha=SHA,
    )
    raw_k2 = (
        f"QTTS_CP_SPEC_K2 frame=4 rounds={len(proposed)} "
        f"proposed={len(proposed)} accepted={len(accepted)} "
        f"bonus={len(bonus)} rejected=0 committed_residual=15 "
        f"ar_steps={ar_steps} ar_through={ar_through}"
    )
    return (
        parse_k2_log_entries(raw_k2),
        parse_depth_stats(raw_depth),
    )


def test_p11b_native_k2_hybrid_depth_ledger_parses_on_policy():
    pure_rows, pure_depths = make_pair(mode="k2")
    hy_rows, hy_depths = make_pair(mode="hybrid")
    pure = check_visit("k2", pure_rows, pure_depths, SHA, 3)
    hybrid = check_visit("hybrid", hy_rows, hy_depths, SHA, 3)
    assert pure["frames"] == hybrid["frames"] == 1
    assert pure["ar_steps"] == 0
    assert hybrid["ar_steps"] == 2
    assert hybrid["spec_rounds"] == 6
    assert hybrid["target_calls"] == 8
    assert pure["target_calls"] == 7
    assert hybrid["depth_activity"]["2"]["ar"] == 1
    assert hybrid["depth_activity"]["4"]["proposed"] == 1
    assert hybrid["depth_activity"]["5"]["bonus"] == 1
    assert check_visit("native", [], [], SHA, 3)["frames"] == 0


def test_p11b_depth_parser_rejects_mixed_source_or_lost_codebooks():
    rows, depths = make_pair(mode="hybrid")
    assert decode_depths("-") == []
    with pytest.raises(ValueError):
        decode_depths("3,2")
    with pytest.raises(ValueError):
        decode_depths("2,2")
    with pytest.raises(RuntimeError, match="source/counters"):
        check_visit("hybrid", rows, depths, "b" * 40, 3)
    bad = dict(depths[0])
    bad["bonus"] = [5]
    with pytest.raises(RuntimeError, match="source/counters"):
        check_visit("hybrid", rows, [bad], SHA, 3)
    malformed = (
        f"QTTS_CP_K2_DEPTHS frame=7 ar_through=3 ar=2,3 "
        f"proposed=4,6 accepted=4 bonus=5 "
        f"source_sha={SHA}"
    )
    with pytest.raises(ValueError, match="conservation"):
        parse_depth_stats(malformed)

# SPDX-License-Identifier: Apache-2.0
"""P16A fail-closed CPU contracts for physical GPU kernel evidence."""
from __future__ import annotations

import inspect

import pytest

from experiments.qwen3_tts_cp_spec_k2.p16a_trace_audit import (
    BONUS_MARKER, ROUND_MARKER, audit_markers,
    candidate_signatures, parse_nvtx_kernel_csv,
    verify_head_kernel_signature,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p16a_profile import (
    expected_bonus_markers, profile_branch_costs, TRIAL_ORDERS,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p15_two_round import (
    gpu_round, two_gpu_rounds,
)

HEADER = (
    "NVTX Range,Style,PID,TID,NVTX Inst,Kern Inst,"
    "Total Time (ns),Avg (ns),Med (ns),Min (ns),Max (ns),"
    "StdDev (ns),Kernel Name"
)


def sample_trace():
    lines = [
        "Processing Nsight Systems output...",
        HEADER,
    ]
    cases = []
    for branch in ("AA", "AR", "RA", "RR"):
        for arm in ("conditional", "unconditional"):
            name = f"P16A_CASE_{branch}_{arm.upper()}"
            bonus = expected_bonus_markers(branch, arm)
            cases.append({
                "nvtx_range": name,
                "transition": branch, "arm": arm,
                "marked_replays": 1,
                "expected_round_start_marker_nodes": 2,
                "expected_bonus_head_marker_nodes": bonus,
            })
            for count, kernel in (
                (2, f"void {ROUND_MARKER}(float*)"),
                (bonus, f"void {BONUS_MARKER}(float*)"),
                (4 + bonus, "real_lm_head_gemm_kernel"),
                (12, "predictor_fused_layer_kernel"),
            ):
                if count:
                    lines.append(
                        f"{name},PushPop,4106413,7,1,{count},"
                        f"1000,100,100,100,100,0,{kernel}"
                    )
    return "\n".join(lines) + "\n", cases


def test_p16a_all_branch_marker_counts_not_cpu_flags():
    expected = {
        "AA": (2, 2),
        "AR": (1, 2),
        "RA": (1, 2),
        "RR": (0, 2),
    }
    for branch, (cond, uncond) in expected.items():
        assert expected_bonus_markers(branch, "conditional") == cond
        assert expected_bonus_markers(branch, "unconditional") == uncond
    with pytest.raises(ValueError):
        expected_bonus_markers("AX", "conditional")


def test_p16a_nsys_csv_parser_and_positive_negative_controls():
    data, cases = sample_trace()
    grouped = parse_nvtx_kernel_csv(data)
    assert len(grouped) == 8
    outcome = audit_markers(grouped, cases)
    assert len(outcome) == 8
    assert outcome["P16A_CASE_RR_CONDITIONAL"]["physical_bonus_marker_kernels"] == 0
    assert outcome["P16A_CASE_RR_UNCONDITIONAL"]["physical_bonus_marker_kernels"] == 2
    head = verify_head_kernel_signature(grouped, "real_lm_head_gemm")
    assert head["extra_kernel_nodes_per_rejected_bonus"] == 1
    assert head["delta_unconditional_minus_conditional"] == {
        "AA": 0, "AR": 1, "RA": 1, "RR": 2,
    }
    assert any(
        "real_lm_head_gemm" in row["kernel_name"]
        for row in candidate_signatures(grouped)
    )


def test_p16a_trace_fail_closed_for_missing_gpu_node_marker():
    data, cases = sample_trace()
    grouped = parse_nvtx_kernel_csv(data)
    # False case assertion can't be faked by "bonus_executed=0" metadata.
    grouped["P16A_CASE_AA_CONDITIONAL"].pop(
        next(k for k in grouped["P16A_CASE_AA_CONDITIONAL"] if BONUS_MARKER in k)
    )
    with pytest.raises(ValueError, match="physical marker"):
        audit_markers(grouped, cases)
    with pytest.raises(ValueError, match="incomplete"):
        audit_markers(
            {k: v for k, v in grouped.items() if not k.endswith("_UNCONDITIONAL")},
            cases,
        )


def test_p16a_nsys_invalid_header_or_no_gpu_nodes_is_not_pass():
    with pytest.raises(ValueError, match="mapping unavailable"):
        parse_nvtx_kernel_csv("Range,Host event count\nP16A_TRACE,8\n")
    with pytest.raises(ValueError, match="mapping unavailable"):
        parse_nvtx_kernel_csv(HEADER + "\n")


def test_p16a_manual_head_regex_must_not_point_to_markers():
    data, _ = sample_trace()
    grouped = parse_nvtx_kernel_csv(data)
    with pytest.raises(ValueError, match="instrument markers"):
        verify_head_kernel_signature(grouped, "p16a_bonus_head_marker")
    with pytest.raises(ValueError, match="no physical kernel"):
        verify_head_kernel_signature(grouped, "nonexistent_lm_head_kernel")


def test_p16a_matched_abba_and_graph_code_not_cpu_branching():
    assert TRIAL_ORDERS == (
        ("conditional", "unconditional"),
        ("unconditional", "conditional"),
        ("unconditional", "conditional"),
        ("conditional", "unconditional"),
    )
    round_code = inspect.getsource(gpu_round)
    chain_code = inspect.getsource(two_gpu_rounds)
    profile_code = inspect.getsource(profile_branch_costs)
    assert "bonus_policy == \"conditional\"" in round_code
    assert "bonus_policy == \"always_compute_mask\"" in round_code
    assert "mark_bonus(after_hidden)" in round_code
    assert "mark_round(hidden)" in round_code
    assert "conditional_graph=graph" in inspect.getsource(
        __import__(
            "sglang_omni.models.qwen3_tts.code_predictor_spec_p15_two_round",
            fromlist=["capture_two_rounds"],
        ).capture_two_rounds
    )
    assert "graph.replay()" in profile_code
    assert "torch.cuda.nvtx.range_push(\"P16A_TRACE\")" in profile_code
    assert "trace_markers=True" in profile_code or (
        'trace_markers=trace_markers' in profile_code
    )
    for source in (round_code, chain_code):
        for forbidden in (
            ".item(", ".cpu(", "torch.multinomial(",
            "torch.cuda.synchronize(",
        ):
            assert forbidden not in source

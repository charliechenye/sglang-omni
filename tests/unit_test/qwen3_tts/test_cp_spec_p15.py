# SPDX-License-Identifier: Apache-2.0
"""P15 contract tests: independent report gates and GPU-only source structure."""
from __future__ import annotations

import copy
import inspect

import pytest

from experiments.qwen3_tts_cp_spec_k2.p15_two_round_gate import (
    GO, NO_GO, PASS, INCOMPLETE, SCHEMA, validate_probe,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p15_two_round import (
    FIRST_DEPTH, FIRST_CACHE_LEN, ROUND_FIELDS, CASES,
    gpu_round, two_gpu_rounds,
)

SHA = "a" * 40
PID = 123
GPU = "NVIDIA H200"


def synthetic_row(transitions=("AA", "AR", "RA", "RR")):
    cases = []
    counts = {key: 0 for key in ("AA", "AR", "RA", "RR")}
    for index in range(16):
        transition = transitions[index % len(transitions)]
        first_accept = transition[0] == "A"
        second_accept = transition[1] == "A"
        first_depth = 10 + int(first_accept)
        first_offset = 11 + int(first_accept)
        cases.append({
            "case_index": index,
            "uniforms": [[0.125, 0.25, 0.375, 0.625]] * 2,
            "transition": transition,
            "full_physical_kv_parity": True,
            "independent_dense_sampler_parity": True,
            "device_state_consumed_by_round2": True,
            "round1": {
                "draft": 10, "committed": 11, "accepted": first_accept,
                "bonus": 12 if first_accept else 0,
                "bonus_executed": int(first_accept),
                "next_cache_len": first_offset,
                "next_depth": first_depth,
            },
            "round2": {
                "draft": 13, "committed": 14, "accepted": second_accept,
                "bonus": 15 if second_accept else 0,
                "bonus_executed": int(second_accept),
                "next_cache_len": first_offset + 1 + int(second_accept),
                "next_depth": first_depth + 1 + int(second_accept),
            },
        })
        counts[transition] += 1
    status = PASS if len(transitions) == 4 else INCOMPLETE
    return {
        "schema": SCHEMA, "status": status, "source_sha": SHA,
        "pid": PID, "gpu": GPU,
        "predictor_layers": 5,
        "backend": "matched_fused_predictor_device_offset",
        "initial_depth": 9, "initial_cache_len": 10,
        "sampling": {"top_k": 50},
        "captured_graph_count": 1,
        "capture_contains_gpu_second_round_cond": True,
        "capture_contains_conditional_bonus": True,
        "bonus_kernel_skip_trace_proven": False,
        "native_seeded_sampler_parity_proven": False,
        "cases": cases, "branch_counts": counts,
        "cuda_graph_event_ms_blocks": [0.2, 0.21, 0.2, 0.22],
        "median_cuda_graph_event_ms": 0.205,
    }


def test_p15_cpu_source_contract_causal_and_gpu_only():
    assert FIRST_DEPTH == 9 and FIRST_CACHE_LEN == 10
    assert len(CASES) == 16
    assert len(ROUND_FIELDS) == 16
    round_code = inspect.getsource(gpu_round)
    chain_code = inspect.getsource(two_gpu_rounds)
    assert round_code.index("q_logits = head(hidden)") < round_code.index(
        "dynamic_fused_predictor("
    ) < round_code.index("p_logits = head(before)") < round_code.index(
        "decision = gpu_accept("
    )
    assert "bonus_head(after_hidden)" in round_code
    assert "cond_op(" in round_code
    assert "bonus_no" in round_code
    assert "dynamic_fused_predictor(" in round_code
    assert "first[ROUND_FIELDS.index(\"next_cache_len\")]" in chain_code
    assert "first[ROUND_FIELDS.index(\"next_hidden\")]" in chain_code
    assert "first[ROUND_FIELDS.index(\"next_embedding\")]" in chain_code
    assert "first[ROUND_FIELDS.index(\"next_sum\")]" in chain_code
    assert "cond_op(" in chain_code
    for code in (round_code, chain_code):
        for forbidden in (
            ".item(", ".cpu(", ".tolist(", "torch.cuda.synchronize(",
            "torch.multinomial(", "round_idx.item(", "cache_len_gpu.item(",
        ):
            assert forbidden not in code


def test_p15_cpu_full_four_transitions_go_not_production():
    row = synthetic_row()
    assert validate_probe(row, sha=SHA, pid=PID, gpu=GPU) == GO
    assert "14DEPTH_OR_E2E" in row["status"]
    assert row["bonus_kernel_skip_trace_proven"] is False
    assert row["native_seeded_sampler_parity_proven"] is False


def test_p15_cpu_insufficient_transition_coverage_no_go():
    row = synthetic_row(("AR", "RA"))
    assert validate_probe(row, sha=SHA, pid=PID, gpu=GPU) == NO_GO
    row["status"] = PASS
    with pytest.raises(ValueError, match="exaggerates"):
        validate_probe(row, sha=SHA, pid=PID, gpu=GPU)


@pytest.mark.parametrize("field,value", [
    ("source_sha", "b" * 40),
    ("captured_graph_count", 2),
    ("bonus_kernel_skip_trace_proven", True),
    ("native_seeded_sampler_parity_proven", True),
    ("capture_contains_gpu_second_round_cond", False),
])
def test_p15_cpu_fail_closed_on_mixed_evidence(field, value):
    row = synthetic_row()
    row[field] = value
    with pytest.raises(ValueError, match="mixed-source"):
        validate_probe(row, sha=SHA, pid=PID, gpu=GPU)


def test_p15_cpu_case_falsified_state_or_kv_must_fail():
    for modify in ("kv", "depth", "bonus", "dense"):
        row = synthetic_row()
        row = copy.deepcopy(row)
        case = row["cases"][0]
        if modify == "kv":
            case["full_physical_kv_parity"] = False
        elif modify == "depth":
            case["round2"]["next_cache_len"] += 1
        elif modify == "bonus":
            case["round1"]["bonus_executed"] = 0
        else:
            case["independent_dense_sampler_parity"] = False
        with pytest.raises(ValueError):
            validate_probe(row, sha=SHA, pid=PID, gpu=GPU)

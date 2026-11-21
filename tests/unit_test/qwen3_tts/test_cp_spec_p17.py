# SPDX-License-Identifier: Apache-2.0
"""P17 CPU source, acceptance and source-locked wide-cost gates."""
from __future__ import annotations

import copy
import inspect
import statistics
from types import SimpleNamespace

import pytest
import torch

from experiments.qwen3_tts_cp_spec_k2.p17_multiwidth_gate import (
    SCHEMA, PROBE_PASS, validate_probe,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p17_wide import (
    WIDTHS, UNIFORM_COUNT, prefix_accept_count, check_wide_backend,
    make_q_proposals, wide_device_fused_predictor,
    make_wide_round,
)

SHA = "a" * 40
PID = 12345
GPU = "NVIDIA H200"


def synthetic_probe():
    widths = {}
    for w in WIDTHS:
        cases = []
        for i in range(16):
            prefix = i % w
            cases.append({
                "index": i, "accepted_prefix": prefix,
                "committed_count": prefix+1,
                "first_rejection": prefix if prefix < w-1 else None,
                "bonus_committed": prefix == w-1,
            })
        hist = {str(x): sum(row["committed_count"] == x for row in cases)
                for x in range(1, w+1)}
        accept = {str(x): sum(row["accepted_prefix"] == x for row in cases)
                  for x in range(w)}
        timing = {}
        medians = {}
        for name in ("full_round", "proposal_only", "verifier_only"):
            indices = (0,5,10,15) if name != "verifier_only" else (0,)
            timing[name] = [
                {
                    "case_index": idx, "repeat_index": rep,
                    "cuda_event_ms": 0.100 + 0.025*w + rep*0.002 + idx*0.00001,
                }
                for idx in indices for rep in range(4)
            ]
            medians[name] = statistics.median(
                row["cuda_event_ms"] for row in timing[name]
            )
        mean = statistics.mean(x["committed_count"] for x in cases)
        widths[str(w)] = {
            "on_policy_real_checkpoint": True,
            "batched_multihead_and_static_fused_parity": True,
            "full_physical_kv_parity": True,
            "independent_cpu_accept_residual_parity": True,
            "case_count": 16,
            "accepted_prefix_histogram": accept,
            "committed_count_histogram": hist,
            "mean_committed_per_round_fixed_checkpoint": mean,
            "timed_full_and_proposal_case_indices": [0, 5, 10, 15],
            "verifier_only_frozen_case_index": 0,
            "cuda_graph_event_ms_blocks": timing,
            "cuda_graph_event_ms_medians": medians,
            "full_round_ms_per_committed_code_SCREEN_ONLY":
                medians["full_round"] / mean,
            "cases": cases,
        }
    return {
        "schema": SCHEMA, "status": PROBE_PASS,
        "source_sha": SHA, "pid": PID, "gpu": GPU,
        "last_committed_depth": 9, "cache_len": 10,
        "predictor_layers": 5,
        "backend": "P17_real_fused_bf16_5layer_wide",
        "sampling": {"top_k": 50},
        "uniform_cases_per_width": 16,
        "widths": widths,
    }


def test_p17_cpu_all_width_source_locked_gate():
    assert WIDTHS == (2,3,4) and UNIFORM_COUNT == 13
    validate_probe(synthetic_probe(), sha=SHA, pid=PID, gpu=GPU)


@pytest.mark.parametrize("flags,expected", [
    ((), 0), ((False,), 0), ((True,), 1),
    ((True,False,True), 1), ((True,True,False), 2),
    ((True,True,True), 3), ((False,True,True), 0),
])
def test_p17_first_rejection_stops_commitment(flags, expected):
    assert prefix_accept_count(flags) == expected


def test_p17_cpu_source_contract_true_multihead_single_wide():
    q_source = inspect.getsource(make_q_proposals)
    verify_source = inspect.getsource(wide_device_fused_predictor)
    round_source = inspect.getsource(make_wide_round)
    assert "old_hidden.expand(n, -1, -1)" in q_source
    assert "head_bank[:n]" in q_source
    assert "fast_selector(" in q_source
    assert "wide_device_fused_predictor(" in round_source
    assert round_source.index("make_q_proposals(") < round_source.index(
        "wide_device_fused_predictor("
    ) < round_source.index("batched_head_logits(")
    assert "torch.cumprod" in round_source
    assert "gpu_accept(" in round_source
    assert "torch.where(\n        full_accept" in round_source
    assert "decode_attention_with_device_end" in verify_source
    for source in (q_source, verify_source, round_source):
        for forbidden in (
            ".item(", ".cpu(", ".tolist(", "torch.multinomial(",
            "torch.cuda.synchronize(",
        ):
            assert forbidden not in source


def test_p17_no_hidden_width_downgrade():
    class FakeModel:
        predictor_cutedsl_layers = None
        predictor_fused_layers = SimpleNamespace(
            covers=lambda n: n <= 4,
            shape=SimpleNamespace(num_q_heads=16, num_kv_heads=4),
        )
        code_predictor = SimpleNamespace(
            model=SimpleNamespace(layers=[0,1,2,3,4]),
        )
        predictor_k_cache = torch.zeros((5,1,16,4,128))
        predictor_v_cache = torch.zeros((5,1,16,4,128))
    model = FakeModel()
    for width in WIDTHS:
        check_wide_backend(model, width, cache_len=10)
    model.predictor_fused_layers.shape.num_kv_heads = 2
    with pytest.raises(RuntimeError, match="ATTENTION_TILE_TOO_NARROW"):
        check_wide_backend(model, width=4, cache_len=10)


@pytest.mark.parametrize("mutator", [
    lambda row: row.update(source_sha="b"*40),
    lambda row: row["widths"].pop("4"),
    lambda row: row["widths"]["4"]["cases"][0].update(committed_count=5),
    lambda row: row["widths"]["3"]["committed_count_histogram"].update({"1": 999}),
    lambda row: row["widths"]["2"].update(full_physical_kv_parity=False),
    lambda row: row["widths"]["4"]["cuda_graph_event_ms_blocks"]["full_round"].pop(),
    lambda row: row["widths"]["4"]["cuda_graph_event_ms_medians"].update(full_round=999.),
])
def test_p17_cpu_source_and_acceptance_evidence_fail_closed(mutator):
    report = copy.deepcopy(synthetic_probe())
    mutator(report)
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate_probe(report, sha=SHA, pid=PID, gpu=GPU)

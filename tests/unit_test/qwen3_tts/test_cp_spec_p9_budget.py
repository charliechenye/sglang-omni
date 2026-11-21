# SPDX-License-Identifier: Apache-2.0
"""P9 component budget screen refuses missing evidence and E2E speedup claims."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.qwen3_tts_cp_spec_k2.p9_budget_screen import (
    NO_GAIN, POSSIBLE, analyze, linear_cost, write_report,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p9a import (
    SCHEMA as A_SCHEMA,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p9b import (
    SCHEMA as B_SCHEMA,
)


def make_evidence(root: Path, *, sampler: float = 0.02, slope: float = 0.08):
    a = root / "four"
    b = root / "sampler"
    base = root / "graph"
    for d in (a, b, base):
        d.mkdir()
    a_report = {
        "schema": A_SCHEMA,
        "status": "P9A_FOUR_REAL_PREDICTOR_OFFSET_GRAPH_PARITY_PASS_NOT_FULL_K2_SPEEDUP",
        "source_sha": "a" * 40,
        "gpu": "NVIDIA H200",
        "predictor_layers": 5,
        "cases": [{"pass": True}] * 6,
        "cost_by_dynamic_stage_sequence": [
            {
                "case": str(i), "model_calls": n,
                "cuda_event_ms_per_graph": 0.01 + slope * n,
            }
            for i, n in enumerate((2, 3, 4, 3, 3, 3))
        ],
    }
    (a / "four_depth_pid100_frame0001.json").write_text(json.dumps(a_report))
    for depth in (2, 10):
        row = {
            "schema": B_SCHEMA,
            "status": "P9B_REAL_LOGITS_GPU_PROPOSAL_ACCEPT_RESIDUAL_AND_RNG_PASS_NOT_FULL_K2",
            "source_sha": "a" * 40,
            "gpu": "NVIDIA H200",
            "proposal_depth": depth,
            "independent_graph_rng_replay_unique": True,
            "output_input_parity": True,
            "checks": [{"pass": True}] * 4,
            "gpu_sampler_with_independent_rng_graph_ms": sampler,
            "overlap": .5,
            "distribution_p_max_abs": 0.,
            "distribution_q_max_abs": 0.,
        }
        (b / f"sample_pid100_frame0001_depth{depth:02d}.json").write_text(json.dumps(row))
    graph = {
        "kind": "baseline_full_predictor_graph_on",
        "source_sha": "b" * 40,
        "device": "NVIDIA H200",
        "frames": 32,
        "graph_replays": 32,
        "median_cuda_event_ms": 2.2266,
    }
    (base / "baseline_graph_pid100.json").write_text(json.dumps(graph))
    p2 = root / "p2.json"
    p2.write_text(json.dumps({
        "source_sha": "c" * 40,
        "frames": 2,
        "total_proposals": 20,
        "total_accepted": 8,
        "total_rejected": 12,
        "total_bonus": 8,
        "depths": [
            {"residual_depth": depth,
             "proposals": 2 if depth < 8 else 1}
            for depth in range(2, 16)
        ],
    }))
    # Sum the per-depth n exactly to 20 (6*2+8*1).
    return a, b, base, p2


def test_regression_linear_cost_fits_per_round_and_intercept():
    intercept, slope = linear_cost([
        (2, .35), (3, .52), (4, .69),
    ])
    assert intercept == pytest.approx(.01)
    assert slope == pytest.approx(.17)
    with pytest.raises(ValueError, match="2, 3 AND 4"):
        linear_cost([(2,.2),(3,.3),(3,.4)])


def test_component_screen_can_fail_and_cannot_claim_serving_speedup(tmp_path):
    a,b,base,p2 = make_evidence(tmp_path)
    output = analyze(a,b,base,p2)
    assert output["status"] == POSSIBLE
    assert output["source_locked_p9_components"]
    assert output["p9a_model_round_slope_ms"] == pytest.approx(.08)
    assert output["historical_accepted_proposals_per_frame"] == 10
    assert output["remaining_baseline_budget_ms_per_frame"] > 0
    assert "NOT AN ACTUAL K2 SPEEDUP TEST" in output["limits"][0]
    file = write_report(tmp_path / "report", output)
    assert file.exists()
    with pytest.raises(FileExistsError):
        write_report(tmp_path / "report", output)

    a_file = next(a.glob("four_depth*.json"))
    row = json.loads(a_file.read_text())
    for cost in row["cost_by_dynamic_stage_sequence"]:
        cost["cuda_event_ms_per_graph"] = .01 + 0.20 * cost["model_calls"]
    a_file.write_text(json.dumps(row))
    negative = analyze(a,b,base,p2)
    assert negative["status"] == NO_GAIN
    assert negative["remaining_baseline_budget_ms_per_frame"] < 0


def test_budget_rejects_missing_real_checkpoint_evidence(tmp_path):
    a,b,base,p2 = make_evidence(tmp_path)
    (next(b.glob("*depth10.json"))).unlink()
    with pytest.raises(RuntimeError, match="missing/mixed"):
        analyze(a,b,base,p2)

    a_file = next(a.glob("four_depth*.json"))
    row = json.loads(a_file.read_text())
    row["status"] = "P9A_FAILED"
    a_file.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="missing/mixed"):
        analyze(a,b,base,p2)

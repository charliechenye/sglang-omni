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
from experiments.qwen3_tts_cp_spec_k2.p4_cost_screen import validate_acceptance

RECOVERED_P2 = (
    Path(__file__).resolve().parents[3]
    / "experiments/qwen3_tts_cp_spec_k2/data/p2_720_historical_aggregate_reconstructed.json"
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
        "source_sha": "a" * 40,
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
        cost["cuda_event_ms_per_graph"] = .01 + 0.25 * cost["model_calls"]
    a_file.write_text(json.dumps(row))
    negative = analyze(a,b,base,p2)
    assert negative["status"] == NO_GAIN
    assert negative["remaining_baseline_budget_ms_per_frame"] < 0


def test_budget_rejects_missing_real_checkpoint_evidence(tmp_path):
    a,b,base,p2 = make_evidence(tmp_path)
    old = next(base.glob("baseline_graph*.json"))
    record = json.loads(old.read_text())
    record["source_sha"] = "z" * 40
    old.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="SAME source SHA"):
        analyze(a,b,base,p2)
    record["source_sha"] = "a" * 40
    old.write_text(json.dumps(record))
    (next(b.glob("*depth10.json"))).unlink()
    with pytest.raises(RuntimeError, match="missing/mixed"):
        analyze(a,b,base,p2)

    a_file = next(a.glob("four_depth*.json"))
    row = json.loads(a_file.read_text())
    row["status"] = "P9A_FAILED"
    a_file.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="missing/mixed"):
        analyze(a,b,base,p2)


def test_historical_p2_720_aggregate_recovers_only_reported_counts():
    """No synthetic per-frame parity proof: preserve exact historical census."""
    raw = json.loads(RECOVERED_P2.read_text())
    result = validate_acceptance(raw)
    assert raw["provenance"]["is_original_report"] is False
    assert raw["provenance"]["is_raw_trace"] is False
    assert raw["source_sha"] == "398297b1d4d2a44d10d1056da5607a7d4ac08e41"
    assert result["frames"] == 720
    assert result["proposals"] == 6974
    assert result["accepted"] == 3420
    assert result["rejected"] == 3554
    assert result["bonus"] == 3106
    assert result["codes_per_round"] == pytest.approx(10080 / 6974)
    assert sum(row["accepted"] for row in result["depths"]) == 3420
    assert sum(row["rejected"] for row in result["depths"]) == 3554
    assert sum(row["bonus_skipped_proposals"] for row in result["depths"]) == 3106
    assert all(
        row["proposals"] + row["bonus_skipped_proposals"] == 720
        for row in result["depths"]
    )


def test_p9_budget_explicitly_identifies_reconstructed_p2_provenance(tmp_path):
    a, b, base, _ = make_evidence(tmp_path)
    report = analyze(a, b, base, RECOVERED_P2)
    assert report["p2_acceptance_reconstructed"] is True
    assert report["p2_acceptance_provenance"] == (
        "RECONSTRUCTED_PRIOR_USER_REPORTED_AGGREGATE_NOT_RAW_FRAMES"
    )
    assert report["historical_accepted_proposals_per_frame"] == pytest.approx(6974 / 720)
    assert report["p2_acceptance_sha"] != report["p9_source_sha"]
    markdown = write_report(tmp_path / "reconstructed_report", report).read_text()
    assert "RECONSTRUCTED_PRIOR_USER_REPORTED_AGGREGATE_NOT_RAW_FRAMES" in markdown
    assert "original per-frame traces/parity proof do NOT exist" in markdown


def test_p9_rejects_corrupted_historical_aggregate(tmp_path):
    a, b, base, _ = make_evidence(tmp_path)
    broken = json.loads(RECOVERED_P2.read_text())
    broken["depths"][0]["accepted"] += 1
    wrong = tmp_path / "broken.json"
    wrong.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Historical P2 reconstruction"):
        analyze(a, b, base, wrong)

    broken = json.loads(RECOVERED_P2.read_text())
    broken["provenance"]["is_raw_trace"] = True
    wrong.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Historical P2 reconstruction"):
        analyze(a, b, base, wrong)

# SPDX-License-Identifier: Apache-2.0
"""P10 complete source-locked component screen is honest and fail-closed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.qwen3_tts_cp_spec_k2.p10_cost_gate import (
    PASS, STOP, analyze, write,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p10a_dynamic_kv import (
    SCHEMA as A_SCHEMA,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p10b_sparse_sampler import (
    SCHEMA as B_SCHEMA,
)

P2 = (
    Path(__file__).resolve().parents[3]
    / "experiments/qwen3_tts_cp_spec_k2/data/"
    "p2_720_historical_aggregate_reconstructed.json"
)


def evidence(root, *, dynamic=.15, static=.14, sparse=.03, dense=.11):
    paths = [root / x for x in ("a", "b", "base_a", "base_b")]
    for path in paths:
        path.mkdir()
    a, b, base_a, base_b = paths
    sha = "a" * 40
    gpu = "NVIDIA H200"
    a_data = {
        "schema": A_SCHEMA,
        "status": "P10A_ONE_GRAPH_GPU_DYNAMIC_KV_FUSED_PREDICTOR_PARITY_PASS_NOT_FULL_K2_OR_SERVING_SPEEDUP",
        "source_sha": sha,
        "gpu": gpu,
        "predictor_layers": 5,
        "backend": "same_production_fused_predictor_layers",
        "cases": [
            {
                "depth": x, "pass": True,
                "dynamic_cuda_graph_ms": dynamic,
                "fixed_cuda_graph_ms": static,
            } for x in (2,6,10,14)
        ],
    }
    (a / "dynamic_kv_pid100_frame0001.json").write_text(json.dumps(a_data))
    for depth in (2,10):
        b_data = {
            "schema": B_SCHEMA,
            "status": "P10B_REAL_PQ_SPARSE_TOPK_SPEC_SAMPLER_PARITY_PASS_NOT_FULL_K2_OR_NATIVE_RNG",
            "source_sha": sha, "gpu": gpu,
            "proposal_depth": depth,
            "top_k": 50, "vocab": 2048,
            "private_rng_replay_unique": True,
            "input_logits_immutable": True,
            "checks": [{"pass": True}] * 4,
            "p_max_abs": 0., "q_max_abs": 0.,
            "overlap": .5,
            "sparse_fused_cuda_graph_ms": sparse,
            "p9b_dense_reference_cuda_graph_ms": dense,
        }
        (b / f"sparse_pid100_frame0001_depth{depth:02d}.json").write_text(
            json.dumps(b_data)
        )
    for folder, measurement in ((base_a,2.24),(base_b,2.23)):
        (folder / "baseline_graph_pid100.json").write_text(json.dumps({
            "kind": "baseline_full_predictor_graph_on",
            "frames": 32, "graph_replays": 32,
            "source_sha": sha, "device": gpu,
            "median_cuda_event_ms": measurement,
        }))
    return a,b,base_a,base_b


def test_p10_component_budget_honest_positive_and_negative(tmp_path):
    a,b,base_a,base_b = evidence(tmp_path)
    report = analyze(a,b,base_a,base_b,P2)
    assert report["status"] == PASS
    assert report["remaining_graph_on_budget_ms_per_frame"] > .30
    assert report["historical_p2_aggregate_reconstructed"] is True
    assert report["historical_rounds_per_frame"] == pytest.approx(6974/720)
    assert report["p10a_dynamic_median_ms_per_round"] == pytest.approx(.15)
    assert report["p10b_sparse_sampler_median_ms_per_round"] == pytest.approx(.03)
    path = write(tmp_path / "summary", report)
    assert "A positive reserve DOES NOT prove full GPU-resident K2" in path.read_text()
    assert "NO" not in report["status"]  # positive cost gate, still not E2E evidence
    assert report["limits"][-1].startswith("Graph-ON baseline is")
    with pytest.raises(FileExistsError):
        write(tmp_path / "summary", report)

    a_json = next(a.glob("dynamic_kv*.json"))
    record = json.loads(a_json.read_text())
    for row in record["cases"]:
        row["dynamic_cuda_graph_ms"] = .24
    a_json.write_text(json.dumps(record))
    b_json = next(b.glob("sparse*depth02.json"))
    record_b = json.loads(b_json.read_text())
    record_b["sparse_fused_cuda_graph_ms"] = .12
    b_json.write_text(json.dumps(record_b))
    status = analyze(a,b,base_a,base_b,P2)
    assert status["status"] == STOP
    assert status["remaining_graph_on_budget_ms_per_frame"] < .30


def test_p10_rejects_sha_mismatch_missing_parity_and_baseline_drift(tmp_path):
    a,b,base_a,base_b = evidence(tmp_path)
    b_json = next(b.glob("sparse*depth02.json"))
    row = json.loads(b_json.read_text())
    row["source_sha"] = "z"*40
    b_json.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="source SHA mismatch"):
        analyze(a,b,base_a,base_b,P2)
    row["source_sha"] = "a"*40
    row["checks"][0]["pass"] = False
    b_json.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="parity"):
        analyze(a,b,base_a,base_b,P2)
    row["checks"][0]["pass"] = True
    b_json.write_text(json.dumps(row))

    baseline_file = next(base_b.glob("baseline_graph*.json"))
    baseline = json.loads(baseline_file.read_text())
    baseline["median_cuda_event_ms"] = 2.7
    baseline_file.write_text(json.dumps(baseline))
    with pytest.raises(RuntimeError, match="drift"):
        analyze(a,b,base_a,base_b,P2)


def test_p10_does_not_accept_fabricated_raw_p2_cohort(tmp_path):
    a,b,base_a,base_b = evidence(tmp_path)
    raw = json.loads(P2.read_text())
    raw["provenance"]["is_raw_trace"] = True
    path = tmp_path / "fabricated.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="historical"):
        analyze(a,b,base_a,base_b,path)

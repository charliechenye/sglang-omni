# SPDX-License-Identifier: Apache-2.0
"""P18 CPU acceptance/cost gates; no H200 result is implied."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from experiments.qwen3_tts_cp_spec_k2.p18_analyze import (
    clustered_ci, seed_for, validate_rows, widths,
)
from experiments.qwen3_tts_cp_spec_k2.p18_cost_model import (
    estimated_cost, paired_bootstrap,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p18_census import (
    armed, feasible_widths, uniform_seed,
)
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

SHA = "a" * 40


def test_p18_only_legal_end_of_model_widths():
    assert feasible_widths(1) == (2, 3, 4)
    assert feasible_widths(11) == (2, 3, 4)
    assert feasible_widths(12) == (2, 3)
    assert feasible_widths(13) == (2,)
    assert feasible_widths(14) == ()
    for d in range(1, 15):
        assert feasible_widths(d) == widths(d)
    with pytest.raises(ValueError):
        feasible_widths(0)
    with pytest.raises(ValueError):
        feasible_widths(15)


def test_p18_uniforms_are_paired_and_source_locked():
    for depth in range(1, 15):
        key = ("p00_s20261123", 17, depth)
        assert uniform_seed(SHA, *key) == seed_for(SHA, *key)
        assert uniform_seed(SHA, *key) == uniform_seed(SHA, *key)
    assert uniform_seed(SHA, "case-A", 17, 2) != uniform_seed(
        SHA, "case-B", 17, 2
    )
    assert uniform_seed(SHA, "case-A", 17, 2) != uniform_seed(
        "b" * 40, "case-A", 17, 2
    )


def test_p18_arm_only_when_both_env_and_file_exist(tmp_path, monkeypatch):
    arm = tmp_path / "ARM_P18"
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P18_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P18_ARM_FILE", str(arm))
    assert armed() is False
    arm.touch()
    assert armed() is True
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_P18_DIR")
    assert armed() is False


def test_p18_native_ar_hook_precedes_kv_append():
    source = inspect.getsource(Qwen3TTSTalker.code_predictor_forward_incremental)
    assert "native_code=pos_codes[:, layer_idx + 1]" in source
    assert source.index("maybe_census(") < source.index(
        "last_hidden = self.predictor_forward_tokens(",
        source.index("maybe_census("),
    )
    top = inspect.getsource(Qwen3TTSTalker.code_predictor_forward)
    assert top.index("P18_REQUIRES_NATIVE_AR_EAGER_K2_DISABLED") < top.index(
        "speculative_k2("
    )


def test_p18_request_cluster_bootstrap_not_frame_bootstrap():
    groups = {
        "request_A": [1.0] * 8,
        "request_B": [3.0] * 8,
        "request_C": [5.0] * 8,
    }
    out = clustered_ci(groups, seed=22, repeats=200)
    assert out["mean"] == 3
    assert out["independent_request_clusters"] == 3
    assert out["frames_within_clusters_NOT_independent"] == 24
    lo, hi = out["cluster_bootstrap_95pct"]
    assert 1 <= lo <= 3 <= hi <= 5


def test_p18_paired_width_cost_uses_request_groups():
    cases = {
        "r1": {2: [1, 1], 3: [1, 1], 4: [1, 1]},
        "r2": {2: [2, 2], 3: [2, 2], 4: [2, 2]},
        "r3": {2: [1, 2], 3: [2, 2], 4: [3, 3]},
    }
    scores = paired_bootstrap(
        cases, {2: 0.25, 3: 0.29, 4: 0.32}, seed=3
    )
    assert set(scores) == {"2", "3", "4"}
    assert scores["2"]["advantage_vs_K2_paired_request_bootstrap"]["nominal"] == 0
    assert (
        scores["3"]["advantage_vs_K2_paired_request_bootstrap"]
        ["interval_95pct_ACCEPTANCE_ONLY"][0]
        <= scores["3"]["advantage_vs_K2_paired_request_bootstrap"]
        ["interval_95pct_ACCEPTANCE_ONLY"][1]
    )


def test_p18_cost_extrapolation_is_not_misidentified_as_measurement():
    cost = {"depths": {
        1: {w: {"median_ms": 0.2 + w * 0.02,
                "checkpoint_medians_ms": [0.2, 0.3]}
            for w in (2, 3, 4)},
        9: {w: {"median_ms": 0.22 + w * 0.02,
                "checkpoint_medians_ms": [0.21, 0.31]}
            for w in (2, 3, 4)},
    }}
    assert estimated_cost(cost, 1, 2)["cost_provenance"] == "P17_ACTUAL_DEPTH"
    assert estimated_cost(cost, 5, 2)["cost_provenance"] == (
        "P17_INTERPOLATED_UNMATCHED"
    )
    assert estimated_cost(cost, 14, 2)["cost_provenance"] == (
        "P17_TAIL_EXTRAPOLATED_UNMATCHED"
    )


def _synthetic_smoke(tmp_path: Path) -> Path:
    """Four independent requests x two frames x 14 depths."""
    root = tmp_path / "smoke"
    (root / "census" / "rows").mkdir(parents=True)
    (root / "census" / "failures").mkdir()
    (root / "config.json").write_text(json.dumps({
        "source_sha": SHA, "native_k2": "OFF",
        "trajectory": "native-ar-eager-only", "frames_per_request": 2,
    }))
    (root / "source.sha").write_text(SHA + "\n")
    visits = []
    for p in range(2):
        for sd in (20261123, 20261124):
            case_id = f"p{p:02d}_s{sd}"
            prompt = f"p{p}"
            req_hash = hashlib.sha256(case_id.encode()).hexdigest()
            nonce = hashlib.sha256(
                f"{SHA}|{case_id}|p18a-v1".encode()
            ).hexdigest()
            visits.append({
                "case_id": case_id, "prompt_id": prompt,
                "seed": sd, "request_hash": req_hash,
                "source_sha": SHA, "frames": 2, "depth_rows": 28,
                "finish_reason": "stop",
            })
            for frame in (10, 11):
                for depth in range(1, 15):
                    kinds = {}
                    for w in widths(depth):
                        flags = [False] * (w - 1)
                        kinds[str(w)] = {
                            "status": "INDEPENDENT_P17_DISTRIBUTION_AND_KV_PARITY_PASS",
                            "accepted_flags": flags, "accepted_prefix": 0,
                            "committed_count": 1, "first_rejection": 0,
                            "bonus_committed": False,
                        }
                    row = {
                        "schema": "qwen3tts-p18a-native-ar-eager-all-depth-v1",
                        "source_sha": SHA, "trajectory":
                            "NATIVE_AR_EAGER_CODEBOOK_SAMPLING",
                        "probe": "P17_INDEPENDENT_ORACLE_COUNTERFACTUAL",
                        "gpu": "NVIDIA H200", "pid": 17,
                        "case_id": case_id, "prompt_id": prompt,
                        "requested_seed": sd, "request_nonce": nonce,
                        "request_hash": req_hash,
                        "frame": frame, "last_committed_depth": depth,
                        "cache_len": depth + 1,
                        "native_sub_sampling_seed": sd,
                        "native_committed_code": 123,
                        "uniform_seed": seed_for(SHA, case_id, frame, depth),
                        "sampling": {"top_k": 50, "temperature": 0.9,
                                     "top_p": 1.0},
                        "widths": kinds,
                    }
                    name = f"{case_id}_f{frame:03d}_d{depth:02d}.json"
                    (root / "census" / "rows" / name).write_text(
                        json.dumps(row)
                    )
    (root / "visits.json").write_text(json.dumps({
        "schema": "p18a-native-eager-http-driver-v1",
        "source_sha": SHA, "mode": "smoke",
        "status": "P18A_NATIVE_AR_ALL_DEPTH_CENSUS_EVIDENCE_CAPTURED",
        "trajectory": "NATIVE_AR_EAGER_NOT_GRAPH_SERVING",
        "frames_per_request": 2, "visits": visits,
    }))
    return root


def test_p18_smoke_gate_and_independent_request_count(tmp_path):
    root = _synthetic_smoke(tmp_path)
    visits, rows = validate_rows(root)
    assert len(visits["visits"]) == 4
    assert len(rows) == 4 * 2 * 14
    assert all(not r["widths"] for r in rows
               if r["last_committed_depth"] == 14)


def test_p18_missing_depth_or_wrong_source_fails_closed(tmp_path):
    root = _synthetic_smoke(tmp_path)
    one = next((root / "census" / "rows").glob("*.json"))
    one.unlink()
    with pytest.raises(RuntimeError, match="INCOMPLETE_ROW_COUNT"):
        validate_rows(root)
    root = _synthetic_smoke(tmp_path / "second")
    one = next((root / "census" / "rows").glob("*.json"))
    record = json.loads(one.read_text())
    record["source_sha"] = "b" * 40
    one.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="BAD_ROW_PROVENANCE"):
        validate_rows(root)


def test_p18_seeds_must_actually_differ_inside_native_sampler(tmp_path):
    root = _synthetic_smoke(tmp_path)
    for path in (root / "census" / "rows").glob("*.json"):
        row = json.loads(path.read_text())
        row["native_sub_sampling_seed"] = 42
        path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="SEEDS_DID_NOT_PRODUCE_DISTINCT"):
        validate_rows(root)


def test_p18_worker_native_sampling_seed_must_stay_constant_per_case(tmp_path):
    root = _synthetic_smoke(tmp_path)
    one = next((root / "census" / "rows").glob("*.json"))
    record = json.loads(one.read_text())
    record["native_sub_sampling_seed"] += 1
    one.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="SUBSAMPLING_SEED_DRIFT"):
        validate_rows(root)

# SPDX-License-Identifier: Apache-2.0
"""P19 CPU tests: true depth-transition DP and strict paired cost evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.qwen3_tts_cp_spec_k2.p18_analyze import widths
from experiments.qwen3_tts_cp_spec_k2.p18_analyze import seed_for
from experiments.qwen3_tts_cp_spec_k2.p19_analyze import (
    STATUS as COST_PASS, analyze as analyze_costs, validate_costs,
)
from experiments.qwen3_tts_cp_spec_k2.p19_dp import (
    RESULT, analyze as analyze_dp, bellman, fixed_width_policy,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_cost import (
    BLOCKS, REPEATS, SCHEMA,
)

SHA = "a" * 40


def _dp_inputs() -> tuple[dict, dict]:
    costs, probs = {}, {}
    for depth in range(1, 15):
        costs[depth] = {"AR": 1.0}
        probs[depth] = {}
        for w in widths(depth):
            costs[depth][str(w)] = {2: 1.1, 3: 2.7, 4: 4.3}[w]
            # Always accept the first K2 draft, so it advances two codes.
            # K3/K4 make only one progress at higher cost.
            probs[depth][str(w)] = (
                [0.0, 1.0] if w == 2
                else [1.0] + [0.0] * (w - 1)
            )
    return costs, probs


def test_p19_dp_full_accept_skip_over_intermediate_depths():
    costs, probs = _dp_inputs()
    chosen = bellman(costs, probs)
    assert chosen["residual_component_graph_ms"] == pytest.approx(7 * 1.1)
    assert chosen["policy"][1] == "2"
    assert chosen["policy"][13] == "2"
    assert chosen["policy"][14] == "AR"
    assert chosen["per_depth"][1]["action_q_ms"]["2"] < (
        chosen["per_depth"][1]["action_q_ms"]["AR"]
    )


def test_p19_dp_global_skip_beats_greedy_local_ratio():
    costs, probs = _dp_inputs()
    # AR appears locally cheaper at depth1 (1 vs K2 3), but an AR
    # transition lands at very expensive depth2. K2 skips it entirely.
    costs[1]["2"] = 3.0
    for arm in costs[2]:
        costs[2][arm] = 100.0
    for arm in ("3", "4"):
        costs[1][arm] = 250.0
    selected = bellman(costs, probs)
    assert selected["policy"][1] == "2"
    assert selected["per_depth"][1]["action_q_ms"]["2"] < (
        selected["per_depth"][1]["action_q_ms"]["AR"]
    )
    assert costs[1]["2"] / 2 > costs[1]["AR"]


def test_p19_dp_fixed_policy_no_illegal_tail_widths():
    costs, probs = _dp_inputs()
    k4_policy = fixed_width_policy(4)
    assert k4_policy[11] == "4"
    assert all(k4_policy[d] == "AR" for d in (12, 13, 14))
    result = bellman(costs, probs, k4_policy)
    assert result["residual_component_graph_ms"] > 0
    bad = {**k4_policy, 14: "4"}
    with pytest.raises(ValueError, match="ILLEGAL_FIXED_POLICY"):
        bellman(costs, probs, bad)


def test_p19_dp_rejects_unmeasured_and_bad_histograms():
    costs, probs = _dp_inputs()
    costs[4].pop("AR")
    with pytest.raises(ValueError, match="INVALID_ACTION_SET"):
        bellman(costs, probs)
    costs, probs = _dp_inputs()
    probs[9]["3"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="BAD_PREFIX_DISTRIBUTION"):
        bellman(costs, probs)
    costs, probs = _dp_inputs()
    costs[7]["AR"] = float("nan")
    with pytest.raises(ValueError, match="INVALID_COST"):
        bellman(costs, probs)


def _fake_source_locked_run(tmp_path: Path) -> Path:
    """4 real-shaped request clusters x 2 frames x 14 depths.
    Costs measured from FIRST real frame per request, exactly as P19A.
    """
    root = tmp_path / "run"
    for suffix in (
        "census/rows", "census/failures",
        "matched_cost/rows", "matched_cost/failures",
    ):
        (root / suffix).mkdir(parents=True, exist_ok=True)
    (root / "source.sha").write_text(SHA + "\n")
    (root / "config.json").write_text(json.dumps({
        "source_sha": SHA, "native_k2": "OFF",
        "trajectory": "native-ar-eager-only",
        "frames_per_request": 2,
    }))
    (root / "matched_cost" / "config.json").write_text(json.dumps({
        "schema": "qwen3tts-p19a-matched-cost-config-v1",
        "source_sha": SHA, "gpu": "NVIDIA H200",
        "trajectory": "P18_NATIVE_AR_EAGER_SAME_CHECKPOINT",
        "graph_costs": "independently_captured_fixed_state_NOT_FULL_FRAME",
        "native_step_scope": "predictor+lm_head+real_seeded_sampler+embedding",
        "p18_frames_per_request": 2, "p19_cost_frames_per_request": 1,
    }))
    visits = []
    for prompt_index in range(2):
        for seed in (20261123, 20261124):
            case = f"p{prompt_index:02d}_s{seed}"
            prompt = f"prompt{prompt_index}"
            reqhash = hashlib.sha256(case.encode()).hexdigest()
            nonce = hashlib.sha256(
                f"{SHA}|{case}|p18a-v1".encode()
            ).hexdigest()
            visits.append({
                "case_id": case, "prompt_id": prompt,
                "seed": seed, "request_hash": reqhash,
                "source_sha": SHA, "frames": 2,
                "depth_rows": 28, "finish_reason": "stop",
            })
            for frame in (10, 11):
                for depth in range(1, 15):
                    accepted = {}
                    for w in widths(depth):
                        accepted[str(w)] = {
                            "status": "INDEPENDENT_P17_DISTRIBUTION_AND_KV_PARITY_PASS",
                            "accepted_flags": [False] * (w-1),
                            "accepted_prefix": 0, "committed_count": 1,
                            "first_rejection": 0, "bonus_committed": False,
                        }
                    record = {
                        "schema": "qwen3tts-p18a-native-ar-eager-all-depth-v1",
                        "source_sha": SHA,
                        "trajectory": "NATIVE_AR_EAGER_CODEBOOK_SAMPLING",
                        "probe": "P17_INDEPENDENT_ORACLE_COUNTERFACTUAL",
                        "gpu": "NVIDIA H200", "pid": 12345,
                        "case_id": case, "prompt_id": prompt,
                        "requested_seed": seed, "request_nonce": nonce,
                        "request_hash": reqhash, "frame": frame,
                        "last_committed_depth": depth, "cache_len": depth+1,
                        "native_sub_sampling_seed": seed,
                        "native_committed_code": 123,
                        "uniform_seed": seed_for(SHA, case, frame, depth),
                        "sampling": {
                            "top_k": 50, "temperature": 0.9, "top_p": 1.0,
                        },
                        "widths": accepted,
                    }
                    (root / "census" / "rows" /
                     f"{case}_f{frame}_d{depth}.json").write_text(
                        json.dumps(record)
                    )
                    if frame != 10:
                        continue
                    arms = {}
                    for a in ("AR", *(str(w) for w in widths(depth))):
                        baseline = {
                            "AR": 0.100, "2": 0.250,
                            "3": 0.290, "4": 0.325,
                        }[a]
                        blocks = [{
                            "block": block, "replays": REPEATS,
                            "cuda_event_ms": baseline + block*0.0001,
                        } for block in range(BLOCKS)]
                        arms[a] = {
                            "parity": "CAPTURED_REPLAY_VS_EAGER_OUTPUT_AND_FULL_KV_PASS",
                            "replays_per_block": REPEATS,
                            "cuda_graph_event_ms_blocks": blocks,
                            "median_cuda_event_ms": (
                                blocks[1]["cuda_event_ms"] +
                                blocks[2]["cuda_event_ms"]
                            ) / 2,
                        }
                    costrow = {
                        "schema": SCHEMA, "source_sha": SHA,
                        "pid": 12345, "gpu": "NVIDIA H200",
                        "case_id": case, "prompt_id": prompt,
                        "request_hash": reqhash, "request_nonce": nonce,
                        "seed": seed, "frame": frame,
                        "last_committed_depth": depth, "cache_len": depth+1,
                        "trajectory": "NATIVE_AR_EAGER_CHECKPOINT_NOT_SPEC_ROLLOUT",
                        "cost_metric": "CUDA_GRAPH_EVENT_ENVELOPE_FIXED_CHECKPOINT",
                        "matched_native_sampler": "QWEN3_TTS_REAL_SEEDED_SUBTALKER",
                        "uniforms": "P18_SHARED_QP_FIXED",
                        "frames_per_request_measured": 1,
                        "arms": arms,
                    }
                    (root / "matched_cost" / "rows" /
                     f"{case}_f{frame}_d{depth}.json").write_text(
                        json.dumps(costrow)
                    )
    (root / "visits.json").write_text(json.dumps({
        "schema": "p18a-native-eager-http-driver-v1",
        "status": "P18A_NATIVE_AR_ALL_DEPTH_CENSUS_EVIDENCE_CAPTURED",
        "source_sha": SHA, "mode": "smoke",
        "trajectory": "NATIVE_AR_EAGER_NOT_GRAPH_SERVING",
        "frames_per_request": 2,
        "seeds": [20261123, 20261124],
        "visits": visits,
    }))
    return root


def test_p19a_cost_validator_matches_same_p18_native_checkpoint(tmp_path):
    root = _fake_source_locked_run(tmp_path)
    visits, accept, cost = validate_costs(root)
    assert len(visits["visits"]) == 4
    assert len(accept) == 4 * 2 * 14
    assert len(cost) == 4 * 14
    assert set(cost[-1]["arms"]) == {"AR", *(str(w) for w in
                                             widths(cost[-1]["last_committed_depth"]))}


def test_p19a_fail_closed_missing_or_corrupted_matching_cost(tmp_path):
    root = _fake_source_locked_run(tmp_path)
    path = next((root / "matched_cost" / "rows").glob("*.json"))
    path.unlink()
    with pytest.raises(RuntimeError, match="MISSING_COST_DEPTH_ROWS"):
        validate_costs(root)

    root = _fake_source_locked_run(tmp_path / "second")
    path = next((root / "matched_cost" / "rows").glob("*.json"))
    value = json.loads(path.read_text())
    value["source_sha"] = "b" * 40
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="COST_PROVENANCE_DRIFT"):
        validate_costs(root)

    root = _fake_source_locked_run(tmp_path / "third")
    path = next((root / "matched_cost" / "rows").glob("*.json"))
    value = json.loads(path.read_text())
    value["arms"]["AR"]["median_cuda_event_ms"] = 999
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="INVALID_COST_OR_PARITY"):
        validate_costs(root)


def test_p19_analysis_smoke_is_not_serving_go(tmp_path):
    root = _fake_source_locked_run(tmp_path)
    cost_report = analyze_costs(root, root / "matched_cost" / "analysis")
    assert cost_report["status"] == COST_PASS
    dp_report = analyze_dp(root, root / "dp")
    assert dp_report["status"] == RESULT
    assert dp_report["reference"]["initial_pair_included"] is False
    assert dp_report["reference"]["native_whole_predictor_graph_compared"] is False
    assert dp_report["research_decision"] == "SMOKE_NO_SELECTION"
    assert dp_report["nominal_adaptive"]["nominal_improvement_vs_native_AR_components"] >= 0

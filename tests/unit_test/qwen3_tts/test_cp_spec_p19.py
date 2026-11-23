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
    RESULT, _case_tables, analyze as analyze_dp, bellman, fixed_width_policy,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_cost import (
    BLOCKS, REPEATS, SCHEMA, select_cost_frame, _matched_eager_arm,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_profile import (
    PROFILE_ENV, PROFILES, DEFAULT_OPS, DEFAULT_REPLAYS, MIN_REPLAY_OPS,
    current_profile, profile_counts, validate_profile,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_diag import (
    MODES as DIAG_MODES, diagnostic_config, event as diag_event,
    maybe_tail_runner_event,
)
from experiments.qwen3_tts_cp_spec_k2.p19_diag_coverage import (
    GRAPH_STAGES, EAGER_STAGES, DEPTH_STAGES,
    arms_at_depth, validate_diagnostic_trace, stages_for_profile,
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


def test_p19_dp_ar_only_tail_depth_has_no_speculative_histogram(tmp_path):
    """Depth 14 has measured AR cost but zero legal K widths / prefix rows.

    Regression: an empty histogram dict used to fail the unconditional
    len({len(v) for v in native_probs.values()}) == 1 check.
    """
    root = _fake_source_locked_run(tmp_path)
    driver, acceptance, cost_rows = validate_costs(root)
    per_case, prompts, prompt_map = _case_tables(
        driver, acceptance, cost_rows,
    )
    assert len(per_case) == 4
    assert len(prompts) == 2
    assert len(prompt_map) == 4
    for case in per_case:
        assert set(per_case[case][14]["cost"]) == {"AR"}
        assert per_case[case][14]["counts"] == {}
        # The penultimate depth must still require native AR AND K2.
        assert set(per_case[case][13]["cost"]) == {"AR", "2"}
        assert set(per_case[case][13]["counts"]) == {"2"}
        assert per_case[case][13]["counts"]["2"] == [2, 0]


def test_p19_dp_rejects_missing_earlier_depth_speculative_cost(tmp_path):
    """The AR-only tail exception must NOT hide missing K2 at depth13."""
    root = _fake_source_locked_run(tmp_path)
    driver, acceptance, cost_rows = validate_costs(root)
    for row in cost_rows:
        if row["last_committed_depth"] == 13:
            row["arms"].pop("2")
    with pytest.raises(RuntimeError, match="P19_DP_UNPAIRED_REAL_NATIVE_CASES"):
        _case_tables(driver, acceptance, cost_rows)


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


def test_p19_cuda700_diag_modes_gate_and_legal_depth(monkeypatch):
    for key in (
        "SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM",
        "SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH",
    ):
        monkeypatch.delenv(key, raising=False)
    assert diagnostic_config() is None
    assert len(DIAG_MODES) == 12
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH", "1")
    for mode in DIAG_MODES:
        monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", mode)
        assert diagnostic_config() == (mode, 1)
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", "K4_GRAPH")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH", "12")
    with pytest.raises(RuntimeError, match="ILLEGAL_K_AT_DEPTH"):
        diagnostic_config()
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", "K2_GRAPH")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH", "14")
    with pytest.raises(RuntimeError, match="ILLEGAL_K_AT_DEPTH"):
        diagnostic_config()
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", "AR_GRAPH")
    assert diagnostic_config() == ("AR_GRAPH", 14)
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", "GRAPH_ALL")
    with pytest.raises(RuntimeError, match="UNKNOWN_DIAGNOSTIC_ARM"):
        diagnostic_config()


def test_p19_diag_cpu_only_durable_phase_trace(tmp_path, monkeypatch):
    from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_diag import (
        SCHEMA as DIAG_SCHEMA,
    )
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", "AR_GRAPH")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH", "9")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P19_COST_DIR", str(tmp_path))
    diag_event("P19_ARM_CAPTURE_BEGIN", depth=9, frame=17, arm="AR")
    diag_event("P19_ARM_FIRST_REPLAY_SYNC_PASS", depth=9, frame=17, arm="AR")
    files = list((tmp_path / "diagnostic").glob("worker_pid*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(x) for x in files[0].read_text().splitlines()]
    assert [r["stage"] for r in rows] == [
        "P19_ARM_CAPTURE_BEGIN", "P19_ARM_FIRST_REPLAY_SYNC_PASS",
    ]
    assert all(r["schema"] == DIAG_SCHEMA for r in rows)
    assert all(r["depth"] == 9 and r["arm"] == "AR" for r in rows)
    assert rows[1]["monotonic_ns"] >= rows[0]["monotonic_ns"]




def test_p19_d09_diagnostic_selects_its_first_real_frame():
    """Regression for D9 no-arm false failure: prior code hardcoded depth=1."""
    selected = set()
    assert not select_cost_frame(
        selected, frame=37, depth=1, max_frames=1, diagnostic_depth=9
    )
    assert selected == set()
    assert select_cost_frame(
        selected, frame=37, depth=9, max_frames=1, diagnostic_depth=9
    )
    assert selected == {37}
    assert not select_cost_frame(
        selected, frame=38, depth=9, max_frames=1, diagnostic_depth=9
    )
    assert not select_cost_frame(
        selected, frame=37, depth=10, max_frames=1, diagnostic_depth=9
    )


def test_p19_formal_cost_frame_selection_remains_depth1_anchored():
    """No changes to full all-depth P19A cohort or 1/2 frame count."""
    selected = set()
    assert not select_cost_frame(
        selected, frame=37, depth=9, max_frames=1
    )
    assert selected == set()
    assert select_cost_frame(
        selected, frame=37, depth=1, max_frames=1
    )
    assert select_cost_frame(
        selected, frame=37, depth=9, max_frames=1
    )
    assert select_cost_frame(
        selected, frame=37, depth=14, max_frames=1
    )
    assert not select_cost_frame(
        selected, frame=38, depth=1, max_frames=1
    )
    assert selected == {37}


def test_p19_diagnostic_depth_selection_supports_ar_tail_and_rejects_invalid():
    selected = set()
    assert select_cost_frame(
        selected, frame=37, depth=14, max_frames=1, diagnostic_depth=14
    )
    assert selected == {37}
    with pytest.raises(ValueError, match="INVALID_DIAGNOSTIC_TARGET_DEPTH"):
        select_cost_frame(
            set(), frame=37, depth=9, max_frames=1, diagnostic_depth=15
        )
    with pytest.raises(ValueError, match="INVALID_COST_FRAME_SELECTION"):
        select_cost_frame(set(), frame=37, depth=0, max_frames=1)


def test_p19a_formal_analyzer_refuses_diagnostic_root(tmp_path):
    root = _fake_source_locked_run(tmp_path)
    cfg_path = root / "matched_cost" / "config.json"
    config = json.loads(cfg_path.read_text())
    config["diagnostic"] = {
        "arm": "AR_GRAPH", "depth": 1, "fence": 0,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    cfg_path.write_text(json.dumps(config))
    with pytest.raises(
        RuntimeError, match="DIAGNOSTIC_MODE_NO_FORMAL_COST",
    ):
        validate_costs(root)


def _synthetic_cumulative_trace(last_depth: int = 14) -> list[dict]:
    """Strictly one frame, one worker, every expected depth/Graph phase."""
    events = []
    seq = 1
    def add(stage, depth, arm):
        nonlocal seq
        events.append({
            "stage": stage, "depth": depth, "arm": arm,
            "frame": 37, "case_id": "p00_s20261123",
            "pid": 12345, "monotonic_ns": seq,
        })
        seq += 1

    for depth in range(1, last_depth + 1):
        add(DEPTH_STAGES[0], depth, "ALL_DEPTHS_GRAPH")
        for arm in arms_at_depth(depth):
            for phase in GRAPH_STAGES:
                add(phase, depth, arm)
        add(DEPTH_STAGES[1], depth, "ALL_DEPTHS_GRAPH")
        add(DEPTH_STAGES[2], depth, "ALL_DEPTHS_GRAPH")
    return events


def test_p19_all_depths_graph_has_50_legal_arms_and_ar_only_tail():
    assert sum(len(arms_at_depth(d)) for d in range(1, 15)) == 50
    assert arms_at_depth(11) == ("AR", "2", "3", "4")
    assert arms_at_depth(12) == ("AR", "2", "3")
    assert arms_at_depth(13) == ("AR", "2")
    assert arms_at_depth(14) == ("AR",)
    assert len(DIAG_MODES) == 12
    assert "ALL_DEPTHS_GRAPH" in DIAG_MODES


def test_p19_all_depths_full_trace_single_frame_complete():
    config = {
        "arm": "ALL_DEPTHS_GRAPH", "depth": 14,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    events = _synthetic_cumulative_trace()
    result = validate_diagnostic_trace(events, config)
    assert result["depths_verified"] == list(range(1, 15))
    assert result["graph_arms_verified"] == 50
    assert result["frame"] == 37
    assert result["worker_pid"] == 12345
    assert result["result"] == "P19_DIAGNOSTIC_TRACE_COVERAGE_PASS_NO_COST_SELECTION"


def test_p19_all_depths_trace_refuses_missing_arm_missing_depth_and_mixed_frame():
    config = {
        "arm": "ALL_DEPTHS_GRAPH", "depth": 14,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    events = _synthetic_cumulative_trace()
    absent_arm = [
        ev for ev in events if not (
            ev["depth"] == 9 and ev["arm"] == "4"
            and ev["stage"] == "P19_ARM_GRAPH_RELEASE_AND_SYNC_PASS"
        )
    ]
    with pytest.raises(RuntimeError, match="PHASE_COVERAGE_INCOMPLETE"):
        validate_diagnostic_trace(absent_arm, config)
    absent_depth = [ev for ev in events if ev["depth"] != 12]
    with pytest.raises(RuntimeError, match="SELECTED_DEPTH_COUNT_MISMATCH"):
        validate_diagnostic_trace(absent_depth, config)
    mixed = [dict(ev) for ev in events]
    for ev in mixed:
        if ev["depth"] == 9 and ev["stage"] == "P19_COST_TARGET_FRAME_SELECTED":
            ev["frame"] = 38
            break
    with pytest.raises(RuntimeError, match="MIXED_NATIVE_FRAME_OR_WORKER"):
        validate_diagnostic_trace(mixed, config)
    duplicate = events + [dict(events[0])]
    with pytest.raises(RuntimeError, match="SELECTED_DEPTH_COUNT_MISMATCH"):
        validate_diagnostic_trace(duplicate, config)


def test_p19_all_depths_selects_only_one_initial_frame_across_14_depths():
    selected = set()
    for depth in range(1, 15):
        assert select_cost_frame(
            selected, frame=37, depth=depth, max_frames=1,
            diagnostic_depth=None,
        )
        assert not select_cost_frame(
            selected, frame=38, depth=depth, max_frames=1,
            diagnostic_depth=None,
        )
    assert selected == {37}


def test_p19_all_depths_rejects_nontemporal_capture_and_replay():
    cfg = {
        "arm": "ALL_DEPTHS_GRAPH", "depth": 14,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    events = _synthetic_cumulative_trace()
    for ev in events:
        if (ev["depth"] == 9 and ev["arm"] == "3"
                and ev["stage"] == "P19_ARM_REPLAY_EAGER_PARITY_PASS"):
            ev["monotonic_ns"] = 1
            break
    with pytest.raises(RuntimeError, match="ARM_PHASE_ORDER_INVALID"):
        validate_diagnostic_trace(events, cfg)


def test_p19_all_depths_prefix_trace_can_bisect_without_faking_full_14():
    config = {
        "arm": "ALL_DEPTHS_GRAPH", "depth": 8,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    result = validate_diagnostic_trace(_synthetic_cumulative_trace(8), config)
    assert result["depths_verified"] == list(range(1, 9))
    assert result["graph_arms_verified"] == 32
    # Running a partial prefix must not accidentally be reported as D1..14.
    with pytest.raises(RuntimeError, match="SELECTED_DEPTH_COUNT_MISMATCH"):
        validate_diagnostic_trace(_synthetic_cumulative_trace(8), {
            **config, "depth": 14,
        })


def test_p19_all_depths_eager_control_has_50_native_arms_no_graph():
    config = {
        "arm": "ALL_DEPTHS_EAGER", "depth": 14,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    events = []
    counter = 0
    for depth in range(1, 15):
        for stage in (DEPTH_STAGES[0],):
            counter += 1
            events.append({
                "stage": stage, "depth": depth,
                "arm": "ALL_DEPTHS_EAGER", "pid": 321,
                "case_id": "p00_s20261123", "frame": 37,
                "monotonic_ns": counter,
            })
        for arm in arms_at_depth(depth):
            for phase in EAGER_STAGES:
                counter += 1
                events.append({
                    "stage": phase, "depth": depth, "arm": arm,
                    "pid": 321, "case_id": "p00_s20261123",
                    "frame": 37, "monotonic_ns": counter,
                })
        for stage in DEPTH_STAGES[1:]:
            counter += 1
            events.append({
                "stage": stage, "depth": depth,
                "arm": "ALL_DEPTHS_EAGER", "pid": 321,
                "case_id": "p00_s20261123", "frame": 37,
                "monotonic_ns": counter,
            })
    result = validate_diagnostic_trace(events, config)
    assert result["graph_arms_verified"] == 0
    assert result["eager_arms_verified"] == 50
    assert result["depths_verified"] == list(range(1, 15))
    omitted = [
        event for event in events if not (
            event["stage"] == "P19_EAGER_ONLY_DEVICE_SYNC_PASS"
            and event["depth"] == 8 and event["arm"] == "3"
        )
    ]
    with pytest.raises(RuntimeError, match="PHASE_COVERAGE_INCOMPLETE"):
        validate_diagnostic_trace(omitted, config)


def test_p19_native_next_frame_trace_does_not_forge_50_graph_arms():
    cfg = {
        "arm": "ALL_DEPTHS_GRAPH", "depth": 14,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    events = _synthetic_cumulative_trace()
    copied = dict(events[-1])
    copied.update({
        "stage": "P19_NATIVE_NEXT_PAIR_PREDICTOR_FORWARD_RETURNED",
        "depth": 0, "arm": "NATIVE_NEXT_FRAME",
        "monotonic_ns": events[-1]["monotonic_ns"] + 1,
    })
    events.append(copied)
    validated = validate_diagnostic_trace(events, cfg)
    assert validated["graph_arms_verified"] == 50
    # The next-frame marker is evidence of execution, not a Graph arm.
    assert sum(e["stage"] == "P19_ARM_SELECTED" for e in events) == 50


def test_p19_native_next_frame_hooks_wired_under_diagnostic_only():
    import inspect
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
    from sglang_omni.models.qwen3_tts import code_predictor_spec_p19_cost

    source = inspect.getsource(
        Qwen3TTSTalker.code_predictor_forward_incremental
    )
    assert 'p19_pending = getattr(' in source
    assert 'p19_mark("PAIR_PREDICTOR_FORWARD_RETURNED", fence=True)' in source
    assert 'p19_mark("CODE1_SAMPLER_RETURNED", fence=True)' in source
    assert 'p19_mark("BEFORE_P18_DEPTH1_ITEM", fence=True)' in source
    assert '"SGLANG_OMNI_QTTS_CP_P19_NATIVE_FENCE"' in source
    cost = inspect.getsource(code_predictor_spec_p19_cost.maybe_measure)
    assert 'model._p19_native_boundary_next_req = request_id' in cost
    assert 'model._p19_native_boundary_case_id = case["case_id"]' in cost


def test_p19_tail_runner_no_gpu_durable_phase_and_request_gate(
    tmp_path, monkeypatch,
):
    """CPU-only test: phase events are fsynced and request-scoped."""
    class MinimalModel:
        _p19_native_boundary_next_req = "engine-request-1"
        _p19_native_boundary_from_frame = 37
        _p19_native_boundary_case_id = "p00_s20261123"

    model = MinimalModel()
    monkeypatch.setenv(
        "SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", "ALL_DEPTHS_GRAPH"
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH", "14"
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_QTTS_CP_P19_COST_DIR", str(tmp_path)
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_QTTS_CP_P19_TAIL_FENCE", "0"
    )
    assert maybe_tail_runner_event(
        model, "CODE15_SAMPLER_RETURNED",
        device="cpu", request_id="engine-request-1",
    )
    files = list((tmp_path / "diagnostic").glob("worker_pid*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text().splitlines()[0])
    assert record["stage"] == "P19_NATIVE_TAIL_CODE15_SAMPLER_RETURNED"
    assert record["depth"] == 15
    assert record["frame"] == 37
    assert record["arm"] == "NATIVE_TAIL"
    assert record["case_id"] == "p00_s20261123"
    with pytest.raises(RuntimeError, match="TAIL_RUNNER_REQUEST_ID_DRIFT"):
        maybe_tail_runner_event(
            model, "BAD_REQUEST", device="cpu", request_id="another"
        )
    assert len(files[0].read_text().splitlines()) == 1
    monkeypatch.setenv(
        "SGLANG_OMNI_QTTS_CP_P19_TAIL_FENCE", "not-a-valid-option"
    )
    with pytest.raises(RuntimeError, match="TAIL_FENCE_MUST_BE_0_OR_1"):
        maybe_tail_runner_event(model, "BAD_ENV", device="cpu")
    assert len(files[0].read_text().splitlines()) == 1


def test_p19_tail_runner_disabled_without_graph_diagnostic(
    tmp_path, monkeypatch,
):
    class MinimalModel:
        _p19_native_boundary_next_req = "engine-request-1"
        _p19_native_boundary_from_frame = 37
        _p19_native_boundary_case_id = "p00_s20261123"

    monkeypatch.delenv(
        "SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM", raising=False
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_QTTS_CP_P19_COST_DIR", str(tmp_path)
    )
    assert not maybe_tail_runner_event(
        MinimalModel(), "NO_RESEARCH_ENV", device="cpu"
    )
    assert not list(tmp_path.rglob("*.jsonl"))


def test_p19_tail_native_code15_and_feedback_hooks_present():
    """Confirm instrumentation stays behind explicit research-only gates."""
    import inspect
    from sglang_omni.models.qwen3_tts.model_runner import (
        Qwen3TTSModelRunner,
    )
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    native = inspect.getsource(
        Qwen3TTSTalker.code_predictor_forward_incremental
    )
    assert '"SGLANG_OMNI_QTTS_CP_P19_TAIL_FENCE"' in native
    for stage in (
        "D14_P19_PROBE_RETURNED",
        "FINAL_PREDICTOR_RETURNED",
        "CODE15_LMHEAD_RETURNED",
        "CODE15_SAMPLER_RETURNED",
        "NATIVE_PREDICTOR_OUTPUT_READY",
    ):
        assert f'"{stage}"' in native
    runner = inspect.getsource(
        Qwen3TTSModelRunner._p19_native_tail_mark
    )
    assert '"ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER"' in runner
    for method, stage in (
        ("collect_codes", "COLLECT_CODES_PREDICTOR_RETURNED"),
        ("post_process_outputs", "POST_PROCESS_CODES_EVENT_RECORDED"),
        ("write_feedback_buffers", "WRITE_FEEDBACK_ROW_IDS_READY"),
        ("custom_decode_forward", "MAIN_TALKER_FORWARD_BEGIN"),
        ("post_decode", "MAIN_TALKER_STEP_RETURNED_BEFORE_COLLECT"),
        ("sample_next_token_ids", "MAIN_TALKER_FORWARD_RETURNED_BEFORE_SAMPLE"),
        ("sample_next_token_ids", "MAIN_SEMANTIC_SAMPLER_RETURNED"),
        ("collect_codes", "NEXT_FRAME_BEFORE_CODE_PREDICTOR"),
    ):
        assert stage in inspect.getsource(
            getattr(Qwen3TTSModelRunner, method)
        )


def test_p19_main_talker_instrumentation_preserves_default_forward_dispatch():
    """Qwen runner must still use the ORIGINAL tp_worker forward path.

    Only source-locked diagnostic hooks are allowed, not a new model backend.
    """
    import inspect
    from types import SimpleNamespace
    from sglang_omni.models.qwen3_tts.model_runner import (
        Qwen3TTSModelRunner,
    )
    observed = []

    class FakeSelf:
        def _p19_native_tail_mark(self, stage, *, device, requests):
            observed.append((stage, device, requests))

    forward_batch = SimpleNamespace(
        input_ids=SimpleNamespace(device="cpu")
    )
    requests = []
    ret = Qwen3TTSModelRunner.custom_decode_forward(
        FakeSelf(), forward_batch, None, requests
    )
    assert ret is None
    assert observed == [("MAIN_TALKER_FORWARD_BEGIN", "cpu", requests)]

    source = inspect.getsource(
        Qwen3TTSModelRunner.sample_next_token_ids
    )
    assert "super().sample_next_token_ids(" in source
    assert '"MAIN_TALKER_FORWARD_RETURNED_BEFORE_SAMPLE"' in source
    assert '"MAIN_SEMANTIC_SAMPLER_RETURNED"' in source
    assert "return sampled" in source


def test_p19_collector_boundaries_guard_next_frame_before_predictor():
    import inspect
    from sglang_omni.models.qwen3_tts.model_runner import (
        Qwen3TTSModelRunner,
    )
    source = inspect.getsource(Qwen3TTSModelRunner.collect_codes)
    assert source.index('"NEXT_FRAME_TOKEN_IDS_STAGED"') < source.index(
        '"NEXT_FRAME_SAMPLE_POSITIONS_READY"'
    )
    assert source.index('"NEXT_FRAME_SAMPLE_POSITIONS_READY"') < (
        source.index('"NEXT_FRAME_BEFORE_CODE_PREDICTOR"')
    )
    assert source.index('"NEXT_FRAME_BEFORE_CODE_PREDICTOR"') < (
        source.index("self.model.code_predictor_forward(")
    )


def test_p19_lifecycle_profile_counts_and_fail_closed_validation(monkeypatch):
    assert DEFAULT_REPLAYS == 61
    assert DEFAULT_OPS == 65
    assert MIN_REPLAY_OPS == 5
    assert profile_counts("ALL_DEPTHS_GRAPH", "DEFAULT") == {
        "eager_operations": 3, "captures": 1,
        "graph_replays": 61, "operation_equivalents": 65,
    }
    assert profile_counts("ALL_DEPTHS_GRAPH", "MIN_REPLAY") == {
        "eager_operations": 3, "captures": 1,
        "graph_replays": 1, "operation_equivalents": 5,
    }
    assert profile_counts("ALL_DEPTHS_EAGER", "MATCHED_EAGER") == {
        "eager_operations": 65, "captures": 0,
        "graph_replays": 0, "operation_equivalents": 65,
    }
    assert len(PROFILES) == 5
    for profile in ("GC_GUARD", "RETAIN_GRAPH"):
        assert validate_profile("ALL_DEPTHS_GRAPH", 14, profile) == profile
    with pytest.raises(RuntimeError, match="PROFILE_MODE_MISMATCH"):
        validate_profile("ALL_DEPTHS_GRAPH", 14, "MATCHED_EAGER")
    with pytest.raises(RuntimeError, match="PROFILE_MODE_MISMATCH"):
        validate_profile("ALL_DEPTHS_EAGER", 14, "GC_GUARD")
    with pytest.raises(RuntimeError, match="TERMINAL_DEPTH14"):
        validate_profile("ALL_DEPTHS_GRAPH", 9, "MIN_REPLAY")
    with pytest.raises(RuntimeError, match="UNKNOWN_PROFILE"):
        validate_profile("ALL_DEPTHS_GRAPH", 14, "SILENT_UNKNOWN")
    monkeypatch.setenv(PROFILE_ENV, "GC_GUARD")
    with pytest.raises(RuntimeError, match="REQUIRES_DIAGNOSTIC_MODE"):
        current_profile(None)
    assert current_profile(("ALL_DEPTHS_GRAPH", 14)) == "GC_GUARD"
    monkeypatch.delenv(PROFILE_ENV)
    assert current_profile(None) == "DEFAULT"


def test_p19_matched_eager_executes_exact_65_real_calls(monkeypatch):
    """CPU mock: no Graph or CUDA is required for exact invocation count."""
    import torch
    calls, restores, events = [], [], []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    def op():
        calls.append(len(calls))
        return {"not_gpu": len(calls)}

    def restore():
        restores.append(len(calls))

    _matched_eager_arm(
        operation=op, restore=restore, device="cpu", mark=events.append
    )
    assert len(calls) == DEFAULT_OPS == 65
    assert len(restores) == 1 + 1 + 2 + 1 + 1 + 4
    assert events[0] == "P19_ARM_MATCHED_EAGER_REFERENCE_SYNC_PASS"
    assert events[-1] == "P19_ARM_MATCHED_EAGER_TOTAL_65_SYNC_PASS"
    assert [e for e in events if "BLOCK_" in e] == [
        f"P19_ARM_MATCHED_EAGER_BLOCK_{i}_SYNC_PASS" for i in range(4)
    ]


def _p19_profile_trace(mode, profile):
    """Fully ordered synthetic trace, exactly one real frame/worker."""
    n, events = 0, []

    def add(stage, depth, arm):
        nonlocal n
        n += 1
        events.append({
            "stage": stage, "depth": depth, "arm": arm,
            "frame": 37, "case_id": "p00_s20261123",
            "pid": 123, "monotonic_ns": n,
        })

    for depth in range(1, 15):
        add(DEPTH_STAGES[0], depth, mode)
        for arm in arms_at_depth(depth):
            for stage in stages_for_profile(mode, profile):
                add(stage, depth, arm)
        if profile == "RETAIN_GRAPH" and depth == 14:
            add("P19_RETAINED_50_GRAPHS_LIVE_UNTIL_WORKER_EXIT",
                depth, mode)
        add(DEPTH_STAGES[1], depth, mode)
        add(DEPTH_STAGES[2], depth, mode)
    return events


@pytest.mark.parametrize(
    "mode,profile,eager,capture,replays",
    [
        ("ALL_DEPTHS_GRAPH", "DEFAULT", 150, 50, 3050),
        ("ALL_DEPTHS_GRAPH", "MIN_REPLAY", 150, 50, 50),
        ("ALL_DEPTHS_GRAPH", "GC_GUARD", 150, 50, 3050),
        ("ALL_DEPTHS_GRAPH", "RETAIN_GRAPH", 150, 50, 3050),
        ("ALL_DEPTHS_EAGER", "MATCHED_EAGER", 3250, 0, 0),
    ],
)
def test_p19_lifecycle_coverage_exact_profile_markers(
    mode, profile, eager, capture, replays,
):
    cfg = {
        "arm": mode, "depth": 14, "profile": profile,
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    events = _p19_profile_trace(mode, profile)
    got = validate_diagnostic_trace(events, cfg)
    assert got["profile"] == profile
    assert got["expected_eager_operations"] == eager
    assert got["expected_captures"] == capture
    assert got["expected_graph_replays"] == replays
    assert got["expected_total_arm_operations"] == eager + capture + replays
    assert got["graph_arms_verified"] + got["eager_arms_verified"] == 50
    # Fail closed when the profile-specific evidence is absent.
    unique = {
        "DEFAULT": "P19_ARM_TIMING_BLOCK_3_SYNC_PASS",
        "MIN_REPLAY": "P19_ARM_MIN_REPLAY_NO_TIMING_PASS",
        "GC_GUARD": "P19_ARM_GC_CAPTURE_GUARD_EXIT",
        "RETAIN_GRAPH": "P19_ARM_GRAPH_RETAINED_AND_SYNC_PASS",
        "MATCHED_EAGER": "P19_ARM_MATCHED_EAGER_TOTAL_65_SYNC_PASS",
    }[profile]
    missing = [
        e for e in events
        if not (e["depth"] == 9 and e["arm"] == "AR"
                and e["stage"] == unique)
    ]
    with pytest.raises(RuntimeError, match="PHASE_COVERAGE_INCOMPLETE"):
        validate_diagnostic_trace(missing, cfg)


def test_p19_retained_graph_count_requires_proven_50_refs():
    cfg = {
        "arm": "ALL_DEPTHS_GRAPH", "depth": 14,
        "profile": "RETAIN_GRAPH",
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
    traces = [
        e for e in _p19_profile_trace("ALL_DEPTHS_GRAPH", "RETAIN_GRAPH")
        if e["stage"] != "P19_RETAINED_50_GRAPHS_LIVE_UNTIL_WORKER_EXIT"
    ]
    with pytest.raises(RuntimeError, match="PHASE_COVERAGE_INCOMPLETE"):
        validate_diagnostic_trace(traces, cfg)


def test_p19_lifecycle_shell_and_client_stay_no_cost_and_source_locked():
    import inspect
    from experiments.qwen3_tts_cp_spec_k2 import p19_diag_once
    from sglang_omni.models.qwen3_tts import code_predictor_spec_p19_cost

    src = inspect.getsource(code_predictor_spec_p19_cost.maybe_measure)
    assert 'profile = current_profile(diagnostic)' in src
    assert 'if diagnostic is not None:' in src
    assert 'return\n        report = {' in src
    once = inspect.getsource(p19_diag_once.run)
    assert 'P19_PROFILE_UNMATCHED_LAUNCH_CONSTRAINTS' in once
    assert 'P19_full_cost_rows_should_be_zero' in once
    assert 'PROFILE_PASS[profile]' in once

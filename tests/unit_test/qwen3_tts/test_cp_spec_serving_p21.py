# SPDX-License-Identifier: Apache-2.0
"""CPU-only P21 real-serving K2 integration and ABBA evidence contracts.

H200 CUDA correctness/quality/performance remains a separate mandatory gate.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from sglang_omni.models.qwen3_tts.code_predictor_spec_serving_p21 import (
    _request_seed, armed,
)
from experiments.qwen3_tts_cp_spec_k2.p21_serving_abba import (
    STATUS_PASS, compute_report, PHASES,
)


def test_p21_armed_fail_closed_and_baseline_off(monkeypatch, tmp_path):
    env = "SGLANG_OMNI_QTTS_CP_P21_SERVING_K2"
    arm_env = "SGLANG_OMNI_QTTS_CP_P21_ARM_FILE"
    sha_env = "SGLANG_OMNI_QTTS_CP_P21_SOURCE_SHA"
    monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv(arm_env, raising=False)
    monkeypatch.delenv(sha_env, raising=False)
    assert not armed()
    monkeypatch.setenv(env, "invalid")
    with pytest.raises(RuntimeError, match="FLAG_MUST_BE_ZERO_OR_ONE"):
        armed()
    monkeypatch.setenv(env, "1")
    with pytest.raises(RuntimeError, match="ABSOLUTE_ARM_PATH"):
        armed()
    arm = tmp_path / "ARM"
    monkeypatch.setenv(arm_env, str(arm))
    # A on same P21-capable worker: no post-health ARM means native path.
    assert not armed()
    arm.touch()
    with pytest.raises(RuntimeError, match="REQUIRES_SOURCE_SHA"):
        armed()
    monkeypatch.setenv(sha_env, "a" * 40)
    assert armed()
    arm.unlink()
    assert not armed()
    monkeypatch.setenv(env, "0")
    arm.touch()
    assert not armed()


def test_p21_seed_is_deterministic_private_and_position_sensitive():
    assert _request_seed(20261123, 37) == _request_seed(20261123, 37)
    assert _request_seed(20261123, 37) != _request_seed(20261123, 38)
    assert _request_seed(20261123, 37) != _request_seed(20261124, 37)
    assert 0 <= _request_seed(1, 1) < 2**63


def _synthetic_abba():
    cases = [
        {"id": "english", "seed": 20261123, "text": "Hello"},
        {"id": "mandarin", "seed": 20261124, "text": "你好"},
    ]
    rows = []
    for visit, phase in enumerate(PHASES):
        for i, case in enumerate(cases):
            rows.append({
                "visit": visit,
                "phase": phase,
                "case_id": case["id"], "seed": case["seed"],
                "e2e_wall_s": (1.0 if phase == "A" else 1.1) + i * 0.1,
                "audio_duration_s": 2.0,
                "http_rtf": 0.5,
                "wav_sha256": hashlib.sha256(
                    f"{case['id']}|{phase}".encode()
                ).hexdigest(),
            })
    return cases, rows


def test_p21_abba_source_matched_c1_screen_no_quality_claim():
    cases, rows = _synthetic_abba()
    report = compute_report(rows, cases)
    assert report["status"] == STATUS_PASS
    assert report["overall"]["A"]["count"] == 4
    assert report["overall"]["B"]["count"] == 4
    assert report["candidate_speed_screen"] == (
        "B_NO_MEANINGFUL_C1_LATENCY_WIN"
    )
    assert report["quality_certificate"].startswith("NONE")
    assert report["within_arm_wav_hash_repeatable"] is True
    assert report["median_case_b_over_a"] > 1.0
    assert report["median_case_relative_latency_change_pct"] > 0


def test_p21_abba_fails_closed_on_incomplete_or_unstable_audio():
    cases, rows = _synthetic_abba()
    with pytest.raises(RuntimeError, match="MISSING_ROWS"):
        compute_report(rows[:-1], cases)
    mutant = [dict(row) for row in rows]
    mutant[-1]["wav_sha256"] = "x" * 64
    with pytest.raises(RuntimeError, match="SEEDED_WAV_DRIFT"):
        compute_report(mutant, cases)


def test_p21_real_serving_hooks_before_all_p18_p19_probes():
    """P21 uses returned output buffers, not a counterfactual cost path."""
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
    from sglang_omni.models.qwen3_tts import (
        code_predictor_spec_serving_p21 as serving,
    )

    dispatch = inspect.getsource(Qwen3TTSTalker.code_predictor_forward)
    assert "SGLANG_OMNI_QTTS_CP_P21_SERVING_K2" in dispatch
    assert dispatch.index("if p21_armed():") < dispatch.index(
        "# P18A: opt-in native-AR eager trajectories"
    )
    assert "return p21_serving_k2(" in dispatch
    assert dispatch.index("return p21_serving_k2(") < dispatch.index(
        "self.predictor_forward_graphed("
    )
    src = inspect.getsource(serving.serving_k2)
    assert "P21_REQUIRES_NO_P18_P19_OR_LEGACY_K2_PROBES" in src
    assert "codes[:, depth + 1: depth + 1 + committed].copy_(" in src
    assert "embeddings.copy_(step[\"next_sum\"])" in src
    assert "return codes, embeddings" in src
    assert "P21_UNCOMMITTED_SPECULATIVE_CODE" in src
    runner = inspect.getsource(Qwen3TTSModelRunner.collect_codes)
    assert 'stats["native_graph_replays"] +=' in runner
    assert 'stats["candidate_frames"] += 1' in runner
    terminal = inspect.getsource(Qwen3TTSModelRunner.on_request_finished)
    assert "P21_REAL_SERVING_REQUEST" in terminal
    # A/B must never secretly turn on P18/P19 and then call this harness.
    launcher = (
        Path(__file__).resolve().parents[3]
        / "experiments/qwen3_tts_cp_spec_k2/p21_serving_launch.sh"
    ).read_text()
    assert 'unset "$key"' in launcher
    assert 'SGLANG_OMNI_QTTS_CP_SPEC_K=0' in launcher
    assert 'SGLANG_OMNI_QTTS_CP_P21_SERVING_K2=1' in launcher

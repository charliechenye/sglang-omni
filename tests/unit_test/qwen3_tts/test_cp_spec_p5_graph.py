# SPDX-License-Identifier: Apache-2.0
"""P5 CPU evidence gates and a small CUDA Graph K/V ownership test."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import sglang_omni.models.qwen3_tts.sglang_model as qtts_module
from experiments.qwen3_tts_cp_spec_k2.p5_graph_screen import (
    STATUS, analyze, write_report,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_graph_cost import (
    SCHEMA, config, maybe_profile_packed,
)


def acceptance() -> dict:
    return {
        "source_sha": "1" * 40,
        "frames": 2,
        "total_proposals": 28,
        "total_accepted": 0,
        "total_rejected": 28,
        "total_bonus": 0,
        "depths": [
            {"residual_depth": x, "proposals": 2}
            for x in range(2, 16)
        ],
    }


def baseline_record() -> dict:
    return {
        "schema": SCHEMA,
        "pid": 271,
        "kind": "baseline_full_predictor_graph_on",
        "source_sha": "2" * 40,
        "device": "NVIDIA H200",
        "frames": 32,
        "graph_replays": 32,
        "median_cuda_event_ms": 1.4,
    }


def packed_record(depth: int, *, two_ms: float = 0.07) -> dict:
    return {
        "schema": SCHEMA,
        "pid": 271,
        "kind": "single_vs_packed_verifier_cuda_graph",
        "source_sha": "2" * 40,
        "device": "NVIDIA H200",
        "proposal_depth": depth,
        "cache_len": depth,
        "serial_one_graph_ms": 0.06,
        "packed_two_graph_ms": two_ms,
        "packed_to_one_ratio": two_ms / 0.06,
    }


def test_config_disabled_and_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_P5_DIR", raising=False)
    assert config() is None
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_GRAPH_DEPTHS", "2,6,10,14")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_REPEATS", "64")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_BASE_FRAMES", "32")
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_P4_DIR", raising=False)
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR", raising=False)
    assert config()["depths"] == (2, 6, 10, 14)
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P4_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="without P4"):
        config()
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_P4_DIR", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_GRAPH_DEPTHS", "2,2")
    with pytest.raises(ValueError, match="unique"):
        config()


def test_p5_screen_bounds_and_no_false_speedup(tmp_path: Path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "baseline_graph_pid271.json").write_text(json.dumps(baseline_record()))
    for d in (2, 6, 10, 14):
        (profiles / f"packed_graph_pid271_frame00001_depth{d:02d}.json").write_text(
            json.dumps(packed_record(d))
        )
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps(acceptance()))
    result = analyze(profiles, cohort, min_probes=4)
    assert result["status"] == STATUS
    assert result["baseline_graph_frames"] == 32
    assert result["k2_proposals_per_frame_from_prior_cohort"] == 14
    assert result["k2_verifier_graph_only_extrapolated_ms_per_frame"] == pytest.approx(14 * 0.07)
    assert result["optimistic_baseline_over_k2_verifier_stack_quotient"] == pytest.approx(1.4 / (14 * .07))
    assert not result["source_sha_same_as_prior_cohort"]
    assert write_report(tmp_path / "report", result).is_file()
    with pytest.raises(FileExistsError):
        write_report(tmp_path / "report", result)

    invalid = baseline_record()
    invalid["graph_replays"] = 0
    (profiles / "baseline_graph_pid271.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="replay coverage"):
        analyze(profiles, cohort, min_probes=4)

    # Missing prior-cohort source identity is never accepted as a source-matched run.
    invalid["graph_replays"] = 32
    (profiles / "baseline_graph_pid271.json").write_text(json.dumps(invalid))
    (profiles / "packed_graph_pid271_frame00001_depth14.json").unlink()
    with pytest.raises(RuntimeError, match="captured-graph probes"):
        analyze(profiles, cohort, min_probes=4)


@pytest.mark.accelerator
def test_p5_cuda_captured_single_and_packed_restores_kv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    if not torch.cuda.is_available():
        pytest.skip("H200/CUDA required")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        DTYPE, HIDDEN, build_talker, with_predictor_layers,
    )
    monkeypatch.setattr(
        qtts_module, "apply_qk_norm", lambda q, k, **kwargs: (q, k)
    )
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_P4_DIR", raising=False)
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_GRAPH_DEPTHS", "2")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_REPEATS", "8")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P5_BASE_FRAMES", "8")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "2" * 40)
    model = with_predictor_layers(build_talker(torch.device("cuda")), 2)
    generator = torch.Generator(device="cpu").manual_seed(20261121)
    inputs = torch.randn(
        (1, 4, HIDDEN), dtype=torch.float32, generator=generator
    ).to("cuda", DTYPE)
    with torch.inference_mode():
        model.predictor_forward_tokens(
            token_embeds=inputs[:, :2].clone(), batch_size=1, cache_len=0
        )
        packed = model.predictor_forward_tokens(
            token_embeds=inputs[:, 2:4].clone(), batch_size=1, cache_len=2
        ).clone()
        k_saved = model.predictor_k_cache[:, :1, :4].clone()
        v_saved = model.predictor_v_cache[:, :1, :4].clone()
        maybe_profile_packed(
            model,
            frame=1, depth=1, cache_len=2,
            last_embedding=inputs[:, 2:3].clone(),
            proposed_embedding=inputs[:, 3:4].clone(),
        )
    rows = list((tmp_path / "graphs").glob("packed_graph_pid*_frame00001_depth02.json"))
    assert len(rows) == 1
    report = json.loads(rows[0].read_text())
    assert report["status"].startswith("P5_INDIVIDUAL_VERIFIER")
    assert report["serial_one_graph_ms"] > 0
    assert report["packed_two_graph_ms"] > 0
    assert report["source_sha"] == "2" * 40
    torch.testing.assert_close(
        model.predictor_k_cache[:, :1, :4], k_saved, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache[:, :1, :4], v_saved, rtol=0, atol=0
    )
    assert torch.isfinite(packed).all().item()

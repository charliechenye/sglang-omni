# SPDX-License-Identifier: Apache-2.0
"""CPU and H200 integration guards for opt-in K2 real target diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.qwen3_tts.sglang_model as model_module
from experiments.qwen3_tts_cp_spec_k2.analyze_depth_trace import analyze, write_report
from sglang_omni.models.qwen3_tts.code_predictor_spec_diagnostics import (
    SCHEMA,
    assert_diagnostic_parity,
    check_embedding_sum,
    check_packed_verifier,
    frame_enabled,
    validate_depth_events,
    write_trace,
)


def make_events() -> list[dict]:
    # 14 residual codes after code1: accept at depth2 -> bonus depth3,
    # then reject every depth4..15 (13 proposals, one bonus).
    events = [{
        "proposal_depth": 2, "expected_overlap": .4,
        "accepted": True, "bonus_depth": 3,
        "target_mass_on_q_support": .6,
        "proposal_mass_on_p_support": .6,
    }]
    events += [{
        "proposal_depth": d, "expected_overlap": .1,
        "accepted": False, "bonus_depth": None,
        "target_mass_on_q_support": .3,
        "proposal_mass_on_p_support": .3,
    } for d in range(4, 16)]
    validate_depth_events(events)
    return events


def test_trace_depth_accounting_and_invalid_sequences() -> None:
    events = make_events()
    assert len(events) == 13
    with pytest.raises(ValueError, match="wrong proposal depth"):
        validate_depth_events([{**events[0], "proposal_depth": 3}] + events[1:])
    with pytest.raises(ValueError, match="missing bonus"):
        validate_depth_events([{**events[0], "bonus_depth": None}] + events[1:])
    with pytest.raises(ValueError, match="invalid per-depth"):
        validate_depth_events([{**events[0], "expected_overlap": 1.01}] + events[1:])


def test_real_embeddings_reconstruct_in_bf16_order() -> None:
    torch.manual_seed(20261121)
    tables = torch.randn(15, 32, 8).to(torch.bfloat16)
    model = SimpleNamespace(
        config=SimpleNamespace(num_code_groups=16),
        code_predictor=SimpleNamespace(
            model=SimpleNamespace(
                codec_embedding=SimpleNamespace(weight=tables)
            )
        ),
    )
    first = torch.randn(1, 1, 8).to(torch.bfloat16)
    codes = torch.arange(16, dtype=torch.long).unsqueeze(0)
    summation = first[:, 0].clone()
    for d in range(1, 16):
        summation.add_(tables[d - 1, codes[0, d]][None])
    result = check_embedding_sum(model, first, codes, summation)
    assert result["close"] and result["max_abs"] == 0
    with pytest.raises(ValueError, match="codec group"):
        check_embedding_sum(model, first, codes[:, :-1], summation)


def test_trace_files_no_overwrite_and_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR", raising=False)
    assert frame_enabled(1) == (None, False)
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_K2_DIAG_MAX_FRAMES", "3")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_K2_PARITY_MAX_FRAMES", "1")
    assert frame_enabled(1) == (tmp_path.resolve(), True)
    assert frame_enabled(2) == (tmp_path.resolve(), False)
    assert frame_enabled(4) == (None, False)
    file = write_trace(tmp_path, 1, {"frame": 1})
    assert json.loads(file.read_text())["schema"] == SCHEMA
    with pytest.raises(FileExistsError):
        write_trace(tmp_path, 1, {"frame": 1})
    with pytest.raises(ValueError):
        monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_K2_PARITY_MAX_FRAMES", "4")
        frame_enabled(1)


def test_cpu_analyzer_does_not_double_count_bonus_or_unverified_parity(tmp_path: Path):
    folder = tmp_path / "frames"
    events = make_events()
    row = {
        "frame": 1,
        "source_sha": "1" * 40,
        "total_residual_codes": 15,
        "proposed": 13,
        "accepted": 1,
        "rejected": 12,
        "bonus": 1,
        "rounds": 13,
        "per_depth": events,
        "output_codes": list(range(16)),
        "embedding_sum": {"close": True, "max_abs": 0.0},
        "target_parity": {"pass": True, "executed": False, "checks": []},
    }
    write_trace(folder, 1, row)
    result = analyze(folder, expect_source_sha="1" * 40, min_frames=1)
    assert result["frames"] == 1
    assert result["parity_checked_frames"] == 0
    assert result["total_proposals"] == 13
    assert result["total_bonus"] == 1
    assert result["status"].endswith("PARITY_NOT_CHECKED")
    by_depth = {r["residual_depth"]: r for r in result["depths"]}
    assert by_depth[2]["accepted"] == 1
    assert by_depth[3]["bonus_skipped_proposals"] == 1
    assert by_depth[3]["proposals"] == 0
    assert by_depth[4]["proposals"] == 1
    write_report(tmp_path / "summary", result)
    assert "Bonus skips" in (tmp_path / "summary" / "summary.md").read_text()
    with pytest.raises(RuntimeError, match="source SHA mismatch"):
        analyze(folder, expect_source_sha="2" * 40, min_frames=1)
    with pytest.raises(RuntimeError, match="require"):
        analyze(folder, expect_source_sha=None, min_frames=2)
    write_trace(folder, 1, {"status": "P1_FAIL"}, failed=True)
    with pytest.raises(RuntimeError, match="parity failures"):
        analyze(folder, expect_source_sha=None, min_frames=1)


@pytest.mark.accelerator
def test_live_gpu_parity_checker_restores_kv_and_compares_same_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("real CUDA required")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        DTYPE, HIDDEN, build_talker, with_predictor_layers,
    )

    monkeypatch.setattr(
        model_module, "apply_qk_norm",
        lambda q, k, **_: (q, k),
    )
    torch.manual_seed(20261121)
    model = with_predictor_layers(build_talker(torch.device("cuda")), 2)
    device = torch.device("cuda")
    all_embeds = torch.randn(1, 4, HIDDEN, device=device).to(DTYPE)
    with torch.inference_mode():
        model.predictor_forward_tokens(
            token_embeds=all_embeds[:, :2].clone(), batch_size=1, cache_len=0
        )
        packed = model.predictor_forward_tokens(
            token_embeds=all_embeds[:, 2:4].clone(), batch_size=1, cache_len=2
        ).clone()
        k_before = model.predictor_k_cache[:, :1, :4].clone()
        v_before = model.predictor_v_cache[:, :1, :4].clone()
        row = check_packed_verifier(
            model,
            cache_len=2, depth=1,
            last_embedding=all_embeds[:, 2:3].clone(),
            proposed_embedding=all_embeds[:, 3:4].clone(),
            packed_hidden=packed,
        )
        assert row["pass"], row
        assert row["checks"]["target_logits"]["close"]
        torch.testing.assert_close(k_before, model.predictor_k_cache[:, :1, :4], atol=0, rtol=0)
        torch.testing.assert_close(v_before, model.predictor_v_cache[:, :1, :4], atol=0, rtol=0)
        # Deliberately corrupt the purported packed hidden; checker must
        # diagnose rather than silently repair or accept.
        bad = check_packed_verifier(
            model, cache_len=2, depth=1,
            last_embedding=all_embeds[:, 2:3].clone(),
            proposed_embedding=all_embeds[:, 3:4].clone(),
            packed_hidden=packed + 2.0,
        )
        assert not bad["pass"]
        with pytest.raises(RuntimeError, match="PARITY_FAIL"):
            assert_diagnostic_parity(bad)

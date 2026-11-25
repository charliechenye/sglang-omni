# SPDX-License-Identifier: Apache-2.0
"""P4 CPU cost accounting and H200 fixed-KV cost-probe safety gates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import sglang_omni.models.qwen3_tts.sglang_model as model_module
from experiments.qwen3_tts_cp_spec_k2.p4_cost_screen import (
    STATUS, analyze, validate_acceptance, write_report,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_perf import (
    SCHEMA, parse_depths, profile_state, read_config,
)


def acceptance_fixture() -> dict:
    # One complete frame, rejecting all 14 K2 draft tokens: 14 proposals,
    # zero bonus. This tests math, NOT an actual acceptance-rate scenario.
    return {
        "source_sha": "a" * 40, "frames": 1,
        "total_proposals": 14, "total_accepted": 0,
        "total_rejected": 14, "total_bonus": 0,
        "depths": [
            {"residual_depth": d, "proposals": 1}
            for d in range(2, 16)
        ],
    }


def cost_row(depth: int, *, packed_ms: float = 0.6) -> dict:
    return {
        "schema": SCHEMA, "pid": 303, "source_sha": "b" * 40,
        "frame": 1, "proposal_depth": depth, "cache_len": depth,
        "gpu": "NVIDIA H200", "dtype": "torch.bfloat16",
        "measurements": {
            name: {
                "cuda_event_ms": value, "host_wall_ms": value * 1.5,
                "blocks": 2,
            }
            for name, value in (
                ("serial_one_stack", 1.0),
                ("serial_two_stack", 2.0),
                ("packed_two_stack", packed_ms),
                ("draft_head", 0.15),
                ("draft_select_debug", 0.4),
                ("accept_debug", 0.5),
            )
        },
    }


def test_p4_is_off_by_default_and_rejects_conflicting_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_P4_DIR", raising=False)
    assert read_config() is None
    assert parse_depths("2,6,10,14") == (2, 6, 10, 14)
    with pytest.raises(ValueError):
        parse_depths("2,2,14")
    with pytest.raises(ValueError):
        parse_depths("1,16")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P4_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P4_MAX_STATES", "4")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P4_REPEATS", "12")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P4_WARMUP", "3")
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR", raising=False)
    assert read_config()["max_states"] == 4
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="separate runs"):
        read_config()


def test_weighted_cost_screen_does_not_claim_serving_speedup(tmp_path: Path) -> None:
    src = tmp_path / "prior_report.json"
    src.write_text(json.dumps(acceptance_fixture()))
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    for d in (2, 6, 10, 14):
        (profiles / f"perf_pid303_frame00001_depth{d:02d}.json").write_text(
            json.dumps(cost_row(d))
        )
    result = analyze(profiles, src, min_probes=4)
    assert result["status"] == STATUS
    assert result["on_policy_accounting"]["codes_per_round"] == 1
    assert result["measured_state_count"] == 4
    assert not result["same_source"]
    assert result["weighted_fixed_state_screen"]["baseline_serial_stack_ms"] == pytest.approx(14.0)
    assert result["weighted_fixed_state_screen"]["candidate_packed_stack_ms"] == pytest.approx(14.0 * .6)
    assert result["weighted_fixed_state_screen"]["serial_stack_over_packed_stack_quotient"] == pytest.approx(1.0 / .6)
    assert result["weighted_fixed_state_screen"]["serial_stack_over_packed_plus_head_quotient"] == pytest.approx(1.0 / .75)
    write_report(tmp_path / "report", result)
    assert "NOT GPU busy" in (tmp_path / "report" / "summary.md").read_text()
    with pytest.raises(FileExistsError):
        write_report(tmp_path / "report", result)

    # A wide2 call slower than a one-token target call cannot be hidden by
    # a purported acceptance-rate multiplier. No ranking is inferred here.
    for file in profiles.glob("*.json"):
        row = json.loads(file.read_text())
        row["measurements"]["packed_two_stack"]["cuda_event_ms"] = 2.0
        file.write_text(json.dumps(row))
    negative = analyze(profiles, src, min_probes=4)
    assert negative["weighted_fixed_state_screen"]["serial_stack_over_packed_stack_quotient"] == pytest.approx(0.5)


def test_acceptance_accounting_checks_the_full_residual_census() -> None:
    good = validate_acceptance(acceptance_fixture())
    assert good["codes_per_round"] == 1
    corrupt = acceptance_fixture()
    corrupt["total_bonus"] = 1
    with pytest.raises(ValueError, match="counters"):
        validate_acceptance(corrupt)


@pytest.mark.accelerator
def test_gpu_probe_does_not_mutate_the_living_checkpoint_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        DTYPE, HIDDEN, PRED_VOCAB, build_talker, with_predictor_layers,
    )
    monkeypatch.setattr(
        model_module, "apply_qk_norm",
        lambda q, k, **_: (q, k),
    )
    device = torch.device("cuda")
    model = with_predictor_layers(build_talker(device), 2)
    torch.manual_seed(20261121)
    inputs = torch.randn(1, 4, HIDDEN, dtype=torch.float32, device=device).to(DTYPE)
    with torch.inference_mode():
        model.predictor_forward_tokens(
            token_embeds=inputs[:, :2].clone(), batch_size=1, cache_len=0
        )
        hidden = model.predictor_forward_tokens(
            token_embeds=inputs[:, 2:4].clone(), batch_size=1, cache_len=2
        )
        k_saved = model.predictor_k_cache[:, :1, :4].clone()
        v_saved = model.predictor_v_cache[:, :1, :4].clone()
        p = torch.ones((1, PRED_VOCAB), device=device) / PRED_VOCAB
        q = p.clone()
        result = profile_state(
            model,
            proposal_depth=2, cache_len=2,
            last_embedding=inputs[:, 2:3].clone(),
            proposed_embedding=inputs[:, 3:4].clone(),
            draft_hidden=hidden[:, 0:1].clone(),
            target_head=model.code_predictor.lm_head[1],
            p=p, q=q, proposal=torch.tensor([3], device=device),
            temperature=.9, top_k=5, top_p=1.,
            repeats=3, warmup=1,
        )
    assert set(result["measurements"]) == {
        "serial_one_stack", "serial_two_stack", "packed_two_stack",
        "draft_head", "draft_select_debug", "accept_debug"
    }
    torch.testing.assert_close(
        model.predictor_k_cache[:, :1, :4], k_saved, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache[:, :1, :4], v_saved, atol=0, rtol=0
    )
    assert result["measurements"]["packed_two_stack"]["cuda_event_ms"] > 0

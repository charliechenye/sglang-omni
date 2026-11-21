# SPDX-License-Identifier: Apache-2.0
"""P12 H200 single-distribution selection and real one-vs-two Predictor cost."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import struct
import wave
from argparse import Namespace

import pytest
import torch

from experiments.qwen3_tts_cp_spec_k2.p12_quality_audit import (
    analyze as analyze_quality, edit_distance, tokenize,
    transcript_features, waveform_features,
)
from experiments.qwen3_tts_cp_spec_k2.p12_live_probe import validate_probe
from sglang_omni.models.qwen3_tts.code_predictor_spec_p12a_selector import (
    DRAWS, STAGES, profile_stage,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p12b_timing import (
    capture_ar_vs_packed,
)


def test_p12_quality_wer_and_trigram_repetition_cpu():
    assert tokenize("Hello, HELLO! it's me.") == [
        "hello", "hello", "it's", "me",
    ]
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    d = transcript_features(
        "we are ready we are ready we are ready", "we are ready"
    )
    assert d["wer"] > 1.
    assert any(x["count"] >= 2 for x in d["repeated_trigrams"])


def test_p12_wav_integrity_rms_trailing_silence(tmp_path):
    dest = tmp_path / "real.wav"
    sound = [8000] * 2400 + [0] * 2400
    with wave.open(str(dest), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(struct.pack("<" + "h"*len(sound), *sound))
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    result = waveform_features(dest, sha)
    assert result["sha256_verified"] is True
    assert result["duration_s"] == pytest.approx(.2)
    assert result["silence_fraction_20ms"] == pytest.approx(.5)
    assert result["trailing_silence_s"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="WAV SHA mismatch"):
        waveform_features(dest, "0"*64)


def test_p12_historical_wav_audit_does_not_require_p12b_gpu_results(tmp_path):
    # Regression: P11B WAVs already exist, but the optional P12B -022
    # H200 AR-vs-packed directory has NOT been collected. This must
    # work without that flag and explicitly say target cost UNMEASURED.
    audio = tmp_path / "audio"
    audio.mkdir()
    sound = [8000] * 2400 + [0] * 2400
    visits = []
    for seed in (20261120, 20261121):
        for i, mode in enumerate(
            ("native", "k2", "hybrid", "hybrid", "k2", "native")
        ):
            tag = f"seed{seed}_{i:02d}_{mode}"
            wav = audio / f"{tag}.wav"
            with wave.open(str(wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(struct.pack("<" + "h" * len(sound), *sound))
            visits.append({
                "tag": tag,
                "seed": seed,
                "mode": mode,
                "duration_s": 0.2,
                "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            })
    report = audio / "report.json"
    report.write_text(json.dumps({
        "status": (
            "P11B_REAL_ON_POLICY_HYBRID_AR_K2_"
            "ALL_STOP_ACCOUNTING_PASS_NOT_PARITY_OR_SPEED"
        ),
        "source_sha": "a" * 40,
        "seeds": [20261120, 20261121],
        "visits": visits,
    }))
    args = Namespace(
        p11b_report=report,
        p12b_timing_dir=None,
        speaker_embeddings=None,
        asr="none",
        asr_model="small.en",
        asr_device="cpu",
        reference_text="hello",
        silence_threshold_dbfs=-45.0,
    )
    result = analyze_quality(args)
    assert result["status"] == "P12B_WAV_STATS_PASS_ASR_OR_SPEAKER_UNMEASURED"
    assert result["asr_engine"] == "NOT_MEASURED"
    assert result["speaker_similarity_status"] == "NOT_MEASURED"
    assert result["target_forward_component_projection"] is None
    assert len(result["seed_comparison"]) == 6
    assert all(
        row["historical_projected_target_cuda_ms_per_codec_frame"] is None
        for row in result["seed_comparison"]
    )
    assert all(row["same_audio_statistics"] for row in result["seed_comparison"])

    # Explicitly specifying missing/incomplete GPU evidence must STOP
    # with an actionable error instead of claiming a measurement.
    args.p12b_timing_dir = tmp_path / "p12b-022" / "ar_vs_pair"
    with pytest.raises(FileNotFoundError, match="P12B_TIMING_EVIDENCE_MISSING"):
        analyze_quality(args)
    args.p12b_timing_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="P12B_TIMING_EVIDENCE_INCOMPLETE"):
        analyze_quality(args)


def test_p12_selector_has_no_cpu_sync_in_device_kernels():
    from sglang_omni.models.qwen3_tts import (
        code_predictor_spec_p12a_selector as mod,
    )
    assert len(DRAWS) == 4
    assert STAGES == ("q_before_verifier", "p_after_verifier")
    for fn in (mod.fast_selector, mod.single_top50_kernel.fn):
        src = inspect.getsource(fn)
        assert ".item(" not in src
        assert "torch.multinomial" not in src
    assert "tl.topk(" in inspect.getsource(mod.single_top50_kernel.fn)
    # Two source-specific hook points prevent a simultaneous p/q
    # dual-topk micrograph from being mistaken for actual causal path.
    source = inspect.getsource(mod.maybe_observe_q)
    assert "_qt_p12a_pending" in source
    source = inspect.getsource(mod.maybe_observe_p)
    assert "pending" in source


def test_p12_selector_tie_no_selection_keeps_cost_but_never_passes():
    stages = {
        name: {
            "status": "P12A_TIE_OR_DISTRIBUTION_MISMATCH_NO_SELECTION",
            "selection_eligible": False,
            "selected_ids_match": False,
            "selected_support_match": False,
            "probability_match": False,
            "fixed_uniform_draws": [0.125, 0.625, 0.995, 0.333],
            "times_ms": {
                "torch_topk_reference": [.05] * 4,
                "triton_single_top50": [.02] * 4,
            },
        }
        for name in ("q_before_verifier", "p_after_verifier")
    }
    row = {
        "schema": "qwen3tts-cpspec-p12a-causal-stage-single-distribution-top50-v1",
        "status": "P12A_CAUSAL_REAL_Q_PRE_AND_P_POST_NO_SELECTION_TIE_OR_DRAW_MISMATCH",
        "selection_eligible": False, "stages": stages,
    }
    assert validate_probe(row, mode="selector") is False
    row["selection_eligible"] = True
    with pytest.raises(ValueError, match="inconsistent"):
        validate_probe(row, mode="selector")


def test_p12_report_gates_refuse_false_component_pass():
    good = {
        "schema": "qwen3tts-cpspec-p12b-real-target-ar-vs-packed-verifier-cost-v1",
        "status": "P12B_REAL_TARGET_AR_ONE_VS_PACKED_TWO_KV_PARITY_PASS_COMPONENT_TIMING_NOT_E2E",
        "predictor_layers": 5, "parity_pass": True,
        "checks": {str(i): {"close": True} for i in range(9)},
        "median_cuda_event_ms": {"ar_one": .1, "packed_two": .14},
    }
    validate_probe(good, mode="ar")
    good["parity_pass"] = False
    with pytest.raises(ValueError, match="parity"):
        validate_probe(good, mode="ar")


@pytest.mark.accelerator
@pytest.mark.parametrize("top_p", [1.0, 0.9])
def test_p12a_h200_tie_free_single_distribution_top50_graph(top_p):
    if not torch.cuda.is_available():
        pytest.skip("H200 CUDA required")
    # Strictly unique values: CUDA/Torch topk tied BF16 outcomes are a
    # *separate fail-closed production blocker*, not concealed here.
    idx = torch.arange(2048, device="cuda", dtype=torch.float32)
    logits = -(idx * .015) + .0002 * torch.sin(idx)
    logits = logits.view(1, -1).contiguous()
    report = profile_stage(
        logits, temperature=.9, top_p=top_p, repeats=8
    )
    assert report["status"].startswith(
        "P12A_SINGLE_DISTRIBUTION_TOP50_EXACT_MATCH_PASS"
    )
    assert report["selected_ids_match"] is True
    assert report["selected_support_match"] is True
    assert report["probability_match"] is True
    assert report["fixed_uniform_draws"] == list(DRAWS)
    assert all(
        len(blocks) == 4 and all(math.isfinite(x) and x > 0 for x in blocks)
        for blocks in report["times_ms"].values()
    )


@pytest.mark.accelerator
def test_p12b_h200_fixture_one_vs_pair_kv_and_graph_restore(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA H200 required")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        DTYPE, HIDDEN, HEAD_DIM, MAX_BS, NUM_KV_HEADS,
        build_talker, with_predictor_layers,
    )
    import sglang_omni.models.qwen3_tts.sglang_model as model_module

    monkeypatch.setattr(
        model_module, "apply_qk_norm", lambda q, k, **_: (q, k)
    )
    dev = torch.device("cuda")
    model = with_predictor_layers(build_talker(dev), 2)
    capacity = 18
    pos = torch.arange(capacity, device=dev, dtype=torch.long)
    model.predictor_positions = pos
    model.predictor_position_rows = pos[:, None].expand(
        capacity, MAX_BS
    ).contiguous()
    model.predictor_cache_slots = (
        torch.arange(MAX_BS, device=dev, dtype=torch.long)[None,:] * capacity
        + pos[:,None]
    ).contiguous()
    model.predictor_k_cache = torch.zeros(
        (2, MAX_BS, capacity, NUM_KV_HEADS, HEAD_DIM), device=dev, dtype=DTYPE
    )
    model.predictor_v_cache = torch.zeros_like(model.predictor_k_cache)
    model.predictor_k_rows = [
        layer.view(MAX_BS * capacity, -1)
        for layer in model.predictor_k_cache
    ]
    model.predictor_v_rows = [
        layer.view(MAX_BS * capacity, -1)
        for layer in model.predictor_v_cache
    ]
    torch.manual_seed(20261122)
    pair = torch.randn(1, 2, HIDDEN, device=dev).to(DTYPE)
    with torch.inference_mode():
        for depth in (0, 2, 4, 6):
            model.predictor_forward_tokens(
                token_embeds=pair.clone(), batch_size=1, cache_len=depth
            )
        old_k = model.predictor_k_cache[:, :1, :16].clone()
        old_v = model.predictor_v_cache[:, :1, :16].clone()
        result = capture_ar_vs_packed(
            model, last_embedding=pair[:, :1, :].clone(),
            proposed_embedding=pair[:, 1:, :].clone(),
            cache_len=8, repeats=8,
        )
    assert result["parity_pass"] is True
    assert result["status"].startswith(
        "P12B_REAL_TARGET_AR_ONE_VS_PACKED_TWO_KV_PARITY_PASS"
    )
    assert result["predictor_layers"] == 2
    assert len(result["event_ms_by_block"]["ar_one"]) == 4
    assert len(result["event_ms_by_block"]["packed_two"]) == 4
    torch.testing.assert_close(
        model.predictor_k_cache[:, :1, :16], old_k, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache[:, :1, :16], old_v, atol=0, rtol=0
    )

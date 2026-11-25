# SPDX-License-Identifier: Apache-2.0
"""CPU-only guards for complete-output K2 smoke and model log evidence."""
from __future__ import annotations

import base64
import io
import wave

import pytest

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    decode_single_batch_response,
    parse_k2_log_entries,
    wav_info,
)


def sample_wav(samples: int = 1920) -> bytes:
    data = io.BytesIO()
    with wave.open(data, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * samples)
    return data.getvalue()


def test_single_item_batch_exposes_true_finish_reason_and_valid_audio() -> None:
    audio = sample_wav(samples=1920)
    fixture = {
        "total": 1,
        "succeeded": 1,
        "failed": 0,
        "results": [{
            "index": 0, "status": "success",
            "format": "wav", "finish_reason": "stop",
            "audio_data": base64.b64encode(audio).decode("ascii"),
        }],
    }
    recovered, finish = decode_single_batch_response(fixture)
    assert recovered == audio and finish == "stop"
    stats = wav_info(recovered)
    assert stats["sample_rate"] == 24000
    assert stats["duration_s"] == pytest.approx(0.08)
    assert stats["approx_80ms_codec_frames"] == pytest.approx(1.0)
    assert stats["pcm_samples"] == 1920

    fixture["results"][0]["finish_reason"] = "length"
    _, finish = decode_single_batch_response(fixture)
    assert finish == "length"
    assert finish != "stop"  # caller must fail complete-output gate

    fixture["results"][0].pop("finish_reason")
    with pytest.raises(ValueError, match="finish_reason"):
        decode_single_batch_response(fixture)


def test_batch_must_not_silently_return_partial_or_bad_audio() -> None:
    fixture = {
        "total": 1, "succeeded": 0, "failed": 1,
        "results": [{
            "index": 0, "status": "error",
            "error": {"message": "CUDA failure"},
        }],
    }
    with pytest.raises(ValueError):
        decode_single_batch_response(fixture)
    with pytest.raises(ValueError, match="WAV"):
        wav_info(b"not wav")
    with pytest.raises(ValueError):
        wav_info(sample_wav(0))


def test_cp_k2_log_parser_and_invariants() -> None:
    text = (
        "2026-11-21 INFO main "
        "QTTS_CP_SPEC_K2 frame=1 rounds=13 proposed=13 "
        "accepted=1 bonus=1 rejected=12 committed_residual=15"
    )
    parsed = parse_k2_log_entries(text)
    assert len(parsed) == 1
    assert parsed[0]["accepted"] == 1
    assert parsed[0]["bonus"] == 1
    assert parse_k2_log_entries("no spec logs") == []

    invalid = text.replace("accepted=1", "accepted=99")
    with pytest.raises(ValueError, match="inconsistent"):
        parse_k2_log_entries(invalid)
    invalid = text.replace("committed_residual=15", "committed_residual=14")
    with pytest.raises(ValueError, match="inconsistent"):
        parse_k2_log_entries(invalid)

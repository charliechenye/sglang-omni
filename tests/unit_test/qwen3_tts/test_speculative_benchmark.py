# SPDX-License-Identifier: Apache-2.0
"""Pure CPU checks for Qwen3-TTS serving benchmark artifacts."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from sglang_omni.utils.qwen3_tts_speculative_benchmark import (
    SpeechResult,
    parse_wav_audio,
    summarize_speech_results,
    write_json_new,
)


def make_pcm_wav(samples: np.ndarray, sample_rate_hz: int = 24000) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(samples.astype("<i2").tobytes())
    return stream.getvalue()


def test_benchmark_parses_and_checks_pcm_wav() -> None:
    samples = np.array([0, 4096, -4096, 8192], dtype=np.int16)

    duration, sample_rate, channels, width, rms, peak, clipping = parse_wav_audio(
        make_pcm_wav(samples)
    )

    assert duration == pytest.approx(4 / 24000)
    assert sample_rate == 24000
    assert channels == 1
    assert width == 2
    assert rms > 0.0
    assert peak == pytest.approx(0.25)
    assert clipping == 0.0


def test_benchmark_rejects_silent_audio() -> None:
    with pytest.raises(RuntimeError, match="silent audio"):
        parse_wav_audio(make_pcm_wav(np.zeros(8, dtype=np.int16)))


def test_benchmark_rejects_truncated_audio() -> None:
    audio = make_pcm_wav(np.array([1024, -1024, 2048], dtype=np.int16))

    with pytest.raises(RuntimeError, match="truncated WAV"):
        parse_wav_audio(audio[:-2])


def test_audio_summary_retains_request_metrics_and_does_not_claim_ttfa() -> None:
    results = [
        SpeechResult(
            request_index=0,
            latency_seconds=0.8,
            audio_bytes=1200,
            audio_duration_seconds=1.0,
            sample_rate_hz=24000,
            channels=1,
            sample_width_bytes=2,
            rms=0.1,
            peak=0.3,
            clipping_fraction=0.0,
        ),
        SpeechResult(
            request_index=1,
            latency_seconds=1.2,
            audio_bytes=2400,
            audio_duration_seconds=2.0,
            sample_rate_hz=24000,
            channels=1,
            sample_width_bytes=2,
            rms=0.2,
            peak=0.5,
            clipping_fraction=0.0,
        ),
    ]

    summary = summarize_speech_results(results)

    assert summary["response_count"] == 2
    assert summary["audio_duration_seconds_total"] == 3.0
    assert summary["real_time_factor_p50"] == pytest.approx(0.7)
    assert summary["ttfa_measured"] is False
    request_rows = summary["requests"]
    assert isinstance(request_rows, list)
    assert request_rows[1]["request_index"] == 1


def test_benchmark_artifacts_do_not_overwrite_existing_files(tmp_path) -> None:
    output_path = tmp_path / "run.json"
    write_json_new(output_path, {"status": "complete"})

    with pytest.raises(FileExistsError):
        write_json_new(output_path, {"status": "replacement"})

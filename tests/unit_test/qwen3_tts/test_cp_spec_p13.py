# SPDX-License-Identifier: Apache-2.0
"""P13 offline pause/timestamp evidence — CPU only; no model or GPU needed."""
from __future__ import annotations

import hashlib
import json
import wave
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.qwen3_tts_cp_spec_k2.p13_acoustic_timeline import (
    DEFAULT_THRESHOLDS,
    analyze,
    compare_word_timelines,
    first_mode_visits,
    pause_context,
    quiet_regions,
    read_validated_wav,
    rms_db_20ms,
    text_tokens,
    word_timeline,
    write_pcm_clip,
)

SHA = "a" * 40


def _word(text, start, end):
    return {
        "text": text, "normalized": text.lower(),
        "start_s": start, "end_s": end,
    }


def _wav(path: Path, seconds: float, *, quiet_start=.4, quiet_end=.6):
    frames = int(round(seconds * 24000))
    samples = np.full((frames,), 10000, dtype="<i2")
    a, b = int(round(quiet_start * 24000)), int(round(quiet_end * 24000))
    samples[max(0, a):min(frames, b)] = 0
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(samples.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cohort(tmp_path: Path):
    root = tmp_path / "smoke"
    root.mkdir()
    seed = 20261121
    sequence = ("native", "k2", "hybrid", "hybrid", "k2", "native")
    visits = []
    for i, mode in enumerate(sequence):
        seconds = 1.0 if mode != "hybrid" else 1.4
        tag = f"seed{seed}_{i:02d}_{mode}"
        p = root / f"{tag}.wav"
        sha = _wav(p, seconds, quiet_start=.4, quiet_end=(
            .9 if mode == "hybrid" else .6
        ))
        visits.append({
            "seed": seed, "tag": tag, "mode": mode,
            "finish_reason": "stop",
            "wav_sha256": sha,
            "duration_s": seconds,
        })
    report = root / "report.json"
    report.write_text(json.dumps({
        "status": (
            "P11B_REAL_ON_POLICY_HYBRID_AR_K2_"
            "ALL_STOP_ACCOUNTING_PASS_NOT_PARITY_OR_SPEED"
        ),
        "source_sha": SHA, "seeds": [seed],
        "visits": visits,
    }))
    return report


def _args(source: Path, target: Path):
    return Namespace(
        p11b_report=source,
        p12_asr_report=None,
        out=target,
        seed=20261121,
        thresholds="-35,-40,-45,-50",
        min_pause_ms=150,
        clip_count=2,
        asr="none",
        asr_model="small.en",
        asr_device="cpu",
        asr_input="file",
    )


def test_p13_source_attestation_and_first_visits(tmp_path):
    p = _cohort(tmp_path)
    data = json.loads(p.read_text())
    visits = first_mode_visits(data, seed=20261121)
    assert set(visits) == {"native", "k2", "hybrid"}
    assert visits["k2"]["tag"] == "seed20261121_01_k2"
    assert visits["hybrid"]["tag"] == "seed20261121_02_hybrid"
    modified = json.loads(p.read_text())
    modified["visits"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="NONSTOP"):
        first_mode_visits(modified, seed=20261121)
    original = p.parent / f"{visits['hybrid']['tag']}.wav"
    loaded = read_validated_wav(original, visits["hybrid"])
    assert loaded["duration_s"] == pytest.approx(1.4)
    bad = dict(visits["hybrid"], wav_sha256="0"*64)
    with pytest.raises(ValueError, match="SHA256_MISMATCH"):
        read_validated_wav(original, bad)


def test_p13_20ms_rms_pause_threshold_sweep():
    # 10 windows above threshold, 12 windows of RMS ~ -44dBFS and
    # 18 windows above threshold. 150ms -> 160ms effective min pause.
    samples = np.r_[
        np.full(10 * 480, .5, dtype=np.float32),
        np.full(12 * 480, .006, dtype=np.float32),
        np.full(18 * 480, .5, dtype=np.float32),
    ]
    db, leftover = rms_db_20ms(samples)
    assert leftover == 0.
    assert len(db) == 40
    assert tuple(DEFAULT_THRESHOLDS) == (-35., -40., -45., -50.)
    at35 = quiet_regions(db, threshold_dbfs=-35, min_pause_ms=150)
    at50 = quiet_regions(db, threshold_dbfs=-50, min_pause_ms=150)
    assert at35["min_pause_effective_ms"] == 160
    assert at35["long_quiet_count"] == 1
    assert at35["long_quiet_total_s"] == pytest.approx(.24)
    assert at35["regions"][0]["start_s"] == pytest.approx(.2)
    assert at35["regions"][0]["end_s"] == pytest.approx(.44)
    assert at50["quiet_total_s"] == 0
    assert at50["long_quiet_count"] == 0


def test_p13_whisper_timeline_accounts_for_exact_audio_delta():
    k2words = [_word("hello", .1, .3), _word("world", .5, .7)]
    hywords = [_word("hello", .15, .45), _word("world", .8, 1.1)]
    a = {"words": k2words, "duration_s": 1.0}
    b = {"words": hywords, "duration_s": 1.5}
    timeline = word_timeline(k2words, duration_s=1.0)
    assert timeline["timeline_reconstructed_s"] == pytest.approx(1.)
    outcome = compare_word_timelines(a, b)
    assert outcome["status"].endswith("WORD_TIMING_MATCH")
    assert outcome["duration_delta_s"] == pytest.approx(.5)
    assert sum(outcome["decomposition_delta_s"].values()) == pytest.approx(.5)
    assert outcome["decomposition_delta_s"]["interword_signed_s"] == pytest.approx(.15)
    assert outcome["decomposition_delta_s"]["word_spans_s"] == pytest.approx(.2)
    ctxt = pause_context({"start_s": .3, "end_s": .5}, k2words)
    assert ctxt["label"] == "BETWEEN_ESTIMATED_WORDS"
    assert ctxt["prev_word"] == "hello"
    assert ctxt["next_word"] == "world"
    assert text_tokens("Code Predictor!") == ["code", "predictor"]


def test_p13_changed_asr_words_never_positionally_align():
    k2 = {"words": [_word("we", .1, .2)], "duration_s": .5}
    hybrid = {"words": [_word("they", .1, .3)], "duration_s": .8}
    report = compare_word_timelines(k2, hybrid)
    assert report["status"].startswith("P13_ASR_WORD_SEQUENCE_MISMATCH")
    assert "decomposition_delta_s" not in report


def test_p13_full_offline_wav_without_asr_passes_and_never_overwrites(tmp_path):
    source = _cohort(tmp_path)
    output = tmp_path / "p13"
    report = analyze(_args(source, output))
    assert report["status"] == "P13_RMS_SWEEP_ONLY_WORD_TIMESTAMPS_NOT_MEASURED"
    assert report["seed"] == 20261121
    assert report["p11b_source_sha"] == SHA
    assert report["k2_hybrid_whisper_word_timing_comparison"] is None
    assert len(report["k2_hybrid_threshold_comparison"]) == 4
    assert report["k2_hybrid_threshold_comparison"][2][
        "duration_delta_s"
    ] == pytest.approx(.4)
    assert report["k2_hybrid_threshold_comparison"][2][
        "total_below_threshold_delta_s"
    ] > 0
    assert (output / "report.json").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "rms_20ms.csv").is_file()
    assert (output / "long_quiet_regions.csv").is_file()
    assert (output / "whisper_words.csv").is_file()
    assert report["clip_manifest"]
    for clip in report["clip_manifest"]:
        with wave.open(clip["clip_path"], "rb") as w:
            assert w.getframerate() == 24000
        assert hashlib.sha256(Path(clip["clip_path"]).read_bytes()).hexdigest() == (
            clip["sha256"]
        )
    with pytest.raises(FileExistsError):
        analyze(_args(source, output))


class _FakeASR:
    """Offline fake words: verifies generator materialization and transcript gate."""

    def transcribe(self, source, **kwargs):
        assert kwargs["word_timestamps"] is True
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["vad_filter"] is False
        path = str(source)
        duration = 1.4 if "hybrid" in path else 1.0
        words = [
            SimpleNamespace(
                word=" Hello", start=.1, end=.28, probability=.99
            ),
            SimpleNamespace(
                word=" world.", start=.7 if duration > 1 else .55,
                end=duration-.1, probability=.98,
            ),
        ]
        segments = [SimpleNamespace(text="Hello world.", words=words)]
        # Always an iterable; Faster-Whisper returns a lazy generator.
        return iter(segments), SimpleNamespace(language="en")


def test_p13_fake_word_timestamp_report_marks_estimated_not_forced(tmp_path):
    source = _cohort(tmp_path)
    args = _args(source, tmp_path / "p13-asr")
    args.asr = "faster-whisper"
    result = analyze(args, asr_model_override=_FakeASR())
    assert result["status"].startswith(
        "P13_RMS_SWEEP_AND_WHISPER_WORD_TIMESTAMPS_MEASURED"
    )
    compare = result["k2_hybrid_whisper_word_timing_comparison"]
    assert compare["status"].endswith("WORD_TIMING_MATCH")
    assert sum(compare["decomposition_delta_s"].values()) == pytest.approx(.4)
    assert result["modes_data"]["hybrid"]["asr"]["word_count"] == 2
    assert "NOT independent forced alignment" not in result["status"]
    assert len((args.out / "whisper_words.csv").read_text().splitlines()) == 7

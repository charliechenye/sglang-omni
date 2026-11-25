# SPDX-License-Identifier: Apache-2.0
"""P13 — offline acoustic pause & Whisper word-timeline diagnosis (NO TTS).

Reads the already-generated P11B WAVs + source-SHA and byte SHA256
from its PASS report. Extracts first native/K2/Hybrid visit for one seed.
Measures low-RMS regions at -35/-40/-45/-50 dBFS on exact 20ms windows;
identifies >=150ms low-RMS intervals (160ms effective at 20ms resolution);
optionally obtains word timestamps from faster-whisper with
word_timestamps=True, compares same-word temporal gaps and durations,
and exports short original-PCM clips for listening.

Not a speech VAD, forced alignment, acoustic stop-causality proof,
speaker similarity test, or serving performance benchmark. The original
WAVs are never mutated. No GPU or Omni worker is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import wave
from pathlib import Path

import numpy as np

SCHEMA = "qwen3tts-cpspec-p13-offline-rms-whisper-word-timeline-v1"
MODES = ("native", "k2", "hybrid")
RATE = 24000
WINDOW_MS = 20
DEFAULT_THRESHOLDS = (-35.0, -40.0, -45.0, -50.0)
DEFAULT_SOURCE_SHA = "50dae7a07484dd82c6d200a818569be705ea89e7"


def word_normalize(text: str) -> str:
    """Punctuation-insensitive lexical word (never rewrite transcript)."""
    return "".join(re.findall(r"[a-z0-9']+", text.lower()))


def text_tokens(text: str) -> list[str]:
    return [w for w in (word_normalize(x) for x in text.split()) if w]


def first_mode_visits(p11b: dict, *, seed: int) -> dict[str, dict]:
    """Ground first visit for each mode in actual P11B 6-visit recording."""
    if not p11b.get("status", "").startswith(
        "P11B_REAL_ON_POLICY_HYBRID_AR_K2_ALL_STOP_ACCOUNTING_PASS"
    ):
        raise ValueError("P13_NEEDS_P11B_COMPLETE_NATURAL_STOP_PASS")
    sha = p11b.get("source_sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("P13_P11B_SOURCE_SHA_NOT_ATTESTED")
    visits = [x for x in p11b.get("visits", []) if x.get("seed") == seed]
    if len(visits) != 6 or [x.get("mode") for x in visits] != [
        "native", "k2", "hybrid", "hybrid", "k2", "native"
    ]:
        raise ValueError("P13_EXPECTS_SIX_ORDERED_NATIVE_K2_HYBRID_VISITS_PER_SEED")
    chosen = {}
    for pos, (idx, mode) in enumerate(((0, "native"), (1, "k2"), (2, "hybrid"))):
        visit = visits[idx]
        expected_tag = f"seed{seed}_{idx:02d}_{mode}"
        if (
            visit.get("tag") != expected_tag
            or visit.get("finish_reason") != "stop"
            or not isinstance(visit.get("wav_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", visit["wav_sha256"])
        ):
            raise ValueError(f"P13_BAD_SOURCE_VISIT_OR_NONSTOP: {expected_tag}")
        chosen[mode] = visit
    return chosen


def read_validated_wav(path: Path, visit: dict) -> dict:
    """Immutable PCM byte evidence, both hash and decoded duration verified."""
    content = path.read_bytes()
    actual_sha = hashlib.sha256(content).hexdigest()
    if actual_sha != visit["wav_sha256"]:
        raise ValueError(f"P13_WAV_SHA256_MISMATCH: {path}")
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1 or reader.getsampwidth() != 2
            or reader.getframerate() != RATE or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"P13_EXPECTS_MONO_24KHZ_PCM16: {path}")
        frames = reader.getnframes()
        pcm_bytes = reader.readframes(frames)
    if frames < 1 or len(pcm_bytes) != frames * 2:
        raise ValueError(f"P13_TRUNCATED_PCM_WAV: {path}")
    duration = frames / RATE
    if abs(duration - float(visit["duration_s"])) > .001:
        raise ValueError(
            f"P13_REPORTED_WAV_DURATION_MISMATCH: {path} "
            f"actual={duration} reported={visit['duration_s']}"
        )
    pcm = np.frombuffer(pcm_bytes, dtype="<i2")
    samples = pcm.astype(np.float32) / 32768.0
    return {
        "path": str(path), "sha256": actual_sha,
        "duration_s": duration, "frames": frames,
        "raw_pcm_bytes": pcm_bytes, "float_audio": samples,
    }


def rms_db_20ms(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Per 20ms RMS from PCM (not a speech VAD); leftover samples explicit."""
    n = (len(samples) // 480) * 480
    if n == 0:
        raise ValueError("P13_WAV_TOO_SHORT_FOR_20MS_RMS")
    blocks = samples[:n].reshape(-1, 480).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(blocks), axis=1))
    decibels = 20.0 * np.log10(np.maximum(rms, 1.0e-12))
    return decibels, (len(samples) - n) / RATE


def quiet_regions(
    db: np.ndarray, *, threshold_dbfs: float,
    min_pause_ms: int = 150,
) -> dict:
    if db.ndim != 1 or not np.isfinite(db).all():
        raise ValueError("P13_FINITE_MONO_RMS_WINDOWS_REQUIRED")
    if not 20 <= min_pause_ms <= 5000:
        raise ValueError("P13_MIN_PAUSE_MS_MUST_BE_20_TO_5000")
    required_windows = math.ceil(min_pause_ms / WINDOW_MS)
    quiet = db < threshold_dbfs
    edges = np.diff(np.r_[False, quiet, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    all_regions = [
        {
            "start_s": round(int(start) * .02, 5),
            "end_s": round(int(end) * .02, 5),
            "duration_s": round((int(end) - int(start)) * .02, 5),
            "windows": int(end) - int(start),
        }
        for start, end in zip(starts, ends)
    ]
    long_regions = [
        r for r in all_regions if r["windows"] >= required_windows
    ]
    trailing_frames = (
        int(len(db) - starts[-1]) if len(starts) and ends[-1] == len(db)
        else 0
    )
    leading_frames = (
        int(ends[0]) if len(starts) and starts[0] == 0 else 0
    )
    return {
        "threshold_dbfs": threshold_dbfs,
        "window_ms": WINDOW_MS,
        "min_pause_ms": min_pause_ms,
        "min_pause_effective_ms": required_windows * WINDOW_MS,
        "quiet_total_s": round(float(quiet.sum()) * .02, 5),
        "quiet_fraction": float(quiet.mean()),
        "long_quiet_total_s": round(
            sum(r["duration_s"] for r in long_regions), 5
        ),
        "long_quiet_count": len(long_regions),
        "leading_quiet_s": round(leading_frames * .02, 5),
        "trailing_quiet_s": round(trailing_frames * .02, 5),
        "regions": long_regions,
        "quiet_all_regions_count": len(all_regions),
    }


def _valid_word_timing(word: dict, *, duration: float) -> bool:
    return (
        word.get("normalized") != ""
        and all(
            isinstance(word.get(k), (float, int))
            and math.isfinite(word[k])
            for k in ("start_s", "end_s")
        )
        and 0 <= word["start_s"] <= word["end_s"] <= duration + .15
    )


def transcribe_words(model, path: Path, *, audio_input: str) -> dict:
    """Whisper estimated timestamps; NOT a ground-truth forced aligner."""
    source = str(path)
    if audio_input == "pcm-array":
        from experiments.qwen3_tts_cp_spec_k2.p12_quality_audit import (
            wav_24k_pcm16_to_whisper_16k_array,
        )
        source = wav_24k_pcm16_to_whisper_16k_array(path)
    elif audio_input != "file":
        raise ValueError("P13_ASR_INPUT_MUST_BE_FILE_OR_PCM_ARRAY")
    try:
        segments, info = model.transcribe(
            source,
            language="en",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        # faster-whisper returns a lazy generator; process ALL segments.
        materialized = list(segments)
    except TypeError as exc:
        if "metadata_errors" in str(exc):
            raise RuntimeError(
                "P13_PYAV19_INCOMPATIBLE: pin 'av==18.1.0' inside the "
                "independent ~/.venv-qwen3tts-asr environment OR use "
                "--asr-input pcm-array with SciPy. Do not modify Omni's venv."
            ) from exc
        raise
    words = []
    for segment in materialized:
        if segment.words is None:
            raise RuntimeError(
                "P13_WORD_TIMESTAMPS_MISSING_FROM_FASTER_WHISPER"
            )
        for w in segment.words:
            normalized = word_normalize(w.word)
            if not normalized:
                continue
            row = {
                "text": w.word.strip(),
                "normalized": normalized,
                "start_s": float(w.start),
                "end_s": float(w.end),
                "probability": (
                    None if w.probability is None else float(w.probability)
                ),
            }
            words.append(row)
    return {
        "transcript": " ".join(s.text.strip() for s in materialized),
        "words": words,
        "word_count": len(words),
        "timestamp_method": (
            "faster-whisper word_timestamps=True cross-attention/DTW; "
            "estimated word boundaries, NOT independent forced alignment"
        ),
        "language": getattr(info, "language", "unknown"),
    }


def word_timeline(words: list[dict], *, duration_s: float) -> dict:
    if not words or not all(
        _valid_word_timing(w, duration=duration_s) for w in words
    ):
        raise ValueError("P13_INVALID_OR_EMPTY_WHISPER_WORD_BOUNDARIES")
    # Keep signed word gaps: when Whisper word spans overlap, a negative
    # gap preserves the exact decomposition identity rather than
    # incorrectly attributing extra time to positive pauses.
    word_total = sum(w["end_s"] - w["start_s"] for w in words)
    word_gaps = [
        {
            "left": words[i]["normalized"],
            "right": words[i + 1]["normalized"],
            "start_s": words[i]["end_s"],
            "end_s": words[i + 1]["start_s"],
            "signed_gap_s": words[i + 1]["start_s"] - words[i]["end_s"],
        }
        for i in range(len(words) - 1)
    ]
    raw_gap_total = sum(x["signed_gap_s"] for x in word_gaps)
    first = words[0]["start_s"]
    last = duration_s - words[-1]["end_s"]
    reconstructed = first + word_total + raw_gap_total + last
    if abs(reconstructed - duration_s) > .0001:
        raise ValueError("P13_WHISPER_TIMELINE_NOT_DECOMPOSABLE")
    return {
        "leading_s": first,
        "trailing_s": last,
        "word_span_total_s": word_total,
        "signed_interword_gaps_s": raw_gap_total,
        "overlap_gap_count": sum(
            x["signed_gap_s"] < 0 for x in word_gaps
        ),
        "word_gaps": word_gaps,
        "timeline_reconstructed_s": reconstructed,
    }


def compare_word_timelines(k2: dict, hybrid: dict) -> dict:
    a = k2.get("words", [])
    b = hybrid.get("words", [])
    tokens_a, tokens_b = (
        [w["normalized"] for w in vals] for vals in (a, b)
    )
    if tokens_a != tokens_b:
        return {
            "status": "P13_ASR_WORD_SEQUENCE_MISMATCH_NO_POSITIONAL_ATTRIBUTION",
            "k2_tokens": tokens_a, "hybrid_tokens": tokens_b,
            "note": (
                "Do not align by index if tokenization or ASR output differs. "
                "A real forced aligner could be added for independent timing."
            ),
        }
    ta = word_timeline(a, duration_s=k2["duration_s"])
    tb = word_timeline(b, duration_s=hybrid["duration_s"])
    word_changes = [
        {
            "word_index": i,
            "word": aw["normalized"],
            "k2_start_s": aw["start_s"],
            "hybrid_start_s": bw["start_s"],
            "k2_span_s": aw["end_s"] - aw["start_s"],
            "hybrid_span_s": bw["end_s"] - bw["start_s"],
            "span_delta_s": (
                bw["end_s"] - bw["start_s"]
                - (aw["end_s"] - aw["start_s"])
            ),
            "start_shift_s": bw["start_s"] - aw["start_s"],
        }
        for i, (aw, bw) in enumerate(zip(a, b))
    ]
    gap_changes = [
        {
            "left_word_index": i,
            "between": f"{a[i]['normalized']} / {a[i + 1]['normalized']}",
            "k2_signed_gap_s": ta["word_gaps"][i]["signed_gap_s"],
            "hybrid_signed_gap_s": tb["word_gaps"][i]["signed_gap_s"],
            "gap_delta_s": (
                tb["word_gaps"][i]["signed_gap_s"]
                - ta["word_gaps"][i]["signed_gap_s"]
            ),
            "k2_gap_start_s": a[i]["end_s"],
            "hybrid_gap_start_s": b[i]["end_s"],
        }
        for i in range(len(a) - 1)
    ]
    delta = hybrid["duration_s"] - k2["duration_s"]
    parts = {
        "leading_s": tb["leading_s"] - ta["leading_s"],
        "interword_signed_s": (
            tb["signed_interword_gaps_s"] - ta["signed_interword_gaps_s"]
        ),
        "word_spans_s": (
            tb["word_span_total_s"] - ta["word_span_total_s"]
        ),
        "trailing_s": tb["trailing_s"] - ta["trailing_s"],
    }
    if abs(sum(parts.values()) - delta) > .0001:
        raise RuntimeError("P13_WORD_TIME_DIFFERENCE_ACCOUNTING_FAIL")
    return {
        "status": "P13_WHISPER_ESTIMATED_POSITIONAL_WORD_TIMING_MATCH",
        "word_count": len(a),
        "duration_delta_s": delta,
        "decomposition_delta_s": parts,
        "overlapping_word_gap_count_k2": ta["overlap_gap_count"],
        "overlapping_word_gap_count_hybrid": tb["overlap_gap_count"],
        "top_positive_gap_deltas": sorted(
            gap_changes, key=lambda x: x["gap_delta_s"], reverse=True
        )[:10],
        "top_positive_word_span_deltas": sorted(
            word_changes, key=lambda x: x["span_delta_s"], reverse=True
        )[:10],
        "all_gap_changes": gap_changes,
        "all_word_changes": word_changes,
        "limitations": [
            "Whisper word boundaries can shift when pauses or prosody shift",
            "Signed interword gaps can be negative with overlapping estimates",
            "A word-duration delta is NOT guaranteed voiced-speech duration",
            "Exact algebraic reconstruction is not causality or a forced aligner",
        ],
    }


def pause_context(region: dict, words: list[dict]) -> dict:
    if not words:
        return {"label": "WORD_TIMESTAMPS_NOT_MEASURED"}
    start, end = region["start_s"], region["end_s"]
    overlaps = [
        w for w in words
        if min(end, w["end_s"]) > max(start, w["start_s"]) + 1.0e-5
    ]
    if overlaps:
        return {
            "label": "OVERLAPS_ESTIMATED_WORD_TIME",
            "words": [w["normalized"] for w in overlaps[:6]],
        }
    left = [w for w in words if w["end_s"] <= start]
    right = [w for w in words if w["start_s"] >= end]
    if not left:
        label = "BEFORE_FIRST_ESTIMATED_WORD"
    elif not right:
        label = "AFTER_LAST_ESTIMATED_WORD"
    else:
        label = "BETWEEN_ESTIMATED_WORDS"
    return {
        "label": label,
        "prev_word": left[-1]["normalized"] if left else None,
        "next_word": right[0]["normalized"] if right else None,
    }


def write_pcm_clip(
    *, pcm: bytes, total_frames: int,
    start_s: float, end_s: float, out: Path, pad_s: float = .25,
) -> dict:
    first = max(0, math.floor((start_s - pad_s) * RATE))
    last = min(total_frames, math.ceil((end_s + pad_s) * RATE))
    if last <= first:
        raise ValueError("P13_EMPTY_CLIP")
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm[first * 2:last * 2])
    return {
        "clip_path": str(out),
        "clip_start_s": first / RATE,
        "clip_end_s": last / RATE,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def csv_file(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze(args, *, asr_model_override=None) -> dict:
    report_path = args.p11b_report.expanduser().resolve(strict=True)
    cohort = json.loads(report_path.read_text())
    seed = int(args.seed)
    selected = first_mode_visits(cohort, seed=seed)
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"P13 output already exists, preserve evidence: {output}"
        )
    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    if (
        len(thresholds) != 4 or len(set(thresholds)) != 4
        or any(not -80 <= t <= -15 for t in thresholds)
        or not 20 <= args.min_pause_ms <= 5000
        or not 0 <= args.clip_count <= 20
    ):
        raise ValueError("P13 thresholds need four distinct -80..-15 dBFS values")
    baseline = {}
    for mode in MODES:
        visit = selected[mode]
        p = report_path.parent / f"{visit['tag']}.wav"
        baseline[mode] = read_validated_wav(p, visit)

    model = asr_model_override
    if args.asr == "faster-whisper" and model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "P13_ASR_NOT_INSTALLED: run with the independent "
                "~/.venv-qwen3tts-asr/bin/python, or choose --asr none. "
                "Do not change the Omni serving venv."
            ) from exc
        model = WhisperModel(
            args.asr_model, device=args.asr_device,
            compute_type="int8" if args.asr_device == "cpu" else "float16",
        )
    if args.asr not in ("none", "faster-whisper"):
        raise ValueError("P13_ASR_MODE_UNSUPPORTED")
    if args.asr == "none" and asr_model_override is not None:
        raise ValueError("P13 cannot inject ASR model when --asr none")

    old_transcripts = {}
    if args.p12_asr_report is not None:
        previous = json.loads(
            args.p12_asr_report.expanduser().resolve(strict=True).read_text()
        )
        if previous.get("p11b_source_sha") != cohort["source_sha"]:
            raise ValueError("P13_P12_TRANSCRIPT_PRIOR_SOURCE_SHA_MISMATCH")
        for x in previous.get("items", []):
            if x.get("seed") == seed and x.get("tag") == selected.get(x.get("mode"), {}).get("tag"):
                old_transcripts[x["mode"]] = x.get("transcript")

    measured = {}
    rms_csv, regions_csv, word_csv = [], [], []
    for mode in MODES:
        item = baseline[mode]
        db, tail = rms_db_20ms(item["float_audio"])
        words_result = None
        if model is not None:
            words_result = transcribe_words(
                model, Path(item["path"]), audio_input=args.asr_input
            )
            if words_result["words"]:
                word_timeline(
                    words_result["words"], duration_s=item["duration_s"]
                )
            else:
                raise RuntimeError(f"P13_ASR_NO_WORD_TIMESTAMPS: mode={mode}")
        words = words_result["words"] if words_result else []
        for i, value in enumerate(db):
            rms_csv.append({
                "seed": seed, "mode": mode,
                "window_start_s": round(i * .02, 5),
                "window_end_s": round((i + 1) * .02, 5),
                "rms_dbfs": float(value),
            })
        for idx, w in enumerate(words):
            word_csv.append({
                "seed": seed, "mode": mode, "word_index": idx,
                **w,
            })
        sweep = {}
        for threshold in thresholds:
            stats = quiet_regions(
                db, threshold_dbfs=threshold,
                min_pause_ms=args.min_pause_ms,
            )
            for r in stats["regions"]:
                r["whisper_context"] = pause_context(r, words)
                regions_csv.append({
                    "seed": seed, "mode": mode,
                    "threshold_dbfs": threshold,
                    **{k: r[k] for k in (
                        "start_s", "end_s", "duration_s"
                    )},
                    "context": r["whisper_context"]["label"],
                    "prev_word": r["whisper_context"].get("prev_word", ""),
                    "next_word": r["whisper_context"].get("next_word", ""),
                })
            sweep[str(threshold)] = stats
        baseline[mode] = {
            "wav_path": item["path"], "wav_sha256": item["sha256"],
            "wav_duration_s": item["duration_s"],
            "rms_trailing_unprocessed_s": tail,
        }
        if words_result:
            baseline[mode]["asr"] = {
                **words_result,
                "matches_p12_earlier_transcript_normalized": (
                    text_tokens(words_result["transcript"]) ==
                    text_tokens(old_transcripts[mode])
                    if mode in old_transcripts else None
                ),
            }
        else:
            baseline[mode]["asr"] = None
        baseline[mode]["threshold_sweep"] = sweep
        measured[mode] = {
            **baseline[mode],
            "_original": item,
        }

    k2, hybrid = measured["k2"], measured["hybrid"]
    comparisons = []
    for threshold in thresholds:
        key = str(threshold)
        a, b = k2["threshold_sweep"][key], hybrid["threshold_sweep"][key]
        duration_delta = (
            hybrid["wav_duration_s"] - k2["wav_duration_s"]
        )
        quiet_delta = b["quiet_total_s"] - a["quiet_total_s"]
        comparisons.append({
            "threshold_dbfs": threshold,
            "duration_delta_s": round(duration_delta, 5),
            "total_below_threshold_delta_s": round(quiet_delta, 5),
            "total_above_threshold_delta_s": round(
                duration_delta - quiet_delta, 5
            ),
            "extra_time_fraction_below_threshold_if_positive_delta": (
                quiet_delta / duration_delta if duration_delta > 0 else None
            ),
            "long_quiet_delta_s": round(
                b["long_quiet_total_s"] - a["long_quiet_total_s"], 5
            ),
            "long_quiet_count_delta": (
                b["long_quiet_count"] - a["long_quiet_count"]
            ),
            "k2_long_regions": a["long_quiet_count"],
            "hybrid_long_regions": b["long_quiet_count"],
        })

    word_comparison = None
    if model is not None:
        word_comparison = compare_word_timelines(
            {
                "words": k2["asr"]["words"],
                "duration_s": k2["wav_duration_s"],
            },
            {
                "words": hybrid["asr"]["words"],
                "duration_s": hybrid["wav_duration_s"],
            },
        )
    status = (
        "P13_RMS_SWEEP_AND_WHISPER_WORD_TIMESTAMPS_MEASURED_NOT_CAUSAL_PROOF"
        if model is not None
        else "P13_RMS_SWEEP_ONLY_WORD_TIMESTAMPS_NOT_MEASURED"
    )
    if word_comparison is not None and word_comparison["status"].startswith(
        "P13_ASR_WORD_SEQUENCE_MISMATCH"
    ):
        status = "P13_RMS_MEASURED_ASR_WORD_SEQUENCE_MISMATCH_NO_WORD_ATTRIBUTION"
    report = {
        "schema": SCHEMA, "status": status,
        "source_p11b_report": str(report_path),
        "p11b_source_sha": cohort["source_sha"],
        "seed": seed, "modes": list(MODES),
        "uses_only_original_first_visit_per_mode": True,
        "two_repeated_seed_visits_are_not_independent": True,
        "asr_model": args.asr_model if model is not None else "NOT_MEASURED",
        "asr_device": args.asr_device if model is not None else "NOT_MEASURED",
        "asr_input": args.asr_input if model is not None else "NOT_MEASURED",
        "rms_thresholds_dbfs": thresholds,
        "min_pause_ms_requested": args.min_pause_ms,
        "modes_data": {
            mode: {
                k: v for k, v in measured[mode].items()
                if k != "_original"
            }
            for mode in MODES
        },
        "k2_hybrid_threshold_comparison": comparisons,
        "k2_hybrid_whisper_word_timing_comparison": word_comparison,
        "clip_manifest": [],
        "limits": [
            "RMS low-amplitude windows are not verified speech silence/VAD",
            "Minimum 150ms threshold is quantized to 160ms on 20ms windows",
            "Whisper word timestamps use cross-attention and DTW estimates, "
            "not a separate forced aligner or ground truth",
            "No new TTS generation; original WAV SHA256 and duration validated",
            "One selected seed is NOT a statistically independent quality cohort",
            "No speaker similarity, native seeded RNG parity, or GPU speedup",
        ],
    }
    # Create output only after all source and ASR validation so dependency
    # errors do not leave empty output directories that block simple reruns.
    output.mkdir(parents=True, exist_ok=False)
    csv_file(
        output / "rms_20ms.csv", rms_csv,
        ["seed", "mode", "window_start_s", "window_end_s", "rms_dbfs"],
    )
    csv_file(
        output / "long_quiet_regions.csv", regions_csv,
        [
            "seed", "mode", "threshold_dbfs", "start_s", "end_s",
            "duration_s", "context", "prev_word", "next_word",
        ],
    )
    csv_file(
        output / "whisper_words.csv", word_csv,
        [
            "seed", "mode", "word_index", "text", "normalized",
            "start_s", "end_s", "probability",
        ],
    )
    clip_thr = -45.0 if -45.0 in thresholds else thresholds[0]
    for mode in MODES:
        original = measured[mode]["_original"]
        ranked = sorted(
            baseline[mode]["threshold_sweep"][str(clip_thr)]["regions"],
            key=lambda x: x["duration_s"], reverse=True
        )[: args.clip_count]
        for idx, region in enumerate(ranked, start=1):
            clip_path = output / "clips" / (
                f"seed{seed}_{mode}_quiet{idx:02d}_"
                f"from{region['start_s']:.2f}_to{region['end_s']:.2f}.wav"
            )
            clip = write_pcm_clip(
                pcm=original["raw_pcm_bytes"],
                total_frames=original["frames"],
                start_s=region["start_s"],
                end_s=region["end_s"],
                out=clip_path,
            )
            report["clip_manifest"].append({
                "mode": mode, "threshold_dbfs": clip_thr,
                "quiet_interval": region, **clip,
                "listen_instruction": (
                    "Contextual excerpt from original WAV; direct "
                    "same-absolute-time comparison across modes does "
                    "not align the same words."
                ),
            })
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.md").write_text(render_summary(report))
    print(f"P13_STATUS={status} REPORT={output/'summary.md'}", flush=True)
    return report


def render_summary(r: dict) -> str:
    lines = [
        "# P13 — Offline acoustic pause / word timeline diagnosis",
        "", f"**{r['status']}**", "",
        f"P11B source SHA: {r['p11b_source_sha']}",
        f"Seed: {r['seed']} (first visit per mode only)",
        f"Word timestamps: {r['asr_model']}",
        "",
        "## K2 vs Hybrid RMS threshold sensitivity",
        "",
        "| Threshold dBFS | Extra WAV s | Extra below threshold s | Extra above threshold s | Extra long-quiet s | Hybrid long regions |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r["k2_hybrid_threshold_comparison"]:
        lines.append(
            f"| {row['threshold_dbfs']:.0f} | "
            f"{row['duration_delta_s']:+.2f} | "
            f"{row['total_below_threshold_delta_s']:+.2f} | "
            f"{row['total_above_threshold_delta_s']:+.2f} | "
            f"{row['long_quiet_delta_s']:+.2f} | "
            f"{row['hybrid_long_regions']} |"
        )
    lines += [
        "", "## Long quiet regions at -45 dBFS (or first supplied threshold)",
        "",
        "| Mode | Start s | End s | Quiet seconds | Estimated word context |",
        "|---|---:|---:|---:|---|",
    ]
    thr = -45.0 if -45.0 in r["rms_thresholds_dbfs"] else r["rms_thresholds_dbfs"][0]
    for mode in MODES:
        regs = r["modes_data"][mode]["threshold_sweep"][str(thr)]["regions"]
        for row in sorted(regs, key=lambda x: x["duration_s"], reverse=True)[:8]:
            ctxt = row["whisper_context"]
            desc = ctxt["label"]
            if ctxt.get("prev_word") or ctxt.get("next_word"):
                desc += (
                    f" ({ctxt.get('prev_word') or '?'} / "
                    f"{ctxt.get('next_word') or '?'})"
                )
            lines.append(
                f"| {mode} | {row['start_s']:.2f} | "
                f"{row['end_s']:.2f} | {row['duration_s']:.2f} | {desc} |"
            )
    comparison = r["k2_hybrid_whisper_word_timing_comparison"]
    if comparison is None:
        lines += ["", "## Word timestamps", "",
                  "**NOT_MEASURED** — add real faster-whisper ASR to this offline run."]
    elif comparison["status"].startswith("P13_ASR_WORD_SEQUENCE_MISMATCH"):
        lines += [
            "", "## Word-level alignment", "",
            "**NO_POSITIONAL_ATTRIBUTION** — ASR token sequences differ; "
            "do not align words by index.",
            "", f"K2 tokens: {comparison['k2_tokens']}",
            f"Hybrid tokens: {comparison['hybrid_tokens']}",
        ]
    else:
        parts = comparison["decomposition_delta_s"]
        lines += [
            "", "## Whisper-estimated timing decomposition (not verified speech duration)",
            "",
            f"Words aligned: {comparison['word_count']}; "
            f"extra WAV: {comparison['duration_delta_s']:+.3f}s",
            "",
            "| Component | Hybrid minus K2 seconds |",
            "|---|---:|",
        ]
        for key, value in parts.items():
            lines.append(f"| {key} | {value:+.3f} |")
        lines += [
            "", "### Largest estimated inter-word gap changes", "",
            "| Between words | K2 gap s | Hybrid gap s | Extra s |",
            "|---|---:|---:|---:|",
        ]
        for change in comparison["top_positive_gap_deltas"][:8]:
            lines.append(
                f"| {change['between']} | "
                f"{change['k2_signed_gap_s']:.2f} | "
                f"{change['hybrid_signed_gap_s']:.2f} | "
                f"{change['gap_delta_s']:+.2f} |"
            )
        lines += [
            "", "### Largest estimated word span changes", "",
            "| Word | K2 span s | Hybrid span s | Extra s |",
            "|---|---:|---:|---:|",
        ]
        for change in comparison["top_positive_word_span_deltas"][:8]:
            lines.append(
                f"| {change['word']} | "
                f"{change['k2_span_s']:.2f} | "
                f"{change['hybrid_span_s']:.2f} | "
                f"{change['span_delta_s']:+.2f} |"
            )
    lines += [
        "", "## Exported evidence", "",
        "- report.json — full source-locked measurements and estimated word alignment",
        "- rms_20ms.csv — complete raw 20ms RMS profiles per mode",
        "- long_quiet_regions.csv — long low-RMS intervals at every threshold",
        "- whisper_words.csv — ASR word-time estimates (empty if ASR disabled)",
        "- clips/*.wav — selected original-PCM quiet excerpts with context; "
        "the original WAV files are unchanged",
        "", "## Limits", "",
    ]
    lines += ["- " + x for x in r["limits"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p11b-report", type=Path, required=True)
    p.add_argument("--p12-asr-report", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--thresholds", default="-35,-40,-45,-50")
    p.add_argument("--min-pause-ms", type=int, default=150)
    p.add_argument("--clip-count", type=int, default=5)
    p.add_argument("--asr", choices=("none", "faster-whisper"),
                   default="faster-whisper")
    p.add_argument("--asr-model", default="small.en")
    p.add_argument("--asr-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--asr-input", choices=("file", "pcm-array"), default="file")
    args = p.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

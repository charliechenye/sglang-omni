# SPDX-License-Identifier: Apache-2.0
"""P12B: evidence-preserving quality review of existing P11B native/K2/hybrid WAVs.

Reads the already generated P11B C1 WAVs; never synthesizes new speech.
Validates the original sha256 and WAV format. Reports silence duration,
trailing silence, frame RMS, repeatability and utterance duration.
Optional faster-whisper transcript WER and repeated phrases if installed.
Optional EXTERNALLY measured speaker embeddings can be passed via JSON;
the script does not pretend waveform correlation is speaker similarity.

Missing ASR or speaker embeddings is NOT a pass: mark those quality gates
UNMEASURED. Every original WAV is immutable, outputs use a NEW directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import wave
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import TEXT

SCHEMA = "qwen3tts-cpspec-p12b-quality-review-v1"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (word != other),
            ))
        previous = current
    return previous[-1]


def transcript_features(transcript: str, reference: str) -> dict:
    words = tokenize(transcript)
    correct = tokenize(reference)
    if not correct:
        raise ValueError("P12B reference text must not be empty")
    trigrams = Counter(
        tuple(words[i:i+3]) for i in range(max(0, len(words) - 2))
    )
    repeat3 = [
        {"text": " ".join(k), "count": n}
        for k, n in trigrams.items() if n >= 2
    ]
    return {
        "transcript": transcript,
        "word_count": len(words),
        "wer": edit_distance(correct, words) / len(correct),
        "repeated_trigrams": sorted(
            repeat3, key=lambda item:(-item["count"], item["text"])
        ),
    }


def waveform_features(path: Path, expected_sha: str, *, threshold_dbfs=-45.) -> dict:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha:
        raise ValueError(f"WAV SHA mismatch for {path}: {sha} != {expected_sha}")
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getframerate() != 24000
            or handle.getsampwidth() != 2
            or handle.getcomptype() != "NONE"
        ):
            raise ValueError(f"P12B unexpected PCM WAV format: {path}")
        frames = handle.getnframes()
        samples = np.frombuffer(
            handle.readframes(frames), dtype="<i2"
        ).astype(np.float32) / 32768.
    block = 480  # 20ms at 24k
    trim = len(samples) // block * block
    if not trim:
        raise ValueError(f"empty/short audio: {path}")
    rms = np.sqrt(np.mean(samples[:trim].reshape(-1, block) ** 2, axis=1))
    silence = 20 * np.log10(np.maximum(rms, 1.0e-12)) < threshold_dbfs
    end = len(silence)
    while end > 0 and silence[end-1]:
        end -= 1
    duration = len(samples) / 24000.
    return {
        "path": str(path),
        "sha256_verified": True,
        "duration_s": duration,
        "duration_80ms_codec_frames_approx": duration / .08,
        "silence_threshold_dbfs": threshold_dbfs,
        "silence_fraction_20ms": float(silence.mean()),
        "trailing_silence_s": float((len(silence) - end) * .02),
        "mean_rms": float(np.sqrt(np.mean(samples**2))),
        "peak_abs": float(np.max(np.abs(samples))),
        "sample_count": len(samples),
        "sample_rate": 24000,
        "note": "RMS silence is not transcript quality or speaker identity",
    }


def cosine(x: list[float], y: list[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if (
        a.ndim != 1 or b.ndim != 1 or a.shape != b.shape
        or a.size < 8 or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
        or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0
    ):
        raise ValueError("invalid source-attributed speaker embedding")
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def asr_engine(name: str, model_path: str, device: str):
    if name == "none":
        return None
    if name != "faster-whisper":
        raise ValueError(f"unsupported ASR engine {name!r}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "P12B ASR not installed: use --asr none or install faster-whisper in venv"
        ) from exc
    return WhisperModel(model_path, device=device, compute_type="float16" if device=="cuda" else "int8")


def analyze(args) -> dict:
    source = args.p11b_report.expanduser().resolve(strict=True)
    report = json.loads(source.read_text())
    folder = source.parent
    if not report.get("status", "").startswith(
        "P11B_REAL_ON_POLICY_HYBRID_AR_K2_ALL_STOP_ACCOUNTING_PASS"
    ):
        raise ValueError("P12B requires passed P11B natural-stop/trace report")
    sha = report.get("source_sha")
    if not isinstance(sha, str) or len(sha) != 40:
        raise ValueError("P11B report lacks source SHA")
    if len(report.get("seeds", [])) < 2:
        raise ValueError("P12B requires at least 2 distinct seeds")
    if len(report.get("visits", [])) != 6 * len(report["seeds"]):
        raise ValueError("P12B P11B visit count inconsistent")
    model = asr_engine(args.asr, args.asr_model, args.asr_device)
    speaker = None
    if args.speaker_embeddings:
        spath = args.speaker_embeddings.expanduser().resolve(strict=True)
        speaker = json.loads(spath.read_text())
        if not isinstance(speaker, dict) or not speaker.get("model_id"):
            raise ValueError("Speaker embeddings need externally recorded model_id and vectors mapping")
    output = {
        "schema": SCHEMA,
        "status": "P12B_WAV_STATS_PASS_ASR_OR_SPEAKER_UNMEASURED",
        "p11b_source_sha": sha,
        "input_p11b_report": str(source),
        "reference_text": args.reference_text,
        "asr_engine": args.asr if model else "NOT_MEASURED",
        "speaker_embeddings": speaker.get("model_id") if speaker else "NOT_MEASURED",
        "items": [],
        "seed_comparison": [],
    }
    for row in report["visits"]:
        path = folder / f"{row['tag']}.wav"
        features = waveform_features(
            path, row["wav_sha256"], threshold_dbfs=args.silence_threshold_dbfs
        )
        if abs(features["duration_s"] - row["duration_s"]) > .001:
            raise ValueError(f"P12B WAV metadata differs from saved report: {path}")
        entry = {
            "seed": row["seed"], "mode": row["mode"],
            "tag": row["tag"], **features,
        }
        if model is not None:
            # Explicitly measured: no fabricated ASR/WER when model absent.
            segments, info = model.transcribe(
                str(path), language="en",
                beam_size=5, vad_filter=False, condition_on_previous_text=False,
            )
            transcript = " ".join(x.text.strip() for x in segments)
            entry.update(transcript_features(transcript, args.reference_text))
        output["items"].append(entry)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in output["items"]:
        grouped[row["seed"]][row["mode"]].append(row)
    for seed, modes in grouped.items():
        if set(modes) != {"native", "k2", "hybrid"}:
            raise ValueError(f"P12B missing modes for seed {seed}")
        base = modes["native"][0]
        for mode in ("native", "k2", "hybrid"):
            visits = modes[mode]
            if len(visits) != 2:
                raise ValueError(f"P12B expected 2 repeated visits for {seed}/{mode}")
            # Reproducibility is not independent cohort sampling.
            same = visits[0]["sha256_verified"] and (
                visits[0]["duration_s"] == visits[1]["duration_s"]
                and visits[0]["silence_fraction_20ms"] == visits[1]["silence_fraction_20ms"]
            )
            output["seed_comparison"].append({
                "seed": seed, "mode": mode,
                "two_repeated_visits_not_independent": True,
                "same_audio_statistics": bool(same),
                "duration_s": visits[0]["duration_s"],
                "duration_delta_vs_native_s": (
                    visits[0]["duration_s"] - base["duration_s"]
                ),
                "duration_delta_vs_native_pct": (
                    visits[0]["duration_s"] / base["duration_s"] - 1
                ) * 100,
                "silence_fraction": visits[0]["silence_fraction_20ms"],
                "trailing_silence_s": visits[0]["trailing_silence_s"],
                "wer": visits[0].get("wer"),
                "repeat3": visits[0].get("repeated_trigrams"),
                "flag_duration_gt_20pct": (
                    visits[0]["duration_s"] > 1.2 * base["duration_s"]
                ),
            })
        if speaker:
            emb = speaker.get("vectors", {})
            reference = emb.get(modes["native"][0]["tag"])
            if reference is None:
                raise ValueError(f"Speaker embed missing native reference seed {seed}")
            for record in output["seed_comparison"]:
                if record["seed"] == seed:
                    other = emb.get(modes[record["mode"]][0]["tag"])
                    if other is None:
                        raise ValueError(f"Speaker embedding missing {record['mode']} for {seed}")
                    record["speaker_cosine_vs_native_external"] = cosine(reference, other)
    if model is not None:
        output["status"] = (
            "P12B_ASR_WER_AND_WAV_STATS_MEASURED_SPEAKER_EXTERNAL_OR_UNMEASURED_"
            "NOT_A_HYBRID_SELECTION"
        )
    if not speaker:
        output["speaker_similarity_status"] = "NOT_MEASURED"
    else:
        output["speaker_similarity_status"] = "EXTERNAL_EMBEDDINGS_UNVALIDATED_MODEL_PROVENANCE"
    output["limits"] = [
        "Two repeated visits per seed are not four independent samples",
        "Silence and duplicate audio segments alone cannot prove linguistic quality",
        "ASR/English WER is unmeasured unless faster-whisper explicitly enabled",
        "Speaker similarity is unmeasured without external embeddings",
        "All durations are output AUDIO seconds, not HTTP latency/QPS",
        "No native seeded sampler parity or lossless distribution claim",
    ]
    return output


def write(out: Path, report: dict) -> Path:
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# P12B real Hybrid AR/K2 audio quality audit",
        "", f"**{report['status']}**", "",
        f"P11B source SHA: {report['p11b_source_sha']}",
        f"ASR: {report['asr_engine']}",
        f"Speaker similarity: {report['speaker_similarity_status']}",
        "",
        "| Seed | Mode | WAV seconds | vs Native | 20ms silence fraction | Trailing silence | ASR WER | >20% longer |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["seed_comparison"]:
        lines.append(
            f"| {row['seed']} | {row['mode']} | {row['duration_s']:.3f} | "
            f"{row['duration_delta_vs_native_pct']:+.1f}% | "
            f"{row['silence_fraction']:.1%} | {row['trailing_silence_s']:.2f}s | "
            f"{(f'{row['wer']:.2%}' if row['wer'] is not None else 'NOT_MEASURED')} | "
            f"{row['flag_duration_gt_20pct']} |"
        )
    lines.extend(["", "## Limits", ""] + [
        "- " + x for x in report["limits"]
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return out / "summary.md"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p11b-report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-text", default=TEXT)
    p.add_argument("--asr", choices=("none","faster-whisper"), default="none")
    p.add_argument("--asr-model", default="small.en")
    p.add_argument("--asr-device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--speaker-embeddings", type=Path, default=None)
    p.add_argument("--silence-threshold-dbfs", type=float, default=-45.)
    args = p.parse_args()
    if not -80 <= args.silence_threshold_dbfs <= -15:
        p.error("silence threshold must be -80..-15 dBFS")
    report = analyze(args)
    file = write(args.out, report)
    print(f"P12B_QUALITY_STATUS={report['status']} REPORT={file}")


if __name__ == "__main__":
    main()

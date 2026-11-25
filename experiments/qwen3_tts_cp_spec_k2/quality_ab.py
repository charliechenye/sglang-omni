# SPDX-License-Identifier: Apache-2.0
"""P3 multi-prompt C1 quality ABBA and optional *separate* ASR scoring.

Generate phase: a single normal Omni TTS server, ARM only for K2 calls.
The authoritative finish_reason comes from the existing one-item batch
speech API. This is NOT a throughput benchmark, proof of quality, or speaker
similarity measurement.

Score phase: use a separate, already running OpenAI-compatible ASR endpoint
(e.g. Omni Qwen3-ASR on another port, or after the TTS server stops). Do not
mix ASR model latency with TTS latency. WER/CER is text intelligibility only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .quality_smoke import (
    MODEL,
    PHASES,
    _http_call,
    parse_k2_log_entries,
    wav_info,
)

DEFAULT_CORPUS = (
    {
        "id": "en_std_1", "language": "en", "voice": "Vivian",
        "seed": 20261120,
        "text": "Our team shipped a new feature in November. We will monitor its quality next week.",
    },
    {
        "id": "en_std_2", "language": "en", "voice": "Vivian",
        "seed": 20261121,
        "text": "The conference starts at nine thirty in the morning, and the second session begins at eleven.",
    },
    {
        "id": "zh_std_1", "language": "zh", "voice": "Vivian",
        "seed": 20261122,
        "text": "今天我们正在测试语音生成系统。希望它能准确读出这段中文，并保持自然的语调。",
    },
    {
        "id": "zh_std_2", "language": "zh", "voice": "Vivian",
        "seed": 20261123,
        "text": "请把这句话完整地读出来。这个实验主要比较生成质量、停顿和语音的清晰度。",
    },
)


def load_corpus(path: Path | None) -> list[dict[str, Any]]:
    source = (
        list(DEFAULT_CORPUS)
        if path is None
        else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    )
    if not source:
        raise ValueError("empty P3 text corpus")
    ids = set()
    for case in source:
        if not isinstance(case, dict):
            raise ValueError("corpus row must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", identifier) is None:
            raise ValueError("case id must be simple alphanumeric/underscore/hyphen")
        if identifier in ids:
            raise ValueError(f"duplicate case: {identifier}")
        ids.add(identifier)
        if case.get("language") not in ("en", "zh"):
            raise ValueError("only en / zh are supported by this reference scorer")
        if not isinstance(case.get("text"), str) or not case["text"].strip():
            raise ValueError("nonempty text required")
        if not isinstance(case.get("voice"), str) or not case["voice"]:
            raise ValueError("voice must be specified")
        if type(case.get("seed")) is not int or case["seed"] < 0:
            raise ValueError("nonnegative deterministic seed required")
    return source


def normalize_tokens(text: str, language: str) -> list[str]:
    """Reference WER/CER normalization; transcript scoring, not audio quality."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    if language == "zh":
        # Chinese CER over Han + Latin/digits; do not insert fake word tokens.
        return [c for c in text if c.isalnum()]
    if language == "en":
        # Normalize Unicode punctuation into space and split spoken word units.
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
    raise ValueError(f"unsupported scoring language: {language}")


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, token in enumerate(reference, start=1):
        current = [row]
        for col, h in enumerate(hypothesis, start=1):
            current.append(min(
                previous[col] + 1,
                current[col - 1] + 1,
                previous[col - 1] + (token != h),
            ))
        previous = current
    return previous[-1]


def score_transcript(reference: str, hypothesis: str, lang: str) -> dict[str, Any]:
    truth = normalize_tokens(reference, lang)
    recognized = normalize_tokens(hypothesis, lang)
    if not truth:
        raise ValueError("empty normalized reference")
    distance = edit_distance(truth, recognized)
    return {
        "metric": "CER" if lang == "zh" else "WER",
        "errors": distance,
        "reference_units": len(truth),
        "error_rate": distance / len(truth),
        "recognized_text": hypothesis,
    }


def generate(args: argparse.Namespace) -> dict:
    root = args.out.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"output already exists: {root}")
    arm = args.arm_file.expanduser().resolve()
    log = args.server_log.expanduser().resolve()
    if arm.exists():
        raise RuntimeError("stale ARM marker before experiment")
    if not arm.parent.is_dir() or not log.is_file():
        raise FileNotFoundError("ARM parent and exact server log must exist")
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as health:
        if health.status != 200:
            raise RuntimeError("TTS server not healthy")
    corpus = load_corpus(args.corpus)
    root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "P3_INCOMPLETE_NO_QUALITY_CLAIM",
        "model": args.model,
        "corpus": corpus,
        "visits": [],
        "max_new_tokens": args.max_new_tokens,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "notes": [
            "Same server and C1 sequential requests; no production throughput claim",
            "WAV or audio length equality is not a quality or distribution parity proof",
            "WER/CER can only be scored after external ASR transcriptions",
            "Speaker similarity, MOS and audio naturalness are not measured",
        ],
    }
    try:
        for case in corpus:
            for idx, phase in enumerate(PHASES):
                tag = f"{case['id']}_{idx:02d}_{phase}"
                print(f"P3_TTS_BEGIN={tag}", flush=True)
                payload = {
                    "model": args.model,
                    "input": case["text"],
                    "voice": case["voice"],
                    "response_format": "wav",
                    "seed": case["seed"],
                    "max_new_tokens": args.max_new_tokens,
                    "stage_params": {
                        "tts_engine": {
                            "subtalker_dosample": True,
                            "subtalker_temperature": args.temperature,
                            "subtalker_top_k": args.top_k,
                            "subtalker_top_p": args.top_p,
                        }
                    },
                }
                prior = len(parse_k2_log_entries(log.read_text(errors="replace")))
                if arm.exists():
                    raise RuntimeError(f"stale ARM marker before {tag}")
                try:
                    if phase == "spec":
                        arm.touch(exist_ok=False)
                    audio, finish, seconds = _http_call(args.base_url, payload)
                    stats = wav_info(audio)
                finally:
                    arm.unlink(missing_ok=True)
                after = len(parse_k2_log_entries(log.read_text(errors="replace")))
                active = after - prior
                result = {
                    "tag": tag, "id": case["id"], "phase": phase,
                    "language": case["language"],
                    "seed": case["seed"],
                    "input_text": case["text"],
                    "finish_reason": finish,
                    "new_k2_frames": active,
                    "debug_seconds_not_perf": round(seconds, 6),
                    "audio_path": str((root / (tag + ".wav")).resolve()),
                    "sha256": hashlib.sha256(audio).hexdigest(),
                    **stats,
                }
                (root / (tag + ".wav")).write_bytes(audio)
                report["visits"].append(result)
                # Persist partial evidence after *every* successful visit.
                (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"P3_TTS_DONE={tag} finish={finish} duration={stats['duration_s']:.3f}s "
                    f"k2_frames={active}", flush=True
                )
                if finish != "stop" or (phase == "spec" and active <= 0) or (
                    phase == "base" and active != 0
                ):
                    raise RuntimeError(f"P3 completeness / K2 routing FAIL at {tag}")
    except Exception as exc:
        report["status"] = "P3_GENERATION_ERROR_DO_NOT_USE_FOR_QUALITY_SELECTION"
        report["error"] = str(exc)
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["status"] = "P3_ALL_NATURAL_STOP_ASR_NOT_YET_SCORED_NO_QUALITY_CLAIM"
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def transcribe(asr_base_url: str, model: str, wav: bytes, lang: str) -> str:
    """OpenAI-compatible /v1/audio/transcriptions with a WAV file upload."""
    boundary = "----cp-spec-quality-" + uuid.uuid4().hex
    def field(name: str, data: bytes, *, filename: str | None = None) -> bytes:
        prefix = f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\""
        if filename:
            prefix += f"; filename=\"{filename}\"\r\nContent-Type: audio/wav"
        return (prefix + "\r\n\r\n").encode() + data + b"\r\n"
    body = (
        field("model", model.encode())
        + field("language", lang.encode())
        + field("file", wav, filename="sample.wav")
        + f"--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        asr_base_url.rstrip("/") + "/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ASR HTTP {exc.code}: {exc.read()[:800]!r}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        raise ValueError("ASR endpoint must return JSON with text; cannot score")
    return raw["text"]


def score(args: argparse.Namespace) -> dict:
    report_path = args.report.expanduser().resolve(strict=True)
    root = report_path.parent
    report = json.loads(report_path.read_text())
    if report.get("status") not in (
        "P3_ALL_NATURAL_STOP_ASR_NOT_YET_SCORED_NO_QUALITY_CLAIM",
        "P3_ALL_NATURAL_STOP_ASR_SCORED_WER_CER_ONLY",
    ):
        raise RuntimeError("generation has not passed natural-stop gate")
    results: list[dict] = []
    try:
        for row in report["visits"]:
            audio_path = Path(row["audio_path"]).resolve(strict=True)
            if audio_path.parent != root:
                raise ValueError("refusing WAV outside this P3 run directory")
            recognized = transcribe(
                args.asr_base_url, args.asr_model, audio_path.read_bytes(),
                row["language"],
            )
            result = score_transcript(
                row["input_text"], recognized, row["language"]
            )
            results.append({"tag": row["tag"], "phase": row["phase"],
                            "id": row["id"], "language": row["language"], **result})
            print(f"P3_ASR_DONE={row['tag']} {result['metric']}={result['error_rate']:.3f}", flush=True)
    except Exception as exc:
        failure = {"status": "P3_ASR_FAILED_NO_QUALITY_SELECTION",
                   "exception": str(exc), "partial_results": results}
        (root / "asr_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    by_lang_phase: dict[tuple[str, str], dict[str, int]] = {}
    for r in results:
        key = (r["language"], r["phase"])
        agg = by_lang_phase.setdefault(key, {"errors": 0, "reference_units": 0, "samples": 0})
        agg["errors"] += r["errors"]
        agg["reference_units"] += r["reference_units"]
        agg["samples"] += 1
    summary = [
        {"language": lang, "phase": phase, **counts,
         "error_rate": counts["errors"] / counts["reference_units"]}
        for (lang, phase), counts in sorted(by_lang_phase.items())
    ]
    output = {
        "status": "P3_ALL_NATURAL_STOP_ASR_SCORED_WER_CER_ONLY",
        "asr_endpoint": args.asr_base_url,
        "asr_model": args.asr_model,
        "rows": results,
        "summary": summary,
        "limitations": [
            "WER/CER is intelligibility only, not speaker similarity or naturalness.",
            "A TTS transcript compared to input text is not a speech reference recording.",
            "One reference text per seed; outputs within arms can be identical.",
            "No independent MOS, objective speaker verification or distribution equivalence.",
        ],
    }
    dest = root / "asr_quality.json"
    if dest.exists():
        raise FileExistsError("existing ASR output must not be overwritten")
    dest.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    g = subs.add_parser("generate", help="TTS ABBA across multiple prompts; no ASR")
    g.add_argument("--base-url", default="http://localhost:8000")
    g.add_argument("--arm-file", required=True, type=Path)
    g.add_argument("--server-log", required=True, type=Path)
    g.add_argument("--out", required=True, type=Path)
    g.add_argument("--corpus", type=Path, default=None)
    g.add_argument("--model", default=MODEL)
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=1.0)
    a = subs.add_parser("score", help="after generation: use independent ASR API")
    a.add_argument("--report", required=True, type=Path)
    a.add_argument("--asr-base-url", required=True)
    a.add_argument("--asr-model", required=True)
    args = p.parse_args()
    if args.action == "generate":
        if args.max_new_tokens < 256 or args.max_new_tokens > 2048:
            p.error("full-stop test requires max_new_tokens 256..2048")
        result = generate(args)
        print(f"{result['status']}: {args.out / 'report.json'}")
    else:
        result = score(args)
        print(f"{result['status']}: {args.report.parent / 'asr_quality.json'}")


if __name__ == "__main__":
    main()

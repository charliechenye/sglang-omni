# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS CP K2 C1 complete-output acceptance gate, not a performance test.

Unlike the historical /v1/audio/speech WAV-only smoke, this sends one item at
a time through /v1/audio/speech/batch. That endpoint calls the *same*
client.speech path and exposes the official finish_reason ("stop"/"length")
in each result. This prevents a successful HTTP 200 from hiding output
truncation at max_new_tokens. Each response still contains one C1 audio sample.

The default max_new_tokens=512 is a ceiling, not a requested duration.
A terminal length stop is always a failed completeness gate, even if the
output is a plausible WAV. Do not use resulting wall time as a QPS estimate.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

STATUS_PASS = "QTTS_CP_K2_C1_ALL_STOP_WITH_CANDIDATE_ACTIVITY_NOT_PARITY_OR_SPEED"
STATUS_INCOMPLETE = "QTTS_CP_K2_C1_LENGTH_OR_UNKNOWN_FINISH_NO_QUALITY_SELECTION"
STATUS_NO_ACTIVITY = "QTTS_CP_K2_C1_NO_CANDIDATE_EXECUTION_PROOF"
STATUS_FAILURE = "QTTS_CP_K2_C1_HTTP_OR_AUDIO_FAILURE"

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TEXT = (
    "We are testing a CodePredictor speculative decoder on a real sampled "
    "text to speech request, including a full natural stopping condition."
)
PHASES = ("base", "spec", "spec", "base")
K2_LOG_RE = re.compile(
    r"QTTS_CP_SPEC_K2 frame=(\d+) rounds=(\d+) proposed=(\d+) "
    r"accepted=(\d+) bonus=(\d+) rejected=(\d+) committed_residual=(\d+)"
)


def decode_single_batch_response(response_json: dict[str, Any]) -> tuple[bytes, str]:
    """Only accept one real successful C1 response and a typed finish reason."""
    if (
        not isinstance(response_json, dict)
        or response_json.get("total") != 1
        or response_json.get("succeeded") != 1
        or response_json.get("failed") != 0
        or not isinstance(response_json.get("results"), list)
        or len(response_json["results"]) != 1
    ):
        raise ValueError(f"batch request failed or returned wrong size: {str(response_json)[:700]}")
    item = response_json["results"][0]
    if not isinstance(item, dict) or item.get("index") != 0 or item.get("status") != "success":
        raise ValueError(f"invalid C1 item result: {str(item)[:700]}")
    finish = item.get("finish_reason")
    if not isinstance(finish, str) or not finish:
        raise ValueError("missing typed finish_reason; do not infer stop from HTTP 200")
    if item.get("format") != "wav":
        raise ValueError(f"expected wav, got {item.get('format')!r}")
    data = item.get("audio_data")
    if not isinstance(data, str) or not data:
        raise ValueError("missing base64 audio_data")
    try:
        audio = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ValueError("invalid audio_data base64") from exc
    return audio, finish.lower()


def wav_info(audio: bytes) -> dict[str, Any]:
    """Validate actual WAV metadata, not inferred sample rate from byte count."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as w:
            channels = w.getnchannels()
            rate = w.getframerate()
            width = w.getsampwidth()
            frames = w.getnframes()
            compression = w.getcomptype()
    except Exception as exc:
        raise ValueError("invalid/truncated WAV response") from exc
    if channels != 1 or rate != 24000 or width != 2 or compression != "NONE" or frames < 1:
        raise ValueError(
            f"unexpected Qwen3-TTS format: channels={channels}, "
            f"rate={rate}, sample_width={width}, frames={frames}, compression={compression}"
        )
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "pcm_samples": frames,
        "duration_s": frames / rate,
        "approx_80ms_codec_frames": frames / 1920,
        "wav_bytes": len(audio),
    }


def parse_k2_log_entries(s: str) -> list[dict[str, int]]:
    """Count only K2 frames and reject malformed impossible counter values."""
    parsed = []
    for m in K2_LOG_RE.finditer(s):
        frame, rounds, proposed, accepted, bonus, rejected, committed = map(int, m.groups())
        if (
            rounds < 1
            or proposed < 1
            or accepted < 0
            or rejected < 0
            or bonus < 0
            or committed != 15
            or accepted + rejected != proposed
            or accepted + bonus + rejected != committed - 1
        ):
            raise ValueError(
                "K2 frame counters inconsistent; cannot trust candidate activation"
            )
        parsed.append({
            "frame": frame,
            "rounds": rounds,
            "proposed": proposed,
            "accepted": accepted,
            "bonus": bonus,
            "rejected": rejected,
            "committed_residual": committed,
        })
    return parsed


def _http_call(base_url: str, payload: dict[str, Any]) -> tuple[bytes, str, float]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/audio/speech/batch",
        data=json.dumps({"items": [payload]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=360) as result:
            if result.status != 200:
                raise RuntimeError(f"batch HTTP {result.status}")
            response = json.loads(result.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"batch HTTP {exc.code}: {exc.read()[:1200]!r}"
        ) from exc
    wav, finish = decode_single_batch_response(response)
    return wav, finish, time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output exists, do not overwrite: {output}")
    arm = args.arm_file.expanduser().resolve()
    if arm.exists():
        raise RuntimeError(f"ARM must be absent before start: {arm}")
    if not arm.parent.is_dir():
        raise FileNotFoundError(f"ARM parent directory missing: {arm.parent}")
    server_log = args.server_log.expanduser().resolve() if args.server_log else None
    if server_log is not None and not server_log.is_file():
        raise FileNotFoundError(f"server log missing: {server_log}")
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError("server not healthy; do not ARM")
    output.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": "wav",
        "seed": args.seed,
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
    (output / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    result: dict[str, Any] = {
        "status": STATUS_FAILURE,
        "note": "C1 sampled audio completion + K2 activity, not serving performance",
        "visits": [],
        "request": payload,
    }
    any_error = False
    for idx, phase in enumerate(PHASES):
        tag = f"{idx:02d}_{phase}"
        if arm.exists():
            raise RuntimeError("ARM marker unexpectedly present before a visit")
        prior = (
            len(parse_k2_log_entries(server_log.read_text(errors="replace")))
            if server_log else None
        )
        print(f"QTTS_K2_QUALITY_BEGIN={tag}", flush=True)
        try:
            if phase == "spec":
                arm.touch(exist_ok=False)
            audio, finish, elapsed = _http_call(args.base_url, payload)
            info = wav_info(audio)
            (output / f"{tag}.wav").write_bytes(audio)
            after = (
                len(parse_k2_log_entries(server_log.read_text(errors="replace")))
                if server_log else None
            )
            new_k2 = after - prior if after is not None and prior is not None else None
            visit = {
                "tag": tag,
                "finish_reason": finish,
                "wav_sha256": hashlib.sha256(audio).hexdigest(),
                "wall_s_debug_not_perf": round(elapsed, 6),
                "new_k2_log_frames": new_k2,
                **info,
            }
            if new_k2 is not None and (
                (phase == "spec" and new_k2 <= 0)
                or (phase == "base" and new_k2 != 0)
            ):
                visit["activity_gate"] = "FAIL_ARM_OR_LOG_ROUTING"
            else:
                visit["activity_gate"] = (
                    "K2_LOG_OBSERVED" if phase == "spec" and new_k2 is not None
                    else "NOT_CHECKED" if new_k2 is None else "BASELINE_NO_K2"
                )
            result["visits"].append(visit)
            print(
                f"QTTS_K2_QUALITY_DONE={tag} finish={finish} "
                f"audio_s={info['duration_s']:.3f} k2_frames={new_k2}",
                flush=True,
            )
        except Exception as exc:
            any_error = True
            result["error"] = {"visit": tag, "detail": str(exc)}
            print(f"QTTS_K2_QUALITY_ERROR={tag} {exc}", flush=True)
            break
        finally:
            arm.unlink(missing_ok=True)
    complete = len(result["visits"]) == 4
    finished = complete and all(v["finish_reason"] == "stop" for v in result["visits"])
    any_activity_problem = any(
        v["activity_gate"] == "FAIL_ARM_OR_LOG_ROUTING" for v in result["visits"]
    )
    if not any_error:
        if not complete or not finished:
            result["status"] = STATUS_INCOMPLETE
        elif any_activity_problem:
            result["status"] = STATUS_NO_ACTIVITY
        else:
            result["status"] = STATUS_PASS
    result["complete_stop_gate"] = finished
    result["k2_activity_gate"] = (
        "CHECKED" if server_log is not None else "NOT_CHECKED"
    )
    result["limits"] = [
        "A 1-item batch request invokes the same underlying client.speech path, "
        "but is a separate HTTP endpoint from the previous WAV-only smoke.",
        "All visits share one live server; latency includes Python sync and "
        "is NOT a valid E2E perf ABBA.",
        "Stop means natural completed output, not parity or correct phonemes.",
        "B/C WAV hash inequality is expected under independent spec RNG.",
        "Prior A1 10.2% first-future-code p/q overlap does not predict full "
        "15-depth acceptance; use the K2 server frame counters.",
    ]
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm-file", type=Path, required=True)
    p.add_argument("--server-log", type=Path, default=None)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--voice", default="Vivian")
    p.add_argument("--text", default=TEXT)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    args = p.parse_args()
    if not (128 <= args.max_new_tokens <= 2048):
        p.error("max_new_tokens must be 128..2048 for this full-stop gate")
    if not (0 < args.temperature <= 5 and 0 < args.top_p <= 1):
        p.error("invalid sampled temperature/top-p")
    if args.top_k < -1:
        p.error("invalid top-k")
    report = run(args)
    print(json.dumps(report, indent=2))
    if report["status"] != STATUS_PASS:
        p.exit(
            status=2,
            message=f"\n{report['status']} — check report.json, not a pass\n",
        )


if __name__ == "__main__":
    main()

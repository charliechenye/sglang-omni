# SPDX-License-Identifier: Apache-2.0
"""P21 real serving C1 native-Graph vs integrated K2 A-B-B-A HTTP latency.

- SAME H200 worker/source/seed/prompt/config. A = normal native Graph-ON,
  B = real committed K2 output via post-health P21_ARM.
- Requires actual worker per-request candidate/Graph activity evidence.
- Requires natural stop, complete WAV, stable deterministic within-arm hashes.
- No P18/P19 cost probe, no GPU timing microbench, no production/quality GO.
- C1 HTTP E2E latency only; no C8/C16 QPS, TTFC, speaker score or ASR CER.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from urllib.request import urlopen

from .p18_run import PROMPTS
from .quality_smoke import MODEL, _http_call, wav_info

PHASES = ("A", "B", "B", "A")
STATUS_PASS = "P21_REAL_SERVING_C1_ABBA_SCREEN_NOT_QUALITY_OR_PRODUCTION_GO"
STATUS_FAIL = "P21_REAL_SERVING_ABBA_GATE_FAILED_NO_SPEEDUP_CLAIM"
RE = re.compile(
    r"P21_REAL_SERVING_REQUEST request_hash=[0-9a-f]{12} (?P<fields>.*)"
)


def _request_events(log: Path) -> list[dict[str, int]]:
    events = []
    for match in RE.finditer(log.read_text(errors="replace")):
        data = {}
        for token in match.group("fields").split():
            if "=" not in token:
                raise RuntimeError("P21_INVALID_REQUEST_ACTIVITY_LOG")
            key, value = token.split("=", 1)
            if not value.isdecimal() or key in data:
                raise RuntimeError("P21_INVALID_REQUEST_COUNTER")
            data[key] = int(value)
        if not all(k in data for k in (
            "candidate_frames", "native_graph_replays",
            "native_eager_calls", "predictor_calls",
            "k2_rounds", "accepted_drafts", "rejected_drafts",
        )):
            raise RuntimeError("P21_MISSING_ACTIVITY_METRICS")
        events.append(data)
    return events


def _http_with_proof(
    *, base_url: str, server_log: Path, phase: str,
    case: dict, max_tokens: int,
) -> dict:
    before = len(_request_events(server_log))
    payload = {
        "model": MODEL,
        "input": case["text"],
        "voice": "Vivian",
        "response_format": "wav",
        "seed": case["seed"],
        "max_new_tokens": max_tokens,
        "stage_params": {
            "tts_engine": {
                "subtalker_dosample": True,
                "subtalker_temperature": 0.9,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
            }
        },
    }
    wav, finish, elapsed = _http_call(base_url, payload)
    if finish != "stop":
        raise RuntimeError(f"P21_NONNATURAL_AUDIO_FINISH {phase} {finish}")
    audio = wav_info(wav)
    events = []
    for _ in range(50):
        events = _request_events(server_log)
        if len(events) >= before + 1:
            break
        time.sleep(0.1)
    if len(events) != before + 1:
        raise RuntimeError(
            f"P21_MISSING_OR_EXTRA_WORKER_ACTIVITY "
            f"before={before} after={len(events)}"
        )
    stats = events[-1]
    if stats["predictor_calls"] < 2:
        raise RuntimeError("P21_INSUFFICIENT_REAL_PREDICTOR_CALLS")
    if phase == "A":
        if (stats["candidate_frames"] != 0
                or stats["native_graph_replays"] < 1):
            raise RuntimeError("P21_A_NOT_NATIVE_GRAPH_ON")
    elif phase == "B":
        if (stats["candidate_frames"] < 1
                or stats["k2_rounds"] < 1
                or stats["native_graph_replays"] != 0):
            raise RuntimeError("P21_B_NOT_REAL_SPECULATIVE_SERVING")
        if stats["candidate_frames"] > stats["predictor_calls"]:
            raise RuntimeError("P21_SPEC_FRAME_COUNTER_INCONSISTENT")
        if stats["accepted_drafts"] + stats["rejected_drafts"] != stats["k2_rounds"]:
            raise RuntimeError("P21_SPEC_ACCEPTANCE_COUNTER_INCONSISTENT")
    else:
        raise RuntimeError("P21_UNKNOWN_ABBA_PHASE")
    if not (math.isfinite(elapsed) and elapsed > 0):
        raise RuntimeError("P21_INVALID_E2E_LATENCY")
    return {
        "case_id": case["id"], "seed": case["seed"],
        "phase": phase,
        "finish_reason": finish,
        "e2e_wall_s": elapsed,
        "audio_duration_s": audio["duration_s"],
        "http_rtf": elapsed / audio["duration_s"],
        "wav_sha256": hashlib.sha256(wav).hexdigest(),
        "wav_bytes": len(wav), "worker": stats,
    }


def _summarize(rows: list[dict]) -> dict:
    durations = [r["e2e_wall_s"] for r in rows]
    return {
        "count": len(rows),
        "e2e_wall_p50_s": statistics.median(durations),
        "e2e_wall_p95_s": sorted(durations)[
            math.ceil(0.95 * len(durations)) - 1
        ],
        "http_rtf_p50": statistics.median(
            r["http_rtf"] for r in rows
        ),
    }


def compute_report(rows: list[dict], corpus: list[dict]) -> dict:
    if len(rows) != 4 * len(corpus):
        raise RuntimeError("P21_ABBA_MISSING_ROWS")
    expected = [
        (visit, case["id"], case["seed"])
        for visit in range(4)
        for case in corpus
    ]
    actual = [
        (r["visit"], r["case_id"], r["seed"]) for r in rows
    ]
    if expected != actual:
        raise RuntimeError("P21_ABBA_CASE_ORDER_OR_SEED_DRIFT")
    if any(r["phase"] != PHASES[r["visit"]] for r in rows):
        raise RuntimeError("P21_ABBA_ARM_PHASE_DRIFT")
    paired = []
    for case in corpus:
        each = {
            visit: next(
                r for r in rows
                if r["case_id"] == case["id"] and r["visit"] == visit
            )
            for visit in range(4)
        }
        # Repeatability is an observation, not proof of phoneme parity.
        hash_a = each[0]["wav_sha256"] == each[3]["wav_sha256"]
        hash_b = each[1]["wav_sha256"] == each[2]["wav_sha256"]
        a = (each[0]["e2e_wall_s"] + each[3]["e2e_wall_s"]) / 2
        b = (each[1]["e2e_wall_s"] + each[2]["e2e_wall_s"]) / 2
        paired.append({
            "case_id": case["id"], "seed": case["seed"],
            "a_mean_s": a, "b_mean_s": b,
            "b_over_a": b / a,
            "a_hash_repeatable": hash_a,
            "b_hash_repeatable": hash_b,
            "a_audio_s": each[0]["audio_duration_s"],
            "b_audio_s": each[1]["audio_duration_s"],
        })
    if not all(
        p["a_hash_repeatable"] and p["b_hash_repeatable"]
        for p in paired
    ):
        raise RuntimeError(
            "P21_WITHIN_ARM_SEEDED_WAV_DRIFT_NO_SPEEDUP_CLAIM"
        )
    summaries = {
        phase: _summarize([r for r in rows if r["phase"] == phase])
        for phase in ("A", "B")
    }
    ratio = statistics.median(p["b_over_a"] for p in paired)
    return {
        "status": STATUS_PASS,
        "quality_certificate": "NONE_NOT_ASR_OR_SPEAKER_EQUIVALENCE",
        "scope": "REAL_HTTP_SERVING_C1_ONLY_NOT_C8_C16_QPS_OR_STREAMING_TTFC",
        "causal_policy": "EXPERIMENTAL_HOST_DRIVEN_K2_NOT_GRAPH_FULL_DECODER",
        "paired": paired, "overall": summaries,
        "median_case_b_over_a": ratio,
        "median_case_relative_latency_change_pct": 100 * (ratio - 1),
        "within_arm_wav_hash_repeatable": all(
            p["a_hash_repeatable"] and p["b_hash_repeatable"] for p in paired
        ),
        "candidate_speed_screen": (
            "B_FASTER_THAN_A_BY_5_PERCENT_NOT_PRODUCTION_GO"
            if ratio < 0.95 else "B_NO_MEANINGFUL_C1_LATENCY_WIN"
        ),
        "notes": [
            "A and B are real codec outputs on the same serving worker",
            "B uses private speculative RNG, not native seed bitwise parity",
            "No natural stop or activity gate may be skipped",
            "HTTP wall includes base64/WAV/vocoder/network; no GPU event claim",
            "No ASR CER/WER, speaker similarity, MOS or quality equivalence",
            "C1 and one ABBA block only; no independent production perf GO",
        ],
    }


def run(root: Path, base_url: str, mode: str, max_tokens: int) -> dict:
    root = root.expanduser().resolve(strict=True)
    sha = (root / "source.sha").read_text().strip()
    local = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    cfg = json.loads((root / "run.json").read_text())
    if (not sha or len(sha) != 40 or sha != local
            or cfg.get("source_sha") != sha
            or cfg.get("P18_P19") != "DISABLED"):
        raise RuntimeError("P21_ABBA_SOURCE_OR_PROBE_PROVENANCE_MISMATCH")
    if mode not in ("smoke", "main") or max_tokens < 512:
        raise ValueError("P21_ABBA_INVALID_MODE_OR_LENGTH_CAP")
    arm = root / "P21_ARM"
    server_log = root / "server.log"
    report_dir = root / "abba"
    if arm.exists() or report_dir.exists() or not server_log.exists():
        raise RuntimeError("P21_ABBA_REQUIRES_FRESH_WORKER_ARM_AND_REPORT")
    with urlopen(base_url.rstrip("/") + "/health", timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError("P21_ABBA_SERVER_NOT_HEALTHY")
    prompts = PROMPTS[:2] if mode == "smoke" else PROMPTS
    seeds = (20261123, 20261124, 20261125)
    corpus = [
        {"id": name, "text": text, "seed": seeds[i % len(seeds)]}
        for i, (name, text) in enumerate(prompts)
    ]
    report_dir.mkdir(mode=0o700)
    report = {
        "schema": "qwen3tts-p21-real-serving-e2e-c1-abba-v1",
        "status": STATUS_FAIL,
        "source_sha": sha, "mode": mode, "corpus": corpus,
        "visits": [], "max_new_tokens": max_tokens,
        "original_graph_baseline_expected": True,
    }
    try:
        for visit, phase in enumerate(PHASES):
            if phase == "B":
                if not arm.exists():
                    arm.touch(exist_ok=False)
            elif arm.exists():
                arm.unlink()
            if (phase == "A" and arm.exists()) or (
                phase == "B" and not arm.exists()
            ):
                raise RuntimeError("P21_ABBA_ARM_STATE_INVALID")
            # Same prewarm prompt, excluded from timing. The first B
            # prewarms cached K2 heads/backend and cannot bias B visit1.
            warmup = {
                "id": "_warmup", "seed": 20261124,
                "text": "A brief warmup of the real speech serving path."
            }
            warm = _http_with_proof(
                base_url=base_url, server_log=server_log,
                phase=phase, case=warmup, max_tokens=max_tokens,
            )
            print(
                f"P21_ABBA_WARMUP_PASS visit={visit} phase={phase} "
                f"latency_s={warm['e2e_wall_s']:.3f}", flush=True,
            )
            for case in corpus:
                row = _http_with_proof(
                    base_url=base_url, server_log=server_log,
                    phase=phase, case=case, max_tokens=max_tokens,
                )
                row["visit"] = visit
                report["visits"].append(row)
                (report_dir / "partial.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
                print(
                    f"P21_ABBA_ROW visit={visit} phase={phase} "
                    f"case={case['id']} "
                    f"e2e_s={row['e2e_wall_s']:.3f} "
                    f"wav_s={row['audio_duration_s']:.3f} "
                    f"graph={row['worker']['native_graph_replays']} "
                    f"k2={row['worker']['candidate_frames']}",
                    flush=True,
                )
        report.update(compute_report(report["visits"], corpus))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if arm.exists():
            arm.unlink()
        (report_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print("P21_ABBA_RESULT=" + report["status"], flush=True)
        print("P21_ABBA_REPORT=" + str(report_dir / "report.json"), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=("smoke", "main"), default="smoke")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()
    report = run(
        args.root, args.base_url, args.mode, args.max_new_tokens
    )
    print(
        "P21_C1_MEDIAN_CASE_B_OVER_A="
        f"{report['median_case_b_over_a']:.5f}", flush=True,
    )
    print("P21_C1_SPEED_SCREEN=" + report["candidate_speed_screen"], flush=True)


if __name__ == "__main__":
    main()

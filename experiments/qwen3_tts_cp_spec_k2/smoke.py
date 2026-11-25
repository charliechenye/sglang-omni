# SPDX-License-Identifier: Apache-2.0
"""C1 no-spec -> K2 -> K2 -> no-spec local smoke (not a perf benchmark).

Server must be started with SGLANG_OMNI_QTTS_CP_SPEC_K=2 and its
SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE pointing to --arm-file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _call(endpoint: str, payload: dict) -> tuple[bytes, float]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            audio = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code}: {exc.read()[:700]!r}"
        ) from exc
    seconds = time.perf_counter() - start
    if not audio or len(audio) < 44:
        raise RuntimeError("empty/invalid audio output")
    return audio, seconds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--arm-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261120)
    args = p.parse_args()
    if not 16 <= args.max_new_tokens <= 512:
        p.error("max_new_tokens must be in 16..512")
    output = args.out.expanduser().resolve()
    if output.exists():
        p.error("output already exists: never overwrite evidence")
    arm = args.arm_file.expanduser().resolve()
    if arm.exists():
        p.error("remove stale ARM marker before running")
    if not arm.parent.exists():
        p.error("ARM marker parent directory must already exist")
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("server not healthy; do not ARM")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": args.model,
        "input": (
            "We are testing a real Qwen3 TTS CodePredictor speculative decoding "
            "implementation. The target is a sampled audio generation request."
        ),
        "voice": "Vivian",
        "response_format": "wav",
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "stage_params": {
            "tts_engine": {
                "subtalker_dosample": True,
                "subtalker_temperature": 0.9,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
            },
        },
    }
    (output / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    evidence = []
    for idx, phase in enumerate(["base", "spec", "spec", "base"]):
        if arm.exists():
            raise RuntimeError("stale ARM marker detected")
        print(f"QTTS_K2_VISIT_BEGIN={idx:02d}_{phase}", flush=True)
        try:
            if phase == "spec":
                arm.touch(exist_ok=False)
            audio, seconds = _call(
                args.base_url.rstrip("/") + "/v1/audio/speech", payload
            )
        except Exception as exc:
            failure = {
                "status": "QTTS_CP_SPEC_K2_SMOKE_ABORTED_DO_NOT_USE_FOR_PERF",
                "failed_visit": f"{idx:02d}_{phase}",
                "exception": str(exc),
                "completed_visits": evidence,
                "arm_removed": True,
            }
            (output / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            print(f"QTTS_K2_VISIT_ERROR={idx:02d}_{phase}: {exc}", flush=True)
            raise
        finally:
            arm.unlink(missing_ok=True)
        (output / f"{idx:02d}_{phase}.wav").write_bytes(audio)
        evidence.append({
            "arm": phase == "spec",
            "phase": phase,
            "seconds_debug_not_perf": seconds,
            "bytes": len(audio),
            "sha256": hashlib.sha256(audio).hexdigest(),
        })
        print(f"QTTS_K2_VISIT_DONE={idx:02d}_{phase}", flush=True)
    report = {
        "status": "QTTS_CP_SPEC_K2_SMOKE_OUTPUTS_ONLY_NO_DISTRIBUTION_OR_SPEED_PROOF",
        "run_order": "ABBA",
        "input": payload,
        "visits": evidence,
        "limitations": [
            "These are four requests within one shared server, not fresh-server ABBA.",
            "CUDA event and HTTP wall times include Python debugging and CPU sync.",
            "Different WAV bytes are expected under independent sampling, not a parity failure.",
            "Check QTTS_CP_SPEC_K2 frame logs to prove that candidate truly ran.",
            "Need logits/KV parity, broad output quality, stability, then performance.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

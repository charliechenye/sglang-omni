# SPDX-License-Identifier: Apache-2.0
"""P19 CUDA700: exactly ONE post-health native-AR diagnostic HTTP request.

Never produces P19A/P19B PASS or uses a partially captured cost table.
Pair with a fresh H200 worker started by:
  P19_DIAG_ARM=AR_GRAPH P19_DIAG_DEPTH=1 P19_DIAG_FENCE=0 \
    P18_CAPTURE_FRAMES=2 P19_COST_FRAMES=1 bash p19_launch.sh ROOT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path
from urllib.request import urlopen

from experiments.qwen3_tts_cp_spec_k2.p18_run import PROMPTS, SEEDS_SMOKE
from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, _http_call, wav_info,
)

STATUS_OK = "P19_SINGLE_REQUEST_COMPLETED_DIAGNOSTIC_ONLY_NO_COST_SELECTION"
STATUS_BAD = "P19_CUDA700_OR_OTHER_DIAGNOSTIC_FAILURE_NO_SELECTION"
SCHEMA = "qwen3tts-p19-cuda700-one-request-v1"


def trace_tail(root: Path, count: int = 40) -> list[dict]:
    items = []
    for file in (root / "matched_cost" / "diagnostic").glob(
        "worker_pid*.jsonl"
    ):
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(item)
    return sorted(
        items, key=lambda row: row["monotonic_ns"]
    )[-count:]


def run(root: Path, base_url: str) -> dict:
    root = root.expanduser().resolve(strict=True)
    config = json.loads((root / "matched_cost" / "config.json").read_text())
    diagnostic = config.get("diagnostic")
    sha = (root / "source.sha").read_text().strip()
    local_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    if (not isinstance(diagnostic, dict)
            or diagnostic.get("policy")
                != "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION"
            or config.get("source_sha") != sha
            or local_sha != sha
            or len(sha) != 40):
        raise RuntimeError("P19_DIAGNOSTIC_SOURCE_OR_LAUNCH_MISMATCH")
    arm_path = root / "ARM_P18"
    case_path = root / "ACTIVE_CASE.json"
    diag_path = root / "matched_cost" / "diagnostic" / "client.json"
    if (diag_path.exists() or arm_path.exists() or case_path.exists()
            or (root / "visits.json").exists()
            or any((root / "census" / "rows").glob("*.json"))
            or any((root / "matched_cost" / "rows").glob("*.json"))):
        raise RuntimeError("P19_DIAGNOSTIC_REQUIRES_FRESH_WORKER_ROOT")
    with urlopen(base_url.rstrip("/") + "/health", timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError("P19_DIAGNOSTIC_SERVER_NOT_HEALTHY")

    prompt_id, prompt = PROMPTS[0]
    seed = SEEDS_SMOKE[0]
    case_id = f"p00_s{seed}"
    nonce = hashlib.sha256(f"{sha}|{case_id}|p18a-v1".encode()).hexdigest()
    identity = {
        "case_id": case_id, "prompt_id": prompt_id,
        "seed": seed, "request_nonce": nonce,
    }
    payload = {
        "model": MODEL, "input": prompt, "voice": "Vivian",
        "response_format": "wav", "seed": seed, "max_new_tokens": 768,
        "stage_params": {"tts_engine": {
            "subtalker_dosample": True,
            "subtalker_temperature": 0.9,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
        }},
    }
    result = {
        "schema": SCHEMA,
        "status": STATUS_BAD, "source_sha": sha,
        "diagnostic": diagnostic, "case": identity,
        "no_cost_or_dp_selection": True, "request": payload,
    }
    failure = None
    try:
        with case_path.open("x", encoding="utf-8") as out:
            json.dump(identity, out)
            out.write("\n")
        arm_path.touch(exist_ok=False)
        print(
            f"P19_DIAG_HTTP_BEGIN mode={diagnostic['arm']} "
            f"depth={diagnostic['depth']}", flush=True,
        )
        wav, finish, wall = _http_call(base_url, payload)
        if finish != "stop":
            raise RuntimeError(
                f"P19_DIAGNOSTIC_NON_NATURAL_STOP: {finish}"
            )
        result.update({
            "status": STATUS_OK,
            "finish_reason": finish,
            "wav_sha256": hashlib.sha256(wav).hexdigest(),
            "http_wall_DEBUG_NOT_PERFORMANCE_s": wall,
            **wav_info(wav),
        })
    except BaseException as exc:
        failure = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        arm_path.unlink(missing_ok=True)
        case_path.unlink(missing_ok=True)
        result["census_rows"] = len(list(
            (root / "census" / "rows").glob("*.json")
        ))
        result["P19_full_cost_rows_should_be_zero"] = len(list(
            (root / "matched_cost" / "rows").glob("*.json")
        ))
        result["trace_tail"] = trace_tail(root)
        if result["P19_full_cost_rows_should_be_zero"] != 0:
            result["status"] = STATUS_BAD
            result["error"] = "P19_DIAGNOSTIC_ILLEGAL_FORMAL_COST_RECORD"
            failure = RuntimeError(result["error"])
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with diag_path.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, sort_keys=True)
            out.write("\n")
        print(f"P19_DIAG_STATUS={result['status']}", flush=True)
        print(f"P19_DIAG_REPORT={diag_path}", flush=True)
        for item in result["trace_tail"][-18:]:
            print(
                "P19_DIAG_PHASE="
                f"{item.get('stage')} depth={item.get('depth')} "
                f"frame={item.get('frame')} arm={item.get('arm')}",
                flush=True,
            )
    if failure is not None:
        raise failure
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    run(args.root, args.base_url)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""P18A: sequential native-AR on-policy requests, fail-closed evidence driver.

SMOKE: 2 prompts x 2 seeds x 2 codec frames. MAIN: 8 prompts x 3 seeds x
8 codec frames. Frames are WITHIN-REQUEST clusters, not independent
prompts; one shared counterfactual RNG tuple per real depth checkpoint.
The real decoder remains native AR EAGER while armed; the entire run is
NOT a throughput or speech-quality experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, _http_call, wav_info,
)

PROMPTS = (
    ("english_engineering",
     "The engineer checked the last performance trace, verified the "
     "memory allocation, and explained why changing one small kernel "
     "could alter a much larger generation pipeline."),
    ("english_story",
     "On a quiet afternoon, the old bookstore opened its doors. A "
     "curious visitor found a handwritten note inside a novel and "
     "started following the clues through the neighborhood."),
    ("english_product",
     "Imagine a voice assistant that responds reliably even when "
     "network conditions change. The product team must balance "
     "latency, consistency, and the clarity of every spoken answer."),
    ("english_weather",
     "The weekend forecast calls for cloudy skies and intermittent "
     "rain. If the weather improves by Sunday morning, we can walk "
     "through the park before meeting friends for lunch."),
    ("mandarin_engineering",
     "今天我们正在测试语音生成模型在不同解码深度下的表现。"
     "我们希望比较候选标记的接受概率，并且确保所有检查都不会改变最终生成的语音。"),
    ("mandarin_story",
     "小林走进安静的图书馆，发现桌子上放着一本旧书。"
     "书页之间夹着一张纸条，上面写着关于城市历史的几个有趣问题。"),
    ("english_numbers",
     "The report contains twelve observations from three different "
     "experiments. We should not confuse repeated samples from one "
     "request with independent observations across several requests."),
    ("english_explanation",
     "To understand the experiment, begin with the native audio "
     "tokens, construct a few speculative proposals, and compare "
     "their exact probability distributions against an independent "
     "reference implementation."),
)
SEEDS_SMOKE = (20261123, 20261124)
SEEDS_MAIN = (20261123, 20261124, 20261125)
MAX_FRAMES = {"smoke": 2, "main": 8}
MODES = {"smoke": (PROMPTS[:2], SEEDS_SMOKE),
         "main": (PROMPTS, SEEDS_MAIN)}
SCHEMA = "p18a-native-eager-http-driver-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def source_pin(root: Path) -> tuple[str, dict]:
    config = json.loads((root / "config.json").read_text())
    sha = (root / "source.sha").read_text().strip()
    local = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    if (len(sha) != 40 or sha != local
            or config.get("source_sha") != sha
            or config.get("trajectory") != "native-ar-eager-only"
            or config.get("native_k2") != "OFF"
            or config.get("gpu") != "NVIDIA H200"):
        raise RuntimeError("P18_SOURCE_OR_NATIVE_EAGER_LAUNCH_MISMATCH")
    return sha, config


def rows_for_case(root: Path, case_id: str) -> list[dict]:
    rows = []
    for path in sorted((root / "census" / "rows").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("case_id") == case_id:
            rows.append(row)
    return rows


def run(root: Path, mode: str, *, base_url: str) -> dict:
    root = root.expanduser().resolve(strict=True)
    sha, config = source_pin(root)
    if mode not in MODES:
        raise ValueError("P18 mode must be smoke or main")
    frames = MAX_FRAMES[mode]
    if config.get("frames_per_request") != frames:
        raise RuntimeError(
            f"P18_LAUNCH_FRAMES_MISMATCH: mode={mode} wants {frames}, "
            f"server={config.get('frames_per_request')}"
        )
    report_file = root / "visits.json"
    marker = root / "ACTIVE_CASE.json"
    arm = root / "ARM_P18"
    if (report_file.exists() or marker.exists() or arm.exists()
            or list((root / "census" / "rows").glob("*.json"))
            or list((root / "census" / "failures").glob("*"))):
        raise RuntimeError("P18 requires FRESH server and unmodified census")
    with urlopen(base_url.rstrip("/") + "/health", timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError("P18 server health check failed")
    prompt_set, seeds = MODES[mode]
    report = {
        "schema": SCHEMA, "status": "P18A_INCOMPLETE_NO_SELECTION",
        "source_sha": sha, "mode": mode,
        "frames_per_request": frames,
        "trajectory": "NATIVE_AR_EAGER_NOT_GRAPH_SERVING",
        "prompts": [
            {"id": name, "text": text, "text_sha256": sha256_text(text)}
            for name, text in prompt_set
        ],
        "seeds": list(seeds), "visits": [],
    }
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for pidx, (prompt_id, text) in enumerate(prompt_set):
            for seed in seeds:
                case_id = f"p{pidx:02d}_s{seed}"
                nonce = sha256_text(f"{sha}|{case_id}|p18a-v1")
                case = {
                    "case_id": case_id, "prompt_id": prompt_id,
                    "seed": seed, "request_nonce": nonce,
                }
                with marker.open("x", encoding="utf-8") as fh:
                    json.dump(case, fh)
                    fh.write("\n")
                arm.touch(exist_ok=False)
                print(f"P18A_CASE_BEGIN={case_id}", flush=True)
                payload = {
                    "model": MODEL, "input": text, "voice": "Vivian",
                    "response_format": "wav", "seed": seed,
                    "max_new_tokens": 768,
                    "stage_params": {"tts_engine": {
                        "subtalker_dosample": True,
                        "subtalker_temperature": 0.9,
                        "subtalker_top_k": 50,
                        "subtalker_top_p": 1.0,
                    }},
                }
                try:
                    audio, finish, wall = _http_call(base_url, payload)
                finally:
                    arm.unlink(missing_ok=True)
                    marker.unlink(missing_ok=True)
                if finish != "stop":
                    raise RuntimeError(
                        f"P18A_NON_NATURAL_STOP: {case_id} finish={finish}"
                    )
                rows = rows_for_case(root, case_id)
                ids = {r["request_hash"] for r in rows}
                depths = {(r["frame"], r["last_committed_depth"]) for r in rows}
                selected_frames = {r["frame"] for r in rows}
                expected_rows = frames * 14
                if (len(rows) != expected_rows
                        or len(depths) != expected_rows
                        or len(ids) != 1
                        or len(selected_frames) != frames
                        or sorted({d for _, d in depths}) != list(range(1, 15))
                        or any(r.get("source_sha") != sha for r in rows)
                        or any(r.get("request_nonce") != nonce for r in rows)):
                    raise RuntimeError(
                        f"P18A_INCOMPLETE_REAL_DEPTH_CENSUS: case={case_id} "
                        f"rows={len(rows)}/{expected_rows} "
                        f"frames={len(selected_frames)}/{frames} "
                        f"request_ids={len(ids)}"
                    )
                report["visits"].append({
                    "case_id": case_id, "prompt_id": prompt_id,
                    "seed": seed, "request_hash": next(iter(ids)),
                    "finish_reason": finish, "source_sha": sha,
                    "frames": frames, "depth_rows": len(rows),
                    "wav_sha256": hashlib.sha256(audio).hexdigest(),
                    "http_wall_s_DEBUG_NOT_PERFORMANCE": wall,
                    **wav_info(audio),
                })
                report_file.write_text(json.dumps(report, indent=2) + "\n")
                print(f"P18A_CASE_PASS={case_id} rows={len(rows)}", flush=True)
        report["status"] = "P18A_NATIVE_AR_ALL_DEPTH_CENSUS_EVIDENCE_CAPTURED"
    except BaseException as exc:
        report["status"] = "P18A_FIRST_ERROR_NO_SELECTION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"P18A_VISITS_DONE={len(report['visits'])} report={report_file}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mode", choices=tuple(MODES), required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    run(a.root, a.mode, base_url=a.base_url)


if __name__ == "__main__":
    main()

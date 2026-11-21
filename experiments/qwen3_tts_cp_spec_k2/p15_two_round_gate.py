# SPDX-License-Identifier: Apache-2.0
"""P15 real H200 2-round K2 conditional Graph source-locked smoke gate.

Starts from an already healthy same-source worker. Baseline Graph-ON
captures from P9, P15 on-policy only on the debug K2 arm. There is NO
serving speedup claim, native RNG parity claim, or kernel-skip trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from urllib.request import urlopen

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

SCHEMA = "qwen3tts-cpspec-p15-two-round-device-branch-real-fused-v1"
PASS = (
    "P15_REAL_TWO_ROUND_GPU_CAUSAL_GRAPH_PARITY_ALL4_"
    "NOT_SEEDED_RNG_OR_14DEPTH_OR_E2E"
)
INCOMPLETE = "P15_REAL_TWO_ROUND_PARITY_PASS_BRANCH_COVERAGE_INCOMPLETE"
GO = "P15_TWO_ROUND_GO_TO_SEEDED_RNG_AND_KERNEL_TRACE_NOT_14DEPTH_OR_E2E"
NO_GO = "P15_TWO_ROUND_COVERAGE_INCOMPLETE_NO_GO"
FAIL = "P15_TWO_ROUND_SOURCE_LOCKED_VERIFICATION_FAIL"
PHASES = ("base", "spec", "spec", "base")
BRANCHES = {"AA", "AR", "RA", "RR"}


def validate_probe(row: dict, *, sha: str, pid: int, gpu: str) -> str:
    """Never promote a partial/proxy probe to a full P15 result."""
    counts = row.get("branch_counts")
    cases = row.get("cases")
    blocks = row.get("cuda_graph_event_ms_blocks")
    if (
        row.get("schema") != SCHEMA
        or row.get("status") not in (PASS, INCOMPLETE)
        or row.get("source_sha") != sha
        or row.get("pid") != pid
        or row.get("gpu") != gpu
        or row.get("predictor_layers") != 5
        or row.get("backend") != "matched_fused_predictor_device_offset"
        or row.get("initial_depth") != 9
        or row.get("initial_cache_len") != 10
        or row.get("sampling", {}).get("top_k") != 50
        or row.get("captured_graph_count") != 1
        or row.get("capture_contains_gpu_second_round_cond") is not True
        or row.get("capture_contains_conditional_bonus") is not True
        or row.get("bonus_kernel_skip_trace_proven") is not False
        or row.get("native_seeded_sampler_parity_proven") is not False
        or not isinstance(counts, dict)
        or set(counts) != BRANCHES
        or not isinstance(cases, list)
        or len(cases) != 16
        or not isinstance(blocks, list)
        or len(blocks) != 4
    ):
        raise ValueError("P15 incomplete or mixed-source Graph evidence")
    if any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0
        for x in blocks
    ):
        raise ValueError("P15 invalid Graph event envelope")
    if (
        not math.isfinite(row.get("median_cuda_graph_event_ms", float("nan")))
        or row["median_cuda_graph_event_ms"] <= 0
        or any(type(v) is not int or v < 0 for v in counts.values())
        or sum(counts.values()) != len(cases)
    ):
        raise ValueError("P15 branch census or Graph median invalid")
    computed = {key: 0 for key in BRANCHES}
    for index, case in enumerate(cases):
        if (
            case.get("case_index") != index
            or case.get("transition") not in BRANCHES
            or case.get("full_physical_kv_parity") is not True
            or case.get("independent_dense_sampler_parity") is not True
            or case.get("device_state_consumed_by_round2") is not True
            or len(case.get("uniforms", [])) != 2
            or any(len(u) != 4 for u in case["uniforms"])
        ):
            raise ValueError("P15 two-round case failed correctness gate")
        for name in ("round1", "round2"):
            data = case.get(name, {})
            if (
                not isinstance(data.get("draft"), int)
                or not isinstance(data.get("committed"), int)
                or not isinstance(data.get("bonus"), int)
                or type(data.get("accepted")) is not bool
                or data.get("bonus_executed") != int(data["accepted"])
                or not 0 <= data["draft"] < 2048
                or not 0 <= data["committed"] < 2048
                or not 0 <= data["bonus"] < 2048
            ):
                raise ValueError("P15 invalid committed/bonus code evidence")
        first = case["round1"]
        second = case["round2"]
        if (
            first["next_cache_len"] != 11 + int(first["accepted"])
            or first["next_depth"] != 10 + int(first["accepted"])
            or second["next_cache_len"] != (
                first["next_cache_len"] + 1 + int(second["accepted"])
            )
            or second["next_depth"] != (
                first["next_depth"] + 1 + int(second["accepted"])
            )
        ):
            raise ValueError("P15 GPU logical depth/KV path is inconsistent")
        computed[case["transition"]] += 1
    if computed != counts:
        raise ValueError("P15 branch counts disagree with case evidence")
    full = all(counts[x] > 0 for x in BRANCHES)
    if full != (row["status"] == PASS):
        raise ValueError("P15 result exaggerates branch coverage")
    return GO if full else NO_GO


def summarize(result: dict) -> str:
    row = result["two_round_probe"]
    counts = row["branch_counts"]
    return "\n".join([
        "# P15 source-locked real two-round GPU causal Graph",
        "",
        f"**{result['status']}**",
        "",
        f"Source SHA: {result['source_sha']}",
        f"GPU: {result['gpu']}",
        f"PID: {result['pid']}",
        f"Native full Predictor Graph ON (context only): "
        f"{result['native_graph_ms_per_frame']:.5f} ms/frame",
        f"P15 2-round GPU Graph event envelope (NOT E2E): "
        f"{row['median_cuda_graph_event_ms']:.5f} ms",
        "",
        "| Transition | AA | AR | RA | RR |",
        "|---|---:|---:|---:|---:|",
        f"| Cases | {counts['AA']} | {counts['AR']} | "
        f"{counts['RA']} | {counts['RR']} |",
        "",
        "- Round 2 consumes GPU-selected Round 1 hidden/embedding/sum/KV offset.",
        "- Independent CPU dense q/p, exact codes/decisions and matched "
        "P10A KV tolerance checked for every observed case.",
        "- Rejection bonus is structured as a CUDA conditional branch, "
        "not an unconditional mask; **Nsight kernel-skip proof is missing**.",
        "- Externally fixed GPU uniforms; **not native seeded request RNG**.",
        "- This is two rounds only. No 14-depth completion, no serving "
        "speedup and no audio quality equivalence.",
        "- Missing AA/AR/RA/RR transition means NO GO, even if "
        "all observed cases have numerical parity.",
        "",
    ]) + "\n"


def run(args):
    out = args.out.expanduser().resolve()
    base_arm = args.base_arm.expanduser().resolve()
    spec_arm = args.spec_arm.expanduser().resolve()
    base_dir = args.baseline_dir.expanduser().resolve()
    probe_dir = args.probe_dir.expanduser().resolve()
    logfile = args.server_log.expanduser().resolve(strict=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    if len(sha) != 40 or sha != args.expect_source_sha:
        raise RuntimeError("P15 local HEAD/SHA drift; start a fresh worker")
    if (
        out.exists() or base_dir.exists() or probe_dir.exists()
        or base_arm == spec_arm
        or base_arm.exists() or spec_arm.exists()
        or not base_arm.parent.is_dir()
        or not spec_arm.parent.is_dir()
    ):
        raise RuntimeError("P15 requires absent distinct ARM and output paths")
    with urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("P15 worker unhealthy")
    payload = {
        "model": args.model, "input": args.text, "voice": args.voice,
        "response_format": "wav", "seed": args.seed,
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
    out.mkdir(parents=True, exist_ok=False)
    (out / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    result = {
        "status": FAIL, "source_sha": sha, "request": payload,
        "visits": [], "kind": "two_round_real_gpu_causal_graph_not_e2e",
    }
    try:
        for index, phase in enumerate(PHASES):
            if base_arm.exists() or spec_arm.exists():
                raise RuntimeError("P15 ARM state leaked between visits")
            arm = base_arm if phase == "base" else spec_arm
            tag = f"{index:02d}_{phase}"
            prior = len(parse_k2_log_entries(
                logfile.read_text(errors="replace")
            ))
            print(f"P15_VISIT_BEGIN={tag}", flush=True)
            try:
                arm.touch(exist_ok=False)
                wav, finish, wall = _http_call(args.base_url, payload)
            finally:
                arm.unlink(missing_ok=True)
            new = parse_k2_log_entries(
                logfile.read_text(errors="replace")
            )[prior:]
            if (
                finish != "stop"
                or (phase == "base" and new)
                or (phase == "spec" and not new)
                or any(event["ar_steps"] != 0 for event in new)
            ):
                raise RuntimeError(
                    f"P15 {tag} natural-stop/K2 ARM mismatch "
                    f"finish={finish} K2frames={len(new)}"
                )
            (out / f"{tag}.wav").write_bytes(wav)
            result["visits"].append({
                "phase": phase, "tag": tag, "finish_reason": finish,
                "k2_frames": len(new),
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                "debug_http_wall_NOT_PERF": wall, **wav_info(wav),
            })
            (out / "report.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(f"P15_VISIT_DONE={tag} finish={finish}", flush=True)

        base_files = list(base_dir.glob("baseline_graph_pid*.json"))
        probe_files = list(probe_dir.glob(
            "two_round_pid*_frame*_depth10.json"
        ))
        failures = list(probe_dir.glob("failure_pid*.txt"))
        if len(base_files) != 1 or len(probe_files) != 1 or failures:
            raise RuntimeError(
                f"P15 requires one native Graph baseline, one P15 depth10 "
                f"probe and no failures; got base={len(base_files)}, "
                f"probe={len(probe_files)}, failures={len(failures)}"
            )
        base = json.loads(base_files[0].read_text())
        row = json.loads(probe_files[0].read_text())
        if (
            base.get("kind") != "baseline_full_predictor_graph_on"
            or base.get("source_sha") != sha
            or base.get("frames", 0) < 8
            or base.get("graph_replays") != base.get("frames")
            or not math.isfinite(base.get("median_cuda_event_ms", math.nan))
            or base["median_cuda_event_ms"] <= 0
        ):
            raise RuntimeError("P15 native optimized Graph-ON baseline invalid")
        decision = validate_probe(
            row, sha=sha, pid=base["pid"], gpu=base["device"],
        )
        result.update({
            "status": decision,
            "pid": base["pid"], "gpu": base["device"],
            "native_graph_ms_per_frame": base["median_cuda_event_ms"],
            "two_round_probe": row,
        })
    except BaseException as exc:
        result["status"] = FAIL
        result["error"] = str(exc)
        (out / "report.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        raise
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "summary.md").write_text(summarize(result))
    print(f"P15_STATUS={result['status']} REPORT={out/'summary.md'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--base-arm", type=Path, required=True)
    parser.add_argument("--spec-arm", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--expect-source-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--voice", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    if (
        not 256 <= args.max_new_tokens <= 2048
        or args.top_k != 50
        or not 0 < args.temperature <= 5
        or not 0 < args.top_p <= 1
    ):
        parser.error("P15 requires sampled C1 topk50, T>0 and top-p in (0,1]")
    run(args)


if __name__ == "__main__":
    main()

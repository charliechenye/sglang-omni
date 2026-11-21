# SPDX-License-Identifier: Apache-2.0
"""P17 source-locked H200 same-worker native/K2 ABBA + multi-width cost gate.

This validator is deliberately a *screen*, not a 14-depth/E2E GO:
depth2/10 checkpoints from 2 frames each do not establish all-depth
acceptance, full request RNG equivalence or true HTTP speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen
import subprocess

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

SCHEMA = "qwen3tts-cpspec-p17-real-multihead-single-wide-verifier-v1"
PROBE_PASS = "P17_REAL_MULTIWIDTH_SAMPLE_AND_COST_PARITY_PASS_NOT_E2E_OR_SPEEDUP"
PASS = "P17_K2_K3_K4_REAL_WIDE_CORRECTNESS_AND_COST_SCREEN_NOT_FULL_DECODER"
FAIL = "P17_SOURCE_LOCKED_MULTIWIDTH_SCREEN_FAILED"
PHASES = ("base", "spec", "spec", "base")
WIDTHS = ("2", "3", "4")
DEPTHS = (1, 9)
FRAMES_PER_DEPTH = 2
DRAW_CASES = 16


def validate_probe(row: dict, *, sha: str, pid: int, gpu: str) -> None:
    widths = row.get("widths", {})
    if (
        row.get("schema") != SCHEMA
        or row.get("status") != PROBE_PASS
        or row.get("source_sha") != sha
        or row.get("pid") != pid
        or row.get("gpu") != gpu
        or row.get("last_committed_depth") not in DEPTHS
        or row.get("cache_len") != row["last_committed_depth"] + 1
        or row.get("predictor_layers") != 5
        or row.get("backend") != "P17_real_fused_bf16_5layer_wide"
        or row.get("sampling", {}).get("top_k") != 50
        or row.get("uniform_cases_per_width") != DRAW_CASES
        or set(widths) != set(WIDTHS)
    ):
        raise ValueError("P17 incomplete or mixed-source multiwidth evidence")
    for width in WIDTHS:
        data = widths[width]
        w = int(width)
        cases = data.get("cases", [])
        hist = data.get("committed_count_histogram")
        accept_hist = data.get("accepted_prefix_histogram")
        timings = data.get("cuda_graph_event_ms_blocks", {})
        medians = data.get("cuda_graph_event_ms_medians", {})
        if (
            data.get("on_policy_real_checkpoint") is not True
            or data.get("batched_multihead_and_static_fused_parity") is not True
            or data.get("full_physical_kv_parity") is not True
            or data.get("independent_cpu_accept_residual_parity") is not True
            or data.get("case_count") != DRAW_CASES
            or not isinstance(cases, list) or len(cases) != DRAW_CASES
            or not isinstance(hist, dict)
            or set(hist) != {str(i) for i in range(1, w+1)}
            or not isinstance(accept_hist, dict)
            or set(accept_hist) != {str(i) for i in range(w)}
            or set(timings) != {"full_round", "proposal_only", "verifier_only"}
            or set(medians) != set(timings)
            or data.get("timed_full_and_proposal_case_indices")
            != [0, 5, 10, 15]
            or data.get("verifier_only_frozen_case_index") != 0
        ):
            raise ValueError(f"P17_K{width}_CHECKPOINT_OR_TIMING_INCOMPLETE")
        counts = [0]*(w+1)
        accepted_counts = [0]*w
        for index, record in enumerate(cases):
            count, prefix = (
                record.get("committed_count"), record.get("accepted_prefix")
            )
            if (
                record.get("index") != index
                or type(count) is not int or not 1 <= count <= w
                or type(prefix) is not int or not 0 <= prefix < w
                or count != prefix + 1
                or record.get("bonus_committed") is not (prefix == w-1)
                or record.get("first_rejection") != (
                    prefix if prefix < w-1 else None
                )
            ):
                raise ValueError(f"P17_K{width}_ACCEPTANCE_ACCOUNTING_BAD")
            counts[count] += 1
            accepted_counts[prefix] += 1
        if (
            any(hist[str(i)] != counts[i] for i in range(1,w+1))
            or any(accept_hist[str(i)] != accepted_counts[i]
                   for i in range(w))
        ):
            raise ValueError(f"P17_K{width}_ACCEPTANCE_HISTOGRAM_MISMATCH")
        mean_comm = statistics.mean(
            x["committed_count"] for x in cases
        )
        if data["mean_committed_per_round_fixed_checkpoint"] != mean_comm:
            raise ValueError("P17_K% s mean accounting drift" % width)
        for name in timings:
            values = timings[name]
            expected_indices = [0, 5, 10, 15] if name != "verifier_only" else [0]
            if (
                len(values) != len(expected_indices)*4
                or sorted({v.get("case_index") for v in values}) != expected_indices
                or any(
                    v.get("repeat_index") not in (0, 1, 2, 3)
                    or type(v.get("cuda_event_ms")) not in (int,float)
                    or not math.isfinite(v["cuda_event_ms"])
                    or v["cuda_event_ms"] <= 0 for v in values
                )
                or medians[name] != statistics.median(
                    v["cuda_event_ms"] for v in values
                )
            ):
                raise ValueError(f"P17_K{width}_{name}_TIMING_NOT_SOURCE_LOCKED")
        ratio = medians["full_round"] / mean_comm
        if data["full_round_ms_per_committed_code_SCREEN_ONLY"] != ratio:
            raise ValueError("P17_K% s cost ratio inconsistent" % width)


def summarize(result: dict) -> str:
    lines = [
        "# P17 — NVIDIA-style multi-draft K2/K3/K4 one-wide-verifier H200",
        "",
        f"**{result['status']}**",
        "",
        f"Source SHA: {result['source_sha']}",
        f"PID: {result['pid']}; GPU: {result['gpu']}",
        f"Native optimized Predictor Graph ON: "
        f"{result['native_full_graph_ms']:.5f} ms/frame (context ONLY)",
        "",
        "| Last committed depth | Width | Frozen checkpoints | "
        "Fixed-u mean committed/round | Full wide round CUDA-event ms | "
        "Batched q head+draft only ms | Verifier only ms | "
        "Full-round ms/committed code (screen) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        for width in WIDTHS:
            row = result["depth_summary"][str(depth)][width]
            x = row["median_event_ms"]
            lines.append(
                f"| {depth} | K={width} | {row['checkpoints']} | "
                f"{row['mean_committed']:.3f} | "
                f"{x['full_round']:.5f} | "
                f"{x['proposal_only']:.5f} | "
                f"{x['verifier_only']:.5f} | "
                f"{row['ms_per_committed_code_SCREEN_ONLY']:.5f} |"
            )
    lines += [
        "",
        "## Decision limits",
        "",
        "- Genuine C1 H200 on-policy K2 checkpoints at depths2 and10, "
        "two different codec frames each; each has 16 *fixed* GPU-uniform "
        "samples. These are NOT 64 independent requests.",
        "- All sampled drafts: multiple distinct head weights applied to "
        "the SAME old hidden via a batched GPU GEMM. Exactly ONE "
        "K-token real fused verifier per round, not a chain of K=2.",
        "- Each case checks independent native LM-head probability, "
        "static-END fused verifier, target and q distributions, CPU "
        "positive residual and full K/V restoration.",
        "- The P17 acceptance/bonus graph unconditionally computes all "
        "K-1 acceptance kernels and final bonus distribution; unlike "
        "NVIDIA's fused proposal/accept CUDA kernels it is NOT fully optimized.",
        "- Component event envelopes are separate graphs and are NOT "
        "additive. Full-round timing samples inputs 0/5/10/15, not just "
        "the last uniform. No HTTP speedup estimate.",
        "- An average over only depths2/10 cannot be extrapolated to "
        "all residual codebooks. No native seeded RNG, complete frame "
        "decoder, production serving throughput, or speech quality proof.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    out = args.out.expanduser().resolve()
    root = args.probe_dir.expanduser().resolve()
    base_dir = args.baseline_dir.expanduser().resolve()
    logfile = args.server_log.expanduser().resolve(strict=True)
    base_arm = args.base_arm.expanduser().resolve()
    spec_arm = args.spec_arm.expanduser().resolve()
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    if sha != args.expect_source_sha or len(sha) != 40:
        raise RuntimeError("P17 local repository source SHA drift")
    if (
        out.exists() or root.exists() or base_dir.exists()
        or base_arm.exists() or spec_arm.exists()
        or base_arm == spec_arm
        or not base_arm.parent.is_dir() or not spec_arm.parent.is_dir()
    ):
        raise RuntimeError("P17 requires fresh distinct evidence paths")
    with urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("P17 H200 server health check failed")
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
            },
        },
    }
    report = {
        "status": FAIL,
        "source_sha": sha,
        "kind": "P17_REAL_MULTIWIDTH_COMPONENT_ONLY",
        "request": payload, "visits": [],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "request.json").write_text(
        json.dumps(payload, indent=2) + "\n",
    )
    try:
        for idx, phase in enumerate(PHASES):
            if base_arm.exists() or spec_arm.exists():
                raise RuntimeError("P17 overlapping ARM files")
            arm = base_arm if phase == "base" else spec_arm
            tag = f"{idx:02d}_{phase}"
            prior = len(parse_k2_log_entries(
                logfile.read_text(errors="replace")
            ))
            print(f"P17_VISIT_BEGIN={tag}", flush=True)
            try:
                arm.touch(exist_ok=False)
                wav, finish, elapsed = _http_call(args.base_url, payload)
            finally:
                arm.unlink(missing_ok=True)
            new = parse_k2_log_entries(
                logfile.read_text(errors="replace")
            )[prior:]
            if (
                finish != "stop"
                or (phase == "base" and new)
                or (phase == "spec" and not new)
                or any(x.get("ar_steps") != 0 for x in new)
            ):
                raise RuntimeError(
                    f"P17 {tag} natural stop or K2 ARM accounting failed: "
                    f"finish={finish} k2frames={len(new)}"
                )
            (out / f"{tag}.wav").write_bytes(wav)
            report["visits"].append({
                "phase": phase, "tag": tag,
                "finish_reason": finish, "k2_frames": len(new),
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                "http_debug_wall_NOT_PERFORMANCE": elapsed,
                **wav_info(wav),
            })
            (out / "report.json").write_text(
                json.dumps(report, indent=2) + "\n",
            )
            print(f"P17_VISIT_DONE={tag}", flush=True)

        base_files = list(base_dir.glob("baseline_graph_pid*.json"))
        probes = sorted(root.glob("wide_pid*_frame*_depth*.json"))
        failures = list(root.glob("failure_pid*.txt"))
        if len(base_files) != 1 or len(probes) != 2*FRAMES_PER_DEPTH or failures:
            raise RuntimeError(
                "P17 requires one native Graph baseline, four distinct "
                f"on-policy checkpoints, no failures; baselines={len(base_files)} "
                f"probes={len(probes)} failures={len(failures)}"
            )
        base = json.loads(base_files[0].read_text())
        if (
            base.get("kind") != "baseline_full_predictor_graph_on"
            or base.get("source_sha") != sha
            or base.get("frames", 0) < 8
            or base.get("graph_replays") != base.get("frames")
            or not math.isfinite(base.get("median_cuda_event_ms", float("nan")))
            or base["median_cuda_event_ms"] <= 0
        ):
            raise RuntimeError("P17 native Graph ON source-matched baseline bad")
        rows = [json.loads(path.read_text()) for path in probes]
        for row in rows:
            validate_probe(row, sha=sha, pid=base["pid"], gpu=base["device"])
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["last_committed_depth"]].append(row)
        if set(grouped) != set(DEPTHS) or any(
            len(grouped[d]) != FRAMES_PER_DEPTH or
            len({x["frame"] for x in grouped[d]}) != FRAMES_PER_DEPTH
            for d in DEPTHS
        ):
            raise RuntimeError("P17 checkpoint depth/frame provenance incomplete")
        summary = {}
        for depth in DEPTHS:
            summary[str(depth)] = {}
            for width in WIDTHS:
                cells = [x["widths"][width] for x in grouped[depth]]
                mean_committed = statistics.mean(
                    x["mean_committed_per_round_fixed_checkpoint"]
                    for x in cells
                )
                medians = {
                    stage: statistics.median(
                        x["cuda_graph_event_ms_medians"][stage]
                        for x in cells
                    ) for stage in (
                        "full_round", "proposal_only", "verifier_only",
                    )
                }
                summary[str(depth)][width] = {
                    "checkpoints": len(cells),
                    "mean_committed": mean_committed,
                    "median_event_ms": medians,
                    "ms_per_committed_code_SCREEN_ONLY":
                        medians["full_round"] / mean_committed,
                }
        report.update({
            "status": PASS,
            "gpu": base["device"], "pid": base["pid"],
            "native_full_graph_ms": base["median_cuda_event_ms"],
            "native_full_graph_normalized_14_AR_STEPS_CONTEXT_ONLY":
                base["median_cuda_event_ms"] / 14,
            "depth_summary": summary,
            "real_on_policy_checkpoint_rows": rows,
            "limitations": [
                "Only two correlated real codec frames at depths2/10",
                "16 fixed draws per frozen checkpoint, no on-policy full corpus",
                "No native seeded RNG or full 14-depth K2/K3/K4 implementation",
                "CUDA event envelopes are not HTTP serving performance",
                "No quality/listening or ASR equivalence",
            ],
        })
    except BaseException as err:
        report["status"] = FAIL
        report["error"] = str(err)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.md").write_text(summarize(report) + "\n")
    print(f"P17_STATUS={PASS} REPORT={out/'summary.md'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--base-arm", required=True, type=Path)
    parser.add_argument("--spec-arm", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--expect-source-sha", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--voice", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.)
    args = parser.parse_args()
    if (
        args.top_k != 50 or not 256 <= args.max_new_tokens <= 2048
        or not 0 < args.temperature <= 5 or not 0 < args.top_p <= 1
    ):
        parser.error("P17 requires C1 sampled topk50, valid T and top_p")
    run(args)


if __name__ == "__main__":
    main()

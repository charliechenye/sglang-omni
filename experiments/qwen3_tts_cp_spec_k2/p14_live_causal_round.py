# SPDX-License-Identifier: Apache-2.0
"""P14 source-locked H200 one-ROUND causal CUDA Graph acceptance.

The *same* optimized Qwen3-TTS worker first records the complete native
Graph-ON Predictor as a reference. Then opt-in real C1 sampled K2
requests capture a single CUDA Graph at residual depths2 and10:
q head -> sampled draft -> actual draft embedding -> packed verifier ->
p head -> GPU accept/residual -> bonus code + next logical GPU state.
Compare this entire matched graph Torch Topk vs P12A Triton Topk, AND
a separate private RNG candidate graph. Restore physical K/V after
every probe. This is NOT the full 14-depth GPU loop or serving QPS.

Both baseline and K2/debug must naturally stop; source SHA, worker
PID/GPU and exact 5-layer parity must match. Do not claim performance
from the deliberately instrumented HTTP wall latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from urllib.request import urlopen

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

PASS = "P14_CAUSAL_ONE_ROUND_GPU_GRAPH_SOURCE_LOCKED_PASS_NOT_14DEPTH_OR_E2E"
FAIL = "P14_CAUSAL_ONE_ROUND_NO_FULL_GPU_K2_SELECTION"
PHASES = ("base", "spec", "spec", "base")
SCHEMA = "qwen3tts-cpspec-p14-one-round-causal-q-proposal-verifier-p-accept-bonus-v1"


def validate_probe(row: dict, *, sha: str, pid: int, gpu: str) -> None:
    cases = row.get("fixed_uniform_cases", [])
    times = row.get("median_cuda_graph_ms", {})
    blocks = row.get("cuda_event_ms_blocks", {})
    if (
        row.get("schema") != SCHEMA
        or row.get("status") !=
        "P14_REAL_CAUSAL_ONE_ROUND_Q_VERIFIER_P_RESIDUAL_BONUS_"
        "GPU_GRAPH_PARITY_PASS_NOT_14DEPTH_OR_E2E"
        or row.get("source_sha") != sha
        or row.get("pid") != pid
        or row.get("gpu") != gpu
        or row.get("predictor_layers") != 5
        or row.get("backend") != "matched_fused_predictor_layers_static_offset"
        or row.get("captured_graph_count") != 3
        or row.get("private_rng_replay_unique") is not True
        or row.get("proposal_depth") not in (2, 10)
        or row.get("cache_len_at_entry") != row.get("proposal_depth")
        or row.get("sampling", {}).get("top_k") != 50
        or len(cases) != 4
        or set(times) != {
            "torch_reference", "triton_candidate", "triton_private_rng"
        }
        or set(blocks) != set(times)
    ):
        raise ValueError("P14 incomplete or mixed-source causal Graph evidence")
    for name in times:
        value = times[name]
        if (
            type(value) not in (float, int) or not math.isfinite(value)
            or value <= 0 or len(blocks[name]) != 4
            or any(
                type(v) not in (float, int) or not math.isfinite(v) or v <= 0
                for v in blocks[name]
            )
        ):
            raise ValueError(f"P14 bad or missing same-process event times: {name}")
    for case in cases:
        if (
            len(case.get("uniforms", [])) != 4
            or case.get("q_p_distribution_parity") is not True
            or case.get("target_hidden_parity") is not True
            or case.get("complete_kv_parity") is not True
            or case.get("committed_embedding_sum_parity") is not True
            or case.get("bonus_committed") is not case.get("accepted")
            or not isinstance(case.get("draft"), int)
            or not isinstance(case.get("committed"), int)
            or not 0 <= case["draft"] < 2048
            or not 0 <= case["committed"] < 2048
            or not math.isfinite(case.get("accept_probability", float("nan")))
        ):
            raise ValueError("P14 one-round correctness/branch evidence incomplete")
    decisions = row.get("sampled_branch_count")
    if not isinstance(decisions, dict) or (
        decisions.get("accepted", 0) + decisions.get("rejected", 0)
        != len(cases)
    ):
        raise ValueError("P14 branch census inconsistent")
    # Some on-policy depths lack q/p support overlap. Rejection-only
    # is a legitimate measurement; NEVER falsely claim both branches.
    if row.get("relative_fast_vs_torch") != (
        times["triton_candidate"] / times["torch_reference"]
    ):
        raise ValueError("P14 rounded/inconsistent Graph speed ratio")


def run(args):
    out = args.out.expanduser().resolve()
    base_arm = args.base_arm.expanduser().resolve()
    spec_arm = args.spec_arm.expanduser().resolve()
    base_dir = args.baseline_dir.expanduser().resolve()
    probe_dir = args.probe_dir.expanduser().resolve()
    logfile = args.server_log.expanduser().resolve(strict=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if sha != args.expect_source_sha or len(sha) != 40:
        raise RuntimeError("P14 local experiment SHA drift; restart fresh server")
    if (
        out.exists() or base_dir.exists() or probe_dir.exists()
        or base_arm == spec_arm
        or base_arm.exists() or spec_arm.exists()
        or not base_arm.parent.is_dir() or not spec_arm.parent.is_dir()
    ):
        raise RuntimeError("P14 requires new evidence folders and distinct absent ARM files")
    with urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("P14 Omni server not healthy")

    payload = {
        "model": args.model, "input": args.text,
        "voice": args.voice, "response_format": "wav",
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
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
    report = {
        "status": "P14_EVIDENCE_INCOMPLETE",
        "source_sha": sha,
        "request": payload, "visits": [],
        "kind": "real_on_policy_causal_one_round_component_gpu_graph",
    }
    try:
        for idx, phase in enumerate(PHASES):
            if base_arm.exists() or spec_arm.exists():
                raise RuntimeError("P14 arm leak across HTTP visits")
            arm = base_arm if phase == "base" else spec_arm
            tag = f"{idx:02d}_{phase}"
            prior = len(parse_k2_log_entries(
                logfile.read_text(errors="replace")
            ))
            print(f"P14_CAUSAL_BEGIN={tag}", flush=True)
            try:
                arm.touch(exist_ok=False)
                wav, reason, elapsed = _http_call(args.base_url, payload)
            finally:
                arm.unlink(missing_ok=True)
            new = parse_k2_log_entries(
                logfile.read_text(errors="replace")
            )[prior:]
            if (
                reason != "stop"
                or (phase == "base" and new)
                or (phase == "spec" and len(new) == 0)
                or any(v["ar_steps"] != 0 for v in new)
            ):
                raise RuntimeError(
                    f"P14 {tag} natural-stop/real K2 arm evidence FAIL "
                    f"finish={reason} debug_k2_frames={len(new)}"
                )
            (out / f"{tag}.wav").write_bytes(wav)
            report["visits"].append({
                "phase": phase, "tag": tag,
                "finish_reason": reason,
                "k2_frames": len(new),
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                "debug_http_wall_NOT_PERF": elapsed,
                **wav_info(wav),
            })
            (out / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                f"P14_CAUSAL_DONE={tag} stop={reason} k2frames={len(new)}",
                flush=True,
            )

        base_files = list(base_dir.glob("baseline_graph_pid*.json"))
        probe_files = sorted(probe_dir.glob(
            "causal_pid*_frame*_depth*.json"
        ))
        failures = list(probe_dir.glob("failure_pid*.txt"))
        if len(base_files) != 1 or len(probe_files) != 2 or failures:
            raise RuntimeError(
                f"P14 required one real Graph-ON baseline, exactly two "
                f"causal probes, no failures. baseline={len(base_files)} "
                f"probe={len(probe_files)} failures={len(failures)}"
            )
        baseline = json.loads(base_files[0].read_text())
        if (
            baseline.get("kind") != "baseline_full_predictor_graph_on"
            or baseline.get("source_sha") != sha
            or baseline.get("frames", 0) < 8
            or baseline.get("graph_replays") != baseline.get("frames")
            or not math.isfinite(baseline.get("median_cuda_event_ms", math.nan))
            or baseline["median_cuda_event_ms"] <= 0
        ):
            raise ValueError("P14 complete native Predictor Graph-ON baseline invalid")
        rows = [json.loads(f.read_text()) for f in probe_files]
        if {r.get("proposal_depth") for r in rows} != {2, 10}:
            raise ValueError("P14 needs on-policy depth2 AND depth10 checkpoint")
        for row in rows:
            validate_probe(
                row, sha=sha,
                pid=baseline["pid"], gpu=baseline["device"]
            )
        report.update({
            "status": PASS,
            "gpu": baseline["device"],
            "pid": baseline["pid"],
            "native_full_predictor_graph_on_ms_per_frame": (
                baseline["median_cuda_event_ms"]
            ),
            "causal_graph_rounds": rows,
            "limits": [
                "All q-to-verifier-to-p dependencies are IN ONE causal GPU Graph",
                "Single static K2 proposal depth only, not next GPU KV continuation",
                "Conditional bonus head is evaluated always, even for rejection",
                "RNG private graph is not native SGLang seeded sampler RNG",
                "Original debug K2 output is untouched; HTTP wall not perf",
                "Cannot multiply round cost into production E2E proof",
            ],
        })
    except Exception as exc:
        report["status"] = FAIL
        report["error"] = str(exc)
        (out / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        raise

    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.md").write_text(summarize(report))
    print(f"P14_CAUSAL_STATUS={PASS} REPORT={out/'summary.md'}")
    return report


def summarize(report: dict) -> str:
    lines = [
        "# P14 H200 actual causal q→verifier→p→accept/residual→bonus ONE GRAPH",
        "", f"**{report['status']}**", "",
        f"Source SHA: {report['source_sha']}",
        f"GPU: {report['gpu']}",
        f"Native optimized full Predictor Graph ON: "
        f"{report['native_full_predictor_graph_on_ms_per_frame']:.5f}ms/frame",
        "",
        "| Proposal depth | Torch full causal Graph ms | Triton full causal Graph ms | Triton + private RNG ms | Candidate/Torch | Accepted fixed-u | Rejected fixed-u |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in sorted(
        report["causal_graph_rounds"], key=lambda r: r["proposal_depth"]
    ):
        time = x["median_cuda_graph_ms"]
        lines.append(
            f"| {x['proposal_depth']} | "
            f"{time['torch_reference']:.5f} | "
            f"{time['triton_candidate']:.5f} | "
            f"{time['triton_private_rng']:.5f} | "
            f"{x['relative_fast_vs_torch']:.3f}x | "
            f"{x['sampled_branch_count']['accepted']} | "
            f"{x['sampled_branch_count']['rejected']} |"
        )
    lines += [
        "", "## Decision: NOT a 14-depth device-branching GPU decoder",
        "",
        "- This is the first full causal one-ROUND CUDA Graph with actual draft-conditional verifier input, p distribution and positive-residual acceptance.",
        "- All fixed-uniform cases must pass independent Torch dense distribution, CPU accept/residual, hidden, code, embedding, logical KV depth and entire physical K/V parity.",
        "- GPU scalar next_cache_len/next_depth is OUTPUT only; not fed to a subsequent verifier. Rejected draft KV suffix is not consumed on-device.",
        "- Bonus head/draw is evaluated for both accept and reject, then bonus contribution masked when rejected. This static-graph overhead can erase savings.",
        "- Private graph CUDA RNG advances independently but is not Omni seeded sampler equivalent.",
        "- Complete optimized Predictor Graph ON is a context baseline; it is NOT directly comparable to a single round, and HTTP debug timing contains CPU sync.",
        "- Do not claim any production speedup or merge until the entire 14-depth GPU loop with dynamic KV and seeded sampling is validated and measured.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--base-arm", type=Path, required=True)
    ap.add_argument("--spec-arm", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--probe-dir", type=Path, required=True)
    ap.add_argument("--server-log", type=Path, required=True)
    ap.add_argument("--expect-source-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--voice", default="Vivian")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.)
    args = ap.parse_args()
    if (
        not 256 <= args.max_new_tokens <= 2048
        or not 0 < args.temperature <= 5
        or args.top_k != 50 or not 0 < args.top_p <= 1
    ):
        ap.error("P14 only supports C1 sampled topk50, T>0, top_p in (0,1]")
    run(args)


if __name__ == "__main__":
    main()

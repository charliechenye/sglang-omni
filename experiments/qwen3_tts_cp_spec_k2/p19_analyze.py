# SPDX-License-Identifier: Apache-2.0
"""P19A strict same-worker native sampled AR-step vs wide round cost validation.

The per-arm numbers are independently captured CUDA Graph *components* at
frozen native-eager checkpoints, NOT a matched full-Graph Predictor and
NOT HTTP throughput. Preserve complete source/checkpoint/seed/probe provenance.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.qwen3_tts_cp_spec_k2.p18_analyze import (
    DEPTHS, hierarchical_ci, validate_rows, widths,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p19_cost import (
    BLOCKS, REPEATS, SCHEMA,
)

STATUS = "P19A_SAME_NATIVE_CHECKPOINT_GRAPH_COMPONENT_COST_PASS_NOT_E2E"
CONFIG_SCHEMA = "qwen3tts-p19a-matched-cost-config-v1"


def validate_costs(root: Path) -> tuple[dict, list[dict], list[dict]]:
    root = root.expanduser().resolve(strict=True)
    driver, acceptance = validate_rows(root)
    cfg = json.loads((root / "matched_cost" / "config.json").read_text())
    src = (root / "source.sha").read_text().strip()
    count = cfg.get("p19_cost_frames_per_request")
    mode = driver["mode"]
    p18_frames = {"smoke": 2, "main": 8}[mode]
    if (cfg.get("schema") != CONFIG_SCHEMA or cfg.get("source_sha") != src
            or cfg.get("gpu") != "NVIDIA H200"
            or cfg.get("trajectory") != "P18_NATIVE_AR_EAGER_SAME_CHECKPOINT"
            or cfg.get("graph_costs") !=
                "independently_captured_fixed_state_NOT_FULL_FRAME"
            or cfg.get("native_step_scope")
                != "predictor+lm_head+real_seeded_sampler+embedding"
            or cfg.get("p18_frames_per_request") != p18_frames
            or type(count) is not int or count not in (1, 2)
            or count > p18_frames
            or any((root / "matched_cost" / "failures").glob("*"))):
        raise RuntimeError("P19A_CONFIG_SOURCE_OR_FAILURE_GATE")
    files = sorted((root / "matched_cost" / "rows").glob("*.json"))
    expected = len(driver["visits"]) * count * len(DEPTHS)
    if len(files) != expected:
        raise RuntimeError(f"P19A_MISSING_COST_DEPTH_ROWS {len(files)}/{expected}")
    visits = {v["case_id"]: v for v in driver["visits"]}
    accept_map = {
        (row["case_id"], row["frame"], row["last_committed_depth"]): row
        for row in acceptance
    }
    rows, observed = [], set()
    bycase = defaultdict(set)
    pids = set()
    for file in files:
        row = json.loads(file.read_text())
        case = row.get("case_id")
        frame = row.get("frame")
        depth = row.get("last_committed_depth")
        if case not in visits:
            raise RuntimeError("P19A_UNKNOWN_REAL_REQUEST")
        previous = accept_map.get((case, frame, depth))
        if previous is None or type(frame) is not int or depth not in DEPTHS:
            raise RuntimeError(f"P19A_UNMATCHED_NATIVE_CHECKPOINT {file}")
        visit = visits[case]
        if (row.get("schema") != SCHEMA
                or row.get("source_sha") != src
                or row.get("gpu") != "NVIDIA H200"
                or row.get("pid") != previous.get("pid")
                or row.get("request_hash") != previous.get("request_hash")
                or row.get("prompt_id") != previous.get("prompt_id")
                or row.get("request_nonce") != previous.get("request_nonce")
                or row.get("seed") != visit["seed"]
                or row.get("cache_len") != depth + 1
                or row.get("trajectory") !=
                  "NATIVE_AR_EAGER_CHECKPOINT_NOT_SPEC_ROLLOUT"
                or row.get("cost_metric") !=
                  "CUDA_GRAPH_EVENT_ENVELOPE_FIXED_CHECKPOINT"
                or row.get("matched_native_sampler") !=
                  "QWEN3_TTS_REAL_SEEDED_SUBTALKER"
                or row.get("uniforms") != "P18_SHARED_QP_FIXED"
                or row.get("frames_per_request_measured") != count
                or set(row.get("arms", {})) != {
                    "AR", *(str(w) for w in widths(depth))
                }):
            raise RuntimeError(f"P19A_COST_PROVENANCE_DRIFT: {file}")
        identity = (case, frame, depth)
        if identity in observed:
            raise RuntimeError("P19A_DUPLICATE_MEASURED_NATIVE_STATE")
        observed.add(identity)
        bycase[case].add((frame, depth))
        pids.add(row["pid"])
        for arm, data in row["arms"].items():
            blocks = data.get("cuda_graph_event_ms_blocks")
            value = data.get("median_cuda_event_ms")
            if (data.get("parity") !=
                    "CAPTURED_REPLAY_VS_EAGER_OUTPUT_AND_FULL_KV_PASS"
                    or not isinstance(blocks, list)
                    or len(blocks) != BLOCKS
                    or data.get("replays_per_block") != REPEATS
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0
                    or any(b.get("block") != i or
                           b.get("replays") != REPEATS or
                           type(b.get("cuda_event_ms")) not in (int, float) or
                           not math.isfinite(b["cuda_event_ms"]) or
                           b["cuda_event_ms"] <= 0
                           for i, b in enumerate(blocks))
                    or not math.isclose(
                        value,
                        statistics.median(b["cuda_event_ms"] for b in blocks),
                        rel_tol=1e-12,
                    )):
                raise RuntimeError(f"P19A_INVALID_COST_OR_PARITY {file} arm={arm}")
        rows.append(row)
    if len(pids) != 1:
        raise RuntimeError("P19A_MULTIPLE_WORKER_PID_MIXED")
    for case in visits:
        frames = sorted({
            row["frame"] for row in acceptance if row["case_id"] == case
        })
        expected_states = {
            (f, d) for f in frames[:count] for d in DEPTHS
        }
        if bycase[case] != expected_states:
            raise RuntimeError(f"P19A_COST_CHECKPOINT_COVERAGE_DRIFT {case}")
    return driver, acceptance, rows


def analyze(root: Path, out: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    out = out.expanduser().resolve()
    if out.exists():
        raise RuntimeError("P19A requires absent analysis directory")
    driver, acceptance, cost_rows = validate_costs(root)
    prompts = {v["case_id"]: v["prompt_id"] for v in driver["visits"]}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in cost_rows:
        for arm, metric in row["arms"].items():
            groups[row["last_committed_depth"]][arm][row["case_id"]].append(
                metric["median_cuda_event_ms"]
            )
    result = {
        "status": STATUS,
        "source_sha": driver["source_sha"],
        "mode": driver["mode"],
        "gpu": "NVIDIA H200",
        "worker_pid": cost_rows[0]["pid"],
        "independent_prompt_clusters": len(set(prompts.values())),
        "request_clusters": len(prompts),
        "measured_frames_per_request": cost_rows[0][
            "frames_per_request_measured"
        ],
        "native_ar_step_scope": (
            "real_predictor_forward+head+seeded_sampler+embedding"
        ),
        "spec_round_scope": (
            "P17_batched_drafts+ONE_wide_fused_verifier+accept_bonus+state"
        ),
        "comparison_limit": (
            "INDIVIDUAL_FIXED_CHECKPOINT_CUDA_GRAPH_COMPONENTS_NOT_NATIVE_"
            "FULL_GRAPH_OR_HTTP_SERVING"
        ),
        "depths": {},
    }
    table = [
        "# P19A — Same-source matched checkpoint AR vs wide graph costs",
        "",
        f"**{STATUS}**",
        "",
        f"Source SHA: {result['source_sha']}; worker PID: "
        f"{result['worker_pid']}; GPU: H200",
        f"Prompt clusters: {result['independent_prompt_clusters']}, "
        f"requests: {result['request_clusters']}, "
        f"measured frames/request: {result['measured_frames_per_request']}",
        "",
        "| Last committed depth | Native AR next-step ms | K2 ms | "
        "K3 ms | K4 ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        data = {}
        for arm in ("AR", *(str(w) for w in widths(depth))):
            by_case = dict(groups[depth][arm])
            if set(by_case) != set(prompts):
                raise RuntimeError("P19A_UNPAIRED_COST_CASES")
            stats = hierarchical_ci(
                by_case, prompts, seed=20261123 + 101*depth +
                (0 if arm == "AR" else int(arm)), repeats=1000,
            )
            data[arm] = stats
        result["depths"][str(depth)] = data
        fmt = lambda arm: (
            f"{data[arm]['mean']:.5f}" if arm in data else "—"
        )
        table.append(
            f"| {depth} | {fmt('AR')} | {fmt('2')} | "
            f"{fmt('3')} | {fmt('4')} |"
        )
    table += [
        "",
        "## Strict limits",
        "",
        "- Every row is measured at the SAME real native AR eager "
        "checkpoint and original physical KV prefix, with exact "
        "captured-vs-eager output and KV parity.",
        "- Native AR next-step uses the actual seeded sampler. Each K arm "
        "uses P17 full-round fixed-uniform speculative sampling. Their "
        "separate graph replays are not a causal integrated decoder.",
        "- Each arm includes GPU Graph launch envelope for that specific "
        "component, not full-frame Graph-ON capture.",
        "- First codebook pair, policy changes, cross-depth caching, "
        "launch/control overhead of a complete runtime, HTTP latency "
        "and audio quality remain unmeasured.",
        "- Hierarchical confidence intervals are prompt then seed, "
        "not independent frame resamples. No production speedup claim.",
        "",
    ]
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (out / "summary.md").write_text("\n".join(table))
    print(f"P19A_STATUS={STATUS}")
    print(f"P19A_REPORT={out/'summary.md'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    analyze(a.root, a.out)


if __name__ == "__main__":
    main()

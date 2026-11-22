# SPDX-License-Identifier: Apache-2.0
"""P18B: paired request-cluster adaptive WIDTH cost screen; NOT a serving GO.

Uses P18A native-AR-EAGER all-depth acceptance and the *prior* P17 H200
CUDA-Graph wide-round envelopes at depths 1/9. The latter are not matched
to each P18 depth, so extrapolated values are only hypothesis-generation.
Native AR per-depth Graph-ON cost is UNMEASURED; do not silently use
2.232ms / 14 as a matched baseline, infer HTTP speedup, or ship policy.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.qwen3_tts_cp_spec_k2.p18_analyze import (
    DEPTHS, widths, validate_rows,
)

STATUS = "P18B_PAIRED_REQUEST_WIDTH_COST_SCREEN_NOT_NATIVE_AR_OR_E2E_GO"
P17_STATUS = "P17_K2_K3_K4_REAL_WIDE_CORRECTNESS_AND_COST_SCREEN_NOT_FULL_DECODER"
P17_SHA = "de4075501633a4d5054d820ea2eb13039fb5d8dc"
RESAMPLES = 1000
MIN_IMPROVEMENT = 0.05


def read_p17_cost(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve(strict=True).read_text())
    if (value.get("status") != P17_STATUS
            or value.get("source_sha") != P17_SHA
            or value.get("gpu") != "NVIDIA H200"
            or value.get("native_full_graph_ms", 0) <= 0
            or not isinstance(value.get("real_on_policy_checkpoint_rows"), list)):
        raise RuntimeError("P18B_INVALID_P17_H200_SOURCE_LOCKED_COST")
    rows = value["real_on_policy_checkpoint_rows"]
    if len(rows) != 4 or sorted(
        row.get("last_committed_depth") for row in rows
    ) != [1, 1, 9, 9]:
        raise RuntimeError("P18B_P17_MISSING_FOUR_REAL_CHECKPOINTS")
    costs = {}
    for depth in (1, 9):
        summary = value.get("depth_summary", {}).get(str(depth), {})
        costs[depth] = {}
        for width in (2, 3, 4):
            cell = summary.get(str(width), {})
            median = cell.get("median_event_ms", {}).get("full_round")
            raw = [
                row.get("widths", {}).get(str(width), {})
                .get("cuda_graph_event_ms_medians", {})
                .get("full_round")
                for row in rows if row["last_committed_depth"] == depth
            ]
            if (type(median) not in (int, float) or median <= 0
                    or len(raw) != 2
                    or any(type(x) not in (int, float) or x <= 0 for x in raw)):
                raise RuntimeError("P18B_P17_COST_ENVELOPE_INCOMPLETE")
            costs[depth][width] = {
                "median_ms": float(median),
                "checkpoint_medians_ms": [float(x) for x in raw],
            }
    return {"source_sha": value["source_sha"],
            "native_graph_ms_per_frame_CONTEXT_ONLY": value["native_full_graph_ms"],
            "depths": costs}


def estimated_cost(cost: dict, depth: int, width: int) -> dict:
    """Bounded interpolation between measured P17 positions; tail extrapolated."""
    early = cost["depths"][1][width]
    late = cost["depths"][9][width]
    weight = min(1., max(0., (depth - 1) / 8.))
    ms = early["median_ms"] * (1 - weight) + late["median_ms"] * weight
    history = early["checkpoint_medians_ms"] + late["checkpoint_medians_ms"]
    return {
        "assumed_full_round_ms": ms,
        "p17_checkpoint_range_ms_NOT_CI": [min(history), max(history)],
        "cost_provenance": (
            "P17_ACTUAL_DEPTH" if depth in (1, 9)
            else "P17_INTERPOLATED_UNMATCHED" if depth < 9
            else "P17_TAIL_EXTRAPOLATED_UNMATCHED"
        ),
    }


def paired_bootstrap(
    case_values: dict[str, dict[int, list[int]]],
    costs: dict[int, float], seed: int,
    prompt_of_case: dict[str, str] | None = None,
) -> dict:
    """Paired K2 baseline; optional hierarchical prompt->seed bootstrap."""
    names = sorted(case_values)
    if len(names) < 2 or 2 not in costs:
        raise RuntimeError("P18B_NEEDS_PAIRED_K2_REQUESTS")
    for name in names:
        if set(case_values[name]) != set(costs):
            raise RuntimeError("P18B_UNPAIRED_WIDTHS")
        if len({len(v) for v in case_values[name].values()}) != 1:
            raise RuntimeError("P18B_UNPAIRED_NATIVE_FRAMES")
    per_case = {
        name: {
            width: statistics.mean(case_values[name][width])
            for width in costs
        } for name in names
    }

    def calc(selected: list[str]) -> dict[int, float]:
        means = {
            width: statistics.mean(per_case[name][width] for name in selected)
            for width in costs
        }
        return {
            width: costs[width] / mean for width, mean in means.items()
        }

    original = calc(names)
    rng = random.Random(seed)
    boot = defaultdict(list)
    prompt_groups = defaultdict(list)
    if prompt_of_case is not None:
        if set(prompt_of_case) != set(names):
            raise RuntimeError("P18B_PROMPT_IDENTITY_MISMATCH")
        for name in names:
            prompt_groups[prompt_of_case[name]].append(name)
        if (len(prompt_groups) < 2
                or len({len(v) for v in prompt_groups.values()}) != 1):
            raise RuntimeError("P18B_NEEDS_BALANCED_PROMPT_SEED_CLUSTERS")
    prompts = sorted(prompt_groups)
    for _ in range(RESAMPLES):
        if prompt_of_case is None:
            drawn = [names[rng.randrange(len(names))] for _ in names]
        else:
            sampled = [prompts[rng.randrange(len(prompts))]
                       for _ in prompts]
            drawn = []
            for prompt in sampled:
                members = prompt_groups[prompt]
                drawn += [members[rng.randrange(len(members))]
                          for _ in members]
        surrogate = calc(drawn)
        for width in costs:
            boot[width].append(1 - surrogate[width] / surrogate[2])
    results = {}
    for width, value in original.items():
        diffs = sorted(boot[width])
        low = diffs[int((RESAMPLES - 1) * 0.025)]
        high = diffs[int((RESAMPLES - 1) * 0.975)]
        results[str(width)] = {
            "assumed_ms_per_committed": value,
            "advantage_vs_K2_paired_request_bootstrap": {
                "nominal": 1 - value / original[2],
                "interval_95pct_ACCEPTANCE_ONLY": [low, high],
                "primary_resampling_unit": (
                    "PROMPT_THEN_SEED" if prompt_of_case is not None
                    else "REQUEST_CONDITIONAL_ON_PROMPTS"
                ),
            },
        }
    return results


def analyze(root: Path, p17_report: Path, out: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    out = out.expanduser().resolve()
    if out.exists():
        raise RuntimeError("P18B requires fresh output dir")
    visits, native_rows = validate_rows(root)
    p17 = read_p17_cost(p17_report)
    if len({r["request_hash"] for r in native_rows}) != len(visits["visits"]):
        raise RuntimeError("P18B_REQUEST_HASH_CLUSTER_MISMATCH")
    result = {
        "status": STATUS, "p18_source_sha": visits["source_sha"],
        "p17_cost_source_sha": p17["source_sha"],
        "p17_native_full_predictor_graph_ms_per_frame_CONTEXT_ONLY":
            p17["native_graph_ms_per_frame_CONTEXT_ONLY"],
        "independent_requests": len(visits["visits"]),
        "mode": visits["mode"],
        "cost_model": "P17_WIDTH_ENVELOPE_X_P18_NATIVE_EAGER_ACCEPTANCE",
        "native_AR_step_ms": None,
        "native_AR_comparison_status": "UNMEASURED_NO_AR_VS_SPEC_GO",
        "causal_spec_policy_counterfactual_status":
            "NOT_ESTABLISHED_NATIVE_AR_CHECKPOINTS_ONLY",
        "policy": {},
    }
    groups = defaultdict(list)
    for row in native_rows:
        groups[row["last_committed_depth"]].append(row)
    prompt_of_case = {
        v["case_id"]: v["prompt_id"] for v in visits["visits"]
    }
    for d in DEPTHS:
        possible = widths(d)
        if not possible:
            result["policy"][str(d)] = {
                "legal_widths": [],
                "AR": "MANDATORY_LAST_CODEBOOK_BOUNDARY",
                "best_K_width_nominal": None,
                "robust_K_better_than_K2": [],
            }
            continue
        per_case = defaultdict(lambda: defaultdict(list))
        cost_model = {}
        for w in possible:
            cost_model[w] = estimated_cost(p17, d, w)
            for row in groups[d]:
                per_case[row["case_id"]][w].append(
                    row["widths"][str(w)]["committed_count"]
                )
        paired = paired_bootstrap(
            per_case, {w: cost_model[w]["assumed_full_round_ms"]
                       for w in possible},
            seed=20261123 + 101*d,
            prompt_of_case=prompt_of_case,
        )
        best = min(possible, key=lambda w: (
            paired[str(w)]["assumed_ms_per_committed"], w
        ))
        robust = [
            w for w in possible if w != 2
            and (paired[str(w)]
                 ["advantage_vs_K2_paired_request_bootstrap"]
                 ["interval_95pct_ACCEPTANCE_ONLY"][0]
                 > MIN_IMPROVEMENT)
        ]
        result["policy"][str(d)] = {
            "legal_widths": list(possible),
            "AR": "UNPRICED_REQUIRES_MATCHED_PER_DEPTH_NATIVE_GRAPH_COST",
            "best_K_width_nominal": best,
            "robust_K_better_than_K2": (
                robust if visits["mode"] == "main" else []
            ),
            "smoke_never_selects_policy": visits["mode"] == "smoke",
            "widths": {
                str(w): {**cost_model[w], **paired[str(w)]}
                for w in possible
            },
        }
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(
        result, sort_keys=True, indent=2,
    ) + "\n")
    table = [
        "# P18B — Paired request adaptive-width cost screen",
        "",
        f"**{STATUS}**",
        "",
        f"P18 SHA: {visits['source_sha']}; P17 cost SHA: {p17['source_sha']}",
        f"Independent request clusters: {len(visits['visits'])}; "
        f"mode: {visits['mode']}",
        "",
        "| Last depth | K2 assumed ms/code | K3 assumed ms/code | "
        "K4 assumed ms/code | Nominal best K | Robust >5% vs K2 | "
        "P17 cost provenance |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for d in DEPTHS:
        row = result["policy"][str(d)]
        if not row["legal_widths"]:
            table.append(f"| {d} | — | — | — | AR mandatory | — | — |")
            continue
        ms = lambda w: (
            f"{row['widths'][str(w)]['assumed_ms_per_committed']:.5f}"
            if str(w) in row["widths"] else "—"
        )
        cost_src = row["widths"]["2"]["cost_provenance"]
        robust = (",".join(f"K{w}" for w in row["robust_K_better_than_K2"])
                  or "none")
        table.append(
            f"| {d} | {ms(2)} | {ms(3)} | {ms(4)} | "
            f"K{row['best_K_width_nominal']} | {robust} | {cost_src} |"
        )
    table += [
        "",
        "## Decision",
        "",
        "- **NO PRODUCTION GO**. We have not measured native Graph-ON AR "
        "step costs at each depth, nor a full speculative native decoder.",
        "- The P17 wide-round cost envelope was measured ONLY at depth 1 "
        "and 9 with a DIFFERENT source SHA; all other cost rows are "
        "interpolated or extrapolated assumptions, NOT measurements.",
        "- P18A acceptance was measured on NATIVE AR EAGER trajectories "
        "using exact P17 q/p, static verifier, CPU residual and physical "
        "KV parity. Spec policy would change later checkpoints.",
        "- PRIMARY paired CI resamples PROMPTS then SEEDS, retaining "
        "paired widths and correlated frames. Eight main prompts are "
        "NOT 24 independent linguistic prompt clusters. The CI covers "
        "acceptance, NOT GPU cost variability or causal rollout.",
        "- A width is robust vs K2 only when paired acceptance-only 95% "
        "lower improvement exceeds 5% AND data is from the main cohort. "
        "This is still a research hypothesis, not an optimization GO.",
        "- AR is deliberately unpriced; never compare against the "
        "2.23ms complete native Predictor Graph divided by 14 as a "
        "matched per-code serving baseline.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(table))
    print(f"P18B_STATUS={STATUS}")
    print(f"P18B_REPORT={out/'summary.md'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--p17-cost-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    analyze(a.root, a.p17_cost_report, a.out)


if __name__ == "__main__":
    main()

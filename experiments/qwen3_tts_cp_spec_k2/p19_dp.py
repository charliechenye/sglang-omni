# SPDX-License-Identifier: Apache-2.0
"""P19B: depth-transition Bellman screen using SAME-run P18 acceptance/P19 costs.

Crucial distinctions:
 * A K-wide round commits 1..K codes; it skips intermediate depth states.
 * Native AR at depth d always commits exactly ONE next codebook code.
 * Acceptance is from native-AR trajectories; the state distribution under
   a speculative rollout is NOT observed. Thus the Bellman model is a
   stationary counterfactual screen, NOT causal performance evidence.
 * All costs are separate captured CUDA Graph *components*, not a fused
   whole frame, and the initial native pair is excluded from BOTH arms.
 * Bootstrap samples PROMPTS, then SEEDS, preserving all frames/arms
   from each request and using the fixed nominal policy for uncertainty.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.qwen3_tts_cp_spec_k2.p18_analyze import (
    DEPTHS, widths,
)
from experiments.qwen3_tts_cp_spec_k2.p19_analyze import (
    STATUS as COST_PASS, validate_costs,
)

RESULT = "P19B_ALL_DEPTH_MATCHED_COMPONENT_DP_SCREEN_NOT_E2E_GO"
RESAMPLES = 1000
TERMINAL = 15
CANDIDATE_THRESHOLD = 0.15


def _check_inputs(costs: dict, probs: dict) -> None:
    if set(costs) != set(DEPTHS) or set(probs) != set(DEPTHS):
        raise ValueError("P19_DP_REQUIRES_DEPTHS_1_THROUGH_14")
    for depth in DEPTHS:
        legal = {"AR", *(str(w) for w in widths(depth))}
        if set(costs[depth]) != legal or set(probs[depth]) != {
            str(w) for w in widths(depth)
        }:
            raise ValueError(f"P19_DP_INVALID_ACTION_SET_DEPTH_{depth}")
        for arm, ms in costs[depth].items():
            if type(ms) not in (float, int) or not math.isfinite(ms) or ms <= 0:
                raise ValueError(f"P19_DP_INVALID_COST_D{depth}_{arm}")
        for sw, freq in probs[depth].items():
            width = int(sw)
            if (len(freq) != width or any(
                not isinstance(x, (float, int)) or not math.isfinite(x)
                or x < 0 for x in freq
            ) or not math.isclose(sum(freq), 1., abs_tol=1e-8)):
                raise ValueError(f"P19_DP_BAD_PREFIX_DISTRIBUTION_D{depth}_K{width}")


def bellman(costs: dict, probs: dict,
            policy: dict[int, str] | None = None) -> dict:
    """Backward DP on last committed depth d; terminal after code15.

    If policy is None, optimize across AR/K2/K3/K4 at each depth.
    Otherwise evaluate EXACT same frozen policy (no re-selection).
    """
    _check_inputs(costs, probs)
    if policy is not None and set(policy) != set(DEPTHS):
        raise ValueError("P19_DP_FIXED_POLICY_MISSING_DEPTHS")
    future = {TERMINAL: 0.0}
    chosen = {}
    breakdown = {}
    for depth in reversed(DEPTHS):
        actions = {}
        for arm in ("AR", *(str(w) for w in widths(depth))):
            if arm == "AR":
                downstream = future[depth + 1]
            else:
                downstream = sum(
                    frequency * future[depth + advance]
                    for advance, frequency in enumerate(probs[depth][arm], 1)
                )
            actions[arm] = costs[depth][arm] + downstream
        if policy is None:
            # Stable tie-break: native AR first, then K2, K3 and K4.
            arm = min(actions, key=actions.__getitem__)
        else:
            arm = policy[depth]
            if arm not in actions:
                raise ValueError(f"P19_ILLEGAL_FIXED_POLICY_AT_DEPTH_{depth}")
        future[depth] = actions[arm]
        chosen[depth] = arm
        breakdown[depth] = {
            "chosen": arm, "v_ms": future[depth],
            "action_q_ms": actions,
        }
    return {
        "residual_component_graph_ms": future[1],
        "policy": chosen,
        "per_depth": breakdown,
    }


def fixed_width_policy(width: int) -> dict[int, str]:
    if width not in (2, 3, 4):
        raise ValueError("P19_FIXED_WIDTH_MUST_BE_2_TO_4")
    return {
        depth: (str(width) if width in widths(depth) else "AR")
        for depth in DEPTHS
    }


def _ci(values: list[float]) -> list[float]:
    ordered = sorted(values)
    if len(ordered) < 100:
        raise ValueError("P19_DP_MIN_BOOTSTRAP_DRAWS_100")
    return [
        ordered[int(0.025 * (len(ordered)-1))],
        ordered[int(0.975 * (len(ordered)-1))],
    ]


def _case_tables(driver: dict, accept_rows: list[dict],
                 cost_rows: list[dict]) -> tuple[dict, dict, dict]:
    visits = {v["case_id"]: v for v in driver["visits"]}
    prompt_map = {case: v["prompt_id"] for case, v in visits.items()}
    costs_per_case = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prefixes_per_case = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in accept_rows:
        case, depth = row["case_id"], row["last_committed_depth"]
        for width in widths(depth):
            prefixes_per_case[case][depth][str(width)].append(
                int(row["widths"][str(width)]["committed_count"])
            )
    for row in cost_rows:
        case, depth = row["case_id"], row["last_committed_depth"]
        for arm, metric in row["arms"].items():
            costs_per_case[case][depth][arm].append(
                metric["median_cuda_event_ms"]
            )
    per_case = {}
    for case in sorted(visits):
        per_case[case] = {}
        for depth in DEPTHS:
            expected_costs = {"AR", *(str(w) for w in widths(depth))}
            native_costs = costs_per_case[case][depth]
            native_probs = prefixes_per_case[case][depth]
            # Depth 14 has NO legal speculative width. Its native_probs
            # must be empty, while native AR cost must still be present.
            # Do not require a non-empty speculative histogram at this
            # terminal-adjacent state. Every earlier depth keeps strict
            # non-empty and balanced acceptance/cost checks.
            if (set(native_costs) != expected_costs
                    or set(native_probs) != expected_costs - {"AR"}
                    or len({len(v) for v in native_costs.values()}) != 1
                    or any(not values for values in native_costs.values())
                    or (bool(native_probs) and (
                        len({len(v) for v in native_probs.values()}) != 1
                        or any(not values for values in native_probs.values())
                    ))):
                raise RuntimeError("P19_DP_UNPAIRED_REAL_NATIVE_CASES")
            per_case[case][depth] = {
                "cost": {a: statistics.mean(v)
                         for a, v in native_costs.items()},
                "counts": {
                    sw: [int(v.count(j)) for j in range(1, int(sw) + 1)]
                    for sw, v in native_probs.items()
                },
            }
    prompts = defaultdict(list)
    for case, prompt in prompt_map.items():
        prompts[prompt].append(case)
    if (len(prompts) not in (2, 8)
            or len({len(x) for x in prompts.values()}) != 1):
        raise RuntimeError("P19_UNBALANCED_PROMPT_SEED_COHORT")
    return per_case, prompts, prompt_map


def _aggregate(per_case: dict, selected: list[str]) -> tuple[dict, dict]:
    """Each selected request contributes one cost value and one frame histogram."""
    if not selected:
        raise ValueError("P19_DP_NO_CLUSTER_DRAW")
    cost, probability = {}, {}
    for depth in DEPTHS:
        arms = {"AR", *(str(w) for w in widths(depth))}
        cost[depth] = {
            a: statistics.mean(
                per_case[case][depth]["cost"][a] for case in selected
            ) for a in arms
        }
        probability[depth] = {}
        for width in widths(depth):
            sw = str(width)
            freqs = [
                (per_case[case][depth]["counts"][sw][j] /
                 sum(per_case[case][depth]["counts"][sw]))
                for case in selected
                for j in range(width)
            ]
            # Reshape grouped counts per request: the same case may be
            # resampled multiple times, but its frames stay together.
            probability[depth][sw] = [
                statistics.mean(freqs[j::width]) for j in range(width)
            ]
    return cost, probability


def _hierarchical_draw(prompts: dict, rng: random.Random) -> list[str]:
    names = sorted(prompts)
    picked = [names[rng.randrange(len(names))] for _ in names]
    result = []
    for prompt in picked:
        cases = sorted(prompts[prompt])
        result.extend(cases[rng.randrange(len(cases))] for _ in cases)
    return result


def analyze(root: Path, out: Path, *, smoke_allow_dp: bool = True) -> dict:
    root = root.expanduser().resolve(strict=True)
    out = out.expanduser().resolve()
    if out.exists():
        raise RuntimeError("P19B requires absent result directory")
    driver, accept_rows, cost_rows = validate_costs(root)
    report_file = root / "matched_cost" / "analysis" / "report.json"
    if not report_file.is_file():
        raise RuntimeError("P19B_REQUIRES_P19A_VALIDATED_COST_REPORT")
    cost_attest = json.loads(report_file.read_text())
    if (cost_attest.get("status") != COST_PASS
            or cost_attest.get("source_sha") != driver["source_sha"]
            or cost_attest.get("worker_pid") != cost_rows[0]["pid"]):
        raise RuntimeError("P19B_P19A_ATTESTATION_MISMATCH")
    if driver["mode"] == "smoke" and not smoke_allow_dp:
        raise RuntimeError("P19B_SMOKE_NO_POLICY_SELECTION")
    per_case, prompts, _ = _case_tables(driver, accept_rows, cost_rows)
    case_ids = sorted(per_case)
    costs, probs = _aggregate(per_case, case_ids)
    optimized = bellman(costs, probs)
    base_policy = {d: "AR" for d in DEPTHS}
    native = bellman(costs, probs, base_policy)
    fixed = {
        f"K{w}": bellman(costs, probs, fixed_width_policy(w))
        for w in (2, 3, 4)
    }
    nominal_policy = optimized["policy"]
    baseline = native["residual_component_graph_ms"]
    nominal_ms = optimized["residual_component_graph_ms"]
    nominal_saving = 1 - nominal_ms / baseline
    rng = random.Random(20261123)
    saving_boot, native_boot, adaptive_boot = [], [], []
    best_case_optimistic = []
    action_freq = defaultdict(Counter)
    for _ in range(RESAMPLES):
        draw = _hierarchical_draw(prompts, rng)
        cost_b, prob_b = _aggregate(per_case, draw)
        n = bellman(cost_b, prob_b, base_policy)
        # CRITICAL: evaluate the frozen nominal policy, not a new
        # overfit minimum chosen on each resample.
        a = bellman(cost_b, prob_b, nominal_policy)
        optimistic = bellman(cost_b, prob_b)
        n_ms = n["residual_component_graph_ms"]
        a_ms = a["residual_component_graph_ms"]
        native_boot.append(n_ms)
        adaptive_boot.append(a_ms)
        saving_boot.append(1 - a_ms / n_ms)
        best_case_optimistic.append(
            1 - optimistic["residual_component_graph_ms"] / n_ms
        )
        for depth, arm in optimistic["policy"].items():
            action_freq[depth][arm] += 1

    report = {
        "status": RESULT,
        "mode": driver["mode"],
        "source_sha": driver["source_sha"],
        "gpu": "NVIDIA H200",
        "independent_prompt_clusters": len(prompts),
        "conditional_request_clusters": len(case_ids),
        "bootstrap": {
            "resamples": RESAMPLES,
            "method": "hierarchical_prompt_then_seed_paired_cost_and_acceptance",
            "fixed_policy_for_CI": True,
            "cost_variability": "per_request_checkpoint_only_not_deployment",
        },
        "reference": {
            "scope": "RESIDUAL_DEPTHS_1_TO_14_ONLY",
            "terminal_after_code": 15,
            "initial_pair_included": False,
            "native_whole_predictor_graph_compared": False,
            "native_AR_only_component_graph_ms": baseline,
            "native_AR_only_bootstrap_ms_CI": _ci(native_boot),
        },
        "nominal_adaptive": {
            "residual_component_graph_ms": nominal_ms,
            "nominal_improvement_vs_native_AR_components": nominal_saving,
            "fixed_policy_bootstrap_improvement_CI": _ci(saving_boot),
            "fixed_policy_bootstrap_ms_CI": _ci(adaptive_boot),
            "policy_by_last_committed_depth": {
                str(d): nominal_policy[d] for d in DEPTHS
            },
            "depth_q": {
                str(d): optimized["per_depth"][d] for d in DEPTHS
            },
        },
        "fixed_widths_with_AR_at_illegal_tail": {
            label: {
                "residual_component_graph_ms": value[
                    "residual_component_graph_ms"
                ],
                "improvement_vs_native_AR_components": 1 -
                value["residual_component_graph_ms"] / baseline,
            } for label, value in fixed.items()
        },
        "reselected_per_bootstrap_optimistic_upper_bound_only": {
            "improvement_CI": _ci(best_case_optimistic),
            "action_counts_per_depth": {
                str(d): dict(action_freq[d]) for d in DEPTHS
            },
        },
        "research_decision": (
            "SMOKE_NO_SELECTION" if driver["mode"] == "smoke"
            else "FOLLOWUP_CAUSAL_ROLLOUT_REQUIRED"
            if _ci(saving_boot)[0] > CANDIDATE_THRESHOLD
            else "NO_STRONG_COMPONENT_HEADROOM"
        ),
        "limits": [
            "State distribution from native AR, not deployed speculative rollout",
            "Independent frozen-checkpoint CUDA Graph components are not additive production GPU execution",
            "First two codebook tokens and full native Predictor Graph ON are NOT priced",
            "No integrated graph-control or cross-depth runtime transition cost",
            "No CUDA Graph memory/startup, HTTP QPS, voice quality or production speedup",
            "Fixed nominal policy evaluated on bootstrap is in-sample and not independent holdout",
            "Conditional on 8 prompt clusters in main; no general-population claim",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    table = [
        "# P19B — All-depth speculative depth-transition DP",
        "",
        f"**{RESULT}**",
        "",
        f"Source SHA: {report['source_sha']}",
        f"Mode: {report['mode']}; prompt clusters: {len(prompts)}; "
        f"request clusters: {len(case_ids)}",
        "",
        "## Residual component cost model (NOT whole native predictor)",
        "",
        f"Native AR-only residual: **{baseline:.5f} ms**",
        f"Adaptive residual (DP nominal): **{nominal_ms:.5f} ms**",
        f"Nominal component headroom vs AR: **{nominal_saving:.1%}**",
        f"Frozen-policy hierarchical 95% CI of headroom: "
        f"**[{_ci(saving_boot)[0]:.1%}, {_ci(saving_boot)[1]:.1%}]**",
        "",
        "| Last committed depth | Selected action | AR Q(ms) | "
        "K2 Q(ms) | K3 Q(ms) | K4 Q(ms) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        choices = optimized["per_depth"][depth]["action_q_ms"]
        fmt = lambda key: (
            f"{choices[key]:.5f}" if key in choices else "—"
        )
        table.append(
            f"| {depth} | {nominal_policy[depth]} | {fmt('AR')} | "
            f"{fmt('2')} | {fmt('3')} | {fmt('4')} |"
        )
    table += [
        "",
        "## Methodology / NO-GO",
        "",
        "- Bellman V(d) = min[action cost(d) + Σ P(committed=j|d,arm) V(d+j)].",
        "- AR advances exactly one code, K2/3/4 can advance 1..K; "
        "illegal tail widths are not selected.",
        "- Costs from P19A same-source same-worker Graph component captures; "
        "acceptance from paired P18 native-eager checkpoints in this run.",
        "- Bootstrap preserves PROMPT and SEED clusters and all correlated "
        "frames/widths; policy selection is in-sample and optimistic.",
        "- No causal rollout: speculative commit changes later KV states "
        "and future acceptance distributions.",
        "- Native complete Predictor Graph ON 2.232ms/frame is NOT "
        "a scope-matched comparator. No meaningful production % speedup.",
        "",
        f"Research decision: **{report['research_decision']}**",
        "",
    ]
    (out / "summary.md").write_text("\n".join(table))
    print(f"P19B_STATUS={RESULT}")
    print(f"P19B_REPORT={out/'summary.md'}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    analyze(a.root, a.out)


if __name__ == "__main__":
    main()

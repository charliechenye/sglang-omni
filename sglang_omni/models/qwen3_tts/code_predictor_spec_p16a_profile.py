# SPDX-License-Identifier: Apache-2.0
"""P16A: matched conditional-vs-unconditional two-round GPU branch census.

Runs inside the P15 source-locked on-policy probe, AFTER all 16 P15 eager
oracle cases have passed. The "always_compute_mask" graph executes both
bonus LM heads but masks rejected bonuses, preserving the same outputs.
Four graph objects are captured:
  conditional/unmarked = original P15 (timing arm)
  unconditional/unmarked = matched baseline (timing arm)
  conditional/marked and unconditional/marked = Nsight-only trace arms.
Trace is bounded by NVTX P16A_TRACE with named per-cell replay ranges.

Only the unmarked graphs are timed. Marked graphs provide independent
physical branch activity, never a performance number. Nsight trace evidence
must be checked separately before asserting actual bonus-head kernel skip.
"""
from __future__ import annotations

import math
import statistics

import torch
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    CUDAGraphCaptureControlFlowOpDispatchMode,
)

from .code_predictor_spec_p15_two_round import (
    BRANCHES, CASES, ROUND_FIELDS, two_gpu_rounds,
)
from .code_predictor_spec_p16a_markers import mark_round, mark_bonus
from .code_predictor_spec_p7_dispatch import compare

SCHEMA = "qwen3tts-cpspec-p16a-branch-cost-and-nsys-node-trace-v1"
STATUS = "P16A_BRANCH_COST_AND_MARKED_REPLAYS_COMPLETE_NSYS_AUDIT_REQUIRED"
ARMS = ("conditional", "unconditional")
TRIAL_ORDERS = (
    ("conditional", "unconditional"),
    ("unconditional", "conditional"),
    ("unconditional", "conditional"),
    ("conditional", "unconditional"),
)
TIMED_REPLAYS = 16
TRACE_PREFIX = "P16A_CASE_"


def expected_bonus_markers(transition: str, arm: str) -> int:
    if transition not in BRANCHES or arm not in ARMS:
        raise ValueError("P16A invalid transition/arm")
    return transition.count("A") if arm == "conditional" else 2


def _freeze_outputs(outputs):
    return tuple({
        name: tensor.detach().clone()
        for name, tensor in zip(ROUND_FIELDS, row)
    } for row in outputs)


def _check_matched(
    a_outputs, b_outputs, a_k, a_v, b_k, b_v, *,
    prefix_k, prefix_v,
):
    strict_fields = (
        "draft", "committed", "accepted", "bonus",
        "next_cache_len", "next_depth", "bonus_executed",
    )
    for idx, (a, b) in enumerate(zip(a_outputs, b_outputs), start=1):
        for key in strict_fields:
            if not torch.equal(a[key], b[key]):
                raise RuntimeError(
                    f"P16A_ROUND{idx}_{key.upper()}_TOKEN_OR_LOGICAL_STATE_FAIL"
                )
        for key in (
            "q_logits", "p_logits", "q_probs", "p_probs",
            "next_hidden", "next_embedding", "next_sum",
        ):
            if not compare(a[key], b[key])["close"]:
                raise RuntimeError(
                    f"P16A_ROUND{idx}_{key.upper()}_MATCHED_PARITY_FAIL"
                )
        for key in ("q_ids", "p_ids"):
            if not torch.equal(a[key], b[key]):
                raise RuntimeError(f"P16A_ROUND{idx}_{key.upper()}_SUPPORT_FAIL")
        for key in ("q_probs", "p_probs"):
            if not torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-6):
                raise RuntimeError(
                    f"P16A_ROUND{idx}_{key.upper()}_DENSE_1E6_FAIL"
                )
    for key, x, y, original in (
        ("K", a_k, b_k, prefix_k),
        ("V", a_v, b_v, prefix_v),
    ):
        if not compare(x, y)["close"]:
            raise RuntimeError(f"P16A_FULL_PHYSICAL_{key}_KV_PARITY_FAIL")
        if not torch.equal(x[:, :, :10], original[:, :, :10]):
            raise RuntimeError(f"P16A_COMMITTED_PREFIX_{key}_MUTATED")


@torch.no_grad()
def profile_branch_costs(
    *, model, conditional_graph, conditional_outputs, stream,
    uniforms_gpu: torch.Tensor, reference_cases: list[dict],
    restore, snapshot, depth: int, cache_start: torch.Tensor,
    indices: torch.Tensor, hidden: torch.Tensor,
    embedding: torch.Tensor, summed: torch.Tensor,
    temperature: float, top_p: float,
    original_k: torch.Tensor, original_v: torch.Tensor,
) -> dict:
    """Measure unmarked arm pairs and emit 8 marked trace ranges."""
    dev = hidden.device
    if len(reference_cases) != 16 or len(CASES) != 16:
        raise RuntimeError("P16A_REQUIRES_ALL_16_INDEPENDENT_P15_CASES")
    chosen = {}
    for case in reference_cases:
        label = case["transition"]
        if label not in BRANCHES:
            raise RuntimeError("P16A_INVALID_ORACLE_TRANSITION")
        chosen.setdefault(label, case["case_index"])
    if set(chosen) != set(BRANCHES):
        raise RuntimeError(
            f"P16A_REQUIRES_ALL_FOUR_BRANCHES: got={sorted(chosen)}"
        )

    # Markers JIT-warmed before graph capture; do not profile compilation.
    with torch.inference_mode():
        mark_round(hidden)
        mark_bonus(hidden)
    torch.cuda.synchronize(dev)

    def capture_variant(*, bonus_policy: str, trace_markers: bool):
        restore()
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(
                graph, stream=stream, capture_error_mode="thread_local",
            ),
            CUDAGraphCaptureControlFlowOpDispatchMode(),
            torch.inference_mode(),
        ):
            outputs = two_gpu_rounds(
                model, first_depth=depth,
                cache_len_gpu=cache_start,
                hidden=hidden, last_embedding=embedding,
                accumulated=summed, uniforms=uniforms_gpu,
                indices_gpu=indices,
                temperature=temperature, top_p=top_p,
                bonus_policy=bonus_policy,
                trace_markers=trace_markers,
            )
        torch.cuda.synchronize(dev)
        return graph, outputs

    g_uncond, out_uncond = capture_variant(
        bonus_policy="always_compute_mask", trace_markers=False,
    )
    g_cond_marked, out_cond_marked = capture_variant(
        bonus_policy="conditional", trace_markers=True,
    )
    g_uncond_marked, out_uncond_marked = capture_variant(
        bonus_policy="always_compute_mask", trace_markers=True,
    )
    holders = {
        "conditional": (conditional_graph, conditional_outputs),
        "unconditional": (g_uncond, out_uncond),
        "conditional_marked": (g_cond_marked, out_cond_marked),
        "unconditional_marked": (g_uncond_marked, out_uncond_marked),
    }
    # Verify all 16: P15 already validated the original conditional
    # against an independent eager oracle. Each new arm MUST match it
    # in code/sampler/hidden/sum AND all physical K/V.
    for case in reference_cases:
        uniforms_gpu.copy_(torch.tensor(
            CASES[case["case_index"]], device=dev, dtype=torch.float32,
        ))
        recorded = None
        for name in holders:
            graph, outputs = holders[name]
            restore()
            graph.replay()
            torch.cuda.synchronize(dev)
            row = (_freeze_outputs(outputs), *snapshot())
            if recorded is None:
                recorded = row
            else:
                _check_matched(
                    recorded[0], row[0],
                    recorded[1], recorded[2], row[1], row[2],
                    prefix_k=original_k, prefix_v=original_v,
                )
    # Every performance sample resets the same original prefix. No
    # Nsight markers or CPU oracle are included in the timed intervals.
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "all_16_matched_against_independent_P15_oracle": True,
        "graph_count_total": 4,
        "timed_graphs_unmarked": True,
        "trace_graphs_marked_not_timed": True,
        "branch_case_indices": chosen,
        "cases": {},
        "trace_ranges": [],
        "nsys_node_trace_verified": False,
        "bonus_head_kernel_skip_proven": False,
        "native_seeded_rng_verified": False,
        "timing_kind": "two-round CUDA event envelope only; not HTTP QPS",
    }
    for label in ("AA", "AR", "RA", "RR"):
        idx = chosen[label]
        uniforms_gpu.copy_(
            torch.tensor(CASES[idx], device=dev, dtype=torch.float32)
        )
        times = {arm: [] for arm in ARMS}
        for order in TRIAL_ORDERS:
            for arm in order:
                restore()
                graph = holders[arm][0]
                for _ in range(2):
                    graph.replay()
                torch.cuda.synchronize(dev)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(TIMED_REPLAYS):
                    graph.replay()
                end.record()
                end.synchronize()
                measured = begin.elapsed_time(end) / TIMED_REPLAYS
                if not math.isfinite(measured) or measured <= 0:
                    raise RuntimeError("P16A_INVALID_CUDA_EVENT_BLOCK")
                times[arm].append(measured)
        med = {arm: statistics.median(xs) for arm, xs in times.items()}
        result["cases"][label] = {
            "case_index": idx,
            "uniforms": [list(v) for v in CASES[idx]],
            "accept_count": label.count("A"),
            "rejection_count": label.count("R"),
            "cuda_event_ms_blocks": times,
            "median_cuda_event_ms": med,
            "unconditional_minus_conditional_ms": (
                med["unconditional"] - med["conditional"]
            ),
        }

    # This range is intentionally bounded and must be captured by Nsight
    # launched with --capture-range=nvtx --nvtx-capture=P16A_TRACE and
    # NSYS_NVTX_PROFILER_REGISTER_ONLY=0. Each range encloses exactly ONE
    # marked graph replay; both arches receive the same fixed uniforms.
    torch.cuda.nvtx.range_push("P16A_TRACE")
    try:
        for label in ("AA", "AR", "RA", "RR"):
            idx = chosen[label]
            uniforms_gpu.copy_(torch.tensor(
                CASES[idx], device=dev, dtype=torch.float32,
            ))
            for arm in ARMS:
                restore()
                torch.cuda.synchronize(dev)
                name = f"{TRACE_PREFIX}{label}_{arm.upper()}"
                graph = holders[f"{arm}_marked"][0]
                torch.cuda.nvtx.range_push(name)
                try:
                    graph.replay()
                    torch.cuda.synchronize(dev)
                finally:
                    torch.cuda.nvtx.range_pop()
                result["trace_ranges"].append({
                    "nvtx_range": name,
                    "transition": label,
                    "arm": arm,
                    "expected_round_start_marker_nodes": 2,
                    "expected_bonus_head_marker_nodes": (
                        expected_bonus_markers(label, arm)
                    ),
                    "marked_replays": 1,
                })
    finally:
        torch.cuda.nvtx.range_pop()
    restore()
    return result

# SPDX-License-Identifier: Apache-2.0
"""P11A: same-GPU graph replay attribution of real Qwen3-TTS K2 sampling.

Measure the verified sparse top-50 sampler's two torch.topk calls, Triton
post-topk proposal/accept/residual kernel, private CUDA RNG, and entire
sparse+RNG graph independently, using ACTUAL on-policy C1 p/q logits at
depths 2 and 10. Compare against the same-process P10B dense reference.
Never replace live K2 output, request RNG, KV, or serving path.

All timing is CUDA-event graph envelopes on frozen real logits, NOT a full
on-policy GPU K2 serving performance result. Components are intentionally
NOT assumed additive: graph launches and allocator dependencies overlap.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import traceback
from pathlib import Path

import torch

from .code_predictor_spec import sampling_probs
from .code_predictor_spec_p10b_sparse_sampler import (
    BLOCK, K, capture_real_sparse_sampler, sparse_accept_residual_kernel,
)

ENV = "SGLANG_OMNI_QTTS_CP_P11A_DIR"
SCHEMA = "qwen3tts-cpspec-p11a-real-sampler-cost-attribution-v1"
CELLS = ("topk_pair", "post_topk_triton", "private_rng", "full_sparse")
ORDERS = (
    CELLS,
    tuple(reversed(CELLS)),
    ("full_sparse", "private_rng", "topk_pair", "post_topk_triton"),
    ("post_topk_triton", "topk_pair", "private_rng", "full_sparse"),
)


def topk_pair(
    p: torch.Tensor, q: torch.Tensor, temperature: float,
):
    """The same two sorted topk(50) calls as the P10B serving probe."""
    return (
        *torch.topk(p.float() / temperature, K, dim=-1, sorted=True),
        *torch.topk(q.float() / temperature, K, dim=-1, sorted=True),
    )


def post_topk(
    precomputed: tuple[torch.Tensor, ...],
    uniforms: torch.Tensor,
    top_p: float,
):
    """Identical fused Triton kernel to P10B, on persistent topk outputs."""
    pv, pi, qv, qi = precomputed
    device = pv.device
    code = torch.empty((1,), device=device, dtype=torch.long)
    draft = torch.empty_like(code)
    accepted = torch.empty((1,), device=device, dtype=torch.bool)
    prob_p = torch.empty((BLOCK,), device=device, dtype=torch.float32)
    prob_q = torch.empty_like(prob_p)
    mass = torch.empty((1,), device=device, dtype=torch.float32)
    sparse_accept_residual_kernel[(1,)](
        pv, pi, qv, qi, uniforms,
        code, draft, accepted, prob_p, prob_q, mass,
        TOP_P=top_p, K_COUNT=K, B=BLOCK,
        num_warps=8, enable_fp_fusion=False,
    )
    return code, draft, accepted, prob_p, prob_q, mass


def capture_real_sampler_attribution(
    *,
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repeats: int = 64,
) -> dict:
    if (
        logits_p.shape != (1, 2048)
        or logits_q.shape != logits_p.shape
        or not logits_p.is_cuda or not logits_q.is_cuda
        or logits_p.dtype not in (torch.bfloat16, torch.float32)
        or logits_q.dtype != logits_p.dtype
        or top_k != K
        or not 16 <= repeats <= 256
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("P11A needs real CUDA [1,2048] p/q topk50, outside capture")
    device = logits_p.device
    p, q = logits_p.detach().clone(), logits_q.detach().clone()
    original_p, original_q = p.clone(), q.clone()
    # Reuse the already proven P10B parity and same-worker dense-vs-sparse
    # control as a strict prerequisite before costing the individual stages.
    prior = capture_real_sparse_sampler(
        logits_p=p, logits_q=q,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repeats=repeats,
    )
    if (
        not prior.get("status", "").startswith(
            "P10B_REAL_PQ_SPARSE_TOPK_SPEC_SAMPLER_PARITY_PASS"
        )
        or not prior.get("private_rng_replay_unique")
        or not all(c.get("pass") is True for c in prior.get("checks", []))
    ):
        raise RuntimeError("P11A P10B component parity prerequisite failed")

    uniforms = torch.tensor(
        [[.125, .250, .375]], device=device, dtype=torch.float32
    )
    prepared = topk_pair(p, q, temperature)
    p_ref = sampling_probs(
        p, temperature=temperature, top_k=top_k, top_p=top_p
    )
    q_ref = sampling_probs(
        q, temperature=temperature, top_k=top_k, top_p=top_p
    )

    def topk_cell():
        return topk_pair(p, q, temperature)

    def post_cell():
        return post_topk(prepared, uniforms, top_p)

    rng1 = torch.Generator(device=device).manual_seed(20261120)
    rng2 = torch.Generator(device=device).manual_seed(20261121)

    def rng_cell():
        return torch.rand(
            (1, 3), generator=rng1, device=device, dtype=torch.float32
        )

    def full_cell():
        sampled = torch.rand(
            (1, 3), generator=rng2, device=device, dtype=torch.float32
        )
        ranks = topk_pair(p, q, temperature)
        return sampled, post_topk(ranks, sampled, top_p)

    funcs = {
        "topk_pair": topk_cell,
        "post_topk_triton": post_cell,
        "private_rng": rng_cell,
        "full_sparse": full_cell,
    }
    # Warm up outside capture to prevent Triton JIT inside cudaStreamCapture.
    with torch.inference_mode():
        for fn in funcs.values():
            for _ in range(3):
                fn()
    torch.cuda.synchronize(device)

    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    captured: dict[str, tuple[torch.cuda.CUDAGraph, object]] = {}
    for name, fn in funcs.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            g, stream=side, capture_error_mode="thread_local"
        ), torch.inference_mode():
            output = fn()
        captured[name] = g, output
    for g, _ in captured.values():
        g.replay()
    torch.cuda.synchronize(device)

    top_vals, top_idx, q_vals, q_idx = captured["topk_pair"][1]
    expected_top = topk_pair(p, q, temperature)
    for actual, target in zip(
        (top_vals, top_idx, q_vals, q_idx), expected_top
    ):
        torch.testing.assert_close(actual, target, atol=0, rtol=0)

    code, draft, accepted, rank_p, rank_q, mass = captured["post_topk_triton"][1]
    recon_p = torch.zeros_like(p_ref).scatter(
        -1, prepared[1], rank_p[:K].view(1, K)
    )
    recon_q = torch.zeros_like(q_ref).scatter(
        -1, prepared[3], rank_q[:K].view(1, K)
    )
    if not (
        torch.allclose(recon_p, p_ref, rtol=1e-6, atol=1e-6)
        and torch.allclose(recon_q, q_ref, rtol=1e-6, atol=1e-6)
    ):
        raise RuntimeError("P11A post-topk isolated graph distribution mismatch")
    if not math.isfinite(float(mass.item())):
        raise RuntimeError("P11A invalid residual mass")

    # RNG-only and full-graph RNG must advance with CUDA graph replay.
    samples = []
    for _ in range(4):
        captured["private_rng"][0].replay()
        captured["full_sparse"][0].replay()
        torch.cuda.synchronize(device)
        samples.append((
            tuple(captured["private_rng"][1].flatten().cpu().tolist()),
            tuple(captured["full_sparse"][1][0].flatten().cpu().tolist()),
        ))
    if (
        len(set(a for a, _ in samples)) != 4
        or len(set(b for _, b in samples)) != 4
    ):
        raise RuntimeError("P11A private graph RNG replay did not advance")

    def measure(graph: torch.cuda.CUDAGraph) -> float:
        for _ in range(6):
            graph.replay()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / repeats
        if not math.isfinite(ms) or ms <= 0:
            raise RuntimeError("P11A invalid measured CUDA event")
        return ms

    blocks = {cell: [] for cell in CELLS}
    for order in ORDERS:
        for cell in order:
            blocks[cell].append(measure(captured[cell][0]))
    measured = {
        cell: {
            "block_cuda_event_ms": blocks[cell],
            "median_cuda_event_ms": statistics.median(blocks[cell]),
        }
        for cell in CELLS
    }
    full_ms = measured["full_sparse"]["median_cuda_event_ms"]
    p10_sparse = prior["sparse_fused_cuda_graph_ms"]
    if (
        not torch.equal(p, original_p) or not torch.equal(q, original_q)
    ):
        raise RuntimeError("P11A mutated persistent source logits")
    return {
        "schema": SCHEMA,
        "status": "P11A_REAL_PQ_TOPK_POST_RNG_ATTRIBUTION_PASS_NOT_E2E",
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "gpu": torch.cuda.get_device_name(device),
        "vocab": p.shape[-1],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "overlap": prior["overlap"],
        "repeat_count_per_block": repeats,
        "measurements": measured,
        "topk_to_full_envelope_ratio": (
            measured["topk_pair"]["median_cuda_event_ms"] / full_ms
        ),
        "post_to_full_envelope_ratio": (
            measured["post_topk_triton"]["median_cuda_event_ms"] / full_ms
        ),
        "p10b_same_worker_sparse_graph_ms": p10_sparse,
        "p10b_same_worker_dense_graph_ms": (
            prior["p9b_dense_reference_cuda_graph_ms"]
        ),
        "p10b_numerical_prerequisite": prior["status"],
        "private_rng_independent_unique_replays": True,
        "torch_reference_distribution_parity": True,
        "original_pq_unchanged": True,
        "limitations": [
            "Independent graph timings do not decompose additively",
            "Topk pair uses two PyTorch topk(50), not a dual-topk Triton kernel",
            "Input is frozen actual on-policy p/q at selected depth",
            "Private RNG is not the SGLang native seeded sampler RNG",
            "No integrated proposal/target dependency, KV, bonus or HTTP speedup",
        ],
    }


def maybe_probe(
    model,
    *,
    frame_index: int,
    proposal_depth: int,
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    root = os.environ.get(ENV)
    if not root:
        return
    if any(os.environ.get(key) for key in (
        "SGLANG_OMNI_QTTS_CP_P4_DIR",
        "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_P7_DIR",
        "SGLANG_OMNI_QTTS_CP_P8_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR",
        "SGLANG_OMNI_QTTS_CP_P9B_DIR",
        "SGLANG_OMNI_QTTS_CP_P10A_DIR",
        "SGLANG_OMNI_QTTS_CP_P10B_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    )):
        raise RuntimeError("P11A must run without other K2 diagnostic captures")
    seen = getattr(model, "_qt_cp_p11a_seen", None)
    if seen is None:
        seen = set()
        model._qt_cp_p11a_seen = seen
    if proposal_depth not in (2, 10) or proposal_depth in seen:
        return
    dest = Path(root).expanduser().resolve()
    try:
        report = capture_real_sampler_attribution(
            logits_p=logits_p, logits_q=logits_q,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repeats=64,
        )
    except BaseException:
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        (dest / f"failure_pid{os.getpid()}_depth{proposal_depth:02d}.txt").write_text(
            traceback.format_exc()
        )
        raise
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    report.update({
        "frame": frame_index,
        "proposal_depth": proposal_depth,
        "pid": os.getpid(),
    })
    path = dest / (
        f"attr_pid{os.getpid()}_frame{frame_index:04d}_depth{proposal_depth:02d}.json"
    )
    with path.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    seen.add(proposal_depth)

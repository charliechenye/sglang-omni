# SPDX-License-Identifier: Apache-2.0
"""P10B: sparse top-50 speculative GPU proposal/accept/positive-residual draw.

The data source is ACTUAL on-policy target/draft CodePredictor logits.
The candidate computes two CUDA torch.topk(50) arrays and performs
top-p, softmax, q-proposal, min(1,p/q) accept, and (p-q)+ residual draw
with ONE Triton kernel operating only on those 50+50 selected tokens.
It avoids the old P9B 2x 2048-wide dense scatters, cumsums and full-vocab
searchsorted. The full candidate is CUDA-graph capturable with a private
GPU RNG and is measured against the EXISTING dense P9B reference Graph
within the SAME model/H200 process.

NOT an all-in-one topk-fused sampler: torch.topk still runs separately.
NOT an integrated GPU-resident K2 loop: it doesn't modify live token,
KV, bonus or model RNG. Fails closed on distribution/sampling mismatch
and refuses to claim actual serving acceleration.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .code_predictor_spec import sampling_probs
from .code_predictor_spec_p9b import (
    TEST_UNIFORMS, cpu_reference, gpu_round,
)

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

ENV = "SGLANG_OMNI_QTTS_CP_P10B_DIR"
SCHEMA = "qwen3tts-cpspec-p10b-sparse-topk-fused-accept-correction-v1"
K = 50
BLOCK = 64


@triton.jit
def sparse_accept_residual_kernel(
    P_VALUES, P_IDS, Q_VALUES, Q_IDS, U,
    COMMITTED, PROPOSAL, SURVIVED,
    P_RANK_PROB, Q_RANK_PROB, RESIDUAL_MASS,
    TOP_P: tl.constexpr,
    K_COUNT: tl.constexpr,
    B: tl.constexpr,
):
    """One CTA, one C1 categorical draw; 50x50 match is register-only.

    P_VALUES/Q_VALUES are already temperature-scaled FP32 torch.topk
    sorted results. Reuses the SAME chosen top-k sets and tie order as
    PyTorch reference. Probability vectors remain sparse; positive
    residual support is a subset of P's K candidates.
    """
    r = tl.arange(0, B)
    valid = r < K_COUNT
    pv = tl.load(P_VALUES + r, mask=valid, other=float("-inf"))
    qv = tl.load(Q_VALUES + r, mask=valid, other=float("-inf"))
    pi = tl.load(P_IDS + r, mask=valid, other=-1).to(tl.int32)
    qi = tl.load(Q_IDS + r, mask=valid, other=-2).to(tl.int32)

    pe = tl.exp(pv - tl.max(pv, axis=0))
    qe = tl.exp(qv - tl.max(qv, axis=0))
    p = pe / tl.sum(pe, axis=0)
    q = qe / tl.sum(qe, axis=0)
    if TOP_P < 1.0:
        # Exactly top-p-before-next-token mass check used by reference.
        pr = tl.cumsum(p, axis=0) - p
        qr = tl.cumsum(q, axis=0) - q
        p = tl.where(pr < TOP_P, p, 0.0)
        q = tl.where(qr < TOP_P, q, 0.0)
        p = p / tl.sum(p, axis=0)
        q = q / tl.sum(q, axis=0)

    tl.store(P_RANK_PROB + r, p, mask=valid)
    tl.store(Q_RANK_PROB + r, q, mask=valid)

    u_draft = tl.load(U + 0)
    u_accept = tl.load(U + 1)
    u_correct = tl.load(U + 2)
    # The existing P9B reference samples a DENSE vocab-ID-ordered
    # 2048-wide CDF. Top-k is in probability order instead. Preserve
    # identical inverse-CDF coupling for the deterministic fixed-u
    # parity oracle using only 50 sparse terms:
    # q_cdf_at_i = sum(q_j for token_id_j <= token_id_i).
    q_cdf_by_id = tl.sum(
        tl.where(
            (qi[None, :] <= qi[:, None]) & valid[None, :],
            q[None, :], 0.0,
        ), axis=1,
    )
    draft = tl.min(
        tl.where(valid & (q_cdf_by_id >= u_draft), qi, 2147483647), axis=0
    )
    draft = tl.where(
        draft == 2147483647, tl.max(tl.where(valid, qi, 0), axis=0), draft
    )
    # The draft is now selected by vocabulary token ID (not top-k rank).
    # Gather q(x) from q's sparse support using that same token ID.
    # Using the removed 'proposed_rank' from the old rank-CDF path made
    # this kernel fail Triton compilation on H200.
    qx = tl.sum(tl.where(valid & (qi == draft), q, 0.0), axis=0)
    px = tl.sum(tl.where(pi == draft, p, 0.0), axis=0)
    probability = tl.minimum(1.0, px / tl.maximum(qx, 1.0e-20))
    accepted = u_accept < probability

    # The draft q support is only 50 entries. Match those IDs directly
    # against the p top-50 set to calculate (p(x)-q(x))+ accurately,
    # avoiding construction of a 2048-wide p/q probability vector.
    same = (pi[:, None] == qi[None, :]) & valid[:, None] & valid[None, :]
    q_on_p = tl.sum(tl.where(same, q[None, :], 0.0), axis=1)
    positive = tl.maximum(p - q_on_p, 0.0)
    mass = tl.sum(positive, axis=0)
    corrected_prob = tl.where(
        mass > 1.0e-12, positive / tl.maximum(mass, 1.0e-12), p
    )
    # Residual support is p's top-50, but reference CDF is again in
    # ascending vocabulary ID order rather than probability rank.
    corrected_cdf_by_id = tl.sum(
        tl.where(
            (pi[None, :] <= pi[:, None]) & valid[None, :],
            corrected_prob[None, :], 0.0,
        ), axis=1,
    )
    correction = tl.min(
        tl.where(
            valid & (corrected_cdf_by_id >= u_correct),
            pi, 2147483647,
        ), axis=0,
    )
    correction = tl.where(
        correction == 2147483647,
        tl.max(tl.where(valid, pi, 0), axis=0),
        correction,
    )
    result = tl.where(accepted, draft, correction)
    tl.store(COMMITTED, result.to(tl.int64))
    tl.store(PROPOSAL, draft.to(tl.int64))
    tl.store(SURVIVED, accepted)
    tl.store(RESIDUAL_MASS, mass)


def sparse_gpu_round(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if (
        logits_p.shape != logits_q.shape
        or logits_p.ndim != 2 or logits_p.shape[0] != 1
        or logits_p.shape[1] != 2048
        or logits_p.dtype not in (torch.bfloat16, torch.float32)
        or logits_q.dtype != logits_p.dtype
        or not logits_p.is_cuda or not logits_q.is_cuda
        or uniforms.shape != (1, 3) or not uniforms.is_cuda
        or uniforms.dtype != torch.float32
        or not math.isfinite(temperature) or temperature <= 0
        or not math.isfinite(top_p) or not 0.0 < top_p <= 1
        or top_k != K
    ):
        raise ValueError("P10B requires C1 CUDA BF16/FP32 logits [1,2048], top_k=50")
    # Keep torch.topk exactly as in current P9B/reference (including
    # ties). Fuse the entire post-topk speculative sample into one kernel.
    pval, pidx = torch.topk(
        logits_p.float() / temperature, K, dim=-1, sorted=True
    )
    qval, qidx = torch.topk(
        logits_q.float() / temperature, K, dim=-1, sorted=True
    )
    pprob = torch.empty((BLOCK,), device=logits_p.device, dtype=torch.float32)
    qprob = torch.empty_like(pprob)
    committed = torch.empty((1,), device=logits_p.device, dtype=torch.long)
    draft = torch.empty_like(committed)
    survived = torch.empty((1,), device=logits_p.device, dtype=torch.bool)
    mass = torch.empty((1,), device=logits_p.device, dtype=torch.float32)
    sparse_accept_residual_kernel[(1,)](
        pval, pidx, qval, qidx, uniforms,
        committed, draft, survived, pprob, qprob, mass,
        TOP_P=top_p, K_COUNT=K, B=BLOCK,
        num_warps=8, enable_fp_fusion=False,
    )
    return committed, survived, draft, pidx, qidx, pprob, qprob, mass


def capture_real_sparse_sampler(
    *, logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repeats: int = 64,
) -> dict:
    if (
        logits_p.shape != logits_q.shape
        or logits_p.shape != (1, 2048)
        or not logits_p.is_cuda or not logits_q.is_cuda
        or not 16 <= repeats <= 256
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("P10B needs CUDA real K2 p/q [1,2048] with 16..256 repeats")
    device = logits_p.device
    p = logits_p.detach().clone().contiguous()
    q = logits_q.detach().clone().contiguous()
    p_before = p.clone()
    q_before = q.clone()
    u = torch.tensor(TEST_UNIFORMS[0], device=device, dtype=torch.float32).view(1, 3)
    p_ref = sampling_probs(p, temperature=temperature, top_k=top_k, top_p=top_p)
    q_ref = sampling_probs(q, temperature=temperature, top_k=top_k, top_p=top_p)
    report = {
        "schema": SCHEMA,
        "status": "P10B_NOT_VALIDATED",
        "gpu": torch.cuda.get_device_name(device),
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "vocab": 2048,
        "overlap": float(torch.minimum(p_ref, q_ref).sum().item()),
        "checks": [],
        "repeats": repeats,
    }
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))

    with torch.inference_mode():
        for _ in range(3):
            sparse_gpu_round(
                p, q, u, temperature=temperature, top_k=top_k, top_p=top_p
            )
            gpu_round(
                p, q, u, temperature=temperature, top_k=top_k, top_p=top_p
            )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        graph, stream=side, capture_error_mode="thread_local"
    ), torch.inference_mode():
        candidate = sparse_gpu_round(
            p, q, u, temperature=temperature, top_k=top_k, top_p=top_p
        )
    graph.replay()
    torch.cuda.synchronize(device)

    def collect_candidate():
        token, accept, draft, pidx, qidx, ps, qs, mass = candidate
        # Convert only for numerical validation; this is NOT in CUDA graph.
        full_p = torch.zeros_like(p_ref).scatter(
            -1, pidx, ps[:K].view(1, K)
        )
        full_q = torch.zeros_like(q_ref).scatter(
            -1, qidx, qs[:K].view(1, K)
        )
        return full_p, full_q, {
            "draft": int(draft.item()),
            "accepted": bool(accept.item()),
            "committed": int(token.item()),
        }, float(mass.item())

    got_p, got_q, _, initial_mass = collect_candidate()
    report["p_max_abs"] = float((got_p - p_ref).abs().max().item())
    report["q_max_abs"] = float((got_q - q_ref).abs().max().item())
    if not torch.allclose(got_p, p_ref, rtol=1e-6, atol=1e-6) or not torch.allclose(
        got_q, q_ref, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError(
            f"P10B_SPARSE_DISTRIBUTION_FAIL p={report['p_max_abs']} "
            f"q={report['q_max_abs']} no-tolerance-relax"
        )
    if not math.isfinite(initial_mass) or initial_mass < 0:
        raise RuntimeError("P10B invalid positive residual mass")

    for triple in TEST_UNIFORMS:
        u.copy_(torch.tensor(triple, device=device, dtype=torch.float32).view(1, 3))
        graph.replay()
        torch.cuda.synchronize(device)
        _, _, got, mass = collect_candidate()
        expected = cpu_reference(p_ref, q_ref, triple)
        ok = got == expected
        report["checks"].append({
            "uniforms": list(triple),
            "actual": got,
            "expected": expected,
            "positive_residual_mass": mass,
            "pass": ok,
        })
        if not ok:
            raise RuntimeError(
                f"P10B_SPARSE_PROPOSAL_ACCEPT_RESIDUAL_PARITY_FAIL "
                f"uniforms={triple} actual={got} expected={expected}"
            )

    # Both graphs draw their own CUDA randoms, not the real model RNG.
    def make_random_graph(fn, rng_seed):
        rng = torch.Generator(device=device).manual_seed(rng_seed)
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            cuda_graph, stream=side, capture_error_mode="thread_local"
        ), torch.inference_mode():
            fresh = torch.rand((1, 3), generator=rng, device=device)
            out = fn(
                p, q, fresh, temperature=temperature, top_k=top_k, top_p=top_p
            )
        return cuda_graph, fresh, out, rng

    sparse_graph, sparse_uniforms, _, sparse_rng = make_random_graph(
        sparse_gpu_round, 20261122
    )
    reference_graph, reference_uniforms, _, reference_rng = make_random_graph(
        gpu_round, 20261123
    )
    observations = []
    for _ in range(4):
        sparse_graph.replay()
        torch.cuda.synchronize(device)
        observations.append(tuple(sparse_uniforms.flatten().cpu().tolist()))
    if len(set(observations)) != 4:
        raise RuntimeError("P10B random graph RNG did not advance")
    report["private_rng_replay_unique"] = True

    def graph_ms(graph_to_time):
        for _ in range(8):
            graph_to_time.replay()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph_to_time.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeats

    report["sparse_fused_cuda_graph_ms"] = graph_ms(sparse_graph)
    report["p9b_dense_reference_cuda_graph_ms"] = graph_ms(reference_graph)
    report["sparse_over_dense_ratio"] = (
        report["sparse_fused_cuda_graph_ms"]
        / report["p9b_dense_reference_cuda_graph_ms"]
    )
    report["input_logits_immutable"] = (
        torch.equal(p, p_before) and torch.equal(q, q_before)
    )
    if not report["input_logits_immutable"]:
        raise RuntimeError("P10B mutated original p/q logits")
    report["status"] = (
        "P10B_REAL_PQ_SPARSE_TOPK_SPEC_SAMPLER_PARITY_PASS_"
        "NOT_FULL_K2_OR_NATIVE_RNG"
    )
    report["limitations"] = [
        "Only post-topk operators are fused; two PyTorch topk launches remain",
        "On-policy K2 p/q logits, but sampling outcome is not committed into live request",
        "Top-k/softmax/top-p distribution matches existing Torch reference at 1e-6",
        "Independent private graph RNG is NOT the production fused-sampler RNG contract",
        "Dynamic K/V, bonus and K2 depth updates not integrated",
        "CUDA-event graph timing is a component cost, NOT E2E QPS",
    ]
    return report


def maybe_probe(
    model: "Qwen3TTSTalker", *,
    frame_index: int,
    proposal_depth: int,
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
):
    root = os.environ.get(ENV)
    if not root:
        return
    if any(os.environ.get(name) for name in (
        "SGLANG_OMNI_QTTS_CP_P4_DIR", "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_P7_DIR", "SGLANG_OMNI_QTTS_CP_P8_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR", "SGLANG_OMNI_QTTS_CP_P9B_DIR",
        "SGLANG_OMNI_QTTS_CP_P10A_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    )):
        raise RuntimeError("P10B must be isolated from other K2 profiling probes")
    seen = getattr(model, "_qt_cp_p10b_seen", None)
    if seen is None:
        seen = set()
        model._qt_cp_p10b_seen = seen
    if proposal_depth not in (2, 10) or proposal_depth in seen:
        return
    dest = Path(root).expanduser().resolve()
    try:
        report = capture_real_sparse_sampler(
            logits_p=logits_p,
            logits_q=logits_q,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
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
        "pid": os.getpid(),
        "frame": frame_index,
        "proposal_depth": proposal_depth,
    })
    with (dest / (
        f"sparse_pid{os.getpid()}_frame{frame_index:04d}_depth{proposal_depth:02d}.json"
    )).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    seen.add(proposal_depth)

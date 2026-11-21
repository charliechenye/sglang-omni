# SPDX-License-Identifier: Apache-2.0
"""P12A H200 one-distribution top-50 selection; source-locked, FAIL-CLOSED.

The real K2 causal path produces q BEFORE speculative verification and p
AFTER verification. This diagnostic observes both stages at the exact
same live (frame,depth) but NEVER changes the live proposal, model,
KV cache, sampler RNG, or audio. It compares a single-CTA Triton
top-50+softmax+top-p+fixed-uniform draft selector to the current torch
topk(50) + softmax + identical inverse-CDF selection. CUDA Graph replay
event envelopes are measured separately for q and p stages.

Key distinction: Torch topk on tied BF16 scores need not use the same
tie order as the candidate's deterministic token-ID tie break. Exact
selected IDs and per-token p/q probabilities MUST match. Fail with
P12A_TIE_ORDER_MISMATCH (not a claimed numerical parity pass) rather
than silently changing the speculative target distribution.

Still NOT an integrated causal GPU K2 loop, HTTP performance, native
seeded RNG parity, or full end-to-end acceleration proof.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path

import torch
import triton
import triton.language as tl

from .code_predictor_spec import sampling_probs

ENV = "SGLANG_OMNI_QTTS_CP_P12A_DIR"
SCHEMA = "qwen3tts-cpspec-p12a-causal-stage-single-distribution-top50-v1"
K = 50
RANK_BLOCK = 64
VOCAB = 2048
DRAWS = (0.125, 0.625, 0.995, 0.333)
STAGES = ("q_before_verifier", "p_after_verifier")


@triton.jit
def single_top50_kernel(
    LOGITS, U, IDS, VALUES, PROBS, CODE,
    TEMPERATURE: tl.constexpr, TOP_P: tl.constexpr,
    VOCAB_WIDTH: tl.constexpr, K_COUNT: tl.constexpr,
    B: tl.constexpr,
):
    """2048-score descending bitonic-sort candidate, one CTA, vocab-ID CDF.

    Positive IEEE float bit order is monotonic. Negative floats are
    inverted to monotonic unsigned ordering. Low 32 bits prefer smaller
    vocab ID on exact ties, explicitly NOT presumed torch.topk tie order.
    """
    v = tl.arange(0, VOCAB_WIDTH)
    score = tl.load(LOGITS + v).to(tl.float32) / TEMPERATURE
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where(
        (bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000,
    ).to(tl.uint64)
    packed = (ordered << 32) | (0xFFFFFFFF - v.to(tl.uint32)).to(tl.uint64)
    sorted_keys = tl.sort(packed, descending=True)
    r = tl.arange(0, B)
    chosen = tl.gather(sorted_keys, r, axis=0)
    code = (0xFFFFFFFF - (chosen & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    chosen_score = tl.load(LOGITS + code, mask=r < K_COUNT, other=0).to(tl.float32)
    chosen_score = tl.where(r < K_COUNT, chosen_score / TEMPERATURE, float("-inf"))
    weights = tl.exp(chosen_score - tl.max(chosen_score, axis=0))
    prob = weights / tl.sum(weights, axis=0)
    if TOP_P < 1.0:
        remove = tl.cumsum(prob, axis=0) - prob >= TOP_P
        prob = tl.where(remove, 0.0, prob)
        prob = prob / tl.sum(prob, axis=0)
    tl.store(IDS + r, code, mask=r < K_COUNT)
    tl.store(VALUES + r, chosen_score, mask=r < K_COUNT)
    tl.store(PROBS + r, prob, mask=r < K_COUNT)
    # Preserve existing P9B/P10B deterministic inverse-CDF coupling:
    # sample from the sparse top50 in ASCENDING *vocabulary ID* order.
    cdf = tl.sum(
        tl.where(
            (code[None, :] <= code[:, None]) & (r[None, :] < K_COUNT),
            prob[None, :], 0.0,
        ),
        axis=1,
    )
    uniform = tl.load(U)
    selected = tl.min(
        tl.where((r < K_COUNT) & (cdf >= uniform), code, 2147483647),
        axis=0,
    )
    selected = tl.where(
        selected == 2147483647,
        tl.max(tl.where(r < K_COUNT, code, 0), axis=0),
        selected,
    )
    tl.store(CODE, selected.to(tl.int64))


@triton.jit
def sample_sparse_torch_top50(
    IDS, PROBS, U, CODE,
    K_COUNT: tl.constexpr, B: tl.constexpr,
):
    """Reference fixed-uniform draw over Torch topk support in vocab-ID order."""
    r = tl.arange(0, B)
    ids = tl.load(IDS + r, mask=r < K_COUNT, other=0).to(tl.int32)
    prob = tl.load(PROBS + r, mask=r < K_COUNT, other=0.0)
    cdf = tl.sum(
        tl.where(
            (ids[None, :] <= ids[:, None]) & (r[None, :] < K_COUNT),
            prob[None, :], 0.0,
        ),
        axis=1,
    )
    u = tl.load(U)
    selected = tl.min(
        tl.where((r < K_COUNT) & (cdf >= u), ids, 2147483647), axis=0
    )
    selected = tl.where(
        selected == 2147483647,
        tl.max(tl.where(r < K_COUNT, ids, 0), axis=0),
        selected,
    )
    tl.store(CODE, selected.to(tl.int64))


def fast_selector(
    logits: torch.Tensor, uniform: torch.Tensor,
    *, temperature: float, top_p: float,
):
    if (
        logits.shape != (1, VOCAB) or not logits.is_cuda
        or logits.dtype not in (torch.bfloat16, torch.float32)
        or uniform.shape != (1,) or not uniform.is_cuda
        or uniform.dtype != torch.float32
        or not math.isfinite(temperature) or temperature <= 0
        or not math.isfinite(top_p) or not 0 < top_p <= 1
    ):
        raise ValueError("P12A requires real C1 CUDA logits [1,2048] and FP32 uniform")
    dev = logits.device
    idx = torch.empty((1, K), device=dev, dtype=torch.int32)
    val = torch.empty((1, K), device=dev, dtype=torch.float32)
    prob = torch.empty_like(val)
    token = torch.empty((1,), device=dev, dtype=torch.long)
    single_top50_kernel[(1,)](
        logits, uniform, idx, val, prob, token,
        TEMPERATURE=temperature, TOP_P=top_p,
        VOCAB_WIDTH=VOCAB, K_COUNT=K, B=RANK_BLOCK,
        num_warps=8, enable_fp_fusion=False,
    )
    return idx, val, prob, token


def torch_selector(
    logits: torch.Tensor, uniform: torch.Tensor,
    *, temperature: float, top_p: float,
):
    scores, idx = torch.topk(
        logits.float() / temperature, K, dim=-1, sorted=True
    )
    prob = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        remove = torch.cumsum(prob, dim=-1) - prob >= top_p
        prob = torch.where(remove, 0.0, prob)
        prob = prob / prob.sum(-1, keepdim=True)
    code = torch.empty((1,), device=logits.device, dtype=torch.long)
    sample_sparse_torch_top50[(1,)](
        idx, prob, uniform, code,
        K_COUNT=K, B=RANK_BLOCK, num_warps=4,
    )
    return idx, scores, prob, code


def _full_dist(ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    full = torch.zeros((1, VOCAB), dtype=torch.float32, device=probs.device)
    return full.scatter(-1, ids.to(torch.long), probs)


def profile_stage(
    logits: torch.Tensor, *,
    temperature: float, top_p: float, repeats: int,
) -> dict:
    if not 8 <= repeats <= 256 or torch.cuda.is_current_stream_capturing():
        raise ValueError("P12A requires 8..256 repeats outside CUDA capture")
    dev = logits.device
    copy = logits.detach().clone().contiguous()
    original = copy.clone()
    u = torch.tensor([DRAWS[0]], device=dev, dtype=torch.float32)
    # Reference probabilities are produced outside capture; never
    # claim equivalence from just one random code sample.
    ref = sampling_probs(copy, temperature=temperature, top_k=K, top_p=top_p)
    with torch.inference_mode():
        for _ in range(2):
            fast_selector(copy, u, temperature=temperature, top_p=top_p)
            torch_selector(copy, u, temperature=temperature, top_p=top_p)
    torch.cuda.synchronize(dev)

    graphs = {}
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    for label, fn in (
        ("torch_topk_reference", torch_selector),
        ("triton_single_top50", fast_selector),
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, stream=side, capture_error_mode="thread_local"
        ), torch.inference_mode():
            outputs = fn(copy, u, temperature=temperature, top_p=top_p)
        graphs[label] = (graph, outputs)
    # Capture is NOT execution; initialize both outputs via replay.
    for graph, _ in graphs.values():
        graph.replay()
    torch.cuda.synchronize(dev)

    xidx, xvals, xprob, _ = graphs["triton_single_top50"][1]
    yidx, yvals, yprob, _ = graphs["torch_topk_reference"][1]
    # On BF16 ties, torch.topk can choose a different (but same-score)
    # ID. That changes q/p support, so it is NOT a lossless replacement.
    ids_equal = bool(torch.equal(xidx.to(torch.long), yidx))
    probs_equal = bool(torch.allclose(
        _full_dist(xidx, xprob), ref, atol=1e-6, rtol=1e-6
    ))
    torch_probs_equal = bool(torch.allclose(
        _full_dist(yidx, yprob), ref, atol=1e-6, rtol=1e-6
    ))
    if not torch_probs_equal:
        raise RuntimeError("P12A reference topk+top-p output does not match P9B")
    mismatch = not (ids_equal and probs_equal)
    draws_match = True
    mismatched_draws = []
    for draw in DRAWS:
        u.fill_(draw)
        for graph, _ in graphs.values():
            graph.replay()
        torch.cuda.synchronize(dev)
        got = int(graphs["triton_single_top50"][1][3].item())
        expect = int(graphs["torch_topk_reference"][1][3].item())
        if got != expect:
            draws_match = False
            mismatched_draws.append({
                "uniform": draw, "candidate_token": got,
                "torch_reference_token": expect,
            })

    def measure(graph: torch.cuda.CUDAGraph):
        for _ in range(6):
            graph.replay()
        torch.cuda.synchronize(dev)
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(repeats):
            graph.replay()
        t1.record()
        t1.synchronize()
        return t0.elapsed_time(t1) / repeats

    measures = {"torch_topk_reference": [], "triton_single_top50": []}
    for order in (
        ("torch_topk_reference", "triton_single_top50"),
        ("triton_single_top50", "torch_topk_reference"),
        ("triton_single_top50", "torch_topk_reference"),
        ("torch_topk_reference", "triton_single_top50"),
    ):
        for name in order:
            measures[name].append(measure(graphs[name][0]))
    if not torch.equal(copy, original):
        raise RuntimeError("P12A candidate mutated original real logits")
    import statistics
    medians = {name: statistics.median(series) for name, series in measures.items()}
    if any(not math.isfinite(val) or val <= 0 for val in medians.values()):
        raise RuntimeError("P12A invalid measured graph time")
    top_values = torch.topk(
        copy.float() / temperature, K + 1, dim=-1, sorted=True
    )[0][0].cpu().tolist()
    tied_within_topk = sum(
        top_values[i] == top_values[i + 1] for i in range(K - 1)
    )
    tied_at_boundary = top_values[K-1] == top_values[K]
    exact = not mismatch and draws_match
    return {
        "status": (
            "P12A_SINGLE_DISTRIBUTION_TOP50_EXACT_MATCH_PASS"
            if exact else
            "P12A_TIE_OR_DISTRIBUTION_MISMATCH_NO_SELECTION"
        ),
        "probability_match": probs_equal,
        "selected_ids_match": ids_equal,
        "deterministic_draws_match": draws_match,
        "mismatched_draws": mismatched_draws,
        "tied_pairs_inside_top50": tied_within_topk,
        "tie_at_top50_boundary": tied_at_boundary,
        "fixed_uniform_draws": list(DRAWS),
        "times_ms": measures,
        "median_ms": medians,
        "fast_over_reference_ratio": (
            medians["triton_single_top50"] / medians["torch_topk_reference"]
        ),
        "selection_eligible": exact,
    }


def maybe_observe_q(
    model, *,
    frame: int, depth: int, logits: torch.Tensor,
    temperature: float, top_k: int, top_p: float,
) -> None:
    root = os.environ.get(ENV)
    if not root:
        return
    if top_k != K or model.config.num_code_groups != 16:
        raise ValueError("P12A only supports actual top-k50 16-codebook C1 sampling")
    if any(os.environ.get(e) for e in (
        "SGLANG_OMNI_QTTS_CP_P11A_DIR", "SGLANG_OMNI_QTTS_CP_P10B_DIR",
        "SGLANG_OMNI_QTTS_CP_P9B_DIR", "SGLANG_OMNI_QTTS_CP_P10A_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR", "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
        "SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR",
    )):
        raise RuntimeError("P12A cannot coexist with any other graph/cost probe")
    seen = getattr(model, "_qt_p12a_seen", set())
    if depth not in (2, 10) or depth in seen:
        return
    if getattr(model, "_qt_p12a_pending", None) is not None:
        raise RuntimeError("P12A pending q was not consumed by the true verifier")
    model._qt_p12a_pending = {
        "frame": frame, "depth": depth,
        "logits_q": logits.detach().clone(),
        "temperature": temperature, "top_p": top_p,
    }


def maybe_observe_p(
    model, *,
    frame: int, depth: int, logits: torch.Tensor,
    temperature: float, top_k: int, top_p: float,
) -> None:
    if not os.environ.get(ENV) or depth not in (2, 10):
        return
    pending = getattr(model, "_qt_p12a_pending", None)
    if pending is None:
        if depth in getattr(model, "_qt_p12a_seen", set()):
            return
        raise RuntimeError("P12A p observed without q-before-verifier")
    if (
        pending["frame"] != frame or pending["depth"] != depth
        or pending["temperature"] != temperature or pending["top_p"] != top_p
        or top_k != K
    ):
        raise RuntimeError("P12A broken q->real verifier->p causal matching")
    root = Path(os.environ[ENV]).expanduser().resolve()
    try:
        result = {
            "schema": SCHEMA,
            "status": "P12A_NOT_YET_FINAL",
            "frame": frame, "depth": depth,
            "source_sha": os.environ.get(
                "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
            ),
            "gpu": torch.cuda.get_device_name(logits.device),
            "pid": os.getpid(),
            "stages": {
                "q_before_verifier": profile_stage(
                    pending["logits_q"], temperature=temperature,
                    top_p=top_p, repeats=32
                ),
                "p_after_verifier": profile_stage(
                    logits, temperature=temperature,
                    top_p=top_p, repeats=32
                ),
            },
            "limitations": [
                "q and p are captured from real on-policy sides of verifier",
                "Selector graphs are replayed independently; NOT one fused causal K2",
                "Triton ties must match original PyTorch topk support exactly",
                "Fixed uniform inverse-CDF, not native fused seeded RNG",
                "No live code/KV/RNG mutation; no serving throughput proof",
            ],
        }
        exact = all(
            row["selection_eligible"]
            for row in result["stages"].values()
        )
        result["selection_eligible"] = exact
        result["status"] = (
            "P12A_CAUSAL_REAL_Q_PRE_AND_P_POST_COMPONENT_GATE_PASS_NOT_FULL_K2"
            if exact else
            "P12A_CAUSAL_REAL_Q_PRE_AND_P_POST_NO_SELECTION_TIE_OR_DRAW_MISMATCH"
        )
    except BaseException:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (root / f"failure_pid{os.getpid()}_depth{depth:02d}.txt").write_text(
            traceback.format_exc()
        )
        model._qt_p12a_pending = None
        raise
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    file = root / f"selector_pid{os.getpid()}_frame{frame:04d}_depth{depth:02d}.json"
    with file.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    seen = getattr(model, "_qt_p12a_seen", set())
    seen.add(depth)
    model._qt_p12a_seen = seen
    model._qt_p12a_pending = None

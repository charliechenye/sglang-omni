# SPDX-License-Identifier: Apache-2.0
"""P14: ONE genuine causal CodePredictor K2 round in ONE CUDA Graph.

This is a source-locked, opt-in H200 research probe inside the existing
C1 sampled debug K2 path, on an ACTUAL on-policy cache/hidden checkpoint.

q head -> q Top50 and GPU proposal -> real proposal embedding ->
real packed two-token Predictor verifier (its second input DEPENDS ON
the sampled q token) -> target p head -> p Top50 support ->
GPU accept / positive-residual draw -> optional bonus head / sample ->
GPU-computed logical next depth/cache_len, next hidden, projected
embedding and residual-code sum. There is NO Python decision, CPU
synchronization or separate q/p graph across these operations.

Both graph arms are matched: source Torch Top50+same sparse accept
versus fast P12A Triton Top64->Top50+same sparse accept. They replay
with identical fixed GPU input uniforms and identical KV prefix.
A separate private-GPU-RNG candidate graph is captured to show that
replay advances PRIVATE RNG state; it is NOT native seeded RNG parity.

The bonus head is evaluated UNCONDITIONALLY in the static graph, then
masked when rejected. This is correct for sampled returned state but
adds needless rejected-round work. Existing request RNG, output codes,
and output summation are NEVER touched; only temporary predictor KV
suffixes are written and restored before live K2 continues.

NOT a 14-depth GPU loop; next cache_len/depth remain device tensors
but no next Predictor call consumes them in this probe. No conditional
bonus skip, dynamic device KV ownership for subsequent rounds, native
sampler RNG equivalence, serving latency or production speedup claim.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .code_predictor_spec import sampling_probs
from .code_predictor_spec_p12a_selector import (
    K, RANK_BLOCK, VOCAB, fast_selector, torch_selector, _full_dist,
)
from .code_predictor_spec_p7_dispatch import compare
from .code_predictor_spec_p9b import cpu_reference

ENV = "SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR"
SCHEMA = "qwen3tts-cpspec-p14-one-round-causal-q-proposal-verifier-p-accept-bonus-v1"
DEPTHS = (2, 10)
TRIPLES = (
    (0.125, 0.250, 0.375, 0.625),
    (0.625, 0.750, 0.875, 0.125),
    (0.995, 0.015, 0.515, 0.333),
    (0.333, 0.333, 0.333, 0.995),
)


@triton.jit
def p14_support_top50(
    LOGITS, IDS, PROBS,
    TEMPERATURE: tl.constexpr, TOP_P: tl.constexpr,
    VOCAB_WIDTH: tl.constexpr, K_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """GPU p support ONLY, same P12A score order, no redundant p draw."""
    ids = tl.arange(0, VOCAB_WIDTH)
    values = tl.load(LOGITS + ids).to(tl.float32) / TEMPERATURE
    bits = values.to(tl.uint32, bitcast=True)
    ordered = tl.where(
        (bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000,
    ).to(tl.uint64)
    packed = (
        (ordered << 32) |
        (0xFFFFFFFF - ids.to(tl.uint32)).to(tl.uint64)
    )
    keys = tl.topk(packed, BLOCK, descending=True)
    r = tl.arange(0, BLOCK)
    code = (0xFFFFFFFF - (keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    score = tl.load(LOGITS + code, mask=r < K_COUNT, other=0).to(tl.float32)
    score = tl.where(r < K_COUNT, score / TEMPERATURE, float("-inf"))
    exps = tl.exp(score - tl.max(score, axis=0))
    p = exps / tl.sum(exps, axis=0)
    if TOP_P < 1.0:
        remove = tl.cumsum(p, axis=0) - p >= TOP_P
        p = tl.where(remove, 0.0, p)
        p = p / tl.sum(p, axis=0)
    tl.store(IDS + r, code, mask=r < K_COUNT)
    tl.store(PROBS + r, p, mask=r < K_COUNT)


def single_support(
    logits: torch.Tensor, *, temperature: float, top_p: float,
    fast: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fast:
        if (
            logits.shape != (1, VOCAB)
            or logits.dtype not in (torch.float32, torch.bfloat16)
            or not logits.is_cuda
        ):
            raise ValueError("P14 p support requires C1 CUDA 2048 logits")
        indices = torch.empty((1, K), dtype=torch.int32, device=logits.device)
        probs = torch.empty((1, K), dtype=torch.float32, device=logits.device)
        p14_support_top50[(1,)](
            logits, indices, probs,
            TEMPERATURE=temperature, TOP_P=top_p,
            VOCAB_WIDTH=VOCAB, K_COUNT=K, BLOCK=RANK_BLOCK,
            num_warps=8, enable_fp_fusion=False,
        )
        return indices, probs
    values, indices = torch.topk(
        logits.float() / temperature, K, dim=-1, sorted=True
    )
    probs = torch.softmax(values, dim=-1)
    if top_p < 1.0:
        remove = torch.cumsum(probs, dim=-1) - probs >= top_p
        probs = torch.where(remove, 0.0, probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return indices, probs


@triton.jit
def p14_accept_residual_from_sparse(
    P_IDS, P_PROBS, Q_IDS, Q_PROBS, U, Q_DRAFT,
    COMMITTED, ACCEPTED, CORRECTED,
    ACCEPT_PROB, POSITIVE_MASS,
    K_COUNT: tl.constexpr, BLOCK: tl.constexpr,
):
    """GPU-causal draft already sampled before verifier; no second q draw.

    P/Q are normalized sparse top-50 probabilities after top-p.
    Given q's actual sampled token, accept min(1,p(x)/q(x)) and
    otherwise sample exactly normalized positive (p-q)+ by vocab-ID CDF.
    """
    r = tl.arange(0, BLOCK)
    valid = r < K_COUNT
    pi = tl.load(P_IDS + r, valid, other=-1).to(tl.int32)
    qi = tl.load(Q_IDS + r, valid, other=-2).to(tl.int32)
    p = tl.load(P_PROBS + r, valid, other=0.0)
    q = tl.load(Q_PROBS + r, valid, other=0.0)
    draft = tl.load(Q_DRAFT).to(tl.int32)
    qx = tl.sum(tl.where(valid & (qi == draft), q, 0.0), axis=0)
    px = tl.sum(tl.where(valid & (pi == draft), p, 0.0), axis=0)
    probability = tl.minimum(1.0, px / tl.maximum(qx, 1.0e-20))
    u_accept = tl.load(U + 1)
    survived = u_accept < probability

    same = (
        (pi[:, None] == qi[None, :]) &
        valid[:, None] & valid[None, :]
    )
    q_on_p = tl.sum(tl.where(same, q[None, :], 0.0), axis=1)
    positive = tl.maximum(p - q_on_p, 0.0)
    mass = tl.sum(positive, axis=0)
    corrected_probs = tl.where(
        mass > 1.0e-12,
        positive / tl.maximum(mass, 1.0e-12),
        p,
    )
    cdf = tl.sum(
        tl.where(
            (pi[None, :] <= pi[:, None]) & valid[None, :],
            corrected_probs[None, :], 0.0,
        ),
        axis=1,
    )
    u_residual = tl.load(U + 2)
    corrected = tl.min(
        tl.where(
            valid & (cdf >= u_residual),
            pi, 2147483647,
        ), axis=0,
    )
    corrected = tl.where(
        corrected == 2147483647,
        tl.max(tl.where(valid, pi, 0), axis=0),
        corrected,
    )
    result = tl.where(survived, draft, corrected)
    tl.store(COMMITTED, result.to(tl.int64))
    tl.store(ACCEPTED, survived)
    tl.store(CORRECTED, corrected.to(tl.int64))
    tl.store(ACCEPT_PROB, probability)
    tl.store(POSITIVE_MASS, mass)


def gpu_accept(
    p_ids: torch.Tensor, p_probs: torch.Tensor,
    q_ids: torch.Tensor, q_probs: torch.Tensor,
    u: torch.Tensor, q_draft: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dev = q_draft.device
    committed = torch.empty((1,), device=dev, dtype=torch.long)
    accepted = torch.empty((1,), device=dev, dtype=torch.bool)
    corrected = torch.empty_like(committed)
    probability = torch.empty((1,), device=dev, dtype=torch.float32)
    mass = torch.empty_like(probability)
    p14_accept_residual_from_sparse[(1,)](
        p_ids, p_probs, q_ids, q_probs, u, q_draft,
        committed, accepted, corrected, probability, mass,
        K_COUNT=K, BLOCK=RANK_BLOCK,
        num_warps=8, enable_fp_fusion=False,
    )
    return {
        "committed": committed, "accepted": accepted,
        "residual_correction": corrected,
        "accept_probability": probability,
        "residual_mass": mass,
    }


def make_causal_round(model, *,
    depth: int, cache_len: int,
    draft_hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated_before: torch.Tensor,
    uniforms: torch.Tensor, temperature: float, top_p: float,
    fast: bool,
) -> dict[str, torch.Tensor]:
    """One complete causal K2 round; static offset, all GPU dependency.

    Depth is last committed residual code (1 or 9 in P14).
    The returned device scalar state covers acceptance/rejection/bonus,
    but a next verifier GPU call has NOT consumed it yet.
    """
    if not 1 <= depth <= 13 or cache_len != depth + 1:
        raise ValueError("P14 requires real K2 committed depth and KV offset")
    if (
        uniforms.shape != (4,) or not uniforms.is_cuda
        or uniforms.dtype != torch.float32
        or last_embedding.shape[:2] != (1, 1)
        or draft_hidden.shape != last_embedding.shape
        or accumulated_before.shape[0] != 1
    ):
        raise ValueError("P14 requires one CUDA row, 4 explicit uniforms")

    head = model.code_predictor.lm_head[depth]
    bonus_head = model.code_predictor.lm_head[depth + 1]
    codec_tables = model.code_predictor.model.codec_embedding.weight
    proj = model.predictor_projected_embeddings
    predictor_dtype = model.predictor_k_cache.dtype

    def projected_embed(code, index):
        if proj is not None:
            return F.embedding(
                code, proj[index]
            ).unsqueeze(1).contiguous()
        raw = F.embedding(code, codec_tables[index])
        return model.code_predictor.project_input(
            raw.unsqueeze(1).to(dtype=predictor_dtype)
        ).contiguous()

    # First q sample is a real GPU dependency of the verifier INPUT.
    draft_logits, _ = head(draft_hidden)
    q_logits = draft_logits[:, -1, :]
    q_idx, _, q_probs, proposal = (
        fast_selector(
            q_logits, uniforms[0:1],
            temperature=temperature, top_p=top_p
        ) if fast else
        torch_selector(
            q_logits, uniforms[0:1],
            temperature=temperature, top_p=top_p
        )
    )
    draft_embedding = projected_embed(proposal, depth)
    packed = torch.cat((last_embedding, draft_embedding), dim=1)
    verified = model.predictor_forward_tokens(
        token_embeds=packed, batch_size=1, cache_len=cache_len
    )
    before_proposal = verified[:, 0:1, :].clone()
    after_proposal = verified[:, 1:2, :].clone()
    logits_target, _ = head(before_proposal)
    p_logits = logits_target[:, -1, :]
    p_idx, p_probs = single_support(
        p_logits, temperature=temperature, top_p=top_p, fast=fast,
    )
    decision = gpu_accept(
        p_idx, p_probs, q_idx, q_probs, uniforms, proposal
    )
    committed = decision["committed"]
    accepted = decision["accepted"]
    bonus_logits, _ = bonus_head(after_proposal)
    bonus_idx, _, bonus_probs, bonus_code = (
        fast_selector(
            bonus_logits[:, -1, :], uniforms[3:4],
            temperature=temperature, top_p=top_p
        ) if fast else
        torch_selector(
            bonus_logits[:, -1, :], uniforms[3:4],
            temperature=temperature, top_p=top_p
        )
    )

    raw_committed = F.embedding(
        committed, codec_tables[depth]
    ).to(dtype=accumulated_before.dtype)
    raw_bonus = F.embedding(
        bonus_code, codec_tables[depth + 1]
    ).to(dtype=accumulated_before.dtype)
    next_sum = accumulated_before + raw_committed
    next_sum = next_sum + torch.where(
        accepted.view(1, 1), raw_bonus, torch.zeros_like(raw_bonus)
    )
    committed_proj = projected_embed(committed, depth)
    bonus_proj = projected_embed(bonus_code, depth + 1)
    next_last_embedding = torch.where(
        accepted.reshape(1, 1, 1), bonus_proj, committed_proj
    )
    next_hidden = torch.where(
        accepted.reshape(1, 1, 1), after_proposal, before_proposal
    )
    next_cache_len = cache_len + 1 + accepted.to(dtype=torch.long)
    next_depth = depth + 1 + accepted.to(dtype=torch.long)
    return {
        "draft": proposal,
        "committed": committed,
        "accepted": accepted,
        "corrected": decision["residual_correction"],
        "bonus": bonus_code,
        "bonus_ids": bonus_idx,
        "bonus_probs": bonus_probs,
        "bonus_logits": bonus_logits[:, -1, :],
        "bonus_valid": accepted.clone(),
        "accept_probability": decision["accept_probability"],
        "residual_mass": decision["residual_mass"],
        "q_logits": q_logits, "p_logits": p_logits,
        "q_ids": q_idx, "q_probs": q_probs,
        "p_ids": p_idx, "p_probs": p_probs,
        "before_proposal": before_proposal,
        "after_proposal": after_proposal,
        "next_hidden": next_hidden,
        "next_embedding": next_last_embedding,
        "next_sum": next_sum,
        "next_cache_len": next_cache_len,
        "next_depth": next_depth,
        "sampled_uniforms": uniforms,
    }


def _distribution(ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return _full_dist(ids, probs)


def _strict_close(name: str, a: torch.Tensor, b: torch.Tensor, *,
                  atol: float = 0., rtol: float = 0.):
    if not torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False):
        deviation = (a.float() - b.float()).abs().max().item()
        raise RuntimeError(
            f"P14_{name}_PARITY_FAIL max_abs={deviation:.8g} "
            f"atol={atol} rtol={rtol}; no silent tolerance relaxation"
        )


@torch.no_grad()
def capture_real_causal_round(
    model, *,
    depth: int,
    cache_len: int,
    draft_hidden: torch.Tensor,
    last_embedding: torch.Tensor,
    accumulated_before: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repeats: int = 32,
) -> dict:
    proposal_depth = depth + 1
    cache_k, cache_v = model.predictor_k_cache, model.predictor_v_cache
    if (
        proposal_depth not in DEPTHS or cache_len != depth + 1
        or top_k != K or not 0 < top_p <= 1.0 or temperature <= 0
        or not draft_hidden.is_cuda or not last_embedding.is_cuda
        or not accumulated_before.is_cuda
        or draft_hidden.shape != last_embedding.shape
        or draft_hidden.shape[:2] != (1, 1)
        or cache_k.shape[2] < 16
        or not 8 <= repeats <= 256
        or torch.cuda.is_current_stream_capturing()
        or model.predictor_cutedsl_layers is not None
        or model.predictor_fused_layers is None
        or not model.predictor_fused_layers.covers(2)
        or len(model.code_predictor.model.layers) != 5
    ):
        raise RuntimeError(
            "P14_UNSUPPORTED_STATE_OR_BACKEND: need actual C1 H200 five-layer "
            "fused Predictor real K2 prefix, topk50, depth2/10 and static "
            "two-token verifier; no fallback to an unmatched kernel backend"
        )

    dev = last_embedding.device
    old_k = cache_k[:, :1, :16].detach().clone()
    old_v = cache_v[:, :1, :16].detach().clone()
    hidden = draft_hidden.detach().clone().contiguous()
    last = last_embedding.detach().clone().contiguous()
    summed = accumulated_before.detach().clone().contiguous()
    saved = (hidden.clone(), last.clone(), summed.clone())
    uniforms = torch.tensor(TRIPLES[0], device=dev, dtype=torch.float32)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    graph_outputs = {}
    graph_holders = {}

    def restore():
        cache_k[:, :1, :16].copy_(old_k)
        cache_v[:, :1, :16].copy_(old_v)

    def snapshot():
        return cache_k[:, :1, :16].clone(), cache_v[:, :1, :16].clone()

    def invoke(fast: bool, u: torch.Tensor):
        return make_causal_round(
            model, depth=depth, cache_len=cache_len,
            draft_hidden=hidden, last_embedding=last,
            accumulated_before=summed, uniforms=u,
            temperature=temperature, top_p=top_p, fast=fast,
        )

    report = {
        "schema": SCHEMA,
        "status": "P14_NOT_YET_VALIDATED",
        "proposal_depth": proposal_depth,
        "cache_len_at_entry": cache_len,
        "predictor_layers": len(model.code_predictor.model.layers),
        "backend": "matched_fused_predictor_layers_static_offset",
        "sampling": {
            "temperature": temperature,
            "top_k": top_k, "top_p": top_p,
        },
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "gpu": torch.cuda.get_device_name(dev),
        "fixed_uniform_cases": [],
        "captured_graph_count": 3,
        "replays_per_block": repeats,
    }

    try:
        # JIT warmup outside CUDA graph capture. All paths execute real
        # kernels and write only temporary KV suffix slots; always restore.
        with torch.inference_mode():
            for fast in (False, True):
                restore()
                invoke(fast, uniforms)
        torch.cuda.synchronize(dev)

        for label, fast in (("torch_reference", False), ("triton_candidate", True)):
            restore()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                g, stream=stream, capture_error_mode="thread_local"
            ), torch.inference_mode():
                outputs = invoke(fast, uniforms)
            graph_holders[label] = g
            graph_outputs[label] = outputs
        restore()

        private_generator = torch.Generator(device=dev).manual_seed(20261122)
        g_random = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            g_random, stream=stream, capture_error_mode="thread_local"
        ), torch.inference_mode():
            private_uniforms = torch.rand(
                (4,), device=dev, dtype=torch.float32,
                generator=private_generator,
            )
            random_result = invoke(True, private_uniforms)
        graph_holders["triton_private_rng"] = g_random
        graph_outputs["triton_private_rng"] = random_result
        torch.cuda.synchronize(dev)

        case_decisions = {"accepted": 0, "rejected": 0}
        for quadruple in TRIPLES:
            uniforms.copy_(
                torch.tensor(quadruple, device=dev, dtype=torch.float32)
            )
            cases = {}
            for label in ("torch_reference", "triton_candidate"):
                restore()
                graph_holders[label].replay()
                torch.cuda.synchronize(dev)
                result = graph_outputs[label]
                kv = snapshot()
                cases[label] = {
                    "result": result,
                    "kv": kv,
                }
            torch_row = cases["torch_reference"]["result"]
            triton_row = cases["triton_candidate"]["result"]
            strict_fields = (
                "draft", "committed", "accepted", "bonus",
                "corrected", "next_cache_len", "next_depth",
                "next_hidden", "next_embedding", "next_sum",
                "before_proposal", "after_proposal",
            )
            for name in strict_fields:
                _strict_close(
                    name.upper(), triton_row[name], torch_row[name]
                )
            for side in ("q", "p", "bonus"):
                triton_prob = _distribution(
                    triton_row[f"{side}_ids"], triton_row[f"{side}_probs"]
                )
                torch_prob = _distribution(
                    torch_row[f"{side}_ids"], torch_row[f"{side}_probs"]
                )
                _strict_close(
                    f"{side.upper()}_DENSE_DISTRIBUTION",
                    triton_prob, torch_prob, atol=1e-6, rtol=1e-6
                )
                # Independent reference from the full logits, beyond
                # two graph variants of the same sparse logic.
                original_prob = sampling_probs(
                    torch_row[f"{side}_logits"],
                    temperature=temperature, top_k=K, top_p=top_p,
                )
                _strict_close(
                    f"{side.upper()}_PYTORCH_SAMPLING_REFERENCE",
                    triton_prob, original_prob, atol=1e-6, rtol=1e-6,
                )
            for index, letter in enumerate(("k", "v")):
                actual = cases["triton_candidate"]["kv"][index]
                expected = cases["torch_reference"]["kv"][index]
                _strict_close(f"FULL_{letter.upper()}_TORCH", actual, expected)
                _strict_close(
                    f"COMMITTED_PREFIX_{letter.upper()}",
                    actual[:, :, :cache_len],
                    (old_k if index == 0 else old_v)[:, :, :cache_len],
                )

            p_ref = sampling_probs(
                torch_row["p_logits"], temperature=temperature,
                top_k=K, top_p=top_p,
            )
            q_ref = sampling_probs(
                torch_row["q_logits"], temperature=temperature,
                top_k=K, top_p=top_p,
            )
            independent = cpu_reference(p_ref, q_ref, quadruple[:3])
            bonus_ref = sampling_probs(
                torch_row["bonus_logits"], temperature=temperature,
                top_k=K, top_p=top_p,
            )[0].float().cpu()
            expected_bonus = int(torch.searchsorted(
                bonus_ref.cumsum(-1).contiguous(),
                torch.tensor(quadruple[3], dtype=torch.float32)
            ).clamp(max=VOCAB-1).item())
            got_bonus = int(triton_row["bonus"][0].item())
            if got_bonus != expected_bonus:
                raise RuntimeError(
                    f"P14_INDEPENDENT_BONUS_DRAW_FAIL "
                    f"u={quadruple[3]} candidate={got_bonus} expected={expected_bonus}"
                )
            expected_sum = (
                saved[2] +
                F.embedding(
                    triton_row["committed"],
                    model.code_predictor.model.codec_embedding.weight[depth]
                ).to(dtype=saved[2].dtype)
            )
            if bool(triton_row["accepted"][0].item()):
                expected_sum = expected_sum + F.embedding(
                    triton_row["bonus"],
                    model.code_predictor.model.codec_embedding.weight[depth+1],
                ).to(dtype=saved[2].dtype)
            _strict_close("INDEPENDENT_COMMITTED_CODE_SUM", triton_row["next_sum"], expected_sum)
            got = {
                "draft": int(triton_row["draft"][0].item()),
                "accepted": bool(triton_row["accepted"][0].item()),
                "committed": int(triton_row["committed"][0].item()),
            }
            if got != independent:
                raise RuntimeError(
                    f"P14_CPU_INDEPENDENT_ACCEPT_RESIDUAL_FAIL "
                    f"u={quadruple} actual={got} independent={independent}"
                )
            valid_next = (
                (got["accepted"] and
                 int(triton_row["next_cache_len"].item()) == cache_len + 2
                 and int(triton_row["next_depth"].item()) == depth + 2)
                or
                (not got["accepted"] and
                 int(triton_row["next_cache_len"].item()) == cache_len + 1
                 and int(triton_row["next_depth"].item()) == depth + 1)
            )
            if not valid_next:
                raise RuntimeError("P14_GPU_LOGICAL_KV_BONUS_BRANCH_STATE_FAIL")
            branch = "accepted" if got["accepted"] else "rejected"
            case_decisions[branch] += 1
            report["fixed_uniform_cases"].append({
                "uniforms": list(quadruple),
                **got,
                "bonus_code_computed_even_if_rejected": int(
                    triton_row["bonus"][0].item()
                ),
                "bonus_committed": got["accepted"],
                "accept_probability": float(
                    triton_row["accept_probability"][0].item()
                ),
                "residual_mass": float(
                    triton_row["residual_mass"][0].item()
                ),
                "q_p_bonus_distribution_parity": True,
                "independent_bonus_sampling_parity": True,
                "independent_embedding_sum_parity": True,
                "target_hidden_parity": True,
                "complete_kv_parity": True,
                "committed_embedding_sum_parity": True,
            })

        report["sampled_branch_count"] = case_decisions
        # The private generator is independent of the request RNG, and
        # graph replay must advance its per-graph draw sequence.
        draws = []
        for _ in range(5):
            restore()
            g_random.replay()
            torch.cuda.synchronize(dev)
            draws.append(tuple(float(x) for x in private_uniforms.cpu().tolist()))
        if len(set(draws)) != len(draws):
            raise RuntimeError("P14_PRIVATE_GRAPH_RNG_NOT_ADVANCING")
        report["private_rng_replay_unique"] = True

        def replay_ms(g: torch.cuda.CUDAGraph) -> float:
            for _ in range(6):
                g.replay()
            torch.cuda.synchronize(dev)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                g.replay()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / repeats
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError("P14_INVALID_CUDA_GRAPH_EVENT_COST")
            return ms

        trials = {
            key: [] for key in (
                "torch_reference", "triton_candidate", "triton_private_rng"
            )
        }
        orders = (
            ("torch_reference", "triton_candidate", "triton_private_rng"),
            ("triton_private_rng", "triton_candidate", "torch_reference"),
            ("triton_candidate", "torch_reference", "triton_private_rng"),
            ("triton_private_rng", "torch_reference", "triton_candidate"),
        )
        for order in orders:
            for label in order:
                restore()
                trials[label].append(replay_ms(graph_holders[label]))
        medians = {
            label: statistics.median(records)
            for label, records in trials.items()
        }
        report["cuda_event_ms_blocks"] = trials
        report["median_cuda_graph_ms"] = medians
        report["relative_fast_vs_torch"] = (
            medians["triton_candidate"] / medians["torch_reference"]
        )
        report["one_round_event_delta_ms"] = (
            medians["triton_candidate"] - medians["torch_reference"]
        )
        report["private_vs_fixed_rng_delta_ms"] = (
            medians["triton_private_rng"] - medians["triton_candidate"]
        )
        report["status"] = (
            "P14_REAL_CAUSAL_ONE_ROUND_Q_VERIFIER_P_RESIDUAL_BONUS_"
            "GPU_GRAPH_PARITY_PASS_NOT_14DEPTH_OR_E2E"
        )
        report["limitations"] = [
            "Only one real round from original committed depth checkpoint",
            "Static KV cache_len; next GPU scalar not fed to next Predictor call",
            "Bonus head/draw computed unconditionally and masked on rejection",
            "Private graph RNG proves advancement, not native seeded RNG parity",
            "RNG replay and every graph capture write temporary KV, restored",
            "No 14-depth GPU conditional loop, no live token/code/embedding mutation",
            "No real E2E HTTP speedup, no speaker/quality parity inference",
        ]
        if not (
            torch.equal(hidden, saved[0])
            and torch.equal(last, saved[1])
            and torch.equal(summed, saved[2])
        ):
            raise RuntimeError("P14_ON_POLICY_INPUT_TENSORS_MUTATED")
        return report
    finally:
        # Explicit temporary KV ownership fence: no speculative
        # candidate physical cache slot may leak into the live K2 path.
        torch.cuda.synchronize(dev)
        restore()
        torch.cuda.synchronize(dev)
        _strict_close("RESTORED_K_CACHE", cache_k[:, :1, :16], old_k)
        _strict_close("RESTORED_V_CACHE", cache_v[:, :1, :16], old_v)


def maybe_probe(
    model, *,
    frame: int, proposal_depth: int,
    cache_len: int, draft_hidden: torch.Tensor,
    last_embedding: torch.Tensor, accumulated_before: torch.Tensor,
    temperature: float, top_k: int, top_p: float,
) -> None:
    root = os.environ.get(ENV)
    if not root:
        return
    conflicts = (
        "SGLANG_OMNI_QTTS_CP_P4_DIR",
        "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_P7_DIR",
        "SGLANG_OMNI_QTTS_CP_P8_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR",
        "SGLANG_OMNI_QTTS_CP_P9B_DIR",
        "SGLANG_OMNI_QTTS_CP_P10A_DIR",
        "SGLANG_OMNI_QTTS_CP_P10B_DIR",
        "SGLANG_OMNI_QTTS_CP_P11A_DIR",
        "SGLANG_OMNI_QTTS_CP_P12A_DIR",
        "SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE",
    )
    if any(os.environ.get(key) for key in conflicts):
        raise RuntimeError(
            "P14_CONFLICTING_HISTORICAL_PROBE_OR_HYBRID_CONFIGURATION"
        )
    seen = getattr(model, "_qt_p14_seen_depths", None)
    if seen is None:
        seen = set()
        model._qt_p14_seen_depths = seen
    if proposal_depth not in DEPTHS or proposal_depth in seen:
        return
    out = Path(root).expanduser().resolve()
    try:
        report = capture_real_causal_round(
            model, depth=proposal_depth-1, cache_len=cache_len,
            draft_hidden=draft_hidden, last_embedding=last_embedding,
            accumulated_before=accumulated_before,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repeats=32,
        )
    except BaseException:
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        (out / f"failure_pid{os.getpid()}_depth{proposal_depth:02d}.txt").write_text(
            traceback.format_exc()
        )
        raise
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    report.update({"pid": os.getpid(), "frame": frame})
    file = out / (
        f"causal_pid{os.getpid()}_frame{frame:04d}_depth{proposal_depth:02d}.json"
    )
    with file.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    seen.add(proposal_depth)

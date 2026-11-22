# SPDX-License-Identifier: Apache-2.0
"""P17 research-only NVIDIA-style multi-head, single-wide-verifier K2/3/4.

For last committed code depth d (zero-based LM head index d), ALL K-1
drafts use the SAME old hidden and distinct heads[d:d+K-1]. They are
sampled BEFORE ONE packed K-token verifier at the existing committed
KV offset, then its distinct target heads produce p_0..p_(K-2) and
the bonus p_(K-1). Each conditional acceptance is exact positive
residual; a first rejection stops commitment, not physical verification.

This is a SOURCE-LOCKED component research probe, NOT a production
speculative decoder, native seeded sampler, optimized fused multi-
proposal acceptance kernel or full 14-depth GPU loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang_omni.utils import predictor_layers as pl
from .code_predictor_spec_p10a_dynamic_kv import (
    decode_attention_with_device_end,
)
from .code_predictor_spec_p12a_selector import fast_selector
from .code_predictor_spec_p14_causal_round import (
    single_support, gpu_accept,
)
from .code_predictor_spec_p15_two_round import _project

WIDTHS = (2, 3, 4)
VOCAB = 2048
TOP_K = 50
UNIFORM_COUNT = 13  # 3 drafts x 4 uniforms + bonus
SCHEMA = "qwen3tts-cpspec-p17-real-multihead-single-wide-verifier-v1"


def prefix_accept_count(flags: tuple[bool, ...]) -> int:
    """Independent CPU gate: accept until FIRST rejection, then stop."""
    result = 0
    for accepted in flags:
        if not accepted:
            break
        result += 1
    return result


def check_wide_backend(model, width: int, cache_len: int) -> None:
    """Never silently replace the optimized serving fused backend."""
    fused = model.predictor_fused_layers
    if width not in WIDTHS or (
        model.predictor_cutedsl_layers is not None
        or fused is None or not fused.covers(width)
        or len(model.code_predictor.model.layers) != 5
        or model.predictor_k_cache.shape[2] < cache_len + width
        or model.predictor_v_cache.shape[2] < cache_len + width
    ):
        raise RuntimeError("P17_FUSED_WIDE_VERIFIER_BACKEND_UNSUPPORTED")
    shape = fused.shape
    group = shape.num_q_heads // shape.num_kv_heads
    if width * group > pl.ATTENTION_ROWS:
        raise RuntimeError(
            "P17_FUSED_ATTENTION_TILE_TOO_NARROW: requires new matched "
            f"static attention tile for width={width}, group={group}; "
            "do not silently truncate query rows"
        )


def prepare_head_bank(model, *, depth: int, width: int) -> torch.Tensor:
    """Prepack actual loaded future LM-head weights, no fake draft model."""
    weights = []
    heads = model.code_predictor.lm_head
    for j in range(width):
        head = heads[depth + j]
        weight = getattr(head, "weight", None)
        if (
            weight is None or weight.ndim != 2
            or weight.shape[0] != VOCAB or weight.device.type != "cuda"
            or getattr(head, "bias", None) is not None
        ):
            raise RuntimeError("P17_PLAIN_TP1_LM_HEAD_WEIGHTS_REQUIRED")
        weights.append(weight.detach())
    return torch.stack(weights, dim=0).contiguous()


def batched_head_logits(
    states: torch.Tensor, bank: torch.Tensor,
) -> torch.Tensor:
    """One GPU batched GEMM over distinct heads on [rows,1,H] states."""
    if (
        states.ndim != 3 or states.shape[1] != 1
        or bank.ndim != 3 or states.shape[0] != bank.shape[0]
        or states.shape[2] != bank.shape[2]
        or states.dtype != bank.dtype
    ):
        raise ValueError("P17_BATCHED_HEAD_LAYOUT_MISMATCH")
    return torch.bmm(states, bank.transpose(1, 2))[:, 0, :].contiguous()


def make_q_proposals(
    model, *, depth: int, width: int, old_hidden: torch.Tensor,
    head_bank: torch.Tensor, uniforms: torch.Tensor,
    temperature: float, top_p: float, accumulation_dtype: torch.dtype,
):
    """Independent head rows share SAME hidden; no verifier-dependent draft."""
    n = width - 1
    q_logits = batched_head_logits(old_hidden.expand(n, -1, -1),
                                   head_bank[:n])
    q_ids, q_probs, drafts, embeds = [], [], [], []
    for j in range(n):
        ids, _, probs, code = fast_selector(
            q_logits[j:j+1], uniforms[4*j:4*j+1],
            temperature=temperature, top_p=top_p,
        )
        q_ids.append(ids[0])
        q_probs.append(probs[0])
        drafts.append(code[0])
        _, projected = _project(
            model, depth + j, code, accumulation_dtype,
        )
        embeds.append(projected)
    return (
        q_logits, torch.stack(q_ids), torch.stack(q_probs),
        torch.stack(drafts), torch.cat(embeds, dim=1),
    )


def wide_device_fused_predictor(
    model, *, packed: torch.Tensor, cache_len_gpu: torch.Tensor,
    offsets_gpu: torch.Tensor,
) -> torch.Tensor:
    """P10A device-END generalized without changing production path."""
    width = int(packed.shape[1])
    # This check is host shape/contract only, not GPU data branching.
    if packed.shape[0] != 1 or packed.dtype != model.predictor_k_cache.dtype:
        raise ValueError("P17_VERIFY_REQUIRES_C1_NATIVE_DTYPE")
    if offsets_gpu.shape != (width,) or cache_len_gpu.ndim != 0:
        raise ValueError("P17_DEVICE_OFFSET_SHAPE_MISMATCH")
    if cache_len_gpu.dtype != torch.long:
        raise ValueError("P17_DEVICE_OFFSET_INT64_REQUIRED")
    # True on-policy cache_len was validated on the host before capture.
    # Never branch on a device scalar or assume a magic static offset here.
    check_wide_backend(model, width, 0)
    fused = model.predictor_fused_layers
    shape = fused.shape
    end = cache_len_gpu + width
    positions = cache_len_gpu + offsets_gpu
    hidden = packed.clone().reshape(width, shape.hidden_size)
    residual = fused.residual[:width]
    q_out = fused.q[:width]
    attention_output = fused.attention_output[:width]
    activated = fused.activated[:width]
    for i, layer in enumerate(model.code_predictor.model.layers):
        attn = layer.self_attn
        k = model.predictor_k_cache[i, :1]
        v = model.predictor_v_cache[i, :1]
        pl.attention_inputs(
            x=hidden, tokens_per_row=width,
            norm=layer.input_layernorm, attention=attn,
            q_out=q_out, positions=positions, k_cache=k, v_cache=v,
            partials=fused.partials, sum_sq_partials=fused.sum_sq_partials,
            counters=fused.counters, shape=shape,
            fitted_stages=fused.fitted_stages,
        )
        use_gdc = is_arch_support_pdl()
        decode_attention_with_device_end[(1, shape.num_kv_heads)](
            q_out, k, v,
            k.stride(0), k.stride(1), k.stride(2),
            attention_output, end, shape.head_dim**-0.5,
            T=width, HEAD_DIM=shape.head_dim,
            NUM_Q_HEADS=shape.num_q_heads, NUM_KV_HEADS=shape.num_kv_heads,
            Q_ROW_STRIDE=q_out.stride(0),
            BLOCK_Q=pl.ATTENTION_ROWS,
            BLOCK_S=triton.next_power_of_2(k.shape[1]),
            USE_GDC=use_gdc, num_warps=pl.NUM_WARPS,
            launch_pdl=use_gdc,
        )
        pl.o_proj_add(
            attention_output=attention_output.view(
                1, width, shape.num_q_heads, shape.head_dim,
            ).transpose(1, 2),
            tokens_per_row=width, weight=attn.o_proj.weight,
            residual_in=hidden, residual_out=residual,
            partials=fused.partials, counters=fused.counters,
            shape=shape, fitted_stages=fused.fitted_stages,
        )
        hidden = residual
        pl.mlp_up(
            residual=residual, norm=layer.post_attention_layernorm,
            weight=layer.mlp.gate_up_proj.weight,
            activated=activated, shape=shape,
            fitted_stages=fused.fitted_stages,
        )
        pl.down_add(
            activated=activated, weight=layer.mlp.down_proj.weight,
            residual=residual, partials=fused.partials,
            counters=fused.counters, shape=shape,
            fitted_stages=fused.fitted_stages,
        )
    return model.code_predictor.model.norm(residual).reshape(
        1, width, shape.hidden_size,
    ).clone()


def static_fused_verifier(
    model, *, packed: torch.Tensor, cache_len: int,
) -> torch.Tensor:
    """Independent STATIC-END fused kernel/reference, C1 width 2/3/4."""
    width = int(packed.shape[1])
    check_wide_backend(model, width, cache_len)
    return model.predictor_fused_layers.forward(
        layers=model.code_predictor.model.layers,
        final_norm=model.code_predictor.model.norm,
        token_embeds=packed.clone(), batch_size=1,
        cache_len=cache_len,
        positions=model.predictor_positions[cache_len:cache_len+width],
        k_cache=model.predictor_k_cache,
        v_cache=model.predictor_v_cache,
    ).clone()


def make_wide_round(
    model, *, depth: int, width: int,
    cache_len_gpu: torch.Tensor,
    old_hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated: torch.Tensor, uniforms: torch.Tensor,
    head_bank: torch.Tensor, offset_ids: torch.Tensor,
    temperature: float, top_p: float,
) -> dict[str, torch.Tensor]:
    """Real parallel q heads + ONE causal wide verify + exact GPU residual."""
    n = width - 1
    q_logits, q_ids, q_probs, drafts, draft_embeddings = make_q_proposals(
        model, depth=depth, width=width, old_hidden=old_hidden,
        head_bank=head_bank, uniforms=uniforms,
        temperature=temperature, top_p=top_p,
        accumulation_dtype=accumulated.dtype,
    )
    packed = torch.cat((last_embedding, draft_embeddings), dim=1)
    verified = wide_device_fused_predictor(
        model, packed=packed, cache_len_gpu=cache_len_gpu,
        offsets_gpu=offset_ids,
    )
    p_logits = batched_head_logits(
        verified[0].unsqueeze(1), head_bank,
    )
    target_ids, target_probs, accepts, candidates = [], [], [], []
    for j in range(n):
        pids, probs = single_support(
            p_logits[j:j+1],
            temperature=temperature, top_p=top_p, fast=True,
        )
        result = gpu_accept(
            pids, probs, q_ids[j:j+1], q_probs[j:j+1],
            uniforms[4*j:4*j+4], drafts[j:j+1],
        )
        target_ids.append(pids[0])
        target_probs.append(probs[0])
        accepts.append(result["accepted"][0])
        candidates.append(result["committed"][0])
    flags = torch.stack(accepts)
    survived = torch.cumprod(flags.to(torch.long), dim=0)
    active = torch.cat((torch.ones_like(survived[:1]), survived[:-1]))
    accepted_prefix = survived.sum()
    committed_count = accepted_prefix + 1
    candidates = torch.stack(candidates)
    negative = torch.full_like(candidates, -1)
    chosen = torch.where(active.bool(), candidates, negative)

    # Bonus head at the verifier's trailing row, only LOGICALLY committed
    # after every draft is accepted. Physical work is unconditional, like
    # the NVIDIA wide verifier's already-computed final p row.
    _, _, _, bonus = fast_selector(
        p_logits[n:n+1], uniforms[4*n:4*n+1],
        temperature=temperature, top_p=top_p,
    )
    # cumprod on Long flags returns an int64 tensor. torch.where requires
    # a boolean condition on both the bonus-code and bonus-embedding paths.
    # Keep this as a device scalar; never synchronize to the host.
    full_accept = survived[-1].to(dtype=torch.bool).reshape(())
    code_list = torch.cat((
        chosen,
        torch.where(full_accept, bonus[0], torch.full_like(bonus[0], -1))
        .reshape(1),
    ))
    raw_sum = accumulated
    projected_options = []
    for j in range(n):
        raw, projection = _project(
            model, depth+j, candidates[j:j+1], accumulated.dtype,
        )
        raw_sum = raw_sum + torch.where(
            active[j].reshape(()).bool(), raw,
            torch.zeros_like(raw),
        )
        projected_options.append(projection[0, 0])
    bonus_raw, bonus_projection = _project(
        model, depth+n, bonus, accumulated.dtype,
    )
    raw_sum = raw_sum + torch.where(
        full_accept, bonus_raw, torch.zeros_like(bonus_raw),
    )
    projected_options.append(bonus_projection[0, 0])
    chosen_pos = (committed_count - 1).reshape(1, 1).expand(
        1, projected_options[0].numel()
    )
    projected_next = torch.gather(
        torch.stack(projected_options), 0, chosen_pos,
    ).reshape(1, 1, -1)
    # A rejection at draft j leaves the verifier row j as next draft
    # hidden. Full acceptance leaves the final verifier row n.
    next_hidden = torch.gather(
        verified[0], 0, chosen_pos,
    ).reshape(1, 1, -1)
    return {
        "drafts": drafts,
        "accepted_flags": flags,
        "accepted_prefix": accepted_prefix,
        "committed_count": committed_count,
        "committed_codes": code_list,
        "bonus_code": bonus,
        "q_logits": q_logits,
        "p_logits": p_logits,
        "q_ids": q_ids,
        "q_probs": q_probs,
        "p_ids": torch.stack(target_ids),
        "p_probs": torch.stack(target_probs),
        "verified_hidden": verified,
        "next_hidden": next_hidden,
        "next_embedding": projected_next,
        "next_sum": raw_sum,
        "next_depth": cache_len_gpu - 1 + committed_count,
        "next_cache_len": cache_len_gpu + committed_count,
    }

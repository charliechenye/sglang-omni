# SPDX-License-Identifier: Apache-2.0
"""P10A: real 5-layer Predictor dynamic CUDA-device KV offset, no conditional tree.

One FIXED CUDA graph, a single GPU scalar (cache_len), and a Triton attention
kernel reading an on-device END pointer. Reuses the SAME optimized
FusedPredictorLayers QKV/norm, cache ownership and GEMV/MLP kernels as the
existing static fixed-offset verifier. Does not use 4-way cond/switch or CPU
.item() per frame. Test-only C1 probe from a live on-policy depth-14 K2 state.

This is a *component parity and cost gate*, NOT a full GPU speculative
decoder, RNG/accept/bonus implementation, or serving throughput proof.
The matched backend is the FusedPredictorLayers path. If the running model
instead uses CuTeDSL or plain unfused kernels, FAIL rather than benchmark
a different implementation against its existing path.
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
from sglang.kernels.jit.utils import is_arch_support_pdl
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from sglang_omni.utils import predictor_layers as pl

from .code_predictor_spec_p7_dispatch import compare

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

ENV = "SGLANG_OMNI_QTTS_CP_P10A_DIR"
SCHEMA = "qwen3tts-cpspec-p10a-dynamic-device-kv-fused-predictor-v1"
OFFSETS = (2, 6, 10, 14)


@triton.jit
def decode_attention_with_device_end(
    Q, K_CACHE, V_CACHE,
    cache_stride_row, cache_stride_slot, cache_stride_head,
    OUT, END_PTR, scale,
    T: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_ROW_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    """Exact math/layout of existing decode_attention_kernel, but END is GPU data.

    All lanes load the SAME int64 scalar end; inactive KV is masked before
    MMA, and each query sees only causal slots end-T+t. This avoids scalar
    extraction on the CPU and uses the same CTA/Pipeline kernel schedule.
    """
    row_group = tl.program_id(0)
    kv_head = tl.program_id(1)
    group: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    if USE_GDC:
        gdc_wait()
    rq = tl.arange(0, BLOCK_Q)
    rs = tl.arange(0, BLOCK_S)
    rd = tl.arange(0, HEAD_DIM)
    end = tl.load(END_PTR).to(tl.int32)
    token = rq // group
    is_query = rq < T * group
    q_offsets = (
        (row_group * T + token)[:, None] * Q_ROW_STRIDE
        + (kv_head * group + rq % group)[:, None] * HEAD_DIM
        + rd[None, :]
    )
    out_offsets = (
        (row_group * T + token)[:, None] * (NUM_Q_HEADS * HEAD_DIM)
        + (kv_head * group + rq % group)[:, None] * HEAD_DIM
        + rd[None, :]
    )
    cache_offsets = (
        row_group * cache_stride_row + kv_head * cache_stride_head
        + rs[:, None] * cache_stride_slot + rd[None, :]
    )
    q = tl.load(Q + q_offsets, mask=is_query[:, None], other=0.0)
    k = tl.load(K_CACHE + cache_offsets, mask=(rs < end)[:, None], other=0.0)
    v = tl.load(V_CACHE + cache_offsets, mask=(rs < end)[:, None], other=0.0)
    scores = tl.dot(q, tl.trans(k)) * scale
    visible = rs[None, :] < end - T + token[:, None] + 1
    scores = tl.where(visible, scores, float("-inf"))
    weights = libdevice.exp(scores - tl.max(scores, 1)[:, None])
    attended = tl.dot(weights.to(tl.bfloat16), v) / tl.sum(weights, 1)[:, None]
    tl.store(OUT + out_offsets, attended.to(tl.bfloat16), mask=is_query[:, None])
    if USE_GDC:
        gdc_launch_dependents()


def dynamic_decode_attention(
    *, q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    end_ptr: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Launch with fixed grid and device end scalar; no host-dependent routing."""
    if (
        end_ptr.ndim != 0 or not end_ptr.is_cuda
        or end_ptr.dtype not in (torch.int32, torch.int64)
        or q.shape[0] != 2 or k_cache.shape[0] != 1
        or k_cache.shape != v_cache.shape
    ):
        raise ValueError("P10A dynamic attention requires 2 q rows and C1 KV")
    kv_heads, head_dim = k_cache.shape[2:]
    use_gdc = is_arch_support_pdl()
    decode_attention_with_device_end[(1, kv_heads)](
        q, k_cache, v_cache,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        out, end_ptr, head_dim ** -0.5,
        T=2,
        HEAD_DIM=head_dim,
        NUM_Q_HEADS=q.shape[1] // head_dim,
        NUM_KV_HEADS=kv_heads,
        Q_ROW_STRIDE=q.stride(0),
        BLOCK_Q=pl.ATTENTION_ROWS,
        BLOCK_S=triton.next_power_of_2(k_cache.shape[1]),
        USE_GDC=use_gdc,
        num_warps=pl.NUM_WARPS,
        launch_pdl=use_gdc,
    )


def dynamic_fused_predictor(
    model: "Qwen3TTSTalker",
    *,
    pair: torch.Tensor,
    cache_len_gpu: torch.Tensor,
    indices_gpu: torch.Tensor,
) -> torch.Tensor:
    """Same fused 5-layer path as FusedPredictorLayers.forward, GPU end/positions.

    Never reads cache_len_gpu on CPU. Does not mutate pair; the fused
    layers' existing owner scratch buffers are shared with static oracle
    but never executed concurrently.
    """
    fused = model.predictor_fused_layers
    if fused is None or not fused.covers(2):
        raise RuntimeError("P10A needs optimized fused Predictor backend")
    shape = fused.shape
    positions = cache_len_gpu + indices_gpu
    end = cache_len_gpu + 2
    hidden = pair.clone().reshape(2, shape.hidden_size)
    residual = fused.residual[:2]
    q_out = fused.q[:2]
    attention_output = fused.attention_output[:2]
    activated = fused.activated[:2]
    for layer_idx, layer in enumerate(model.code_predictor.model.layers):
        attn = layer.self_attn
        k = model.predictor_k_cache[layer_idx, :1]
        v = model.predictor_v_cache[layer_idx, :1]
        pl.attention_inputs(
            x=hidden,
            tokens_per_row=2,
            norm=layer.input_layernorm,
            attention=attn,
            q_out=q_out,
            positions=positions,
            k_cache=k,
            v_cache=v,
            partials=fused.partials,
            sum_sq_partials=fused.sum_sq_partials,
            counters=fused.counters,
            shape=shape,
            fitted_stages=fused.fitted_stages,
        )
        dynamic_decode_attention(
            q=q_out, k_cache=k, v_cache=v, end_ptr=end,
            out=attention_output,
        )
        pl.o_proj_add(
            attention_output=attention_output.view(
                1, 2, shape.num_q_heads, shape.head_dim,
            ).transpose(1, 2),
            tokens_per_row=2,
            weight=attn.o_proj.weight,
            residual_in=hidden,
            residual_out=residual,
            partials=fused.partials,
            counters=fused.counters,
            shape=shape,
            fitted_stages=fused.fitted_stages,
        )
        hidden = residual
        pl.mlp_up(
            residual=residual,
            norm=layer.post_attention_layernorm,
            weight=layer.mlp.gate_up_proj.weight,
            activated=activated,
            shape=shape,
            fitted_stages=fused.fitted_stages,
        )
        pl.down_add(
            activated=activated,
            weight=layer.mlp.down_proj.weight,
            residual=residual,
            partials=fused.partials,
            counters=fused.counters,
            shape=shape,
            fitted_stages=fused.fitted_stages,
        )
    return model.code_predictor.model.norm(residual).reshape(1, 2, shape.hidden_size).clone()


def capture_real_dynamic_kv(
    model: "Qwen3TTSTalker",
    *,
    token_embeds: torch.Tensor,
    repeats: int = 32,
) -> dict:
    if (
        token_embeds.shape[:2] != (1, 2)
        or token_embeds.device.type != "cuda"
        or token_embeds.dtype != model.predictor_k_cache.dtype
        or model.predictor_k_cache.shape[2] < 16
        or not 8 <= repeats <= 256
        or torch.cuda.is_current_stream_capturing()
        or model.predictor_fused_layers is None
        or not model.predictor_fused_layers.covers(2)
        or model.predictor_cutedsl_layers is not None
    ):
        raise RuntimeError(
            "P10A_BACKEND_OR_SHAPE_UNSUPPORTED: optimized fused Predictor C1 "
            "required; do not replace production CuTeDSL or other backend"
        )
    device = token_embeds.device
    k_cache, v_cache = model.predictor_k_cache, model.predictor_v_cache
    original_k = k_cache[:, :1, :16].clone()
    original_v = v_cache[:, :1, :16].clone()
    pair = token_embeds.detach().clone().contiguous()
    initial_pair = pair.clone()
    offset = torch.tensor(2, device=device, dtype=torch.int64)
    indices_gpu = torch.arange(2, device=device, dtype=torch.int64)
    def restore():
        k_cache[:, :1, :16].copy_(original_k)
        v_cache[:, :1, :16].copy_(original_v)
    def snapshot():
        return (k_cache[:, :1, :16].clone(), v_cache[:, :1, :16].clone())
    def candidate():
        return dynamic_fused_predictor(
            model, pair=pair, cache_len_gpu=offset, indices_gpu=indices_gpu
        )
    def static(depth: int):
        return model.predictor_forward_tokens(
            token_embeds=pair.clone(), batch_size=1, cache_len=depth
        ).clone()
    report = {
        "schema": SCHEMA,
        "status": "P10A_NOT_VALIDATED",
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "gpu": torch.cuda.get_device_name(device),
        "predictor_layers": len(model.code_predictor.model.layers),
        "backend": "same_production_fused_predictor_layers",
        "offsets": list(OFFSETS),
        "cases": [],
        "repeats": repeats,
    }
    capture_completed = False
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    try:
        # Compile kernels and warm up all four offsets before capturing.
        with torch.inference_mode():
            for depth in OFFSETS:
                restore()
                offset.fill_(depth)
                candidate()
                restore()
                static(depth)
        torch.cuda.synchronize(device)
        restore()
        g_dynamic = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            g_dynamic, stream=side, capture_error_mode="thread_local"
        ), torch.inference_mode():
            out_dynamic = candidate()
        static_graphs = {}
        for depth in OFFSETS:
            restore()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                g, stream=side, capture_error_mode="thread_local"
            ), torch.inference_mode():
                out = static(depth)
            static_graphs[depth] = (g, out)
        torch.cuda.synchronize(device)
        capture_completed = True

        def replay_cost(g: torch.cuda.CUDAGraph) -> float:
            for _ in range(4):
                g.replay()
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                g.replay()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) / repeats

        for depth in OFFSETS:
            restore()
            oracle = static(depth).clone()
            oracle_k, oracle_v = snapshot()

            restore()
            offset.fill_(depth)
            g_dynamic.replay()
            torch.cuda.synchronize(device)
            dyn_out = out_dynamic.clone()
            dyn_k, dyn_v = snapshot()

            restore()
            gs, gs_out = static_graphs[depth]
            gs.replay()
            torch.cuda.synchronize(device)
            static_out = gs_out.clone()
            static_k, static_v = snapshot()

            checks = {
                "dynamic_hidden_vs_eager": compare(dyn_out, oracle),
                "dynamic_k_vs_eager": compare(dyn_k, oracle_k),
                "dynamic_v_vs_eager": compare(dyn_v, oracle_v),
                "fixed_graph_hidden_vs_eager": compare(static_out, oracle),
                "fixed_graph_k_vs_eager": compare(static_k, oracle_k),
                "fixed_graph_v_vs_eager": compare(static_v, oracle_v),
                "original_prefix_k_untouched": compare(
                    dyn_k[:, :, :depth], original_k[:, :, :depth],
                ),
                "original_prefix_v_untouched": compare(
                    dyn_v[:, :, :depth], original_v[:, :, :depth],
                ),
                "original_input_untouched": compare(pair, initial_pair),
            }
            # We reuse the exact same fused model kernels, weight format,
            # RoPE/cache indices and BF16 rounding. Device-END must be
            # bit-identical to the static-end reference, not merely close.
            passed = all(
                x["close"] and x["max_abs"] == 0.0
                for x in checks.values()
            )
            case = {"depth": depth, "checks": checks, "pass": passed}
            report["cases"].append(case)
            if not passed:
                bad = {name: v["max_abs"] for name, v in checks.items() if not v["close"]}
                raise RuntimeError(
                    f"P10A_DYNAMIC_DEVICE_KV_PARITY_FAIL depth={depth} bad={bad}"
                )

        # Accurate matched static vs dynamic event envelopes from same
        # worker and same weights, isolated from oracle execution.
        for row in report["cases"]:
            depth = row["depth"]
            restore()
            offset.fill_(depth)
            row["dynamic_cuda_graph_ms"] = replay_cost(g_dynamic)
            restore()
            row["fixed_cuda_graph_ms"] = replay_cost(static_graphs[depth][0])
            row["delta_dynamic_minus_fixed_ms"] = (
                row["dynamic_cuda_graph_ms"] - row["fixed_cuda_graph_ms"]
            )
        report["status"] = (
            "P10A_ONE_GRAPH_GPU_DYNAMIC_KV_FUSED_PREDICTOR_PARITY_PASS_"
            "NOT_FULL_K2_OR_SERVING_SPEEDUP"
        )
        report["limitations"] = [
            "Supports optimized FusedPredictorLayers, fails closed for alternate backends",
            "One captured graph and true GPU device depth, on real frozen K2 embeddings",
            "Independent fixed-depth captured baseline for every offset",
            "No GPU sampling / bonus / dynamic K2 loop or serving speedup",
            "CUDA-event replay envelopes, not model kernel-busy time",
        ]
    finally:
        # A failed capture can invalidate the stream and cleanup could hide
        # the first error; after successful capture always restore original
        # live model KV, including on a parity failure.
        if capture_completed:
            torch.cuda.synchronize(device)
            restore()
            torch.cuda.synchronize(device)
    return report


def maybe_probe(
    model: "Qwen3TTSTalker", *,
    frame_index: int,
    proposal_depth: int,
    token_embeds: torch.Tensor,
) -> None:
    root = os.environ.get(ENV)
    if not root:
        return
    if any(os.environ.get(name) for name in (
        "SGLANG_OMNI_QTTS_CP_P4_DIR", "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_P7_DIR", "SGLANG_OMNI_QTTS_CP_P8_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR", "SGLANG_OMNI_QTTS_CP_P9B_DIR",
        "SGLANG_OMNI_QTTS_CP_P10B_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    )):
        raise RuntimeError("P10A cannot coexist with other K2 profiling probes")
    if proposal_depth != 14 or getattr(model, "_qt_cp_p10a_done", False):
        return
    dest = Path(root).expanduser().resolve()
    try:
        result = capture_real_dynamic_kv(
            model, token_embeds=token_embeds, repeats=32
        )
    except BaseException:
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        (dest / f"failure_pid{os.getpid()}.txt").write_text(traceback.format_exc())
        raise
    dest.mkdir(parents=True, exist_ok=False)
    result.update({"pid": os.getpid(), "frame": frame_index})
    file = dest / f"dynamic_kv_pid{os.getpid()}_frame{frame_index:04d}.json"
    with file.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    model._qt_cp_p10a_done = True

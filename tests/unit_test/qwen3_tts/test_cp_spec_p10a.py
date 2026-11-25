# SPDX-License-Identifier: Apache-2.0
"""P10A matched dynamic-device KV attention vs native static Predictor kernel."""
from __future__ import annotations

import inspect

import pytest
import torch

from sglang_omni.models.qwen3_tts.code_predictor_spec_p10a_dynamic_kv import (
    OFFSETS, dynamic_decode_attention, dynamic_fused_predictor,
)
from sglang_omni.utils import predictor_layers as pl


def test_p10a_cpu_checks_no_python_depth_extract_in_hot_path():
    assert OFFSETS == (2, 6, 10, 14)
    source = inspect.getsource(dynamic_fused_predictor)
    assert "cache_len_gpu + indices_gpu" in source
    assert "cache_len_gpu + 2" in source
    assert ".item(" not in source
    assert "torch.cond" not in source
    assert "cond_op" not in source
    attention_source = inspect.getsource(dynamic_decode_attention)
    assert "end_ptr=end" not in attention_source  # launcher uses direct GPU argument
    assert "end_ptr," in attention_source


@pytest.mark.accelerator
def test_p10a_h200_one_captured_dynamic_gpu_end_matches_static_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/H200 required")
    if not pl.HAS_TRITON:
        pytest.skip("Triton dynamic attention backend unavailable")
    torch.manual_seed(20261122)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_heads = 32
    kv_heads = 4
    head_dim = 128
    slots = 17
    q = torch.randn(
        2, q_heads * head_dim, device=device, dtype=dtype
    ).contiguous()
    k = torch.randn(
        1, slots, kv_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    original_k = k.clone()
    original_v = v.clone()
    depth = torch.tensor(2, device=device, dtype=torch.int64)
    end_ptr = depth + 2
    result = torch.empty_like(q)
    with torch.inference_mode():
        dynamic_decode_attention(
            q=q, k_cache=k, v_cache=v, end_ptr=end_ptr, out=result,
        )
        for off in OFFSETS:
            static_out = torch.empty_like(q)
            pl.decode_attention(
                q=q, tokens_per_row=2, k_cache=k, v_cache=v,
                end=off + 2, attention_output=static_out,
            )
    torch.cuda.synchronize(device)
    # end_ptr must be computed INSIDE the capture from the changing GPU depth.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        end = depth + 2
        dynamic_decode_attention(
            q=q, k_cache=k, v_cache=v, end_ptr=end, out=result
        )
    for off in OFFSETS:
        depth.fill_(off)
        graph.replay()
        torch.cuda.synchronize(device)
        expected = torch.empty_like(q)
        pl.decode_attention(
            q=q, tokens_per_row=2, k_cache=k, v_cache=v,
            end=off + 2, attention_output=expected,
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(k, original_k, atol=0, rtol=0)
    torch.testing.assert_close(v, original_v, atol=0, rtol=0)


@pytest.mark.accelerator
def test_p10a_rejects_unmatched_nonfused_backend():
    # The model-level gate is exercised on the actual five-layer checkpoint
    # through post-health P10A ARM. It must never silently benchmark a
    # different backend from the one serving production requests.
    from types import SimpleNamespace
    from sglang_omni.models.qwen3_tts.code_predictor_spec_p10a_dynamic_kv import (
        capture_real_dynamic_kv,
    )

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    fake = SimpleNamespace(
        predictor_k_cache=torch.zeros((1, 1, 17, 4, 128), device="cuda", dtype=torch.bfloat16),
        predictor_v_cache=torch.zeros((1, 1, 17, 4, 128), device="cuda", dtype=torch.bfloat16),
        predictor_fused_layers=None,
        predictor_cutedsl_layers=None,
    )
    pair = torch.zeros((1, 2, 1024), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="P10A_BACKEND_OR_SHAPE_UNSUPPORTED"):
        capture_real_dynamic_kv(fake, token_embeds=pair, repeats=8)

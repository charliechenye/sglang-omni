# SPDX-License-Identifier: Apache-2.0
"""P7 fixed-depth real Predictor CUDA conditional branch diagnostics."""
from __future__ import annotations

import pytest
import torch

import sglang_omni.models.qwen3_tts.sglang_model as model_module
from sglang_omni.models.qwen3_tts.code_predictor_spec_p7_dispatch import (
    capture_two_offsets, compare, SCHEMA,
)


def test_p7_compare_nonempty_cpu_tensors() -> None:
    a = torch.tensor([1., 2.])
    assert compare(a, a.clone())["close"]
    assert compare(a, a.clone())["max_abs"] == 0.0
    assert not compare(a, a + 0.2)["close"]
    with pytest.raises(ValueError):
        compare(a, torch.zeros(3))
    with pytest.raises(ValueError):
        compare(torch.empty(0), torch.empty(0))


@pytest.mark.accelerator
def test_p7_h200_basic_conditional_cuda_graph_selector_changes_output():
    """Distinguish PyTorch cond-node selection from Predictor state corruption."""
    if not torch.cuda.is_available():
        pytest.skip("requires H200/CUDA")
    from torch._higher_order_ops.cond import cond_op
    from torch._higher_order_ops.cudagraph_conditional_nodes import (
        CUDAGraphCaptureControlFlowOpDispatchMode,
    )

    device = torch.device("cuda")
    selector = torch.ones((), device=device, dtype=torch.bool)
    data = torch.tensor([3], device=device, dtype=torch.int64)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))

    def left(x):
        return x + 11

    def right(x):
        return x + 22

    with torch.inference_mode():
        left(data)
        right(data)
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(
            graph, stream=side, capture_error_mode="thread_local"
        ),
        CUDAGraphCaptureControlFlowOpDispatchMode(),
        torch.inference_mode(),
    ):
        output = cond_op(selector, left, right, (data,))

    for chosen, expected in ((True, 14), (False, 25), (True, 14)):
        selector.fill_(chosen)
        graph.replay()
        torch.cuda.synchronize()
        assert output.item() == expected


@pytest.mark.accelerator
def test_p7_h200_real_predictor_conditional_offsets_and_kv_restore(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires real H200/CUDA")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        DTYPE, HIDDEN, NUM_KV_HEADS, HEAD_DIM, MAX_BS, build_talker,
        with_predictor_layers,
    )

    monkeypatch.setattr(
        model_module, "apply_qk_norm", lambda q, k, **_: (q, k),
    )
    dev = torch.device("cuda")
    model = with_predictor_layers(build_talker(dev), 2)
    capacity = 18
    pos = torch.arange(capacity, device=dev, dtype=torch.long)
    # The factory fixture originally has NUM_CODE_GROUPS+1 positions (5).
    # Extending cache slots without extending predictor_positions silently
    # invalidates the cache_len=6 branch.
    model.predictor_positions = pos
    model.predictor_position_rows = pos[:, None].expand(capacity, MAX_BS).contiguous()
    model.predictor_cache_slots = (
        torch.arange(MAX_BS, device=dev, dtype=torch.long)[None, :] * capacity
        + pos[:, None]
    ).contiguous()
    model.predictor_k_cache = torch.zeros(
        (2, MAX_BS, capacity, NUM_KV_HEADS, HEAD_DIM), device=dev, dtype=DTYPE
    )
    model.predictor_v_cache = torch.zeros_like(model.predictor_k_cache)
    model.predictor_k_rows = [
        x.view(MAX_BS * capacity, -1) for x in model.predictor_k_cache
    ]
    model.predictor_v_rows = [
        x.view(MAX_BS * capacity, -1) for x in model.predictor_v_cache
    ]
    torch.manual_seed(20261121)
    embeds = torch.randn(1, 2, HIDDEN, device=dev).to(DTYPE)
    unmodified_embeds = embeds.clone()
    # Explicitly fill the legitimate prefix before the live P7 verifier.
    # Fused BF16 Predictor may overwrite residual input, so each call must
    # have private input and leave embeds immutable.
    with torch.inference_mode():
        for cache_len in (0, 2, 4):
            model.predictor_forward_tokens(
                token_embeds=embeds.clone(), batch_size=1, cache_len=cache_len
            )
        torch.testing.assert_close(embeds, unmodified_embeds, rtol=0, atol=0)
        before_k = model.predictor_k_cache[:, :1, :8].clone()
        before_v = model.predictor_v_cache[:, :1, :8].clone()
        report = capture_two_offsets(model, fixed_embeds=embeds, repeats=8)
    assert report["status"].startswith("P7_REAL_MODEL_TWO_FIXED_DEPTH_CUDA_COND_PASS")
    assert [x["offset"] for x in report["cases"]] == [2, 6, 2]
    assert all(x["pass"] for x in report["cases"])
    assert all(x["checks"]["hidden"]["max_abs"] == 0 for x in report["cases"])
    assert all(
        x["checks"]["persistent_input_unchanged"]["close"]
        for x in report["cases"]
    )
    assert all(
        x["checks"]["original_kv_prefix_k_untouched"]["close"]
        and x["checks"]["original_kv_prefix_v_untouched"]["close"]
        for x in report["cases"]
    )
    torch.testing.assert_close(embeds, unmodified_embeds, atol=0, rtol=0)
    assert set(report["conditional_cuda_event_ms_by_offset"]) == {"2", "6"}
    torch.testing.assert_close(
        model.predictor_k_cache[:, :1, :8], before_k, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache[:, :1, :8], before_v, atol=0, rtol=0
    )

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
    # Explicitly fill the legitimate prefix before the live P7 verifier.
    with torch.inference_mode():
        for cache_len in (0, 2, 4):
            model.predictor_forward_tokens(
                token_embeds=embeds, batch_size=1, cache_len=cache_len
            )
        before_k = model.predictor_k_cache[:, :1, :8].clone()
        before_v = model.predictor_v_cache[:, :1, :8].clone()
        report = capture_two_offsets(model, fixed_embeds=embeds, repeats=8)
    assert report["status"].startswith("P7_REAL_MODEL_TWO_FIXED_DEPTH_CUDA_COND_PASS")
    assert [x["offset"] for x in report["cases"]] == [2, 6, 2]
    assert all(x["pass"] for x in report["cases"])
    assert all(x["checks"]["hidden"]["max_abs"] == 0 for x in report["cases"])
    assert set(report["conditional_cuda_event_ms_by_offset"]) == {"2", "6"}
    torch.testing.assert_close(
        model.predictor_k_cache[:, :1, :8], before_k, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache[:, :1, :8], before_v, atol=0, rtol=0
    )

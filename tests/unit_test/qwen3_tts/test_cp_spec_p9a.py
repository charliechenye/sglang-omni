# SPDX-License-Identifier: Apache-2.0
"""P9A four real Predictor depths: CPU sequence and H200 GPU KV parity."""
from __future__ import annotations

import pytest
import torch

import sglang_omni.models.qwen3_tts.sglang_model as model_module
from sglang_omni.models.qwen3_tts.code_predictor_spec_p9a import (
    CASE_MASKS, OFFSETS, capture_four_real_offsets, reference,
)


def test_p9a_cpu_four_depth_dispatch_and_stage_skip_accounting():
    assert OFFSETS == (2, 6, 10, 14)
    assert len(CASE_MASKS) == 6
    for label, mask in CASE_MASKS:
        row = reference(mask)
        assert 2 <= row["rounds"] <= 4, label
        assert row["rounds"] + row["stage_skips"] == 4, label
        assert all(x in OFFSETS for x in row["offsets"])
        assert len(set(row["offsets"])) == row["rounds"]
    assert reference((False,)*4)["offsets"] == list(OFFSETS)
    assert reference((True,)*4)["offsets"] == [2, 10]
    assert reference((False, True, False, False))["offsets"] == [2, 6, 14]
    with pytest.raises(ValueError, match="four binary"):
        reference((True, False))
    with pytest.raises(ValueError, match="four binary"):
        reference((True, 1, False, False))


@pytest.mark.accelerator
def test_p9a_h200_four_offset_nested_model_whole_kv_parity(
    monkeypatch: pytest.MonkeyPatch,
):
    if not torch.cuda.is_available():
        pytest.skip("H200/CUDA not available")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        DTYPE, HIDDEN, HEAD_DIM, MAX_BS, NUM_KV_HEADS,
        build_talker, with_predictor_layers,
    )
    monkeypatch.setattr(
        model_module, "apply_qk_norm", lambda q, k, **_: (q, k)
    )
    dev = torch.device("cuda")
    model = with_predictor_layers(build_talker(dev), 2)
    capacity = 18
    positions = torch.arange(capacity, device=dev, dtype=torch.long)
    model.predictor_positions = positions
    model.predictor_position_rows = (
        positions[:, None].expand(capacity, MAX_BS).contiguous()
    )
    model.predictor_cache_slots = (
        torch.arange(MAX_BS, device=dev, dtype=torch.long)[None, :] * capacity
        + positions[:, None]
    ).contiguous()
    model.predictor_k_cache = torch.zeros(
        (2, MAX_BS, capacity, NUM_KV_HEADS, HEAD_DIM), device=dev, dtype=DTYPE
    )
    model.predictor_v_cache = torch.zeros_like(model.predictor_k_cache)
    model.predictor_k_rows = [
        layer.view(MAX_BS * capacity, -1) for layer in model.predictor_k_cache
    ]
    model.predictor_v_rows = [
        layer.view(MAX_BS * capacity, -1) for layer in model.predictor_v_cache
    ]
    torch.manual_seed(20261122)
    embeds = torch.randn(1, 2, HIDDEN, device=dev).to(DTYPE)
    original = embeds.clone()
    with torch.inference_mode():
        # Prepare 14 committed prefix slots, like a genuine K2 proposal at
        # depth 14. Every original prefix row is real model-generated K/V.
        for cache_len in range(0, 14, 2):
            model.predictor_forward_tokens(
                token_embeds=embeds.clone(), batch_size=1, cache_len=cache_len
            )
        old_k = model.predictor_k_cache[:, :1, :16].clone()
        old_v = model.predictor_v_cache[:, :1, :16].clone()
        result = capture_four_real_offsets(
            model, token_embeds=embeds, repeats=8
        )
    assert result["status"].startswith(
        "P9A_FOUR_REAL_PREDICTOR_OFFSET_GRAPH_PARITY_PASS"
    )
    assert len(result["cases"]) == 6
    assert all(row["pass"] for row in result["cases"]), result["cases"]
    assert all(
        check["max_abs"] == 0
        for row in result["cases"] for check in row["checks"].values()
    )
    assert {
        row["model_calls"] for row in result["cost_by_dynamic_stage_sequence"]
    } == {2, 3, 4}
    torch.testing.assert_close(
        model.predictor_k_cache[:, :1, :16], old_k, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache[:, :1, :16], old_v, atol=0, rtol=0
    )
    torch.testing.assert_close(embeds, original, atol=0, rtol=0)

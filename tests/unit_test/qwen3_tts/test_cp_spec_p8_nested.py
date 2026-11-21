# SPDX-License-Identifier: Apache-2.0
"""P8 bounded real Predictor CUDA WHILE+IF integration gate."""
from __future__ import annotations

import pytest
import torch

import sglang_omni.models.qwen3_tts.sglang_model as model_module
from sglang_omni.models.qwen3_tts.code_predictor_spec_p8_nested import (
    STATES, capture_nested_real_predictor, expected_offsets,
)


def test_p8_cpu_reference_covers_one_or_two_dynamic_rounds():
    assert len(STATES) == 6
    assert expected_offsets((True, False), 1) == [2]
    assert expected_offsets((False, True), 1) == [6]
    assert expected_offsets((True, False), 2) == [2, 6]
    assert expected_offsets((False, True), 2) == [6, 2]
    assert expected_offsets((True, True), 2) == [2, 2]
    assert expected_offsets((False, False), 2) == [6, 6]
    with pytest.raises(ValueError):
        expected_offsets((True,), 1)
    with pytest.raises(ValueError):
        expected_offsets((True, False), 3)
    with pytest.raises(ValueError):
        expected_offsets((True, 0), 2)


@pytest.mark.accelerator
def test_p8_h200_nested_while_cond_real_model_parity_and_kv(
    monkeypatch: pytest.MonkeyPatch,
):
    if not torch.cuda.is_available():
        pytest.skip("requires NVIDIA H200 CUDA")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        DTYPE, HIDDEN, HEAD_DIM, MAX_BS, NUM_KV_HEADS,
        build_talker, with_predictor_layers,
    )

    monkeypatch.setattr(
        model_module, "apply_qk_norm", lambda q, k, **kwargs: (q, k),
    )
    dev = torch.device("cuda")
    model = with_predictor_layers(build_talker(dev), 2)
    capacity = 18
    pos = torch.arange(capacity, device=dev, dtype=torch.long)
    model.predictor_positions = pos
    model.predictor_position_rows = (
        pos[:, None].expand(capacity, MAX_BS).contiguous()
    )
    model.predictor_cache_slots = (
        torch.arange(MAX_BS, device=dev, dtype=torch.long)[None, :] * capacity
        + pos[:, None]
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
    torch.manual_seed(20261121)
    embeds = torch.randn(1, 2, HIDDEN, device=dev).to(DTYPE)
    embeddings_before = embeds.clone()
    with torch.inference_mode():
        # Initialize distinct K/V for prefix tokens 0..5.
        for cache_len in (0, 2, 4):
            model.predictor_forward_tokens(
                token_embeds=embeds.clone(), batch_size=1, cache_len=cache_len
            )
        old_k = model.predictor_k_cache[:, :1, :8].clone()
        old_v = model.predictor_v_cache[:, :1, :8].clone()
        report = capture_nested_real_predictor(
            model, fixed_embeds=embeds, repeats=8
        )
    assert report["status"].startswith("P8_NESTED_WHILE_IF_REAL_5_LAYER_MODEL_PASS")
    assert report["predictor_layers"] == 2  # fixture only, not actual model
    assert len(report["cases"]) == len(STATES) == 6
    assert all(row["pass"] for row in report["cases"]), report["cases"]
    assert {
        row["label"] for row in report["cases"]
    } == {label for label, _, _ in STATES}
    assert all(
        row["checks"]["k_entire_8slot"]["max_abs"] == 0
        and row["checks"]["v_entire_8slot"]["max_abs"] == 0
        and row["checks"]["persistent_pair_unchanged"]["max_abs"] == 0
        for row in report["cases"]
    )
    assert all(
        value > 0 for value in report["cuda_graph_event_ms_per_case"].values()
    )
    torch.testing.assert_close(
        model.predictor_k_cache[:, :1, :8], old_k, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache[:, :1, :8], old_v, atol=0, rtol=0
    )
    torch.testing.assert_close(embeds, embeddings_before, atol=0, rtol=0)

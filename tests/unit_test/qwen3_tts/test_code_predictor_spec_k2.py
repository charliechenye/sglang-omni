# SPDX-License-Identifier: Apache-2.0
"""Research-only Qwen3-TTS CodePredictor K2 correctness and regression gates."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import sglang_omni.models.qwen3_tts.sglang_model as model_module
from sglang_omni.models.qwen3_tts.code_predictor_spec import (
    accept_or_residual,
    embed_residual_code,
    offset_causal_mask,
    overlap_probability,
    sampling_probs,
)


def test_offset_causal_mask_keeps_prefix_and_hides_future() -> None:
    allowed = offset_causal_mask(2, 2, torch.device("cpu"))
    assert allowed.tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]
    torch.manual_seed(13)
    q = torch.randn(1, 1, 2, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    packed = F.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed, is_causal=False
    )
    first = F.scaled_dot_product_attention(q[:, :, :1], k[:, :, :3], v[:, :, :3])
    second = F.scaled_dot_product_attention(q[:, :, 1:], k, v)
    torch.testing.assert_close(packed, torch.cat((first, second), dim=2), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        offset_causal_mask(-1, 2, torch.device("cpu"))


def test_target_sampling_distribution_and_one_proposal_overlap() -> None:
    logits_p = torch.log(torch.tensor([[.8, .2, .001]], dtype=torch.float))
    logits_q = torch.log(torch.tensor([[.3, .7, .001]], dtype=torch.float))
    p = sampling_probs(logits_p, temperature=1., top_k=2, top_p=1.)
    q = sampling_probs(logits_q, temperature=1., top_k=2, top_p=1.)
    assert p[0, 2] == q[0, 2] == 0
    assert overlap_probability(p, q).item() == pytest.approx(.5, abs=1e-5)
    assert p.argmax(-1).item() != q.argmax(-1).item()
    clipped = sampling_probs(
        torch.log(torch.tensor([[.5, .3, .2]])),
        temperature=1., top_k=3, top_p=.6
    )
    assert clipped.tolist()[0] == pytest.approx([.625, .375, 0.], abs=1e-5)
    with pytest.raises(ValueError):
        sampling_probs(torch.randn(1, 4), temperature=0, top_k=2, top_p=1.)


def test_reject_residual_returns_target_marginal_not_q_marginal() -> None:
    p = torch.tensor([[.8, .2]], dtype=torch.float)
    q = torch.tensor([[.3, .7]], dtype=torch.float)
    g = torch.Generator(device="cpu").manual_seed(713)
    counts = [0, 0]
    accepted = 0
    for _ in range(2500):
        proposal = torch.multinomial(q, 1, generator=g).view(-1)
        committed, yes = accept_or_residual(proposal, p, q, g)
        counts[int(committed.item())] += 1
        accepted += int(yes)
    assert counts[0] / 2500 == pytest.approx(.8, abs=.045)
    assert accepted / 2500 == pytest.approx(.5, abs=.045)
    same = torch.tensor([[.3, .7]], dtype=torch.float)
    for _ in range(10):
        selected, accepted = accept_or_residual(torch.tensor([1]), same, same, g)
        assert accepted and selected.item() == 1


def test_full_16_group_last_codec_depth_uses_fallback_projection() -> None:
    """Regression for HTTP 500: depth 15 cannot index 14 projected rows."""
    torch.manual_seed(20261121)
    total_residual_depths = 15  # num_code_groups=16
    vocab, codec_hidden, predictor_hidden = 8, 4, 6
    tables = torch.randn(total_residual_depths, vocab, codec_hidden)
    projection = torch.nn.Linear(codec_hidden, predictor_hidden, bias=True)
    with torch.no_grad():
        preprojected = projection(tables[:-1]).detach().clone()
    assert tables.shape[0] == 15
    assert preprojected.shape[0] == 14

    code = torch.tensor([3], dtype=torch.long)
    raw14, projected14 = embed_residual_code(
        depth=14, code=code, codec_tables=tables,
        projected_tables=preprojected, project_input=projection,
        accumulation_dtype=torch.float32, predictor_dtype=torch.float32,
    )
    torch.testing.assert_close(raw14, tables[13, 3].unsqueeze(0))
    torch.testing.assert_close(
        projected14, preprojected[13, 3].reshape(1, 1, predictor_hidden)
    )

    # First previously crashing path: proposal for final residual code.
    # Second previously crashing path: the same last code was committed.
    for _ in range(2):
        raw15, projected15 = embed_residual_code(
            depth=15, code=code, codec_tables=tables,
            projected_tables=preprojected, project_input=projection,
            accumulation_dtype=torch.float32, predictor_dtype=torch.float32,
        )
        torch.testing.assert_close(raw15, tables[14, 3].unsqueeze(0))
        torch.testing.assert_close(
            projected15, projection(raw15.unsqueeze(1))
        )
        assert projected15.shape == (1, 1, predictor_hidden)

    with pytest.raises(ValueError, match="invalid residual depth"):
        embed_residual_code(
            depth=16, code=code, codec_tables=tables,
            projected_tables=preprojected, project_input=projection,
            accumulation_dtype=torch.float32, predictor_dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="exactly one fewer depth"):
        embed_residual_code(
            depth=15, code=code, codec_tables=tables,
            projected_tables=torch.zeros(15, vocab, predictor_hidden),
            project_input=projection,
            accumulation_dtype=torch.float32, predictor_dtype=torch.float32,
        )


def test_embedding_last_depth_without_preprojected_tables() -> None:
    tables = torch.arange(15 * 8 * 4).float().view(15, 8, 4) / 100
    projection = torch.nn.Identity()
    raw, projected = embed_residual_code(
        depth=15, code=torch.tensor([7]), codec_tables=tables,
        projected_tables=None, project_input=projection,
        accumulation_dtype=torch.float32, predictor_dtype=torch.float32,
    )
    torch.testing.assert_close(raw, tables[14, 7][None])
    torch.testing.assert_close(projected, raw[:, None])


@pytest.fixture
def patch_qk_norm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        model_module,
        "apply_qk_norm",
        lambda q, k, **_: (q, k),
    )


@pytest.mark.accelerator
def test_packed_two_token_verify_matches_serial_with_cached_prefix(patch_qk_norm):
    if not torch.cuda.is_available():
        pytest.skip("requires H200/CUDA")
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        BF16_GEMM_ROUNDING,
        DTYPE,
        HIDDEN,
        build_talker,
        with_predictor_layers,
    )

    dev = torch.device("cuda")
    packed_model = with_predictor_layers(build_talker(dev), 2)
    serial_model = with_predictor_layers(build_talker(dev), 2)
    torch.manual_seed(394)
    tokens = torch.randn(1, 4, HIDDEN, dtype=torch.float32).to(dev, DTYPE)
    with torch.inference_mode():
        _ = packed_model.predictor_forward_tokens(
            token_embeds=tokens[:, :2].clone(), batch_size=1, cache_len=0
        )
        _ = serial_model.predictor_forward_tokens(
            token_embeds=tokens[:, :2].clone(), batch_size=1, cache_len=0
        )
        batched_verify = packed_model.predictor_forward_tokens(
            token_embeds=tokens[:, 2:4].clone(), batch_size=1, cache_len=2
        ).clone()
        serial_a = serial_model.predictor_forward_tokens(
            token_embeds=tokens[:, 2:3].clone(), batch_size=1, cache_len=2
        ).clone()
        serial_b = serial_model.predictor_forward_tokens(
            token_embeds=tokens[:, 3:4].clone(), batch_size=1, cache_len=3
        ).clone()
    torch.testing.assert_close(
        batched_verify, torch.cat([serial_a, serial_b], dim=1),
        **BF16_GEMM_ROUNDING
    )
    torch.testing.assert_close(
        packed_model.predictor_k_cache[:, :1, :4],
        serial_model.predictor_k_cache[:, :1, :4],
        **BF16_GEMM_ROUNDING
    )
    torch.testing.assert_close(
        packed_model.predictor_v_cache[:, :1, :4],
        serial_model.predictor_v_cache[:, :1, :4],
        **BF16_GEMM_ROUNDING
    )

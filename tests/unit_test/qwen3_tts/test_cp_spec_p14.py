# SPDX-License-Identifier: Apache-2.0
"""P14 causal one-round GPU graph: CPU invariants + H200 strict parity."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.code_predictor_spec_p9b import cpu_reference
from sglang_omni.models.qwen3_tts.code_predictor_spec import sampling_probs
from sglang_omni.models.qwen3_tts.code_predictor_spec_p14_causal_round import (
    DEPTHS, K, TRIPLES, SCHEMA, capture_real_causal_round,
    gpu_accept, make_causal_round, single_support,
)


def test_p14_cpu_contract_no_static_decoupled_q_and_p():
    assert DEPTHS == (2, 10)
    assert K == 50 and len(TRIPLES) == 4
    assert "causal" in SCHEMA
    source = inspect.getsource(make_causal_round)
    order = [
        "draft_logits, _ = head(draft_hidden)",
        "q_idx, _, q_probs, proposal",
        "draft_embedding = projected_embed(proposal, depth)",
        "verified = model.predictor_forward_tokens(",
        "logits_target, _ = head(before_proposal)",
        "p_idx, p_probs = single_support(",
        "decision = gpu_accept(",
        "bonus_logits, _ = bonus_head(after_proposal)",
        "next_cache_len = cache_len + 1 + accepted.to(dtype=torch.long)",
    ]
    indexes = [source.index(item) for item in order]
    assert indexes == sorted(indexes)
    assert ".item(" not in source
    assert "torch.multinomial(" not in source
    assert "torch.cuda.synchronize(" not in source
    assert "sum_before" not in source


def test_p14_cpu_selected_proposal_and_acceptance_accounting():
    # q has support0..49, p has support25..74; both uniform on top50.
    # Proposal 4 must reject (p(4)=0); proposal 29 must accept (p/q=1).
    q = torch.zeros((1, 2048), dtype=torch.float32)
    p = torch.zeros_like(q)
    q[0, :50] = 1 / 50.
    p[0, 25:75] = 1 / 50.
    rejected = cpu_reference(p, q, (0.10, 0.50, 0.625))
    accepted = cpu_reference(p, q, (0.60, 0.00, 0.625))
    assert rejected["accepted"] is False
    assert accepted["accepted"] is True
    assert rejected["draft"] < 25
    assert 25 <= accepted["draft"] < 50
    assert rejected["committed"] >= 25
    assert accepted["committed"] == accepted["draft"]


@pytest.mark.accelerator
def test_p14_h200_sparse_accept_and_correct_both_decisions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA H200 required")
    dev = torch.device("cuda")
    pids = torch.arange(25, 75, dtype=torch.int32, device=dev).view(1, 50)
    qids = torch.arange(0, 50, dtype=torch.int32, device=dev).view(1, 50)
    pp = torch.full((1, 50), .02, device=dev)
    qp = pp.clone()
    p = torch.zeros((1, 2048), device=dev).scatter(
        -1, pids.long(), pp
    )
    q = torch.zeros_like(p).scatter(-1, qids.long(), qp)
    for draft, triple, surviving in (
        (4, (0.10, 0.50, 0.625, 0.125), False),
        (29, (0.60, 0.00, 0.625, 0.125), True),
    ):
        u = torch.tensor(triple, device=dev, dtype=torch.float32)
        actual = gpu_accept(
            pids, pp, qids, qp, u,
            torch.tensor([draft], device=dev, dtype=torch.long),
        )
        torch.cuda.synchronize(dev)
        expected = cpu_reference(p, q, triple[:3])
        assert expected["draft"] == draft
        assert bool(actual["accepted"][0].item()) is surviving
        assert bool(actual["accepted"][0].item()) == expected["accepted"]
        assert int(actual["committed"][0].item()) == expected["committed"]
        assert float(actual["residual_mass"][0].item()) > 0


@pytest.mark.accelerator
def test_p14_h200_single_top50_support_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA H200 required")
    gen = torch.Generator().manual_seed(20261122)
    logits = torch.randn((1, 2048), generator=gen).cuda().float()
    for top_p in (1.0, 0.9):
        ids, probs = single_support(
            logits, temperature=.9, top_p=top_p, fast=True
        )
        ref = sampling_probs(
            logits, temperature=.9, top_k=50, top_p=top_p
        )
        full = torch.zeros_like(ref).scatter(-1, ids.long(), probs)
        torch.testing.assert_close(full, ref, atol=1e-6, rtol=1e-6)


class _FakePredictor:
    """5-layer-shaped GPU cache used to exercise capture dependencies only.

    NOT the actual pretrained five-layer model and NOT an on-policy
    serving parity test. Real model evidence must come from HTTP P14.
    """

    def __init__(self, device):
        h = 32
        vocab = 2048
        self.predictor_cutedsl_layers = None
        self.predictor_fused_layers = SimpleNamespace(covers=lambda n: n == 2)
        self.predictor_k_cache = torch.zeros(
            (5, 1, 16, 1, h), dtype=torch.float32, device=device
        )
        self.predictor_v_cache = self.predictor_k_cache.clone()
        heads = torch.nn.ModuleList([
            torch.nn.Linear(h, vocab, bias=False, device=device)
            for _ in range(15)
        ])
        with torch.no_grad():
            for head in heads:
                head.weight.mul_(0.02)
        codec = torch.nn.Embedding(
            15 * vocab, h, device=device,
        )
        weights = codec.weight.view(15, vocab, h)
        projected = torch.randn((14, vocab, h), device=device) * .1
        self.predictor_projected_embeddings = projected
        self.code_predictor = SimpleNamespace(
            lm_head=heads,
            model=SimpleNamespace(
                layers=list(range(5)),
                codec_embedding=SimpleNamespace(weight=weights),
            ),
        )
        self.inputs_seen = []

    def predictor_forward_tokens(self, *,
                                 token_embeds, batch_size, cache_len):
        assert batch_size == 1 and token_embeds.shape == (1, 2, 32)
        # True GPU dependency on proposal embedding in SECOND ROW.
        h0 = token_embeds[:, :1, :].clone() + .1
        h1 = token_embeds[:, 1:2, :].clone() + h0 * .1
        result = torch.cat((h0, h1), dim=1)
        cache_shape = (5, 1, 2, 1, 32)
        cache_data = result.reshape((1, 1, 2, 1, 32)).expand(cache_shape)
        self.predictor_k_cache[:, :, cache_len:cache_len+2].copy_(cache_data)
        self.predictor_v_cache[:, :, cache_len:cache_len+2].copy_(cache_data + .1)
        return result


@pytest.mark.accelerator
def test_p14_h200_fake_model_one_graph_full_causal_and_kv_restored():
    if not torch.cuda.is_available():
        pytest.skip("CUDA H200 required")
    dev = torch.device("cuda")
    torch.manual_seed(20261122)
    model = _FakePredictor(dev)
    depth = 1
    k0 = model.predictor_k_cache.clone()
    v0 = model.predictor_v_cache.clone()
    hidden = torch.randn((1, 1, 32), device=dev) * .01
    last = torch.randn_like(hidden) * .01
    summed = torch.randn((1, 32), device=dev) * .01
    report = capture_real_causal_round(
        model, depth=depth, cache_len=2,
        draft_hidden=hidden, last_embedding=last,
        accumulated_before=summed,
        temperature=.9, top_k=50, top_p=1., repeats=8,
    )
    assert report["status"].startswith(
        "P14_REAL_CAUSAL_ONE_ROUND_Q_VERIFIER_P_RESIDUAL_BONUS_"
    )
    assert report["private_rng_replay_unique"] is True
    assert len(report["fixed_uniform_cases"]) == len(TRIPLES)
    assert set(report["median_cuda_graph_ms"]) == {
        "torch_reference", "triton_candidate", "triton_private_rng"
    }
    assert all(x > 0 for x in report["median_cuda_graph_ms"].values())
    assert all(c["q_p_distribution_parity"] for c in report["fixed_uniform_cases"])
    torch.testing.assert_close(
        model.predictor_k_cache, k0, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.predictor_v_cache, v0, atol=0, rtol=0
    )

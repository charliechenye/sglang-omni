# SPDX-License-Identifier: Apache-2.0
"""P10B sparse top-50 CUDA proposal, acceptance, corrected residual and cost."""
from __future__ import annotations

import inspect

import pytest
import torch

from sglang_omni.models.qwen3_tts.code_predictor_spec import sampling_probs
from sglang_omni.models.qwen3_tts.code_predictor_spec_p10b_sparse_sampler import (
    BLOCK, K, TEST_UNIFORMS,
    capture_real_sparse_sampler, sparse_gpu_round,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p9b import cpu_reference


def test_p10b_cpu_sparse_support_rejection_only_on_p_topk():
    # The (p-q)+ support is a strict subset of p's original nonzero
    # categorical support for any normalized p and q.
    p = torch.tensor([.6, .25, .15, 0., 0.])
    q = torch.tensor([0., .1, .2, .45, .25])
    residual = (p - q).clamp_min(0)
    assert torch.equal(residual[~(p > 0)], torch.zeros_like(residual[~(p > 0)]))
    assert residual.sum() > 0
    assert len(TEST_UNIFORMS) == 4
    assert (K, BLOCK) == (50, 64)
    assert cpu_reference(p.unsqueeze(0), q.unsqueeze(0), (.75, .99, .85))["accepted"] is False


def test_p10b_sparse_graph_contains_no_cpu_sync():
    source = inspect.getsource(sparse_gpu_round)
    assert ".item(" not in source
    assert "torch.topk(" in source
    assert "sparse_accept_residual_kernel" in source
    assert "torch.searchsorted(" not in source
    assert "torch.cumsum(" not in source


def test_p10b_qx_uses_vocab_id_after_sparse_cdf_refactor():
    # P10B originally changed its draft from top-k rank to vocabulary
    # ID, but left a stale q lookup using proposed_rank; Triton raised
    # NameError while compiling. A CPU-only source guard catches this
    # before the next expensive H200 capture test.
    from sglang_omni.models.qwen3_tts.code_predictor_spec_p10b_sparse_sampler import (
        sparse_accept_residual_kernel,
    )

    kernel_source = inspect.getsource(sparse_accept_residual_kernel.fn)
    assert "proposed_rank" not in kernel_source
    assert "qi == draft" in kernel_source
    assert "qx = tl.sum(" in kernel_source


@pytest.mark.accelerator
@pytest.mark.parametrize("top_p", (1.0, 0.90))
def test_p10b_h200_real_vocab_probabilities_and_deterministic_draws(top_p):
    if not torch.cuda.is_available():
        pytest.skip("H200/CUDA required")
    gen = torch.Generator(device="cpu").manual_seed(20261122)
    p = torch.randn(1, 2048, generator=gen).to("cuda", torch.bfloat16)
    q = torch.randn(1, 2048, generator=gen).to("cuda", torch.bfloat16)
    p_ref = sampling_probs(p, temperature=.9, top_k=50, top_p=top_p)
    q_ref = sampling_probs(q, temperature=.9, top_k=50, top_p=top_p)
    u = torch.tensor(TEST_UNIFORMS[0], device="cuda").view(1, 3)
    old_p = p.clone()
    old_q = q.clone()
    with torch.inference_mode():
        sparse_gpu_round(
            p, q, u, temperature=.9, top_k=50, top_p=top_p
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        candidate = sparse_gpu_round(
            p, q, u, temperature=.9, top_k=50, top_p=top_p
        )
    for sample in TEST_UNIFORMS:
        u.copy_(torch.tensor(sample, device="cuda").view(1, 3))
        graph.replay()
        torch.cuda.synchronize()
        code, survived, draft, pidx, qidx, ps, qs, mass = candidate
        full_p = torch.zeros_like(p_ref).scatter(-1, pidx, ps[:K].view(1, K))
        full_q = torch.zeros_like(q_ref).scatter(-1, qidx, qs[:K].view(1, K))
        torch.testing.assert_close(full_p, p_ref, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(full_q, q_ref, rtol=1e-6, atol=1e-6)
        expected = cpu_reference(p_ref, q_ref, sample)
        actual = {
            "draft": int(draft.item()),
            "accepted": bool(survived.item()),
            "committed": int(code.item()),
        }
        assert actual == expected
        assert torch.isfinite(mass).all()
    torch.testing.assert_close(p, old_p, rtol=0, atol=0)
    torch.testing.assert_close(q, old_q, rtol=0, atol=0)


@pytest.mark.accelerator
def test_p10b_h200_captured_private_rng_and_same_worker_cost():
    if not torch.cuda.is_available():
        pytest.skip("H200/CUDA required")
    gen = torch.Generator(device="cpu").manual_seed(20261123)
    p = torch.randn(1, 2048, generator=gen).to("cuda", torch.bfloat16)
    q = torch.randn(1, 2048, generator=gen).to("cuda", torch.bfloat16)
    report = capture_real_sparse_sampler(
        logits_p=p, logits_q=q,
        temperature=.9, top_k=50, top_p=1.0, repeats=16,
    )
    assert report["status"].startswith(
        "P10B_REAL_PQ_SPARSE_TOPK_SPEC_SAMPLER_PARITY_PASS"
    )
    assert report["private_rng_replay_unique"] is True
    assert all(x["pass"] for x in report["checks"])
    assert report["input_logits_immutable"] is True
    assert report["p_max_abs"] <= 1e-6
    assert report["q_max_abs"] <= 1e-6
    assert report["sparse_fused_cuda_graph_ms"] > 0
    assert report["p9b_dense_reference_cuda_graph_ms"] > 0

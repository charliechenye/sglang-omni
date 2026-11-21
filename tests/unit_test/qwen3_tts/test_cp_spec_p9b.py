# SPDX-License-Identifier: Apache-2.0
"""P9B real-logits GPU proposal, accept/reject and CUDA Graph RNG gates."""
from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_tts.code_predictor_spec import sampling_probs
from sglang_omni.models.qwen3_tts.code_predictor_spec_p9b import (
    TEST_UNIFORMS, cpu_reference, gpu_probs, gpu_round, profile_real_logits,
)


def test_p9b_cpu_sampler_distribution_parity_and_rejection_math():
    p_logits = torch.log(torch.tensor([[.8, .2, .001, .001]]))
    q_logits = torch.log(torch.tensor([[.3, .7, .001, .001]]))
    args = dict(temperature=1., top_k=2, top_p=1.)
    p = sampling_probs(p_logits, **args)
    q = sampling_probs(q_logits, **args)
    assert torch.allclose(gpu_probs(p_logits, **args), p)
    assert torch.allclose(gpu_probs(q_logits, **args), q)

    # Draft category 0 has high p/q; category 1 can reject.
    # u_q=.6 falls in the q category-1 CDF (q0=.3), u_accept=.99
    # rejects (p1/q1=.2857), residual produces category0.
    uniforms = torch.tensor([[.6, .99, .5]], dtype=torch.float32)
    token, accepted, proposal, calc_p, calc_q = gpu_round(
        p_logits, q_logits, uniforms, **args
    )
    expected = cpu_reference(p, q, (.6, .99, .5))
    assert not expected["accepted"]
    assert proposal.item() == expected["draft"]
    assert token.item() == expected["committed"]
    assert bool(accepted.item()) == expected["accepted"]
    torch.testing.assert_close(calc_p, p)
    torch.testing.assert_close(calc_q, q)

    clipped_args = dict(temperature=1., top_k=3, top_p=.6)
    clipped = torch.log(torch.tensor([[.5, .3, .2]]))
    torch.testing.assert_close(
        gpu_probs(clipped, **clipped_args),
        sampling_probs(clipped, **clipped_args),
    )
    with pytest.raises(ValueError, match="P9B requires"):
        gpu_probs(p_logits, temperature=0., top_k=2, top_p=1.)


def test_p9b_gpu_round_has_no_host_item_in_hot_path(monkeypatch):
    # Static source-level guard: this CPU fixture validates the CUDA-safe
    # operator structure without relying on an H200 context.
    import inspect
    from sglang_omni.models.qwen3_tts import code_predictor_spec_p9b as module
    source = inspect.getsource(module.gpu_round)
    assert ".item(" not in source
    assert "torch.searchsorted" in source
    assert "torch.where(" in source


@pytest.mark.accelerator
def test_p9b_h200_captured_realistic_logits_and_private_graph_rng():
    if not torch.cuda.is_available():
        pytest.skip("H200/CUDA not available")
    generator = torch.Generator(device="cpu").manual_seed(20261122)
    p = torch.randn(1, 2048, generator=generator).to("cuda", torch.bfloat16)
    q = torch.randn(1, 2048, generator=generator).to("cuda", torch.bfloat16)
    report = profile_real_logits(
        logits_p=p,
        logits_q=q,
        temperature=.9,
        top_k=50,
        top_p=1.0,
        repeats=16,
    )
    assert report["status"].startswith(
        "P9B_REAL_LOGITS_GPU_PROPOSAL_ACCEPT_RESIDUAL_AND_RNG_PASS"
    )
    assert all(x["pass"] for x in report["checks"])
    assert len(report["checks"]) == len(TEST_UNIFORMS)
    assert report["independent_graph_rng_replay_unique"] is True
    assert report["output_input_parity"] is True
    assert report["distribution_p_max_abs"] <= 1e-6
    assert report["distribution_q_max_abs"] <= 1e-6
    assert report["gpu_sampler_with_independent_rng_graph_ms"] > 0

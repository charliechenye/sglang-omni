# SPDX-License-Identifier: Apache-2.0
"""P11A real p/q CUDA attribution and P11B actual target-AR state contracts."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.code_predictor_spec_p11a_attribution import (
    CELLS, capture_real_sampler_attribution, topk_pair,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p11b_hybrid import (
    AUDIT, HYBRID_ARM, HYBRID_THROUGH,
    active_ar_through, direct_target_ar_step, stats_line, validate_counts,
)
from experiments.qwen3_tts_cp_spec_k2.quality_smoke import parse_k2_log_entries

SOURCE = "a" * 40


def test_p11a_topk_pair_contract_cpu():
    generator = torch.Generator().manual_seed(20261120)
    p = torch.randn(1, 2048, generator=generator)
    q = torch.randn(1, 2048, generator=generator)
    out = topk_pair(p, q, .9)
    assert len(out) == 4
    assert all(x.shape == (1, 50) for x in out)
    for vals, ix, logits in ((out[0], out[1], p), (out[2], out[3], q)):
        expected_vals, expected_ix = torch.topk(
            logits.float() / .9, 50, dim=-1, sorted=True
        )
        torch.testing.assert_close(vals, expected_vals, rtol=0, atol=0)
        torch.testing.assert_close(ix, expected_ix, rtol=0, atol=0)
    assert "torch.searchsorted" not in inspect.getsource(topk_pair)
    assert set(CELLS) == {
        "topk_pair", "post_topk_triton", "private_rng", "full_sparse"
    }


def test_p11b_hybrid_marker_is_request_scoped(tmp_path, monkeypatch):
    marker = tmp_path / "HYBRID_ARM"
    monkeypatch.delenv(HYBRID_ARM, raising=False)
    monkeypatch.delenv(HYBRID_THROUGH, raising=False)
    assert active_ar_through() == 1
    monkeypatch.setenv(HYBRID_ARM, str(marker))
    monkeypatch.setenv(HYBRID_THROUGH, "3")
    assert active_ar_through() == 1
    marker.touch(exist_ok=False)
    assert active_ar_through() == 3
    marker.unlink()
    assert active_ar_through() == 1
    monkeypatch.setenv(HYBRID_THROUGH, "5")
    with pytest.raises(ValueError, match="2 or 3"):
        active_ar_through()


def test_p11b_strict_14_residual_code_conservation():
    validate_counts(
        ar_steps=0, rounds=10, proposals=10,
        accepted=4, rejected=6, bonus=4,
    )
    validate_counts(
        ar_steps=2, rounds=8, proposals=8,
        accepted=4, rejected=4, bonus=4,
    )
    for wrong in (
        dict(ar_steps=2, rounds=8, proposals=8, accepted=5, rejected=3, bonus=3),
        dict(ar_steps=0, rounds=10, proposals=10, accepted=2, rejected=8, bonus=4),
        dict(ar_steps=1, rounds=9, proposals=8, accepted=4, rejected=4, bonus=5),
    ):
        with pytest.raises(RuntimeError, match="accounting"):
            validate_counts(**wrong)


def test_p11b_direct_ar_updates_cache_hidden_and_commit_state():
    class FakeHead:
        def __call__(self, hidden):
            assert hidden.shape == (1, 1, 8)
            return torch.tensor([[[0.0, 8.0, -3.0, -3.0]]]), None

    class FakeModel:
        def __init__(self):
            self.code_predictor = SimpleNamespace(
                lm_head=[FakeHead(), FakeHead(), FakeHead()],
            )
            self.cache_lengths = []

        def predictor_forward_tokens(self, *, token_embeds, batch_size, cache_len):
            assert batch_size == 1
            assert token_embeds.shape == (1, 1, 8)
            self.cache_lengths.append(cache_len)
            return token_embeds + cache_len

    model = FakeModel()
    committed = []

    def commit(depth, code):
        committed.append((depth, int(code.item())))
        return torch.full((1, 1, 8), float(depth))

    gen = torch.Generator().manual_seed(20261120)
    depth, cache_len = 1, 2
    last_emb = torch.ones((1, 1, 8))
    for expected_depth in (2, 3):
        depth, cache_len, h0, last_emb = direct_target_ar_step(
            model=model, depth=depth, cache_len=cache_len,
            last_embedding=last_emb,
            commit=commit, generator=gen,
            temperature=.9, top_k=4, top_p=1.0,
        )
        assert depth == expected_depth
        assert cache_len == depth + 1
        assert h0.shape == last_emb.shape
        torch.testing.assert_close(last_emb, torch.full_like(last_emb, float(depth)))
    assert model.cache_lengths == [2, 3]
    assert [x[0] for x in committed] == [2, 3]


def test_p11b_legacy_and_hybrid_k2_logs_both_parse():
    old = (
        "QTTS_CP_SPEC_K2 frame=1 rounds=10 proposed=10 "
        "accepted=4 bonus=4 rejected=6 committed_residual=15"
    )
    hybrid = (
        "QTTS_CP_SPEC_K2 frame=2 rounds=8 proposed=8 "
        "accepted=4 bonus=4 rejected=4 committed_residual=15 "
        "ar_steps=2 ar_through=3"
    )
    rows = parse_k2_log_entries(old + "\n" + hybrid + "\n")
    assert len(rows) == 2
    assert rows[0]["ar_steps"] == 0
    assert rows[0]["ar_through"] == 1
    assert rows[1]["ar_steps"] == 2
    assert rows[1]["ar_through"] == 3
    invalid = hybrid.replace("ar_steps=2", "ar_steps=1")
    with pytest.raises(ValueError, match="counters inconsistent"):
        parse_k2_log_entries(invalid)
    assert (
        "ar=2,3 proposed=4,6 accepted=4 bonus=5"
        in stats_line(
            frame=2, ar_through=3, ar=[2, 3],
            proposed=[4, 6], accepted=[4], bonus=[5],
            source_sha=SOURCE,
        )
    )


@pytest.mark.accelerator
def test_p11a_h200_graph_stage_attribution_and_p10b_parity():
    if not torch.cuda.is_available():
        pytest.skip("H200/CUDA required")
    gen = torch.Generator(device="cpu").manual_seed(20261122)
    p = torch.randn(1, 2048, generator=gen).to("cuda", torch.bfloat16)
    q = torch.randn(1, 2048, generator=gen).to("cuda", torch.bfloat16)
    report = capture_real_sampler_attribution(
        logits_p=p, logits_q=q,
        temperature=.9, top_k=50, top_p=1.,
        repeats=16,
    )
    assert report["status"].startswith("P11A_REAL_PQ_TOPK_POST_RNG_ATTRIBUTION_PASS")
    assert report["torch_reference_distribution_parity"] is True
    assert report["private_rng_independent_unique_replays"] is True
    assert report["original_pq_unchanged"] is True
    assert set(report["measurements"]) == set(CELLS)
    assert all(
        len(item["block_cuda_event_ms"]) == 4
        and item["median_cuda_event_ms"] > 0
        for item in report["measurements"].values()
    )
    assert report["p10b_same_worker_sparse_graph_ms"] > 0
    assert report["p10b_same_worker_dense_graph_ms"] > 0

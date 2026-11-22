# SPDX-License-Identifier: Apache-2.0
"""P17 CPU source, acceptance and source-locked wide-cost gates."""
from __future__ import annotations

import copy
import inspect
import statistics
from types import SimpleNamespace

import pytest
import torch

from experiments.qwen3_tts_cp_spec_k2.p17_multiwidth_gate import (
    SCHEMA, PROBE_PASS, validate_probe,
)
from sglang_omni.models.qwen3_tts import code_predictor_spec_p17_wide as p17_wide
from sglang_omni.models.qwen3_tts.code_predictor_spec_p17_wide import (
    WIDTHS, UNIFORM_COUNT, prefix_accept_count, check_wide_backend,
    make_q_proposals, wide_device_fused_predictor,
    make_wide_round,
)

SHA = "a" * 40
PID = 12345
GPU = "NVIDIA H200"


def synthetic_probe():
    widths = {}
    for w in WIDTHS:
        cases = []
        for i in range(16):
            prefix = i % w
            cases.append({
                "index": i, "accepted_prefix": prefix,
                "committed_count": prefix+1,
                "first_rejection": prefix if prefix < w-1 else None,
                "bonus_committed": prefix == w-1,
            })
        hist = {str(x): sum(row["committed_count"] == x for row in cases)
                for x in range(1, w+1)}
        accept = {str(x): sum(row["accepted_prefix"] == x for row in cases)
                  for x in range(w)}
        timing = {}
        medians = {}
        for name in ("full_round", "proposal_only", "verifier_only"):
            indices = (0,5,10,15) if name != "verifier_only" else (0,)
            timing[name] = [
                {
                    "case_index": idx, "repeat_index": rep,
                    "cuda_event_ms": 0.100 + 0.025*w + rep*0.002 + idx*0.00001,
                }
                for idx in indices for rep in range(4)
            ]
            medians[name] = statistics.median(
                row["cuda_event_ms"] for row in timing[name]
            )
        mean = statistics.mean(x["committed_count"] for x in cases)
        widths[str(w)] = {
            "on_policy_real_checkpoint": True,
            "batched_multihead_and_static_fused_parity": True,
            "full_physical_kv_parity": True,
            "independent_cpu_accept_residual_parity": True,
            "case_count": 16,
            "accepted_prefix_histogram": accept,
            "committed_count_histogram": hist,
            "mean_committed_per_round_fixed_checkpoint": mean,
            "timed_full_and_proposal_case_indices": [0, 5, 10, 15],
            "verifier_only_frozen_case_index": 0,
            "cuda_graph_event_ms_blocks": timing,
            "cuda_graph_event_ms_medians": medians,
            "full_round_ms_per_committed_code_SCREEN_ONLY":
                medians["full_round"] / mean,
            "cases": cases,
        }
    return {
        "schema": SCHEMA, "status": PROBE_PASS,
        "source_sha": SHA, "pid": PID, "gpu": GPU,
        "last_committed_depth": 9, "cache_len": 10,
        "predictor_layers": 5,
        "backend": "P17_real_fused_bf16_5layer_wide",
        "sampling": {"top_k": 50},
        "uniform_cases_per_width": 16,
        "widths": widths,
    }


def test_p17_cpu_all_width_source_locked_gate():
    assert WIDTHS == (2,3,4) and UNIFORM_COUNT == 13
    validate_probe(synthetic_probe(), sha=SHA, pid=PID, gpu=GPU)


@pytest.mark.parametrize("flags,expected", [
    ((), 0), ((False,), 0), ((True,), 1),
    ((True,False,True), 1), ((True,True,False), 2),
    ((True,True,True), 3), ((False,True,True), 0),
])
def test_p17_first_rejection_stops_commitment(flags, expected):
    assert prefix_accept_count(flags) == expected


def test_p17_cpu_source_contract_true_multihead_single_wide():
    q_source = inspect.getsource(make_q_proposals)
    verify_source = inspect.getsource(wide_device_fused_predictor)
    round_source = inspect.getsource(make_wide_round)
    assert "old_hidden.expand(n, -1, -1)" in q_source
    assert "head_bank[:n]" in q_source
    assert "fast_selector(" in q_source
    assert "wide_device_fused_predictor(" in round_source
    assert round_source.index("make_q_proposals(") < round_source.index(
        "wide_device_fused_predictor("
    ) < round_source.index("batched_head_logits(")
    assert "torch.cumprod" in round_source
    assert "gpu_accept(" in round_source
    assert "torch.where(\n        full_accept" in round_source
    assert "decode_attention_with_device_end" in verify_source
    for source in (q_source, verify_source, round_source):
        for forbidden in (
            ".item(", ".cpu(", ".tolist(", "torch.multinomial(",
            "torch.cuda.synchronize(",
        ):
            assert forbidden not in source


def test_p17_no_hidden_width_downgrade():
    class FakeModel:
        predictor_cutedsl_layers = None
        predictor_fused_layers = SimpleNamespace(
            covers=lambda n: n <= 4,
            shape=SimpleNamespace(num_q_heads=16, num_kv_heads=4),
        )
        code_predictor = SimpleNamespace(
            model=SimpleNamespace(layers=[0,1,2,3,4]),
        )
        predictor_k_cache = torch.zeros((5,1,16,4,128))
        predictor_v_cache = torch.zeros((5,1,16,4,128))
    model = FakeModel()
    for width in WIDTHS:
        check_wide_backend(model, width, cache_len=10)
    model.predictor_fused_layers.shape.num_kv_heads = 2
    with pytest.raises(RuntimeError, match="ATTENTION_TILE_TOO_NARROW"):
        check_wide_backend(model, width=4, cache_len=10)


@pytest.mark.parametrize("mutator", [
    lambda row: row.update(source_sha="b"*40),
    lambda row: row["widths"].pop("4"),
    lambda row: row["widths"]["4"]["cases"][0].update(committed_count=5),
    lambda row: row["widths"]["3"]["committed_count_histogram"].update({"1": 999}),
    lambda row: row["widths"]["2"].update(full_physical_kv_parity=False),
    lambda row: row["widths"]["4"]["cuda_graph_event_ms_blocks"]["full_round"].pop(),
    lambda row: row["widths"]["4"]["cuda_graph_event_ms_medians"].update(full_round=999.),
])
def test_p17_cpu_source_and_acceptance_evidence_fail_closed(mutator):
    report = copy.deepcopy(synthetic_probe())
    mutator(report)
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate_probe(report, sha=SHA, pid=PID, gpu=GPU)


@pytest.mark.parametrize("width,accepts", [
    (2, (False,)),
    (2, (True,)),
    (3, (False, True)),
    (3, (True, False)),
    (3, (True, True)),
    (4, (False, True, True)),
    (4, (True, False, True)),
    (4, (True, True, False)),
    (4, (True, True, True)),
])
def test_p17_wide_round_bonus_mask_is_bool_and_prefix_is_correct(
    monkeypatch, width, accepts,
):
    """Exercise the REAL round's torch.where/bonus path with CPU fake kernels.

    Old P17 used int64 cumprod[-1] directly as torch.where(condition),
    which raised RuntimeError before any on-policy GPU evidence existed.
    Do not replace or skip the actual make_wide_round call.
    """
    hidden_dim = 3
    n = width - 1
    draft_codes = [11 + j for j in range(n)]
    rejected_codes = [41 + j for j in range(n)]
    bonus_code = 99

    def fake_proposals(model, *, depth, width, old_hidden, head_bank,
                       uniforms, temperature, top_p, accumulation_dtype):
        assert old_hidden.shape == (1, 1, hidden_dim)
        return (
            torch.zeros((n, 2048)),
            torch.zeros((n, 50), dtype=torch.int32),
            torch.full((n, 50), 1 / 50.0),
            torch.tensor(draft_codes, dtype=torch.long),
            torch.ones((1, n, hidden_dim)),
        )

    def fake_verifier(model, *, packed, cache_len_gpu, offsets_gpu):
        assert packed.shape == (1, width, hidden_dim)
        assert cache_len_gpu.dtype == torch.long
        assert offsets_gpu.shape == (width,)
        return torch.arange(width, dtype=torch.float32).reshape(
            1, width, 1
        ).expand(1, width, hidden_dim).clone()

    def fake_logits(states, bank):
        assert states.shape == (width, 1, hidden_dim)
        return torch.zeros((width, 2048))

    def fake_support(logits, *, temperature, top_p, fast):
        return (torch.zeros((1, 50), dtype=torch.int32),
                torch.full((1, 50), 1 / 50.0))

    def fake_accept(p_ids, p_probs, q_ids, q_probs, u, q_draft):
        idx = int(q_draft[0]) - draft_codes[0]
        is_accepted = accepts[idx]
        return {
            "accepted": torch.tensor([is_accepted], dtype=torch.bool),
            "committed": torch.tensor(
                [draft_codes[idx] if is_accepted else rejected_codes[idx]],
                dtype=torch.long,
            ),
        }

    def fake_bonus(logits, uniform, *, temperature, top_p):
        return None, None, None, torch.tensor([bonus_code], dtype=torch.long)

    def fake_project(model, index, code, accum_dtype):
        raw = code.to(dtype=accum_dtype).reshape(1, 1).expand(
            1, hidden_dim
        ).clone()
        return raw, raw.unsqueeze(1).clone()

    monkeypatch.setattr(p17_wide, "make_q_proposals", fake_proposals)
    monkeypatch.setattr(p17_wide, "wide_device_fused_predictor", fake_verifier)
    monkeypatch.setattr(p17_wide, "batched_head_logits", fake_logits)
    monkeypatch.setattr(p17_wide, "single_support", fake_support)
    monkeypatch.setattr(p17_wide, "gpu_accept", fake_accept)
    monkeypatch.setattr(p17_wide, "fast_selector", fake_bonus)
    monkeypatch.setattr(p17_wide, "_project", fake_project)

    result = make_wide_round(
        SimpleNamespace(), depth=1, width=width,
        cache_len_gpu=torch.tensor(2, dtype=torch.long),
        old_hidden=torch.zeros((1, 1, hidden_dim)),
        last_embedding=torch.zeros((1, 1, hidden_dim)),
        accumulated=torch.full((1, hidden_dim), 7.0),
        uniforms=torch.zeros(UNIFORM_COUNT),
        head_bank=torch.zeros((width, hidden_dim, 2048)),
        offset_ids=torch.arange(width, dtype=torch.long),
        temperature=0.9, top_p=1.0,
    )
    prefix = prefix_accept_count(accepts)
    expected_codes = [
        (draft_codes[j] if accepts[j] else rejected_codes[j])
        if j <= prefix else -1
        for j in range(n)
    ]
    expected_codes.append(bonus_code if prefix == n else -1)
    assert result["accepted_flags"].dtype == torch.bool
    assert result["accepted_prefix"].item() == prefix
    assert result["committed_count"].item() == prefix + 1
    assert result["committed_codes"].tolist() == expected_codes
    assert result["next_cache_len"].item() == 2 + prefix + 1
    assert result["next_depth"].item() == 2 + prefix
    expected_last = expected_codes[prefix]
    torch.testing.assert_close(
        result["next_embedding"],
        torch.full((1, 1, hidden_dim), float(expected_last)),
    )
    torch.testing.assert_close(
        result["next_hidden"],
        torch.full((1, 1, hidden_dim), float(prefix)),
    )
    torch.testing.assert_close(
        result["next_sum"],
        torch.full(
            (1, hidden_dim),
            7.0 + sum(code for code in expected_codes if code >= 0),
        ),
    )

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from array import array
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
)
from sglang_omni.models.qwen3_asr.temporal_prefill import Qwen3ASRTemporalPrefillAdder


@dataclass(frozen=True, slots=True)
class PrefillProbe:
    prefix_len: int
    extend_len: int
    is_chunked: bool = False


def make_admission_state() -> (
    tuple[Qwen3ASRTemporalPrefillAdder, Req, Qwen3ASRTemporalRequest]
):
    adder = object.__new__(Qwen3ASRTemporalPrefillAdder)
    adder.can_run_list = []
    adder.running_batch = SimpleNamespace(reqs=[])
    adder.waiting_queue_len = 1
    adder.page_size = 1
    adder.is_hybrid_swa = False
    adder.is_hybrid_ssm_cache = False
    adder.dllm_config = None
    adder.token_to_kv_pool_allocator = object()
    adder.per_req_token_overhead = 0

    prompt = array("q", [10, 11, 12, 13])
    request = Req(
        "final-1",
        "",
        prompt,
        SamplingParams(max_new_tokens=3, temperature=0, top_p=1),
        eos_token_ids={2},
        vocab_size=64,
    )
    request.sampling_params.normalize(tokenizer=None)
    request.full_untruncated_fill_ids = prompt[:]
    request.multimodal_inputs = SimpleNamespace(
        mrope_positions=torch.arange(4).expand(3, -1).clone()
    )
    request.kv.cache_protected_len = 0
    request.kv.cache_inserted_len = 0
    request.kv.kv_allocated_len = 0
    request.kv.kv_committed_len = 0
    temporal = Qwen3ASRTemporalRequest(
        candidate_token_ids=(20, 21), prompt_boundary=len(prompt)
    )
    request.omni_data = Qwen3ASRRequestData(
        req=SimpleNamespace(), temporal_request=temporal
    )
    return adder, request, temporal


@pytest.mark.parametrize(
    ("unsafe_state", "chunked"),
    [
        ("prefix", False),
        ("host_hit", False),
        ("mixed_waiting", False),
        ("mixed_running", False),
        ("sampling", False),
        ("budget", False),
        (None, True),
    ],
)
def test_admission_requires_isolated_greedy_cache_miss(
    unsafe_state: str | None, chunked: bool
) -> None:
    adder, request, temporal = make_admission_state()
    if unsafe_state == "prefix":
        request.prefix_len = 1
    elif unsafe_state == "host_hit":
        request.host_hit_length = 1
    elif unsafe_state == "mixed_waiting":
        adder.waiting_queue_len = 2
    elif unsafe_state == "mixed_running":
        adder.running_batch.reqs = [object()]
    elif unsafe_state == "sampling":
        request.sampling_params.repetition_penalty = 1.1
    elif unsafe_state == "budget":
        request.sampling_params.max_new_tokens = 2
    else:
        assert unsafe_state is None

    assert not adder.can_admit_temporal(request, temporal, chunked)
    assert not temporal.admitted
    assert request.full_untruncated_fill_ids == request.origin_input_ids


@pytest.mark.parametrize(
    ("preflight_ok", "native_admits", "expected_temporal"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_add_one_request_preflights_then_calls_native_once(
    monkeypatch: pytest.MonkeyPatch,
    preflight_ok: bool,
    native_admits: bool,
    expected_temporal: bool,
) -> None:
    adder, request, temporal = make_admission_state()
    native_calls: list[tuple[int, ...]] = []

    def preflight(
        candidate: Req,
        *,
        total_tokens: int,
        host_hit_length: int,
        swa_host_hit_length: int,
        truncation_align_size: int | None,
        has_chunked_req: bool,
    ) -> PrefillProbe | AddReqResult:
        assert total_tokens > len(candidate.full_untruncated_fill_ids)
        assert not has_chunked_req
        if preflight_ok:
            return PrefillProbe(
                prefix_len=candidate.prefix_len,
                extend_len=len(candidate.full_untruncated_fill_ids)
                - candidate.prefix_len,
            )
        else:
            return AddReqResult.NO_TOKEN

    monkeypatch.setattr(adder, "_select_prefill_admission", preflight)
    monkeypatch.setattr(adder, "_mamba_gap_budget_for_req", lambda *a, **k: 0)
    monkeypatch.setattr(adder, "_request_slot_reserve_for_req", lambda *a, **k: 0)

    def native_add(
        native_adder: PrefillAdder,
        candidate: Req,
        has_chunked_req: bool,
        truncation_align_size: int | None,
    ) -> AddReqResult:
        assert not has_chunked_req and truncation_align_size is None
        native_calls.append(tuple(candidate.full_untruncated_fill_ids))
        if native_admits:
            native_adder.can_run_list.append(candidate)
            candidate.extend_end = len(candidate.full_untruncated_fill_ids)
        else:
            pass
        return AddReqResult.CONTINUE

    monkeypatch.setattr(PrefillAdder, "add_one_req", native_add)
    result = adder.add_one_req(request, False, None)

    assert result is AddReqResult.CONTINUE
    assert len(native_calls) == 1
    assert temporal.admitted is expected_temporal
    if expected_temporal:
        assert native_calls == [(10, 11, 12, 13, 20, 21)]
        assert request.multimodal_inputs.mrope_positions[0].tolist() == [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    else:
        assert native_calls == [(10, 11, 12, 13)]
        assert request.full_untruncated_fill_ids == array("q", [10, 11, 12, 13])
        assert request.multimodal_inputs.mrope_positions.shape[1] == 4

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from array import array
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.sampling.sampling_params import SamplingParams

import sglang_omni.models.qwen3_asr.temporal_prefill as temporal_prefill
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
)
from sglang_omni.models.qwen3_asr.temporal_prefill import Qwen3ASRTemporalPrefillAdder
from sglang_omni.proto.realtime import RealtimeTranscriptionContext


@dataclass(frozen=True, slots=True)
class PrefillAdmissionProbe:
    prefix_len: int
    extend_len: int
    is_chunked: bool


def make_temporal_admission_state(
    monkeypatch: pytest.MonkeyPatch,
    prefill_backend: Backend,
) -> tuple[
    Qwen3ASRTemporalPrefillAdder,
    Req,
    Qwen3ASRTemporalRequest,
]:
    class Allocator:
        pass

    monkeypatch.setattr(temporal_prefill, "TokenToKVPoolAllocator", Allocator)
    monkeypatch.setattr(
        temporal_prefill,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend=prefill_backend)
                )
            )
        ),
    )
    adder = object.__new__(Qwen3ASRTemporalPrefillAdder)
    adder.can_run_list = []
    adder.running_batch = SimpleNamespace(
        reqs=[], spec_algorithm=SimpleNamespace(is_none=lambda: True)
    )
    adder.waiting_queue_len = 1
    adder.page_size = 1
    adder.token_to_kv_pool_allocator = Allocator()
    adder.is_hybrid_swa = False
    adder.is_hybrid_ssm_cache = False
    adder.dllm_config = None

    request_id = "session:segment:attempt"
    context = RealtimeTranscriptionContext(
        request_id=request_id,
        segment_id=1,
        attempt=1,
        is_final=True,
        previous_hypothesis="prior words",
        language="English",
    )
    candidate_token_ids = (20, 21)
    temporal_request = Qwen3ASRTemporalRequest(
        context=context,
        candidate_token_ids=candidate_token_ids,
        prompt_boundary=4,
    )
    prompt_token_ids = array("q", [10, 11, 12, 13])
    request = Req(
        request_id,
        "",
        prompt_token_ids,
        SamplingParams(max_new_tokens=3, temperature=0, top_p=1),
        eos_token_ids={2},
        vocab_size=64,
    )
    request.full_untruncated_fill_ids = prompt_token_ids[:]
    request.omni_data = Qwen3ASRRequestData(
        req=SimpleNamespace(), temporal_request=temporal_request
    )
    request.multimodal_inputs = SimpleNamespace(
        mrope_positions=torch.arange(4).expand(3, -1).clone()
    )
    request.kv.cache_protected_len = 0
    request.kv.cache_inserted_len = 0
    request.kv.kv_allocated_len = 0
    request.kv.kv_committed_len = 0
    return adder, request, temporal_request


def test_breakable_prefill_admits_a_supported_temporal_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adder, request, temporal_request = make_temporal_admission_state(
        monkeypatch, Backend.BREAKABLE
    )

    assert request.sampling_params.temperature == 1
    assert request.sampling_params.top_k == 1
    assert request.sampling_params.stop_regex_strs is None
    assert adder.can_admit_temporal(request, temporal_request, has_chunked_req=False)
    assert temporal_request.fallback_reason is None
    assert not temporal_request.admitted
    assert request.full_untruncated_fill_ids == request.origin_input_ids


@pytest.mark.parametrize(
    ("fallback_case", "expected_reason"),
    [
        ("full_graph", "unsupported_prefill_graph_backend"),
        ("page_size", "unsupported_admission_state"),
        ("prefix_cache", "unsupported_admission_state"),
        ("host_cache", "unsupported_admission_state"),
        ("chunked", "unsupported_admission_state"),
        ("mixed_waiting", "unsupported_admission_state"),
        ("mixed_running", "unsupported_admission_state"),
        ("sampling", "unsupported_admission_state"),
        ("output_budget", "unsupported_admission_state"),
        ("attempted", "unsupported_admission_state"),
    ],
)
def test_unsupported_temporal_admission_falls_back_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    fallback_case: str,
    expected_reason: str,
) -> None:
    prefill_backend = (
        Backend.FULL if fallback_case == "full_graph" else Backend.BREAKABLE
    )
    adder, request, temporal_request = make_temporal_admission_state(
        monkeypatch, prefill_backend
    )
    has_chunked_req = False
    if fallback_case == "page_size":
        adder.page_size = 16
    elif fallback_case == "prefix_cache":
        request.prefix_len = 1
    elif fallback_case == "host_cache":
        request.host_hit_length = 1
    elif fallback_case == "chunked":
        has_chunked_req = True
    elif fallback_case == "mixed_waiting":
        adder.waiting_queue_len = 2
    elif fallback_case == "mixed_running":
        adder.running_batch.reqs = [SimpleNamespace()]
    elif fallback_case == "sampling":
        request.sampling_params.top_k = 4
    elif fallback_case == "output_budget":
        request.sampling_params.max_new_tokens = 2
    elif fallback_case == "attempted":
        temporal_request.attempted = True
    else:
        assert fallback_case == "full_graph"

    assert not adder.can_admit_temporal(
        request, temporal_request, has_chunked_req=has_chunked_req
    )
    assert temporal_request.fallback_reason == expected_reason
    assert not temporal_request.admitted
    assert request.full_untruncated_fill_ids == request.origin_input_ids


def install_native_admission_boundary(
    monkeypatch: pytest.MonkeyPatch,
    adder: Qwen3ASRTemporalPrefillAdder,
    *,
    preflight_verdict: AddReqResult | None,
    preflight_is_chunked: bool = False,
    native_verdict: AddReqResult,
    admit_request: bool,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[int]]:
    native_fill_ids: list[tuple[int, ...]] = []
    native_mrope_positions: list[tuple[int, ...]] = []
    native_calls: list[int] = []

    monkeypatch.setattr(adder, "per_req_token_overhead", 0, raising=False)

    def zero_mamba_gap_budget(_request: Req, *, include_host_hit: bool = False) -> int:
        return 0

    def zero_request_slot_reserve(
        _request: Req, *, extend_input_len: int | None = None
    ) -> int:
        return 0

    def select_prefill_admission(
        request: Req,
        *,
        total_tokens: int,
        host_hit_length: int,
        swa_host_hit_length: int,
        truncation_align_size: int | None,
        has_chunked_req: bool,
    ) -> AddReqResult | PrefillAdmissionProbe:
        if preflight_verdict is not None:
            return preflight_verdict
        else:
            pass
        return PrefillAdmissionProbe(
            prefix_len=request.prefix_len,
            extend_len=len(request.full_untruncated_fill_ids) - request.prefix_len,
            is_chunked=preflight_is_chunked,
        )

    monkeypatch.setattr(adder, "_mamba_gap_budget_for_req", zero_mamba_gap_budget)
    monkeypatch.setattr(
        adder, "_request_slot_reserve_for_req", zero_request_slot_reserve
    )
    monkeypatch.setattr(adder, "_select_prefill_admission", select_prefill_admission)

    def native_add_one_req(
        native_adder: PrefillAdder,
        request: Req,
        has_chunked_req: bool,
        truncation_align_size: int | None,
    ) -> AddReqResult:
        assert not has_chunked_req
        assert truncation_align_size is None
        native_calls.append(1)
        native_fill_ids.append(tuple(request.full_untruncated_fill_ids))
        native_mrope_positions.append(
            tuple(request.multimodal_inputs.mrope_positions[0].tolist())
        )
        if admit_request:
            native_adder.can_run_list.append(request)
            request.extend_end = len(request.full_untruncated_fill_ids)
        else:
            pass
        return native_verdict

    monkeypatch.setattr(PrefillAdder, "add_one_req", native_add_one_req)
    return native_fill_ids, native_mrope_positions, native_calls


@pytest.mark.parametrize(
    (
        "preflight_verdict",
        "preflight_is_chunked",
        "native_verdict",
        "native_admits_request",
        "expected_temporal_admission",
        "expected_fallback_reason",
        "expected_native_fill_ids",
    ),
    [
        (
            None,
            False,
            AddReqResult.CONTINUE,
            True,
            True,
            None,
            (10, 11, 12, 13, 20, 21),
        ),
        (
            AddReqResult.NO_TOKEN,
            False,
            AddReqResult.CONTINUE,
            True,
            False,
            "prefill_budget_or_shape",
            (10, 11, 12, 13),
        ),
        (
            None,
            True,
            AddReqResult.CONTINUE,
            True,
            False,
            "prefill_budget_or_shape",
            (10, 11, 12, 13),
        ),
        (
            None,
            False,
            AddReqResult.OTHER,
            False,
            False,
            "native_admission_rejected",
            (10, 11, 12, 13, 20, 21),
        ),
        (
            None,
            False,
            AddReqResult.NO_TOKEN,
            True,
            True,
            None,
            (10, 11, 12, 13, 20, 21),
        ),
    ],
)
def test_add_one_req_commits_or_restores_the_single_native_admission(
    monkeypatch: pytest.MonkeyPatch,
    preflight_verdict: AddReqResult | None,
    preflight_is_chunked: bool,
    native_verdict: AddReqResult,
    native_admits_request: bool,
    expected_temporal_admission: bool,
    expected_fallback_reason: str | None,
    expected_native_fill_ids: tuple[int, ...],
) -> None:
    adder, request, temporal_request = make_temporal_admission_state(
        monkeypatch, Backend.BREAKABLE
    )
    origin_input_ids = request.origin_input_ids[:]
    canonical_fill_ids = request.full_untruncated_fill_ids[:]
    canonical_mrope_positions = request.multimodal_inputs.mrope_positions.clone()
    native_fill_ids, native_mrope_positions, native_calls = (
        install_native_admission_boundary(
            monkeypatch,
            adder,
            preflight_verdict=preflight_verdict,
            preflight_is_chunked=preflight_is_chunked,
            native_verdict=native_verdict,
            admit_request=native_admits_request,
        )
    )

    result = adder.add_one_req(request, False, None)

    assert result is native_verdict
    assert native_calls == [1]
    assert native_fill_ids == [expected_native_fill_ids]
    assert request.origin_input_ids == origin_input_ids
    assert isinstance(request.origin_input_ids, array)
    assert isinstance(request.full_untruncated_fill_ids, array)
    assert temporal_request.admitted is expected_temporal_admission
    assert temporal_request.attempted is expected_temporal_admission
    assert temporal_request.fallback_reason == expected_fallback_reason
    assert (request in adder.can_run_list) is native_admits_request
    if expected_temporal_admission:
        assert request.full_untruncated_fill_ids == array(
            "q", [*canonical_fill_ids, 20, 21]
        )
        assert native_mrope_positions == [(0, 1, 2, 3, 4, 5)]
    else:
        assert request.full_untruncated_fill_ids == canonical_fill_ids
        assert torch.equal(
            request.multimodal_inputs.mrope_positions, canonical_mrope_positions
        )
        expected_native_mrope = (
            (0, 1, 2, 3, 4, 5)
            if preflight_verdict is None and not preflight_is_chunked
            else (0, 1, 2, 3)
        )
        assert native_mrope_positions == [expected_native_mrope]


def test_add_one_req_restores_state_when_preflight_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adder, request, temporal_request = make_temporal_admission_state(
        monkeypatch, Backend.BREAKABLE
    )
    canonical_fill_ids = request.full_untruncated_fill_ids[:]
    origin_input_ids = request.origin_input_ids[:]
    canonical_mrope_positions = request.multimodal_inputs.mrope_positions.clone()
    native_fill_ids, _, native_calls = install_native_admission_boundary(
        monkeypatch,
        adder,
        preflight_verdict=None,
        native_verdict=AddReqResult.CONTINUE,
        admit_request=False,
    )

    def fail_preflight(
        _request: Req,
        *,
        total_tokens: int,
        host_hit_length: int,
        swa_host_hit_length: int,
        truncation_align_size: int | None,
        has_chunked_req: bool,
    ) -> AddReqResult | PrefillAdmissionProbe:
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(adder, "_select_prefill_admission", fail_preflight)
    with pytest.raises(RuntimeError, match="preflight failed"):
        adder.add_one_req(request, False, None)

    assert native_calls == []
    assert native_fill_ids == []
    assert request.origin_input_ids == origin_input_ids
    assert isinstance(request.full_untruncated_fill_ids, array)
    assert request.full_untruncated_fill_ids == canonical_fill_ids
    assert torch.equal(
        request.multimodal_inputs.mrope_positions, canonical_mrope_positions
    )
    assert not temporal_request.admitted
    assert not temporal_request.attempted

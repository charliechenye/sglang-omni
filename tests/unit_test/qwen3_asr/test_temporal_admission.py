# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.cuda_graph_config import Backend

import sglang_omni.models.qwen3_asr.temporal_prefill as temporal_prefill
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
)
from sglang_omni.models.qwen3_asr.temporal_prefill import Qwen3ASRTemporalPrefillAdder
from sglang_omni.proto.realtime import RealtimeTranscriptionContext


def make_temporal_admission_state(
    monkeypatch: pytest.MonkeyPatch,
    prefill_backend: Backend,
) -> tuple[
    Qwen3ASRTemporalPrefillAdder,
    SimpleNamespace,
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
    prompt_token_ids = [10, 11, 12, 13]
    request = SimpleNamespace(
        rid=request_id,
        omni_data=Qwen3ASRRequestData(
            req=SimpleNamespace(), temporal_request=temporal_request
        ),
        origin_input_ids=prompt_token_ids,
        full_untruncated_fill_ids=prompt_token_ids.copy(),
        prefix_len=0,
        host_hit_length=0,
        swa_host_hit_length=0,
        needs_host_load_back=lambda: False,
        output_ids=[],
        kv=SimpleNamespace(
            cache_protected_len=0,
            cache_inserted_len=0,
        ),
        to_finish=None,
        discard_output_reason=None,
        session=None,
        session_id=None,
        is_retracted=False,
        finished=lambda: False,
        return_logprob=False,
        return_sampling_mask=False,
        custom_logit_processor=None,
        grammar=None,
        multimodal_inputs=SimpleNamespace(
            mrope_positions=torch.arange(4).expand(3, -1).clone()
        ),
        sampling_params=SimpleNamespace(
            temperature=1,
            top_p=1,
            top_k=1,
            min_p=0,
            min_new_tokens=0,
            ignore_eos=False,
            repetition_penalty=1,
            frequency_penalty=0,
            presence_penalty=0,
            custom_params={},
            stop_strs=[],
            max_new_tokens=3,
        ),
    )
    return adder, request, temporal_request


def test_breakable_prefill_admits_a_supported_temporal_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adder, request, temporal_request = make_temporal_admission_state(
        monkeypatch, Backend.BREAKABLE
    )

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
    else:
        assert fallback_case == "full_graph"

    assert not adder.can_admit_temporal(
        request, temporal_request, has_chunked_req=has_chunked_req
    )
    assert temporal_request.fallback_reason == expected_reason
    assert not temporal_request.admitted
    assert request.full_untruncated_fill_ids == request.origin_input_ids

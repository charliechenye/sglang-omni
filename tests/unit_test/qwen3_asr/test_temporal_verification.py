# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.models.qwen3_asr.temporal_scheduler import Qwen3ASRTemporalScheduler
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalBatchResult,
    verify_greedy_candidates,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.realtime import RealtimeTranscriptionContext


def test_temporal_verification_commits_only_the_contiguous_target_prefix() -> None:
    """A mismatch must discard later guesses and return its corrective token."""
    cases = [
        ((21, 22), [21, 22, 2], (21, 22), 2, 14),
        ((21,), [21, 30], (21,), 30, 13),
        ((21, 22), [99, 22, 2], (), 99, 12),
        ((21, 22, 23, 24), [21, 22, 99, 24, 2], (21, 22), 99, 14),
        ((21, 22), [21, 2, 99], (21,), 2, 13),
        ((), [2], (), 2, 12),
    ]

    for (
        candidate_token_ids,
        target_token_ids,
        accepted_prefix,
        pending_token_id,
        boundary,
    ) in cases:
        verification = verify_greedy_candidates(
            request_id="final-3-1",
            row_index=0,
            prompt_boundary=12,
            candidate_token_ids=candidate_token_ids,
            target_token_ids=target_token_ids,
        )

        assert verification.accepted_prefix == accepted_prefix
        assert verification.pending_token_id == pending_token_id
        assert verification.owned_kv_boundary == boundary


@pytest.mark.parametrize(
    (
        "enabled",
        "is_final",
        "request_id",
        "max_new_tokens",
        "exclude_candidate_special",
        "expected_candidates",
    ),
    [
        (False, True, "final-3-1", 8, False, ()),
        (True, False, "final-3-1", 8, False, ()),
        (True, True, "stale-3-1", 8, False, ()),
        (True, True, "final-3-1", 1, False, ()),
        (True, True, "final-3-1", 2, False, (24,)),
        (True, True, "final-3-1", 8, False, (24, 25)),
        (True, True, "final-3-1", 8, True, ()),
    ],
)
def test_final_temporal_candidates_require_the_live_enabled_final_request(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    is_final: bool,
    request_id: str,
    max_new_tokens: int,
    exclude_candidate_special: bool,
    expected_candidates: tuple[int, ...],
) -> None:
    class TemporalTokenizer(Tokenizer):
        eos_token_id = 2
        all_special_ids = [2, 42]

        def __len__(self) -> int:
            return 102

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|audio_pad|>"
            return 42

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            if text == "<asr_text>":
                return [100, 101]
            elif text == "abcdef":
                return [20, 21, 22, 23, 24, 25]
            else:
                raise AssertionError(f"unexpected tokenizer input {text!r}")

        def __call__(self, _text: str, *, add_special_tokens: bool = False):
            assert not add_special_tokens
            return SimpleNamespace(input_ids=[11, 42, 12, 13])

        def decode(self, token_ids: list[int], **_: object) -> str:
            return "".join(
                {20: "a", 21: "b", 22: "c", 23: "d", 24: "e", 25: "f"}[token_id]
                for token_id in token_ids
            )

    if exclude_candidate_special:
        TemporalTokenizer.all_special_ids = [2, 42, 24]
    else:
        pass

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=TemporalTokenizer(),
        max_new_tokens=max_new_tokens,
        feature_extractor=lambda *args, **kwargs: SimpleNamespace(
            input_features=torch.zeros((1, 128, 100)),
            attention_mask=torch.ones((1, 100), dtype=torch.long),
        ),
        enable_temporal_final_verification=enabled,
    )
    payload = StagePayload(
        request_id="final-3-1",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={
                "language": "English",
                "_asr_streaming": True,
                "_asr_streaming_prefix_text": "abcdef",
                "_asr_streaming_rollback_tokens": 2,
            },
            realtime_transcription_context=RealtimeTranscriptionContext(
                request_id=request_id,
                segment_id=3,
                attempt=1,
                is_final=is_final,
                previous_hypothesis="abcdef",
                language="English",
            ),
        ),
        data={},
    )

    request_data = request_builder(payload)

    if expected_candidates:
        assert request_data.temporal_request is not None
        assert request_data.temporal_request.candidate_token_ids == expected_candidates
        assert request_data.req.sampling_params.temperature == 1
        assert request_data.req.sampling_params.top_k == 1
    else:
        assert request_data.temporal_request is None
        if enabled and is_final and max_new_tokens == 1:
            assert request_data.temporal_fallback_reason == "output_budget_too_small"
        elif enabled and is_final and exclude_candidate_special:
            assert (
                request_data.temporal_fallback_reason
                == "invalid_temporal_candidate_ids"
            )
        elif enabled and is_final and request_id != "final-3-1":
            assert request_data.temporal_fallback_reason == "invalid_realtime_context"
        else:
            assert request_data.temporal_fallback_reason is None


def test_realtime_context_round_trips_and_rejects_malformed_wire_values() -> None:
    context = RealtimeTranscriptionContext(
        request_id="session-3-1",
        segment_id=3,
        attempt=1,
        is_final=True,
        previous_hypothesis="recognized words",
        language="English",
    )
    request = OmniRequest(
        inputs={"audio_bytes": b"wav"},
        realtime_transcription_context=context,
    )

    assert OmniRequest.from_dict(request.to_dict()) == request
    malformed = request.to_dict()
    malformed_context = malformed["realtime_transcription_context"]
    assert isinstance(malformed_context, dict)
    malformed_context["is_final"] = "true"
    assert OmniRequest.from_dict(malformed).realtime_transcription_context is None


@pytest.mark.parametrize(
    (
        "target_token_ids",
        "accepted_prefix",
        "pending_token_id",
        "cancel_before_commit",
    ),
    [
        ([20, 22, 23, 2], (20,), 22, False),
        ([99, 20, 21, 2], (), 99, False),
        ([20, 21, 22, 2], (20, 21, 22), 2, False),
        ([20, 21, 22, 2], (), 2, True),
    ],
)
def test_temporal_precommit_reclaims_only_the_rejected_suffix(
    target_token_ids: list[int],
    accepted_prefix: tuple[int, ...],
    pending_token_id: int,
    cancel_before_commit: bool,
) -> None:
    request_id = "final-3-1"
    prompt_token_ids = list(range(10))
    candidate_token_ids = (20, 21, 22)
    context = RealtimeTranscriptionContext(
        request_id=request_id,
        segment_id=3,
        attempt=1,
        is_final=True,
        previous_hypothesis="recognized words",
        language="English",
    )
    temporal_request = Qwen3ASRTemporalRequest(
        context=context,
        candidate_token_ids=candidate_token_ids,
        prompt_boundary=len(prompt_token_ids),
        admitted=True,
    )
    verification = verify_greedy_candidates(
        request_id=request_id,
        row_index=0,
        prompt_boundary=len(prompt_token_ids),
        candidate_token_ids=candidate_token_ids,
        target_token_ids=target_token_ids,
    )
    request = SimpleNamespace(
        rid=request_id,
        origin_input_ids=prompt_token_ids,
        full_untruncated_fill_ids=[*prompt_token_ids, *candidate_token_ids],
        output_ids=[],
        extend_end=len(prompt_token_ids) + len(candidate_token_ids),
        kv=SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=0,
            cache_inserted_len=0,
            kv_committed_len=len(prompt_token_ids) + len(candidate_token_ids),
            kv_allocated_len=len(prompt_token_ids) + len(candidate_token_ids),
        ),
        multimodal_inputs=SimpleNamespace(
            mrope_positions=torch.arange(13).expand(3, -1).clone()
        ),
        sampling_params=SimpleNamespace(max_new_tokens=8),
        to_finish="abort" if cancel_before_commit else None,
        is_retracted=False,
        discard_output_reason=None,
        skip_radix_cache_insert=False,
        finished=lambda: request.to_finish is not None,
    )

    def refresh_fill_ids() -> None:
        request.fill_ids = [*request.origin_input_ids, *request.output_ids]

    request.refresh_fill_ids = refresh_fill_ids
    request_data = Qwen3ASRRequestData(
        req=request,
        temporal_request=temporal_request,
    )
    request.omni_data = request_data

    class RecordingAllocator:
        def __init__(self) -> None:
            self.freed: list[torch.Tensor] = []

        def free(self, locations: torch.Tensor) -> None:
            self.freed.append(locations.clone())

    allocator = RecordingAllocator()
    token_row = torch.arange(1, 14, dtype=torch.long)
    scheduler = object.__new__(Qwen3ASRTemporalScheduler)
    scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=token_row.unsqueeze(0).clone()
    )
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.aborted_request_ids = set()
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        reqs=[request],
        seq_lens=torch.tensor([13], dtype=torch.int32),
        seq_lens_cpu=[13],
    )
    batch_result = Qwen3ASRTemporalBatchResult(
        native_result=SimpleNamespace(
            next_token_ids=torch.tensor([pending_token_id], dtype=torch.long)
        ),
        temporal_verification=verification,
    )

    Qwen3ASRTemporalScheduler.precommit_temporal_result(
        scheduler, schedule_batch, batch_result
    )

    committed_prefix = () if cancel_before_commit else accepted_prefix
    owned_boundary = len(prompt_token_ids) + len(committed_prefix)
    assert request.output_ids == list(committed_prefix)
    assert request.full_untruncated_fill_ids == prompt_token_ids
    assert request.kv.kv_committed_len == owned_boundary
    assert request.kv.kv_allocated_len == owned_boundary
    assert schedule_batch.seq_lens_cpu == [owned_boundary]
    assert schedule_batch.seq_lens.tolist() == [owned_boundary]
    assert request.multimodal_inputs.mrope_positions.shape[1] == len(prompt_token_ids)
    assert scheduler.req_to_token_pool.req_to_token[0, owned_boundary:].tolist() == [
        0
    ] * (len(candidate_token_ids) - len(accepted_prefix))
    if len(candidate_token_ids) == len(accepted_prefix):
        assert allocator.freed == []
    else:
        assert len(allocator.freed) == 1
        assert allocator.freed[0].numel() == len(candidate_token_ids) - len(
            accepted_prefix
        )
    assert batch_result.native_result.next_token_ids.tolist() == [pending_token_id]
    assert temporal_request.cancelled is cancel_before_commit

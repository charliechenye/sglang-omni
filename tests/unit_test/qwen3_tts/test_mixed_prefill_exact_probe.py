# SPDX-License-Identifier: Apache-2.0
"""Research-only exact service probe for Qwen3-TTS mixed prefill."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.scheduling.types import RequestOutput


def _request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(
            output_codes=[],
            latest_stream_code_chunk=None,
            pending_feedback_queue=[],
        ),
    )


def test_mixed_prefill_emits_one_batch_codec_frame_ready_event(monkeypatch) -> None:
    emitted: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_tts.model_runner._emit_request_event",
        lambda **kwargs: emitted.append(kwargs),
    )

    requests = [_request("newcomer")] + [
        _request(f"incumbent-{idx:02d}") for idx in range(15)
    ]
    batch_size = len(requests)

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner._has_pending_code_step = False
    runner._pending_mixed_codec_probe = None
    runner._stage_token_ids = lambda result, token_ids: None
    runner._sample_positions = lambda forward_batch, device: torch.zeros(
        batch_size, dtype=torch.long, device=device
    )
    runner.model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=-1),
        code_predictor_forward=lambda *args, **kwargs: None,
        _output_codes=torch.arange(batch_size * 2, dtype=torch.long).reshape(
            batch_size, 2
        ),
        _output_embeds=torch.ones(batch_size, 4),
    )

    result = SimpleNamespace(
        next_token_ids=torch.full((batch_size,), 7, dtype=torch.long),
        logits_output=SimpleNamespace(hidden_states=torch.ones(batch_size, 4)),
    )
    forward_batch = SimpleNamespace()
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_mixed=lambda: True),
        extend_lens=[24] + [1] * 15,
    )

    runner.post_prefill(result, forward_batch, schedule_batch, requests)
    assert runner._pending_mixed_codec_probe is not None

    outputs = {
        request.request_id: RequestOutput(request.request_id, data=7)
        for request in requests
    }
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        outputs,
    )

    assert len(emitted) == 1
    event = emitted[0]
    assert event["request_id"] == "newcomer"
    assert event["event_name"] == "mixed_codec_frame_ready"
    assert isinstance(event["timestamp_ns"], int)
    assert event["metadata"]["forward_mode"] == "MIXED"
    assert event["metadata"]["batch_size"] == 16
    assert event["metadata"]["newcomer_count"] == 1
    assert event["metadata"]["one_token_row_count"] == 15
    assert event["metadata"]["committed_one_token_row_count"] == 15
    assert event["metadata"]["extend_num_tokens"] == 39
    assert runner._pending_mixed_codec_probe is None

    for request in requests[1:]:
        assert len(request.data.output_codes) == 1
        assert len(request.data.pending_feedback_queue) == 1


def test_non_mixed_prefill_does_not_emit_probe(monkeypatch) -> None:
    emitted: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_tts.model_runner._emit_request_event",
        lambda **kwargs: emitted.append(kwargs),
    )

    request = _request("ordinary")
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner._has_pending_code_step = False
    runner._pending_mixed_codec_probe = None
    runner._stage_token_ids = lambda result, token_ids: None
    runner._sample_positions = lambda forward_batch, device: torch.zeros(
        1, dtype=torch.long, device=device
    )
    runner.model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=-1),
        code_predictor_forward=lambda *args, **kwargs: None,
        _output_codes=torch.ones(1, 2, dtype=torch.long),
        _output_embeds=torch.ones(1, 4),
    )

    result = SimpleNamespace(
        next_token_ids=torch.tensor([7], dtype=torch.long),
        logits_output=SimpleNamespace(hidden_states=torch.ones(1, 4)),
    )
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_mixed=lambda: False),
        extend_lens=[24],
    )

    runner.post_prefill(result, SimpleNamespace(), schedule_batch, [request])
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=[request]),
        {"ordinary": RequestOutput("ordinary", data=7)},
    )

    assert emitted == []

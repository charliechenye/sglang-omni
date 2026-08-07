# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for Qwen3-Omni's OmniPrefillInputs adoption contract."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.prefill_inputs import OmniPrefillInputs
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


def _runner(*, capture_hidden: bool = False) -> ThinkerModelRunner:
    runner = object.__new__(ThinkerModelRunner)
    runner._should_capture_hidden = lambda request: capture_hidden
    return runner


def _requests(
    model_inputs_list: list[dict | None],
    *,
    output_modalities: tuple[str, ...] = ("text",),
) -> tuple[SimpleNamespace, SimpleNamespace]:
    requests = []
    schedule_reqs = []
    for index, model_inputs in enumerate(model_inputs_list):
        stage_payload = SimpleNamespace(
            metadata={"output_modalities": list(output_modalities)}
        )
        data = SimpleNamespace(stage_payload=stage_payload)
        requests.append(SimpleNamespace(request_id=f"req-{index}", data=data))
        schedule_reqs.append(
            SimpleNamespace(omni_model_inputs=model_inputs, request_id=f"req-{index}")
        )
    return (
        SimpleNamespace(reqs=schedule_reqs),
        requests,
    )


def _audio_inputs() -> dict:
    return {
        "audio_embeds": torch.zeros(2, 4),
        "audio_feature_lengths": torch.tensor([8]),
        "feature_attention_mask": torch.ones(1, 8, dtype=torch.long),
        "pad_values": {"audio": 1001},
    }


def _forward_batch(*, mm_inputs, batch_size: int = 1, num_tokens: int = 3):
    return SimpleNamespace(
        input_embeds=None,
        mm_inputs=mm_inputs,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        batch_size=batch_size,
    )


def test_text_only_batch_is_payload_compatible() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([None])

    assert runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_audio_to_text_batch_is_payload_compatible() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([_audio_inputs()])

    assert runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_multiple_audio_requests_and_text_audio_mix_are_compatible() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([_audio_inputs(), _audio_inputs(), None])

    assert runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_speech_output_is_rejected_from_payload_path() -> None:
    runner = _runner()
    schedule_batch, requests = _requests(
        [_audio_inputs()], output_modalities=("audio",)
    )

    assert not runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_hidden_capture_is_rejected_from_payload_path() -> None:
    runner = _runner(capture_hidden=True)
    schedule_batch, requests = _requests([_audio_inputs()])

    assert not runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_visual_and_deepstack_inputs_are_rejected() -> None:
    runner = _runner()
    for inputs in (
        {"image_embeds": torch.zeros(1, 4)},
        {"video_embeds": torch.zeros(1, 4)},
        {"deepstack_visual_embeds": [torch.zeros(1, 4)]},
        {"image_deepstack_visual_embeds": [torch.zeros(1, 4)]},
        {"video_deepstack_visual_embeds": [torch.zeros(1, 4)]},
        {"use_audio_in_video": True},
    ):
        schedule_batch, requests = _requests([inputs])
        assert not runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_unknown_auxiliary_input_is_rejected() -> None:
    runner = _runner()
    schedule_batch, requests = _requests(
        [{**_audio_inputs(), "unqualified_aux": torch.zeros(1)}]
    )

    assert not runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_incompatible_request_rejects_the_entire_batch() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([None, {"image_embeds": torch.zeros(1, 4)}])

    assert not runner._can_use_qwen_prefill_payload(schedule_batch, requests)


def test_before_prefill_replaces_only_metadata_shell_and_attaches_payload(
    monkeypatch,
) -> None:
    runner = _runner()
    schedule_batch, requests = _requests([_audio_inputs()])
    runner._build_prefill_input_embeds = lambda forward_batch, batch: torch.zeros(
        len(forward_batch.input_ids), 4
    )
    shell = SimpleNamespace(mm_items=[])
    forward_batch = _forward_batch(mm_inputs=[shell])

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert forward_batch.input_embeds is None
    assert isinstance(forward_batch.mm_inputs, OmniPrefillInputs)
    assert forward_batch.mm_inputs.rids == ("req-0",)
    assert forward_batch.mm_inputs.input_embeds.shape == (3, 4)


def test_before_prefill_fails_closed_on_genuine_multimodal_items() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([_audio_inputs()])
    runner._build_prefill_input_embeds = lambda forward_batch, batch: torch.zeros(
        len(forward_batch.input_ids), 4
    )
    genuine = SimpleNamespace(mm_items=[object()])
    forward_batch = _forward_batch(mm_inputs=[genuine])

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert forward_batch.mm_inputs == [genuine]
    assert forward_batch.input_embeds is None


def test_before_prefill_accepts_no_mm_input_placeholder() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([None])
    runner._build_prefill_input_embeds = lambda forward_batch, batch: torch.zeros(
        len(forward_batch.input_ids), 4
    )
    forward_batch = _forward_batch(mm_inputs=None)

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert isinstance(forward_batch.mm_inputs, OmniPrefillInputs)
    assert forward_batch.input_embeds is None


def test_payload_batch_custom_forward_delegates_to_normal_worker() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([None])
    forward_batch = _forward_batch(mm_inputs=None)
    forward_batch.mm_inputs = OmniPrefillInputs(
        input_embeds=torch.zeros(3, 4), rids=("req-0",)
    )
    schedule_batch.forward_mode = SimpleNamespace(is_extend=lambda: True)

    assert runner.custom_prefill_forward(forward_batch, schedule_batch, requests) is None

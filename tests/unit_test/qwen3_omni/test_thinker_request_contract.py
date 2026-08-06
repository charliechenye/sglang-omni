# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalInputFormat

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    build_sglang_thinker_request,
    make_thinker_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer


def _thinker_config(audio_token_id: int = 77) -> SimpleNamespace:
    return SimpleNamespace(
        image_token_id=55,
        video_token_id=66,
        audio_token_id=audio_token_id,
    )


def _patch_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.request_builders._compute_mrope_positions",
        lambda input_ids, model_inputs, thinker_config: (
            torch.zeros((3, input_ids.numel()), dtype=torch.long),
            torch.tensor(0),
        ),
    )


def _audio_state(
    input_ids: torch.Tensor,
    audio_embeds: torch.Tensor,
    *,
    media_cache_keys: dict[str, str] | None = None,
    extra_model_inputs: dict[str, object] | None = None,
) -> Qwen3OmniPipelineState:
    model_inputs: dict[str, object] = {"audio_embeds": audio_embeds}
    model_inputs.update(extra_model_inputs or {})
    thinker_inputs: dict[str, object] = {"model_inputs": model_inputs}
    if media_cache_keys is not None:
        thinker_inputs["media_cache_keys"] = media_cache_keys
    return Qwen3OmniPipelineState(
        prompt={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "prompt_text": "audio",
        },
        thinker_inputs=thinker_inputs,
    )


def _payload(state: Qwen3OmniPipelineState, *, output_modalities: list[str]) -> StagePayload:
    return StagePayload(
        request_id="request",
        request=OmniRequest(
            inputs=[], metadata={"output_modalities": output_modalities}
        ),
        data=state.to_dict(),
    )


def test_audio_to_text_uses_native_precomputed_item_without_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_builder(monkeypatch)
    audio = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    state = _audio_state(
        torch.tensor([10, 77, 11, 77, 12]),
        audio,
        media_cache_keys={"audio": "audio:cache"},
    )
    payload = _payload(state, output_modalities=["text"])

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )
    request_data = request_builder(payload)
    req = request_data.req
    item = req.multimodal_inputs.mm_items[0]

    assert req.input_embeds is None
    assert req.omni_model_inputs is None
    assert item.format is MultimodalInputFormat.PRECOMPUTED_EMBEDDING
    assert item.precomputed_embeddings is audio
    assert item.model_specific_data["positions_cpu"].tolist() == [1, 3]
    assert request_data.model_inputs.get("audio_embeds") is None
    assert "audio_embeds" not in payload.data["thinker_inputs"]["model_inputs"]


def test_audio_to_text_without_cache_key_keeps_audio_token_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_builder(monkeypatch)
    audio = torch.ones((2, 4))
    state = _audio_state(torch.tensor([10, 77, 11, 77]), audio)
    payload = _payload(state, output_modalities=["text"])

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )
    request_data = request_builder(payload)
    item = request_data.req.multimodal_inputs.mm_items[0]

    assert item.pad_value == 77
    assert item.model_specific_data["positions_cpu"].tolist() == [1, 3]


def test_speech_output_remains_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_builder(monkeypatch)
    audio = torch.ones((2, 4))
    state = _audio_state(torch.tensor([10, 77, 11, 77]), audio)
    payload = _payload(state, output_modalities=["audio"])

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )
    request_data = request_builder(payload)

    assert request_data.req.omni_model_inputs is not None
    assert request_data.req.multimodal_inputs.mm_items == []
    assert "audio_embeds" in payload.data["thinker_inputs"]["model_inputs"]


def test_visual_audio_request_remains_fully_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_builder(monkeypatch)
    audio = torch.ones((2, 4))
    state = _audio_state(
        torch.tensor([10, 77, 11, 77]),
        audio,
        extra_model_inputs={"image_embeds": torch.ones((1, 4))},
    )
    payload = _payload(state, output_modalities=["text"])

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )
    request_data = request_builder(payload)

    assert request_data.req.omni_model_inputs is not None
    assert request_data.req.multimodal_inputs.mm_items == []
    assert "audio_embeds" in payload.data["thinker_inputs"]["model_inputs"]


def test_invalid_audio_row_count_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_builder(monkeypatch)
    state = _audio_state(torch.tensor([10, 77, 11, 77]), torch.ones((1, 4)))
    payload = _payload(state, output_modalities=["text"])

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )
    with pytest.raises(ValueError, match="rows do not match"):
        request_builder(payload)


def test_direct_legacy_builder_compatibility_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_builder(monkeypatch)
    audio = torch.ones((1, 4))
    state = _audio_state(torch.tensor([10, 77, 11]), audio)

    request_data = build_sglang_thinker_request(
        state,
        params={},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )

    assert request_data.req.omni_model_inputs is not None

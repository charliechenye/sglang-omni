# SPDX-License-Identifier: Apache-2.0
"""CPU semantic contracts for the generated Thinker -> Talker payload."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.components.talker_input import (  # noqa: E402
    build_prefill_input,
)
from sglang_omni.models.qwen3_omni.components.talker_prefill import (  # noqa: E402
    TalkerPrefillBuilder,
)
from sglang_omni.models.qwen3_omni.request_builders import (  # noqa: E402
    make_thinker_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload  # noqa: E402

HIDDEN = 4
IM_START = 1
USER = 2
ASSISTANT = 3
IM_END = 99
AUDIO_TOKEN = 40
IMAGE_TOKEN = 41


def _audio_stage_payload(*, stream: bool = False) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=[],
            params={"stream": stream},
            metadata={"output_modalities": ["audio"]},
        ),
        data={},
    )


def _thinker_req_data(*, stream: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=_audio_stage_payload(stream=stream),
    )


def test_make_thinker_stream_output_builder_drops_generated_layer_hidden() -> None:
    builder = make_thinker_stream_output_builder()
    token_id = 123
    embed = torch.tensor([1.0, 2.0, 3.0, 4.0])
    layer_hidden = torch.full((HIDDEN,), 99.0)
    req_output = SimpleNamespace(
        data=token_id,
        extra={"hidden_states": {"embed": embed, 24: layer_hidden}},
    )

    messages = builder("req-1", _thinker_req_data(), req_output)

    assert len(messages) == 1
    message = messages[0]
    assert message.target == "talker_ar"
    assert message.type == "stream"
    assert torch.equal(message.data, embed)
    assert message.metadata == {"token_id": token_id}
    assert "layer_hidden" not in message.metadata


def test_make_thinker_stream_output_builder_preserves_layer_hidden_only_fallback() -> (
    None
):
    builder = make_thinker_stream_output_builder()
    token_id = 321
    layer_hidden = torch.tensor([5.0, 6.0, 7.0, 8.0])
    req_output = SimpleNamespace(
        data=token_id,
        extra={"hidden_states": {24: layer_hidden}},
    )

    messages = builder("req-1", _thinker_req_data(), req_output)

    assert len(messages) == 1
    message = messages[0]
    assert message.target == "talker_ar"
    assert message.type == "stream"
    assert torch.equal(message.data, layer_hidden)
    assert message.metadata == {"token_id": token_id}


def _fake_prefill_builder() -> TalkerPrefillBuilder:
    """Build a CPU-only Talker prefill builder with deterministic projections."""

    builder = object.__new__(TalkerPrefillBuilder)
    builder._device = torch.device("cpu")
    builder._dtype = torch.float32
    builder._model_path = "unused-in-test"
    builder._speaker_map = {}
    builder._audio_token_id = AUDIO_TOKEN
    builder._image_token_id = IMAGE_TOKEN
    builder._video_token_id = None
    builder._im_start_token_id = IM_START
    builder._im_end_token_id = IM_END
    builder._system_token_id = 4
    builder._user_token_id = USER
    builder._assistant_token_id = ASSISTANT
    builder._tts_pad_token_id = 10
    builder._codec_nothink_id = 5
    builder._codec_think_bos_id = 6
    builder._codec_think_eos_id = 7
    builder._codec_pad_id = 8
    builder._codec_bos_id = 9

    def embedding_rows(token_ids: torch.Tensor) -> torch.Tensor:
        rows = token_ids.to(dtype=torch.float32).reshape(-1, 1)
        return rows.expand(-1, HIDDEN).clone()

    def codec_embedding(token_ids: torch.Tensor) -> torch.Tensor:
        rows = token_ids.to(dtype=torch.float32).reshape(-1, 1)
        return rows.expand(-1, HIDDEN).clone()

    builder._load_prompt_token_embeddings = embedding_rows
    builder._model = SimpleNamespace(
        text_projection=lambda rows: rows + 0.5,
        hidden_projection=lambda rows: rows * 3.0,
        get_input_embeddings=lambda: codec_embedding,
    )
    prompt_ids = torch.tensor(
        [IM_START, USER, 10, 11, IM_START, ASSISTANT], dtype=torch.long
    )
    prompt_embed = embedding_rows(prompt_ids)
    prompt_hidden = prompt_embed.clone()
    builder._reconstruct_prompt_states = lambda _state: (
        prompt_ids,
        prompt_embed,
        prompt_hidden,
        {},
    )
    builder.get_tts_special_embeds = lambda: (
        torch.full((1, HIDDEN), 10.0),
        torch.full((1, HIDDEN), 20.0),
        torch.full((1, HIDDEN), 30.0),
    )
    return builder


def _generated_chunks(
    token_ids: list[int],
    *,
    include_layer_hidden: bool,
    poison_data: bool = False,
    poison_layer_hidden: bool = False,
) -> list[SimpleNamespace]:
    chunks = []
    for index, token_id in enumerate(token_ids):
        value = float(index + 1)
        data = torch.full((HIDDEN,), -value if poison_data else value)
        metadata = {"token_id": token_id}
        if include_layer_hidden:
            layer_value = float(index + 101)
            metadata["layer_hidden"] = torch.full(
                (HIDDEN,), -layer_value if poison_layer_hidden else layer_value
            )
        chunks.append(SimpleNamespace(data=data, metadata=metadata))
    return chunks


def _prefill_payload(
    builder: TalkerPrefillBuilder,
    chunks: list[SimpleNamespace],
    *,
    thinker_done: bool = False,
) -> dict:
    return builder.build_prompt_prefill(
        SimpleNamespace(data={}, request=SimpleNamespace(params={})),
        chunks,
        thinker_done=thinker_done,
    )


def _assert_prefill_equal(left: dict, right: dict) -> None:
    for key in ("input_embeds", "input_ids", "tts_pad_embed", "tts_eos_embed"):
        assert torch.equal(left[key], right[key]), key
    left_rows = list(left["pending_text_queue"])
    right_rows = list(right["pending_text_queue"])
    assert len(left_rows) == len(right_rows)
    for left_row, right_row in zip(left_rows, right_rows):
        assert torch.equal(left_row, right_row)


def test_initial_talker_prefill_ignores_generated_layer_hidden() -> None:
    token_ids = [20, 21, 22, 23]
    builder = _fake_prefill_builder()
    baseline = _prefill_payload(
        builder,
        _generated_chunks(token_ids, include_layer_hidden=True),
    )
    candidate = _prefill_payload(
        builder,
        _generated_chunks(token_ids, include_layer_hidden=False),
    )

    _assert_prefill_equal(baseline, candidate)


def test_incremental_talker_projection_ignores_generated_layer_hidden() -> None:
    builder = _fake_prefill_builder()
    baseline = _generated_chunks([7], include_layer_hidden=True)[0]
    candidate = _generated_chunks([7], include_layer_hidden=False)[0]

    torch.testing.assert_close(
        builder.project_assistant_chunk(baseline),
        builder.project_assistant_chunk(candidate),
        rtol=0,
        atol=0,
    )


def test_incremental_talker_projection_uses_token_id_over_primary_data() -> None:
    builder = _fake_prefill_builder()
    normal = _generated_chunks([7], include_layer_hidden=False)[0]
    poisoned = _generated_chunks(
        [7],
        include_layer_hidden=True,
        poison_data=True,
        poison_layer_hidden=True,
    )[0]

    torch.testing.assert_close(
        builder.project_assistant_chunk(normal),
        builder.project_assistant_chunk(poisoned),
        rtol=0,
        atol=0,
    )


def _mixed_prompt_inputs(
    builder: TalkerPrefillBuilder,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    thinker_input_ids = torch.tensor(
        [
            IM_START,
            USER,
            10,
            AUDIO_TOKEN,
            11,
            IMAGE_TOKEN,
            12,
            IM_START,
            ASSISTANT,
            70,
            71,
            72,
            73,
            74,
        ],
        dtype=torch.long,
    )
    thinker_embed = (
        thinker_input_ids.to(dtype=torch.float32)
        .unsqueeze(1)
        .expand(-1, HIDDEN)
        .clone()
    )
    thinker_hidden = torch.arange(
        thinker_input_ids.numel() * HIDDEN, dtype=torch.float32
    ).reshape(-1, HIDDEN)
    return (
        thinker_embed,
        thinker_input_ids,
        thinker_hidden,
        builder.build_multimodal_mask(thinker_input_ids),
    )


def _build_mixed_prefill(
    builder: TalkerPrefillBuilder,
    thinker_embed: torch.Tensor,
    thinker_input_ids: torch.Tensor,
    thinker_hidden: torch.Tensor,
    multimodal_mask: torch.Tensor,
) -> dict:
    tts_bos_embed, tts_eos_embed, tts_pad_embed = builder.get_tts_special_embeds()
    return build_prefill_input(
        thinker_embed=thinker_embed,
        thinker_hidden=thinker_hidden,
        thinker_input_ids=thinker_input_ids,
        multimodal_mask=multimodal_mask,
        text_projection=builder._model.text_projection,
        hidden_projection=builder._model.hidden_projection,
        codec_embed_fn=builder._model.get_input_embeddings(),
        tts_bos_embed=tts_bos_embed,
        tts_eos_embed=tts_eos_embed,
        tts_pad_embed=tts_pad_embed,
        im_start_token_id=builder._im_start_token_id,
        system_token_id=builder._system_token_id,
        user_token_id=builder._user_token_id,
        assistant_token_id=builder._assistant_token_id,
        speaker_id=0,
        codec_nothink_id=5,
        codec_think_bos_id=6,
        codec_think_eos_id=7,
        codec_pad_id=8,
        codec_bos_id=9,
        tts_pad_token_id=10,
        im_end_token_id=builder._im_end_token_id,
    )


def test_prompt_text_hidden_invariance() -> None:
    builder = _fake_prefill_builder()
    thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask = (
        _mixed_prompt_inputs(builder)
    )
    poisoned_hidden = thinker_hidden.clone()
    poisoned_hidden[~multimodal_mask] = -1234.0

    normal = _build_mixed_prefill(
        builder, thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask
    )
    poisoned = _build_mixed_prefill(
        builder,
        thinker_embed,
        thinker_input_ids,
        poisoned_hidden,
        multimodal_mask,
    )

    for key in ("input_embeds", "input_ids", "future_text_rows"):
        assert torch.equal(normal[key], poisoned[key]), key


def test_prompt_multimodal_hidden_liveness() -> None:
    builder = _fake_prefill_builder()
    thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask = (
        _mixed_prompt_inputs(builder)
    )
    changed_hidden = thinker_hidden.clone()
    changed_hidden[multimodal_mask] += torch.tensor(
        [[100.0] * HIDDEN, [200.0] * HIDDEN]
    )

    normal = _build_mixed_prefill(
        builder, thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask
    )
    changed = _build_mixed_prefill(
        builder,
        thinker_embed,
        thinker_input_ids,
        changed_hidden,
        multimodal_mask,
    )

    assert torch.equal(normal["input_ids"], changed["input_ids"])
    assert torch.equal(normal["future_text_rows"], changed["future_text_rows"])
    changed_rows = (normal["input_embeds"] != changed["input_embeds"]).any(dim=1)
    assert changed_rows.nonzero(as_tuple=True)[0].tolist() == [3, 5]

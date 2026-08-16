# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the generated Thinker -> Talker stream payload.

These tests deliberately mutate generated assistant hidden rows while keeping
token IDs fixed.  They are investigation coverage for Track 2.2, not a
production payload change.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker_input import (
    build_prefill_input,
    build_user_part,
    segment_chat_template,
)
from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    TalkerPrefillBuilder,
)


HIDDEN = 4
IM_START = 1
USER = 2
ASSISTANT = 3
IM_END = 99
AUDIO_TOKEN = 40
IMAGE_TOKEN = 41


def _fake_prefill_builder() -> TalkerPrefillBuilder:
    """Build a CPU-only builder with deterministic local embedding rows."""
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

    def embedding_rows(token_ids: torch.Tensor) -> torch.Tensor:
        rows = token_ids.to(dtype=torch.float32).reshape(-1, 1)
        return rows.expand(-1, HIDDEN).clone()

    def codec_embedding(token_ids: torch.Tensor) -> torch.Tensor:
        rows = token_ids.to(dtype=torch.float32).reshape(-1, 1)
        return rows.expand(-1, HIDDEN).clone()

    def get_input_embeddings():
        return codec_embedding

    builder._load_prompt_token_embeddings = embedding_rows
    builder._model = SimpleNamespace(
        text_projection=lambda rows: rows + 0.5,
        hidden_projection=lambda rows: rows * 3.0,
        get_input_embeddings=get_input_embeddings,
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
    tts_specials = (
        torch.full((1, HIDDEN), 10.0),
        torch.full((1, HIDDEN), 20.0),
        torch.full((1, HIDDEN), 30.0),
    )
    builder.get_tts_special_embeds = lambda: tts_specials
    return builder


def _chunks(
    token_ids: list[int],
    *,
    poison_data: bool = False,
    poison_layer: bool = False,
    token_only: bool = False,
) -> list[SimpleNamespace]:
    chunks = []
    for index, token_id in enumerate(token_ids):
        if token_only:
            data = torch.tensor([token_id], dtype=torch.long)
        else:
            value = float(index + 1)
            data = torch.full((HIDDEN,), value if not poison_data else -value)
        metadata = {"token_id": token_id}
        if not token_only:
            layer_value = float(index + 101)
            metadata["layer_hidden"] = torch.full(
                (HIDDEN,), layer_value if not poison_layer else -layer_value
            )
        chunks.append(SimpleNamespace(data=data, metadata=metadata))
    return chunks


def _prefill_payload(
    builder: TalkerPrefillBuilder,
    chunks: list[SimpleNamespace],
    *,
    thinker_done: bool,
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


def _mixed_prompt_inputs(
    builder: TalkerPrefillBuilder,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return one mixed user prompt followed by a multi-token assistant turn."""
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
    multimodal_mask = builder.build_multimodal_mask(thinker_input_ids)
    return thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask


def _build_mixed_prefill(
    builder: TalkerPrefillBuilder,
    thinker_embed: torch.Tensor,
    thinker_input_ids: torch.Tensor,
    thinker_hidden: torch.Tensor,
    multimodal_mask: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
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


def _user_multimodal_output_rows(
    thinker_input_ids: torch.Tensor, multimodal_mask: torch.Tensor
) -> list[int]:
    """Map multimodal user offsets after system segments are omitted."""
    segments = segment_chat_template(
        thinker_input_ids,
        im_start_token_id=IM_START,
        system_token_id=4,
        user_token_id=USER,
        assistant_token_id=ASSISTANT,
    )
    user_index, user_segment = next(
        (index, segment)
        for index, segment in enumerate(segments)
        if segment["role"] == "user"
    )
    output_start = sum(
        segment["end"] - segment["start"]
        for segment in segments[:user_index]
        if segment["role"] != "system"
    )
    user_mask = multimodal_mask[user_segment["start"] : user_segment["end"]]
    return (output_start + user_mask.nonzero(as_tuple=True)[0]).tolist()


def _assert_direct_prefill_equal(
    left: dict[str, torch.Tensor | None], right: dict[str, torch.Tensor | None]
) -> None:
    for key in ("input_embeds", "input_ids", "future_text_rows"):
        left_value = left[key]
        right_value = right[key]
        assert left_value is not None
        assert right_value is not None
        assert torch.equal(left_value, right_value), key


def test_mixed_prompt_has_text_and_multiple_multimodal_positions() -> None:
    builder = _fake_prefill_builder()
    _, thinker_input_ids, _, multimodal_mask = _mixed_prompt_inputs(builder)

    user_ids = thinker_input_ids[2:7]
    user_mask = multimodal_mask[2:7]
    assert user_ids[~user_mask].tolist() == [10, 11, 12]
    assert user_ids[user_mask].tolist() == [AUDIO_TOKEN, IMAGE_TOKEN]


def test_text_prompt_hidden_rows_do_not_affect_talker_prefill() -> None:
    """Poisoning every non-multimodal hidden row leaves Talker input exact."""
    builder = _fake_prefill_builder()
    thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask = (
        _mixed_prompt_inputs(builder)
    )
    poisoned_hidden = thinker_hidden.clone()
    poisoned_hidden[~multimodal_mask] = torch.full_like(
        poisoned_hidden[~multimodal_mask], -1234.0
    )
    assert torch.equal(
        poisoned_hidden[multimodal_mask], thinker_hidden[multimodal_mask]
    )

    normal = _build_mixed_prefill(
        builder,
        thinker_embed,
        thinker_input_ids,
        thinker_hidden,
        multimodal_mask,
    )
    poisoned = _build_mixed_prefill(
        builder,
        thinker_embed,
        thinker_input_ids,
        poisoned_hidden,
        multimodal_mask,
    )

    _assert_direct_prefill_equal(normal, poisoned)


def test_multimodal_prompt_hidden_rows_change_corresponding_talker_rows() -> None:
    """Changing multimodal prompt hidden rows changes only their projections."""
    builder = _fake_prefill_builder()
    thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask = (
        _mixed_prompt_inputs(builder)
    )
    changed_hidden = thinker_hidden.clone()
    changed_hidden[multimodal_mask] += torch.tensor(
        [[100.0] * HIDDEN, [200.0] * HIDDEN]
    )

    normal = _build_mixed_prefill(
        builder,
        thinker_embed,
        thinker_input_ids,
        thinker_hidden,
        multimodal_mask,
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
    multimodal_rows = _user_multimodal_output_rows(
        thinker_input_ids, multimodal_mask
    )
    assert changed_rows.nonzero(as_tuple=True)[0].tolist() == multimodal_rows


def test_sparse_non_multimodal_hidden_is_equivalent_through_prefill() -> None:
    """Keeping only multimodal prompt hidden rows preserves all Talker outputs."""
    builder = _fake_prefill_builder()
    thinker_embed, thinker_input_ids, thinker_hidden, multimodal_mask = (
        _mixed_prompt_inputs(builder)
    )
    sparse_hidden = torch.zeros_like(thinker_hidden)
    sparse_hidden[multimodal_mask] = thinker_hidden[multimodal_mask]

    normal = _build_mixed_prefill(
        builder,
        thinker_embed,
        thinker_input_ids,
        thinker_hidden,
        multimodal_mask,
    )
    sparse = _build_mixed_prefill(
        builder,
        thinker_embed,
        thinker_input_ids,
        sparse_hidden,
        multimodal_mask,
    )

    _assert_direct_prefill_equal(normal, sparse)


@pytest.mark.parametrize("assistant_count", [3, 4, 5])
@pytest.mark.parametrize("thinker_done", [False, True])
def test_initial_prefill_ignores_generated_assistant_hidden_values(
    assistant_count: int, thinker_done: bool
) -> None:
    """Shape-valid mutations do not change the assistant-side prefill result."""
    token_ids = list(range(20, 20 + assistant_count))
    normal = _chunks(token_ids)
    builder = _fake_prefill_builder()

    left = _prefill_payload(builder, normal, thinker_done=thinker_done)
    for mutated in (
        _chunks(token_ids, poison_data=True),
        _chunks(token_ids, poison_layer=True),
        _chunks(token_ids, poison_data=True, poison_layer=True),
    ):
        right = _prefill_payload(builder, mutated, thinker_done=thinker_done)
        _assert_prefill_equal(left, right)


def test_initial_prefill_strips_im_end_and_controls_tts_eos() -> None:
    """The EOS boundary remains independent of the removed hidden payload."""
    chunks = _chunks([20, 21, 22, IM_END])
    builder = _fake_prefill_builder()

    partial = _prefill_payload(builder, chunks, thinker_done=False)
    done = _prefill_payload(builder, chunks, thinker_done=True)

    assert torch.equal(partial["input_embeds"], done["input_embeds"])
    assert torch.equal(partial["input_ids"], done["input_ids"])
    assert len(done["pending_text_queue"]) == len(partial["pending_text_queue"]) + 1
    assert torch.equal(list(done["pending_text_queue"])[-1], done["tts_eos_embed"])


def test_incremental_token_id_dominates_data_and_layer_hidden() -> None:
    """Steady-state projection reconstructs the row from the token ID."""
    builder = _fake_prefill_builder()
    normal = _chunks([7])[0]
    expected = builder.project_assistant_chunk(normal)
    for mutated in (
        _chunks([7], poison_data=True)[0],
        _chunks([7], poison_layer=True)[0],
        _chunks([7], poison_data=True, poison_layer=True)[0],
    ):
        assert torch.equal(expected, builder.project_assistant_chunk(mutated))


def test_incremental_missing_token_id_keeps_legacy_embedding_fallback() -> None:
    """A legacy chunk without metadata still uses its full embedding row."""
    builder = _fake_prefill_builder()
    data = torch.arange(HIDDEN, dtype=torch.float32)
    chunk = SimpleNamespace(data=data, metadata={})

    assert torch.equal(builder.project_assistant_chunk(chunk), data + 0.5)


def test_initial_prefill_missing_token_id_fails_closed() -> None:
    """Initial prefill must not invent a token ID from an embedding payload."""
    builder = _fake_prefill_builder()
    chunk = SimpleNamespace(data=torch.zeros(HIDDEN), metadata={})

    with pytest.raises(KeyError, match="token_id"):
        builder.extract_chunk_token_ids([chunk])


def test_eos_chunk_is_dropped_and_done_appends_tts_eos() -> None:
    builder = _fake_prefill_builder()
    req_data = SimpleNamespace(
        thinker_chunks_done=False,
        pending_text_queue=deque(),
        tts_eos_embed=torch.full((HIDDEN,), 20.0),
    )

    builder.append_text_chunk(req_data, _chunks([IM_END])[0])
    assert len(req_data.pending_text_queue) == 0

    builder.append_text_chunk(req_data, _chunks([7])[0])
    assert len(req_data.pending_text_queue) == 1
    builder.mark_thinker_done(req_data)
    assert req_data.thinker_chunks_done is True
    assert len(req_data.pending_text_queue) == 2
    assert torch.equal(list(req_data.pending_text_queue)[-1], req_data.tts_eos_embed)

    builder.mark_thinker_done(req_data)
    assert len(req_data.pending_text_queue) == 2


def test_prompt_multimodal_hidden_remains_semantically_live() -> None:
    """Prompt multimodal rows still use hidden_projection, unlike text rows."""
    thinker_embed = torch.zeros((2, HIDDEN), dtype=torch.float32)
    hidden_a = torch.ones((2, HIDDEN), dtype=torch.float32)
    hidden_b = hidden_a.clone()
    hidden_b[0] = 9.0
    multimodal_mask = torch.tensor([True, False])

    left = build_user_part(
        thinker_embed=thinker_embed,
        thinker_hidden=hidden_a,
        multimodal_mask=multimodal_mask,
        text_projection=lambda rows: rows + 1.0,
        hidden_projection=lambda rows: rows * 2.0,
    )
    right = build_user_part(
        thinker_embed=thinker_embed,
        thinker_hidden=hidden_b,
        multimodal_mask=multimodal_mask,
        text_projection=lambda rows: rows + 1.0,
        hidden_projection=lambda rows: rows * 2.0,
    )

    assert not torch.equal(left[0], right[0])
    assert torch.equal(left[1], right[1])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Current build_prompt_prefill still stacks assistant hidden rows; "
        "the Track 2.2 candidate should accept token-only prefetched chunks."
    ),
)
def test_token_only_prefetched_chunks_are_accepted_by_initial_prefill() -> None:
    """Desired future contract: token IDs are sufficient for assistant prefill."""
    builder = _fake_prefill_builder()
    result = _prefill_payload(
        builder,
        _chunks([20, 21, 22, 23, 24], token_only=True),
        thinker_done=False,
    )

    assert result["input_embeds"].ndim == 2
    assert result["input_ids"].ndim == 1

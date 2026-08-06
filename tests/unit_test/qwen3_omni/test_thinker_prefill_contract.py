# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import xxhash
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer


class _ExtendMode:
    def is_decode(self) -> bool:
        return False

    def is_target_verify(self) -> bool:
        return False


class _FakeThinkerTextModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.seen_input_ids = None
        self.seen_input_embeds = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        *,
        input_ids,
        positions,
        forward_batch,
        input_embeds,
        pp_proxy_tensors=None,
        input_deepstack_embeds=None,
    ):
        del positions, forward_batch, pp_proxy_tensors, input_deepstack_embeds
        self.seen_input_ids = input_ids
        self.seen_input_embeds = input_embeds
        return input_embeds


def _outer_model() -> tuple[Qwen3OmniThinkerForCausalLM, _FakeThinkerTextModel]:
    outer = Qwen3OmniThinkerForCausalLM.__new__(Qwen3OmniThinkerForCausalLM)
    nn.Module.__init__(outer)
    text_model = _FakeThinkerTextModel()
    outer.model = text_model
    outer.lm_head = nn.Identity()
    outer.logits_processor = lambda input_ids, hidden, lm_head, forward_batch: hidden
    return outer, text_model


def _audio_item(
    embeddings: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=123,
        pad_value=999,
        offsets=offsets,
        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
        precomputed_embeddings=embeddings,
    )


def _forward_batch(
    input_ids: torch.Tensor,
    mm_inputs: list[MultimodalInputs | None],
    *,
    prefix_lens: list[int],
    extend_lens: list[int],
    stable_input_embeds: torch.Tensor | None = None,
):
    return SimpleNamespace(
        input_ids=input_ids,
        input_embeds=stable_input_embeds,
        mm_inputs=mm_inputs,
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens_cpu=extend_lens,
        forward_mode=_ExtendMode(),
        mrope_positions=None,
    )


def _qwen_audio_state(input_ids: torch.Tensor, audio_embeds: torch.Tensor):
    return Qwen3OmniPipelineState(
        prompt={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "prompt_text": "audio",
        },
        thinker_inputs={
            "model_inputs": {"audio_embeds": audio_embeds},
            "media_cache_keys": {"audio": "audio:cache"},
        },
    )


def _thinker_config(audio_token_id: int = 77):
    return SimpleNamespace(
        image_token_id=55,
        video_token_id=66,
        audio_token_id=audio_token_id,
    )


def _patch_sampling_and_mrope(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_text_audio_request_uses_standard_item_and_keeps_live_input_embeds_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sampling_and_mrope(monkeypatch)
    audio_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    original_ids = torch.tensor([10, 77, 11, 77, 12], dtype=torch.long)
    state = _qwen_audio_state(original_ids, audio_embeds)
    payload = StagePayload(
        request_id="audio-text",
        request=OmniRequest(inputs=[], metadata={"output_modalities": ["text"]}),
        data=state.to_dict(),
    )

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )
    request_data = request_builder(payload)

    req = request_data.req
    item = req.multimodal_inputs.mm_items[0]
    expected_hash = xxhash.xxh3_64(b"audio:cache").intdigest()
    expected_pad = 256 + expected_hash % (1 << 62)

    assert req.input_embeds is None
    assert req.omni_model_inputs is None
    assert item.format is MultimodalInputFormat.PRECOMPUTED_EMBEDDING
    assert item.precomputed_embeddings is audio_embeds
    assert item.hash == expected_hash
    assert item.pad_value == expected_pad
    assert item.offsets == [(1, 1), (3, 3)]
    assert request_data.model_inputs.get("audio_embeds") is None
    assert "audio_embeds" not in payload.data["thinker_inputs"]["model_inputs"]
    assert request_data.input_ids.tolist()[1] == expected_pad
    assert request_data.input_ids.tolist()[3] == expected_pad


def test_speech_audio_request_retains_legacy_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sampling_and_mrope(monkeypatch)
    audio_embeds = torch.ones((2, 4))
    state = _qwen_audio_state(torch.tensor([10, 77, 11, 77]), audio_embeds)
    payload = StagePayload(
        request_id="audio-speech",
        request=OmniRequest(inputs=[], metadata={"output_modalities": ["audio"]}),
        data=state.to_dict(),
    )
    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )

    request_data = request_builder(payload)

    assert request_data.req.omni_model_inputs is not None
    assert request_data.req.multimodal_inputs.mm_items == []
    assert "audio_embeds" in payload.data["thinker_inputs"]["model_inputs"]


def test_mixed_visual_audio_request_retains_legacy_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sampling_and_mrope(monkeypatch)
    audio_embeds = torch.ones((2, 4))
    state = _qwen_audio_state(torch.tensor([10, 77, 11, 77]), audio_embeds)
    state.thinker_inputs["model_inputs"]["image_embeds"] = torch.ones((1, 4))
    payload = StagePayload(
        request_id="mixed-visual-audio",
        request=OmniRequest(inputs=[], metadata={"output_modalities": ["text"]}),
        data=state.to_dict(),
    )
    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        thinker_config=_thinker_config(),
    )

    request_data = request_builder(payload)

    assert request_data.req.omni_model_inputs is not None
    assert request_data.req.multimodal_inputs.mm_items == []
    assert "audio_embeds" in payload.data["thinker_inputs"]["model_inputs"]


def test_existing_custom_prefill_hook_is_noop_for_standard_request() -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner._inject_multimodal_embeds = lambda *_args: pytest.fail(
        "standard request must not be intercepted"
    )
    forward_batch = SimpleNamespace(input_embeds=None)
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        reqs=[SimpleNamespace(omni_model_inputs=None)],
    )

    assert runner.custom_prefill_forward(forward_batch, schedule_batch, []) is None
    assert forward_batch.input_embeds is None


def test_qwen_outer_forward_composes_audio_in_eager_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer, text_model = _outer_model()
    source = torch.tensor(
        [[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]]
    )
    input_ids = torch.tensor([1, 999, 2, 999], dtype=torch.long)
    batch = _forward_batch(
        input_ids,
        [MultimodalInputs(mm_items=[_audio_item(source, [(1, 1), (3, 3)])])],
        prefix_lens=[0],
        extend_lens=[4],
    )

    cat_calls = []
    original_cat = torch.cat

    def spy_cat(*args, **kwargs):
        cat_calls.append((args, kwargs))
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", spy_cat)
    result = outer.forward(input_ids, torch.arange(4), batch)

    assert text_model.seen_input_ids is None
    assert text_model.seen_input_embeds is not None
    assert batch.input_embeds is None
    assert torch.equal(text_model.seen_input_embeds[[1, 3]], source)
    assert result is text_model.seen_input_embeds
    assert cat_calls == []


def test_qwen_outer_forward_text_only_uses_normal_embedding_path() -> None:
    outer, text_model = _outer_model()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    batch = _forward_batch(
        input_ids,
        [None],
        prefix_lens=[0],
        extend_lens=[3],
    )

    result = outer.forward(input_ids, torch.arange(3), batch)

    assert result is text_model.seen_input_embeds
    assert text_model.seen_input_ids is None
    assert batch.input_embeds is None


def test_qwen_outer_forward_fills_upstream_stable_buffer() -> None:
    outer, text_model = _outer_model()
    source = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
    input_ids = torch.tensor([1, 999, 2], dtype=torch.long)
    stable = torch.zeros((3, 4))
    batch = _forward_batch(
        input_ids,
        [MultimodalInputs(mm_items=[_audio_item(source, [(1, 1)])])],
        prefix_lens=[0],
        extend_lens=[3],
        stable_input_embeds=stable,
    )

    result = outer.forward(input_ids, torch.arange(3), batch)

    assert result is stable
    assert text_model.seen_input_embeds is stable
    assert torch.equal(stable[1], source[0])


def test_qwen_outer_forward_preserves_legacy_base_for_mixed_batch() -> None:
    outer, text_model = _outer_model()
    source = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
    input_ids = torch.tensor([1, 999, 2], dtype=torch.long)
    legacy_base = torch.full((3, 4), -7.0)
    batch = _forward_batch(
        input_ids,
        [MultimodalInputs(mm_items=[_audio_item(source, [(1, 1)])])],
        prefix_lens=[0],
        extend_lens=[3],
        stable_input_embeds=legacy_base,
    )

    result = outer.forward(
        input_ids,
        torch.arange(3),
        batch,
        input_embeds=legacy_base,
    )

    assert result is legacy_base
    assert torch.equal(legacy_base[0], torch.full((4,), -7.0))
    assert torch.equal(legacy_base[1], source[0])
    assert torch.equal(legacy_base[2], torch.full((4,), -7.0))
    assert text_model.seen_input_ids is None


def test_qwen_outer_forward_places_only_current_chunk_for_mixed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer, text_model = _outer_model()
    source_a = torch.tensor(
        [
            [10.0, 10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0, 12.0],
        ]
    )
    source_b = torch.tensor(
        [
            [20.0, 20.0, 20.0, 20.0],
            [21.0, 21.0, 21.0, 21.0],
            [22.0, 22.0, 22.0, 22.0],
            [23.0, 23.0, 23.0, 23.0],
        ]
    )
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    batch = _forward_batch(
        input_ids,
        [
            MultimodalInputs(mm_items=[_audio_item(source_a, [(1, 3)])]),
            MultimodalInputs(mm_items=[_audio_item(source_b, [(2, 5)])]),
        ],
        prefix_lens=[1, 4],
        extend_lens=[3, 2],
    )

    index_copy_calls = []
    cat_calls = []
    original_index_copy = torch.Tensor.index_copy_
    original_cat = torch.cat

    def spy_index_copy(self, dim, index, source):
        index_copy_calls.append((dim, index.clone(), source.clone()))
        return original_index_copy(self, dim, index, source)

    def spy_cat(*args, **kwargs):
        cat_calls.append((args, kwargs))
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_copy_", spy_index_copy)
    monkeypatch.setattr(torch, "cat", spy_cat)
    outer.forward(input_ids, torch.arange(5), batch)

    assert len(index_copy_calls) == 1
    assert len(cat_calls) == 1
    assert torch.equal(text_model.seen_input_embeds[0:3], source_a)
    assert torch.equal(text_model.seen_input_embeds[3:5], source_b[2:4])

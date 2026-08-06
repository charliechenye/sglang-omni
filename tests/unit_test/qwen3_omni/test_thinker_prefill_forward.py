# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
    _config_has_mrope,
    _qwen_mrope_enabled,
)


class _Mode:
    def __init__(self, *, decode: bool = False, target_verify: bool = False):
        self._decode = decode
        self._target_verify = target_verify

    def is_decode(self) -> bool:
        return self._decode

    def is_target_verify(self) -> bool:
        return self._target_verify


class _EmbedTokens(nn.Module):
    num_embeddings = 128

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)


class _InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _EmbedTokens()
        self.pp_group = SimpleNamespace(is_first_rank=True)
        self.calls: list[dict[str, object]] = []

    def get_input_embeddings(self) -> _EmbedTokens:
        return self.embed_tokens

    def forward(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        input_embeds = kwargs["input_embeds"]
        if input_embeds is not None:
            return input_embeds
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        return self.embed_tokens(input_ids)


class _AudioItem:
    modality = SimpleNamespace(name="AUDIO")

    def __init__(self, positions: list[int], embeds: torch.Tensor) -> None:
        self.precomputed_embeddings = embeds
        self.model_specific_data = {
            "positions_cpu": torch.tensor(positions, dtype=torch.long)
        }

    def is_precomputed_embedding(self) -> bool:
        return True


def _outer() -> tuple[Qwen3OmniThinkerForCausalLM, _InnerModel]:
    outer = Qwen3OmniThinkerForCausalLM.__new__(Qwen3OmniThinkerForCausalLM)
    nn.Module.__init__(outer)
    inner = _InnerModel()
    outer.model = inner
    outer.lm_head = object()
    outer.logits_processor = lambda input_ids, hidden, lm_head, forward_batch: (
        input_ids,
        hidden,
        lm_head,
        forward_batch,
    )
    outer.is_mrope_enabled = True
    return outer, inner


def _forward_batch(
    input_ids: torch.Tensor,
    *,
    mm_inputs=None,
    extend_lens: list[int] | None = None,
    prefix_lens: list[int] | None = None,
    mode: _Mode | None = None,
    input_embeds: torch.Tensor | None = None,
):
    return SimpleNamespace(
        input_ids=input_ids,
        mm_inputs=mm_inputs,
        input_embeds=input_embeds,
        extend_seq_lens_cpu=extend_lens or [int(input_ids.numel())],
        extend_prefix_lens_cpu=prefix_lens or [0],
        forward_mode=mode or _Mode(),
        mrope_positions=None,
    )


def test_text_only_eager_preserves_inner_input_ids() -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 4, 5], dtype=torch.long)
    batch = _forward_batch(input_ids)

    outer(input_ids, torch.arange(3), batch)

    call = inner.calls[-1]
    assert call["input_ids"] is input_ids
    assert call["input_embeds"] is None


def test_text_only_graph_populates_the_model_argument_buffer() -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 4, 5], dtype=torch.long)
    stable = torch.full((3, 4), -12345.0)
    batch = _forward_batch(input_ids, input_embeds=stable)

    outer(input_ids, torch.arange(3), batch, input_embeds=stable)

    expected = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    assert torch.equal(stable, expected)
    call = inner.calls[-1]
    assert call["input_ids"] is None
    assert call["input_embeds"] is stable


def test_audio_eager_composes_text_and_precomputed_rows() -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 99, 5, 99, 7], dtype=torch.long)
    audio = torch.tensor([[100.0] * 4, [200.0] * 4])
    item = _AudioItem([1, 3], audio)
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[item])],
    )

    outer(input_ids, torch.arange(5), batch)

    expected = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    expected[[1, 3]] = audio
    assert torch.equal(inner.calls[-1]["input_embeds"], expected)
    assert inner.calls[-1]["input_ids"] is None


def test_audio_graph_overwrites_sentinel_in_stable_buffer() -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 99, 5, 99, 7], dtype=torch.long)
    audio = torch.tensor([[100.0] * 4, [200.0] * 4])
    item = _AudioItem([1, 3], audio)
    stable = torch.full((5, 4), -12345.0)
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[item])],
        input_embeds=stable,
    )

    outer(input_ids, torch.arange(5), batch, input_embeds=stable)

    expected = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    expected[[1, 3]] = audio
    assert torch.equal(stable, expected)
    assert inner.calls[-1]["input_embeds"] is stable
    assert inner.calls[-1]["input_ids"] is None


def test_audio_placement_uses_absolute_positions_for_chunked_prefill() -> None:
    outer, inner = _outer()
    # The item describes the full prompt. This batch contains absolute rows 2..4.
    input_ids = torch.tensor([20, 21, 99], dtype=torch.long)
    item = _AudioItem([1, 4], torch.tensor([[100.0] * 4, [200.0] * 4]))
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[item])],
        extend_lens=[3],
        prefix_lens=[2],
    )

    outer(input_ids, torch.arange(3), batch)

    expected = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    expected[2] = torch.tensor([200.0] * 4)
    assert torch.equal(inner.calls[-1]["input_embeds"], expected)


def test_audio_placement_preserves_flat_batch_offsets() -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 4, 5, 99, 7], dtype=torch.long)
    item = _AudioItem([1], torch.tensor([[200.0] * 4]))
    batch = _forward_batch(
        input_ids,
        mm_inputs=[None, SimpleNamespace(mm_items=[item])],
        extend_lens=[2, 3],
        prefix_lens=[0, 0],
    )

    outer(input_ids, torch.arange(5), batch)

    expected = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    expected[3] = torch.tensor([200.0] * 4)
    assert torch.equal(inner.calls[-1]["input_embeds"], expected)


def test_audio_placement_coalesces_multiple_items_into_one_index_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 99, 5, 98, 7], dtype=torch.long)
    first = _AudioItem([1], torch.tensor([[100.0] * 4]))
    second = _AudioItem([3], torch.tensor([[200.0] * 4]))
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[first, second])],
    )

    cat_calls: list[int] = []
    index_copy_calls: list[int] = []
    original_cat = torch.cat
    original_index_copy = torch.Tensor.index_copy_

    def counted_cat(tensors, *args, **kwargs):
        cat_calls.append(len(tensors))
        return original_cat(tensors, *args, **kwargs)

    def counted_index_copy(self, *args, **kwargs):
        index_copy_calls.append(1)
        return original_index_copy(self, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", counted_cat)
    monkeypatch.setattr(torch.Tensor, "index_copy_", counted_index_copy)

    outer(input_ids, torch.arange(5), batch)

    expected = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    expected[1] = first.precomputed_embeddings[0]
    expected[3] = second.precomputed_embeddings[0]
    assert torch.equal(inner.calls[-1]["input_embeds"], expected)
    assert len(index_copy_calls) == 1
    # One source cat and one destination cat; no per-item placement calls.
    assert len(cat_calls) == 2


@pytest.mark.parametrize("audio_tokens", [256, 2912, 8192])
def test_audio_placement_scaling_checkpoints(audio_tokens: int) -> None:
    outer, inner = _outer()
    input_ids = torch.full((audio_tokens,), 99, dtype=torch.long)
    audio = torch.arange(audio_tokens * 4, dtype=torch.float32).reshape(
        audio_tokens, 4
    )
    item = _AudioItem(list(range(audio_tokens)), audio)
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[item])],
    )

    outer(input_ids, torch.arange(audio_tokens), batch)

    assert torch.equal(inner.calls[-1]["input_embeds"], audio)


def test_audio_width_mismatch_is_a_semantic_contract_error() -> None:
    outer, _ = _outer()
    input_ids = torch.tensor([3, 99], dtype=torch.long)
    item = _AudioItem([1], torch.ones((1, 3)))
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[item])],
    )

    with pytest.raises(ValueError, match="width"):
        outer(input_ids, torch.arange(2), batch)


def test_audio_positions_must_be_ordered_source_metadata() -> None:
    outer, _ = _outer()
    input_ids = torch.tensor([3, 99, 98], dtype=torch.long)
    item = _AudioItem([2, 1], torch.ones((2, 4)))
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[item])],
    )

    with pytest.raises(RuntimeError, match="sorted"):
        outer(input_ids, torch.arange(3), batch)


def test_legacy_multimodal_shell_preserves_external_embeddings() -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 4], dtype=torch.long)
    external = torch.full((2, 4), 7.0)
    batch = _forward_batch(
        input_ids,
        mm_inputs=[SimpleNamespace(mm_items=[])],
    )

    outer(input_ids, torch.arange(2), batch, input_embeds=external)

    call = inner.calls[-1]
    assert call["input_ids"] is input_ids
    assert call["input_embeds"] is external
    assert torch.equal(external, torch.full((2, 4), 7.0))


def test_mrope_declaration_accepts_both_upstream_config_shapes() -> None:
    assert _config_has_mrope(
        SimpleNamespace(rope_parameters={"mrope_section": [16, 24, 24]})
    )
    assert _config_has_mrope(
        SimpleNamespace(rope_scaling={"mrope_section": [16, 24, 24]})
    )
    assert not _config_has_mrope(SimpleNamespace(rope_scaling={"factor": 2.0}))


def test_outer_mrope_contract_checks_root_and_text_configs() -> None:
    root = SimpleNamespace(rope_parameters={"mrope_section": [16, 24, 24]})
    text = SimpleNamespace(rope_scaling=None)

    assert _qwen_mrope_enabled(root, text)


def test_outer_forward_uses_mrope_positions_for_the_inner_model() -> None:
    outer, inner = _outer()
    input_ids = torch.tensor([3, 4], dtype=torch.long)
    mrope_positions = torch.arange(6, dtype=torch.long).reshape(3, 2)
    batch = _forward_batch(input_ids)
    batch.mrope_positions = mrope_positions

    outer(input_ids, torch.arange(2), batch)

    assert inner.calls[-1]["positions"] is mrope_positions


def test_upstream_prefill_runner_uses_outer_mrope_contract() -> None:
    runner_module = pytest.importorskip(
        "sglang.srt.model_executor.cuda_graph_runner"
    )
    runner_type = getattr(runner_module, "PrefillCudaGraphRunner", None)
    if runner_type is None:
        pytest.skip("pinned SGLang does not expose PrefillCudaGraphRunner here")

    outer, _ = _outer()
    mrope = torch.arange(12, dtype=torch.long).reshape(3, 4)
    positions = torch.arange(4, dtype=torch.long)
    forward_batch = SimpleNamespace(mrope_positions=mrope, positions=positions)
    runner = runner_type.__new__(runner_type)
    runner.model_runner = SimpleNamespace(model=outer)

    selected = runner._get_layer_model_positions(forward_batch)

    assert selected is mrope

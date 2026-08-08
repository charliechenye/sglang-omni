# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni outer-model and pinned prefill graph contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytest.importorskip("transformers")

from sglang_omni.models.qwen3_omni.components import sglang_thinker  # noqa: E402
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (  # noqa: E402
    Qwen3OmniThinkerForCausalLM,
)
from sglang_omni.models.qwen3_omni.hf_config import (  # noqa: E402
    Qwen3OmniMoeTextConfig,
)


class _FakeInnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)


def _root_config(text_config) -> SimpleNamespace:
    return SimpleNamespace(thinker_config=SimpleNamespace(text_config=text_config))


def _patch_model_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        sglang_thinker,
        "Qwen3MoeLLMModel",
        lambda **kwargs: _FakeInnerModel(),
    )
    monkeypatch.setattr(
        sglang_thinker,
        "ParallelLMHead",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        sglang_thinker,
        "LogitsProcessor",
        lambda config: torch.nn.Identity(),
    )


def test_real_qwen_text_config_sets_mrope_capability(monkeypatch) -> None:
    _patch_model_dependencies(monkeypatch)
    config = Qwen3OmniMoeTextConfig(
        vocab_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
        },
    )

    outer = Qwen3OmniThinkerForCausalLM(_root_config(config))

    assert outer.is_mrope_enabled is True


def test_real_qwen_text_config_without_mrope_is_not_marked(monkeypatch) -> None:
    _patch_model_dependencies(monkeypatch)
    config = Qwen3OmniMoeTextConfig(
        vocab_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
    )

    outer = Qwen3OmniThinkerForCausalLM(_root_config(config))

    assert outer.is_mrope_enabled is False
    assert (
        sglang_thinker._config_uses_mrope(
            SimpleNamespace(rope_parameters={"rope_type": "default"})
        )
        is False
    )


def test_outer_forward_accepts_generic_sidecar_request_ids() -> None:
    outer = Qwen3OmniThinkerForCausalLM.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(outer)
    seen: dict[str, object] = {}

    class _TextModel(torch.nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=None,
            pp_proxy_tensors=None,
            input_deepstack_embeds=None,
        ):
            del positions, forward_batch, pp_proxy_tensors, input_deepstack_embeds
            seen["input_ids"] = input_ids
            seen["input_embeds"] = input_embeds
            return input_embeds

    outer.model = _TextModel()
    outer.lm_head = torch.nn.Identity()
    outer.logits_processor = lambda input_ids, hidden_states, lm_head, forward_batch: (
        input_ids,
        hidden_states,
        lm_head,
        forward_batch,
    )
    forward_batch = SimpleNamespace(mrope_positions=None)
    input_ids = torch.tensor([1, 2])
    input_embeds = torch.ones(2, 4)

    output = outer(
        input_ids,
        torch.tensor([0, 1]),
        forward_batch,
        input_embeds=input_embeds,
        omni_prefill_rids=("request-a",),
    )

    assert seen["input_ids"] is input_ids
    assert seen["input_embeds"] is input_embeds
    assert output[1] is input_embeds


@pytest.mark.parametrize("mrope_enabled", [True, False])
def test_pinned_prefill_runner_selects_positions_from_outer_capability(
    mrope_enabled: bool,
) -> None:
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    outer = SimpleNamespace(is_mrope_enabled=mrope_enabled)
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(model=outer)
    mrope_positions = torch.tensor([[10, 11], [10, 11], [10, 11]])
    positions = torch.tensor([0, 1])
    forward_batch = SimpleNamespace(
        mrope_positions=mrope_positions,
        positions=positions,
    )

    selected = runner._get_layer_model_positions(forward_batch)

    assert selected is (mrope_positions if mrope_enabled else positions)

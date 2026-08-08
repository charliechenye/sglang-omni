# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Qwen3-Omni's breakable-prefill adopter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

nn = torch.nn

from sglang_omni.model_runner.prefill_inputs import (  # noqa: E402
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.models.qwen3_omni.components import (  # noqa: E402
    sglang_thinker,
)
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (  # noqa: E402
    Qwen3OmniThinkerForCausalLM,
)
from sglang_omni.models.qwen3_omni.thinker_model_runner import (  # noqa: E402
    Qwen3OmniThinkerModelRunner,
)


def _runner() -> Qwen3OmniThinkerModelRunner:
    runner = object.__new__(Qwen3OmniThinkerModelRunner)
    runner._image_token_id = 100
    runner._video_token_id = 101
    runner._audio_token_id = 99
    runner._embed_tokens = nn.Embedding(256, 4)
    with torch.no_grad():
        runner._embed_tokens.weight.copy_(
            torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
        )
    return runner


def _forward_batch(input_ids: torch.Tensor, *, batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=input_ids,
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=[object() for _ in range(batch_size)],
        batch_size=batch_size,
        extend_seq_lens_cpu=[],
    )


def _audio_req(
    input_ids: list[int],
    audio_embeds: torch.Tensor,
    *,
    audio_token_id: int = 99,
) -> SimpleNamespace:
    return SimpleNamespace(
        origin_input_ids=input_ids,
        extend_range=SimpleNamespace(start=0, length=len(input_ids)),
        omni_model_inputs={
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": torch.tensor([audio_embeds.shape[0]]),
        },
        _omni_consumed=None,
        inflight_middle_chunks=0,
        audio_token_id=audio_token_id,
    )


def test_pure_text_prefill_skips_sidecar_and_forward_input_embeds() -> None:
    runner = _runner()
    forward_batch = _forward_batch(torch.tensor([1, 2]), batch_size=1)
    schedule_batch = SimpleNamespace(
        reqs=[SimpleNamespace(omni_model_inputs={})],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )

    runner.before_prefill(forward_batch, schedule_batch, [object()])
    result = runner.custom_prefill_forward(
        forward_batch, schedule_batch, [object()]
    )

    assert result is None
    assert get_omni_prefill_inputs(forward_batch) is None
    assert forward_batch.input_embeds is None


def test_audio_prefill_uses_sidecar_without_mutating_upstream_fields() -> None:
    runner = _runner()
    audio_embeds = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
    req = _audio_req([1, 99, 2], audio_embeds)
    req.multimodal_inputs = SimpleNamespace(mrope_position_delta=7)
    forward_batch = _forward_batch(torch.tensor([1, 99, 2]), batch_size=1)
    forward_batch.extend_seq_lens_cpu = [3]
    forward_batch.mrope_positions = torch.tensor([[0, 1, 2]] * 3)
    original_mm_inputs = forward_batch.mm_inputs
    schedule_batch = SimpleNamespace(reqs=[req])

    runner.before_prefill(forward_batch, schedule_batch, [object()])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    assert not hasattr(sidecar, "rids")
    expected = runner._embed_tokens(torch.tensor([1, 99, 2])).detach().clone()
    expected[1] = audio_embeds[0]
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs is original_mm_inputs
    assert forward_batch.mrope_positions.shape == (3, 3)
    assert req.multimodal_inputs.mrope_position_delta == 7


def test_mixed_text_audio_prefill_composes_both_row_kinds() -> None:
    runner = _runner()
    audio_embeds = torch.tensor([[200.0, 201.0, 202.0, 203.0]])
    text_req = SimpleNamespace(omni_model_inputs=None)
    audio_req = _audio_req([3, 99, 4], audio_embeds)
    input_ids = torch.tensor([1, 2, 3, 99, 4])
    forward_batch = _forward_batch(input_ids, batch_size=2)
    forward_batch.extend_seq_lens_cpu = [2, 3]
    schedule_batch = SimpleNamespace(reqs=[text_req, audio_req])

    runner.before_prefill(forward_batch, schedule_batch, [object(), object()])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    expected = runner._embed_tokens(input_ids).detach().clone()
    expected[3] = audio_embeds[0]
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None


@pytest.mark.parametrize(
    "model_inputs",
    [
        {"image_embeds": torch.ones(1, 4)},
        {"video_embeds": torch.ones(1, 4)},
        {"deepstack_visual_embeds": [torch.ones(1, 4)]},
        {"audio_embeds": torch.ones(1, 4), "use_audio_in_video": True},
        {"audio_embeds": torch.ones(4)},
        {"audio_embeds": torch.ones(1, 4), "unknown_aux": True},
    ],
)
def test_unsupported_or_malformed_audio_prefill_falls_back(
    model_inputs: dict[str, object],
) -> None:
    runner = _runner()
    req = SimpleNamespace(
        origin_input_ids=[1, 99, 2],
        extend_range=SimpleNamespace(start=0, length=3),
        omni_model_inputs=model_inputs,
        _omni_consumed=None,
        inflight_middle_chunks=0,
    )
    forward_batch = _forward_batch(torch.tensor([1, 99, 2]), batch_size=1)
    forward_batch.extend_seq_lens_cpu = [3]
    schedule_batch = SimpleNamespace(reqs=[req])

    runner.before_prefill(forward_batch, schedule_batch, [object()])

    assert get_omni_prefill_inputs(forward_batch) is None
    assert forward_batch.input_embeds is None


def test_existing_embedding_or_sidecar_fails_closed() -> None:
    runner = _runner()
    req = _audio_req([1, 99, 2], torch.ones(1, 4))
    schedule_batch = SimpleNamespace(reqs=[req])

    forward_batch = _forward_batch(torch.tensor([1, 99, 2]), batch_size=1)
    forward_batch.input_embeds = torch.zeros(3, 4)
    runner.before_prefill(forward_batch, schedule_batch, [object()])
    assert get_omni_prefill_inputs(forward_batch) is None
    assert forward_batch.input_embeds is not None

    forward_batch = _forward_batch(torch.tensor([1, 99, 2]), batch_size=1)
    payload = OmniPrefillInputs(input_embeds=torch.zeros(3, 4))
    attach_omni_prefill_inputs(forward_batch, payload)
    runner.before_prefill(forward_batch, schedule_batch, [object()])
    assert get_omni_prefill_inputs(forward_batch) is payload


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_input_ids = None
        self.seen_input_embeds = None

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
        self.seen_input_ids = input_ids
        self.seen_input_embeds = input_embeds
        return input_embeds if input_embeds is not None else input_ids


class _FakeLogitsProcessor(nn.Module):
    def forward(self, input_ids, hidden_states, lm_head, forward_batch):
        del input_ids, hidden_states, lm_head, forward_batch
        return "logits"


def test_outer_forward_accepts_generic_sidecar_request_ids() -> None:
    outer = Qwen3OmniThinkerForCausalLM.__new__(Qwen3OmniThinkerForCausalLM)
    nn.Module.__init__(outer)
    text_model = _FakeTextModel()
    outer.model = text_model
    outer.lm_head = nn.Identity()
    outer.logits_processor = _FakeLogitsProcessor()

    input_embeds = torch.ones(2, 4)
    output = outer(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        SimpleNamespace(mrope_positions=None),
        input_embeds=input_embeds,
        omni_prefill_rids=("request-a",),
    )

    assert output == "logits"
    assert text_model.seen_input_embeds is input_embeds


def test_qwen_outer_detects_mrope_from_current_config_shapes(monkeypatch) -> None:
    class _FakeInnerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)

    monkeypatch.setattr(
        sglang_thinker,
        "Qwen3MoeLLMModel",
        lambda **kwargs: _FakeInnerModel(),
    )
    monkeypatch.setattr(
        sglang_thinker,
        "ParallelLMHead",
        lambda *args, **kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        sglang_thinker,
        "LogitsProcessor",
        lambda config: nn.Identity(),
    )

    config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                vocab_size=8,
                hidden_size=4,
                rope_parameters={"mrope_section": [16, 24, 24]},
            )
        )
    )
    outer = Qwen3OmniThinkerForCausalLM(config)

    assert outer.is_mrope_enabled is True
    assert (
        sglang_thinker._config_uses_mrope(
            SimpleNamespace(rope_parameters={"rope_type": "default"})
        )
        is False
    )


def test_pinned_prefill_runner_uses_live_mrope_positions() -> None:
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    outer = SimpleNamespace(is_mrope_enabled=True)
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(model=outer)
    mrope_positions = torch.tensor([[10, 11], [10, 11], [10, 11]])
    positions = torch.tensor([0, 1])
    forward_batch = SimpleNamespace(
        mrope_positions=mrope_positions,
        positions=positions,
    )

    assert runner._get_layer_model_positions(forward_batch) is mrope_positions

    outer.is_mrope_enabled = False
    assert runner._get_layer_model_positions(forward_batch) is positions

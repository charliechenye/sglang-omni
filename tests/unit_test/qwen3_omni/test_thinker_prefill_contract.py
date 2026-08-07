# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for Qwen3-Omni's OmniPrefillInputs adoption contract."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.model_runner import thinker_model_runner as thinker_runner_module
from sglang_omni.model_runner.prefill_inputs import OmniPrefillInputs
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components import sglang_thinker
from sglang_omni.models.qwen3_omni.thinker_model_runner import (
    Qwen3OmniThinkerModelRunner,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _runner(*, capture_hidden: bool = False) -> Qwen3OmniThinkerModelRunner:
    runner = object.__new__(Qwen3OmniThinkerModelRunner)
    runner._should_capture_hidden = lambda request: capture_hidden
    runner._embed_tokens = nn.Embedding(128, 4)
    runner._image_token_id = 91
    runner._video_token_id = 92
    runner._audio_token_id = 93
    return runner


def _base_runner(*, capture_hidden: bool = False) -> ThinkerModelRunner:
    runner = object.__new__(ThinkerModelRunner)
    runner._should_capture_hidden = lambda request: capture_hidden
    runner._embed_tokens = nn.Embedding(128, 4)
    runner._image_token_id = 91
    runner._video_token_id = 92
    runner._audio_token_id = 93
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


def _forward_batch(
    *,
    mm_inputs,
    batch_size: int = 1,
    num_tokens: int = 3,
    input_ids: torch.Tensor | None = None,
):
    if input_ids is None:
        input_ids = torch.zeros(num_tokens, dtype=torch.long)
    return SimpleNamespace(
        input_embeds=None,
        mm_inputs=mm_inputs,
        input_ids=input_ids,
        batch_size=batch_size,
        extend_seq_lens_cpu=[num_tokens],
        extend_prefix_lens_cpu=[0],
        positions=torch.arange(num_tokens),
        mrope_positions=None,
    )


def _legacy_request_pair(model_inputs: dict, input_ids: list[int]):
    schedule_req = SimpleNamespace(
        origin_input_ids=list(input_ids),
        omni_model_inputs=model_inputs,
        _omni_consumed=None,
        _omni_mm_positions={
            "image": torch.tensor(
                [i for i, token in enumerate(input_ids) if token == 91],
                dtype=torch.long,
            ),
            "video": torch.tensor(
                [i for i, token in enumerate(input_ids) if token == 92],
                dtype=torch.long,
            ),
            "audio": torch.tensor(
                [i for i, token in enumerate(input_ids) if token == 93],
                dtype=torch.long,
            ),
        },
        inflight_middle_chunks=0,
        request_id="req-0",
    )
    request = SimpleNamespace(
        request_id="req-0",
        data=SimpleNamespace(
            stage_payload=SimpleNamespace(metadata={"output_modalities": ["text"]})
        ),
    )
    schedule_batch = SimpleNamespace(
        reqs=[schedule_req],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )
    forward_batch = _forward_batch(
        mm_inputs=None,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        num_tokens=len(input_ids),
    )
    return forward_batch, schedule_batch, [request], schedule_req


def _real_mrope_audio_prefill_state():
    from sglang.srt.managers.schedule_batch import MultimodalInputs

    audio_rows = torch.tensor([[10.0, 11.0, 12.0, 13.0]])
    model_inputs = {
        **_audio_inputs(),
        "audio_embeds": audio_rows,
    }
    forward_batch, schedule_batch, requests, schedule_req = _legacy_request_pair(
        model_inputs, [7, 93, 7]
    )

    shell = MultimodalInputs(mm_items=[])
    shell.mrope_positions = torch.tensor(
        [[100, 101, 102], [200, 201, 202], [300, 301, 302]],
        dtype=torch.long,
    )
    delta = torch.tensor([17], dtype=torch.long)
    shell.mrope_position_delta = delta
    schedule_req.multimodal_inputs = shell
    forward_batch.mm_inputs = [shell]
    current_prefill_positions = torch.tensor(
        [[400, 401, 402], [500, 501, 502], [600, 601, 602]],
        dtype=torch.long,
    )
    forward_batch.mrope_positions = current_prefill_positions
    return (
        forward_batch,
        schedule_batch,
        requests,
        schedule_req,
        shell,
        delta,
        audio_rows,
        current_prefill_positions,
    )


def test_text_only_batch_is_payload_compatible() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([None])

    assert runner._can_use_prefill_payload(schedule_batch, requests)


@pytest.mark.parametrize(
    "model_inputs",
    [
        {"audio_feature_lengths": torch.tensor([8])},
        {"audio_embeds": None},
        {"audio_embeds": [[0.0, 1.0]]},
    ],
    ids=["missing-audio-embeds", "none-audio-embeds", "non-tensor-audio-embeds"],
)
def test_incomplete_audio_payload_is_not_graph_compatible(model_inputs) -> None:
    runner = _runner()
    schedule_batch, requests = _requests([model_inputs])

    assert not runner._can_use_prefill_payload(schedule_batch, requests)


def test_multiple_audio_requests_and_text_audio_mix_are_compatible() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([_audio_inputs(), _audio_inputs(), None])

    assert runner._can_use_prefill_payload(schedule_batch, requests)


def test_speech_output_is_rejected_from_payload_path() -> None:
    runner = _runner()
    schedule_batch, requests = _requests(
        [_audio_inputs()], output_modalities=("audio",)
    )

    assert not runner._can_use_prefill_payload(schedule_batch, requests)


def test_hidden_capture_is_rejected_from_payload_path() -> None:
    runner = _runner(capture_hidden=True)
    schedule_batch, requests = _requests([_audio_inputs()])

    assert not runner._can_use_prefill_payload(schedule_batch, requests)


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
        assert not runner._can_use_prefill_payload(schedule_batch, requests)


def test_unknown_auxiliary_input_is_rejected() -> None:
    runner = _runner()
    schedule_batch, requests = _requests(
        [{**_audio_inputs(), "unqualified_aux": torch.zeros(1)}]
    )

    assert not runner._can_use_prefill_payload(schedule_batch, requests)


def test_incompatible_request_rejects_the_entire_batch() -> None:
    runner = _runner()
    schedule_batch, requests = _requests([None, {"image_embeds": torch.zeros(1, 4)}])

    assert not runner._can_use_prefill_payload(schedule_batch, requests)


@pytest.mark.parametrize(
    ("name", "input_ids", "model_inputs", "output_modalities", "expected"),
    [
        ("text", [7, 7], {}, ("text",), True),
        (
            "audio",
            [7, 93, 7],
            _audio_inputs(),
            ("text",),
            True,
        ),
        (
            "text-audio",
            [7, 93, 7, 93],
            _audio_inputs(),
            ("text",),
            True,
        ),
        (
            "image",
            [7, 91, 7],
            {"image_embeds": torch.zeros(1, 4)},
            ("text",),
            False,
        ),
        (
            "video",
            [7, 92, 7],
            {"video_embeds": torch.zeros(1, 4)},
            ("text",),
            False,
        ),
        (
            "image-audio",
            [7, 91, 93, 7],
            {**_audio_inputs(), "image_embeds": torch.zeros(1, 4)},
            ("text",),
            False,
        ),
        (
            "speech",
            [7, 93, 7],
            _audio_inputs(),
            ("audio",),
            False,
        ),
    ],
)
def test_classifier_matches_actual_qwen_request_builder_outputs(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    input_ids: list[int],
    model_inputs: dict,
    output_modalities: tuple[str, ...],
    expected: bool,
) -> None:
    """Exercise the classifier with the same Req/data objects the scheduler uses."""
    del name
    from sglang_omni.models.qwen3_omni.request_builders import (
        build_sglang_thinker_request,
    )
    from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer, make_qwen_state

    # Sampling normalization is unrelated to this contract and otherwise
    # requires a full tokenizer implementation.
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )

    state = make_qwen_state(
        prompt={
            "prompt_text": "test",
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        },
        thinker_inputs={"model_inputs": model_inputs},
    )
    req_data = build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 2, "temperature": 0.0},
        tokenizer=FakeQwenTokenizer(vocab_size=128),
        vocab_size=128,
        request_id="req-builder",
        thinker_config=None,
    )
    req_data.stage_payload = StagePayload(
        request_id="req-builder",
        request=OmniRequest(
            inputs=None,
            metadata={"output_modalities": list(output_modalities)},
        ),
        data=None,
    )
    schedule_batch = SimpleNamespace(reqs=[req_data.req])
    requests = [SimpleNamespace(request_id="req-builder", data=req_data)]

    assert _runner()._can_use_prefill_payload(schedule_batch, requests) is expected


def test_before_prefill_attaches_audio_payload_and_preserves_request_mrope_delta() -> (
    None
):
    runner = _runner()
    (
        forward_batch,
        schedule_batch,
        requests,
        schedule_req,
        shell,
        delta,
        audio_rows,
        current_prefill_positions,
    ) = _real_mrope_audio_prefill_state()

    assert shell.mrope_positions is not None
    assert shell.mrope_position_delta is delta
    assert "mrope_position_delta" not in vars(forward_batch)

    runner.before_prefill(forward_batch, schedule_batch, requests)

    payload = forward_batch.mm_inputs
    assert isinstance(payload, OmniPrefillInputs)
    assert payload.rids == ("req-0",)
    assert torch.equal(payload.input_embeds[1:2], audio_rows)
    assert forward_batch.input_embeds is None
    assert forward_batch.mrope_positions is current_prefill_positions
    assert schedule_req.multimodal_inputs is shell
    assert schedule_req.multimodal_inputs.mrope_position_delta is delta


def test_before_prefill_fails_closed_without_materialized_mrope_positions() -> None:
    runner = _runner()
    (
        forward_batch,
        schedule_batch,
        requests,
        schedule_req,
        shell,
        delta,
        _,
        _,
    ) = _real_mrope_audio_prefill_state()
    forward_batch.mrope_positions = None
    original_mm_inputs = forward_batch.mm_inputs
    original_model_inputs = schedule_req.omni_model_inputs
    original_consumed = schedule_req._omni_consumed
    original_positions = schedule_req._omni_mm_positions

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert forward_batch.mm_inputs is original_mm_inputs
    assert forward_batch.mm_inputs[0] is shell
    assert forward_batch.input_embeds is None
    assert schedule_req.omni_model_inputs is original_model_inputs
    assert schedule_req._omni_consumed is original_consumed
    assert schedule_req._omni_mm_positions is original_positions
    assert schedule_req.multimodal_inputs is shell
    assert schedule_req.multimodal_inputs.mrope_position_delta is delta


def test_before_prefill_does_not_consume_state_when_input_embeds_is_set() -> None:
    runner = _runner()
    input_ids = [7, 93, 7]
    model_inputs = _audio_inputs()
    forward_batch, schedule_batch, requests, schedule_req = _legacy_request_pair(
        model_inputs, input_ids
    )
    existing_input_embeds = torch.randn(3, 4)
    forward_batch.input_embeds = existing_input_embeds
    shell = SimpleNamespace(mm_items=[])
    forward_batch.mm_inputs = [shell]
    consumed = {"audio": 1}
    schedule_req._omni_consumed = consumed
    model_inputs_before = schedule_req.omni_model_inputs
    positions_before = schedule_req._omni_mm_positions

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert forward_batch.input_embeds is existing_input_embeds
    assert forward_batch.mm_inputs == [shell]
    assert schedule_req.omni_model_inputs is model_inputs_before
    assert schedule_req._omni_consumed is consumed
    assert schedule_req._omni_mm_positions is positions_before


def test_before_prefill_fails_closed_on_genuine_multimodal_items() -> None:
    from sglang.srt.managers.schedule_batch import MultimodalInputs

    runner = _runner()
    schedule_batch, requests = _requests([_audio_inputs()])
    schedule_req = schedule_batch.reqs[0]
    schedule_req.origin_input_ids = [7, 93, 7]
    schedule_req._omni_consumed = {"audio": 1}
    schedule_req._omni_mm_positions = {
        "image": torch.empty(0, dtype=torch.long),
        "video": torch.empty(0, dtype=torch.long),
        "audio": torch.tensor([1]),
    }
    schedule_req.inflight_middle_chunks = 0
    genuine = MultimodalInputs(mm_items=[object()])
    mm_inputs = [genuine]
    forward_batch = _forward_batch(
        mm_inputs=mm_inputs,
        input_ids=torch.tensor([7, 93, 7]),
    )
    original_model_inputs = schedule_req.omni_model_inputs
    original_consumed = schedule_req._omni_consumed
    original_positions = schedule_req._omni_mm_positions

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert forward_batch.mm_inputs is mm_inputs
    assert forward_batch.mm_inputs[0] is genuine
    assert forward_batch.input_embeds is None
    assert schedule_req.omni_model_inputs is original_model_inputs
    assert schedule_req._omni_consumed is original_consumed
    assert schedule_req._omni_mm_positions is original_positions


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

    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, requests) is None
    )


@pytest.mark.parametrize(
    ("token_id", "embed_key"),
    [(93, "audio_embeds"), (91, "image_embeds")],
)
def test_ordinary_multimodal_prefill_delegates_with_input_embeds(
    token_id: int, embed_key: str
) -> None:
    runner = _base_runner()
    input_ids = [7, token_id, 7]
    model_inputs = {embed_key: torch.randn(1, 4)}
    forward_batch, schedule_batch, requests, schedule_req = _legacy_request_pair(
        model_inputs, input_ids
    )

    result = runner.custom_prefill_forward(forward_batch, schedule_batch, requests)

    assert result is None
    assert forward_batch.input_embeds is not None
    assert torch.allclose(forward_batch.input_embeds[1], model_inputs[embed_key][0])
    assert schedule_req.omni_model_inputs is None


def test_qwen_runner_falls_back_to_shared_legacy_prefill() -> None:
    runner = _runner()
    forward_batch, schedule_batch, requests, schedule_req = _legacy_request_pair(
        {"audio_embeds": torch.randn(1, 4)}, [7, 93, 7]
    )

    result = runner.custom_prefill_forward(forward_batch, schedule_batch, requests)

    assert result is None
    assert forward_batch.input_embeds is not None
    assert schedule_req.omni_model_inputs is None


def test_deepstack_prefill_uses_direct_forward_and_attention_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _base_runner()
    input_ids = [7, 91, 7]
    model_inputs = {
        "image_embeds": torch.randn(1, 4),
        "image_deepstack_visual_embeds": [torch.randn(1, 2)],
    }
    forward_batch, schedule_batch, requests, _ = _legacy_request_pair(
        model_inputs, input_ids
    )

    class FakeOuter:
        lm_head = object()

        def __init__(self):
            self.model_calls = 0
            self.logits_calls = 0

        def model(self, **kwargs):
            self.model_calls += 1
            assert kwargs["input_ids"] is None
            assert kwargs["input_deepstack_embeds"].shape == (3, 2)
            return torch.zeros(3, 4)

        def logits_processor(self, *args):
            self.logits_calls += 1
            return "deepstack-logits"

    outer = FakeOuter()
    metadata_calls = 0

    class FakeAttentionBackend:
        def init_forward_metadata(self, batch):
            nonlocal metadata_calls
            assert batch is forward_batch
            metadata_calls += 1

    runner._outer_model = outer
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=FakeAttentionBackend())
    )
    context_enters = 0

    @contextlib.contextmanager
    def fake_attention_context(attn_backend):
        nonlocal context_enters
        assert isinstance(attn_backend, FakeAttentionBackend)
        context_enters += 1
        yield

    monkeypatch.setattr(
        thinker_runner_module, "attn_forward_context", fake_attention_context
    )

    result = runner.custom_prefill_forward(forward_batch, schedule_batch, requests)

    assert result.logits_output == "deepstack-logits"
    assert result.can_run_cuda_graph is False
    assert metadata_calls == 1
    assert outer.model_calls == 1
    assert outer.logits_calls == 1
    assert context_enters == 1


def test_payload_and_ordinary_delegation_do_not_enter_manual_attention_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_runner = _runner()
    ordinary_runner = _base_runner()
    context_enters = 0

    @contextlib.contextmanager
    def fake_attention_context(attn_backend):
        del attn_backend
        nonlocal context_enters
        context_enters += 1
        yield

    monkeypatch.setattr(
        thinker_runner_module, "attn_forward_context", fake_attention_context
    )

    payload_schedule, payload_requests = _requests([None])
    payload_batch = _forward_batch(mm_inputs=None)
    payload_batch.mm_inputs = OmniPrefillInputs(
        input_embeds=torch.zeros(3, 4), rids=("req-0",)
    )
    payload_schedule.forward_mode = SimpleNamespace(is_extend=lambda: True)
    assert (
        payload_runner.custom_prefill_forward(
            payload_batch, payload_schedule, payload_requests
        )
        is None
    )

    input_ids = [7, 93, 7]
    ordinary_batch, ordinary_schedule, ordinary_requests, _ = _legacy_request_pair(
        {"audio_embeds": torch.randn(1, 4)}, input_ids
    )
    assert (
        ordinary_runner.custom_prefill_forward(
            ordinary_batch, ordinary_schedule, ordinary_requests
        )
        is None
    )
    assert context_enters == 0


def test_qwen_outer_model_consumes_payload_as_single_prefill_embedding_source() -> None:
    class FakeTextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[str, object] = {}

        def forward(self, **kwargs):
            self.seen = kwargs
            return torch.zeros(3, 4)

    model = object.__new__(sglang_thinker.Qwen3OmniThinkerForCausalLM)
    nn.Module.__init__(model)
    text_model = FakeTextModel()
    model.model = text_model
    model.lm_head = object()
    logits_calls: list[tuple[object, ...]] = []

    def fake_logits_processor(*args):
        logits_calls.append(args)
        return "logits"

    model.logits_processor = fake_logits_processor
    input_ids = torch.tensor([11, 12, 13])
    payload_embeds = torch.randn(3, 4)
    current_prefill_positions = torch.arange(9).reshape(3, 3)
    forward_batch = SimpleNamespace(
        mrope_positions=current_prefill_positions,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        mm_inputs=OmniPrefillInputs(
            input_embeds=payload_embeds,
            rids=("req-0",),
        ),
    )

    result = model(
        input_ids=input_ids,
        positions=torch.arange(3),
        forward_batch=forward_batch,
    )

    assert result == "logits"
    assert text_model.seen["input_ids"] is None
    assert text_model.seen["positions"] is current_prefill_positions
    assert text_model.seen["input_embeds"] is payload_embeds
    assert logits_calls[0][0] is input_ids


def test_qwen_mrope_config_detection_is_semantic() -> None:
    assert sglang_thinker._config_uses_mrope(
        SimpleNamespace(rope_parameters={"mrope_section": [16, 24, 24]})
    )
    assert not sglang_thinker._config_uses_mrope(
        SimpleNamespace(rope_parameters={"rope_type": "default"})
    )


def test_qwen_outer_model_and_pinned_runner_share_mrope_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTextModel(nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
            self.embed_tokens = nn.Embedding(128, 4)

    monkeypatch.setattr(sglang_thinker, "Qwen3MoeLLMModel", FakeTextModel)
    monkeypatch.setattr(
        sglang_thinker,
        "LogitsProcessor",
        lambda config: SimpleNamespace(config=config),
    )
    text_config = SimpleNamespace(
        tie_word_embeddings=True,
        vocab_size=128,
        hidden_size=4,
        rope_parameters={"mrope_section": [16, 24, 24]},
    )
    config = SimpleNamespace(thinker_config=SimpleNamespace(text_config=text_config))
    outer = sglang_thinker.Qwen3OmniThinkerForCausalLM(config)

    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    runner = object.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(model=outer)
    forward_batch = SimpleNamespace(
        positions=torch.arange(3),
        mrope_positions=torch.arange(9).reshape(3, 3),
    )

    assert outer.is_mrope_enabled is True
    assert (
        runner._get_layer_model_positions(forward_batch)
        is forward_batch.mrope_positions
    )

# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Qwen3-Omni's prefill sidecar adopter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import MultimodalInputs  # noqa: E402

from sglang_omni.model_runner.prefill_inputs import (  # noqa: E402
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.model_runner.thinker_model_runner import (  # noqa: E402
    ThinkerModelRunner,
)
from sglang_omni.models.qwen3_omni.merge import build_thinker_inputs  # noqa: E402
from sglang_omni.models.qwen3_omni.payload_types import (  # noqa: E402
    Qwen3OmniPipelineState,
)
from sglang_omni.models.qwen3_omni.request_builders import (  # noqa: E402
    build_sglang_thinker_request,
)
from sglang_omni.models.qwen3_omni.thinker_model_runner import (  # noqa: E402
    Qwen3OmniThinkerModelRunner,
)
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer  # noqa: E402


def _runner() -> Qwen3OmniThinkerModelRunner:
    runner = object.__new__(Qwen3OmniThinkerModelRunner)
    runner._image_token_id = 100
    runner._video_token_id = 101
    runner._audio_token_id = 99
    runner._embed_tokens = torch.nn.Embedding(256, 4)
    with torch.no_grad():
        runner._embed_tokens.weight.copy_(
            torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
        )
    return runner


def _multimodal_inputs() -> MultimodalInputs:
    return MultimodalInputs(mm_items=[])


def _forward_batch(
    input_ids: list[int],
    *,
    lengths: list[int],
    starts: list[int],
    mm_inputs: list[MultimodalInputs] | None = None,
) -> SimpleNamespace:
    if mm_inputs is None:
        mm_inputs = [_multimodal_inputs() for _ in lengths]
    return SimpleNamespace(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=mm_inputs,
        batch_size=len(lengths),
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=list(starts),
        positions=torch.arange(len(input_ids), dtype=torch.long),
        mrope_positions=torch.arange(
            3 * len(input_ids), dtype=torch.long
        ).reshape(3, len(input_ids)),
    )


def _schedule(reqs: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=reqs,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )


def _audio_req(
    origin_input_ids: list[int],
    audio_embeds: torch.Tensor,
    *,
    start: int = 0,
    length: int | None = None,
    inflight_middle_chunks: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        origin_input_ids=origin_input_ids,
        extend_range=SimpleNamespace(
            start=start,
            length=len(origin_input_ids) if length is None else length,
        ),
        omni_model_inputs={
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": torch.tensor([audio_embeds.shape[0]]),
        },
        _omni_consumed=None,
        inflight_middle_chunks=inflight_middle_chunks,
    )


def _run_before(
    runner: Qwen3OmniThinkerModelRunner,
    forward_batch: SimpleNamespace,
    reqs: list[SimpleNamespace],
) -> None:
    schedule_batch = _schedule(reqs)
    runner.before_prefill(forward_batch, schedule_batch, reqs)


def test_pure_text_prefill_skips_sidecar_and_forward_input_embeds() -> None:
    runner = _runner()
    req = SimpleNamespace(omni_model_inputs={})
    forward_batch = _forward_batch([1, 2], lengths=[2], starts=[0])
    schedule_batch = _schedule([req])

    runner.before_prefill(forward_batch, schedule_batch, [req])
    result = runner.custom_prefill_forward(forward_batch, schedule_batch, [req])

    assert result is None
    assert get_omni_prefill_inputs(forward_batch) is None
    assert forward_batch.input_embeds is None


def test_audio_prefill_uses_sidecar_without_mutating_upstream_fields() -> None:
    runner = _runner()
    audio_embeds = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
    mm_inputs = _multimodal_inputs()
    req = _audio_req([1, 99, 2], audio_embeds)
    req.multimodal_inputs = mm_inputs
    forward_batch = _forward_batch(
        [1, 99, 2], lengths=[3], starts=[0], mm_inputs=[mm_inputs]
    )
    original_mrope_positions = forward_batch.mrope_positions

    _run_before(runner, forward_batch, [req])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    assert not hasattr(sidecar, "rids")
    expected = runner._embed_tokens(torch.tensor([1, 99, 2])).detach().clone()
    expected[1] = audio_embeds[0]
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs[0] is mm_inputs
    assert req.multimodal_inputs is mm_inputs
    assert forward_batch.mrope_positions is original_mrope_positions


def test_chunk_before_audio_uses_standard_text_path_and_preserves_state() -> None:
    runner = _runner()
    audio_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    req = _audio_req([1, 99, 2, 3, 99], audio_embeds, start=0, length=1)
    original_inputs = req.omni_model_inputs
    forward_batch = _forward_batch([1], lengths=[1], starts=[0])
    schedule_batch = _schedule([req])

    runner.before_prefill(forward_batch, schedule_batch, [req])
    result = runner.custom_prefill_forward(forward_batch, schedule_batch, [req])

    assert result is None
    assert get_omni_prefill_inputs(forward_batch) is None
    assert req.omni_model_inputs is original_inputs
    assert req._omni_consumed is None


def test_chunk_containing_audio_attaches_exact_rows_and_advances_cursor() -> None:
    runner = _runner()
    audio_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    req = _audio_req([1, 99, 2, 3, 99], audio_embeds, start=1, length=1)
    forward_batch = _forward_batch([99], lengths=[1], starts=[1])

    _run_before(runner, forward_batch, [req])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    torch.testing.assert_close(sidecar.input_embeds, audio_embeds[:1])
    assert req._omni_consumed == {"audio": 1}


def test_chunk_after_audio_uses_standard_path_and_preserves_state() -> None:
    runner = _runner()
    audio_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    req = _audio_req([1, 99, 2, 3, 99], audio_embeds, start=1, length=1)
    audio_batch = _forward_batch([99], lengths=[1], starts=[1])
    _run_before(runner, audio_batch, [req])
    assert req._omni_consumed == {"audio": 1}

    req.extend_range = SimpleNamespace(start=2, length=2)
    after_batch = _forward_batch([2, 3], lengths=[2], starts=[2])
    original_inputs = req.omni_model_inputs
    _run_before(runner, after_batch, [req])

    assert get_omni_prefill_inputs(after_batch) is None
    assert req.omni_model_inputs is original_inputs
    assert req._omni_consumed == {"audio": 1}


def test_mixed_text_and_audio_prefill_composes_one_sidecar() -> None:
    runner = _runner()
    text_req = SimpleNamespace(omni_model_inputs=None)
    audio_req = _audio_req([3, 99, 4], torch.tensor([[200.0] * 4]))
    forward_batch = _forward_batch(
        [1, 2, 3, 99, 4], lengths=[2, 3], starts=[0, 0]
    )
    schedule_batch = _schedule([text_req, audio_req])

    runner.before_prefill(forward_batch, schedule_batch, [text_req, audio_req])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    expected = runner._embed_tokens(forward_batch.input_ids).detach().clone()
    expected[3] = audio_req.omni_model_inputs["audio_embeds"][0]
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None


def test_mixed_text_and_audio_without_live_audio_stays_standard() -> None:
    runner = _runner()
    text_req = SimpleNamespace(omni_model_inputs=None)
    audio_req = _audio_req([1, 2, 99], torch.ones((1, 4)), start=0, length=2)
    forward_batch = _forward_batch(
        [1, 2, 1, 2], lengths=[2, 2], starts=[0, 0]
    )
    schedule_batch = _schedule([text_req, audio_req])

    runner.before_prefill(forward_batch, schedule_batch, [text_req, audio_req])

    assert get_omni_prefill_inputs(forward_batch) is None
    assert audio_req._omni_consumed is None


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
def test_unsupported_or_malformed_prefill_delegates_to_eager(
    monkeypatch, model_inputs
) -> None:
    runner = _runner()
    req = SimpleNamespace(
        origin_input_ids=[1, 99, 2],
        extend_range=SimpleNamespace(start=0, length=3),
        omni_model_inputs=model_inputs,
        _omni_consumed=None,
        inflight_middle_chunks=1,
    )
    forward_batch = _forward_batch([1, 99, 2], lengths=[3], starts=[0])
    schedule_batch = _schedule([req])
    called = []

    def eager_fallback(self, forward_batch, schedule_batch, requests):
        called.append((forward_batch, schedule_batch, requests))
        return "eager"

    monkeypatch.setattr(ThinkerModelRunner, "custom_prefill_forward", eager_fallback)

    runner.before_prefill(forward_batch, schedule_batch, [req])
    result = runner.custom_prefill_forward(forward_batch, schedule_batch, [req])

    assert result == "eager"
    assert len(called) == 1
    assert get_omni_prefill_inputs(forward_batch) is None


def test_malformed_consumed_offset_delegates_without_mutating_request(
    monkeypatch,
) -> None:
    runner = _runner()
    req = _audio_req([1, 99, 2], torch.ones((1, 4)))
    req._omni_consumed = {"audio": 2}
    forward_batch = _forward_batch([1, 99, 2], lengths=[3], starts=[0])
    schedule_batch = _schedule([req])
    called = []
    monkeypatch.setattr(
        ThinkerModelRunner,
        "custom_prefill_forward",
        lambda self, *args: called.append(args) or "eager",
    )

    runner.before_prefill(forward_batch, schedule_batch, [req])
    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, [req])
        == "eager"
    )
    assert req._omni_consumed == {"audio": 2}
    assert called


def test_available_omni_positions_are_consumed_without_mutating_metadata() -> None:
    runner = _runner()
    req = _audio_req([1, 99, 2], torch.ones((1, 4)))
    positions = {"audio": torch.tensor([1], dtype=torch.long)}
    req._omni_mm_positions = positions
    forward_batch = _forward_batch([1, 99, 2], lengths=[3], starts=[0])

    _run_before(runner, forward_batch, [req])

    assert get_omni_prefill_inputs(forward_batch) is not None
    assert req._omni_mm_positions is positions


def test_unexpected_composition_failure_propagates() -> None:
    runner = _runner()
    req = _audio_req([1, 99, 2], torch.ones((1, 4)))
    forward_batch = _forward_batch([1, 99, 2], lengths=[3], starts=[0])

    def fail(*args):
        raise RuntimeError("composition failed")

    runner._inject_multimodal_embeds = fail
    with pytest.raises(RuntimeError, match="composition failed"):
        _run_before(runner, forward_batch, [req])


def test_existing_forward_embed_or_sidecar_fails_closed() -> None:
    runner = _runner()
    req = _audio_req([1, 99, 2], torch.ones((1, 4)))
    schedule_batch = _schedule([req])

    forward_batch = _forward_batch([1, 99, 2], lengths=[3], starts=[0])
    forward_batch.input_embeds = torch.zeros(3, 4)
    runner.before_prefill(forward_batch, schedule_batch, [req])
    assert get_omni_prefill_inputs(forward_batch) is None
    assert forward_batch.input_embeds is not None

    forward_batch = _forward_batch([1, 99, 2], lengths=[3], starts=[0])
    payload = OmniPrefillInputs(input_embeds=torch.zeros(3, 4))
    attach_omni_prefill_inputs(forward_batch, payload)
    runner.before_prefill(forward_batch, schedule_batch, [req])
    assert get_omni_prefill_inputs(forward_batch) is payload


def test_real_thinker_merge_fields_reach_runner_request(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    prompt_ids = torch.tensor([1, 99, 2], dtype=torch.long)
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": prompt_ids, "attention_mask": torch.ones(3)},
        mm_inputs={"audio": {}},
    )
    audio_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    model_inputs = build_thinker_inputs(
        state,
        {
            "audio_encoder": {
                "audio_embeds": audio_embeds,
                "audio_feature_lengths": torch.tensor([2]),
            }
        },
    )
    state.thinker_inputs = {"model_inputs": model_inputs}
    request_data = build_sglang_thinker_request(
        state,
        params={},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
    )
    req = request_data.req
    req.extend_range = SimpleNamespace(start=0, length=3)
    req.inflight_middle_chunks = 1
    forward_batch = _forward_batch([1, 99, 2], lengths=[3], starts=[0])
    runner = _runner()

    _run_before(runner, forward_batch, [req])

    assert set(req.omni_model_inputs) == {
        "audio_embeds",
        "audio_feature_lengths",
    }
    assert get_omni_prefill_inputs(forward_batch) is not None

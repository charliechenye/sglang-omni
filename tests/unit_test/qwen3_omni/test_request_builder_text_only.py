# SPDX-License-Identifier: Apache-2.0
"""Request-builder contracts for Qwen3-Omni text-only inputs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang_omni.models.qwen3_omni.request_builders import (  # noqa: E402
    build_sglang_thinker_request,
    build_thinker_request,
)
from tests.unit_test.fixtures.qwen_fakes import (  # noqa: E402
    FakeQwenTokenizer,
    make_qwen_state,
)


def test_explicit_empty_model_inputs_remain_text_only() -> None:
    state = make_qwen_state(
        thinker_inputs={
            "model_inputs": {},
            "media_cache_keys": {"audio": "audio:cache"},
        }
    )

    request = build_thinker_request(state, params={})

    assert request.model_inputs == {}


def test_missing_model_inputs_key_keeps_legacy_fallback() -> None:
    audio_embeds = torch.ones((1, 4))
    state = make_qwen_state(
        thinker_inputs={
            "audio_embeds": audio_embeds,
            "capture_model_output_keys": ("hidden_states",),
        }
    )

    request = build_thinker_request(state, params={})

    assert set(request.model_inputs) == {"audio_embeds"}
    assert request.model_inputs["audio_embeds"] is audio_embeds


@pytest.mark.parametrize("malformed", [None, [], "audio"])
def test_malformed_explicit_model_inputs_raise_type_error(malformed) -> None:
    state = make_qwen_state(thinker_inputs={"model_inputs": malformed})

    with pytest.raises(TypeError, match="model_inputs must be a dict"):
        build_thinker_request(state, params={})

    with pytest.raises(TypeError, match="model_inputs must be a dict"):
        build_sglang_thinker_request(
            state,
            params={},
            tokenizer=FakeQwenTokenizer(),
            vocab_size=256,
        )


def test_sglang_text_only_request_has_no_omni_model_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    state = make_qwen_state(thinker_inputs={"model_inputs": {}})

    request = build_sglang_thinker_request(
        state,
        params={},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
    )

    assert request.model_inputs == {}
    assert request.req.omni_model_inputs is None

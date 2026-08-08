# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3-Omni text-only request classification."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang_omni.models.qwen3_omni.payload_types import (  # noqa: E402
    Qwen3OmniPipelineState,
)
from sglang_omni.models.qwen3_omni.request_builders import (  # noqa: E402
    build_sglang_thinker_request,
    build_thinker_request,
)


def _text_only_state() -> Qwen3OmniPipelineState:
    return Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([1, 2])},
        thinker_inputs={
            "model_inputs": {},
            "capture_model_output_keys": ("hidden_states",),
            "unrelated_metadata": "must-not-be-forwarded",
        },
    )


def test_explicit_empty_model_inputs_stays_text_only() -> None:
    request = build_thinker_request(_text_only_state(), params={})

    assert request.model_inputs == {}


def test_sglang_text_only_request_does_not_acquire_thinker_metadata(monkeypatch):
    from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer

    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    request = build_sglang_thinker_request(
        _text_only_state(),
        params={"max_new_tokens": 2},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
    )

    assert request.model_inputs == {}
    assert request.req.omni_model_inputs is None

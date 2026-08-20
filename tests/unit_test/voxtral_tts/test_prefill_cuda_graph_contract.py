# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for Voxtral's breakable-prefill adoption."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.voxtral_tts import CAPABILITIES
from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner
from sglang_omni.models.voxtral_tts.pipeline.engine_builder import (
    VoxtralTtsEngineBuilder,
)

HIDDEN = 4
VOCAB = 128
AUDIO_TOKEN_ID = 24


def _runner() -> VoxtralTTSModelRunner:
    runner = object.__new__(VoxtralTTSModelRunner)
    embedding = torch.nn.Embedding(VOCAB, HIDDEN)
    runner.model = SimpleNamespace(get_input_embeddings=lambda: embedding)
    return runner


def _request(
    full_ids: list[int],
    *,
    prefix_len: int = 0,
    voice_embedding: torch.Tensor | None = None,
) -> SimpleNamespace:
    req = SimpleNamespace(
        prefix_indices=list(range(prefix_len)),
        extend_range=SimpleNamespace(length=len(full_ids) - prefix_len),
    )
    data = SimpleNamespace(
        req=req,
        input_ids=torch.tensor(full_ids),
        voice_embedding=voice_embedding,
        audio_token_id=AUDIO_TOKEN_ID,
    )
    return SimpleNamespace(data=data)


def _batch(requests: list[SimpleNamespace]) -> SimpleNamespace:
    chunks = []
    for sched_req in requests:
        data = sched_req.data
        prefix_len = len(data.req.prefix_indices)
        req_len = int(data.req.extend_range.length)
        chunks.append(data.input_ids[prefix_len : prefix_len + req_len])
    input_ids = torch.cat(chunks)
    return SimpleNamespace(
        input_ids=input_ids,
        input_embeds=None,
        replace_embeds=None,
    )


def test_before_prefill_attaches_composed_embeddings_to_sidecar() -> None:
    runner = _runner()
    voice_one = torch.arange(8, dtype=torch.float32).reshape(2, HIDDEN) + 100
    voice_two = torch.arange(16, dtype=torch.float32).reshape(4, HIDDEN) + 200
    requests = [
        _request(
            [7, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 9],
            voice_embedding=voice_one,
        ),
        _request(
            [8, AUDIO_TOKEN_ID, 10, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID],
            prefix_len=2,
            voice_embedding=voice_two,
        ),
    ]
    forward_batch = _batch(requests)
    expected = runner._build_prefill_input_embeds(forward_batch, requests).clone()

    runner.before_prefill(forward_batch, None, requests)

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None
    torch.testing.assert_close(sidecar.input_embeds[1:3], voice_one)
    torch.testing.assert_close(sidecar.input_embeds[5:7], voice_two[1:3])


def test_voxtral_breakable_prefill_is_opt_in() -> None:
    builder = VoxtralTtsEngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")

    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is True
    assert builder.supports_breakable_prefill_cuda_graph is True
    assert "cuda_graph_backend_prefill" not in defaults
    assert "cuda_graph_bs_prefill" not in defaults


def test_forward_selects_logical_endpoints_from_padded_extend() -> None:
    from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel

    model = VoxtralSGLangTTSModel.__new__(VoxtralSGLangTTSModel)
    captured_hidden = torch.arange(8 * HIDDEN, dtype=torch.float32).reshape(8, HIDDEN)
    model.language_model = lambda **kwargs: captured_hidden
    forward_batch = SimpleNamespace(
        input_ids=torch.arange(8, dtype=torch.long),
        extend_seq_lens=torch.tensor([2, 3], dtype=torch.long),
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_extend=lambda: True,
        ),
    )

    output = model.forward(
        forward_batch.input_ids,
        torch.arange(8, dtype=torch.long),
        forward_batch,
        input_embeds=torch.zeros((5, HIDDEN)),
    )

    torch.testing.assert_close(output.hidden_states, captured_hidden[[1, 4]])
    assert output.next_token_logits.shape == (2, 1)

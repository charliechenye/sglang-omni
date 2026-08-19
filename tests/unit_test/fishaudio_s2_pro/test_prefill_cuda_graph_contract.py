# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
    S2ProSGLangTextModel,
    _S2ProTransformerBody,
)

HIDDEN_SIZE = 3
VOCAB_SIZE = 8


class _ForwardMode:
    def is_extend(self) -> bool:
        return True


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, positions, hidden_states, forward_batch, residual):
        del positions, forward_batch, residual
        return hidden_states + self.delta, None


class _SyntheticNorm(nn.Module):
    def forward(self, hidden_states, residual):
        del residual
        return hidden_states, None


def _minimal_model() -> S2ProSGLangTextModel:
    model = S2ProSGLangTextModel.__new__(S2ProSGLangTextModel)
    nn.Module.__init__(model)
    model.vocab_size = VOCAB_SIZE
    model.hidden_size = HIDDEN_SIZE
    model.num_layers = 1
    model.tie_word_embeddings = True
    model._vq_ready = False
    model.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(
                VOCAB_SIZE, HIDDEN_SIZE
            )
        )
    model.start_layer = 0
    model.end_layer = 1
    model.layers = nn.ModuleList([_SyntheticLayer()])
    model.norm = _SyntheticNorm()
    model.model = _S2ProTransformerBody(model)
    return model


def _forward_batch(num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.arange(num_tokens, dtype=torch.long),
        input_embeds=None,
        extend_seq_lens=torch.tensor([2, 3], dtype=torch.long),
        forward_mode=_ForwardMode(),
    )


def test_fish_prefill_graph_body_matches_transformer_path() -> None:
    model = _minimal_model()
    hidden_states = torch.arange(5 * HIDDEN_SIZE, dtype=torch.float32).reshape(
        5, HIDDEN_SIZE
    )
    batch = _forward_batch(5)

    output = model.model.forward(
        batch.input_ids,
        torch.arange(5),
        batch,
        input_embeds=hidden_states,
    )

    assert torch.equal(output, hidden_states + model.layers[0].delta)


def test_fish_prefill_graph_replay_keeps_outer_tail_eager() -> None:
    model = _minimal_model()
    batch = _forward_batch(8)
    replay_hidden = torch.arange(8 * HIDDEN_SIZE, dtype=torch.float32).reshape(
        8, HIDDEN_SIZE
    )
    original_forward = model.model.forward
    model.model.forward = lambda *args, **kwargs: replay_hidden
    try:
        output = model.forward(
            batch.input_ids,
            torch.arange(8),
            batch,
            input_embeds=torch.zeros(5, HIDDEN_SIZE),
        )
    finally:
        model.model.forward = original_forward

    expected_hidden = replay_hidden[torch.tensor([1, 4])]
    expected_logits = torch.nn.functional.linear(
        expected_hidden, model.embed_tokens.weight
    )
    assert torch.equal(output.hidden_states, expected_hidden)
    assert torch.equal(output.next_token_logits, expected_logits)

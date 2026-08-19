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


def test_fish_prefill_graph_replay_keeps_outer_tail_eager() -> None:
    model = S2ProSGLangTextModel.__new__(S2ProSGLangTextModel)
    nn.Module.__init__(model)
    model.tie_word_embeddings = True
    model._vq_ready = True
    model.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    model.layers = nn.ModuleList()
    model.model = _S2ProTransformerBody(model)

    replay_hidden = torch.arange(8 * HIDDEN_SIZE, dtype=torch.float32).reshape(
        8, HIDDEN_SIZE
    )
    body = model.model
    original_forward = body.forward

    seen: dict[str, torch.Tensor] = {}

    def decode_codebooks(logits: torch.Tensor, hidden_states: torch.Tensor) -> None:
        seen["logits"] = logits.clone()
        seen["hidden_states"] = hidden_states.clone()

    model._decode_codebooks = decode_codebooks
    batch = SimpleNamespace(
        input_ids=torch.arange(8, dtype=torch.long),
        extend_seq_lens=torch.tensor([2, 3], dtype=torch.long),
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )

    body.forward = lambda *args, **kwargs: replay_hidden
    try:
        output = model.forward(
            batch.input_ids,
            torch.arange(8),
            batch,
            input_embeds=torch.zeros(5, HIDDEN_SIZE),
        )
    finally:
        body.forward = original_forward

    expected_hidden = replay_hidden[torch.tensor([1, 4])]
    expected_logits = torch.nn.functional.linear(
        expected_hidden, model.embed_tokens.weight
    )
    assert torch.equal(output.hidden_states, expected_hidden)
    assert torch.equal(output.next_token_logits, expected_logits)
    assert set(seen) == {"hidden_states", "logits"}
    assert torch.equal(seen["hidden_states"], expected_hidden)
    assert torch.equal(seen["logits"], expected_logits)

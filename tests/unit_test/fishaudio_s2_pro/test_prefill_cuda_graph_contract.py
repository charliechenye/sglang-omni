# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel

HIDDEN_SIZE = 3
VOCAB_SIZE = 8


def test_fish_prefill_graph_replay_keeps_outer_tail_eager() -> None:
    model = S2ProSGLangTextModel.__new__(S2ProSGLangTextModel)
    nn.Module.__init__(model)
    model.tie_word_embeddings = True
    model._vq_ready = False
    model.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

    replay_hidden = torch.arange(8 * HIDDEN_SIZE, dtype=torch.float32).reshape(
        8, HIDDEN_SIZE
    )
    model.model = SimpleNamespace(forward=lambda *args, **kwargs: replay_hidden)
    batch = SimpleNamespace(
        input_ids=torch.arange(8, dtype=torch.long),
        extend_seq_lens=torch.tensor([2, 3], dtype=torch.long),
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )

    output = model.forward(
        batch.input_ids,
        torch.arange(8),
        batch,
        input_embeds=torch.zeros(5, HIDDEN_SIZE),
    )

    expected_hidden = replay_hidden[torch.tensor([1, 4])]
    expected_logits = torch.nn.functional.linear(
        expected_hidden, model.embed_tokens.weight
    )
    assert torch.equal(output.hidden_states, expected_hidden)
    assert torch.equal(output.next_token_logits, expected_logits)

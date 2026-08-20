# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for ZONOS2 breakable prefill CUDA graph adoption."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.zonos2 import callbacks
from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder


def _zonos2_sglang_model_module():
    return pytest.importorskip(
        "sglang_omni.models.zonos2.sglang_model",
        reason="SGLang runtime dependencies are not installed",
    )


class _PassthroughLayer(nn.Module):
    def forward(self, x, residual, router_states, positions, forward_batch):
        del residual, router_states, positions, forward_batch
        return x, torch.zeros_like(x), None


def _minimal_body(module, hidden_size: int = 4):
    body = module.Zonos2TransformerBody.__new__(module.Zonos2TransformerBody)
    nn.Module.__init__(body)
    body.emb_norm_eps = 1e-6
    body.layers = nn.ModuleList([_PassthroughLayer()])
    body.out_norm = nn.Parameter(torch.ones(hidden_size))
    return body


def _minimal_outer(module, body):
    outer = module.Zonos2SGLangModel.__new__(module.Zonos2SGLangModel)
    nn.Module.__init__(outer)
    outer.model = body
    return outer


def test_zonos2_prefill_attaches_embeddings_to_sidecar() -> None:
    expected = torch.randn(4, 4)
    runner = SimpleNamespace(_build_prefill_embeds=lambda *_: expected)
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
    )

    callbacks.zonos2_prefill_forward(runner, forward_batch, None, [])

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert payload.input_embeds is expected
    assert forward_batch.input_embeds is None


def test_zonos2_builder_supports_breakable_prefill() -> None:
    assert Zonos2EngineBuilder.supports_breakable_prefill_cuda_graph is True


def test_zonos2_outer_model_uses_resolved_transformer_body() -> None:
    module = _zonos2_sglang_model_module()
    resolver = pytest.importorskip(
        "sglang.srt.model_loader.utils"
    ).resolve_language_model
    body = _minimal_body(module)
    outer = _minimal_outer(module, body)
    input_ids = torch.zeros(2, dtype=torch.long)
    positions = torch.arange(2, dtype=torch.long)
    forward_batch = SimpleNamespace()
    input_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    resolved = resolver(outer)

    assert resolved is body
    hidden = resolved(input_ids, positions, forward_batch, input_embeds)
    assert torch.is_tensor(hidden)

    result = outer.forward(
        input_ids,
        positions,
        forward_batch,
        input_embeds=input_embeds,
    )

    torch.testing.assert_close(result.hidden_states, hidden)


def test_zonos2_loader_maps_checkpoint_weights_into_transformer_body() -> None:
    module = _zonos2_sglang_model_module()

    def parameter(shape):
        return nn.Parameter(torch.empty(shape))

    attention = SimpleNamespace(
        wq=parameter((4, 4)),
        wkv=parameter((2, 4, 4)),
        wo=parameter((4, 4)),
        gater=parameter((1, 4)),
        temp=parameter((1, 1, 1)),
    )
    layer = SimpleNamespace(
        attention=attention,
        attention_norm=parameter((4,)),
        ffn_norm=parameter((4,)),
        is_moe=False,
        feed_forward=SimpleNamespace(
            w_in=parameter((2, 3, 4)),
            w_out=parameter((4, 3)),
        ),
    )
    loader = SimpleNamespace(
        n_codebooks=0,
        embedders=[SimpleNamespace(weight=parameter((3, 4)))],
        model=SimpleNamespace(
            out_norm=parameter((4,)),
            layers=[layer],
        ),
        multi_output=parameter((2, 4)),
        speaker_lda_projection=SimpleNamespace(
            weight=parameter((2, 3)),
            bias=parameter((2,)),
        ),
        speaker_projection=SimpleNamespace(
            weight=parameter((4, 2)),
            bias=parameter((4,)),
        ),
    )
    checkpoint = {
        "multi_embedder.embedders.0.weight": torch.full((3, 4), 1.0),
        "out_norm.weight": torch.full((4,), 2.0),
        "multi_output.weight": torch.full((2, 4), 3.0),
        "speaker_lda_projection.weight": torch.full((2, 3), 4.0),
        "speaker_projection.weight": torch.full((4, 2), 5.0),
        "speaker_projection.bias": torch.full((4,), 6.0),
        "layers.0.attention.wq.weight": torch.full((4, 4), 7.0),
        "layers.0.attention.wkv.weight": torch.full((2, 4, 4), 8.0),
        "layers.0.attention.wo.weight": torch.full((4, 4), 9.0),
        "layers.0.attention.gater.weight": torch.full((1, 4), 10.0),
        "layers.0.attention.temp": torch.full((1, 1, 1), 11.0),
        "layers.0.attention_norm.weight": torch.full((4,), 12.0),
        "layers.0.ffn_norm.weight": torch.full((4,), 13.0),
        "layers.0.feed_forward.w_in.weight": torch.full((2, 3, 4), 14.0),
        "layers.0.feed_forward.w_out.weight": torch.full((4, 3), 15.0),
    }

    module.Zonos2SGLangModel.load_weights(loader, checkpoint.items())

    torch.testing.assert_close(loader.model.out_norm, checkpoint["out_norm.weight"])
    torch.testing.assert_close(
        loader.model.layers[0].attention.wq,
        checkpoint["layers.0.attention.wq.weight"],
    )
    torch.testing.assert_close(
        loader.model.layers[0].feed_forward.w_in,
        checkpoint["layers.0.feed_forward.w_in.weight"],
    )

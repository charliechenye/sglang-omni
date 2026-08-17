# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for ZONOS2 breakable prefill CUDA graph adoption."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.models.zonos2 import CAPABILITIES, callbacks
from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder


def _forward_batch(*, num_tokens: int = 4, replace_embeds=None) -> SimpleNamespace:
    return SimpleNamespace(
        input_embeds=None,
        replace_embeds=replace_embeds,
        replace_positions=None,
        mm_inputs=[object(), None],
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        batch_size=1,
        rids=["request"],
    )


def _zonos2_sglang_model_module():
    return pytest.importorskip(
        "sglang_omni.models.zonos2.sglang_model",
        reason="SGLang runtime dependencies are not installed",
    )


class _PassthroughLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

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


def test_zonos2_prefill_attaches_composed_embeds_to_private_sidecar() -> None:
    forward_batch = _forward_batch(num_tokens=4)
    mm_inputs = forward_batch.mm_inputs
    requests = [object()]
    embeds = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    calls: list[tuple[object, list[object]]] = []

    def build_prefill_embeds(batch, request_list):
        calls.append((batch, request_list))
        return embeds

    runner = SimpleNamespace(_build_prefill_embeds=build_prefill_embeds)

    assert (
        callbacks.zonos2_prefill_forward(runner, forward_batch, None, requests) is None
    )

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert payload.input_embeds is embeds
    assert payload.input_embeds.shape[0] == len(forward_batch.input_ids)
    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs is mm_inputs
    assert calls == [(forward_batch, requests)]


def test_zonos2_prefill_preserves_shared_sidecar_conflict_checks() -> None:
    forward_batch = _forward_batch(
        replace_embeds=torch.zeros(1, 4, dtype=torch.float32),
    )
    runner = SimpleNamespace(
        _build_prefill_embeds=lambda *_: torch.zeros(4, 4, dtype=torch.float32)
    )

    with pytest.raises(RuntimeError, match="replace_embeds"):
        callbacks.zonos2_prefill_forward(runner, forward_batch, None, [object()])


def test_zonos2_prefill_rejects_malformed_embed_rows() -> None:
    forward_batch = _forward_batch(num_tokens=4)
    runner = SimpleNamespace(
        _build_prefill_embeds=lambda *_: torch.zeros(3, 4, dtype=torch.float32)
    )

    with pytest.raises(RuntimeError, match="extend-window tokens"):
        callbacks.zonos2_prefill_forward(runner, forward_batch, None, [object()])


def test_zonos2_bcg_is_capable_but_not_default_enabled() -> None:
    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is True
    assert (
        Zonos2EngineBuilder.supports_breakable_prefill_cuda_graph
        is CAPABILITIES.supports_breakable_prefill_cuda_graph
    )

    defaults = Zonos2EngineBuilder().generation_defaults(dtype="bfloat16")
    assert "cuda_graph_backend_prefill" not in defaults
    assert "cuda_graph_bs_prefill" not in defaults


def test_zonos2_resolver_discovers_registered_transformer_body() -> None:
    module = _zonos2_sglang_model_module()
    resolver = pytest.importorskip(
        "sglang.srt.model_loader.utils"
    ).resolve_language_model
    body = _minimal_body(module)
    outer = _minimal_outer(module, body)

    resolved = resolver(outer)

    assert resolved is body
    assert resolved is not outer
    assert resolved.layers is body.layers
    assert not hasattr(outer, "language_model")


def test_zonos2_attention_layer_is_discoverable_by_sglang() -> None:
    compute_attention_and_moe_layers = pytest.importorskip(
        "sglang.srt.model_executor.model_runner_components.layer_setup"
    ).compute_attention_and_moe_layers
    sentinel_attention = object()
    body = SimpleNamespace(
        layers=[SimpleNamespace(attention=SimpleNamespace(attn=sentinel_attention))]
    )

    discovered = compute_attention_and_moe_layers(body)

    assert discovered.attention_layers == [sentinel_attention]


def test_zonos2_transformer_body_exposes_bcg_forward_signature() -> None:
    module = _zonos2_sglang_model_module()
    body = _minimal_body(module)
    outer = _minimal_outer(module, body)
    resolver = pytest.importorskip(
        "sglang.srt.model_loader.utils"
    ).resolve_language_model
    discovered_body = resolver(outer)

    assert list(inspect.signature(discovered_body.forward).parameters) == [
        "input_ids",
        "positions",
        "forward_batch",
        "input_embeds",
    ]


def test_zonos2_transformer_body_returns_backend_compatible_tensor() -> None:
    module = _zonos2_sglang_model_module()
    body = _minimal_body(module)
    input_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = body(
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        SimpleNamespace(),
        input_embeds,
    )

    assert torch.is_tensor(output)
    assert output.shape == input_embeds.shape


def test_zonos2_outer_forward_calls_discovered_body(monkeypatch) -> None:
    module = _zonos2_sglang_model_module()
    resolver = pytest.importorskip(
        "sglang.srt.model_loader.utils"
    ).resolve_language_model
    body = _minimal_body(module)
    outer = _minimal_outer(module, body)
    input_ids = torch.zeros(2, dtype=torch.long)
    positions = torch.arange(2, dtype=torch.long)
    forward_batch = _forward_batch(num_tokens=2)
    input_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected_hidden = torch.full((2, 4), 7.0)
    calls = []

    def sentinel(input_ids_arg, positions_arg, batch_arg, embeds_arg):
        calls.append((input_ids_arg, positions_arg, batch_arg, embeds_arg))
        return expected_hidden

    monkeypatch.setattr(body, "forward", sentinel)
    result = outer.forward(
        input_ids,
        positions,
        forward_batch,
        input_embeds=input_embeds,
    )

    assert resolver(outer) is body
    assert len(calls) == 1
    assert calls[0][0] is input_ids
    assert calls[0][1] is positions
    assert calls[0][2] is forward_batch
    assert calls[0][3] is input_embeds
    assert result.hidden_states is expected_hidden
    assert result.next_token_logits.shape == (2, 1)


def test_zonos2_sidecar_reaches_discovered_body_after_admission() -> None:
    module = _zonos2_sglang_model_module()
    sglang_model_runner = pytest.importorskip(
        "sglang_omni.model_runner.sglang_model_runner",
        reason="SGLang runtime dependencies are not installed",
    )
    body = _minimal_body(module)
    outer = _minimal_outer(module, body)
    forward_batch = _forward_batch(num_tokens=2)
    embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    runner_for_callback = SimpleNamespace(
        _build_prefill_embeds=lambda *_: embeds,
    )
    callbacks.zonos2_prefill_forward(
        runner_for_callback, forward_batch, None, [object()]
    )
    assert forward_batch.input_embeds is None

    transport_runner = sglang_model_runner.SGLModelRunner.__new__(
        sglang_model_runner.SGLModelRunner
    )
    transport_runner.support_pp = False
    transport_runner.is_generation = True
    kwargs = transport_runner._extend_forward_kwargs(forward_batch, object())

    seen: list[torch.Tensor] = []

    def sentinel(input_ids, positions, batch, input_embeds):
        del input_ids, positions, batch
        seen.append(input_embeds)
        return torch.zeros_like(input_embeds)

    body.forward = sentinel
    outer.forward(
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        forward_batch,
        **kwargs,
    )

    assert len(seen) == 1
    assert seen[0] is embeds
    assert forward_batch.input_embeds is None


def test_zonos2_sidecar_rejects_upstream_input_embeds_conflict() -> None:
    sglang_model_runner = pytest.importorskip(
        "sglang_omni.model_runner.sglang_model_runner",
        reason="SGLang runtime dependencies are not installed",
    )
    forward_batch = _forward_batch(num_tokens=2)
    forward_batch.input_embeds = torch.zeros(2, 4)
    attach_omni_prefill_inputs(
        forward_batch,
        OmniPrefillInputs(input_embeds=torch.ones(2, 4)),
    )
    runner = sglang_model_runner.SGLModelRunner.__new__(
        sglang_model_runner.SGLModelRunner
    )
    runner.support_pp = False
    runner.is_generation = True

    with pytest.raises(RuntimeError, match="upstream input_embeds"):
        runner._extend_forward_kwargs(forward_batch, object())


def test_zonos2_body_parameters_are_registered_once_under_inner_module() -> None:
    module = _zonos2_sglang_model_module()
    body = _minimal_body(module)
    outer = _minimal_outer(module, body)

    named_parameters = dict(outer.named_parameters())
    body_parameters = dict(body.named_parameters())

    assert set(named_parameters) == {
        "model.layers.0.weight",
        "model.out_norm",
    }
    assert {id(parameter) for parameter in named_parameters.values()} == {
        id(parameter) for parameter in body_parameters.values()
    }
    assert len({id(parameter) for parameter in outer.parameters()}) == len(
        named_parameters
    )
    assert not hasattr(outer, "layers")
    assert not hasattr(outer, "out_norm")


def test_zonos2_loader_maps_legacy_checkpoint_names_into_inner_body() -> None:
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

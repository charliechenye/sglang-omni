# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for ZONOS2 breakable prefill CUDA graph adoption."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
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


def test_zonos2_forward_consumes_late_bound_sidecar_embeds() -> None:
    sglang_model_runner = pytest.importorskip(
        "sglang_omni.model_runner.sglang_model_runner",
        reason="SGLang runtime dependencies are not installed",
    )
    from sglang_omni.models.zonos2.sglang_model import Zonos2SGLangModel

    forward_batch = _forward_batch(num_tokens=2)
    embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    runner_for_callback = SimpleNamespace(
        _build_prefill_embeds=lambda *_: embeds,
    )
    callbacks.zonos2_prefill_forward(
        runner_for_callback, forward_batch, None, [object()]
    )

    transport_runner = sglang_model_runner.SGLModelRunner.__new__(
        sglang_model_runner.SGLModelRunner
    )
    transport_runner.support_pp = False
    transport_runner.is_generation = True
    kwargs = transport_runner._extend_forward_kwargs(forward_batch, object())

    seen: list[torch.Tensor] = []

    class _Layer:
        def __call__(self, x, residual, router_states, positions, batch):
            del residual, router_states, positions, batch
            seen.append(x)
            return x, torch.zeros_like(x), None

    model = SimpleNamespace(
        emb_norm_eps=1e-6,
        layers=[_Layer()],
        out_norm=None,
        _warmup_embed=lambda *_: pytest.fail(
            "late-bound prefill embeddings must bypass _warmup_embed"
        ),
    )

    Zonos2SGLangModel.forward(
        model,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        forward_batch,
        **kwargs,
    )

    assert len(seen) == 1
    expected = torch.nn.functional.rms_norm(embeds, (4,), None, 1e-6)
    assert torch.equal(seen[0], expected)
    assert forward_batch.input_embeds is None

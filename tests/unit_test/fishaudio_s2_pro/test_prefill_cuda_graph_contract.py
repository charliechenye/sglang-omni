# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the FishAudio S2-Pro BCG adopter."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from torch import nn

import sglang_omni.models.fishaudio_s2_pro.engine_builder as fish_engine_builder
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    clear_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.fishaudio_s2_pro import CAPABILITIES
from sglang_omni.models.fishaudio_s2_pro.engine_builder import FishS2ProEngineBuilder
from sglang_omni.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
    S2ProSGLangTextModel,
    _S2ProTransformerView,
)
from tests.unit_test.fixtures.fish_fakes import FakeFishModel, FakeFishReq

HIDDEN = 3
VOCAB = 8


class _ForwardMode:
    def __init__(self, is_extend: bool) -> None:
        self._is_extend = is_extend

    def is_extend(self) -> bool:
        return self._is_extend


class _SyntheticLayer(nn.Module):
    def __init__(self, delta: tuple[float, ...]) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(delta, dtype=torch.float32))

    def forward(self, positions, hidden_states, forward_batch, residual):
        del positions, forward_batch, residual
        return hidden_states + self.delta, None


class _SyntheticNorm(nn.Module):
    def forward(self, hidden_states, residual):
        del residual
        return hidden_states, None


class _SyntheticCodebookEmbeddings:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        return (
            indices.to(dtype=torch.float32)
            .unsqueeze(-1)
            .expand(*indices.shape, self.hidden_size)
        )


def _minimal_model(*, hidden_size: int = HIDDEN) -> S2ProSGLangTextModel:
    model = S2ProSGLangTextModel.__new__(S2ProSGLangTextModel)
    nn.Module.__init__(model)
    model.vocab_size = VOCAB
    model.hidden_size = hidden_size
    model.num_layers = 1
    model.tie_word_embeddings = True
    model._vq_ready = False
    model.embed_tokens = nn.Embedding(VOCAB, hidden_size)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            torch.arange(VOCAB * hidden_size, dtype=torch.float32).reshape(
                VOCAB, hidden_size
            )
        )
    model.start_layer = 0
    model.end_layer = 1
    model.layers = nn.ModuleList(
        [_SyntheticLayer(tuple(float(index + 1) for index in range(hidden_size)))]
    )
    model.norm = _SyntheticNorm()
    model._transformer_view = _S2ProTransformerView(model)
    return model


def _forward_batch(
    *,
    num_input_rows: int,
    extend_seq_lens: list[int],
    is_extend: bool,
    input_embeds: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.arange(num_input_rows, dtype=torch.long),
        input_embeds=input_embeds,
        replace_embeds=None,
        replace_positions=None,
        mm_inputs=[None],
        batch_size=len(extend_seq_lens),
        rids=[f"request-{index}" for index in range(len(extend_seq_lens))],
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.long),
        forward_mode=_ForwardMode(is_extend),
    )


def _fish_request(
    *,
    extend_len: int,
    prefix_indices: list[int] | None = None,
    vq_mask_tokens: torch.Tensor | None = None,
    vq_parts: list[torch.Tensor] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            req=FakeFishReq(
                extend_len=extend_len,
                prefix_indices=prefix_indices,
            ),
            vq_mask_tokens=vq_mask_tokens,
            vq_parts=vq_parts,
        )
    )


def test_fish_before_prefill_transports_old_embedding_result_through_sidecar() -> None:
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    runner._sync_decode_state = lambda requests: None
    request = _fish_request(
        extend_len=3,
        vq_mask_tokens=torch.tensor([True, False, True]),
        vq_parts=[torch.tensor([[7, 8], [9, 10]], dtype=torch.long)],
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
    )

    expected_old_field = runner._build_prefill_input_embeds(
        forward_batch, [request]
    ).clone()
    runner.before_prefill(forward_batch, None, [request])

    prefill_inputs = get_omni_prefill_inputs(forward_batch)
    assert forward_batch.input_embeds is None
    assert prefill_inputs is not None
    assert prefill_inputs.input_embeds.shape == (3, 2)
    assert prefill_inputs.input_embeds.dtype == expected_old_field.dtype
    assert torch.equal(prefill_inputs.input_embeds, expected_old_field)


def test_fish_vq_prefill_composition_preserves_request_offsets_and_prefixes() -> None:
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    requests = [
        _fish_request(
            extend_len=3,
            prefix_indices=[100, 101],
            vq_mask_tokens=torch.tensor([False, True, True, False, True]),
            vq_parts=[torch.tensor([[10, 20, 30], [11, 21, 31]], dtype=torch.long)],
        ),
        _fish_request(
            extend_len=2,
            prefix_indices=[200],
            vq_mask_tokens=torch.tensor([True, False, True]),
            vq_parts=[torch.tensor([[40, 50, 60], [41, 51, 61]], dtype=torch.long)],
        ),
    ]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([10, 11, 12, 20, 21], dtype=torch.long)
    )

    embeds = runner._build_prefill_input_embeds(forward_batch, requests)
    expected = torch.tensor(
        [
            [1020.0, 1021.0],
            [11.0, 11.0],
            [1030.0, 1031.0],
            [20.0, 20.0],
            [1050.0, 1051.0],
        ]
    )
    assert torch.equal(embeds, expected)


def test_fish_transformer_view_matches_sglang_discovery_contract() -> None:
    model = _minimal_model()

    assert model.model is model._transformer_view
    assert model.model.layers is model.layers
    parameters = inspect.signature(model.model.forward).parameters
    assert "input_embeds" in parameters
    assert parameters["input_embeds"].default is None


def test_fish_transformer_view_accepts_sglang_prefill_graph_discovery_writeback() -> (
    None
):
    model = _minimal_model()
    view = model.model

    model.model = view

    assert model.model is view
    assert model._transformer_view is view

    with pytest.raises(ValueError, match="transformer view"):
        model.model = object()

    assert model.model is view


def test_fish_transformer_view_supports_sglang_monkeypatch_lifecycle() -> None:
    model = _minimal_model()
    view = model.model
    batch = _forward_batch(
        num_input_rows=2,
        extend_seq_lens=[2],
        is_extend=False,
        input_embeds=torch.ones(2, HIDDEN),
    )
    original = view.forward
    replay_calls: list[tuple[int, int]] = []

    def replay(input_ids, positions, forward_batch, input_embeds=None):
        del positions, forward_batch, input_embeds
        replay_calls.append((int(input_ids.shape[0]), HIDDEN))
        return torch.full((2, HIDDEN), 77.0)

    view.forward = replay
    try:
        replayed = view.forward(
            batch.input_ids,
            torch.arange(2),
            batch,
            batch.input_embeds,
        )
        assert replay_calls == [(2, HIDDEN)]
        assert torch.equal(replayed, torch.full((2, HIDDEN), 77.0))
    finally:
        view.forward = original

    restored = view.forward(
        batch.input_ids,
        torch.arange(2),
        batch,
        batch.input_embeds,
    )
    torch.testing.assert_close(restored, batch.input_embeds + model.layers[0].delta)


def test_fish_transformer_view_is_not_registered_or_duplicated() -> None:
    model = _minimal_model()
    module_names = dict(model.named_modules())
    parameter_names = dict(model.named_parameters())
    state_names = model.state_dict()

    assert not isinstance(model._transformer_view, nn.Module)
    assert "_transformer_view" not in module_names
    assert "model" not in module_names
    assert list(parameter_names).count("embed_tokens.weight") == 1
    assert list(parameter_names).count("layers.0.delta") == 1
    assert not any(
        name.startswith("_transformer_view.") or name.startswith("model.")
        for name in parameter_names
    )
    assert not any(
        name.startswith("_transformer_view.") or name.startswith("model.")
        for name in state_names
    )


def test_fish_outer_forward_selects_logical_rows_and_matches_expected_logits() -> None:
    model = _minimal_model()
    live_hidden = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)
    batch = _forward_batch(
        num_input_rows=5,
        extend_seq_lens=[2, 3],
        is_extend=True,
    )

    expected_hidden = (live_hidden + model.layers[0].delta)[torch.tensor([1, 4])]
    expected_logits = torch.nn.functional.linear(
        expected_hidden, model.embed_tokens.weight
    )
    output = model.forward(
        batch.input_ids,
        torch.arange(5),
        batch,
        input_embeds=live_hidden,
        omni_prefill_rids=("request-0", "request-1"),
    )

    torch.testing.assert_close(output.hidden_states, expected_hidden)
    torch.testing.assert_close(output.next_token_logits, expected_logits)


def test_fish_padded_bcg_body_feeds_logical_rows_to_eager_tail() -> None:
    model = _minimal_model()
    padded_hidden = torch.arange(8 * HIDDEN, dtype=torch.float32).reshape(8, HIDDEN)
    live_hidden = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)
    batch = _forward_batch(
        num_input_rows=8,
        extend_seq_lens=[2, 3],
        is_extend=True,
        input_embeds=live_hidden,
    )
    original = model.model.forward

    def replay(input_ids, positions, forward_batch, input_embeds=None):
        del input_ids, positions, forward_batch, input_embeds
        return padded_hidden.clone()

    model.model.forward = replay
    try:
        output = model.forward(
            batch.input_ids,
            torch.arange(8),
            batch,
            input_embeds=live_hidden,
            omni_prefill_rids=batch.rids,
        )
    finally:
        model.model.forward = original

    expected_hidden = padded_hidden[torch.tensor([1, 4])]
    expected_logits = torch.nn.functional.linear(
        expected_hidden, model.embed_tokens.weight
    )
    assert torch.equal(output.hidden_states, expected_hidden)
    torch.testing.assert_close(output.next_token_logits, expected_logits)


def test_fish_codebook_tail_receives_only_logical_padded_rows() -> None:
    model = _minimal_model()
    padded_hidden = torch.arange(8 * HIDDEN, dtype=torch.float32).reshape(8, HIDDEN)
    live_hidden = torch.zeros(5, HIDDEN)
    batch = _forward_batch(
        num_input_rows=8,
        extend_seq_lens=[2, 3],
        is_extend=True,
        input_embeds=live_hidden,
    )
    seen: dict[str, torch.Tensor] = {}

    def decode_codebooks(logits, hidden_states):
        seen["logits"] = logits.detach().clone()
        seen["hidden_states"] = hidden_states.detach().clone()

    model._vq_ready = True
    model._decode_codebooks = decode_codebooks
    original = model.model.forward
    model.model.forward = lambda *args, **kwargs: padded_hidden.clone()
    try:
        output = model.forward(
            batch.input_ids,
            torch.arange(8),
            batch,
            input_embeds=live_hidden,
        )
    finally:
        model.model.forward = original

    expected_hidden = padded_hidden[torch.tensor([1, 4])]
    expected_logits = torch.nn.functional.linear(
        expected_hidden, model.embed_tokens.weight
    )
    assert seen["hidden_states"].shape == (2, HIDDEN)
    assert torch.equal(seen["hidden_states"], expected_hidden)
    torch.testing.assert_close(seen["logits"], expected_logits)
    assert torch.equal(output.hidden_states, expected_hidden)


def test_fish_forward_accepts_late_bound_prefill_kwargs() -> None:
    parameters = inspect.signature(S2ProSGLangTextModel.forward).parameters
    assert "input_embeds" in parameters
    assert "omni_prefill_rids" in parameters

    model = _minimal_model()
    batch = _forward_batch(
        num_input_rows=2,
        extend_seq_lens=[2],
        is_extend=True,
    )
    output = model.forward(
        batch.input_ids,
        torch.arange(2),
        batch,
        input_embeds=torch.ones(2, HIDDEN),
        omni_prefill_rids=["request-0"],
    )
    assert output.hidden_states.shape == (1, HIDDEN)


def test_fish_decode_keeps_embedding_vq_transformer_and_sampling_tail() -> None:
    model = _minimal_model()
    model._vq_ready = True
    model._vq_codes = torch.tensor([[1, 2], [0, 1]], dtype=torch.long)
    model._vq_mask = torch.tensor([True, False])
    model._vq_codebook_offsets = torch.tensor([0, 3], dtype=torch.long)
    model._vq_codebook_embeddings = _SyntheticCodebookEmbeddings(HIDDEN)
    model._vq_scale = 0.5
    seen: dict[str, torch.Tensor] = {}

    def decode_codebooks(logits, hidden_states):
        seen["logits"] = logits.detach().clone()
        seen["hidden_states"] = hidden_states.detach().clone()

    model._decode_codebooks = decode_codebooks
    batch = _forward_batch(
        num_input_rows=2,
        extend_seq_lens=[2],
        is_extend=False,
    )
    assert batch.input_embeds is None
    assert get_omni_prefill_inputs(batch) is None

    token_embeds = model.embed_tokens(batch.input_ids)
    offset_parts = model._vq_codes + model._vq_codebook_offsets[None, :]
    vq_sum = (
        offset_parts.to(torch.float32).unsqueeze(-1).expand(2, 2, HIDDEN).sum(dim=1)
    )
    combined = (token_embeds + vq_sum) * model._vq_scale
    expected_input = torch.where(model._vq_mask.unsqueeze(-1), combined, token_embeds)
    expected_hidden = expected_input + model.layers[0].delta
    expected_logits = torch.nn.functional.linear(
        expected_hidden, model.embed_tokens.weight
    )

    output = model.forward(batch.input_ids, torch.arange(2), batch)

    torch.testing.assert_close(output.hidden_states, expected_hidden)
    torch.testing.assert_close(output.next_token_logits, expected_logits)
    torch.testing.assert_close(seen["hidden_states"], expected_hidden)
    assert seen["hidden_states"].shape == (2, HIDDEN)


def test_fish_prefill_sidecar_is_late_bound_for_eager_fallback() -> None:
    model = _minimal_model()
    model_runner = SGLModelRunner.__new__(SGLModelRunner)
    model_runner.support_pp = False
    model_runner.is_generation = True
    batch = _forward_batch(
        num_input_rows=2,
        extend_seq_lens=[2],
        is_extend=True,
    )
    payload = OmniPrefillInputs(input_embeds=torch.ones(2, HIDDEN))
    attach_omni_prefill_inputs(batch, payload)

    kwargs = model_runner._extend_forward_kwargs(batch, object())
    assert kwargs["input_embeds"] is payload.input_embeds
    assert kwargs["omni_prefill_rids"] == batch.rids
    assert batch.input_embeds is None

    output = model.forward(
        batch.input_ids,
        torch.arange(2),
        batch,
        **kwargs,
    )
    assert output.hidden_states.shape == (1, HIDDEN)


def test_fish_prefill_sidecar_is_cleared_by_shared_prepare_finally() -> None:
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    runner._sync_decode_state = lambda requests: None
    request = _fish_request(extend_len=2)
    batch = SimpleNamespace(
        input_ids=torch.tensor([10, 11], dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
    )
    cleanup_seen: list[object] = []

    def fail_forward(*args):
        del args
        raise RuntimeError("synthetic Fish forward failure")

    runner.custom_prefill_forward = fail_forward
    runner.cleanup_prefill = lambda *args: cleanup_seen.append(
        get_omni_prefill_inputs(batch)
    )

    with pytest.raises(RuntimeError, match="synthetic Fish forward failure"):
        runner._prepare_and_forward(
            batch,
            SimpleNamespace(is_prefill_only=True),
            [request],
            True,
        )

    assert cleanup_seen == [None]
    assert get_omni_prefill_inputs(batch) is None


def test_fish_prefill_uses_shared_fail_closed_sidecar_validation() -> None:
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    runner._sync_decode_state = lambda requests: None
    request = _fish_request(extend_len=2)
    batch = SimpleNamespace(
        input_ids=torch.tensor([10, 11], dtype=torch.long),
        input_embeds=None,
        replace_embeds=torch.zeros(1, HIDDEN),
    )

    with pytest.raises(RuntimeError, match="replace_embeds"):
        runner.before_prefill(batch, None, [request])
    assert get_omni_prefill_inputs(batch) is None

    mismatch_batch = SimpleNamespace(
        input_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        replace_embeds=None,
    )
    with pytest.raises(RuntimeError, match="extend-window tokens"):
        attach_omni_prefill_inputs(
            mismatch_batch,
            OmniPrefillInputs(input_embeds=torch.zeros(2, HIDDEN)),
        )

    first = OmniPrefillInputs(input_embeds=torch.zeros(3, HIDDEN))
    second = OmniPrefillInputs(input_embeds=torch.ones(3, HIDDEN))
    attach_omni_prefill_inputs(mismatch_batch, first)
    with pytest.raises(RuntimeError, match="already attached"):
        attach_omni_prefill_inputs(mismatch_batch, second)
    assert get_omni_prefill_inputs(mismatch_batch) is first
    clear_omni_prefill_inputs(mismatch_batch)


def test_fish_declares_bcg_capability_without_a_default_or_guessed_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is True
    assert FishS2ProEngineBuilder.supports_breakable_prefill_cuda_graph is True

    seen_gpu_ids: list[int] = []

    def fake_get_visible_gpu_sm_version(gpu_id: int) -> int:
        seen_gpu_ids.append(gpu_id)
        return 90

    monkeypatch.setattr(
        fish_engine_builder,
        "get_visible_gpu_sm_version",
        fake_get_visible_gpu_sm_version,
    )

    builder = FishS2ProEngineBuilder(max_new_tokens=16, ras_window=4)
    builder.gpu_id = 0
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert seen_gpu_ids == [0]
    assert "cuda_graph_backend_prefill" not in defaults
    assert "cuda_graph_bs_prefill" not in defaults
    assert "cuda_graph_max_bs_prefill" not in defaults

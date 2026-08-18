# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from transformers import WhisperConfig

from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.model_runner.whisper_prefill_cuda_graph_runner import (
    WhisperPrefillCudaGraphRunner,
)
from sglang_omni.models.whisper_asr import sglang_model
from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperSGLangCrossAttention,
    WhisperSGLangSelfAttention,
)


def _tiny_whisper_config() -> WhisperConfig:
    return WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=32,
        max_source_positions=8,
        max_target_positions=8,
        num_mel_bins=4,
    )


def _encoder_batch(
    *,
    encoder_lens: list[int],
    encoder_cached: list[bool],
    encoder_out_cache_loc: torch.Tensor | None,
    has_inputs: bool = True,
) -> SimpleNamespace:
    mm_inputs = None
    if has_inputs:
        mm_inputs = [
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        feature=torch.ones(1),
                        precomputed_embeddings=None,
                    )
                ]
            )
            for _ in encoder_lens
        ]
    return SimpleNamespace(
        batch_size=len(encoder_lens),
        encoder_lens=torch.tensor(encoder_lens, dtype=torch.int64),
        encoder_lens_cpu=encoder_lens,
        encoder_cached=encoder_cached,
        encoder_out_cache_loc=encoder_out_cache_loc,
        mm_inputs=mm_inputs,
    )


def test_whisper_model_exposes_decoder_body_without_duplicate_registration() -> None:
    model = WhisperModel(_tiny_whisper_config())

    assert model.layers is model.decoder.layers
    assert "input_embeds" in inspect.signature(model.forward).parameters
    assert not any(name.startswith("layers.") for name in model.state_dict())
    assert any(name.startswith("decoder.layers.") for name in model.state_dict())


@pytest.mark.parametrize("encoder_tokens", [1, 3])
def test_whisper_cross_attention_caches_encoder_kv_eagerly(
    monkeypatch: pytest.MonkeyPatch,
    encoder_tokens: int,
) -> None:
    attention = WhisperSGLangCrossAttention(_tiny_whisper_config(), layer_id=1)
    states = torch.randn(encoder_tokens, attention.embed_dim)
    cache_loc = torch.arange(encoder_tokens, dtype=torch.int64)
    writes: list[tuple[object, object, torch.Tensor, torch.Tensor]] = []
    token_to_kv_pool = SimpleNamespace(
        set_kv_buffer=lambda layer, loc, key, value: writes.append(
            (layer, loc, key, value)
        )
    )
    monkeypatch.setattr(
        sglang_model,
        "get_attn_backend",
        lambda: SimpleNamespace(token_to_kv_pool=token_to_kv_pool),
    )

    attention.cache_encoder_states(states, cache_loc)

    assert len(writes) == 1
    layer, write_loc, key, value = writes[0]
    assert layer is attention.attn
    assert write_loc.loc is cache_loc
    assert key.shape == (encoder_tokens, attention.num_heads, attention.head_dim)
    assert value.shape == key.shape


@pytest.mark.parametrize(
    "attention_cls",
    [WhisperSGLangSelfAttention, WhisperSGLangCrossAttention],
)
def test_whisper_attention_flattens_breakable_graph_head_output(
    attention_cls: type[torch.nn.Module],
) -> None:
    class _HeadShapedAttention(torch.nn.Module):
        def forward(self, query, key, value, forward_batch):
            del key, value, forward_batch
            return query

    attention = attention_cls(_tiny_whisper_config(), layer_id=0)
    attention.attn = _HeadShapedAttention()
    attention.out_proj = torch.nn.Identity()
    hidden_states = torch.randn(3, attention.embed_dim)

    output = attention(hidden_states, SimpleNamespace())

    assert output.shape == hidden_states.shape


def test_whisper_cross_attention_queries_cached_kv_without_reprojecting() -> None:
    attention = WhisperSGLangCrossAttention(_tiny_whisper_config(), layer_id=1)
    calls: list[tuple[object, object]] = []

    class _CachedAttention(torch.nn.Module):
        def forward(self, query, key, value, forward_batch):
            del query, forward_batch
            calls.append((key, value))
            return torch.zeros(2, attention.num_heads, attention.head_dim)

    attention.attn = _CachedAttention()
    attention.out_proj = torch.nn.Identity()
    output = attention(torch.randn(2, attention.embed_dim), SimpleNamespace())

    assert output.shape == (2, attention.embed_dim)
    assert calls == [(None, None)]


def test_whisper_precomputed_encoder_states_are_cached_before_decoder_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    encoder_states = torch.randn(3, model.config.d_model)
    mm_input = SimpleNamespace(
        mm_items=[
            SimpleNamespace(
                feature=None,
                precomputed_embeddings=encoder_states,
            )
        ]
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_cached=[False],
        encoder_lens=torch.tensor([3]),
        encoder_out_cache_loc=torch.arange(3),
        mm_inputs=[mm_input],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        model.model,
        "cache_encoder_states",
        lambda states, batch: (
            calls.append("cache")
            if states.shape == encoder_states.shape
            and batch is forward_batch.encoder_out_cache_loc
            else (_ for _ in ()).throw(AssertionError("bad encoder cache"))
        ),
    )
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda *args, **kwargs: calls.append("decoder")
        or torch.randn(input_ids.shape[0], model.config.d_model),
    )
    monkeypatch.setattr(
        model.logits_processor,
        "forward",
        lambda *args, **kwargs: calls.append("logits") or object(),
    )

    model(input_ids, positions, forward_batch)

    assert calls == ["cache", "decoder", "logits"]


def test_whisper_encoder_in_prefill_fallback_uses_the_same_kv_cache_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    encoder_states = torch.randn(1, 3, model.config.d_model)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_cached=[False],
        encoder_lens=torch.tensor([3]),
        encoder_out_cache_loc=torch.arange(3),
        mm_inputs=[None],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        model,
        "_batch_audio_inputs",
        lambda batch: (torch.randn(1, 4, 4), [3]),
    )
    monkeypatch.setattr(
        model,
        "_run_encoder",
        lambda features: calls.append("encoder") or encoder_states,
    )
    monkeypatch.setattr(
        model.model,
        "cache_encoder_states",
        lambda states, batch: calls.append("cache"),
    )
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda *args, **kwargs: calls.append("decoder")
        or torch.randn(input_ids.shape[0], model.config.d_model),
    )
    monkeypatch.setattr(
        model.logits_processor,
        "forward",
        lambda *args, **kwargs: calls.append("logits") or object(),
    )

    model(input_ids, positions, forward_batch)

    assert calls == ["encoder", "cache", "decoder", "logits"]


def test_whisper_no_encoder_batch_keeps_skip_cross_attention_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_cached=[True],
        encoder_lens=torch.tensor([0]),
        encoder_out_cache_loc=None,
        mm_inputs=[None],
    )
    skip_values: list[bool] = []
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda *args, **kwargs: skip_values.append(kwargs["skip_cross_attention"])
        or torch.randn(2, model.config.d_model),
    )
    monkeypatch.setattr(
        model.logits_processor,
        "forward",
        lambda *args, **kwargs: object(),
    )

    model(torch.tensor([1, 2]), torch.tensor([0, 1]), forward_batch)

    assert skip_values == [True]


def test_whisper_prefill_capture_populates_encoder_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    capture_batch = SimpleNamespace(
        batch_size=1,
        encoder_lens=None,
        encoder_lens_cpu=None,
        encoder_cached=None,
        encoder_out_cache_loc=None,
    )
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "capture_prepare",
        lambda self, num_tokens: (capture_batch, object()),
    )
    runner.device = "cpu"

    batch, _ = runner.capture_prepare(4)

    assert batch.encoder_lens.tolist() == [1]
    assert batch.encoder_lens_cpu == [1]
    assert batch.encoder_cached == [True]
    assert batch.encoder_out_cache_loc is None


def test_whisper_prefill_runner_registers_self_and_cross_attention_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_attention_layers = [object(), object()]
    cross_attention_layers = [object(), object()]
    decoder_layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(attn=self_attention),
            encoder_attn=SimpleNamespace(attn=cross_attention),
        )
        for self_attention, cross_attention in zip(
            self_attention_layers, cross_attention_layers
        )
    ]
    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                decoder=SimpleNamespace(layers=decoder_layers),
            )
        ),
        attention_layers=[],
    )
    initialized: list[object] = []
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "__init__",
        lambda self, runner: initialized.append(runner),
    )

    WhisperPrefillCudaGraphRunner(model_runner)

    assert model_runner.attention_layers == (
        self_attention_layers + cross_attention_layers
    )
    assert len({id(layer) for layer in model_runner.attention_layers}) == 4
    assert initialized == [model_runner]


@pytest.mark.parametrize(
    ("encoder_lens", "encoder_cached", "cache_loc", "has_inputs", "expected"),
    [
        ([4], [True], None, False, True),
        ([0], [False], None, False, False),
        ([4], [False], None, True, False),
        ([4], [False], torch.arange(4), True, True),
        ([4], [False], torch.arange(4), False, False),
    ],
)
def test_whisper_prefill_admission_requires_encoder_context(
    encoder_lens: list[int],
    encoder_cached: list[bool],
    cache_loc: torch.Tensor | None,
    has_inputs: bool,
    expected: bool,
) -> None:
    batch = _encoder_batch(
        encoder_lens=encoder_lens,
        encoder_cached=encoder_cached,
        encoder_out_cache_loc=cache_loc,
        has_inputs=has_inputs,
    )

    assert WhisperPrefillCudaGraphRunner._encoder_metadata_is_usable(batch) is expected


def test_whisper_prefill_replay_preserves_encoder_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    static_batch = SimpleNamespace(
        encoder_lens=None,
        encoder_lens_cpu=None,
        encoder_cached=None,
        encoder_out_cache_loc=None,
    )
    live_batch = SimpleNamespace(
        encoder_lens=torch.tensor([7, 5]),
        encoder_lens_cpu=[7, 5],
        encoder_cached=[False, True],
        encoder_out_cache_loc=torch.tensor([11, 12, 13]),
    )
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "load_batch",
        lambda self, forward_batch, **kwargs: static_batch,
    )

    replay_batch = runner.load_batch(live_batch)

    assert replay_batch.encoder_lens is live_batch.encoder_lens
    assert replay_batch.encoder_lens_cpu == live_batch.encoder_lens_cpu
    assert replay_batch.encoder_cached == live_batch.encoder_cached
    assert replay_batch.encoder_out_cache_loc is live_batch.encoder_out_cache_loc


@pytest.mark.parametrize(
    ("architecture", "backend", "expected"),
    [
        ("WhisperForConditionalGeneration", "breakable", WhisperPrefillCudaGraphRunner),
        ("WhisperForConditionalGeneration", "disabled", None),
        ("Qwen3ASRForConditionalGeneration", "breakable", None),
    ],
)
def test_model_runner_selects_whisper_prefill_adapter_only_when_needed(
    architecture: str,
    backend: str,
    expected: type[WhisperPrefillCudaGraphRunner] | None,
) -> None:
    runner = object.__new__(SGLModelRunner)
    runner._model_arch_override = architecture
    runner.server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend=backend))
    )

    assert runner._prefill_cuda_graph_runner_cls() is expected


def test_model_runner_restores_prefill_adapter_after_capture_failure() -> None:
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup

    runner = object.__new__(SGLModelRunner)
    runner._model_arch_override = "WhisperForConditionalGeneration"
    runner.server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="breakable"))
    )
    original = cuda_graph_setup.PrefillCudaGraphRunner

    with pytest.raises(RuntimeError, match="capture failed"):
        with runner._prefill_cuda_graph_runner_override():
            assert (
                cuda_graph_setup.PrefillCudaGraphRunner is WhisperPrefillCudaGraphRunner
            )
            raise RuntimeError("capture failed")

    assert cuda_graph_setup.PrefillCudaGraphRunner is original

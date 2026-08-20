# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.runtime_context import get_context, get_parallel
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr.prefill_cuda_graph_runner import (
    WhisperPrefillCudaGraphRunner,
)
from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperForConditionalGeneration,
    WhisperSGLangCrossAttention,
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


@pytest.fixture
def sglang_runtime_context():
    with (
        get_context().override_server_args(),
        get_parallel().override(tp_size=1),
    ):
        yield


def _mm_input(embeddings: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        mm_items=[SimpleNamespace(precomputed_embeddings=embeddings)]
    )


def test_whisper_mixed_prefill_keeps_uncached_encoder_order(
    monkeypatch: pytest.MonkeyPatch,
    sglang_runtime_context,
):
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    expected_states = torch.tensor([[11.0] * 8, [12.0] * 8, [31.0] * 8, [32.0] * 8])
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_cached=[True, False, True, False],
        encoder_lens=torch.tensor([2, 2, 2, 2]),
        encoder_lens_cpu=[2, 2, 2, 2],
        encoder_out_cache_loc=torch.tensor([41, 42, 73, 74]),
        mm_inputs=[
            None,
            _mm_input(expected_states[:2]),
            None,
            _mm_input(expected_states[2:]),
        ],
    )
    cache_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    events: list[str] = []

    def record_cache(states, cache_loc):
        cache_calls.append((states.clone(), cache_loc.clone()))
        events.append("cache")

    for layer in model.model.decoder.layers:
        monkeypatch.setattr(
            layer.encoder_attn,
            "cache_encoder_states",
            record_cache,
        )
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda *_args, **_kwargs: events.append("decoder")
        or torch.randn(2, model.config.d_model),
    )
    monkeypatch.setattr(
        model.logits_processor,
        "forward",
        lambda *_args, **_kwargs: events.append("logits") or object(),
    )

    model(torch.tensor([1, 2]), torch.tensor([0, 1]), forward_batch)

    assert events == ["cache", "cache", "decoder", "logits"]
    assert len(cache_calls) == 2
    for states, cache_loc in cache_calls:
        torch.testing.assert_close(states, expected_states)
        torch.testing.assert_close(cache_loc, forward_batch.encoder_out_cache_loc)


def test_whisper_cross_attention_reads_cached_kv_without_reprojection():
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


def test_whisper_decode_capture_reuses_cached_encoder_kv(
    monkeypatch: pytest.MonkeyPatch,
    sglang_runtime_context,
):
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        encoder_cached=None,
        encoder_lens=torch.tensor([3]),
        encoder_lens_cpu=None,
        encoder_out_cache_loc=None,
        mm_inputs=None,
    )
    cache_calls: list[object] = []
    for layer in model.model.decoder.layers:
        monkeypatch.setattr(
            layer.encoder_attn,
            "cache_encoder_states",
            lambda *args, **kwargs: cache_calls.append((args, kwargs)),
        )
    skip_cross_attention: list[bool] = []
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda *args, **kwargs: skip_cross_attention.append(
            kwargs["skip_cross_attention"]
        )
        or torch.randn(1, model.config.d_model),
    )
    monkeypatch.setattr(
        model.logits_processor,
        "forward",
        lambda *args, **kwargs: object(),
    )

    with model_capture_mode():
        result = model(torch.tensor([1]), torch.tensor([0]), forward_batch)

    assert result is not None
    assert cache_calls == []
    assert skip_cross_attention == [False]


def test_whisper_prefill_runner_exposes_self_and_cross_attention(
    monkeypatch: pytest.MonkeyPatch,
):
    self_attention = object()
    cross_attention = object()
    decoder_layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(attn=self_attention),
            encoder_attn=SimpleNamespace(attn=cross_attention),
        )
    ]
    original_attention_layers = [object()]
    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(decoder=SimpleNamespace(layers=decoder_layers))
        ),
        attention_layers=original_attention_layers,
    )
    monkeypatch.setattr(
        PrefillCudaGraphRunner,
        "__init__",
        lambda self, runner: setattr(self, "attention_layers", runner.attention_layers),
    )
    prefill_runner = WhisperPrefillCudaGraphRunner(model_runner)

    assert model_runner.attention_layers is original_attention_layers
    assert prefill_runner.attention_layers == [self_attention, cross_attention]


def test_whisper_prefill_runner_keeps_encoder_metadata_live(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    capture_batch = SimpleNamespace(batch_size=2)
    monkeypatch.setattr(
        PrefillCudaGraphRunner,
        "capture_prepare",
        lambda self, num_tokens: (capture_batch, object()),
    )
    runner.device = "cpu"
    captured, _ = runner.capture_prepare(4)

    assert captured.encoder_lens.tolist() == [1, 1]
    assert captured.encoder_lens_cpu == [1, 1]

    static_batch = SimpleNamespace()
    live_batch = SimpleNamespace(
        encoder_lens=torch.tensor([7, 5]),
        mm_inputs=[object(), object()],
        encoder_lens_cpu=[7, 5],
        encoder_cached=[False, True],
        encoder_out_cache_loc=torch.tensor([11, 12, 13]),
    )
    monkeypatch.setattr(
        PrefillCudaGraphRunner,
        "load_batch",
        lambda self, forward_batch, **kwargs: static_batch,
    )
    replay_batch = runner.load_batch(live_batch)

    assert replay_batch.encoder_lens_cpu == live_batch.encoder_lens_cpu
    assert replay_batch.encoder_cached == live_batch.encoder_cached
    assert replay_batch.encoder_out_cache_loc is live_batch.encoder_out_cache_loc

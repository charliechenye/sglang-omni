# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.runtime_context import get_context, get_parallel
from transformers import WhisperConfig

from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.whisper_asr import sglang_model
from sglang_omni.models.whisper_asr.prefill_cuda_graph_runner import (
    WhisperPrefillCudaGraphRunner,
)
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


@pytest.fixture
def sglang_runtime_context():
    with (
        get_context().override_server_args(),
        get_parallel().override(tp_size=1),
    ):
        yield


def test_whisper_model_exposes_decoder_body_without_duplicate_registration() -> None:
    model = WhisperModel(_tiny_whisper_config())

    assert model.layers is model.decoder.layers
    assert "input_embeds" in inspect.signature(model.forward).parameters
    assert not any(name.startswith("layers.") for name in model.state_dict())
    assert any(name.startswith("decoder.layers.") for name in model.state_dict())


def test_whisper_decoder_uses_prefill_input_embeds_without_reembedding() -> None:
    model = WhisperModel(_tiny_whisper_config())
    provided = torch.randn(3, model.decoder.embed_tokens.embedding_dim)

    class _IdentityLayer(torch.nn.Module):
        def forward(self, hidden_states, forward_batch, skip_cross_attention=False):
            del forward_batch, skip_cross_attention
            return hidden_states

    model.decoder.layers = torch.nn.ModuleList([_IdentityLayer()])
    model.decoder.layer_norm = torch.nn.Identity()

    def fail_embedding(*args, **kwargs):
        del args, kwargs
        raise AssertionError("decoder re-embedded input_ids despite input_embeds")

    model.decoder.embed_tokens.forward = fail_embedding
    model.decoder.embed_positions.forward = fail_embedding

    output = model(
        torch.tensor([1, 2, 3]),
        torch.tensor([0, 1, 2]),
        SimpleNamespace(),
        input_embeds=provided,
        skip_cross_attention=True,
    )

    torch.testing.assert_close(output, provided)


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


@pytest.mark.parametrize("encoder_tokens", [1, 3])
def test_whisper_cross_attention_caches_encoder_kv_eagerly(
    monkeypatch: pytest.MonkeyPatch,
    encoder_tokens: int,
) -> None:
    attention = WhisperSGLangCrossAttention(_tiny_whisper_config(), layer_id=1)
    states = torch.randn(encoder_tokens, attention.embed_dim)
    cache_loc = torch.arange(encoder_tokens, dtype=torch.int64)
    with torch.no_grad():
        attention.kv_proj.weight.copy_(
            torch.arange(
                2 * attention.embed_dim * attention.embed_dim,
                dtype=states.dtype,
            ).reshape_as(attention.kv_proj.weight)
            / 100.0
        )
        attention.kv_proj.bias.copy_(
            torch.arange(2 * attention.embed_dim, dtype=states.dtype)
        )
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
    expected_key, expected_value = attention.kv_proj(states).chunk(2, dim=-1)
    expected_key = expected_key.view(-1, attention.num_heads, attention.head_dim)
    expected_value = expected_value.view(-1, attention.num_heads, attention.head_dim)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)
    assert key.dtype == states.dtype
    assert value.dtype == states.dtype
    assert key.device == states.device
    assert value.device == states.device


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


def test_whisper_mixed_encoder_batch_projects_only_uncached_requests_in_order(
    monkeypatch: pytest.MonkeyPatch,
    sglang_runtime_context,
) -> None:
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    attention = model.model.decoder.layers[0].encoder_attn
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_cached=[True, False, True, False],
        encoder_lens=torch.tensor([2, 2, 2, 2]),
        encoder_out_cache_loc=torch.tensor([41, 42, 73, 74], dtype=torch.int64),
        mm_inputs=[
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        feature=None,
                        precomputed_embeddings=torch.full((2, 8), -99.0),
                    )
                ]
            ),
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        feature=None,
                        precomputed_embeddings=torch.tensor([[11.0] * 8, [12.0] * 8]),
                    )
                ]
            ),
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        feature=None,
                        precomputed_embeddings=torch.full((2, 8), -77.0),
                    )
                ]
            ),
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        feature=None,
                        precomputed_embeddings=torch.tensor([[31.0] * 8, [32.0] * 8]),
                    )
                ]
            ),
        ],
    )

    states = model._batch_precomputed_encoder_states(batch)
    assert states is not None
    torch.testing.assert_close(
        states,
        torch.tensor([[11.0] * 8, [12.0] * 8, [31.0] * 8, [32.0] * 8]),
    )

    writes: list[tuple[object, object, torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(
        sglang_model,
        "get_attn_backend",
        lambda: SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                set_kv_buffer=lambda layer, loc, key, value: writes.append(
                    (layer, loc, key, value)
                )
            )
        ),
    )
    attention.cache_encoder_states(states, batch.encoder_out_cache_loc)

    assert len(writes) == 1
    layer, write_loc, key, value = writes[0]
    assert layer is attention.attn
    assert write_loc.loc is batch.encoder_out_cache_loc
    assert torch.equal(write_loc.loc, torch.tensor([41, 42, 73, 74]))
    expected_key, expected_value = attention.kv_proj(states).chunk(2, dim=-1)
    torch.testing.assert_close(
        key,
        expected_key.view(-1, attention.num_heads, attention.head_dim),
    )
    torch.testing.assert_close(
        value,
        expected_value.view(-1, attention.num_heads, attention.head_dim),
    )


def test_whisper_precomputed_encoder_states_are_cached_before_decoder_body(
    monkeypatch: pytest.MonkeyPatch,
    sglang_runtime_context,
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
        encoder_lens_cpu=[3],
        encoder_out_cache_loc=torch.arange(3),
        mm_inputs=[mm_input],
    )
    calls: list[str] = []
    for layer in model.model.decoder.layers:
        monkeypatch.setattr(
            layer.encoder_attn,
            "cache_encoder_states",
            lambda states, cache_loc: (
                calls.append("cache")
                if states.shape == encoder_states.shape
                and cache_loc is forward_batch.encoder_out_cache_loc
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

    assert calls == ["cache", "cache", "decoder", "logits"]


def test_whisper_encoder_in_prefill_fallback_uses_the_same_kv_cache_path(
    monkeypatch: pytest.MonkeyPatch,
    sglang_runtime_context,
) -> None:
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    encoder_states = torch.randn(1, 3, model.config.d_model)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_cached=[False],
        encoder_lens=torch.tensor([3]),
        encoder_lens_cpu=[3],
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
    for layer in model.model.decoder.layers:
        monkeypatch.setattr(
            layer.encoder_attn,
            "cache_encoder_states",
            lambda states, cache_loc: calls.append("cache"),
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

    assert calls == ["encoder", "cache", "cache", "decoder", "logits"]


def test_whisper_decode_capture_reuses_cached_encoder_kv(
    monkeypatch: pytest.MonkeyPatch,
    sglang_runtime_context,
) -> None:
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    input_ids = torch.tensor([1])
    positions = torch.tensor([0])
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
    decoder_skip_cross_attention: list[bool] = []
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda *args, **kwargs: decoder_skip_cross_attention.append(
            kwargs["skip_cross_attention"]
        )
        or torch.randn(input_ids.shape[0], model.config.d_model),
    )
    monkeypatch.setattr(
        model.logits_processor,
        "forward",
        lambda *args, **kwargs: object(),
    )

    with model_capture_mode():
        model(input_ids, positions, forward_batch)

    assert cache_calls == []
    assert decoder_skip_cross_attention == [False]


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
    original_attention_layers = [object()]
    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                decoder=SimpleNamespace(layers=decoder_layers),
            )
        ),
        attention_layers=original_attention_layers,
    )
    initialized: list[list[object]] = []

    def fake_init(self, runner):
        self.attention_layers = runner.attention_layers
        initialized.append(self.attention_layers)

    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "__init__",
        fake_init,
    )

    prefill_runner = WhisperPrefillCudaGraphRunner(model_runner)

    assert model_runner.attention_layers is original_attention_layers
    assert prefill_runner.attention_layers == (
        self_attention_layers + cross_attention_layers
    )
    assert initialized == [prefill_runner.attention_layers]


def test_whisper_prefill_replay_restores_only_omitted_encoder_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    base_encoder_lens = object()
    base_mm_inputs = object()
    static_batch = SimpleNamespace(
        encoder_lens=base_encoder_lens,
        mm_inputs=base_mm_inputs,
        encoder_lens_cpu=None,
        encoder_cached=None,
        encoder_out_cache_loc=None,
    )
    live_batch = SimpleNamespace(
        encoder_lens=torch.tensor([7, 5]),
        mm_inputs=[object(), object()],
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

    assert replay_batch.encoder_lens is base_encoder_lens
    assert replay_batch.mm_inputs is base_mm_inputs
    assert replay_batch.encoder_lens_cpu == live_batch.encoder_lens_cpu
    assert replay_batch.encoder_cached == live_batch.encoder_cached
    assert replay_batch.encoder_out_cache_loc is live_batch.encoder_out_cache_loc


@pytest.mark.parametrize("encoder_lens_cpu", [[], [0]])
def test_whisper_prefill_admission_rejects_empty_encoder_context(
    monkeypatch: pytest.MonkeyPatch,
    encoder_lens_cpu: list[int],
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    base_calls: list[object] = []
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "can_run_graph",
        lambda self, forward_batch: base_calls.append(forward_batch) or True,
    )

    assert (
        runner.can_run_graph(SimpleNamespace(encoder_lens_cpu=encoder_lens_cpu))
        is False
    )
    assert base_calls == []


def test_whisper_prefill_admission_delegates_normal_context_to_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    forward_batch = SimpleNamespace(encoder_lens_cpu=[1])
    base_calls: list[object] = []
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "can_run_graph",
        lambda self, batch: base_calls.append(batch) or True,
    )

    assert runner.can_run_graph(forward_batch) is True
    assert base_calls == [forward_batch]


def test_model_runner_restores_prefill_adapter_after_capture_failure() -> None:
    from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup

    runner = object.__new__(SGLModelRunner)
    runner._model_arch_override = "WhisperForConditionalGeneration"
    runner.server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=CudaGraphBackend.BREAKABLE)
        )
    )
    original = cuda_graph_setup.PrefillCudaGraphRunner

    with pytest.raises(RuntimeError, match="capture failed"):
        with runner._whisper_prefill_cuda_graph_runner_override():
            assert (
                cuda_graph_setup.PrefillCudaGraphRunner is WhisperPrefillCudaGraphRunner
            )
            raise RuntimeError("capture failed")

    assert cuda_graph_setup.PrefillCudaGraphRunner is original

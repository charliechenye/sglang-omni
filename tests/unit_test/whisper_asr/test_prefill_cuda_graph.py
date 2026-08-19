# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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


def test_whisper_prefill_admission_accepts_a_real_forward_batch() -> None:
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.tensor([1, 2]),
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([1, 1]),
        out_cache_loc=torch.tensor([5, 6]),
        seq_lens_sum=2,
        encoder_lens=torch.tensor([2, 2]),
        encoder_out_cache_loc=torch.tensor([41, 42]),
        encoder_lens_cpu=[2, 2],
        encoder_cached=[True, False],
        mm_inputs=[
            None,
            MultimodalInputs(
                mm_items=[
                    MultimodalDataItem(
                        modality=Modality.AUDIO,
                        precomputed_embeddings=torch.ones(2, 8),
                    )
                ]
            ),
        ],
    )

    assert WhisperPrefillCudaGraphRunner._encoder_metadata_is_usable(batch)


def test_whisper_prefill_admission_reads_encoder_lengths_from_host_mirror() -> None:
    batch = SimpleNamespace(
        batch_size=1,
        encoder_lens=torch.empty(1, dtype=torch.int64, device="meta"),
        encoder_lens_cpu=[4],
        encoder_cached=[True],
        encoder_out_cache_loc=None,
        mm_inputs=None,
    )

    assert WhisperPrefillCudaGraphRunner._encoder_metadata_is_usable(batch)


def test_whisper_model_exposes_decoder_body_without_duplicate_registration() -> None:
    model = WhisperModel(_tiny_whisper_config())

    assert model.layers is model.decoder.layers
    assert "input_embeds" in inspect.signature(model.forward).parameters
    assert not any(name.startswith("layers.") for name in model.state_dict())
    assert any(name.startswith("decoder.layers.") for name in model.state_dict())


def test_whisper_load_weights_maps_tied_projection_and_preserves_state_dict_names() -> (
    None
):
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    replacement = torch.randn_like(model.model.decoder.embed_tokens.weight)
    position_replacement = torch.randn_like(model.model.decoder.embed_positions.weight)

    model.load_weights(
        [
            ("proj_out.weight", replacement),
            ("model.decoder.embed_positions.weight", position_replacement),
        ]
    )

    torch.testing.assert_close(model.model.decoder.embed_tokens.weight, replacement)
    torch.testing.assert_close(
        model.model.decoder.embed_positions.weight, position_replacement
    )
    assert "proj_out.weight" not in model.state_dict()


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


@pytest.mark.parametrize("encoder_tokens", [1, 3])
def test_whisper_cross_attention_caches_encoder_kv_eagerly(
    monkeypatch: pytest.MonkeyPatch,
    encoder_tokens: int,
) -> None:
    attention = WhisperSGLangCrossAttention(_tiny_whisper_config(), layer_id=1)
    states = torch.randn(encoder_tokens, attention.embed_dim)
    cache_loc = torch.arange(encoder_tokens, dtype=torch.int64)
    with torch.no_grad():
        attention.k_proj.weight.copy_(
            torch.arange(
                attention.embed_dim * attention.embed_dim,
                dtype=states.dtype,
            ).reshape_as(attention.k_proj.weight)
            / 100.0
        )
        attention.v_proj.weight.copy_(torch.flip(attention.k_proj.weight, dims=[0]))
        attention.v_proj.bias.copy_(
            torch.arange(attention.embed_dim, dtype=states.dtype)
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
    expected_key = attention.k_proj(states).view(
        -1, attention.num_heads, attention.head_dim
    )
    expected_value = attention.v_proj(states).view(
        -1, attention.num_heads, attention.head_dim
    )
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)
    assert key.dtype == states.dtype
    assert value.dtype == states.dtype
    assert key.device == states.device
    assert value.device == states.device


@pytest.mark.parametrize(
    ("states", "cache_loc", "error"),
    [
        (torch.empty(0, 8), torch.empty(0, dtype=torch.int64), "non-empty"),
        (torch.ones(2, 8), torch.ones(2, 1, dtype=torch.int64), "flat"),
        (torch.ones(2, 8), torch.ones(1, dtype=torch.int64), "length"),
        (torch.ones(1, 2, 8), torch.ones(2, dtype=torch.int64), "flat"),
    ],
)
def test_whisper_cross_attention_rejects_invalid_kv_writes(
    states: torch.Tensor,
    cache_loc: torch.Tensor,
    error: str,
) -> None:
    attention = WhisperSGLangCrossAttention(_tiny_whisper_config(), layer_id=1)

    with pytest.raises((RuntimeError, ValueError), match=error):
        attention.cache_encoder_states(states, cache_loc)


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


def test_whisper_cross_attention_signature_and_body_never_project_encoder_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert list(inspect.signature(WhisperSGLangCrossAttention.forward).parameters) == [
        "self",
        "hidden_states",
        "forward_batch",
    ]
    attention = WhisperSGLangCrossAttention(_tiny_whisper_config(), layer_id=1)

    def fail_projection(*args, **kwargs):
        del args, kwargs
        raise AssertionError("decoder cross-attention body projected encoder K/V")

    monkeypatch.setattr(attention.k_proj, "forward", fail_projection)
    monkeypatch.setattr(attention.v_proj, "forward", fail_projection)

    class _CachedAttention(torch.nn.Module):
        def forward(self, query, key, value, forward_batch):
            del key, value, forward_batch
            return torch.zeros_like(query)

    attention.attn = _CachedAttention()
    attention.out_proj = torch.nn.Identity()
    output = attention(torch.randn(2, attention.embed_dim), SimpleNamespace())

    assert output.shape == (2, attention.embed_dim)


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


def test_whisper_mixed_encoder_batch_projects_only_uncached_requests_in_order(
    monkeypatch: pytest.MonkeyPatch,
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
    torch.testing.assert_close(
        key,
        attention.k_proj(states).view(-1, attention.num_heads, attention.head_dim),
    )
    torch.testing.assert_close(
        value,
        attention.v_proj(states).view(-1, attention.num_heads, attention.head_dim),
    )


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
    original_attention_layers = [object()]
    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                decoder=SimpleNamespace(layers=decoder_layers),
            )
        ),
        attention_layers=original_attention_layers,
    )

    original_mha_layers = [None, None]
    original_moe_layers = [None, None]
    original_moe_fusions = [None, None]
    original_dsa_indexers = [None, None]
    model_runner.mha_companion_layers = original_mha_layers
    model_runner.moe_layers = original_moe_layers
    model_runner.moe_fusions = original_moe_fusions
    model_runner.dsa_indexers = original_dsa_indexers
    initialized: list[
        tuple[
            object, list[object], list[object], list[object], list[object], list[object]
        ]
    ] = []

    def fake_init(self, runner):
        self.attention_layers = runner.attention_layers
        self.mha_companion_layers = runner.mha_companion_layers
        self.moe_layers = runner.moe_layers
        self.moe_fusions = runner.moe_fusions
        self.dsa_indexers = runner.dsa_indexers
        initialized.append(
            (
                runner,
                self.attention_layers,
                self.mha_companion_layers,
                self.moe_layers,
                self.moe_fusions,
                self.dsa_indexers,
            )
        )

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
    assert len({id(layer) for layer in prefill_runner.attention_layers}) == 4
    assert initialized == [
        (
            model_runner,
            prefill_runner.attention_layers,
            original_mha_layers,
            original_moe_layers,
            original_moe_fusions,
            original_dsa_indexers,
        )
    ]


def test_whisper_prefill_runner_restores_attention_layers_on_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_attention_layers = [object()]
    decoder_layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(attn=object()),
            encoder_attn=SimpleNamespace(attn=object()),
        )
    ]
    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(decoder=SimpleNamespace(layers=decoder_layers))
        ),
        attention_layers=original_attention_layers,
    )

    def fail_init(self, runner):
        del self, runner
        raise RuntimeError("prefill runner init failed")

    monkeypatch.setattr(WhisperPrefillCudaGraphRunner.__mro__[1], "__init__", fail_init)

    with pytest.raises(RuntimeError, match="prefill runner init failed"):
        WhisperPrefillCudaGraphRunner(model_runner)

    assert model_runner.attention_layers is original_attention_layers


@pytest.mark.parametrize(
    ("encoder_lens", "encoder_cached", "cache_loc", "has_inputs", "expected"),
    [
        ([4], [True], None, False, True),
        ([0], [True], None, False, False),
        ([0], [False], None, False, False),
        ([4], [False], None, True, False),
        ([4], [False], torch.arange(4), True, True),
        ([4], [False], torch.arange(4), False, False),
        ([4, 4], [True, False], torch.arange(4), True, True),
        ([4, 4], [True, False], torch.arange(3), True, False),
        ([4, 4], [True, False], torch.arange(4).reshape(2, 2), True, False),
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


@pytest.mark.parametrize(
    ("batch", "base_result"),
    [
        (
            _encoder_batch(
                encoder_lens=[4],
                encoder_cached=[True],
                encoder_out_cache_loc=None,
                has_inputs=False,
            ),
            True,
        ),
        (
            _encoder_batch(
                encoder_lens=[4],
                encoder_cached=[False],
                encoder_out_cache_loc=torch.arange(4),
                has_inputs=True,
            ),
            True,
        ),
        (
            _encoder_batch(
                encoder_lens=[4],
                encoder_cached=[False],
                encoder_out_cache_loc=None,
                has_inputs=True,
            ),
            False,
        ),
        (
            _encoder_batch(
                encoder_lens=[0],
                encoder_cached=[True],
                encoder_out_cache_loc=None,
                has_inputs=False,
            ),
            False,
        ),
    ],
)
def test_whisper_prefill_can_run_graph_delegates_to_base_after_admission(
    monkeypatch: pytest.MonkeyPatch,
    batch: SimpleNamespace,
    base_result: bool,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "can_run_graph",
        lambda self, forward_batch: base_result,
    )

    assert runner.can_run_graph(batch) is base_result


def test_whisper_prefill_admission_keeps_base_runner_fallback_for_large_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    batch = _encoder_batch(
        encoder_lens=[4],
        encoder_cached=[True],
        encoder_out_cache_loc=None,
        has_inputs=False,
    )
    batch.input_ids = torch.zeros(257, dtype=torch.int64)
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "can_run_graph",
        lambda self, forward_batch: len(forward_batch.input_ids) <= 256,
    )

    assert runner.can_run_graph(batch) is False


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

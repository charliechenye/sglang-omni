import gc
import logging
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_mode import ForwardMode
from torch import nn

from sglang_omni.models.qwen3_asr import sglang_model
from sglang_omni.models.qwen3_asr.sglang_model import Qwen3ASRForConditionalGeneration


class RecordingAudioTower(nn.Module):
    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0))
        self.seen_features: torch.Tensor | None = None
        self.seen_lengths: torch.Tensor | None = None

    def forward(
        self,
        input_features: torch.Tensor,
        *,
        feature_lens: torch.Tensor,
    ) -> SimpleNamespace:
        self.seen_features = input_features
        self.seen_lengths = feature_lens
        return SimpleNamespace(last_hidden_state=input_features)


class AudioFeatureModel(Qwen3ASRForConditionalGeneration):
    def __init__(self, audio_tower: RecordingAudioTower) -> None:
        nn.Module.__init__(self)
        self.audio_tower = audio_tower
        self.encoder_graph_runner = None


class EventHandoffModel(Qwen3ASRForConditionalGeneration):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.language_model = nn.Identity()
        self.encoder_graph_runner = None
        self.pre_lm_ready_event_lock = threading.Lock()
        self.pre_lm_ready_events: dict[
            int,
            tuple[
                weakref.ReferenceType[torch.Tensor],
                torch.cuda.Event,
            ],
        ] = {}
        self.pre_lm_event_marker_emitted = False
        self.pre_lm_event_handoff_enabled = True


class RecordingStream:
    def __init__(self) -> None:
        self.waited_events: list[torch.cuda.Event] = []

    def wait_event(self, event: torch.cuda.Event) -> None:
        self.waited_events.append(event)


class RecordingEvent:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


@pytest.mark.parametrize("use_mrope_positions", [True, False])
def test_forward_uses_available_mrope_positions(
    monkeypatch: pytest.MonkeyPatch,
    use_mrope_positions: bool,
) -> None:
    monkeypatch.setattr(sglang_model, "fused_qk_norm_rope", lambda *args: None)
    positions = torch.tensor([4, 5])
    mrope_positions = torch.tensor([[4, 5], [4, 5], [4, 5]])
    forward_batch = SimpleNamespace(
        mrope_positions=mrope_positions if use_mrope_positions else None
    )
    output = torch.tensor([1.0])
    seen_positions = None

    def fake_general_mm_embed_routine(**kwargs):
        nonlocal seen_positions
        seen_positions = kwargs["positions"]
        return output

    monkeypatch.setattr(
        sglang_model,
        "general_mm_embed_routine",
        fake_general_mm_embed_routine,
    )
    model = SimpleNamespace(
        language_model=object(),
        get_audio_feature=object(),
        wait_for_pre_lm_embeddings=lambda _batch: None,
    )

    actual = Qwen3ASRForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([1, 2]),
        positions=positions,
        forward_batch=forward_batch,
    )

    expected_positions = mrope_positions[0] if use_mrope_positions else positions
    assert torch.equal(seen_positions, expected_positions)
    assert seen_positions.dtype == torch.int32
    assert actual is output


def test_forward_keeps_long_positions_for_native_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_model, "fused_qk_norm_rope", None)
    seen_positions = None

    def fake_general_mm_embed_routine(**kwargs):
        nonlocal seen_positions
        seen_positions = kwargs["positions"]
        return torch.tensor([1.0])

    monkeypatch.setattr(
        sglang_model,
        "general_mm_embed_routine",
        fake_general_mm_embed_routine,
    )
    model = SimpleNamespace(
        language_model=object(),
        get_audio_feature=object(),
        wait_for_pre_lm_embeddings=lambda _batch: None,
    )

    Qwen3ASRForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([4, 5], dtype=torch.int32),
        forward_batch=SimpleNamespace(mrope_positions=None),
    )

    assert seen_positions.dtype == torch.long


def test_event_handoff_waits_for_every_producer_before_multimodal_consumption(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    stream = RecordingStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(sglang_model, "fused_qk_norm_rope", None)
    model = EventHandoffModel()
    first_embedding = torch.ones((2, 4))
    second_embedding = torch.ones((3, 4))
    repeated_batch_embedding = torch.ones((1, 4))
    first_event = RecordingEvent()
    second_event = RecordingEvent()
    model.register_pre_lm_ready_event(first_embedding, first_event)
    model.register_pre_lm_ready_event(second_embedding, second_event)
    model.register_pre_lm_ready_event(repeated_batch_embedding, first_event)
    item_values = [
        None,
        SimpleNamespace(precomputed_embeddings=first_embedding),
        SimpleNamespace(precomputed_embeddings=second_embedding),
        SimpleNamespace(precomputed_embeddings=repeated_batch_embedding),
    ]
    forward_batch = SimpleNamespace(
        mm_inputs=[SimpleNamespace(mm_items=item_values)],
        mrope_positions=None,
        forward_mode=ForwardMode.EXTEND,
    )
    expected_output = torch.zeros(1)

    def consume_embeddings(**_kwargs):  # noqa: ANN003
        assert stream.waited_events == [first_event, second_event]
        return expected_output

    monkeypatch.setattr(
        sglang_model,
        "general_mm_embed_routine",
        consume_embeddings,
    )

    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        forward_batch=forward_batch,
    )

    assert output is expected_output
    assert "QWEN3_ASR_P5_EVENT_HANDOFF_ENABLED" in caplog.text


def test_event_handoff_graph_capture_falls_back_to_host_wait(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    stream = RecordingStream()
    event = RecordingEvent()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    model = EventHandoffModel()
    embedding = torch.ones((2, 4))
    model.register_pre_lm_ready_event(embedding, event)
    forward_batch = SimpleNamespace(
        mm_inputs=[
            SimpleNamespace(
                mm_items=[SimpleNamespace(precomputed_embeddings=embedding)]
            )
        ],
        mrope_positions=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
    )

    model.wait_for_pre_lm_embeddings(forward_batch)

    assert stream.waited_events == []
    assert event.synchronize_calls == 1
    assert not model.pre_lm_event_marker_emitted
    assert "QWEN3_ASR_P5_EVENT_HANDOFF_ENABLED" not in caplog.text


def test_event_handoff_skips_decode_only_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = RecordingStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    model = EventHandoffModel()
    embedding = torch.ones((2, 4))
    model.register_pre_lm_ready_event(embedding, RecordingEvent())
    forward_batch = SimpleNamespace(
        mm_inputs=[
            SimpleNamespace(
                mm_items=[SimpleNamespace(precomputed_embeddings=embedding)]
            )
        ],
        mrope_positions=None,
        forward_mode=ForwardMode.DECODE,
    )

    model.wait_for_pre_lm_embeddings(forward_batch)

    assert stream.waited_events == []
    assert not model.pre_lm_event_marker_emitted


def test_event_registry_releases_events_with_their_output_tensor() -> None:
    model = EventHandoffModel()
    embedding = torch.ones((2, 4))
    embedding_identity = id(embedding)
    model.register_pre_lm_ready_event(embedding, RecordingEvent())

    del embedding
    gc.collect()

    assert embedding_identity not in model.pre_lm_ready_events


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_event_handoff_orders_streams_and_retains_the_output_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingCudaStream(torch.cuda.Stream):
        def __init__(self, device: torch.device) -> None:
            super().__init__(device=device)
            self.waited_events: list[torch.cuda.Event] = []

        def wait_event(self, event: torch.cuda.Event) -> None:
            self.waited_events.append(event)
            super().wait_event(event)

    model = EventHandoffModel()
    output = torch.empty((64,), device="cuda")
    output_reference = weakref.ref(output)
    producer_stream = torch.cuda.Stream(device=output.device)
    consumer_stream = RecordingCudaStream(device=output.device)
    producer_event = torch.cuda.Event()
    with torch.cuda.stream(producer_stream):
        output.fill_(3)
        producer_event.record(producer_stream)
    model.register_pre_lm_ready_event(output, producer_event)
    multimodal_item = SimpleNamespace(precomputed_embeddings=output)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mm_inputs=[SimpleNamespace(mm_items=[multimodal_item])],
    )

    with torch.cuda.stream(consumer_stream):
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda _device=None: consumer_stream,
        )
        model.wait_for_pre_lm_embeddings(forward_batch)
        consumed_output = output.clone()
        del output, multimodal_item, forward_batch
    gc.collect()

    assert consumer_stream.waited_events == [producer_event]
    assert output_reference() is None
    assert model.pre_lm_ready_events == {}
    consumer_stream.synchronize()
    assert torch.equal(consumed_output.cpu(), torch.full((64,), 3.0))


def test_fused_asr_qk_norm_rope_is_bound_per_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def original(positions, hidden_states):
        return positions, hidden_states

    supported = SimpleNamespace(
        head_dim=128,
        forward_prepare_native=original,
    )
    unsupported = SimpleNamespace(
        head_dim=96,
        forward_prepare_native=original,
    )
    language_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=supported),
                SimpleNamespace(self_attn=unsupported),
            ]
        )
    )
    monkeypatch.setattr(sglang_model, "fused_qk_norm_rope", lambda *args: None)

    sglang_model.enable_fused_asr_qk_norm_rope(language_model)

    assert (
        supported._asr_unfused_forward_prepare_native is original
    )  # noqa: leading-underscore  # production name
    assert (
        supported.forward_prepare_native.__func__
        is sglang_model.fused_asr_forward_prepare_native
    )
    assert unsupported.forward_prepare_native is original


def test_fused_asr_qk_norm_rope_is_not_bound_without_platform_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def original(positions, hidden_states):
        return positions, hidden_states

    attention = SimpleNamespace(
        head_dim=128,
        forward_prepare_native=original,
    )
    language_model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    )
    monkeypatch.setattr(sglang_model, "fused_qk_norm_rope", None)

    sglang_model.enable_fused_asr_qk_norm_rope(language_model)

    assert attention.forward_prepare_native is original
    assert not hasattr(attention, "_asr_unfused_forward_prepare_native")


def test_fused_asr_qk_norm_rope_falls_back_before_projection() -> None:
    expected = (object(), object(), object())
    attention = SimpleNamespace(
        qkv_proj=lambda hidden_states: pytest.fail(
            f"unexpected projection for {hidden_states.dtype}"
        ),
        _asr_unfused_forward_prepare_native=lambda positions, hidden_states: expected,
    )

    actual = sglang_model.fused_asr_forward_prepare_native(
        attention,
        torch.tensor([0], dtype=torch.int32),
        torch.zeros((1, 1, 8), dtype=torch.float16),
    )

    assert actual is expected


def test_get_audio_feature_preserves_masks_in_mixed_batch() -> None:
    tower = RecordingAudioTower()
    model = AudioFeatureModel(tower)
    items = [
        SimpleNamespace(
            feature=torch.tensor([[[1.0, 2.0, 90.0, 91.0]]]),
            feature_attention_mask=torch.tensor([[1, 1, 0, 0]]),
        ),
        SimpleNamespace(
            feature=torch.tensor([[[3.0, 4.0]]]),
            feature_attention_mask=None,
        ),
    ]

    output = Qwen3ASRForConditionalGeneration.get_audio_feature(model, items)

    assert torch.equal(tower.seen_lengths, torch.tensor([2, 2]))
    assert torch.equal(tower.seen_features, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(output, tower.seen_features)


def test_get_audio_feature_rejects_mismatched_mask_shape() -> None:
    model = AudioFeatureModel(RecordingAudioTower())
    items = [
        SimpleNamespace(
            feature=torch.ones((1, 2, 4)),
            feature_attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
    ]

    with pytest.raises(ValueError, match="feature_attention_mask shape"):
        Qwen3ASRForConditionalGeneration.get_audio_feature(model, items)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_get_audio_feature_normalizes_cpu_masks_for_cuda_features() -> None:
    tower = RecordingAudioTower().cuda()
    model = AudioFeatureModel(tower)
    items = [
        SimpleNamespace(
            feature=torch.tensor([[[1.0, 2.0, 90.0, 91.0]]], device="cuda"),
            feature_attention_mask=torch.tensor([[1, 1, 0, 0]]),
        ),
        SimpleNamespace(
            feature=torch.tensor([[[3.0, 4.0]]], device="cuda"),
            feature_attention_mask=None,
        ),
    ]

    Qwen3ASRForConditionalGeneration.get_audio_feature(model, items)

    assert tower.seen_lengths is not None
    assert tower.seen_lengths.device.type == "cuda"
    assert torch.equal(tower.seen_lengths.cpu(), torch.tensor([2, 2]))
    assert tower.seen_features is not None
    assert torch.equal(
        tower.seen_features.cpu(),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )

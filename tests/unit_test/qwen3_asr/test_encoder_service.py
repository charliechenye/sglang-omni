# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import contextlib
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import msgspec
import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang_omni.models.qwen3_asr import encoder_service
from sglang_omni.models.qwen3_asr.encoder_service import (
    Qwen3ASRPreLMEncoderService,
    build_cache_namespace,
    expected_audio_tokens,
)
from sglang_omni.scheduling.pre_lm_encoder import QueueEntry

HIDDEN_SIZE = 4
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"
requires_accelerator = pytest.mark.skipif(
    DEVICE == "cpu",
    reason="requires cuda or xpu",
)
NAMESPACE = "testns"
SERVICES: list[Qwen3ASRPreLMEncoderService] = []


@pytest.fixture(autouse=True)
def close_services() -> Iterator[None]:
    yield
    for service in SERVICES:
        service.close()
    SERVICES.clear()


class StubModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(2, 2).to(dtype)
        self.config = SimpleNamespace(
            thinker_config=SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=HIDDEN_SIZE)
            )
        )
        self.dtype = dtype
        self.encode_calls = 0
        self.encode_batch_sizes: list[int] = []
        self.fail = False
        self.fail_oom = False
        self.fail_multi_item = False
        self.packed_3d_output = True
        self.encode_gate: threading.Event | None = None
        self.encode_started: threading.Event | None = None
        self.row_offset = 0
        self.encode_delay_s = 0.0
        self.grad_enabled_during_encode: bool | None = None
        self.pre_lm_events: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self.encoder_graph_runner = None

    def get_audio_feature(self, items):  # noqa: ANN001
        self.grad_enabled_during_encode = torch.is_grad_enabled()
        self.encode_calls += 1
        self.encode_batch_sizes.append(len(items))
        if self.encode_started is not None:
            self.encode_started.set()
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            gate.wait(timeout=10)
        if self.encode_delay_s:
            time.sleep(self.encode_delay_s)
        if self.fail_oom:
            raise torch.OutOfMemoryError("encoder OOM")
        if self.fail:
            raise RuntimeError("boom")
        if self.fail_multi_item and len(items) > 1:
            raise RuntimeError("multi-item boom")
        parts = []
        for item in items:
            rows = expected_audio_tokens(item) + self.row_offset
            fill = float((getattr(item, "hash", None) or 0) % 97 + 1)
            parts.append(torch.full((rows, HIDDEN_SIZE), fill, dtype=self.dtype))
        packed = torch.cat(parts, dim=0)
        if self.packed_3d_output:
            # Mirrors the real audio tower: one packed frame stream in, so
            # last_hidden_state is [1, total_tokens, hidden].
            return packed.unsqueeze(0)
        return packed

    def register_pre_lm_ready_event(
        self, embedding: torch.Tensor, event: torch.cuda.Event
    ) -> None:
        self.pre_lm_events[id(embedding)] = (embedding, event)

    def remove_pre_lm_ready_event(self, embedding: torch.Tensor) -> None:
        self.pre_lm_events.pop(id(embedding), None)


def make_service(
    model: StubModel | None = None,
    *,
    cache_max_entries: int = 16,
    cache_max_bytes: int = 1 << 20,
    max_batch_size: int = 8,
    enable_event_handoff: bool = False,
) -> Qwen3ASRPreLMEncoderService:
    service = Qwen3ASRPreLMEncoderService(
        model or StubModel(),
        cache_namespace=NAMESPACE,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
        max_batch_size=max_batch_size,
        enable_event_handoff=enable_event_handoff,
    )
    SERVICES.append(service)
    return service


class RecordingStream:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class RecordingEvent:
    def __init__(self) -> None:
        self.recorded_stream: RecordingStream | None = None
        self.synchronize_calls = 0

    def record(self, stream: RecordingStream) -> None:
        self.recorded_stream = stream

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class FakeCudaParameter:
    device: torch.device = torch.device("cuda:0")
    dtype: torch.dtype = torch.float32


class FakeAudioTower(torch.nn.Module):
    def parameters(self, recurse: bool = True) -> Iterator[FakeCudaParameter]:
        del recurse
        yield FakeCudaParameter()


def test_event_handoff_records_after_attach_and_keeps_events_out_of_item_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubModel()
    service = make_service(model)
    service.enable_event_handoff = True
    stream = RecordingStream()
    event = RecordingEvent()
    service.stream = stream
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    item = make_item(7, 3)
    embedding = torch.ones((3, HIDDEN_SIZE))

    item.precomputed_embeddings = embedding
    service.after_batch_embeddings_attached([item], [embedding])
    service.synchronize_batch()
    encoded_metadata = msgspec.msgpack.encode(item.model_specific_data)

    assert event.recorded_stream is stream
    assert model.pre_lm_events[id(embedding)] == (embedding, event)
    assert stream.synchronize_calls == 0
    assert "producer_event" not in item.model_specific_data
    assert all(value is not event for value in item.model_specific_data.values())
    assert b"producer_event" not in encoded_metadata


def test_event_handoff_disabled_keeps_encoder_stream_synchronization() -> None:
    service = make_service()
    stream = RecordingStream()
    service.stream = stream

    service.synchronize_batch()

    assert not service.enable_event_handoff
    assert stream.synchronize_calls == 1


def test_event_handoff_falls_back_when_encoder_graphs_are_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubModel()
    model.audio_tower = FakeAudioTower()
    model.encoder_graph_runner = object()
    monkeypatch.setattr(
        encoder_service,
        "create_device_stream",
        lambda _device: RecordingStream(),
    )

    service = Qwen3ASRPreLMEncoderService(
        model,
        cache_namespace=NAMESPACE,
        enable_event_handoff=True,
    )
    SERVICES.append(service)

    assert not service.enable_event_handoff
    assert not model.pre_lm_event_handoff_enabled


def test_event_handoff_defers_cpu_cache_until_producer_stream_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_service()
    service.enable_event_handoff = True
    stream = RecordingStream()
    service.stream = stream
    item = make_item(81, 3)
    embedding = torch.ones((3, HIDDEN_SIZE))
    cache_key = service.cache_key(item)
    assert cache_key is not None
    future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
    queue_entry = QueueEntry(item=item, future=future)
    service.inflight[cache_key] = future
    future.add_done_callback(
        lambda completed: service.clear_inflight(cache_key, completed)
    )
    original_put = service.cache.put

    def put_after_stream_sync(cache_key: str, cache_value: torch.Tensor) -> None:
        assert stream.synchronize_calls == 1
        original_put(cache_key, cache_value)

    monkeypatch.setattr(service.cache, "put", put_after_stream_sync)

    service.cache_embedding(item, embedding)

    assert service.cache.get(cache_key) is None
    future.set_result(embedding)
    assert service.inflight[cache_key] is future
    service.on_batch_finished([queue_entry], None, None, 0.0)

    cached = service.cache.get(cache_key)
    assert cached is not None
    assert torch.equal(cached, embedding)
    assert stream.synchronize_calls == 1
    assert cache_key not in service.inflight


@pytest.mark.parametrize("cancelled", [True, False])
def test_cancelled_or_failed_event_handoff_clears_pending_cache_state(
    cancelled: bool,
) -> None:
    model = StubModel()
    service = make_service(model)
    service.enable_event_handoff = True
    stream = RecordingStream()
    service.stream = stream
    item = make_item(82, 3)
    embedding = torch.ones((3, HIDDEN_SIZE))
    event = RecordingEvent()
    model.register_pre_lm_ready_event(embedding, event)
    item.precomputed_embeddings = embedding
    cache_key = service.cache_key(item)
    assert cache_key is not None
    future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
    queue_entry = QueueEntry(item=item, future=future)
    service.inflight[cache_key] = future
    future.add_done_callback(
        lambda completed: service.clear_inflight(cache_key, completed)
    )
    service.cache_embedding(item, embedding)

    failure: Exception | None = None
    if cancelled:
        assert future.cancel()
    else:
        failure = RuntimeError("encode failed")
        future.set_exception(failure)
    service.on_batch_finished([queue_entry], failure, None, 0.0)

    assert service.cache.get(cache_key) is None
    assert cache_key not in service.inflight
    assert service.pending_cache_publications == {}
    assert model.pre_lm_events == {}
    assert stream.synchronize_calls == 0


def test_cache_publication_failure_clears_pending_cache_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_service()
    service.enable_event_handoff = True
    stream = RecordingStream()
    service.stream = stream
    item = make_item(83, 3)
    embedding = torch.ones((3, HIDDEN_SIZE))
    cache_key = service.cache_key(item)
    assert cache_key is not None
    future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
    queue_entry = QueueEntry(item=item, future=future)
    service.inflight[cache_key] = future
    future.add_done_callback(
        lambda completed: service.clear_inflight(cache_key, completed)
    )
    service.cache_embedding(item, embedding)

    def fail_cache_put(cache_key: str, cache_value: torch.Tensor) -> None:
        raise RuntimeError(
            f"cache publication failed for {cache_key} with {tuple(cache_value.shape)}"
        )

    monkeypatch.setattr(service.cache, "put", fail_cache_put)
    future.set_result(embedding)
    service.on_batch_finished([queue_entry], None, None, 0.0)

    assert service.cache.get(cache_key) is None
    assert cache_key not in service.inflight
    assert service.pending_cache_publications == {}
    assert stream.synchronize_calls == 1


def make_item(
    audio_hash: int | None,
    num_audio_tokens: int,
    *,
    with_feature: bool = True,
) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=audio_hash,
        feature=torch.zeros(1, 128, 300) if with_feature else None,
        model_specific_data={
            "audio_fingerprint": str(audio_hash) if audio_hash is not None else None,
            "num_audio_tokens": num_audio_tokens,
        },
    )


def test_encode_attaches_lm_ready_embedding_and_clears_feature() -> None:
    model = StubModel()
    service = make_service(model)
    item = make_item(7, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)
    assert item.precomputed_embeddings.dtype == model.dtype
    assert (
        item.precomputed_embeddings.device
        == next(model.audio_tower.parameters()).device
    )
    assert item.feature is None
    assert item.format.name == "PRECOMPUTED_EMBEDDING"
    assert model.encode_calls == 1
    assert model.grad_enabled_during_encode is False
    assert service.stats()["misses"] == 1


def test_submit_returns_before_encoding_completes() -> None:
    model = StubModel()
    gate = threading.Event()
    model.encode_gate = gate
    service = make_service(model)
    item = make_item(7, 3)

    future = service.submit_item(item)

    assert not future.done()
    assert item.precomputed_embeddings is None
    gate.set()
    future.result(timeout=2)
    assert item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)


def test_async_submissions_form_full_batch_without_blocked_callers() -> None:
    model = StubModel()
    gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = gate
    model.encode_started = encode_started
    service = make_service(model, max_batch_size=8)
    items = [make_item(audio_hash, 3) for audio_hash in range(9)]

    futures = [service.submit_item(items[0])]
    assert encode_started.wait(timeout=2)
    assert model.encode_calls == 1
    futures.extend(service.submit_item(item) for item in items[1:])
    gate.set()
    for future in futures:
        future.result(timeout=2)

    assert model.encode_batch_sizes == [1, 8]


def test_async_single_flight_completes_each_item_future() -> None:
    model = StubModel()
    gate = threading.Event()
    model.encode_gate = gate
    service = make_service(model)
    items = [make_item(123, 3) for _ in range(3)]

    futures = [service.submit_item(item) for item in items]
    assert all(not future.done() for future in futures)
    gate.set()
    for future in futures:
        future.result(timeout=2)

    assert model.encode_calls == 1
    assert service.stats()["merged"] == 2
    assert all(item.precomputed_embeddings is not None for item in items)


def test_close_stops_worker() -> None:
    service = make_service()

    service.close()

    assert not service.thread.is_alive()


def test_batch_context_unwinds_inference_mode_when_stream_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(Qwen3ASRPreLMEncoderService)
    service.stream = SimpleNamespace(device=torch.device("cuda", 0))

    class FakeDeviceModule:
        def __init__(self) -> None:
            self.stream_calls: list[object] = []

        @contextlib.contextmanager
        def stream(self, stream):  # noqa: ANN001, ANN202
            self.stream_calls.append(stream)
            raise RuntimeError("stream context failed")
            yield

    device_module = FakeDeviceModule()
    monkeypatch.setattr(torch, "get_device_module", lambda _device=None: device_module)

    assert not torch.is_inference_mode_enabled()
    with pytest.raises(RuntimeError, match="stream context failed"):
        with service.batch_context():
            pass
    assert not torch.is_inference_mode_enabled()
    assert device_module.stream_calls == [service.stream]


def test_cache_hit_skips_reencode() -> None:
    model = StubModel()
    service = make_service(model)

    first = make_item(11, 3)
    second = make_item(11, 3)
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 1
    assert torch.equal(first.precomputed_embeddings, second.precomputed_embeddings)
    assert second.feature is None
    assert service.stats()["hits"] == 1


def test_lookup_cached_embedding_returns_only_valid_entries() -> None:
    model = StubModel()
    service = make_service(model)
    item = make_item(11, 3)
    service.encode_item(item)

    cached = service.lookup_cached_embedding(item.audio_fingerprint, 3)

    assert cached is not None
    assert torch.equal(cached, item.precomputed_embeddings.cpu())
    assert service.stats()["hits"] == 1

    assert service.lookup_cached_embedding(item.audio_fingerprint, 4) is None
    assert len(service.cache) == 0


def test_cache_key_prefers_full_waveform_fingerprint() -> None:
    model = StubModel()
    service = make_service(model)
    first = make_item(7, 3)
    second = make_item(7, 3)
    first.audio_fingerprint = "full-hash-a"
    second.audio_fingerprint = "full-hash-b"

    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert len(service.cache) == 2


def test_stale_cache_miss_rechecks_before_starting_duplicate_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubModel()
    service = make_service(model)
    stale_miss = threading.Event()
    release_stale_reader = threading.Event()
    original_get = service.cache.get

    def controlled_get(key: str | None):  # noqa: ANN202
        cached = original_get(key)
        if (
            threading.current_thread().name == "stale-cache-reader"
            and not stale_miss.is_set()
        ):
            assert cached is None
            stale_miss.set()
            assert release_stale_reader.wait(timeout=10)
        return cached

    monkeypatch.setattr(service.cache, "get", controlled_get)
    follower_item = make_item(123, 3)
    errors: list[Exception] = []

    def follower() -> None:
        try:
            service.encode_item(follower_item)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=follower, name="stale-cache-reader")
    thread.start()
    assert stale_miss.wait(timeout=10)

    leader_item = make_item(123, 3)
    service.encode_item(leader_item)
    release_stale_reader.set()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert not errors, errors
    assert model.encode_calls == 1
    assert torch.equal(
        leader_item.precomputed_embeddings,
        follower_item.precomputed_embeddings,
    )
    assert service.stats()["hits"] == 1


def test_concurrent_identical_requests_deduplicate_without_cache() -> None:
    model = StubModel()
    gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = gate
    model.encode_started = encode_started
    service = make_service(model, cache_max_entries=0)
    items = [make_item(123, 3) for _ in range(2)]
    errors: list[Exception] = []

    def worker(item: MultimodalDataItem) -> None:
        try:
            service.encode_item(item)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    threads[0].start()
    assert encode_started.wait(timeout=5)
    threads[1].start()
    deadline = time.monotonic() + 5
    while service.stats()["merged"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.stats()["merged"] == 1
    gate.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    assert len(service.cache) == 0
    assert torch.equal(items[0].precomputed_embeddings, items[1].precomputed_embeddings)


def test_submit_item_failure_counts_failed() -> None:
    model = StubModel()
    model.fail = True
    service = make_service(model)

    future = service.submit_item(make_item(88, 3))

    with pytest.raises(RuntimeError, match="boom"):
        future.result(timeout=2)
    assert service.stats()["failed"] == 1


def test_encode_failure_propagates_without_poisoning_cache() -> None:
    model = StubModel()
    model.fail = True
    model.encode_delay_s = 0.05
    service = make_service(model)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(make_item(55, 3))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(errors) == 2
    assert all(isinstance(exc, RuntimeError) and "boom" in str(exc) for exc in errors)
    assert len(service.cache) == 0
    assert service.stats()["failed"] == 2

    model.fail = False
    item = make_item(55, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)


def test_oom_failure_detaches_traceback_and_recovers() -> None:
    model = StubModel()
    model.fail_oom = True
    service = make_service(model)

    with pytest.raises(torch.OutOfMemoryError, match="encoder OOM") as excinfo:
        service.encode_item(make_item(77, 3))

    # note (luojiaxuan): the propagated exception must not pin encoder frames.
    assert excinfo.value.__traceback__ is not None  # pytest's own raise site
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert service.stats()["failed"] == 1

    model.fail_oom = False
    item = make_item(77, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)


def test_merged_follower_token_mismatch_raises_and_counts_failed() -> None:
    model = StubModel()
    model.encode_delay_s = 0.2
    service = make_service(model)
    leader_item = make_item(321, 3)
    follower_item = make_item(321, 5)
    errors: list[Exception] = []

    def leader() -> None:
        try:
            service.encode_item(leader_item)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=leader)
    thread.start()
    deadline = time.monotonic() + 5
    while not service.inflight and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service.inflight, "leader never registered in-flight"

    with pytest.raises(RuntimeError, match="returned an invalid"):
        service.encode_item(follower_item)
    thread.join(timeout=30)

    assert not errors, errors
    assert leader_item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)
    assert follower_item.precomputed_embeddings is None
    stats = service.stats()
    assert stats["merged"] == 1
    assert stats["failed"] == 1


def test_multi_item_batch_failure_retries_per_item_and_counts_stats() -> None:
    model = StubModel()
    model.fail_multi_item = True
    gate = threading.Event()
    model.encode_gate = gate
    service = make_service(model)
    items = [make_item(31, 3), make_item(32, 3), make_item(33, 4)]
    errors: list[Exception] = []

    def worker(item: MultimodalDataItem) -> None:
        try:
            service.encode_item(item)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    # note (luojiaxuan): queue every leader before releasing the gate so the
    # next drain exercises the multi-item retry path.
    deadline = time.monotonic() + 5
    while len(service.inflight) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(service.inflight) == 3, "items never queued"
    gate.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    for item in items:
        assert item.precomputed_embeddings.shape == (
            item.num_audio_tokens,
            HIDDEN_SIZE,
        )
    stats = service.stats()
    assert stats["failed"] == 0
    assert stats["items"] == 3
    assert stats["batches"] == 3
    assert model.encode_calls == 4
    assert len(service.cache) == 3


def test_eviction_under_byte_budget_triggers_reencode() -> None:
    model = StubModel()
    service = make_service(model, cache_max_bytes=100)

    for audio_hash in (1, 2, 3):
        service.encode_item(make_item(audio_hash, 3))
    assert model.encode_calls == 3
    assert service.cache.eviction_count >= 1
    assert len(service.cache) == 2

    service.encode_item(make_item(1, 3))
    assert model.encode_calls == 4


def test_invalid_cache_entry_is_evicted_and_reencoded() -> None:
    model = StubModel()
    service = make_service(model)
    probe = make_item(42, 3)
    service.encode_item(probe)
    assert model.encode_calls == 1
    key = service.cache_key(probe)

    for poison in (
        torch.zeros(5, HIDDEN_SIZE),
        torch.zeros(3, HIDDEN_SIZE + 1),
        torch.zeros(3, HIDDEN_SIZE, dtype=torch.float64),
    ):
        service.cache.put(key, poison)
        item = make_item(42, 3)
        service.encode_item(item)
        assert model.encode_calls == 2
        assert item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, probe.precomputed_embeddings)
        model.encode_calls = 1


def test_token_count_mismatch_fails_loudly() -> None:
    model = StubModel()
    model.row_offset = 1
    service = make_service(model)
    item = make_item(9, 3)

    with pytest.raises(RuntimeError, match="!= expected rows"):
        service.encode_item(item)

    assert item.precomputed_embeddings is None
    assert len(service.cache) == 0


def test_missing_token_count_raises() -> None:
    service = make_service()
    item = MultimodalDataItem(modality=Modality.AUDIO, hash=1)

    with pytest.raises(RuntimeError, match="num_audio_tokens"):
        service.encode_item(item)


def test_item_without_fingerprint_encodes_without_caching() -> None:
    model = StubModel()
    service = make_service(model)

    first = make_item(1, 2)
    second = make_item(1, 2)
    first.audio_fingerprint = None
    second.audio_fingerprint = None
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert first.feature is None
    assert first.precomputed_embeddings.shape == (2, HIDDEN_SIZE)
    assert len(service.cache) == 0


def test_build_cache_namespace_is_stable_and_scoped() -> None:
    model = StubModel()
    frontend = SimpleNamespace(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        nb_max_frames=3000,
        padding_value=0.0,
    )
    base = dict(
        model_path="Qwen/Qwen3-ASR-1.7B",
        feature_extractor=frontend,
        mm_attention_backend=None,
    )

    namespace = build_cache_namespace(model, **base)
    assert namespace == build_cache_namespace(model, **base)
    assert namespace != build_cache_namespace(
        model, **{**base, "model_path": "other/revision"}
    )
    assert namespace != build_cache_namespace(
        model, **{**base, "mm_attention_backend": "triton_attn"}
    )
    assert namespace != build_cache_namespace(StubModel(dtype=torch.bfloat16), **base)
    changed_frontend = SimpleNamespace(**{**vars(frontend), "hop_length": 320})
    assert namespace != build_cache_namespace(
        model, **{**base, "feature_extractor": changed_frontend}
    )
    changed_config = StubModel()
    changed_config.config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=HIDDEN_SIZE)
        ),
        marker="other",
    )
    assert namespace != build_cache_namespace(changed_config, **base)


def test_flat_2d_encoder_output_is_also_accepted() -> None:
    model = StubModel()
    model.packed_3d_output = False
    service = make_service(model)
    item = make_item(8, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)


@pytest.mark.accelerator
@requires_accelerator
def test_the_device_cache_is_really_reclaimed_after_an_oom() -> None:
    """The behaviour the fix exists for, on the live accelerator."""
    device_module = torch.get_device_module(torch.device(DEVICE))
    service = make_service(StubModel().to(DEVICE))

    with device_module.device(DEVICE):
        device_module.synchronize()
        device_module.empty_cache()
        floor = device_module.memory_reserved()
        allocated = device_module.memory_allocated()
        cached_slack = max(0, floor - allocated)
        block = torch.empty(
            cached_slack + 64 * 1024 * 1024,
            dtype=torch.uint8,
            device=DEVICE,
        )
        device_module.synchronize()
        reserved_with_block = device_module.memory_reserved()
        assert reserved_with_block > floor

        del block
        device_module.synchronize()
        reserved_before = device_module.memory_reserved()
        assert reserved_before > floor

        service.recover_after_failure(torch.OutOfMemoryError("encoder OOM"))

        device_module.synchronize()
        assert device_module.memory_reserved() < reserved_before

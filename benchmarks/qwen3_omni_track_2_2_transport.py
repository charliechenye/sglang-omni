# SPDX-License-Identifier: Apache-2.0
"""Two-GPU transport probe for Qwen3-Omni RFC #1018 Track 2.2.

The probe does not load a model.  It starts a Thinker-like sender process on
GPU 0 and a Talker-like receiver process on GPU 1, then uses the production
``CommEngine`` and ``stage_io`` stream-chunk path.  Only tensor contents and
the request scheduler are synthetic.

The primary mode measures A/B/C/D and D/C/B/A order, c1/c8/c32 active request
IDs, five rounds, 100 warmup chunks per arm, and 2,000 measured chunks per
arm.  c32 is a transport interleaving probe, not a complete high-concurrency
serving benchmark.
The trace mode is a smaller diagnostic with trace logging enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import sglang_omni

_SGLANG_OMNI_SOURCE = Path(sglang_omni.__file__).resolve()
try:
    _SGLANG_OMNI_SOURCE.relative_to(_REPOSITORY_ROOT / "sglang_omni")
except ValueError as exc:
    raise RuntimeError(
        "benchmark imported sglang_omni outside the checked-out repository: "
        f"{_SGLANG_OMNI_SOURCE}"
    ) from exc

_BENCHMARK_NAME = "qwen3_omni_track_2_2_transport"
_DEFAULT_WARMUP_CHUNKS = 100
_DEFAULT_MEASURE_CHUNKS = 2_000
_DEFAULT_ROUNDS = 5
_TRACE_WARMUP_CHUNKS = 64
_TRACE_MEASURE_CHUNKS = 100
_TRACE_ROUNDS = 1
_DEFAULT_CONCURRENCIES = (1, 8, 32)
_DEFAULT_CUDA_IPC_SLOT_KB = 64
_DEFAULT_RELAY_CREDITS = 2
_DEFAULT_TIMEOUT_S = 1_800.0
_MATERIAL_CHANGE_PCT = 5.0
_MIN_STABLE_IMPROVEMENT_FRACTION = 0.60

torch: Any = None
CommEngine: Any = None
CommRouter: Any = None
ControlPlaneContext: Any = None
DataAckMessage: Any = None
DataReadyMessage: Any = None
DataRef: Any = None
PullSocket: Any = None
PushSocket: Any = None
TransportKind: Any = None


@dataclass(frozen=True)
class ArmSpec:
    name: str
    description: str
    primary_shape: tuple[int, ...]
    primary_dtype: str
    metadata_layer_hidden: bool
    expected_transport: str
    logical_bytes_per_chunk: int
    tensor_transfers_per_chunk: int


ARM_SPECS: dict[str, ArmSpec] = {
    "A": ArmSpec(
        name="A",
        description="BF16 primary [2048] plus BF16 layer_hidden [2048] metadata",
        primary_shape=(2048,),
        primary_dtype="torch.bfloat16",
        metadata_layer_hidden=True,
        expected_transport="cuda_ipc",
        logical_bytes_per_chunk=8_192,
        tensor_transfers_per_chunk=2,
    ),
    "B": ArmSpec(
        name="B",
        description="BF16 primary [2048] plus token_id metadata",
        primary_shape=(2048,),
        primary_dtype="torch.bfloat16",
        metadata_layer_hidden=False,
        expected_transport="cuda_ipc",
        logical_bytes_per_chunk=4_096,
        tensor_transfers_per_chunk=1,
    ),
    "C": ArmSpec(
        name="C",
        description="persistent CUDA uint8 carrier [1] plus token_id metadata",
        primary_shape=(1,),
        primary_dtype="torch.uint8",
        metadata_layer_hidden=False,
        expected_transport="cuda_ipc",
        logical_bytes_per_chunk=1,
        tensor_transfers_per_chunk=1,
    ),
    "D": ArmSpec(
        name="D",
        description="CPU torch.long carrier [1] plus token_id metadata",
        primary_shape=(1,),
        primary_dtype="torch.int64",
        metadata_layer_hidden=False,
        expected_transport="shm",
        logical_bytes_per_chunk=8,
        tensor_transfers_per_chunk=1,
    ),
}


def _load_runtime() -> None:
    """Load CUDA/runtime dependencies only after CLI qualification succeeds."""

    global CommEngine, CommRouter, ControlPlaneContext, DataAckMessage
    global DataReadyMessage, DataRef, PullSocket, PushSocket, TransportKind, torch
    if torch is not None:
        return
    import torch as torch_module

    from sglang_omni.comm.data_ref import DataRef as data_ref
    from sglang_omni.comm.data_ref import TransportKind as transport_kind
    from sglang_omni.comm.engine import CommEngine as comm_engine
    from sglang_omni.comm.router import CommRouter as comm_router
    from sglang_omni.pipeline.control_plane import (
        ControlPlaneContext as control_plane_context,
    )
    from sglang_omni.pipeline.control_plane import PullSocket as pull_socket
    from sglang_omni.pipeline.control_plane import PushSocket as push_socket
    from sglang_omni.proto import DataAckMessage as data_ack_message
    from sglang_omni.proto import DataReadyMessage as data_ready_message

    torch = torch_module
    CommEngine = comm_engine
    CommRouter = comm_router
    ControlPlaneContext = control_plane_context
    DataAckMessage = data_ack_message
    DataReadyMessage = data_ready_message
    DataRef = data_ref
    PullSocket = pull_socket
    PushSocket = push_socket
    TransportKind = transport_kind


def _torch_dtype(dtype_name: str) -> Any:
    _load_runtime()
    _, attribute = dtype_name.split(".", 1)
    return getattr(torch, attribute)


@dataclass(frozen=True)
class RunSpec:
    run_key: str
    arm: str
    concurrency: int
    round_index: int
    direction: str
    order_position: int
    warmup_chunks: int
    measure_chunks: int

    @property
    def total_chunks(self) -> int:
        return self.warmup_chunks + self.measure_chunks


@dataclass(frozen=True)
class _PendingSend:
    object_id: str
    start_ns: int
    publish_end_ns: int
    phase: str
    completion: asyncio.Future[int]


class _TraceQueueHandler(logging.Handler):
    """Forward existing COMM_TRACE records to the parent process."""

    def __init__(self, trace_queue: Any, role: str) -> None:
        super().__init__(level=logging.INFO)
        self.trace_queue = trace_queue
        self.role = role

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        prefix = "COMM_TRACE "
        if not message.startswith(prefix):
            return
        try:
            event = json.loads(message[len(prefix) :])
        except (TypeError, ValueError):
            return
        if not isinstance(event, dict):
            return
        event["trace_role"] = self.role
        try:
            self.trace_queue.put(event)
        except Exception:
            # Trace collection is diagnostic only; it must not fail the probe.
            return


def _configure_trace(trace_enabled: bool, trace_queue: Any, role: str) -> None:
    os.environ["SGLANG_OMNI_COMM_TRACE"] = "1" if trace_enabled else "0"
    logger = logging.getLogger("sglang_omni.comm_trace")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO if trace_enabled else logging.WARNING)
    if trace_enabled:
        logger.addHandler(_TraceQueueHandler(trace_queue, role))


class _BenchmarkControlPlane:
    """Production-compatible PUSH/PULL stage endpoint for this probe.

    The real stage control plane also owns coordinator and abort sockets.  The
    probe has neither, but it deliberately keeps the repository's real ZMQ
    sockets and msgpack DataReady/DataAck serialization for the stage edge.
    """

    def __init__(
        self,
        *,
        outgoing_endpoint: str,
        incoming_endpoint: str,
        bind_incoming: bool,
    ) -> None:
        self._outgoing = PushSocket(outgoing_endpoint)
        self._incoming = PullSocket(incoming_endpoint, bind=bind_incoming)
        self._object_ids: dict[tuple[str, int], str] = {}
        self._object_keys: dict[str, tuple[str, int]] = {}
        self._ready: dict[str, asyncio.Future[None]] = {}
        self._complete: dict[str, asyncio.Future[int]] = {}
        self._outstanding: set[str] = set()
        self._max_outstanding = 0
        self._fatal: asyncio.Future[None] | None = None

    async def start(self) -> None:
        await self._incoming.start()
        await self._outgoing.connect()

    async def receive(self) -> Any:
        return await self._incoming.recv()

    async def send_to_stage(
        self,
        _stage_name: str,
        _stage_endpoint: str,
        message: DataReadyMessage | DataAckMessage,
    ) -> None:
        if isinstance(message, DataReadyMessage):
            if message.data_ref is None:
                raise ValueError("benchmark data_ready message has no data_ref")
            object_id = message.data_ref.get("object_id")
            if not isinstance(object_id, str):
                raise ValueError("benchmark data_ref object_id must be a string")
            if message.chunk_id is None:
                raise ValueError("benchmark data_ready message has no chunk_id")
            key = (message.request_id, message.chunk_id)
            if key in self._object_ids:
                raise RuntimeError(f"duplicate benchmark data key {key!r}")
            loop = asyncio.get_running_loop()
            self._object_ids[key] = object_id
            self._object_keys[object_id] = key
            self._ready[object_id] = loop.create_future()
            self._complete[object_id] = loop.create_future()
        await self._outgoing.send(message)

    def object_id_for(self, request_id: str, chunk_id: int) -> str:
        try:
            return self._object_ids[(request_id, chunk_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"no published object for {request_id}:{chunk_id}"
            ) from exc

    def mark_ready(self, object_id: str) -> asyncio.Future[int]:
        try:
            ready = self._ready[object_id]
            completion = self._complete[object_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown benchmark object {object_id!r}") from exc
        if not ready.done():
            ready.set_result(None)
        self._outstanding.add(object_id)
        self._max_outstanding = max(self._max_outstanding, len(self._outstanding))
        return completion

    def fatal_future(self) -> asyncio.Future[None] | None:
        return self._fatal

    @property
    def max_outstanding(self) -> int:
        return self._max_outstanding

    def reset_max_outstanding(self) -> None:
        if self._outstanding:
            raise RuntimeError("cannot reset outstanding count with live objects")
        self._max_outstanding = 0

    def retire(self, object_id: str) -> None:
        completion = self._complete.get(object_id)
        if completion is None:
            raise RuntimeError(f"unknown benchmark object {object_id!r}")
        if not completion.done():
            raise RuntimeError(
                f"cannot retire incomplete benchmark object {object_id!r}"
            )
        key = self._object_keys.pop(object_id, None)
        if key is not None:
            self._object_ids.pop(key, None)
        self._ready.pop(object_id, None)
        self._complete.pop(object_id, None)

    async def ack_loop(self, engine: CommEngine) -> None:
        self._fatal = asyncio.get_running_loop().create_future()
        handlers: set[asyncio.Task[None]] = set()
        try:
            while True:
                message = await self.receive()
                if not isinstance(message, DataAckMessage):
                    raise TypeError(
                        "benchmark sender expected DataAckMessage, got "
                        f"{type(message).__name__}"
                    )
                task = asyncio.create_task(self._resolve_ack(message, engine))
                handlers.add(task)
                task.add_done_callback(handlers.discard)
                task.add_done_callback(self._record_ack_failure)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._set_fatal(exc)
            raise
        finally:
            for task in tuple(handlers):
                task.cancel()
            if handlers:
                await asyncio.gather(*handlers, return_exceptions=True)

    async def _resolve_ack(self, message: DataAckMessage, engine: CommEngine) -> None:
        object_id = message.object_id
        ready = self._ready.get(object_id)
        completion = self._complete.get(object_id)
        if ready is None or completion is None:
            raise RuntimeError(f"ACK for unknown benchmark object {object_id!r}")
        await ready
        pending = engine._pending.get(object_id)
        if pending is None:
            raise RuntimeError(f"ACK arrived before pending object {object_id!r}")
        engine.ack_transfer(message)
        if pending.task is not None:
            await pending.task
        release_ns = time.perf_counter_ns()
        self._outstanding.discard(object_id)
        if not completion.done():
            completion.set_result(release_ns)

    def _record_ack_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._set_fatal(error)

    def _set_fatal(self, error: BaseException) -> None:
        if self._fatal is None or self._fatal.done():
            return
        self._fatal.set_exception(error)
        self._fatal.add_done_callback(lambda future: future.exception())

    def close(self) -> None:
        self._incoming.close()
        self._outgoing.close()
        self._object_ids.clear()
        self._object_keys.clear()
        self._ready.clear()
        self._complete.clear()
        self._outstanding.clear()


async def _drain_completion_records(
    records: list[_PendingSend],
    control: _BenchmarkControlPlane,
    ack_release_latencies_ms: list[float],
) -> int:
    """Drain completions and use the resource-release timestamp they publish."""

    pending = {record.completion: record for record in records}
    latest_release_ns = 0
    while pending:
        fatal = control.fatal_future()
        waitables: set[asyncio.Future[Any]] = set(pending)
        if fatal is not None:
            waitables.add(fatal)
        done, _ = await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)
        if fatal is not None and fatal in done:
            error = fatal.exception()
            if error is not None:
                raise error
            raise RuntimeError("benchmark ACK loop failed without an error")
        for completion in done:
            record = pending.pop(completion)
            release_ns = completion.result()
            latest_release_ns = max(latest_release_ns, release_ns)
            control.retire(record.object_id)
            if record.phase == "measure":
                ack_release_latencies_ms.append(
                    (release_ns - record.start_ns) / 1_000_000.0
                )
    return latest_release_ns


def _distribute_chunks(total: int, concurrency: int) -> list[int]:
    if total < concurrency:
        raise ValueError(
            f"chunk count {total} must be at least concurrency {concurrency} "
            "so every request ID participates"
        )
    base, remainder = divmod(total, concurrency)
    return [base + int(index < remainder) for index in range(concurrency)]


def _round_robin_chunk_plan(
    counts: list[int], chunk_offsets: list[int] | None = None
) -> list[tuple[int, int]]:
    """Return one producer's request/chunk order without creating workers."""

    if not counts:
        return []
    offsets = [0] * len(counts) if chunk_offsets is None else chunk_offsets
    if len(offsets) != len(counts):
        raise ValueError("chunk offsets must match request count")
    return [
        (request_index, offsets[request_index] + local_index)
        for local_index in range(max(counts))
        for request_index, request_count in enumerate(counts)
        if local_index < request_count
    ]


def _build_run_specs(
    *,
    direction: str,
    arm_order: tuple[str, ...],
    concurrency_values: tuple[int, ...],
    rounds: int,
    warmup_chunks: int,
    measure_chunks: int,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for round_index in range(rounds):
        for concurrency in concurrency_values:
            for order_position, arm in enumerate(arm_order):
                run_key = f"{direction}-round{round_index}-c{concurrency}-{arm}"
                specs.append(
                    RunSpec(
                        run_key=run_key,
                        arm=arm,
                        concurrency=concurrency,
                        round_index=round_index,
                        direction=direction,
                        order_position=order_position,
                        warmup_chunks=warmup_chunks,
                        measure_chunks=measure_chunks,
                    )
                )
    return specs


def _make_buffers(arm: ArmSpec, *, sender_gpu: int) -> dict[str, torch.Tensor]:
    _load_runtime()
    dtype = _torch_dtype(arm.primary_dtype)
    device = (
        torch.device(f"cuda:{sender_gpu}")
        if arm.expected_transport == "cuda_ipc"
        else torch.device("cpu")
    )
    if arm.name in {"A", "B"}:
        primary = torch.arange(2048, dtype=dtype, device=device)
    elif arm.name == "C":
        primary = torch.tensor([17], dtype=dtype, device=device)
    else:
        primary = torch.tensor([17], dtype=dtype, device=device)
    buffers = {"primary": primary}
    if arm.metadata_layer_hidden:
        buffers["layer_hidden"] = torch.arange(
            2048, dtype=_torch_dtype("torch.bfloat16"), device=device
        )
    return buffers


def _comm_config(
    role: str, *, pool_mb: int | None, slot_kb: int, credits: int
) -> dict[str, Any]:
    config = {
        "worker_id": f"qwen3_track_2_2_{role}_{os.getpid()}",
        "cuda_ipc_slot_size_kb": slot_kb,
        "credits": credits,
    }
    if pool_mb is not None:
        config["cuda_ipc_pool_size_mb"] = pool_mb
    return config


def _make_router(
    *,
    role: str,
    gpu_id: int,
    peer_gpu: int,
    config: dict[str, Any],
) -> CommRouter:
    stage_name, peer_name = (
        ("thinker", "talker") if role == "sender" else ("talker", "thinker")
    )
    return CommRouter(
        stage_name=stage_name,
        gpu_id=gpu_id,
        placement_gpu_id=gpu_id,
        same_process_targets=set(),
        gpu_stage_names={peer_name},
        stage_gpu_ids={peer_name: (peer_gpu,)},
        remote_stage_names=set(),
        comm_config=config,
    )


def _run_key_and_request_index(request_id: str) -> tuple[str, int]:
    run_key, marker, raw_index = request_id.rpartition(":req")
    if not marker or not run_key or not raw_index.isdigit():
        raise ValueError(f"invalid benchmark request_id {request_id!r}")
    return run_key, int(raw_index)


def _validate_received_payload(
    arm: ArmSpec,
    data: torch.Tensor,
    metadata: dict[str, Any] | None,
    *,
    receiver_gpu: int,
    check_contents: bool,
) -> None:
    _load_runtime()
    if not isinstance(metadata, dict):
        raise AssertionError("stream metadata must be a dict")
    if type(metadata.get("token_id")) is not int:
        raise AssertionError("token_id did not remain a Python int")
    if tuple(data.shape) != arm.primary_shape:
        raise AssertionError(
            f"primary shape {tuple(data.shape)} != {arm.primary_shape}"
        )
    if data.dtype != _torch_dtype(arm.primary_dtype):
        raise AssertionError(f"primary dtype {data.dtype} != {arm.primary_dtype}")
    if arm.expected_transport == "cuda_ipc":
        if data.device.type != "cuda" or data.device.index != receiver_gpu:
            raise AssertionError(
                f"CUDA primary arrived on {data.device}, expected cuda:{receiver_gpu}"
            )
    elif data.device.type != "cpu":
        raise AssertionError(f"SHM primary arrived on {data.device}, expected CPU")

    layer_hidden = metadata.get("layer_hidden")
    if arm.metadata_layer_hidden:
        if not isinstance(layer_hidden, torch.Tensor):
            raise AssertionError("layer_hidden metadata tensor is missing")
        if tuple(layer_hidden.shape) != (2048,):
            raise AssertionError("layer_hidden shape is not [2048]")
        if layer_hidden.dtype != _torch_dtype("torch.bfloat16"):
            raise AssertionError("layer_hidden dtype is not BF16")
        if (
            layer_hidden.device.type != "cuda"
            or layer_hidden.device.index != receiver_gpu
        ):
            raise AssertionError(
                f"layer_hidden arrived on {layer_hidden.device}, "
                f"expected cuda:{receiver_gpu}"
            )
    elif layer_hidden is not None:
        raise AssertionError(f"unexpected layer_hidden tensor in arm {arm.name}")

    if not check_contents:
        return
    if arm.name in {"A", "B"}:
        if float(data[0].item()) != 0.0 or float(data[-1].item()) != 2047.0:
            raise AssertionError("BF16 primary content check failed")
        if arm.metadata_layer_hidden:
            assert isinstance(layer_hidden, torch.Tensor)
            if (
                float(layer_hidden[0].item()) != 0.0
                or float(layer_hidden[-1].item()) != 2047.0
            ):
                raise AssertionError("BF16 layer_hidden content check failed")
    elif int(data[0].item()) != 17:
        raise AssertionError(f"carrier content check failed for arm {arm.name}")


def _backend_length(data_ref: DataRef) -> int:
    length = data_ref.buffer.length
    for metadata_ref in data_ref.metadata_tensors:
        length += metadata_ref.ref.buffer.length
    return int(length)


async def _sender_process_async(
    *,
    specs: list[RunSpec],
    data_endpoint: str,
    ack_endpoint: str,
    sender_gpu: int,
    receiver_gpu: int,
    pool_mb: int | None,
    slot_kb: int,
    credits: int,
    timeout_s: float,
    ready_queue: Any,
    start_event: Any,
) -> dict[str, Any]:
    _load_runtime()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    torch.cuda.set_device(sender_gpu)
    router = _make_router(
        role="sender",
        gpu_id=sender_gpu,
        peer_gpu=receiver_gpu,
        config=_comm_config(
            "sender", pool_mb=pool_mb, slot_kb=slot_kb, credits=credits
        ),
    )
    engine = CommEngine(router)
    control = _BenchmarkControlPlane(
        outgoing_endpoint=data_endpoint,
        incoming_endpoint=ack_endpoint,
        bind_incoming=True,
    )
    await control.start()
    ready_queue.put("sender")
    await asyncio.to_thread(start_event.wait)
    ack_task = asyncio.create_task(control.ack_loop(engine))
    buffers = {
        arm: _make_buffers(ARM_SPECS[arm], sender_gpu=sender_gpu) for arm in ARM_SPECS
    }
    direct_cuda_ipc_allowed = router.can_use_direct_cuda_ipc("talker")

    async def send_run(spec: RunSpec) -> dict[str, Any]:
        control.reset_max_outstanding()
        arm = ARM_SPECS[spec.arm]
        transport, relay = router.relay_for_stream(
            "talker", buffers[spec.arm]["primary"]
        )
        if transport.value != arm.expected_transport:
            raise AssertionError(
                f"router selected {transport.value} for arm {arm.name}, expected "
                f"{arm.expected_transport}"
            )
        if transport.value == "cuda_ipc" and direct_cuda_ipc_allowed:
            raise AssertionError(
                "benchmark topology unexpectedly permits direct same-GPU CUDA IPC"
            )

        warmup_counts = _distribute_chunks(spec.warmup_chunks, spec.concurrency)
        measure_counts = _distribute_chunks(spec.measure_chunks, spec.concurrency)
        publish_latencies_ms: list[float] = []
        ack_release_latencies_ms: list[float] = []

        async def send_chunk(
            request_index: int,
            chunk_id: int,
            *,
            phase: str,
        ) -> _PendingSend:
            request_id = f"{spec.run_key}:req{request_index}"
            metadata: dict[str, Any] = {
                "token_id": int(1_000_000 * request_index + chunk_id)
            }
            if arm.metadata_layer_hidden:
                metadata["layer_hidden"] = buffers[spec.arm]["layer_hidden"]
            start_ns = time.perf_counter_ns()
            await engine.send_stream_chunk(
                relay=relay,
                control_plane=control,
                request_id=request_id,
                data=buffers[spec.arm]["primary"],
                target_stage="talker",
                target_endpoint=data_endpoint,
                from_stage="thinker",
                chunk_id=chunk_id,
                metadata=metadata,
                transport=transport,
            )
            publish_end_ns = time.perf_counter_ns()
            object_id = control.object_id_for(request_id, chunk_id)
            completion = control.mark_ready(object_id)
            if phase == "measure":
                publish_latencies_ms.append((publish_end_ns - start_ns) / 1_000_000.0)
            return _PendingSend(
                object_id=object_id,
                start_ns=start_ns,
                publish_end_ns=publish_end_ns,
                phase=phase,
                completion=completion,
            )

        async def produce_phase(
            counts: list[int], *, chunk_offsets: list[int], phase: str
        ) -> tuple[list[_PendingSend], int, int]:
            records: list[_PendingSend] = []
            phase_start_ns = time.perf_counter_ns()
            for request_index, chunk_id in _round_robin_chunk_plan(
                counts, chunk_offsets
            ):
                records.append(
                    await send_chunk(
                        request_index,
                        chunk_id,
                        phase=phase,
                    )
                )
            return records, phase_start_ns, time.perf_counter_ns()

        async def drain(records: list[_PendingSend]) -> int:
            return await _drain_completion_records(
                records, control, ack_release_latencies_ms
            )

        warmup_records, _, _ = await produce_phase(
            warmup_counts, chunk_offsets=[0] * spec.concurrency, phase="warmup"
        )
        await asyncio.wait_for(drain(warmup_records), timeout=timeout_s)

        measure_start_ns = time.perf_counter_ns()
        measure_records, _, measure_publish_end_ns = await produce_phase(
            measure_counts, chunk_offsets=warmup_counts, phase="measure"
        )
        latest_release_ns = await asyncio.wait_for(
            drain(measure_records), timeout=timeout_s
        )
        publish_elapsed_s = max(
            (measure_publish_end_ns - measure_start_ns) / 1_000_000_000.0,
            1e-12,
        )
        measure_end_ns = max(measure_publish_end_ns, latest_release_ns)
        end_to_end_elapsed_s = max(
            (measure_end_ns - measure_start_ns) / 1_000_000_000.0,
            1e-12,
        )
        final_ack_drain_ms = max(
            (latest_release_ns - measure_publish_end_ns) / 1_000_000.0,
            0.0,
        )
        return {
            "run_key": spec.run_key,
            "arm": arm.name,
            "concurrency": spec.concurrency,
            "round": spec.round_index,
            "direction": spec.direction,
            "order_position": spec.order_position,
            "warmup_chunks": spec.warmup_chunks,
            "measure_chunks": spec.measure_chunks,
            "transport": transport.value,
            "relay_type": type(relay).__name__,
            "sender_gpu": sender_gpu,
            "logical_tensor_bytes_per_chunk": arm.logical_bytes_per_chunk,
            "tensor_transfers_per_chunk": arm.tensor_transfers_per_chunk,
            "total_transferred_logical_tensor_bytes": arm.logical_bytes_per_chunk
            * spec.measure_chunks,
            "total_tensor_transfers": (
                arm.tensor_transfers_per_chunk * spec.measure_chunks
            ),
            "publish_latency_ms": _numeric_stats(publish_latencies_ms),
            "ack_release_latency_ms": _numeric_stats(ack_release_latencies_ms),
            "publish_chunks_per_sec": spec.measure_chunks / publish_elapsed_s,
            "end_to_end_drain_chunks_per_sec": (
                spec.measure_chunks / end_to_end_elapsed_s
            ),
            "publish_wall_ms": publish_elapsed_s * 1_000.0,
            "end_to_end_drain_wall_ms": end_to_end_elapsed_s * 1_000.0,
            "final_ack_drain_ms": final_ack_drain_ms,
            "maximum_outstanding_transfer_count": control.max_outstanding,
            "producer_mode": "single_round_robin",
            "sender_router_selected_expected": True,
            "direct_cuda_ipc_allowed": direct_cuda_ipc_allowed,
        }

    try:
        run_results = [await send_run(spec) for spec in specs]
    finally:
        ack_task.cancel()
        await asyncio.gather(ack_task, return_exceptions=True)
        control.close()
        await engine.close()
        ControlPlaneContext.close()
    return {
        "role": "sender",
        "gpu": sender_gpu,
        "device_name": torch.cuda.get_device_name(sender_gpu),
        "cuda_device_count": torch.cuda.device_count(),
        "runs": run_results,
        "direct_cuda_ipc_allowed": direct_cuda_ipc_allowed,
    }


async def _receiver_process_async(
    *,
    specs: list[RunSpec],
    data_endpoint: str,
    ack_endpoint: str,
    sender_gpu: int,
    receiver_gpu: int,
    pool_mb: int | None,
    slot_kb: int,
    credits: int,
    ready_queue: Any,
    start_event: Any,
) -> dict[str, Any]:
    _load_runtime()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    torch.cuda.set_device(receiver_gpu)
    router = _make_router(
        role="receiver",
        gpu_id=receiver_gpu,
        peer_gpu=sender_gpu,
        config=_comm_config(
            "receiver", pool_mb=pool_mb, slot_kb=slot_kb, credits=credits
        ),
    )
    engine = CommEngine(router)
    control = _BenchmarkControlPlane(
        outgoing_endpoint=ack_endpoint,
        incoming_endpoint=data_endpoint,
        bind_incoming=True,
    )
    await control.start()
    ready_queue.put("receiver")
    await asyncio.to_thread(start_event.wait)

    spec_by_key = {spec.run_key: spec for spec in specs}
    expected_messages = sum(spec.total_chunks for spec in specs)
    pending_tasks: set[asyncio.Task[None]] = set()
    receive_errors: list[BaseException] = []
    receive_failure = asyncio.get_running_loop().create_future()
    run_observations: dict[str, dict[str, Any]] = {}
    validated_runs: set[str] = set()

    def observation_for(spec: RunSpec) -> dict[str, Any]:
        observation = run_observations.get(spec.run_key)
        if observation is None:
            observation = {
                "run_key": spec.run_key,
                "arm": spec.arm,
                "concurrency": spec.concurrency,
                "round": spec.round_index,
                "direction": spec.direction,
                "message_count": 0,
                "measured_message_count": 0,
                "transports": set(),
                "relay_types": set(),
                "tensor_transfer_counts": set(),
                "backend_bytes": 0,
                "measured_backend_bytes": 0,
                "logical_bytes": 0,
                "measured_logical_bytes": 0,
                "content_checks": 0,
            }
            run_observations[spec.run_key] = observation
        return observation

    async def receive_one(message: DataReadyMessage) -> None:
        if message.data_ref is None or message.chunk_id is None:
            raise ValueError("benchmark receiver got a non-chunk data_ready message")
        run_key, request_index = _run_key_and_request_index(message.request_id)
        spec = spec_by_key.get(run_key)
        if spec is None:
            raise ValueError(f"unknown benchmark run key {run_key!r}")
        arm = ARM_SPECS[spec.arm]
        data_ref = DataRef.from_dict(message.data_ref)
        if data_ref.transport.value != arm.expected_transport:
            raise AssertionError(
                f"receiver saw {data_ref.transport.value} for arm {arm.name}, "
                f"expected {arm.expected_transport}"
            )
        if data_ref.kind.value != "stream_chunk":
            raise AssertionError(
                f"receiver saw unexpected data kind {data_ref.kind.value}"
            )
        expected_metadata_refs = arm.tensor_transfers_per_chunk - 1
        if len(data_ref.metadata_tensors) != expected_metadata_refs:
            raise AssertionError(
                f"arm {arm.name} has {len(data_ref.metadata_tensors)} metadata "
                f"tensor refs, expected {expected_metadata_refs}"
            )

        relay = router.relay(data_ref.transport)
        data, metadata = await engine.read_stream_chunk(relay=relay, data_ref=data_ref)
        warmup_counts = _distribute_chunks(spec.warmup_chunks, spec.concurrency)
        is_measured = message.chunk_id >= warmup_counts[request_index]
        check_contents = (
            run_key not in validated_runs
            and message.chunk_id == 0
            and warmup_counts[request_index] > 0
        )
        _validate_received_payload(
            arm,
            data,
            metadata,
            receiver_gpu=receiver_gpu,
            check_contents=check_contents,
        )
        if check_contents:
            validated_runs.add(run_key)

        observation = observation_for(spec)
        observation["message_count"] += 1
        observation["measured_message_count"] += int(is_measured)
        observation["transports"].add(data_ref.transport.value)
        observation["relay_types"].add(type(relay).__name__)
        observation["tensor_transfer_counts"].add(1 + len(data_ref.metadata_tensors))
        backend_bytes = _backend_length(data_ref)
        logical_bytes = int(data.numel() * data.element_size())
        if metadata is not None:
            for value in metadata.values():
                if isinstance(value, torch.Tensor):
                    logical_bytes += int(value.numel() * value.element_size())
        observation["backend_bytes"] += backend_bytes
        observation["measured_backend_bytes"] += backend_bytes * int(is_measured)
        observation["logical_bytes"] += logical_bytes
        observation["measured_logical_bytes"] += logical_bytes * int(is_measured)
        observation["content_checks"] += int(check_contents)

        await control.send_to_stage(
            "thinker",
            ack_endpoint,
            DataAckMessage(
                request_id=message.request_id,
                from_stage="talker",
                to_stage="thinker",
                object_id=data_ref.object_id,
                success=True,
            ),
        )

    def receive_task_done(task: asyncio.Task[None]) -> None:
        pending_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            receive_errors.append(error)
            if not receive_failure.done():
                receive_failure.set_result(error)

    async def receive_message() -> DataReadyMessage:
        receive_task = asyncio.create_task(control.receive())
        done, _ = await asyncio.wait(
            {receive_task, receive_failure},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receive_failure in done:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
            raise receive_failure.result()
        message = receive_task.result()
        if not isinstance(message, DataReadyMessage):
            raise TypeError(
                "benchmark receiver expected DataReadyMessage, got "
                f"{type(message).__name__}"
            )
        return message

    try:
        for _ in range(expected_messages):
            if receive_errors:
                raise receive_errors[0]
            message = await receive_message()
            task = asyncio.create_task(receive_one(message))
            pending_tasks.add(task)
            task.add_done_callback(receive_task_done)
        if pending_tasks:
            await asyncio.gather(*tuple(pending_tasks), return_exceptions=True)
        if receive_errors:
            raise receive_errors[0]
    finally:
        for task in tuple(pending_tasks):
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*tuple(pending_tasks), return_exceptions=True)
        control.close()
        await engine.close()
        ControlPlaneContext.close()

    observations: list[dict[str, Any]] = []
    for spec in specs:
        observation = run_observations.get(spec.run_key)
        if observation is None:
            raise RuntimeError(f"receiver produced no observation for {spec.run_key}")
        observation = dict(observation)
        for key in ("transports", "relay_types", "tensor_transfer_counts"):
            observation[key] = sorted(observation[key], key=str)
        observations.append(observation)
    return {
        "role": "receiver",
        "gpu": receiver_gpu,
        "device_name": torch.cuda.get_device_name(receiver_gpu),
        "cuda_device_count": torch.cuda.device_count(),
        "expected_messages": expected_messages,
        "runs": observations,
    }


def _sender_process_entry(
    specs: list[RunSpec],
    data_endpoint: str,
    ack_endpoint: str,
    sender_gpu: int,
    receiver_gpu: int,
    pool_mb: int | None,
    slot_kb: int,
    credits: int,
    timeout_s: float,
    trace_enabled: bool,
    trace_queue: Any,
    ready_queue: Any,
    start_event: Any,
    result_queue: Any,
) -> None:
    _configure_trace(trace_enabled, trace_queue, "sender")
    try:
        result = asyncio.run(
            _sender_process_async(
                specs=specs,
                data_endpoint=data_endpoint,
                ack_endpoint=ack_endpoint,
                sender_gpu=sender_gpu,
                receiver_gpu=receiver_gpu,
                pool_mb=pool_mb,
                slot_kb=slot_kb,
                credits=credits,
                timeout_s=timeout_s,
                ready_queue=ready_queue,
                start_event=start_event,
            )
        )
        result_queue.put({"role": "sender", "ok": True, "result": result})
    except BaseException as exc:
        result_queue.put(
            {
                "role": "sender",
                "ok": False,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        trace_queue.put({"trace_done": "sender"})


def _receiver_process_entry(
    specs: list[RunSpec],
    data_endpoint: str,
    ack_endpoint: str,
    sender_gpu: int,
    receiver_gpu: int,
    pool_mb: int | None,
    slot_kb: int,
    credits: int,
    trace_enabled: bool,
    trace_queue: Any,
    ready_queue: Any,
    start_event: Any,
    result_queue: Any,
) -> None:
    _configure_trace(trace_enabled, trace_queue, "receiver")
    try:
        result = asyncio.run(
            _receiver_process_async(
                specs=specs,
                data_endpoint=data_endpoint,
                ack_endpoint=ack_endpoint,
                sender_gpu=sender_gpu,
                receiver_gpu=receiver_gpu,
                pool_mb=pool_mb,
                slot_kb=slot_kb,
                credits=credits,
                ready_queue=ready_queue,
                start_event=start_event,
            )
        )
        result_queue.put({"role": "receiver", "ok": True, "result": result})
    except BaseException as exc:
        result_queue.put(
            {
                "role": "receiver",
                "ok": False,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        trace_queue.put({"trace_done": "receiver"})


def _percentile(values: Iterable[float], fraction: float) -> float:
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("cannot calculate a percentile from no samples")
    if fraction <= 0.0:
        return samples[0]
    if fraction >= 1.0:
        return samples[-1]
    return samples[max(0, math.ceil(fraction * len(samples)) - 1)]


def _trace_run_key(event: dict[str, Any]) -> str:
    request_id = event.get("request_id")
    if isinstance(request_id, str):
        run_key, marker, _ = request_id.rpartition(":req")
        if marker and run_key:
            return run_key
    return "unscoped"


def _numeric_stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    median = _percentile(values, 0.50)
    deviations = [abs(float(value) - median) for value in values]
    return {
        "count": len(values),
        "median": median,
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
        "mad": _percentile(deviations, 0.50),
        "mean": statistics.fmean(values),
    }


def _summarize_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    field_sources = {
        "bytes": {
            "cuda_ipc_put_async": "bytes",
            "cuda_ipc_get_async": "bytes",
            "cuda_ipc_put_wait_ack": "bytes",
            "cuda_ipc_get_wait_copy": "bytes",
        },
        "slot_count": {"cuda_ipc_pool_alloc": "slot_count"},
        "slot_allocation_wait_ms": {"cuda_ipc_put_async": "acquire_ms"},
        "sender_copy_enqueue_ms": {"cuda_ipc_put_async": "copy_enqueue_ms"},
        "cuda_event_handle_ms": {"cuda_ipc_put_async": "event_handle_ms"},
        "receiver_event_import_ms": {"cuda_ipc_get_async": "event_import_ms"},
        "receiver_copy_enqueue_ms": {"cuda_ipc_get_async": "copy_enqueue_ms"},
        "receiver_gpu_copy_wait_ms": {
            "cuda_ipc_get_wait_copy": "receiver_gpu_wait_copy_ms"
        },
        "ack_resource_hold_ms": {"cuda_ipc_put_wait_ack": "ack_resume_ms"},
    }
    for event in events:
        run_key = _trace_run_key(event)
        role = str(event.get("trace_role", "unknown"))
        key = f"{run_key}:{role}"
        group = grouped.setdefault(key, {"run_key": run_key, "trace_role": role})
        event_name = str(event.get("event", "unknown"))
        counts = group.setdefault("event_counts", collections.Counter())
        counts[event_name] += 1
        for output_name, sources in field_sources.items():
            source_field = sources.get(event_name)
            value = event.get(source_field) if source_field else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                group.setdefault("fields", {}).setdefault(output_name, []).append(
                    float(value)
                )

    summaries = []
    for key in sorted(grouped):
        group = grouped[key]
        event_counts = group.get("event_counts", collections.Counter())
        summaries.append(
            {
                "run_key": group["run_key"],
                "trace_role": group["trace_role"],
                "event_counts": dict(sorted(event_counts.items())),
                "fields": {
                    name: _numeric_stats(values)
                    for name, values in group.get("fields", {}).items()
                },
            }
        )
    return {
        "event_count": len(events),
        "event_counts": dict(
            sorted(
                collections.Counter(
                    str(event.get("event", "unknown")) for event in events
                ).items()
            )
        ),
        "runs": summaries,
    }


def _drain_trace_queue(
    trace_queue: Any,
    trace_events: list[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            item = trace_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if isinstance(item, dict) and "trace_done" not in item:
            trace_events.append(item)
    while True:
        try:
            item = trace_queue.get_nowait()
        except queue.Empty:
            return
        if isinstance(item, dict) and "trace_done" not in item:
            trace_events.append(item)


def _terminate_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=30)


def _run_direction(
    *,
    ctx: mp.context.BaseContext,
    specs: list[RunSpec],
    direction: str,
    sender_gpu: int,
    receiver_gpu: int,
    pool_mb: int | None,
    slot_kb: int,
    credits: int,
    timeout_s: float,
    trace_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_dir = Path(tempfile.mkdtemp(prefix="qwen3-omni-track-2-2-"))
    data_endpoint = f"ipc://{endpoint_dir / 'data.sock'}"
    ack_endpoint = f"ipc://{endpoint_dir / 'ack.sock'}"
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    trace_queue = ctx.Queue()
    start_event = ctx.Event()
    trace_events: list[dict[str, Any]] = []
    drain_stop = threading.Event()
    drain_thread = threading.Thread(
        target=_drain_trace_queue,
        args=(trace_queue, trace_events, drain_stop),
        daemon=True,
    )
    drain_thread.start()
    common_args = (
        specs,
        data_endpoint,
        ack_endpoint,
        sender_gpu,
        receiver_gpu,
        pool_mb,
        slot_kb,
        credits,
    )
    sender = ctx.Process(
        target=_sender_process_entry,
        args=common_args
        + (
            timeout_s,
            trace_enabled,
            trace_queue,
            ready_queue,
            start_event,
            result_queue,
        ),
        name=f"qwen3-transport-{direction}-sender",
    )
    receiver = ctx.Process(
        target=_receiver_process_entry,
        args=common_args
        + (
            trace_enabled,
            trace_queue,
            ready_queue,
            start_event,
            result_queue,
        ),
        name=f"qwen3-transport-{direction}-receiver",
    )
    processes = [sender, receiver]
    result_messages: dict[str, dict[str, Any]] = {}

    def handle_result_message(message: dict[str, Any]) -> None:
        role = str(message.get("role"))
        result_messages[role] = message
        if not message.get("ok", False):
            raise RuntimeError(
                f"{direction} {role} failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )

    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + timeout_s
        ready_roles: set[str] = set()
        while ready_roles != {"sender", "receiver"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {direction} control sockets")
            try:
                ready_roles.add(str(ready_queue.get(timeout=0.2)))
            except queue.Empty:
                while True:
                    try:
                        handle_result_message(result_queue.get_nowait())
                    except queue.Empty:
                        break
                for process in processes:
                    if not process.is_alive() and process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"{direction} child {process.name} exited with "
                            f"code {process.exitcode} before readiness"
                        )
        start_event.set()

        while set(result_messages) != {"sender", "receiver"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out running {direction} transport sequence")
            try:
                handle_result_message(result_queue.get(timeout=0.2))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    raise RuntimeError(f"{direction} children exited without results")
        for process in processes:
            process.join(timeout=30)
            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f"{direction} child {process.name} exited with {process.exitcode}"
                )
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        drain_stop.set()
        drain_thread.join(timeout=5)
        shutil.rmtree(endpoint_dir, ignore_errors=True)
        for resource in (ready_queue, result_queue, trace_queue):
            resource.close()
            resource.join_thread()

    sender_result = result_messages["sender"]["result"]
    receiver_result = result_messages["receiver"]["result"]
    for event in trace_events:
        event["direction"] = direction
    return sender_result["runs"], receiver_result["runs"], trace_events


def _merge_results(
    sender_runs: list[dict[str, Any]], receiver_runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    receiver_by_key = {run["run_key"]: run for run in receiver_runs}
    merged: list[dict[str, Any]] = []
    for sender_run in sender_runs:
        run = dict(sender_run)
        receiver_run = receiver_by_key.get(run["run_key"])
        if receiver_run is None:
            raise RuntimeError(f"missing receiver result for {run['run_key']}")
        expected_transport = ARM_SPECS[run["arm"]].expected_transport
        expected_relay = (
            "CudaIpcRelay" if expected_transport == "cuda_ipc" else "ShmRelay"
        )
        transport_checks = {
            "sender_router_selected_expected": bool(
                run["sender_router_selected_expected"]
                and run["transport"] == expected_transport
            ),
            "pooled_cuda_ipc_not_direct": bool(
                run["transport"] != "cuda_ipc" or not run["direct_cuda_ipc_allowed"]
            ),
            "receiver_data_refs_selected_expected": receiver_run["transports"]
            == [expected_transport],
            "receiver_relay_type_expected": receiver_run["relay_types"]
            == [expected_relay],
            "receiver_tensor_transfer_count_expected": receiver_run[
                "tensor_transfer_counts"
            ]
            == [ARM_SPECS[run["arm"]].tensor_transfers_per_chunk],
            "receiver_message_count_expected": receiver_run["message_count"]
            == run["warmup_chunks"] + run["measure_chunks"],
            "receiver_measured_message_count_expected": receiver_run[
                "measured_message_count"
            ]
            == run["measure_chunks"],
            "receiver_content_check_passed": receiver_run["content_checks"] > 0,
        }
        run["receiver"] = receiver_run
        run["transport_checks"] = transport_checks
        run["transport_check_passed"] = all(transport_checks.values())
        if not run["transport_check_passed"]:
            raise AssertionError(
                f"transport check failed for {run['run_key']}: {transport_checks}"
            )
        merged.append(run)
    return merged


def _aggregate_latency_metric(
    cells: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    medians = [float(cell[metric]["median"]) for cell in cells]
    p95s = [float(cell[metric]["p95"]) for cell in cells]
    return {
        "median": statistics.median(medians),
        "p95": statistics.median(p95s),
        "run_median_mad": float(_numeric_stats(medians)["mad"]),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for run in runs:
        grouped[(run["arm"], int(run["concurrency"]))].append(run)
    aggregates: list[dict[str, Any]] = []
    for (arm, concurrency), cells in sorted(grouped.items()):
        aggregates.append(
            {
                "arm": arm,
                "concurrency": concurrency,
                "runs": len(cells),
                "directions": sorted({cell["direction"] for cell in cells}),
                "publish_latency_ms": _aggregate_latency_metric(
                    cells, "publish_latency_ms"
                ),
                "ack_release_latency_ms": _aggregate_latency_metric(
                    cells, "ack_release_latency_ms"
                ),
                "publish_chunks_per_sec": statistics.median(
                    cell["publish_chunks_per_sec"] for cell in cells
                ),
                "end_to_end_drain_chunks_per_sec": statistics.median(
                    cell["end_to_end_drain_chunks_per_sec"] for cell in cells
                ),
                "final_ack_drain_ms": statistics.median(
                    cell["final_ack_drain_ms"] for cell in cells
                ),
                "maximum_outstanding_transfer_count": max(
                    cell["maximum_outstanding_transfer_count"] for cell in cells
                ),
                "logical_tensor_bytes_per_chunk": cells[0][
                    "logical_tensor_bytes_per_chunk"
                ],
                "tensor_transfers_per_chunk": cells[0]["tensor_transfers_per_chunk"],
                "transport": cells[0]["transport"],
                "producer_modes": sorted({cell["producer_mode"] for cell in cells}),
                "all_transport_checks_passed": all(
                    cell["transport_check_passed"] for cell in cells
                ),
            }
        )
    return aggregates


_PAIRED_METRICS = (
    "publish_latency_ms",
    "ack_release_latency_ms",
    "publish_chunks_per_sec",
    "end_to_end_drain_chunks_per_sec",
)
_LATENCY_METRICS = {"publish_latency_ms", "ack_release_latency_ms"}


def _run_metric_value(run: dict[str, Any], metric: str) -> float:
    value = run[metric]
    if isinstance(value, dict):
        return float(value["median"])
    return float(value)


def _paired_runs(
    runs: list[dict[str, Any]],
    baseline_arm: str,
    candidate_arm: str,
    concurrency: int | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for run in runs:
        run_concurrency = int(run["concurrency"])
        if concurrency is not None and run_concurrency != concurrency:
            continue
        key = (
            str(run["direction"]),
            int(run["round"]),
            run_concurrency,
            str(run["arm"]),
        )
        by_key[key] = run
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    keys = sorted(
        {key[:3] for key in by_key if key[3] in {baseline_arm, candidate_arm}}
    )
    for direction, round_index, run_concurrency in keys:
        baseline = by_key.get((direction, round_index, run_concurrency, baseline_arm))
        candidate = by_key.get((direction, round_index, run_concurrency, candidate_arm))
        if baseline is not None and candidate is not None:
            paired.append((baseline, candidate))
    return paired


def _paired_delta(candidate: float, baseline: float, metric: str) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else math.inf
    if metric in _LATENCY_METRICS:
        return (1.0 - candidate / baseline) * 100.0
    return (candidate / baseline - 1.0) * 100.0


def _paired_metric_stats(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], metric: str
) -> dict[str, Any]:
    deltas = [
        _paired_delta(
            _run_metric_value(candidate, metric),
            _run_metric_value(baseline, metric),
            metric,
        )
        for baseline, candidate in pairs
    ]
    summary = _numeric_stats(deltas)
    if summary is None:
        return {
            "paired_observations": 0,
            "median_paired_percentage_delta": None,
            "p25_paired_percentage_delta": None,
            "p75_paired_percentage_delta": None,
            "mad_paired_percentage_delta": None,
            "improvement_count": 0,
            "improvement_fraction": None,
            "regression_count": 0,
            "stable_effect": False,
        }
    improvement_count = sum(delta >= _MATERIAL_CHANGE_PCT for delta in deltas)
    regression_count = sum(delta <= -_MATERIAL_CHANGE_PCT for delta in deltas)
    return {
        "paired_observations": len(deltas),
        "median_paired_percentage_delta": summary["median"],
        "p25_paired_percentage_delta": summary["p25"],
        "p75_paired_percentage_delta": summary["p75"],
        "mad_paired_percentage_delta": summary["mad"],
        "improvement_count": improvement_count,
        "improvement_fraction": improvement_count / len(deltas),
        "regression_count": regression_count,
        "stable_effect": abs(summary["median"]) > summary["mad"],
    }


def _stable_improvement(stats: dict[str, Any]) -> bool:
    median = stats["median_paired_percentage_delta"]
    mad = stats["mad_paired_percentage_delta"]
    fraction = stats["improvement_fraction"]
    return bool(
        median is not None
        and mad is not None
        and fraction is not None
        and median >= _MATERIAL_CHANGE_PCT
        and fraction >= _MIN_STABLE_IMPROVEMENT_FRACTION
        and median > mad
    )


def _stable_regression(stats: dict[str, Any]) -> bool:
    median = stats["median_paired_percentage_delta"]
    mad = stats["mad_paired_percentage_delta"]
    if median is None or mad is None:
        return False
    observations = int(stats["paired_observations"])
    regressions = int(stats["regression_count"])
    return (
        median <= -_MATERIAL_CHANGE_PCT
        and observations > 0
        and regressions / observations >= _MIN_STABLE_IMPROVEMENT_FRACTION
        and -median > mad
    )


def _stable_non_regression(stats: dict[str, Any]) -> bool:
    median = stats["median_paired_percentage_delta"]
    return median is not None and median > -_MATERIAL_CHANGE_PCT


def _paired_metrics(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    return {metric: _paired_metric_stats(pairs, metric) for metric in _PAIRED_METRICS}


def _status_for_paired_metrics(metrics: dict[str, dict[str, Any]]) -> str:
    publish_facing_good = any(
        _stable_improvement(metrics[metric])
        for metric in (
            "publish_latency_ms",
            "publish_chunks_per_sec",
            "end_to_end_drain_chunks_per_sec",
        )
    )
    throughput_good = all(
        _stable_non_regression(metrics[metric])
        for metric in (
            "publish_chunks_per_sec",
            "end_to_end_drain_chunks_per_sec",
        )
    )
    has_stable_regression = any(
        _stable_regression(metrics[metric]) for metric in _PAIRED_METRICS
    )
    if has_stable_regression:
        return "NO-GO"
    if publish_facing_good and throughput_good:
        return "GO"
    return "MAYBE"


def _decision_for_transition(comparisons: list[dict[str, Any]]) -> str:
    go_count = sum(item["status"] == "GO" for item in comparisons)
    no_go_count = sum(item["status"] == "NO-GO" for item in comparisons)
    if go_count >= 2 and no_go_count == 0:
        return "GO"
    if no_go_count >= 2 and go_count == 0:
        return "NO-GO"
    return "MAYBE"


def _build_transition(
    runs: list[dict[str, Any]], baseline_arm: str, candidate_arm: str
) -> dict[str, Any]:
    concurrencies = sorted({int(run["concurrency"]) for run in runs})
    cells: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        pairs = _paired_runs(runs, baseline_arm, candidate_arm, concurrency=concurrency)
        metrics = _paired_metrics(pairs)
        cells.append(
            {
                "concurrency": concurrency,
                "baseline": baseline_arm,
                "candidate": candidate_arm,
                "paired_observations": len(pairs),
                "metrics": metrics,
                "status": _status_for_paired_metrics(metrics),
            }
        )
    all_pairs = _paired_runs(runs, baseline_arm, candidate_arm)
    return {
        "baseline": baseline_arm,
        "candidate": candidate_arm,
        "cells": cells,
        "overall_paired": {
            "paired_observations": len(all_pairs),
            "metrics": _paired_metrics(all_pairs),
        },
        "decision": _decision_for_transition(cells) if cells else "MAYBE",
    }


def _recommend_candidate(direct_from_a: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    for transition_name, transition in direct_from_a.items():
        metrics = transition["overall_paired"]["metrics"]
        publish_delta = metrics["publish_latency_ms"]["median_paired_percentage_delta"]
        ack_delta = metrics["ack_release_latency_ms"]["median_paired_percentage_delta"]
        publish_throughput = metrics["publish_chunks_per_sec"][
            "median_paired_percentage_delta"
        ]
        drain_throughput = metrics["end_to_end_drain_chunks_per_sec"][
            "median_paired_percentage_delta"
        ]
        publish_facing_deltas = [
            delta
            for delta in (publish_delta, publish_throughput, drain_throughput)
            if delta is not None
        ]
        publish_facing_score = (
            max(publish_facing_deltas) if publish_facing_deltas else None
        )
        throughput_score = (
            min(publish_throughput, drain_throughput)
            if publish_throughput is not None and drain_throughput is not None
            else None
        )
        scores.append(
            {
                "candidate": transition["candidate"],
                "transition": transition_name,
                "decision": transition["decision"],
                "publish_facing_score_pct": publish_facing_score,
                "ack_guardrail_delta_pct": ack_delta,
                "throughput_score_pct": throughput_score,
                "paired_observations": transition["overall_paired"][
                    "paired_observations"
                ],
            }
        )
    rank = {"GO": 2, "MAYBE": 1, "NO-GO": 0}
    scores.sort(
        key=lambda item: (
            rank[item["decision"]],
            (
                item["publish_facing_score_pct"]
                if item["publish_facing_score_pct"] is not None
                else float("-inf")
            ),
            (
                item["throughput_score_pct"]
                if item["throughput_score_pct"] is not None
                else float("-inf")
            ),
        ),
        reverse=True,
    )
    best = scores[0] if scores else None
    if best is None or best["decision"] == "NO-GO":
        return {
            "best_candidate_relative_to_A": best["candidate"] if best else None,
            "recommended_candidate": "A",
            "decision": "NO-GO" if best else "MAYBE",
            "candidate_scores": scores,
            "rationale": "No candidate has a stable direct improvement over A.",
        }
    return {
        "best_candidate_relative_to_A": best["candidate"],
        "recommended_candidate": best["candidate"],
        "decision": best["decision"],
        "candidate_scores": scores,
        "rationale": (
            "Recommendation uses direct A-baseline evidence; incremental C->D "
            "regression does not override a stable A->C result."
        ),
    }


def _build_comparisons(runs: list[dict[str, Any]]) -> dict[str, Any]:
    incremental_pairs = (("A", "B"), ("B", "C"), ("C", "D"))
    direct_pairs = (("A", "B"), ("A", "C"), ("A", "D"))
    incremental = {
        f"{baseline}->{candidate}": _build_transition(runs, baseline, candidate)
        for baseline, candidate in incremental_pairs
    }
    direct_from_a = {
        f"{baseline}->{candidate}": _build_transition(runs, baseline, candidate)
        for baseline, candidate in direct_pairs
    }
    recommendation = _recommend_candidate(direct_from_a)
    return {
        "material_change_threshold_pct": _MATERIAL_CHANGE_PCT,
        "minimum_stable_improvement_fraction": _MIN_STABLE_IMPROVEMENT_FRACTION,
        "rubric": {
            "GO": (
                "at least one publish-facing metric (publish latency, publish "
                "throughput, or end-to-end drain throughput) has a stable >=5% "
                "improvement, neither throughput metric materially regresses, "
                "and ACK/resource latency has no stable material regression"
            ),
            "MAYBE": (
                "effect is marginal, mixed, or ordinary paired variation is "
                "comparable to the nominal effect"
            ),
            "NO-GO": (
                "the compared candidate has a stable material regression; this "
                "only rejects that candidate relative to its baseline"
            ),
        },
        "incremental": incremental,
        "direct_from_A": direct_from_a,
        "recommendation": recommendation,
        "overall": recommendation["decision"],
    }


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "-+-".join("-" * width for width in widths)
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        separator,
    ]
    lines.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _print_summary(
    aggregates: list[dict[str, Any]], comparisons: dict[str, Any]
) -> None:
    rows = [
        [
            cell["arm"],
            str(cell["concurrency"]),
            f"{cell['publish_latency_ms']['median']:.3f}",
            f"{cell['publish_latency_ms']['p95']:.3f}",
            f"{cell['ack_release_latency_ms']['median']:.3f}",
            f"{cell['publish_chunks_per_sec']:.1f}",
            f"{cell['end_to_end_drain_chunks_per_sec']:.1f}",
            str(cell["logical_tensor_bytes_per_chunk"]),
            str(cell["tensor_transfers_per_chunk"]),
            str(cell["maximum_outstanding_transfer_count"]),
            cell["transport"],
            str(cell["runs"]),
        ]
        for cell in aggregates
    ]
    print("Human-readable summary (median across order passes and rounds):")
    print(
        _format_table(
            [
                "arm",
                "reqs",
                "pub med ms",
                "pub p95 ms",
                "ACK med ms",
                "publish/s",
                "drain/s",
                "logical B/chunk",
                "tensors/chunk",
                "max out",
                "transport",
                "cells",
            ],
            rows,
        )
    )
    print("\nAttribution comparisons:")
    comparison_rows: list[list[str]] = []
    for group_name in ("incremental", "direct_from_A"):
        for transition, result in comparisons[group_name].items():
            for cell in result["cells"]:
                metrics = cell["metrics"]
                comparison_rows.append(
                    [
                        group_name,
                        transition,
                        str(cell["concurrency"]),
                        f"{metrics['publish_latency_ms']['median_paired_percentage_delta']:+.2f}%",
                        f"{metrics['ack_release_latency_ms']['median_paired_percentage_delta']:+.2f}%",
                        f"{metrics['end_to_end_drain_chunks_per_sec']['median_paired_percentage_delta']:+.2f}%",
                        f"{metrics['publish_latency_ms']['mad_paired_percentage_delta']:.2f}%",
                        f"{metrics['publish_latency_ms']['improvement_count']}/{metrics['publish_latency_ms']['paired_observations']}",
                        cell["status"],
                    ]
                )
            comparison_rows.append(
                [group_name, transition, "all", "", "", "", "", "", result["decision"]]
            )
    print(
        _format_table(
            [
                "comparison",
                "transition",
                "reqs",
                "publish Δ",
                "ACK Δ",
                "drain/s Δ",
                "publish MAD",
                "improve",
                "decision",
            ],
            comparison_rows,
        )
    )
    recommendation = comparisons["recommendation"]
    print(
        "\nRecommendation: "
        f"{recommendation['recommended_candidate']} "
        f"({recommendation['decision']}); "
        f"best direct candidate={recommendation['best_candidate_relative_to_A']}"
    )
    print(
        "\nInterpret logical bytes only alongside measured transfer count, slot "
        "occupancy, event handling, receiver wait, and ACK/resource-hold trace."
    )


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPOSITORY_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _cuda_device_name(index: int) -> str | None:
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
            return None
        return str(torch.cuda.get_device_name(index))
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def _runtime_provenance() -> dict[str, Any]:
    cuda_version = None
    torch_version = None
    if torch is not None:
        torch_version = str(torch.__version__)
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            cuda_version = str(cuda_version)
    return {
        "git_commit": _git_commit(),
        "repository_root": str(_REPOSITORY_ROOT),
        "sglang_omni_source": str(_SGLANG_OMNI_SOURCE),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_0_name": _cuda_device_name(0),
        "gpu_1_name": _cuda_device_name(1),
    }


def _print_runtime_provenance(provenance: dict[str, Any]) -> None:
    print("Track 2.2 benchmark runtime provenance:")
    print(f"  Git commit: {provenance['git_commit']}")
    print(f"  Repository root: {provenance['repository_root']}")
    print(f"  Imported sglang_omni: {provenance['sglang_omni_source']}")
    print(f"  torch: {provenance['torch_version']}")
    print(f"  CUDA: {provenance['cuda_version']}")
    print(f"  GPU 0: {provenance['gpu_0_name']}")
    print(f"  GPU 1: {provenance['gpu_1_name']}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "trace"), default="primary")
    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
        help="path for machine-readable JSON output",
    )
    parser.add_argument("--warmup-chunks", type=int)
    parser.add_argument("--measure-chunks", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=None,
        help="interleaved request counts; default is 1 8 32",
    )
    parser.add_argument("--sender-gpu", type=int, default=0)
    parser.add_argument("--receiver-gpu", type=int, default=1)
    parser.add_argument(
        "--cuda-ipc-slot-kb", type=int, default=_DEFAULT_CUDA_IPC_SLOT_KB
    )
    parser.add_argument(
        "--cuda-ipc-pool-mb",
        type=int,
        default=None,
        help=(
            "optional CUDA IPC pool size in MiB; unset preserves the production "
            "relay default (1 GiB)"
        ),
    )
    parser.add_argument("--relay-credits", type=int, default=_DEFAULT_RELAY_CREDITS)
    parser.add_argument("--timeout-s", type=float, default=_DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    defaults = (
        (_TRACE_WARMUP_CHUNKS, _TRACE_MEASURE_CHUNKS, _TRACE_ROUNDS)
        if args.mode == "trace"
        else (_DEFAULT_WARMUP_CHUNKS, _DEFAULT_MEASURE_CHUNKS, _DEFAULT_ROUNDS)
    )
    args.warmup_chunks = (
        defaults[0] if args.warmup_chunks is None else args.warmup_chunks
    )
    args.measure_chunks = (
        defaults[1] if args.measure_chunks is None else args.measure_chunks
    )
    args.rounds = defaults[2] if args.rounds is None else args.rounds
    args.concurrency = tuple(
        _DEFAULT_CONCURRENCIES if args.concurrency is None else args.concurrency
    )
    if args.sender_gpu == args.receiver_gpu:
        parser.error("sender and receiver must use different GPU IDs")
    if args.warmup_chunks <= 0 or args.measure_chunks <= 0 or args.rounds <= 0:
        parser.error("warmup-chunks, measure-chunks, and rounds must be positive")
    if not args.concurrency or any(value <= 0 for value in args.concurrency):
        parser.error("concurrency values must be positive")
    if args.warmup_chunks < max(args.concurrency):
        parser.error("warmup-chunks must be >= the largest concurrency")
    if args.measure_chunks < max(args.concurrency):
        parser.error("measure-chunks must be >= the largest concurrency")
    if args.cuda_ipc_slot_kb <= 0 or args.relay_credits <= 0:
        parser.error("relay sizes and credits must be positive")
    if args.cuda_ipc_pool_mb is not None and args.cuda_ipc_pool_mb <= 0:
        parser.error("cuda-ipc-pool-mb must be positive when provided")
    if args.timeout_s <= 0:
        parser.error("timeout-s must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_runtime()
    provenance = _runtime_provenance()
    _print_runtime_provenance(provenance)
    trace_enabled = args.mode == "trace"
    os.environ["SGLANG_OMNI_COMM_TRACE"] = "1" if trace_enabled else "0"
    forward_specs = _build_run_specs(
        direction="forward",
        arm_order=("A", "B", "C", "D"),
        concurrency_values=args.concurrency,
        rounds=args.rounds,
        warmup_chunks=args.warmup_chunks,
        measure_chunks=args.measure_chunks,
    )
    reverse_specs = _build_run_specs(
        direction="reverse",
        arm_order=("D", "C", "B", "A"),
        concurrency_values=args.concurrency,
        rounds=args.rounds,
        warmup_chunks=args.warmup_chunks,
        measure_chunks=args.measure_chunks,
    )
    ctx = mp.get_context("spawn")
    all_runs: list[dict[str, Any]] = []
    all_trace_events: list[dict[str, Any]] = []
    for direction, specs in (("forward", forward_specs), ("reverse", reverse_specs)):
        sender_runs, receiver_runs, trace_events = _run_direction(
            ctx=ctx,
            specs=specs,
            direction=direction,
            sender_gpu=args.sender_gpu,
            receiver_gpu=args.receiver_gpu,
            pool_mb=args.cuda_ipc_pool_mb,
            slot_kb=args.cuda_ipc_slot_kb,
            credits=args.relay_credits,
            timeout_s=args.timeout_s,
            trace_enabled=trace_enabled,
        )
        all_runs.extend(_merge_results(sender_runs, receiver_runs))
        all_trace_events.extend(trace_events)

    aggregates = _aggregate_runs(all_runs)
    comparisons = _build_comparisons(all_runs)
    output = {
        "schema_version": 1,
        "benchmark": _BENCHMARK_NAME,
        "git_commit": provenance["git_commit"],
        "repository_root": provenance["repository_root"],
        "sglang_omni_source": provenance["sglang_omni_source"],
        "provenance": provenance,
        "mode": args.mode,
        "trace_enabled": trace_enabled,
        "architecture": {
            "sender": "Thinker-like spawned process",
            "receiver": "Talker-like spawned process",
            "sender_gpu": args.sender_gpu,
            "receiver_gpu": args.receiver_gpu,
            "control_plane": "repository PushSocket/PullSocket with msgpack messages",
            "data_path": (
                "CommEngine.send_stream_chunk -> "
                "stage_io.write_stream_chunk/read_stream_chunk"
            ),
            "cuda_ipc_pool_mb": args.cuda_ipc_pool_mb,
            "cuda_ipc_pool_policy": (
                "explicit override"
                if args.cuda_ipc_pool_mb is not None
                else "production relay default (1 GiB)"
            ),
            "cuda_ipc_slot_kb": args.cuda_ipc_slot_kb,
            "relay_credits": args.relay_credits,
            "model_weights_loaded": False,
        },
        "config": {
            "warmup_chunks_per_arm": args.warmup_chunks,
            "measure_chunks_per_arm": args.measure_chunks,
            "rounds": args.rounds,
            "concurrency": list(args.concurrency),
            "orders": ["A B C D", "D C B A"],
            "producer_mode": "single_round_robin",
            "timeout_s": args.timeout_s,
        },
        "arms": {
            arm.name: {
                "description": arm.description,
                "primary_shape": list(arm.primary_shape),
                "primary_dtype": arm.primary_dtype,
                "metadata_layer_hidden": arm.metadata_layer_hidden,
                "expected_transport": arm.expected_transport,
                "logical_bytes_per_chunk": arm.logical_bytes_per_chunk,
                "tensor_transfers_per_chunk": arm.tensor_transfers_per_chunk,
            }
            for arm in ARM_SPECS.values()
        },
        "runs": all_runs,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "trace": (
            _summarize_trace(all_trace_events) if trace_enabled else {"event_count": 0}
        ),
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_0_name": provenance["gpu_0_name"],
            "gpu_1_name": provenance["gpu_1_name"],
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _print_summary(aggregates, comparisons)
    print(f"\nMachine-readable JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

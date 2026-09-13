# SPDX-License-Identifier: Apache-2.0
"""Opt-in attribution tracing for the buffered Fun-CosyVoice3 vocoder."""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import torch

ATTRIBUTION_ENV = "SGLANG_COSYVOICE3_ATTRIBUTION"
ATTRIBUTION_PATH_ENV = "SGLANG_COSYVOICE3_ATTRIBUTION_PATH"
ATTRIBUTION_SCHEMA = "cosyvoice3-post2141-attribution-v1"


def enabled_from_env() -> bool:
    """Return whether the experimental recorder is explicitly enabled."""

    return os.environ.get(ATTRIBUTION_ENV) == "1"


def _output_path_from_env() -> str:
    raw_path = os.environ.get(ATTRIBUTION_PATH_ENV)
    if raw_path is None or not raw_path.strip():
        raise ValueError(
            f"{ATTRIBUTION_PATH_ENV} must be set to a valid JSONL file path "
            f"when {ATTRIBUTION_ENV}=1"
        )

    path = Path(os.path.expanduser(raw_path)).resolve(strict=False)
    if path.exists() and path.is_dir():
        raise ValueError(
            f"{ATTRIBUTION_PATH_ENV} must name a file, not a directory: {path}"
        )
    if not path.parent.exists() or not path.parent.is_dir():
        raise ValueError(
            f"{ATTRIBUTION_PATH_ENV} parent directory does not exist: " f"{path.parent}"
        )
    if not os.access(path.parent, os.W_OK):
        raise ValueError(
            f"{ATTRIBUTION_PATH_ENV} parent directory is not writable: "
            f"{path.parent}"
        )
    if path.exists() and not os.access(path, os.W_OK):
        raise ValueError(f"{ATTRIBUTION_PATH_ENV} file is not writable: {path}")
    return str(path)


def _cuda_events_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def _new_cuda_events(
    device: torch.device,
) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if not _cuda_events_supported(device):
        return None, None
    with torch.cuda.device(device):
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )


def _record_event(event: torch.cuda.Event | None, device: torch.device) -> None:
    if event is None:
        return
    with torch.cuda.device(device):
        event.record()


def _elapsed_ms(
    start: torch.cuda.Event | None,
    end: torch.cuda.Event | None,
    device: torch.device,
) -> float | None:
    if start is None or end is None:
        return None
    # Do not synchronize here. Callers invoke this only after the existing
    # waveform .cpu() boundary has naturally waited for the preceding GPU work.
    with torch.cuda.device(device):
        return float(start.elapsed_time(end))


class _NvtxRange:
    def __init__(self, name: str) -> None:
        self._active = False
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.nvtx.range_push(name)
        except (AttributeError, NotImplementedError, RuntimeError):
            return
        self._active = True

    def close(self) -> None:
        if not self._active:
            return
        try:
            torch.cuda.nvtx.range_pop()
        except (AttributeError, NotImplementedError, RuntimeError):
            pass
        self._active = False


class CosyVoice3AttributionRecorder:
    """Record one compact JSON object per completed buffered decode batch."""

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._outer_batch_ids = itertools.count(1)
        self._write_lock = Lock()

    @classmethod
    def from_env(cls) -> "CosyVoice3AttributionRecorder | None":
        if not enabled_from_env():
            return None
        return cls(_output_path_from_env())

    def begin_outer_batch(self, request_count: int) -> "CosyVoice3OuterBatchTrace":
        return CosyVoice3OuterBatchTrace(
            recorder=self,
            outer_batch_id=next(self._outer_batch_ids),
            request_count=request_count,
        )

    def _write(self, record: dict[str, Any]) -> None:
        encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        # O_APPEND plus one write keeps each completed batch as one JSONL record
        # when several vocoder processes share the explicitly requested path.
        with self._write_lock:
            fd = os.open(
                self.output_path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o644,
            )
            try:
                written = os.write(fd, encoded)
                if written != len(encoded):
                    raise OSError(
                        f"short write while writing CosyVoice3 attribution: "
                        f"{written} of {len(encoded)} bytes"
                    )
            finally:
                os.close(fd)


class CosyVoice3OuterBatchTrace:
    def __init__(
        self,
        *,
        recorder: CosyVoice3AttributionRecorder,
        outer_batch_id: int,
        request_count: int,
    ) -> None:
        self.recorder = recorder
        self.outer_batch_id = outer_batch_id
        self.request_count = request_count
        self.host_start_ns = time.perf_counter_ns()
        self.host_end_ns: int | None = None
        self.flow_groups: list[CosyVoice3FlowGroupTrace] = []
        self._nvtx = _NvtxRange(
            f"cosy3/vocoder/outer={outer_batch_id}/B={request_count}"
        )

    def begin_flow_group(
        self,
        *,
        index: int,
        request_indices: list[int],
        total_mel_frames: list[int],
        effective_shape: list[int],
        q16_padded_t: int,
        cuda_graph_key: list[int] | None,
        cuda_graph_resident: bool | None,
        device: torch.device,
    ) -> "CosyVoice3FlowGroupTrace":
        trace = CosyVoice3FlowGroupTrace(
            outer=self,
            index=index,
            request_indices=request_indices,
            total_mel_frames=total_mel_frames,
            effective_shape=effective_shape,
            q16_padded_t=q16_padded_t,
            cuda_graph_key=cuda_graph_key,
            cuda_graph_resident=cuda_graph_resident,
            device=device,
        )
        self.flow_groups.append(trace)
        return trace

    def complete(self, host_end_ns: int) -> None:
        self.host_end_ns = host_end_ns
        self.close_nvtx()
        self.recorder._write(
            {
                "schema": ATTRIBUTION_SCHEMA,
                "kind": "outer_batch",
                "pid": os.getpid(),
                "outer_batch_id": self.outer_batch_id,
                "host_start_ns": self.host_start_ns,
                "host_end_ns": self.host_end_ns,
                "request_count": self.request_count,
                "flow_groups": [trace.to_dict() for trace in self.flow_groups],
            }
        )

    def close_nvtx(self) -> None:
        self._nvtx.close()


class CosyVoice3FlowGroupTrace:
    def __init__(
        self,
        *,
        outer: CosyVoice3OuterBatchTrace,
        index: int,
        request_indices: list[int],
        total_mel_frames: list[int],
        effective_shape: list[int],
        q16_padded_t: int,
        cuda_graph_key: list[int] | None,
        cuda_graph_resident: bool | None,
        device: torch.device,
    ) -> None:
        self.outer_batch_id = outer.outer_batch_id
        self.index = index
        self.request_indices = request_indices
        self.total_mel_frames = total_mel_frames
        self.effective_shape = effective_shape
        self.q16_padded_t = q16_padded_t
        self.cuda_graph_key = cuda_graph_key
        self.cuda_graph_resident = cuda_graph_resident
        self.device = torch.device(device)
        self.host_start_ns = time.perf_counter_ns()
        self.host_end_ns: int | None = None
        self.cuda_elapsed_ms: float | None = None
        self.hift_groups: list[CosyVoice3HiFTGroupTrace] = []
        self._cuda_start, self._cuda_end = _new_cuda_events(self.device)
        _record_event(self._cuda_start, self.device)
        graph_label = (
            "none"
            if cuda_graph_key is None
            else f"{cuda_graph_key[0]}x{cuda_graph_key[1]}"
        )
        self._nvtx = _NvtxRange(
            f"cosy3/flow/outer={self.outer_batch_id}/group={index}/"
            f"B={len(request_indices)}/T={effective_shape[-1]}/cg={graph_label}"
        )

    def finish(self) -> None:
        if self.host_end_ns is not None:
            return
        _record_event(self._cuda_end, self.device)
        self.host_end_ns = time.perf_counter_ns()
        self._nvtx.close()

    def resolve_cuda_elapsed(self) -> None:
        if self.cuda_elapsed_ms is None:
            self.cuda_elapsed_ms = _elapsed_ms(
                self._cuda_start, self._cuda_end, self.device
            )

    def begin_hift_group(
        self,
        *,
        index: int,
        mel_lengths: list[int],
        device: torch.device,
    ) -> "CosyVoice3HiFTGroupTrace":
        trace = CosyVoice3HiFTGroupTrace(
            outer_batch_id=self.outer_batch_id,
            flow_group_index=self.index,
            index=index,
            mel_lengths=mel_lengths,
            device=device,
        )
        self.hift_groups.append(trace)
        return trace

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_batch_id": self.outer_batch_id,
            "flow_group_index": self.index,
            "batch_size": len(self.request_indices),
            "request_indices": self.request_indices,
            "total_mel_frames": self.total_mel_frames,
            "min_total_mel_frames": min(self.total_mel_frames),
            "max_total_mel_frames": max(self.total_mel_frames),
            "effective_shape": self.effective_shape,
            "effective_t": self.effective_shape[-1],
            "q16_padded_t": self.q16_padded_t,
            "cuda_graph_key": self.cuda_graph_key,
            "cuda_graph_resident": self.cuda_graph_resident,
            "host_start_ns": self.host_start_ns,
            "host_end_ns": self.host_end_ns,
            "cuda_elapsed_ms": self.cuda_elapsed_ms,
            "hift_groups": [trace.to_dict() for trace in self.hift_groups],
        }


class CosyVoice3HiFTGroupTrace:
    def __init__(
        self,
        *,
        outer_batch_id: int,
        flow_group_index: int,
        index: int,
        mel_lengths: list[int],
        device: torch.device,
    ) -> None:
        self.outer_batch_id = outer_batch_id
        self.flow_group_index = flow_group_index
        self.index = index
        self.mel_lengths = mel_lengths
        self.max_mel_length = max(mel_lengths)
        self.device = torch.device(device)
        self.host_start_ns: int | None = None
        self.host_end_ns: int | None = None
        self.cuda_elapsed_ms: float | None = None
        self.cpu_materialization_start_ns: int | None = None
        self.cpu_materialization_end_ns: int | None = None
        self.cpu_materialization_ms: float | None = None
        self._cuda_start: torch.cuda.Event | None = None
        self._cuda_end: torch.cuda.Event | None = None
        self._hift_nvtx: _NvtxRange | None = None
        self._d2h_nvtx: _NvtxRange | None = None

    def begin_gpu(self) -> None:
        self.host_start_ns = time.perf_counter_ns()
        self._cuda_start, self._cuda_end = _new_cuda_events(self.device)
        _record_event(self._cuda_start, self.device)
        self._hift_nvtx = _NvtxRange(
            f"cosy3/hift/outer={self.outer_batch_id}/"
            f"flow={self.flow_group_index}/group={self.index}/B={len(self.mel_lengths)}"
        )

    def finish_gpu(self) -> None:
        _record_event(self._cuda_end, self.device)
        self.host_end_ns = time.perf_counter_ns()
        if self._hift_nvtx is not None:
            self._hift_nvtx.close()

    def begin_cpu_materialization(self) -> None:
        self.cpu_materialization_start_ns = time.perf_counter_ns()
        self._d2h_nvtx = _NvtxRange(
            f"cosy3/d2h/outer={self.outer_batch_id}/"
            f"flow={self.flow_group_index}/hift={self.index}"
        )

    def finish_cpu_materialization(self) -> None:
        self.cpu_materialization_end_ns = time.perf_counter_ns()
        if self.cpu_materialization_start_ns is not None:
            self.cpu_materialization_ms = (
                self.cpu_materialization_end_ns - self.cpu_materialization_start_ns
            ) / 1_000_000
        if self._d2h_nvtx is not None:
            self._d2h_nvtx.close()

    def resolve_cuda_elapsed(self) -> None:
        if self.cuda_elapsed_ms is None:
            self.cuda_elapsed_ms = _elapsed_ms(
                self._cuda_start, self._cuda_end, self.device
            )

    def to_dict(self) -> dict[str, Any]:
        padding_frames = self.max_mel_length * len(self.mel_lengths) - sum(
            self.mel_lengths
        )
        return {
            "outer_batch_id": self.outer_batch_id,
            "flow_group_index": self.flow_group_index,
            "hift_group_index": self.index,
            "batch_size": len(self.mel_lengths),
            "mel_lengths": self.mel_lengths,
            "max_mel_length": self.max_mel_length,
            "padding_frames": padding_frames,
            "host_start_ns": self.host_start_ns,
            "host_end_ns": self.host_end_ns,
            "cuda_elapsed_ms": self.cuda_elapsed_ms,
            "cpu_materialization_start_ns": self.cpu_materialization_start_ns,
            "cpu_materialization_end_ns": self.cpu_materialization_end_ns,
            "cpu_materialization_ms": self.cpu_materialization_ms,
        }

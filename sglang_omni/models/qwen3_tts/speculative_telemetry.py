# SPDX-License-Identifier: Apache-2.0
"""Optional request-boundary telemetry for Qwen3-TTS P16 workers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TextIO, TypeAlias, TypedDict

P16_TELEMETRY_SCHEMA = "qtts-p16-worker-v1"
P16_POLICY = "chained-k3-native-target"
P16_COUNTER_NAMES = (
    "completed_requests",
    "total_frames",
    "eligible_frames",
    "spec_frames",
    "native_frames",
    "eligible_native_fallbacks",
    "spec_openings",
    "spec_rounds",
    "spec_graph_replays",
    "proposed",
    "accepted",
    "committed",
    "graph_capture_failures",
    "eager_spec_rounds",
    "invalid_probability_rows",
    "cuda_errors",
)


class P16TelemetryCounters(TypedDict):
    completed_requests: int
    total_frames: int
    eligible_frames: int
    spec_frames: int
    native_frames: int
    eligible_native_fallbacks: int
    spec_openings: int
    spec_rounds: int
    spec_graph_replays: int
    proposed: int
    accepted: int
    committed: int
    graph_capture_failures: int
    eager_spec_rounds: int
    invalid_probability_rows: int
    cuda_errors: int


class P16StartupRecord(TypedDict):
    schema: str
    kind: Literal["startup"]
    enabled: bool
    draft_sha256: str | None
    draft_loaded: bool
    policy: str
    pid: int


class P16GraphDiagnostics(TypedDict):
    opening_pair_graph_captures: int
    full_round_graph_captures: int
    graph_captures: int
    graph_capture_time_ms: float
    graph_capture_memory_allocated_baseline_bytes: int
    graph_capture_memory_allocated_current_bytes: int
    graph_capture_memory_allocated_growth_bytes: int
    graph_capture_memory_reserved_baseline_bytes: int
    graph_capture_memory_reserved_current_bytes: int
    graph_capture_memory_reserved_growth_bytes: int
    graph_capture_memory_budget_bytes: int
    graph_capture_memory_budget_exceeded: bool


class P16CompletionGraphDiagnostics(P16GraphDiagnostics):
    captures_since_previous_request: int


class P16CompletionRecord(TypedDict):
    schema: str
    kind: Literal["request_complete"]
    enabled: bool
    pid: int
    seq: int
    native_reason_counts: dict[str, int]
    counters: P16TelemetryCounters
    graph_diagnostics: P16CompletionGraphDiagnostics | None


P16TelemetryRecord: TypeAlias = P16StartupRecord | P16CompletionRecord


@dataclass(frozen=True)
class P16FrameTelemetry:
    """Device-counter deltas and eligibility for one output codec frame."""

    eligible: bool
    speculative: bool
    native_reason: str | None
    spec_openings: int = 0
    spec_rounds: int = 0
    spec_graph_replays: int = 0
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    graph_capture_failures: int = 0
    eager_spec_rounds: int = 0
    invalid_probability_rows: int = 0
    cuda_errors: int = 0


@dataclass
class RequestTelemetry:
    frames: int = 0
    eligible_frames: int = 0
    spec_frames: int = 0
    native_frames: int = 0
    eligible_native_fallbacks: int = 0
    spec_openings: int = 0
    spec_rounds: int = 0
    spec_graph_replays: int = 0
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    graph_capture_failures: int = 0
    eager_spec_rounds: int = 0
    invalid_probability_rows: int = 0
    cuda_errors: int = 0
    native_reason_counts: dict[str, int] = field(default_factory=dict)


class P16WorkerTelemetry:
    """Exclusive JSONL writer flushed only on worker startup/request completion."""

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool,
        draft_sha256: str | None,
        draft_loaded: bool,
        pid: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("P16 telemetry path must be absolute")
        elif enabled and (draft_sha256 is None or not draft_loaded):
            raise ValueError("enabled P16 telemetry requires a validated loaded draft")
        elif not enabled and (draft_sha256 is not None or draft_loaded):
            raise ValueError("disabled P16 telemetry must not report a loaded draft")
        else:
            pass
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        self.stream: TextIO = os.fdopen(descriptor, "w", encoding="utf-8")
        self.enabled = enabled
        self.pid = os.getpid() if pid is None else pid
        self.sequence = 0
        self.counters: P16TelemetryCounters = {name: 0 for name in P16_COUNTER_NAMES}
        self.pending_worker_counters = {
            "graph_capture_failures": 0,
            "eager_spec_rounds": 0,
            "invalid_probability_rows": 0,
            "cuda_errors": 0,
        }
        self.native_reason_counts: dict[str, int] = {}
        self.last_graph_capture_count = 0
        self.requests: dict[str, RequestTelemetry] = {}
        self.write_record(
            P16StartupRecord(
                schema=P16_TELEMETRY_SCHEMA,
                kind="startup",
                enabled=enabled,
                draft_sha256=draft_sha256,
                draft_loaded=draft_loaded,
                policy=P16_POLICY,
                pid=self.pid,
            )
        )

    def record_frame(self, request_id: str, frame: P16FrameTelemetry) -> None:
        if not request_id:
            raise ValueError("P16 telemetry requires a nonempty request ID")
        elif type(frame.eligible) is not bool or type(frame.speculative) is not bool:
            raise ValueError("P16 frame eligibility flags must be booleans")
        elif frame.speculative and not frame.eligible:
            raise ValueError("an ineligible frame cannot use speculative decoding")
        elif frame.speculative and frame.native_reason is not None:
            raise ValueError("a speculative frame cannot have a native fallback reason")
        elif not frame.speculative and not frame.native_reason:
            raise ValueError("a native frame must have an explicit fallback reason")
        else:
            pass
        frame_counts = (
            frame.spec_openings,
            frame.spec_rounds,
            frame.spec_graph_replays,
            frame.proposed,
            frame.accepted,
            frame.committed,
            frame.graph_capture_failures,
            frame.eager_spec_rounds,
            frame.invalid_probability_rows,
            frame.cuda_errors,
        )
        if any(type(value) is not int or value < 0 for value in frame_counts):
            raise ValueError("P16 telemetry frame counters cannot be negative")
        elif frame.speculative and (
            frame.spec_openings != 1
            or frame.committed != 14
            or frame.spec_rounds != frame.spec_graph_replays
            or frame.spec_rounds + frame.accepted != frame.committed
        ):
            raise ValueError("P16 speculative frame counters violate round invariants")
        elif not frame.speculative and any(frame_counts):
            raise ValueError("native frames cannot contribute speculative counters")
        else:
            pass

        request = self.requests.setdefault(request_id, RequestTelemetry())
        request.frames += 1
        request.eligible_frames += int(frame.eligible)
        request.spec_frames += int(frame.speculative)
        request.native_frames += int(not frame.speculative)
        request.eligible_native_fallbacks += int(
            self.enabled and frame.eligible and not frame.speculative
        )
        request.spec_openings += frame.spec_openings
        request.spec_rounds += frame.spec_rounds
        request.spec_graph_replays += frame.spec_graph_replays
        request.proposed += frame.proposed
        request.accepted += frame.accepted
        request.committed += frame.committed
        request.graph_capture_failures += frame.graph_capture_failures
        request.eager_spec_rounds += frame.eager_spec_rounds
        request.invalid_probability_rows += frame.invalid_probability_rows
        request.cuda_errors += frame.cuda_errors
        if not frame.speculative:
            reason = frame.native_reason
            assert reason is not None
            request.native_reason_counts[reason] = (
                request.native_reason_counts.get(reason, 0) + 1
            )
        else:
            pass

    def note_worker_counters(self, deltas: dict[str, int]) -> None:
        """Retain real failure/capture counters from requests that do not complete."""
        if set(deltas) - set(self.pending_worker_counters):
            raise ValueError("P16 worker counter delta contains unsupported fields")
        else:
            pass
        for name, value in deltas.items():
            if type(value) is not int or value < 0:
                raise ValueError(
                    "P16 worker counter deltas must be nonnegative integers"
                )
            else:
                self.pending_worker_counters[name] += value
        else:
            pass

    def complete_request(
        self,
        request_id: str,
        graph_diagnostics: P16GraphDiagnostics | None = None,
    ) -> None:
        request = self.requests.pop(request_id, RequestTelemetry())
        if request.spec_frames > request.eligible_frames:
            raise RuntimeError("P16 telemetry recorded speculative ineligible frames")
        elif request.frames != request.native_frames + request.spec_frames:
            raise RuntimeError("P16 telemetry frame totals are inconsistent")
        elif request.spec_openings != request.spec_frames:
            raise RuntimeError("P16 telemetry opening count is inconsistent")
        elif request.spec_rounds != request.spec_graph_replays:
            raise RuntimeError("P16 telemetry graph replay count is inconsistent")
        elif request.spec_rounds + request.accepted != request.committed:
            raise RuntimeError("P16 telemetry committed-token count is inconsistent")
        else:
            pass

        self.sequence += 1
        deltas: P16TelemetryCounters = {
            "completed_requests": 1,
            "total_frames": request.frames,
            "eligible_frames": request.eligible_frames,
            "spec_frames": request.spec_frames,
            "native_frames": request.native_frames,
            "eligible_native_fallbacks": request.eligible_native_fallbacks,
            "spec_openings": request.spec_openings,
            "spec_rounds": request.spec_rounds,
            "spec_graph_replays": request.spec_graph_replays,
            "proposed": request.proposed,
            "accepted": request.accepted,
            "committed": request.committed,
            "graph_capture_failures": request.graph_capture_failures,
            "eager_spec_rounds": request.eager_spec_rounds,
            "invalid_probability_rows": request.invalid_probability_rows,
            "cuda_errors": request.cuda_errors,
        }
        for name, value in self.pending_worker_counters.items():
            deltas[name] += value
        else:
            pass
        if deltas["accepted"] > deltas["proposed"]:
            raise RuntimeError("P16 accepted proposals exceed the proposal count")
        elif self.enabled and deltas["eligible_frames"] != deltas["spec_frames"]:
            raise RuntimeError(
                "P16 enabled eligible frames did not all run speculation"
            )
        elif not self.enabled and any(
            deltas[name] != 0
            for name in ("spec_frames", "proposed", "accepted", "committed")
        ):
            raise RuntimeError("P16 disabled worker recorded speculative execution")
        elif any(
            self.counters[name] + deltas[name] != 0
            for name in (
                "eligible_native_fallbacks",
                "graph_capture_failures",
                "eager_spec_rounds",
                "invalid_probability_rows",
                "cuda_errors",
            )
        ):
            raise RuntimeError("P16 telemetry recorded a serving-path failure")
        else:
            pass
        native_reason_total = sum(self.native_reason_counts.values()) + sum(
            request.native_reason_counts.values()
        )
        if (
            native_reason_total
            != self.counters["native_frames"] + deltas["native_frames"]
        ):
            raise RuntimeError("P16 native reason counts do not cover native frames")
        else:
            pass
        for name in P16_COUNTER_NAMES:
            self.counters[name] += deltas[name]
        for name in self.pending_worker_counters:
            self.pending_worker_counters[name] = 0
        else:
            pass
        for reason, count in request.native_reason_counts.items():
            self.native_reason_counts[reason] = (
                self.native_reason_counts.get(reason, 0) + count
            )
        else:
            pass
        if graph_diagnostics is None:
            request_graph_diagnostics = None
        else:
            captures_since_previous_request = (
                graph_diagnostics["graph_captures"] - self.last_graph_capture_count
            )
            if captures_since_previous_request < 0:
                raise RuntimeError("P16 graph capture count decreased")
            else:
                pass
            request_graph_diagnostics: P16CompletionGraphDiagnostics = {
                **graph_diagnostics,
                "captures_since_previous_request": captures_since_previous_request,
            }
            self.last_graph_capture_count = graph_diagnostics["graph_captures"]
        self.write_record(
            P16CompletionRecord(
                schema=P16_TELEMETRY_SCHEMA,
                kind="request_complete",
                enabled=self.enabled,
                pid=self.pid,
                seq=self.sequence,
                native_reason_counts=dict(self.native_reason_counts),
                counters=dict(self.counters),
                graph_diagnostics=request_graph_diagnostics,
            )
        )

    def discard_request(self, request_id: str) -> None:
        self.requests.pop(request_id, None)

    def close(self) -> None:
        self.stream.close()

    def write_record(self, record: P16TelemetryRecord) -> None:
        line = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.stream.write(line + "\n")
        self.stream.flush()

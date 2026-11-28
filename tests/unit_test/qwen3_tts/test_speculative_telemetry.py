# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the optional P16 request telemetry contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import pytest

from sglang_omni.models.qwen3_tts.speculative_telemetry import (
    P16FrameTelemetry,
    P16WorkerTelemetry,
)


class TelemetryCounters(TypedDict):
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


class StartupRecord(TypedDict):
    schema: str
    kind: Literal["startup"]
    enabled: bool
    draft_sha256: str | None
    draft_loaded: bool
    policy: str
    pid: int


class CompletionRecord(TypedDict):
    schema: str
    kind: Literal["request_complete"]
    enabled: bool
    pid: int
    seq: int
    native_reason_counts: dict[str, int]
    counters: TelemetryCounters


TelemetryRecord: TypeAlias = StartupRecord | CompletionRecord


def read_records(path: Path) -> list[TelemetryRecord]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_telemetry_counts_eligible_native_frames_and_owns_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker.jsonl"
    telemetry = P16WorkerTelemetry(
        path,
        enabled=False,
        draft_sha256=None,
        draft_loaded=False,
        pid=123,
    )
    with pytest.raises(FileExistsError):
        P16WorkerTelemetry(
            path,
            enabled=False,
            draft_sha256=None,
            draft_loaded=False,
            pid=456,
        )

    telemetry.record_frame(
        "request-1",
        P16FrameTelemetry(
            eligible=True,
            speculative=False,
            native_reason="feature_off",
        ),
    )
    telemetry.complete_request("request-1")
    telemetry.close()

    startup, completion = read_records(path)
    assert startup["kind"] == "startup"
    assert startup == {
        "schema": "qtts-p16-worker-v1",
        "kind": "startup",
        "enabled": False,
        "draft_sha256": None,
        "draft_loaded": False,
        "policy": "chained-k3-native-target",
        "pid": 123,
    }
    assert completion["kind"] == "request_complete"
    assert completion["native_reason_counts"] == {"feature_off": 1}
    assert completion["counters"]["eligible_frames"] == 1
    assert completion["counters"]["total_frames"] == 1
    assert completion["counters"]["native_frames"] == 1
    assert completion["counters"]["spec_frames"] == 0
    assert completion["counters"]["eligible_native_fallbacks"] == 0


def test_enabled_telemetry_requires_all_eligible_frames_to_run_speculation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "enabled.jsonl"
    telemetry = P16WorkerTelemetry(
        path,
        enabled=True,
        draft_sha256="a" * 64,
        draft_loaded=True,
        pid=123,
    )
    telemetry.record_frame(
        "request-1",
        P16FrameTelemetry(
            eligible=True,
            speculative=True,
            native_reason=None,
            spec_openings=1,
            spec_rounds=13,
            spec_graph_replays=13,
            proposed=30,
            accepted=1,
            committed=14,
        ),
    )
    telemetry.complete_request("request-1")
    telemetry.close()

    _, completion = read_records(path)
    assert completion["kind"] == "request_complete"
    counters = completion["counters"]
    assert counters["eligible_frames"] == counters["spec_frames"] == 1
    assert counters["spec_openings"] == 1
    assert counters["spec_rounds"] == counters["spec_graph_replays"] == 13
    assert counters["proposed"] == 30
    assert counters["accepted"] == 1
    assert counters["committed"] == 14


def test_enabled_native_fallback_and_failed_requests_do_not_complete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed.jsonl"
    telemetry = P16WorkerTelemetry(
        path,
        enabled=True,
        draft_sha256="b" * 64,
        draft_loaded=True,
        pid=123,
    )
    telemetry.record_frame(
        "request-1",
        P16FrameTelemetry(
            eligible=True,
            speculative=False,
            native_reason="unexpected-fallback",
        ),
    )
    with pytest.raises(RuntimeError, match="all run speculation"):
        telemetry.complete_request("request-1")
    telemetry.discard_request("request-cancelled")
    telemetry.close()

    assert [record["kind"] for record in read_records(path)] == ["startup"]


def test_telemetry_rejects_invalid_frame_counters(
    tmp_path: Path,
) -> None:
    telemetry = P16WorkerTelemetry(
        tmp_path / "invalid.jsonl",
        enabled=True,
        draft_sha256="c" * 64,
        draft_loaded=True,
        pid=123,
    )
    with pytest.raises(ValueError, match="frame counters"):
        telemetry.record_frame(
            "request-1",
            P16FrameTelemetry(
                eligible=True,
                speculative=True,
                native_reason=None,
                spec_openings=1,
                spec_rounds=13,
                spec_graph_replays=13,
                proposed=1,
                accepted=2,
                committed=14,
            ),
        )
    telemetry.close()

# SPDX-License-Identifier: Apache-2.0
"""Run identity and finalization lifecycle tests for the request event recorder."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import (
    RequestEventBuffer,
    RequestEventRecorder,
    RequestEventSnapshot,
    get_recorder,
)
from sglang_omni.profiler.event_recorder_writer import DEFERRED_WRITER_RUN_ID_PREFIX
from sglang_omni.profiler.event_recorder_types import RecorderRunState
from tests.unit_test.profiler.event_recorder_test_utils import (
    fail_status_replace,
    read_events,
    read_status_file,
)


class RetirementInspectingWriterGate(threading.Event):
    def __init__(self, recorder: RequestEventRecorder, run_id: str) -> None:
        super().__init__()
        self.recorder = recorder
        self.run_id = run_id
        self.active_state_at_release: RecorderRunState | None = None
        self.run_id_pending_at_release = False
        self.pending_state_at_release: RecorderRunState | None = None

    def set(self) -> None:
        self.active_state_at_release = self.recorder.active_state
        self.run_id_pending_at_release = self.run_id in self.recorder.pending_states
        self.pending_state_at_release = self.recorder.pending_states.get(self.run_id)
        super().set()


def test_zero_event_retired_run_finalizes_complete(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("zero", str(tmp_path), "stage")

    finalization = recorder.deactivate()

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    status = read_status_file(finalization.status_path)
    assert read_events(path) == []
    assert status["accepted_events"] == 0
    assert status["processed_events"] == 0
    assert status["written_events"] == 0
    assert status["unwritten_events"] == 0
    assert status["dropped_events"] == 0
    assert status["diagnostic_writer_mode"] == "normal"
    assert status["complete"] is True


def test_zero_event_deferred_run_finalizes_complete(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    run_id = f"{DEFERRED_WRITER_RUN_ID_PREFIX}zero"
    path = recorder.start(run_id, str(tmp_path), "stage")
    run_state = recorder.active_state
    assert run_state is not None
    assert run_state.writer_gate is not None
    assert run_state.writer_gate.is_set() is False

    finalization = recorder.deactivate()

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    status = read_status_file(finalization.status_path)
    assert read_events(path) == []
    assert status["accepted_events"] == 0
    assert status["processed_events"] == 0
    assert status["written_events"] == 0
    assert status["unwritten_events"] == 0
    assert status["dropped_events"] == 0
    assert status["diagnostic_writer_mode"] == "deferred"
    assert status["complete"] is True


def test_deferred_gate_releases_after_retirement_registration(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    run_id = f"{DEFERRED_WRITER_RUN_ID_PREFIX}retirement-order"
    path = recorder.start(run_id, str(tmp_path), "stage")
    run_state = recorder.active_state
    assert run_state is not None
    assert run_state.writer_gate is not None
    retirement_gate = RetirementInspectingWriterGate(recorder, run_id)
    run_state.writer_gate = retirement_gate

    recorder.emit(request_id="first", stage="stage", event_name="event")
    recorder.emit(request_id="second", stage="stage", event_name="event")
    finalization = recorder.deactivate(run_id=run_id)

    assert finalization is not None
    assert retirement_gate.is_set()
    assert retirement_gate.active_state_at_release is None
    assert retirement_gate.run_id_pending_at_release is True
    assert retirement_gate.pending_state_at_release is run_state
    assert finalization.wait(timeout_seconds=5)

    events = read_events(path)
    status = read_status_file(finalization.status_path)
    assert [event["request_id"] for event in events] == ["first", "second"]
    assert status["accepted_events"] == 2
    assert status["processed_events"] == 2
    assert status["written_events"] == 2
    assert status["complete"] is True


def test_deferred_shared_stages_release_gate_only_after_final_stage(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder()
    run_id = f"{DEFERRED_WRITER_RUN_ID_PREFIX}shared"
    path = recorder.start(run_id, str(tmp_path), "first")
    recorder.start(run_id, str(tmp_path), "second")
    run_state = recorder.active_state
    assert run_state is not None
    assert run_state.writer_gate is not None
    recorder.emit(request_id="first", stage="first", event_name="event")
    recorder.emit(request_id="second", stage="second", event_name="event")

    assert recorder.release_stage("first") is None
    assert recorder.is_active() is True
    assert run_state.writer_gate.is_set() is False
    assert Path(path).exists() is False

    finalization = recorder.release_stage("second")
    assert finalization is not None
    assert run_state.writer_gate.is_set()
    assert finalization.wait(timeout_seconds=5)
    status = read_status_file(finalization.status_path)
    assert status["diagnostic_writer_mode"] == "deferred"
    assert status["accepted_events"] == 2
    assert status["processed_events"] == 2
    assert status["written_events"] == 2
    assert status["complete"] is True
    assert {event["stage"] for event in read_events(path)} == {"first", "second"}


def test_stale_expected_run_id_cannot_enter_a_later_run(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    first_path = recorder.start(
        run_id="first", event_dir=str(tmp_path / "first"), stage="stage"
    )
    recorder.start(run_id="second", event_dir=str(tmp_path / "second"), stage="stage")
    recorder.emit(
        request_id="stale",
        stage="stage",
        event_name="event",
        expected_run_id="first",
    )
    second_path = recorder.active_path()
    recorder.stop()
    assert recorder.wait_for_finalizations(timeout_seconds=5)
    assert read_events(first_path) == []
    assert second_path is not None
    assert read_events(second_path) == []


def test_buffered_event_keeps_original_run_identity_after_rotation(
    tmp_path: Path,
) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    first_path = recorder.start(
        run_id="first", event_dir=str(tmp_path / "first"), stage="stage"
    )
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="stale", metadata={"value": 1}),),
        {},
    )
    second_path = recorder.start(
        run_id="second", event_dir=str(tmp_path / "second"), stage="stage"
    )
    buffer.flush(stage="stage")
    recorder.stop()
    assert recorder.wait_for_finalizations(timeout_seconds=5)
    assert read_events(first_path) == []
    assert read_events(second_path) == []


def test_status_publication_failure_does_not_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="status", event_dir=str(tmp_path), stage="stage")

    fail_status_replace(monkeypatch)
    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is False
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False
    assert list(tmp_path.glob("run_status_*.json")) == []

    # A later successful run must not clear the recorder-level evidence that
    # the previous run failed to publish its completion sidecar.
    monkeypatch.undo()
    next_event_dir = tmp_path / "status-next"
    recorder.start(run_id="status-next", event_dir=str(next_event_dir), stage="stage")
    next_finalization = recorder.deactivate()
    assert next_finalization is not None
    assert next_finalization.wait(timeout_seconds=5) is True
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False


def test_stop_returns_none_when_finalization_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    recorder.start(run_id="status-stop", event_dir=str(tmp_path), stage="stage")

    fail_status_replace(monkeypatch)

    assert recorder.stop() is None

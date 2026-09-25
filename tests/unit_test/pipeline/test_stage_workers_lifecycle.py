# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec
from sglang_omni.scheduling.lifecycle import StartupFinalizable


class _ReadyEvent:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def set(self) -> None:
        self.events.append("ready")


class _FinalizableScheduler(StartupFinalizable):
    def __init__(self, stage_name: str, events: list[str]) -> None:
        self.stage_name = stage_name
        self.events = events

    def finalize_startup(self) -> None:
        self.events.append(f"finalize:{self.stage_name}")


class _CoincidentalScheduler:
    def __init__(self, stage_name: str, events: list[str]) -> None:
        self.stage_name = stage_name
        self.events = events

    def finalize_startup(self) -> None:
        raise AssertionError(f"unexpected finalization: {self.stage_name}")


class _FailingScheduler(StartupFinalizable):
    def finalize_startup(self) -> None:
        raise RuntimeError("startup finalization failed")


class _FakeStage:
    def __init__(
        self,
        stage_name: str,
        scheduler: StartupFinalizable | _CoincidentalScheduler,
        events: list[str],
    ) -> None:
        self.name = stage_name
        self.scheduler = scheduler
        self.events = events
        self.running = False

    async def start(self) -> None:
        self.events.append(f"start:{self.name}")
        self.running = True

    async def run(self) -> None:
        self.events.append(f"run:{self.name}")
        self.running = False

    async def stop(self) -> None:
        self.events.append(f"stop:{self.name}")
        self.running = False


class _RecordingDispatcher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def register_many(self, stages: list[_FakeStage]) -> None:
        del stages
        self.events.append("register")


def test_run_process_finalizes_only_explicit_schedulers_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    schedulers = {
        "first": _FinalizableScheduler("first", events),
        "second": _CoincidentalScheduler("second", events),
    }

    def fake_construct_stage(
        stage_spec: StageLaunchConfig,
        log: logging.Logger,
        *,
        local_dispatcher: _RecordingDispatcher,
    ) -> _FakeStage:
        del log, local_dispatcher
        events.append(f"construct:{stage_spec.stage_name}")
        return _FakeStage(
            stage_spec.stage_name,
            schedulers[stage_spec.stage_name],
            events,
        )

    monkeypatch.setattr(
        stage_workers,
        "LocalStageDispatcher",
        lambda: _RecordingDispatcher(events),
    )
    monkeypatch.setattr(stage_workers, "construct_stage", fake_construct_stage)

    stage_workers.run_process(
        StageWorkerProcessSpec(
            process_name="worker",
            stage_specs=[
                StageLaunchConfig(stage_name="first"),
                StageLaunchConfig(stage_name="second"),
            ],
        ),
        _ReadyEvent(events),
        logging.getLogger(__name__),
    )

    assert events == [
        "construct:first",
        "construct:second",
        "register",
        "finalize:first",
        "start:first",
        "start:second",
        "ready",
        "run:first",
        "run:second",
    ]


def test_run_process_propagates_finalizer_failure_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fake_construct_stage(
        stage_spec: StageLaunchConfig,
        log: logging.Logger,
        *,
        local_dispatcher: _RecordingDispatcher,
    ) -> _FakeStage:
        del log, local_dispatcher
        events.append(f"construct:{stage_spec.stage_name}")
        return _FakeStage(stage_spec.stage_name, _FailingScheduler(), events)

    monkeypatch.setattr(
        stage_workers,
        "LocalStageDispatcher",
        lambda: _RecordingDispatcher(events),
    )
    monkeypatch.setattr(stage_workers, "construct_stage", fake_construct_stage)

    with pytest.raises(RuntimeError, match="startup finalization failed"):
        stage_workers.run_process(
            StageWorkerProcessSpec(
                process_name="worker",
                stage_specs=[StageLaunchConfig(stage_name="first")],
            ),
            _ReadyEvent(events),
            logging.getLogger(__name__),
        )

    assert events == ["construct:first", "register", "stop:first"]

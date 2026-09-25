# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage.runtime import StartupFinalizableScheduler
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec


class ReadyEvent:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def set(self) -> None:
        self.events.append("ready")


class FinalizableScheduler(StartupFinalizableScheduler):
    def __init__(
        self,
        events: list[str],
        *,
        name: str = "scheduler",
        fail: bool = False,
    ) -> None:
        self.events = events
        self.name = name
        self.fail = fail

    def finalize_startup(self) -> None:
        self.events.append(f"finalize:{self.name}")
        if self.fail:
            raise RuntimeError("startup finalization failed")
        else:
            pass


class CoincidentalScheduler:
    def finalize_startup(self) -> None:
        raise AssertionError("non-participating scheduler must not be finalized")


class FakeStage:
    def __init__(
        self,
        stage_name: str,
        scheduler: StartupFinalizableScheduler | CoincidentalScheduler,
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


class RecordingDispatcher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def register_many(self, stages: list[FakeStage]) -> None:
        del stages
        self.events.append("register")


def run_worker(
    monkeypatch: pytest.MonkeyPatch,
    stage_schedulers: dict[str, StartupFinalizableScheduler | CoincidentalScheduler],
    events: list[str],
    stage_names: list[str],
) -> None:
    def fake_construct_stage(
        stage_spec: StageLaunchConfig,
        log: logging.Logger,
        *,
        local_dispatcher: RecordingDispatcher,
    ) -> FakeStage:
        del log, local_dispatcher
        events.append(f"construct:{stage_spec.stage_name}")
        return FakeStage(
            stage_spec.stage_name,
            stage_schedulers[stage_spec.stage_name],
            events,
        )

    monkeypatch.setattr(
        stage_workers,
        "LocalStageDispatcher",
        lambda: RecordingDispatcher(events),
    )
    monkeypatch.setattr(stage_workers, "construct_stage", fake_construct_stage)

    stage_workers.run_process(
        StageWorkerProcessSpec(
            process_name="worker",
            stage_specs=[StageLaunchConfig(stage_name=name) for name in stage_names],
        ),
        ReadyEvent(events),
        logging.getLogger(__name__),
    )


def test_run_process_finalizes_explicit_schedulers_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    run_worker(
        monkeypatch,
        {
            "first": FinalizableScheduler(events, name="first"),
            "second": CoincidentalScheduler(),
        },
        events,
        ["first", "second"],
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


def test_run_process_stops_startup_when_finalization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    with pytest.raises(RuntimeError, match="startup finalization failed"):
        run_worker(
            monkeypatch,
            {"first": FinalizableScheduler(events, fail=True)},
            events,
            ["first"],
        )

    assert events == ["construct:first", "register", "stop:first"]

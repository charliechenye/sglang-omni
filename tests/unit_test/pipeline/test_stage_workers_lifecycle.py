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
        lifecycle_events: list[str],
        *,
        name: str,
        should_fail: bool = False,
    ) -> None:
        self.lifecycle_events = lifecycle_events
        self.name = name
        self.should_fail = should_fail

    def finalize_startup(self) -> None:
        self.lifecycle_events.append(f"finalize:{self.name}")
        if self.should_fail:
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
        lifecycle_events: list[str],
    ) -> None:
        self.name = stage_name
        self.scheduler = scheduler
        self.lifecycle_events = lifecycle_events
        self.running = False

    async def start(self) -> None:
        self.lifecycle_events.append(f"start:{self.name}")
        self.running = True

    async def run(self) -> None:
        self.lifecycle_events.append(f"run:{self.name}")
        self.running = False

    async def stop(self) -> None:
        self.lifecycle_events.append(f"stop:{self.name}")
        self.running = False


class RecordingDispatcher:
    def __init__(self, lifecycle_events: list[str]) -> None:
        self.lifecycle_events = lifecycle_events

    def register_many(self, stages: list[FakeStage]) -> None:
        if stages:
            self.lifecycle_events.append("register")
        else:
            raise AssertionError("stage worker registered no stages")


def run_stage_worker(
    monkeypatch: pytest.MonkeyPatch,
    stage_schedulers: dict[str, StartupFinalizableScheduler | CoincidentalScheduler],
    lifecycle_events: list[str],
    stage_names: list[str],
) -> None:
    def fake_construct_stage(
        stage_spec: StageLaunchConfig,
        log: logging.Logger,
        *,
        local_dispatcher: RecordingDispatcher,
    ) -> FakeStage:
        assert log is not None
        assert local_dispatcher is not None
        lifecycle_events.append(f"construct:{stage_spec.stage_name}")
        return FakeStage(
            stage_spec.stage_name,
            stage_schedulers[stage_spec.stage_name],
            lifecycle_events,
        )

    monkeypatch.setattr(
        stage_workers,
        "LocalStageDispatcher",
        lambda: RecordingDispatcher(lifecycle_events),
    )
    monkeypatch.setattr(stage_workers, "construct_stage", fake_construct_stage)

    stage_workers.run_process(
        StageWorkerProcessSpec(
            process_name="worker",
            stage_specs=[StageLaunchConfig(stage_name=name) for name in stage_names],
        ),
        ReadyEvent(lifecycle_events),
        logging.getLogger(__name__),
    )


def test_run_process_finalizes_explicit_schedulers_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle_events: list[str] = []

    run_stage_worker(
        monkeypatch,
        {
            "first": FinalizableScheduler(lifecycle_events, name="first"),
            "second": CoincidentalScheduler(),
        },
        lifecycle_events,
        ["first", "second"],
    )

    finalize_index = lifecycle_events.index("finalize:first")
    assert lifecycle_events.index("construct:first") < finalize_index
    assert lifecycle_events.index("construct:second") < finalize_index
    assert lifecycle_events.index("register") < finalize_index
    assert finalize_index < lifecycle_events.index("start:first")
    assert finalize_index < lifecycle_events.index("start:second")
    assert finalize_index < lifecycle_events.index("ready")
    assert "finalize:second" not in lifecycle_events


def test_run_process_stops_startup_when_finalization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle_events: list[str] = []

    with pytest.raises(RuntimeError, match="startup finalization failed"):
        run_stage_worker(
            monkeypatch,
            {
                "first": FinalizableScheduler(
                    lifecycle_events,
                    name="first",
                    should_fail=True,
                )
            },
            lifecycle_events,
            ["first"],
        )

    assert "finalize:first" in lifecycle_events
    assert "start:first" not in lifecycle_events
    assert "ready" not in lifecycle_events
    assert "stop:first" in lifecycle_events
    assert lifecycle_events.index("finalize:first") < lifecycle_events.index(
        "stop:first"
    )

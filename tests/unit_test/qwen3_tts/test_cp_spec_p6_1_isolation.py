# SPDX-License-Identifier: Apache-2.0
"""CPU-only P6.1 reference and staged subprocess reporting guards."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.qwen3_tts_cp_spec_k2.p6_1_conditional_isolation import (
    FAIL, PASS, STAGES, coordinator, cpu_depth_reference, masks,
)


def test_reference_mask_census_and_dynamic_skips() -> None:
    decisions = masks()
    assert len(decisions) == 5
    assert cpu_depth_reference(decisions["all_reject"]) == {
        "depth": 15, "rounds": 14, "accepted": 0, "bonus": 0,
    }
    assert cpu_depth_reference(decisions["all_accept"]) == {
        "depth": 15, "rounds": 7, "accepted": 7, "bonus": 7,
    }
    for name, pattern in decisions.items():
        result = cpu_depth_reference(pattern)
        assert result["depth"] == 15, name
        assert 7 <= result["rounds"] <= 14
        assert result["rounds"] + result["bonus"] == 14
        assert result["accepted"] <= result["rounds"]


def test_invalid_cpu_masks_fail() -> None:
    with pytest.raises(ValueError, match="16 binary"):
        cpu_depth_reference([1] * 15)
    with pytest.raises(ValueError, match="16 binary"):
        cpu_depth_reference([1] * 15 + [2])


def test_coordinator_stops_at_first_failed_capture_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd[cmd.index("--stage") + 1])
        out = Path(cmd[cmd.index("--stage-out") + 1])
        out.mkdir(parents=True)
        (out / "report.json").write_text(json.dumps({
            "status": "FAILED",
            "first_error_type": "RuntimeError",
            "first_error_message": "First capture error",
        }))
        return subprocess.CompletedProcess(
            cmd, returncode=2, stdout="first error\n", stderr="stack\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result_dir = tmp_path / "new-run"
    assert coordinator(result_dir, timeout=60) == 2
    assert seen == [STAGES[0]]
    report = json.loads((result_dir / "report.json").read_text())
    assert report["status"] == FAIL
    assert len(report["stages"]) == 1
    assert report["stages"][0]["first_error_message"] == "First capture error"
    assert "First capture error" in (result_dir / "summary.md").read_text()
    with pytest.raises(FileExistsError):
        coordinator(result_dir, timeout=60)


def test_coordinator_never_calls_gpu_after_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    def fake_run(cmd, **kwargs):
        stage = cmd[cmd.index("--stage") + 1]
        seen.append(stage)
        path = Path(cmd[cmd.index("--stage-out") + 1])
        path.mkdir(parents=True)
        status = "PASS" if stage == "plain_graph" else "FAILED"
        (path / "report.json").write_text(json.dumps({"status": status}))
        return subprocess.CompletedProcess(cmd, 0 if status == "PASS" else 2, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert coordinator(tmp_path / "partial", timeout=60) == 2
    assert seen == ["plain_graph", "while_minimal"]


def test_all_stages_must_succeed_for_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd, **kwargs):
        path = Path(cmd[cmd.index("--stage-out") + 1])
        path.mkdir(parents=True)
        (path / "report.json").write_text(json.dumps({"status": "PASS"}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert coordinator(tmp_path / "all", timeout=60) == 0
    report = json.loads((tmp_path / "all" / "report.json").read_text())
    assert report["status"] == PASS
    assert len(report["stages"]) == 3

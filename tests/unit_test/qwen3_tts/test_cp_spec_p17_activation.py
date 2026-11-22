# SPDX-License-Identifier: Apache-2.0
"""P17 activation diagnostics are bounded, exact, and never a parity gate."""
from __future__ import annotations

import json

import pytest

from experiments.qwen3_tts_cp_spec_k2.p17_activation_smoke import (
    STATUS_ARM_MISSING,
    STATUS_ENTERED,
    STATUS_NATIVE,
    STATUS_NOT_SEEN,
    STATUS_PARTIAL,
    _classify,
)
from sglang_omni.models.qwen3_tts.code_predictor_spec_p17_activation import (
    _COUNTS,
    MAX_PER_STATE,
    record,
)


@pytest.mark.parametrize(
    "events,status",
    [
        ([], STATUS_NOT_SEEN),
        ([{"event": "model_runner_collect_before_predictor"}], STATUS_PARTIAL),
        ([{"event": "code_predictor_forward_enter", "spec_arm_exists": False}],
         STATUS_ARM_MISSING),
        ([{"event": "code_predictor_forward_enter", "spec_arm_exists": True},
          {"event": "code_predictor_forward_native_selected"}],
         STATUS_NATIVE),
        ([{"event": "code_predictor_forward_enter", "spec_arm_exists": True},
          {"event": "speculative_k2_enter"}], STATUS_ENTERED),
    ],
)
def test_p17_activation_classification_is_not_gpu_parity(events, status):
    assert _classify(events) == status


def test_p17_activation_records_true_file_state_and_is_bounded(
    tmp_path, monkeypatch,
):
    _COUNTS.clear()
    folder = tmp_path / "activation"
    spec = tmp_path / "ARM_SPEC"
    base = tmp_path / "ARM_BASE"
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR", str(folder))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE", str(spec))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE", str(base))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_SPEC_K", "2")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "a" * 40)
    for _ in range(MAX_PER_STATE + 2):
        record("code_predictor_forward_enter", caller_module=__file__)
    spec.touch()
    for _ in range(MAX_PER_STATE + 2):
        record("code_predictor_forward_enter", caller_module=__file__)
    files = list(folder.glob("dispatch_pid*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(rows) == 2 * MAX_PER_STATE
    assert all(row["source_sha"] == "a" * 40 for row in rows)
    assert all(row["spec_k"] == "2" for row in rows)
    assert all(not row["spec_arm_exists"] for row in rows[:MAX_PER_STATE])
    assert all(row["spec_arm_exists"] for row in rows[MAX_PER_STATE:])
    assert all(row["spec_arm_path"] == str(spec) for row in rows)
    _COUNTS.clear()


def test_p17_activation_disabled_is_noop(tmp_path, monkeypatch):
    _COUNTS.clear()
    monkeypatch.delenv("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR", raising=False)
    record("speculative_k2_enter", caller_module=__file__)
    assert not list(tmp_path.iterdir())
    assert not _COUNTS

# SPDX-License-Identifier: Apache-2.0
"""P22B CPU-only CUDA API/NVTX and native Graph-on ownership regressions."""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.qwen3_tts_cp_spec_k2.p22b_cuda_api_nvtx import (
    analyze_sqlite, classify_api, match_api_stage, STATUS,
)
from sglang_omni.models.qwen3_tts import code_predictor_spec_p22b_native as native


def test_p22b_cuda_api_classifications():
    assert classify_api("cudaLaunchKernel") == "kernel_launch"
    assert classify_api("cuLaunchKernelEx") == "kernel_launch"
    assert classify_api("cudaGraphLaunch_v10000") == "graph_launch"
    assert classify_api("cudaMemcpyAsync") == "memcpy"
    assert classify_api("cudaStreamSynchronize") == "sync_or_wait"
    assert classify_api("cudaMemsetAsync") == "memset"


def _fixture_sqlite(path):
    db = sqlite3.connect(path)
    db.executescript("""
CREATE TABLE StringIds (id INTEGER, value TEXT);
CREATE TABLE NVTX_EVENTS (
 start INTEGER, end INTEGER, eventType INTEGER, globalTid INTEGER,
 text TEXT, textId INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
 start INTEGER, end INTEGER, globalTid INTEGER,
 correlationId INTEGER, nameId INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (
 start INTEGER, end INTEGER, globalTid INTEGER,
 correlationId INTEGER, nameId INTEGER
);
""")
    tid = (123 << 24) | 7
    db.executemany("INSERT INTO StringIds VALUES (?,?)", [
        (1, "cudaLaunchKernel"),
        (2, "cudaMemcpyAsync"),
        (3, "cudaStreamSynchronize"),
        (4, "cudaGraphLaunch"),
        (5, "cuLaunchKernel"),
    ])
    db.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,NULL)",
        [
            (0, 1000, 59, tid, "P22_FULL_CODE_PREDICTOR"),
            (100, 300, 59, tid, "P22_Q_PROPOSALS"),
            (310, 500, 59, tid, "P22_WIDE_VERIFIER"),
            (510, 900, 59, tid, "P22_TARGET_SUPPORT_ACCEPT_BONUS"),
            (1200, 1500, 59, tid, "P22B_NATIVE_CODE_PREDICTOR"),
        ],
    )
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
        [
            (120, 135, tid, 11, 1),
            (350, 370, tid, 12, 1),
            (520, 550, tid, 13, 2),
            (600, 680, tid, 14, 3),
            (1210, 1230, tid, 15, 4),
            (1600, 1700, tid, 16, 2),
        ],
    )
    # The driver API entry for cid=11 must not double count the CUDA launch.
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (?,?,?,?,?)",
        [(121, 135, tid, 11, 5), (850, 875, tid, 17, 5)],
    )
    db.commit()
    db.close()


def test_p22b_real_nsight_cpu_api_attribution_is_not_gpu_time(tmp_path):
    db = tmp_path / "trace.sqlite"
    _fixture_sqlite(db)
    report = analyze_sqlite(db)
    assert report["status"] == STATUS
    assert report["counts"]["driver_duplicate_runtime_skipped"] == 1
    assert report["counts"]["outside_p22_nvtx"] == 1
    assert report["counts"]["assigned_p22_nvtx_calls"] == 6
    assert report["stages"]["P22_Q_PROPOSALS"]["kernel_launch"][
        "summed_cpu_api_ms"
    ] == pytest.approx(0.000015)
    assert report["stages"]["P22_TARGET_SUPPORT_ACCEPT_BONUS"][
        "sync_or_wait"
    ]["calls"] == 1
    assert report["stages"]["P22B_NATIVE_CODE_PREDICTOR"][
        "graph_launch"
    ]["calls"] == 1
    with pytest.raises(RuntimeError, match="MISSING_CUDA_API_STAGE"):
        analyze_sqlite(db, required_stages=("P22_MISSING",))
    # API timestamps live on the CPU thread, not the GPU.
    assert match_api_stage(
        {123: {"starts": [], "rows": [], "max_end": []}},
        456, 100, 150,
    ) is None


def _native_env(monkeypatch, tmp_path):
    arm = tmp_path / "NATIVE_ARM"
    out = tmp_path / "native_records"
    monkeypatch.setenv(native.ENABLE, "1")
    monkeypatch.setenv(native.ARM, str(arm))
    monkeypatch.setenv(native.DIR, str(out))
    monkeypatch.setenv(native.SOURCE, "a" * 40)
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P21_SERVING_K2", "1")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P21_ARM_FILE",
                      str(tmp_path / "ABSENT_K2_ARM"))
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P22_PROFILE", "0")
    return arm, out


def test_p22b_native_capture_one_armed_real_request_only(tmp_path, monkeypatch):
    import torch

    arm, output = _native_env(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(torch.cuda, "cudart", lambda: SimpleNamespace(
        cudaProfilerStart=lambda: calls.append("start") or 0,
        cudaProfilerStop=lambda: calls.append("stop") or 0,
    ))
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda d: calls.append("sync"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_push",
                        lambda name: calls.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop",
                        lambda: calls.append(("pop",)))
    m = SimpleNamespace(predictor_device="mock_device")
    assert native.start_if_armed(m, "rid1") is False
    assert calls == []
    arm.touch()
    assert native.start_if_armed(m, "rid1") is True
    assert native.start_if_armed(m, "rid1") is True
    with native.predictor_span(m, "rid1"):
        pass
    with native.predictor_span(m, "rid1"):
        pass
    expected = {
        "native_graph_replays": 2,
        "predictor_calls": 2,
        "candidate_frames": 0,
    }
    result = native.stop_if_captured(
        m, request_id="rid1", stats=expected,
    )
    assert result["actual_predictor_calls"] == 2
    assert result["cuda_profiler_capture"] is True
    assert result["cpu_predictor_per_frame_ms"] > 0
    assert calls.count("start") == 1
    assert calls.count("stop") == 1
    assert calls.count("sync") == 1
    assert len(list(output.glob("request_*.json"))) == 1
    with pytest.raises(RuntimeError, match="ONLY_ONE_CAPTURE"):
        native.start_if_armed(m, "rid1")


def test_p22b_native_rejects_spec_arm_and_missing_graph(tmp_path, monkeypatch):
    import torch

    arm, _ = _native_env(monkeypatch, tmp_path)
    arm.touch()
    m = SimpleNamespace()
    k2 = tmp_path / "ABSENT_K2_ARM"
    k2.touch()
    with pytest.raises(RuntimeError, match="REFUSES_K2_CANDIDATE_ARM"):
        native.start_if_armed(m, "rid")
    k2.unlink()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: SimpleNamespace(
        cudaProfilerStart=lambda: 0,
        cudaProfilerStop=lambda: 0,
    ))
    native.start_if_armed(m, "rid")
    m._p22b_native_host_pred_calls = 2
    with pytest.raises(RuntimeError, match="GRAPH_AND_FRAME_COVERAGE_FAIL"):
        native.stop_if_captured(
            m, request_id="rid",
            stats={"candidate_frames": 1, "native_graph_replays": 0,
                   "predictor_calls": 2},
        )


def test_p22b_bash_launcher_parses_and_keeps_native_arm_absent():
    root = Path(__file__).resolve().parents[3]
    s = root / "experiments/qwen3_tts_cp_spec_k2/p22b_native_profile_launch.sh"
    p = subprocess.run(["bash", "-n", str(s)], text=True,
                       capture_output=True)
    assert p.returncode == 0, p.stderr
    src = s.read_text()
    assert "experiment/qwen3-tts-p22b-nvidia-gap" in src
    assert "--capture-range=cudaProfilerApi" in src
    assert "--target-processes=all" not in src
    assert 'SGLANG_OMNI_QTTS_CP_SPEC_K=0' in src
    assert 'SGLANG_OMNI_QTTS_CP_P22_PROFILE=0' in src
    assert "P21_ARM_STAYS_ABSENT" in src

# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic Nsight SQLite stage attribution with async kernel timing."""
from __future__ import annotations

import sqlite3

import pytest

from experiments.qwen3_tts_cp_spec_k2.p22_nsys_sqlite import (
    SUCCESS, load_nsight,
)


def test_p22_nsys_runtime_launch_range_not_gpu_timestamp(tmp_path):
    path = tmp_path / "p22.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
CREATE TABLE StringIds (id INTEGER, value TEXT);
CREATE TABLE NVTX_EVENTS (
    start INTEGER, end INTEGER, eventType INTEGER,
    globalTid INTEGER, text TEXT, textId INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (
    start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start INTEGER, end INTEGER, globalPid INTEGER,
    correlationId INTEGER, shortName INTEGER
);
""")
    tid = (123 << 24) | 7
    pid = 123 << 24
    otherpid = 456 << 24
    db.executemany(
        "INSERT INTO StringIds VALUES (?,?)",
        [(1, "kernel_q"), (2, "kernel_wide"), (3, "kernel_target")]
    )
    db.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,NULL)",
        [
            (50, 1000, 59, tid, "P22_FULL_CODE_PREDICTOR"),
            (100, 300, 59, tid, "P22_Q_PROPOSALS"),
            (310, 490, 59, tid, "P22_WIDE_VERIFIER"),
            (500, 680, 59, tid, "P22_TARGET_SUPPORT_ACCEPT_BONUS"),
        ]
    )
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
        [(160, 170, tid, 111), (340, 350, tid, 222),
         (530, 540, tid, 333)]
    )
    # A driver-level wrapper can share the same ID. Runtime API wins.
    db.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (?,?,?,?)",
        (150, 180, tid, 111)
    )
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
        [
            # GPU start AFTER q NVTX ends. Correct mapping uses API time!
            (500, 650, pid, 111, 1),
            (710, 960, pid, 222, 2),
            # The third required group must also have a real CUDA API
            # launch and a GPU kernel. Its GPU execution is deliberately
            # AFTER its CPU NVTX range to check async launch correlation.
            (1200, 1290, pid, 333, 3),
            # SAME correlation ID in a different process cannot inherit
            # the original worker's NVTX range.
            (1000, 1110, otherpid, 111, 1),
        ],
    )
    db.commit()
    db.close()
    report, top = load_nsight(path)
    assert report["status"] == SUCCESS
    assert report["counters"]["all_cuda_kernels"] == 4
    assert report["counters"]["assigned_p22_nvtx_kernels"] == 3
    assert report["counters"]["missing_cuda_api_correlation"] == 1
    assert report["stage_sums"]["P22_Q_PROPOSALS"]["kernels"] == 1
    assert report["stage_sums"]["P22_WIDE_VERIFIER"]["kernels"] == 1
    assert report["stage_sums"]["P22_TARGET_SUPPORT_ACCEPT_BONUS"]["kernels"] == 1
    assert report["stage_sums"]["P22_TARGET_SUPPORT_ACCEPT_BONUS"][
        "summed_kernel_work_ms"
    ] == pytest.approx(0.00009)
    assert report["stage_sums"]["P22_Q_PROPOSALS"][
        "summed_kernel_work_ms"
    ] == 0.00015
    assert top["top_gpu_kernel_work"][0]["stage"] == "P22_WIDE_VERIFIER"

    # Preserve the fail-closed production evidence contract: a valid
    # Q+verifier trace without the target/accept/bonus kernel is NOT PASS.
    db = sqlite3.connect(path)
    db.execute(
        "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId = 333"
    )
    db.commit()
    db.close()
    with pytest.raises(
        RuntimeError,
        match="P22_NSYS_REQUIRED_REAL_K2_KERNEL_GROUP_MISSING "
              "P22_TARGET_SUPPORT_ACCEPT_BONUS",
    ):
        load_nsight(path)


def test_p22_nsys_rejects_corrupted_missing_nvtx_schema(tmp_path):
    path = tmp_path / "invalid.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="REQUIRED_TABLES_MISSING"):
        load_nsight(path)

# SPDX-License-Identifier: Apache-2.0
"""CPU-only Nsight postmortem tests; no H200 nor .nsys-rep required."""
from __future__ import annotations

import json
import sqlite3

from experiments.qwen3_tts_cp_spec_k2.p16a_nsys_diagnose import (
    classify, describe_sqlite,
)

PID = 4131882
TID = 7
GLOBAL_TID = (PID << 24) | TID


def create_sample_sqlite(path, *, with_nvtx=True, with_kernel=True, with_node=True):
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS(
            start INTEGER, end INTEGER, textId INTEGER, text TEXT,
            globalTid INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER, end INTEGER, nameId INTEGER, globalTid INTEGER,
            correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER, end INTEGER, shortName INTEGER,
            correlationId INTEGER, graphNodeId INTEGER, graphId INTEGER
        );
        """
    )
    db.executemany(
        "INSERT INTO StringIds VALUES(?, ?)",
        [
            (1, "P16A_CASE_RR_CONDITIONAL"),
            (2, "cudaGraphLaunch"),
            (3, "void p16a_round_start_marker_kernel()"),
            (4, "void p16a_bonus_head_marker_kernel()"),
        ],
    )
    if with_nvtx:
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES(100, 1000, 1, NULL, ?)",
            (GLOBAL_TID,),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES("
            "200, 220, 2, ?, 17)",
            (GLOBAL_TID,),
        )
    if with_kernel:
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES("
            "250, 310, 3, 17, ?, 31)",
            (44 if with_node else None,),
        )
    db.commit()
    db.close()


def test_p16a_postmortem_census_recovers_actual_saved_trace_schema(tmp_path):
    path = tmp_path / "trace.sqlite"
    create_sample_sqlite(path)
    result = describe_sqlite(path, expected_pid=PID)
    assert result["p16a_nvtx_ranges"] == {"P16A_CASE_RR_CONDITIONAL": 1}
    assert result["p16a_nvtx_pid_census"][str(PID)] == 1
    assert result["graph_node_kernel_count"] == 1
    assert result["classification"] == (
        "P16A_NVTX_AND_GPU_NODE_DATA_PRESENT_CHECK_LAUNCH_CORRELATION"
    )
    correlations = result["graph_launch_correlations_DIAGNOSTIC_NOT_PROOF"]
    assert len(correlations) == 1
    assert correlations[0]["kernel_counts_by_correlation_candidate"] == {
        "17": 1
    }
    assert json.dumps(result)


def test_p16a_postmortem_no_nvtx_or_missing_node_never_passes(tmp_path):
    cases = [
        ("missing_nvtx", False, True, True,
         "P16A_NVTX_CASE_RANGES_MISSING_OR_NOT_CAPTURED"),
        ("missing_kernel", True, False, True,
         "P16A_GPU_KERNEL_ACTIVITY_MISSING"),
        ("missing_graph_node", True, True, False,
         "P16A_CUDA_GRAPH_NODE_ACTIVITY_MISSING"),
    ]
    for label, has_nvtx, has_k, has_node, expected in cases:
        path = tmp_path / f"{label}.sqlite"
        create_sample_sqlite(
            path, with_nvtx=has_nvtx,
            with_kernel=has_k, with_node=has_node,
        )
        result = describe_sqlite(path, expected_pid=PID)
        assert result["classification"] == expected


def test_p16a_postmortem_wrong_pid_must_remain_incomplete(tmp_path):
    path = tmp_path / "mismatch.sqlite"
    create_sample_sqlite(path)
    result = describe_sqlite(path, expected_pid=PID + 1)
    assert result["classification"] == "P16A_NVTX_WORKER_PID_NOT_IN_CAPTURE"
    assert result["graph_node_kernel_count"] == 1

# SPDX-License-Identifier: Apache-2.0
"""Inspect a SAVED P16A Nsight report without any H200 rerun.

Diagnostic only: NVTX events, CUDA kernels, graph-node fields and exact
host CUDA-graph launch correlations are examined separately. NO GPU kernel
skip claim is made by this script. Reports and SQLite schema vary with
Nsight version; preserve all raw outputs for a future verified adapter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

REPORTS = ("nvtx_kern_sum", "nvtx_sum", "cuda_gpu_kern_sum", "cuda_api_sum")
TABLES = (
    "NVTX_EVENTS",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_DRIVER",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    "CUDA_GRAPH_NODE_EVENTS",
    "CUDA_GRAPH_EVENTS",
    "StringIds",
)


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def describe_sqlite(path: Path, *, expected_pid: int | None = None) -> dict:
    """Inspect only the source report's schema and high-level row census."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        present = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        census = {}
        for table in TABLES:
            if table not in present:
                census[table] = {"present": False}
                continue
            cols = _columns(conn, table)
            n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            census[table] = {"present": True, "rows": n, "columns": cols}
        nvtx_names = []
        nvtx_pids = {}
        nvtx_case_ranges = []
        if "NVTX_EVENTS" in present:
            cols = set(_columns(conn, "NVTX_EVENTS"))
            if "text" in cols:
                expr = "NULLIF(n.text,'')"
            else:
                expr = "NULL"
            if "textId" in cols and "StringIds" in present:
                expr = f"COALESCE({expr}, s.value)"
                join = " LEFT JOIN StringIds s ON n.textId=s.id"
            else:
                join = ""
            if "start" in cols and "end" in cols:
                tid = "n.globalTid" if "globalTid" in cols else "NULL"
                select = (
                    f"SELECT {expr} AS label, n.start, n.end, {tid} "
                    f"FROM NVTX_EVENTS n{join} "
                    "WHERE n.start IS NOT NULL AND n.end IS NOT NULL"
                )
                for label, start, end, global_tid in conn.execute(select):
                    if not isinstance(label, str):
                        continue
                    if label.startswith("P16A_"):
                        pid = (
                            (int(global_tid) // 0x1000000) % 0x1000000
                            if global_tid is not None else None
                        )
                        nvtx_names.append(label)
                        nvtx_case_ranges.append({
                            "label": label, "start": start, "end": end,
                            "global_tid": global_tid, "decoded_pid": pid,
                        })
                        nvtx_pids[str(pid)] = nvtx_pids.get(str(pid), 0) + 1
        kernel_names = []
        kernel_node_count = None
        kernel_graph_counts = {}
        if "CUPTI_ACTIVITY_KIND_KERNEL" in present:
            cols = set(_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL"))
            if "graphNodeId" in cols:
                kernel_node_count = conn.execute(
                    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                    "WHERE graphNodeId IS NOT NULL AND graphNodeId != 0"
                ).fetchone()[0]
            for field in ("graphId", "graphNodeId", "correlationId"):
                if field in cols:
                    kernel_graph_counts[field] = conn.execute(
                        f"SELECT COUNT(DISTINCT {field}) "
                        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
                    ).fetchone()[0]
            if "shortName" in cols and "StringIds" in present:
                kernel_names = [
                    {"name": name, "count": count}
                    for name, count in conn.execute(
                        "SELECT s.value, COUNT(*) "
                        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                        "JOIN StringIds s ON k.shortName=s.id "
                        "WHERE s.value LIKE '%p16a_%' "
                        "GROUP BY s.value ORDER BY COUNT(*) DESC"
                    )
                ]
        # Attribute *candidate* graphLaunch calls to exact same-thread NVTX
        # ranges and CUDA kernel correlation IDs. These are diagnostic
        # candidates, not proof of node attribution or skipped LM head.
        launch_candidates = []
        if nvtx_case_ranges:
            for item in nvtx_case_ranges:
                graph_calls = []
                for api_table in (
                    "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"
                ):
                    if api_table not in present or "StringIds" not in present:
                        continue
                    cols = set(_columns(conn, api_table))
                    required = {
                        "start", "end", "nameId", "globalTid", "correlationId",
                    }
                    if not required.issubset(cols):
                        continue
                    for corr, name in conn.execute(
                        f"SELECT a.correlationId, s.value "
                        f"FROM {api_table} a "
                        "JOIN StringIds s ON a.nameId=s.id "
                        "WHERE a.start>=? AND a.end<=? "
                        "AND a.globalTid=? AND LOWER(s.value) LIKE '%graphlaunch%'",
                        (item["start"], item["end"], item["global_tid"]),
                    ):
                        graph_calls.append({
                            "api_table": api_table, "correlationId": corr,
                            "name": name,
                        })
                counts = {}
                if graph_calls and "CUPTI_ACTIVITY_KIND_KERNEL" in present:
                    cols = set(_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL"))
                    if "correlationId" in cols:
                        for call in graph_calls:
                            cid = call["correlationId"]
                            n = conn.execute(
                                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                                "WHERE correlationId=?", (cid,)
                            ).fetchone()[0]
                            counts[str(cid)] = n
                launch_candidates.append({
                    "nvtx_range": item["label"],
                    "decoded_pid": item["decoded_pid"],
                    "expected_pid_match": (
                        item["decoded_pid"] == expected_pid
                        if expected_pid is not None else None
                    ),
                    "same_thread_graph_launch_calls": graph_calls,
                    "kernel_counts_by_correlation_candidate": counts,
                })
        return {
            "source_sqlite": str(path),
            "table_census": census,
            "p16a_nvtx_ranges": dict(Counter(nvtx_names)),
            "p16a_nvtx_pid_census": nvtx_pids,
            "graph_node_kernel_count": kernel_node_count,
            "kernel_distinct_ids": kernel_graph_counts,
            "physical_marker_kernel_names": kernel_names,
            "graph_launch_correlations_DIAGNOSTIC_NOT_PROOF": launch_candidates,
            "classification": classify(
                nvtx_names, census, kernel_node_count, expected_pid, nvtx_pids,
            ),
        }
    finally:
        conn.close()


def classify(names, census, graph_nodes, expected_pid, nvtx_pids) -> str:
    if not names:
        return "P16A_NVTX_CASE_RANGES_MISSING_OR_NOT_CAPTURED"
    if expected_pid is not None and str(expected_pid) not in nvtx_pids:
        return "P16A_NVTX_WORKER_PID_NOT_IN_CAPTURE"
    if census.get("CUPTI_ACTIVITY_KIND_KERNEL", {}).get("rows", 0) == 0:
        return "P16A_GPU_KERNEL_ACTIVITY_MISSING"
    if graph_nodes is None:
        return "P16A_CUDA_GRAPH_NODE_COLUMN_MISSING"
    if graph_nodes == 0:
        return "P16A_CUDA_GRAPH_NODE_ACTIVITY_MISSING"
    return "P16A_NVTX_AND_GPU_NODE_DATA_PRESENT_CHECK_LAUNCH_CORRELATION"


def _run(args: list[str]) -> dict:
    try:
        process = subprocess.run(
            args, capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        return {
            "command": args, "returncode": None,
            "stdout": "", "stderr": repr(exc),
        }
    return {
        "command": args, "returncode": process.returncode,
        "stdout": process.stdout, "stderr": process.stderr,
    }


def diagnose(report: Path, out: Path, *, expected_pid: int | None = None):
    if out.exists():
        raise FileExistsError(f"P16A diagnostic output exists: {out}")
    if not report.is_file() or report.stat().st_size == 0:
        raise FileNotFoundError(f"P16A missing Nsight report: {report}")
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    for name in REPORTS:
        data = _run([
            "nsys", "stats", f"--report={name}", "--format=csv",
            str(report),
        ])
        (out / f"{name}.stdout.txt").write_text(data.pop("stdout"))
        (out / f"{name}.stderr.txt").write_text(data.pop("stderr"))
        checks[name] = {
            **data, "stdout_file": f"{name}.stdout.txt",
            "stderr_file": f"{name}.stderr.txt",
        }
    version = _run(["nsys", "--version"])
    (out / "nsys_version.txt").write_text(
        version["stdout"] + version["stderr"]
    )
    sqlite_out = out / "p16a_trace.sqlite"
    exported = _run([
        "nsys", "export", "--type=sqlite",
        f"--output={sqlite_out}", str(report),
    ])
    (out / "export.stdout.txt").write_text(exported["stdout"])
    (out / "export.stderr.txt").write_text(exported["stderr"])
    hasher = hashlib.sha256()
    with report.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    trace_sha = hasher.hexdigest()
    payload = {
        "status": "P16A_NSYS_POSTMORTEM_DIAGNOSTIC_ONLY",
        "source_report": str(report),
        "source_report_sha256": trace_sha,
        "expected_pid": expected_pid,
        "nsys_version": version["stdout"].strip() or version["stderr"].strip(),
        "stats_command_results": checks,
        "export_returncode": exported["returncode"],
        "nsys_sqlite": None,
    }
    if exported["returncode"] == 0 and sqlite_out.is_file():
        try:
            payload["nsys_sqlite"] = describe_sqlite(
                sqlite_out, expected_pid=expected_pid,
            )
        except (sqlite3.Error, ValueError, TypeError) as exc:
            payload["status"] = "P16A_NSYS_SQLITE_SCHEMA_UNSUPPORTED"
            payload["sqlite_diagnostic_error"] = repr(exc)
    else:
        payload["status"] = "P16A_NSYS_SQLITE_EXPORT_UNAVAILABLE"
    (out / "diagnosis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"P16A_DIAGNOSIS={out/'diagnosis.json'}")
    print(
        "P16A_STATUS=" +
        (
            payload["nsys_sqlite"]["classification"]
            if payload["nsys_sqlite"] else payload["status"]
        )
    )
    print("P16A_NO_KERNEL_SKIP_CLAIM=true")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-pid", type=int)
    args = parser.parse_args()
    diagnose(
        args.nsys_report.expanduser().resolve(),
        args.out.expanduser().resolve(),
        expected_pid=args.expected_pid,
    )


if __name__ == "__main__":
    main()

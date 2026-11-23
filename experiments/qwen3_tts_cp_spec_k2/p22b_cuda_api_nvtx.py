# SPDX-License-Identifier: Apache-2.0
"""P22B read-only Nsight SQLite CUDA-API -> innermost P22 NVTX attribution.

Purpose: distinguish C++/Python CUDA API launch, H2D/D2H memcpy, and
stream-sync CPU-side overhead. A CUDA API's host start/end is matched to
its SAME THREAD'S enclosing NVTX scope; GPU kernel timestamps must never
be used for CPU NVTX containment. API durations are per-thread sums and
NOT the request critical path, GPU kernel time, or achievable savings.

Only source-pinned previously completed P22 Nsight runs are accepted.
No GPU, CUDA profiler, or server is needed for this analysis.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from .p22_nsys_sqlite import (
    API_TABLES, NVTX_TABLE, STRING_TABLE, _columns,
)

SCHEMA = "qwen3tts-p22b-nvtx-cuda-api-v1"
STATUS = "P22B_CUDA_API_NVTX_ATTRIBUTION_NOT_CRITICAL_PATH"
REQUIRED = ("P22_Q_PROPOSALS", "P22_WIDE_VERIFIER",
            "P22_TARGET_SUPPORT_ACCEPT_BONUS")


def classify_api(name: str) -> str:
    n = name.lower()
    if "graphlaunch" in n:
        return "graph_launch"
    if "launchkernel" in n or "launchcooperative" in n:
        return "kernel_launch"
    if "memcpy" in n:
        return "memcpy"
    if "memset" in n:
        return "memset"
    if "synchronize" in n or "waitevent" in n:
        return "sync_or_wait"
    if "event" in n:
        return "event"
    return "other"


def _nvtx_index(db: sqlite3.Connection) -> dict:
    col = _columns(db, NVTX_TABLE)
    if not {"start", "end", "eventType", "globalTid"}.issubset(col):
        raise RuntimeError("P22B_NVTX_SCHEMA_CHANGED")
    if "text" not in col and "textId" not in col:
        raise RuntimeError("P22B_NVTX_LABEL_UNAVAILABLE")
    if "textId" in col:
        label = ("coalesce(n.text,si.value)"
                 if "text" in col else "si.value")
        join = "LEFT JOIN StringIds si ON n.textId=si.id"
    else:
        label, join = "n.text", ""
    sql = (
        f"SELECT n.start,n.end,n.globalTid,{label} FROM {NVTX_TABLE} n "
        f"{join} WHERE n.eventType=59 AND n.end IS NOT NULL "
        "ORDER BY n.globalTid,n.start"
    )
    ranges = defaultdict(list)
    for start, end, tid, name in db.execute(sql):
        if (not isinstance(name, str) or not name.startswith("P22_")
                or type(start) is not int or type(end) is not int
                or end < start):
            continue
        ranges[tid].append((start, end, name))
    if not ranges:
        raise RuntimeError("P22B_NO_REAL_P22_NVTX_RANGES")
    index = {}
    for tid, rows in ranges.items():
        rows.sort(key=lambda r: (r[0], -r[1]))
        maximum = -1
        prefix = []
        for _, end, _ in rows:
            maximum = max(end, maximum)
            prefix.append(maximum)
        index[tid] = {
            "rows": rows, "starts": [r[0] for r in rows],
            "max_end": prefix,
        }
    return index


def match_api_stage(index: dict, tid: int, start: int, end: int) -> str | None:
    """Select deepest enclosing CPU NVTX range; never use GPU timestamps."""
    record = index.get(tid)
    if record is None:
        return None
    j = bisect.bisect_right(record["starts"], start) - 1
    best = None
    best_span = float("inf")
    while j >= 0 and record["max_end"][j] >= end:
        a, b, name = record["rows"][j]
        if a <= start and b >= end and (b - a) < best_span:
            best = name
            best_span = b - a
        j -= 1
    return best


def _api_rows(db: sqlite3.Connection, table: str):
    col = _columns(db, table)
    if not {"start", "end", "globalTid", "correlationId"}.issubset(col):
        raise RuntimeError("P22B_API_SCHEMA_CHANGED_" + table)
    if "nameId" in col:
        expr = "si.value"
        join = "LEFT JOIN StringIds si ON a.nameId=si.id"
    elif "name" in col:
        expr, join = "a.name", ""
    else:
        raise RuntimeError("P22B_API_NAME_SCHEMA_CHANGED_" + table)
    sql = (f"SELECT a.start,a.end,a.globalTid,a.correlationId,{expr} "
           f"FROM {table} a {join}")
    return db.execute(sql)


def analyze_sqlite(db_path: Path) -> dict:
    # Immutable, read-only sqlite. No accidental export/overwrite.
    p = db_path.expanduser().resolve(strict=True)
    if not p.is_file() or p.stat().st_size <= 0:
        raise RuntimeError("P22B_EMPTY_NSYS_SQLITE")
    db = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if NVTX_TABLE not in tables or STRING_TABLE not in tables:
            raise RuntimeError("P22B_NVTX_OR_STRINGS_MISSING")
        if not all(t in tables for t in API_TABLES):
            raise RuntimeError("P22B_CUDA_RUNTIME_OR_DRIVER_TABLE_MISSING")
        if not {"id", "value"}.issubset(_columns(db, STRING_TABLE)):
            raise RuntimeError("P22B_STRINGIDS_SCHEMA_CHANGED")
        scopes = _nvtx_index(db)
        # Driver layer can report a second wrapper for a runtime launch.
        # Exclude matched driver (pid,cid), but do not otherwise suppress
        # bona fide driver-only calls. This is not a unique kernel join.
        runtime_correlations = set()
        runtime_rows = []
        for start, end, tid, cid, name in _api_rows(
            db, API_TABLES[0]
        ):
            runtime_rows.append((start, end, tid, cid, name, "runtime"))
            if type(cid) is int and type(tid) is int:
                runtime_correlations.add((tid >> 24, cid))

        by_stage = defaultdict(
            lambda: defaultdict(lambda: {"calls": 0, "cpu_ns": 0})
        )
        by_api = defaultdict(
            lambda: {"calls": 0, "cpu_ns": 0}
        )
        counts = defaultdict(int)
        counted_ns = 0
        def consume(start, end, tid, cid, name, source):
            nonlocal counted_ns
            if not all(type(v) is int for v in (start, end, tid)):
                raise RuntimeError("P22B_INVALID_CUDA_API_TIMESTAMPS")
            if end < start:
                raise RuntimeError("P22B_NEGATIVE_CUDA_API_DURATION")
            if not isinstance(name, str) or not name:
                raise RuntimeError("P22B_CUDA_API_NAME_MISSING")
            counts["all_api_calls"] += 1
            stage = match_api_stage(scopes, tid, start, end)
            if stage is None:
                counts["outside_p22_nvtx"] += 1
                return
            category = classify_api(name)
            row = by_stage[stage][category]
            row["calls"] += 1
            row["cpu_ns"] += end - start
            named = by_api[(stage, name, source)]
            named["calls"] += 1
            named["cpu_ns"] += end - start
            counts["assigned_p22_nvtx_calls"] += 1
            counted_ns += end - start

        for row in runtime_rows:
            consume(*row)
        for start, end, tid, cid, name in _api_rows(db, API_TABLES[1]):
            if type(cid) is int and type(tid) is int and (
                tid >> 24, cid
            ) in runtime_correlations:
                counts["driver_duplicate_runtime_skipped"] += 1
                continue
            consume(start, end, tid, cid, name, "driver")
        for stage in REQUIRED:
            if stage not in by_stage:
                raise RuntimeError("P22B_MISSING_CUDA_API_STAGE_" + stage)
        if counts["assigned_p22_nvtx_calls"] < 1:
            raise RuntimeError("P22B_NO_ATTRIBUTED_REAL_CUDA_APIS")
        return {
            "schema": SCHEMA, "status": STATUS,
            "semantics": (
                "CPU CUDA API duration summed within NVTX launch scopes, "
                "not GPU elapsed, critical path, or achievable savings"
            ),
            "counts": dict(counts),
            "total_assigned_cpu_api_ms": counted_ns / 1e6,
            "stages": {
                stage: {
                    group: {
                        "calls": row["calls"],
                        "summed_cpu_api_ms": row["cpu_ns"] / 1e6,
                    }
                    for group, row in sorted(groups.items())
                }
                for stage, groups in sorted(by_stage.items())
            },
            "top_api_by_cpu_time": [
                {
                    "stage": stage, "api_name": name, "source": source,
                    "calls": row["calls"],
                    "summed_cpu_api_ms": row["cpu_ns"] / 1e6,
                }
                for (stage, name, source), row in sorted(
                    by_api.items(), key=lambda item: item[1]["cpu_ns"],
                    reverse=True
                )[:40]
            ],
        }
    finally:
        db.close()


def validate_provenance(root: Path, report: dict) -> dict:
    root = root.expanduser().resolve(strict=True)
    cfg = json.loads((root / "run.json").read_text())
    client = json.loads((root / "p22_client.json").read_text())
    source = (root / "source.sha").read_text().strip()
    if (cfg.get("source_sha") != source or
            client.get("source_sha") != source or len(source) != 40 or
            cfg.get("nsys_capture_range") is not True or
            client.get("status") !=
                "P22_ONE_REAL_SERVING_B_REQUEST_PROFILE_CAPTURED_NOT_SPEEDUP"):
        raise RuntimeError("P22B_PROFILE_SOURCE_OR_CAPTURE_PROVENANCE_FAIL")
    frames = client.get("profiled_candidate_frames")
    rounds = client.get("profiled_k2_rounds")
    if type(frames) is not int or type(rounds) is not int or frames < 1 or rounds < 1:
        raise RuntimeError("P22B_INVALID_REAL_REQUEST_COUNTS")
    report.update({
        "source_sha": source, "real_frames": frames, "real_k2_rounds": rounds,
        "mean_k2_rounds_per_frame": rounds / frames,
    })
    for groups in report["stages"].values():
        for row in groups.values():
            row["calls_per_frame"] = row["calls"] / frames
            row["summed_cpu_api_ms_per_frame"] = row["summed_cpu_api_ms"] / frames
            row["calls_per_k2_round"] = row["calls"] / rounds
    return report


def markdown(report: dict) -> str:
    lines = [
        "# P22B — Real serving CUDA API ↔ NVTX attribution",
        "",
        "**Read-only; NOT a GPU, latency or causal speedup measurement.**",
        "",
        f"Source: \`{report['source_sha']}\`",
        f"Frames: {report['real_frames']}; K2 rounds: {report['real_k2_rounds']}",
        f"Assigned API calls: {report['counts']['assigned_p22_nvtx_calls']} "
        f"/ {report['counts']['all_api_calls']}",
        "",
        "| Stage | API group | Calls | Total CPU API ms | ms / frame |",
        "|---|---|---:|---:|---:|",
    ]
    for name, groups in report["stages"].items():
        for category, row in groups.items():
            lines.append(
                f"| {name} | {category} | {row['calls']} "
                f"| {row['summed_cpu_api_ms']:.3f} "
                f"| {row['summed_cpu_api_ms_per_frame']:.4f} |"
            )
    lines += [
        "",
        "CUDA API time may overlap with other CPU work; these categories "
        "cannot be summed into an E2E request speedup.",
        "Kernel execution and GPU wait are independently accounted for "
        "by P22's GPU kernel work trace.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--sqlite", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    src = args.sqlite.expanduser().resolve(strict=True)
    if src.parent != root:
        raise RuntimeError("P22B_NSYS_DB_MUST_BE_IN_SOURCE_ROOT")
    out = args.out.expanduser().resolve()
    if out.exists() or out.parent != root:
        raise FileExistsError("P22B_ABSENT_OUTPUT_WITHIN_SOURCE_ROOT_REQUIRED")
    report = validate_provenance(root, analyze_sqlite(src))
    out.mkdir(parents=False, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "summary.md").write_text(markdown(report))
    print("P22B_STATUS=" + report["status"])
    print("P22B_SUMMARY=" + str(out / "summary.md"))


if __name__ == "__main__":
    main()

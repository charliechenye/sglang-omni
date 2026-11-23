# SPDX-License-Identifier: Apache-2.0
"""P22 optional Nsight Systems SQLite GPU kernel-work attribution.

Maps GPU kernel -> CUDA runtime/driver API by correlationId, then maps
that API's *CPU timestamp/thread* to the innermost P22 NVTX push/pop
range (eventType 59). GPU timestamp containment in CPU NVTX ranges is
WRONG for asynchronous launches and is never used here.

Kernel duration sum = GPU KERNEL WORK, not GPU elapsed envelope, SM busy,
critical path or E2E speedup. Missing API correlation is explicitly counted.
Fail closed on schema drift, empty trace, insufficient stage coverage.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
API_TABLES = (
    "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER",
)
NVTX_TABLE = "NVTX_EVENTS"
STRING_TABLE = "StringIds"
SUCCESS = "P22_NSYS_SOURCE_SCOPED_KERNEL_WORK_ATTRIBUTION_NOT_GPU_ELAPSED"


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def load_nsight(db_path: Path) -> tuple[dict, dict]:
    uri = "file:" + str(db_path.resolve(strict=True)) + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    try:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {KERNEL_TABLE, NVTX_TABLE, STRING_TABLE}.issubset(tables):
            raise RuntimeError("P22_NSYS_SQLITE_REQUIRED_TABLES_MISSING")
        if not {"start", "end", "correlationId", "shortName"}.issubset(
            _columns(db, KERNEL_TABLE)
        ):
            raise RuntimeError("P22_NSYS_KERNEL_SCHEMA_CHANGED")
        if not {"start", "end", "eventType", "globalTid"}.issubset(
            _columns(db, NVTX_TABLE)
        ):
            raise RuntimeError("P22_NSYS_NVTX_SCHEMA_CHANGED")
        if not {"id", "value"}.issubset(_columns(db, STRING_TABLE)):
            raise RuntimeError("P22_NSYS_STRINGIDS_SCHEMA_CHANGED")
        ncol = _columns(db, NVTX_TABLE)
        text_expr = (
            "coalesce(n.text, s.value)" if {"text", "textId"}.issubset(ncol)
            else ("s.value" if "textId" in ncol else "n.text")
        )
        text_join = (
            "LEFT JOIN StringIds s ON n.textId=s.id"
            if "textId" in ncol else ""
        )
        events = defaultdict(list)
        sql = (
            f"SELECT n.start,n.end,n.globalTid,{text_expr} "
            f"FROM {NVTX_TABLE} n {text_join} "
            "WHERE n.eventType=59 AND n.end IS NOT NULL "
            "ORDER BY n.globalTid,n.start"
        )
        for start, end, tid, label in db.execute(sql):
            if (not isinstance(label, str)
                    or not label.startswith("P22_")
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or end < start):
                continue
            events[tid].append((start, end, label))
        if not events:
            raise RuntimeError("P22_NSYS_NO_P22_NVTX_WORKER_RANGES")
        index = {}
        for tid, rows in events.items():
            rows.sort()
            prefix = []
            maximum = -1
            for _, end, _ in rows:
                maximum = max(maximum, end)
                prefix.append(maximum)
            index[tid] = {
                "starts": [r[0] for r in rows],
                "ranges": rows,
                "prefix_max_end": prefix,
            }

        # CUDA graph nodes may have a correlationId shared by many kernels.
        # One API launch -> multiple GPU kernels is valid. A collision of
        # distinct API launches with the SAME correlationId is ambiguous.
        api = {}
        ambiguous = set()
        for table in API_TABLES:
            if table not in tables:
                continue
            if not {"start", "end", "globalTid", "correlationId"}.issubset(
                _columns(db, table)
            ):
                continue
            for cid, begin, end, tid in db.execute(
                f"SELECT correlationId,start,end,globalTid FROM {table} "
                "WHERE correlationId IS NOT NULL"
            ):
                item = (begin, end, tid)
                if cid in api and api[cid] != item:
                    ambiguous.add(cid)
                else:
                    api[cid] = item

        for cid in ambiguous:
            api.pop(cid, None)

        def find_stage(item):
            begin, end, tid = item
            row = index.get(tid)
            if row is None:
                return None
            j = bisect.bisect_right(row["starts"], begin) - 1
            while j >= 0 and row["prefix_max_end"][j] >= end:
                start, finish, name = row["ranges"][j]
                if start <= begin and finish >= end:
                    return name
                j -= 1
            return None

        stage = {}
        for cid, item in api.items():
            label = find_stage(item)
            if label is not None:
                stage[cid] = label

        by_stage = defaultdict(
            lambda: {"kernels": 0, "sum_gpu_kernel_work_ns": 0}
        )
        kernel_names = Counter()
        counts = Counter()
        query = (
            f"SELECT k.start,k.end,k.correlationId,s.value "
            f"FROM {KERNEL_TABLE} k LEFT JOIN StringIds s "
            "ON k.shortName=s.id"
        )
        for start, end, cid, name in db.execute(query):
            if end < start:
                raise RuntimeError("P22_NSYS_INVALID_KERNEL_TIMESTAMP")
            counts["all_cuda_kernels"] += 1
            if cid in ambiguous:
                counts["ambiguous_api_correlation"] += 1
                continue
            if cid not in api:
                counts["missing_cuda_api_correlation"] += 1
                continue
            label = stage.get(cid)
            if label is None:
                counts["no_p22_nvtx_launch_range"] += 1
                continue
            counts["assigned_p22_nvtx_kernels"] += 1
            entry = by_stage[label]
            entry["kernels"] += 1
            entry["sum_gpu_kernel_work_ns"] += end - start
            kernel_names[(label, name or "UNNAMED")] += end - start
        if not counts["all_cuda_kernels"]:
            raise RuntimeError("P22_NSYS_NO_CUDA_KERNELS_CAPTURED")
        if not any(
            by_stage.get(label, {}).get("kernels", 0) > 0
            for label in ("P22_WIDE_VERIFIER", "P22_Q_PROPOSALS",
                          "P22_TARGET_SUPPORT_ACCEPT_BONUS")
        ):
            raise RuntimeError(
                "P22_NSYS_NO_REAL_WIDE_K2_KERNEL_STAGE_CORRELATION"
            )
        return {
            "schema": "p22-nsys-cuda-kernel-work-attribution-v1",
            "status": SUCCESS,
            "counters": dict(counts),
            "stage_sums": {
                name: {
                    "kernels": row["kernels"],
                    "summed_kernel_work_ms": (
                        row["sum_gpu_kernel_work_ns"] / 1_000_000
                    ),
                }
                for name, row in sorted(by_stage.items())
            },
        }, {
            "top_gpu_kernel_work": [
                {"stage": stage, "kernel": name,
                 "sum_gpu_kernel_work_ms": duration / 1_000_000}
                for (stage, name), duration in kernel_names.most_common(25)
            ],
        }
    finally:
        db.close()


def analyze(root: Path, sqlite_path: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    cfg = json.loads((root / "run.json").read_text())
    p22 = json.loads((root / "p22_client.json").read_text())
    sha = (root / "source.sha").read_text().strip()
    if (cfg.get("source_sha") != sha
            or p22.get("source_sha") != sha
            or cfg.get("nsys_capture_range") is not True
            or p22.get("status") !=
               "P22_ONE_REAL_SERVING_B_REQUEST_PROFILE_CAPTURED_NOT_SPEEDUP"):
        raise RuntimeError("P22_NSYS_UNMATCHED_PROFILE_SOURCE_OR_CAPTURE")
    if sqlite_path.parent.resolve() != root:
        raise RuntimeError("P22_NSYS_SQLITE_MUST_BE_EXPORTED_INSIDE_ROOT")
    report, top = load_nsight(sqlite_path)
    report.update(top)
    report.update({
        "source_sha": sha,
        "profiled_candidate_frames": p22["profiled_candidate_frames"],
        "profiled_k2_rounds": p22["profiled_k2_rounds"],
        "note": (
            "Durations sum kernel execution work per stage and may overlap. "
            "Not kernel busy time, GPU critical-path latency, nor request E2E."
        ),
    })
    for row in report["stage_sums"].values():
        row["summed_gpu_work_ms_per_frame"] = (
            row["summed_kernel_work_ms"]
            / report["profiled_candidate_frames"]
        )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--sqlite", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    out = a.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError("P22_NSYS_REPORT_REFUSES_OVERWRITE")
    report = analyze(a.root, a.sqlite)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print("P22_NSYS_STATUS=" + report["status"])
    for stage, row in sorted(report["stage_sums"].items()):
        print(
            f"P22_GPU_KERNEL_WORK {stage} "
            f"kernels={row['kernels']} "
            f"sum_ms={row['summed_kernel_work_ms']:.3f} "
            f"per_frame_ms={row['summed_gpu_work_ms_per_frame']:.3f}"
        )
    print("P22_NSYS_JSON=" + str(out))


if __name__ == "__main__":
    main()

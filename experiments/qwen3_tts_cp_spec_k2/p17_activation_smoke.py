# SPDX-License-Identifier: Apache-2.0
"""P17 one-base/one-spec activation-only smoke on a FRESH diag worker.

This does NOT run P17 K2/K3/K4 Graph capture and cannot produce a
performance/parity PASS. It proves actual worker entry and ARM visibility,
independently of ordinary server log formatting/buffering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from experiments.qwen3_tts_cp_spec_k2.quality_smoke import (
    MODEL, TEXT, _http_call, parse_k2_log_entries, wav_info,
)

STATUS_ENTERED = "P17_ACTIVATION_K2_ENTERED_NOT_MULTIWIDTH_PARITY"
STATUS_NOT_SEEN = "P17_ACTIVATION_DISPATCH_NOT_OBSERVED"
STATUS_ARM_MISSING = "P17_ACTIVATION_ARM_NOT_VISIBLE_TO_WORKER"
STATUS_NATIVE = "P17_ACTIVATION_NATIVE_FALLBACK_WITH_ARM_PRESENT"
STATUS_PARTIAL = "P17_ACTIVATION_CODE_PREDICTOR_NOT_ENTERED"
STATUS_FAILED = "P17_ACTIVATION_SMOKE_INCOMPLETE"
STATUS_STARTUP_MISSING = "P17_ACTIVATION_WORKER_STARTUP_NOT_INSTRUMENTED"
STATUS_EVENT_WINDOW = "P17_ACTIVATION_EVENTS_OUTSIDE_REQUEST_WINDOW"
STARTUP_EVENTS = frozenset((
    "engine_builder_make_model_runner",
    "model_runner_initialized",
))


def _collect(
    root: Path, *, start_ns: int | None = None, end_ns: int | None = None,
) -> list[dict]:
    folder = root / "activation"
    output = []
    for path in sorted(folder.glob("dispatch_pid*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise RuntimeError("P17 malformed activation trace")
            timestamp = event.get("monotonic_ns")
            if not isinstance(timestamp, int):
                raise RuntimeError("P17 activation event missing clock")
            if ((start_ns is None or start_ns <= timestamp)
                    and (end_ns is None or timestamp <= end_ns)):
                output.append(event)
    return sorted(output, key=lambda e: e["monotonic_ns"])


def _classify(events: list[dict]) -> str:
    if not events:
        return STATUS_NOT_SEEN
    labels = {e["event"] for e in events}
    if "speculative_k2_enter" in labels:
        return STATUS_ENTERED
    entered = [
        e for e in events if e["event"] == "code_predictor_forward_enter"
    ]
    if not entered:
        return STATUS_PARTIAL
    if not any(e.get("spec_arm_exists") is True for e in entered):
        return STATUS_ARM_MISSING
    return STATUS_NATIVE


def run(root: Path, out: Path, *, base_url: str) -> dict:
    root, out = root.expanduser().resolve(), out.expanduser().resolve()
    if out.exists():
        raise RuntimeError(f"P17 diagnostic output already exists: {out}")
    sha_path = root / "source.sha"
    log_path = root / "server.log"
    if not sha_path.is_file() or not log_path.is_file():
        raise FileNotFoundError("P17 source-locked worker evidence missing")
    pin = sha_path.read_text().strip()
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if len(pin) != 40 or sha != pin:
        raise RuntimeError("P17 client code SHA differs from running worker")
    diag = root / "activation"
    if not diag.is_dir():
        raise RuntimeError(
            "P17_ACTIVATION_DIR_NOT_CREATED: start Terminal A with "
            "P17_ACTIVATION_DIAG=1 before running this smoke"
        )
    launcher_pin = (diag / "launcher-source.sha")
    if (not launcher_pin.is_file()
            or launcher_pin.read_text().strip() != pin):
        raise RuntimeError("P17_ACTIVATION_LAUNCHER_SHA_ATTESTATION_MISSING")
    if (root / "ARM_BASE").exists() or (root / "ARM_SPEC").exists():
        raise RuntimeError("P17 stale ARM file before smoke")
    if (root / "smoke").exists():
        raise RuntimeError("P17 activation-only cannot share ABBA evidence")
    with urlopen(base_url.rstrip("/") + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("P17 activation worker unhealthy")
    out.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": MODEL, "input": TEXT, "voice": "Vivian",
        "response_format": "wav", "seed": 20261121,
        "max_new_tokens": 512,
        "stage_params": {
            "tts_engine": {
                "subtalker_dosample": True,
                "subtalker_temperature": 0.9,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
            },
        },
    }
    startup = _collect(root)
    startup_counts = Counter(row["event"] for row in startup)
    report = {
        "status": STATUS_FAILED,
        "schema": "qwen3tts-p17-activation-smoke-v1",
        "worker_source_sha": pin,
        "experiment_root": str(root),
        "kind": "activation_entry_only_NOT_PARITY_OR_PERFORMANCE",
        "raw_startup_events": startup,
        "raw_startup_counts": dict(startup_counts),
        "visits": [],
    }
    if not STARTUP_EVENTS.issubset(startup_counts):
        report["status"] = STATUS_STARTUP_MISSING
        report["missing_startup_events"] = sorted(
            STARTUP_EVENTS.difference(startup_counts)
        )
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "summary.md").write_text(
            "# P17 activation worker startup preflight\n\n"
            f"**{STATUS_STARTUP_MISSING}**\n\n"
            f"Source SHA: {pin}\n"
            f"Raw startup counts: {dict(startup_counts)}\n"
            f"Missing: {report['missing_startup_events']}\n"
            "No HTTP requests sent; investigate worker env/module import.\n"
        )
        print(f"P17_ACTIVATION_STATUS={STATUS_STARTUP_MISSING}")
        print(f"P17_ACTIVATION_REPORT={out/'summary.md'}")
        return report
    try:
        for label, arm_name in (
            ("00_base", "ARM_BASE"), ("01_spec", "ARM_SPEC")
        ):
            arm = root / arm_name
            before_raw = log_path.read_text(errors="replace")
            previous_frames = len(parse_k2_log_entries(before_raw))
            arm.touch(exist_ok=False)
            start_ns = time.monotonic_ns()
            print(f"P17_ACTIVATION_VISIT_BEGIN={label}", flush=True)
            try:
                audio, finish, wall = _http_call(base_url, payload)
                end_ns = time.monotonic_ns()
            finally:
                arm.unlink(missing_ok=True)
            events = _collect(root, start_ns=start_ns, end_ns=end_ns)
            full_snapshot = _collect(root)
            all_counts = Counter(x["event"] for x in full_snapshot)
            for event in events:
                if event.get("source_sha") != pin:
                    raise RuntimeError(
                        "P17 activation events mismatch server source SHA"
                    )
                expected_arm = root / (
                    "ARM_SPEC" if label == "01_spec" else "ARM_BASE"
                )
                field = (
                    "spec_arm_path" if label == "01_spec" else "base_arm_path"
                )
                if event.get(field) != str(expected_arm):
                    raise RuntimeError(
                        f"P17 worker ARM path mismatch: {field}={event.get(field)}"
                    )
            new_frames = len(parse_k2_log_entries(
                log_path.read_text(errors="replace")
            )) - previous_frames
            states = Counter(event["event"] for event in events)
            record = {
                "label": label,
                "finish_reason": finish,
                "wav_sha256": hashlib.sha256(audio).hexdigest(),
                "wall_s_NOT_PERFORMANCE": wall,
                "native_k2_log_frames": new_frames,
                "worker_event_counts": dict(states),
                "all_raw_event_counts_after_visit": dict(all_counts),
                "all_raw_event_count_after_visit": len(full_snapshot),
                "worker_events": events,
                "diagnostic_classification": (
                    (_classify(events) if events else STATUS_EVENT_WINDOW)
                    if label == "01_spec" else "BASELINE"
                ),
                **wav_info(audio),
            }
            report["visits"].append(record)
            (out / f"{label}.wav").write_bytes(audio)
            (out / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                f"P17_ACTIVATION_VISIT_DONE={label} "
                f"stop={finish} raw_k2={new_frames} "
                f"events={dict(states)}",
                flush=True,
            )
            if finish != "stop":
                raise RuntimeError("P17 diagnostic request did not naturally stop")
        base, spec = report["visits"]
        # The original strict K2 gate stays unchanged. A diagnostic K2
        # entry is NOT considered a GPU correctness/speedup pass.
        if any(
            e["event"] == "speculative_k2_enter"
            for e in base["worker_events"]
        ):
            raise RuntimeError("P17_BASELINE_ARM_LEAKED_INTO_K2")
        report["status"] = spec["diagnostic_classification"]
    except BaseException as exc:
        report["status"] = STATUS_FAILED
        report["first_error"] = f"{type(exc).__name__}: {exc}"
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.md").write_text(
        "# P17 activation-only worker diagnosis\n\n"
        f"**{report['status']}**\n\n"
        f"Source SHA: {pin}\n"
        f"Startup event counts: {dict(startup_counts)}\n"
        f"Baseline event counts: {base['worker_event_counts']}\n"
        f"Spec event counts: {spec['worker_event_counts']}\n"
        f"All raw events after spec: {spec['all_raw_event_counts_after_visit']}\n"
        f"Spec native K2 log frames: {spec['native_k2_log_frames']}\n\n"
        "No GPU multiwidth parity or performance inference.\n"
    )
    print(f"P17_ACTIVATION_STATUS={report['status']}")
    print(f"P17_ACTIVATION_REPORT={out/'summary.md'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    run(args.root, args.out, base_url=args.base_url)


if __name__ == "__main__":
    main()

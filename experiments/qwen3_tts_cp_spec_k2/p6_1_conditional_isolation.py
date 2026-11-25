# SPDX-License-Identifier: Apache-2.0
"""P6.1 — Isolate PyTorch 2.14 CUDA conditional-node capture failures.

The previous P6 used a plain torch.cuda.graph context. Unlike Inductor's
CUDAGraphTrees, it did NOT install the control-flow dispatch mode that
lowers torch.while_loop to a CUDA WHILE conditional node. It also hid the
first exception behind cudaErrorStreamCaptureInvalidated.

This standalone H200 diagnostic runs three gates in SEPARATE FRESH PROCESSES:
  plain_graph: prove ordinary capture/replay works.
  while_minimal: capture a real CUDA WHILE with a dynamic scalar trip count.
  depth_k2: capture a GPU decision-mask loop with accept + bonus skip behavior.

Uses the same explicit internal dispatch mode as PyTorch 2.14 Inductor:
CUDAGraphCaptureControlFlowOpDispatchMode, together with the corresponding
higher-order while_loop_op. No reinstall/root, no TTS weights or server.

This mode is private/prototype API. Passing the gate proves control-flow
compatibility, not model capture, full K2 correctness or speedup.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

STAGES = ("plain_graph", "while_minimal", "depth_k2")
PASS = "P61_CUDA_DYNAMIC_WHILE_CAPTURE_AND_REPLAY_PASS_NOT_MODEL_SPEEDUP"
FAIL = "P61_CAPTURE_OR_RUNTIME_FAIL_NOT_HARDWARE_NEGATIVE_PROOF"


def cpu_depth_reference(decisions: list[int]) -> dict[str, int]:
    if len(decisions) != 16 or any(v not in (0, 1) for v in decisions):
        raise ValueError("expected 16 binary decisions")
    d = 1
    rounds = accepted = bonus = 0
    while d < 15:
        proposed = d + 1
        chosen = decisions[proposed]
        b = int(bool(chosen) and proposed < 15)
        d = proposed + b
        rounds += 1
        accepted += chosen
        bonus += b
    assert rounds + bonus == 14
    return {"depth": d, "rounds": rounds, "accepted": accepted, "bonus": bonus}


def masks() -> dict[str, list[int]]:
    return {
        "all_reject": [0] * 16,
        "all_accept": [0] + [1] * 15,
        "alternating": [0] + [i % 2 for i in range(1, 16)],
        "late_accept": [0] + [int(i >= 6) for i in range(1, 16)],
        "late_reject": [0] + [int(i < 6) for i in range(1, 16)],
    }


def device_depth_hop(while_loop_op, state, depth0, zero):
    """No GPU->CPU sync, all carried inputs and outputs stay fixed shaped."""
    def condition(depth, rounds, accepted, bonus, decision_table):
        return depth < 15

    def body(depth, rounds, accepted, bonus, decision_table):
        proposed = depth + 1
        choice = decision_table.gather(0, proposed.reshape(1)).reshape(())
        extra = (choice & (proposed < 15)).to(torch.int64)
        return (
            proposed + extra,
            rounds + 1,
            accepted + choice.to(torch.int64),
            bonus + extra,
        )

    import torch

    return while_loop_op(
        condition,
        body,
        (depth0, zero, zero.clone(), zero.clone()),
        (state,),
    )


def stage_main(stage: str, stage_dir: Path) -> int:
    import torch

    report: dict[str, Any] = {
        "stage": stage,
        "status": "FAILED",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "first_error_traceback": None,
        "cases": [],
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        device = torch.device("cuda:0")
        report["gpu"] = torch.cuda.get_device_name(device)
        # A normal graph gate isolates CUDA env problems from conditional
        # control-flow problems.
        torch.cuda.synchronize()
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        if stage == "plain_graph":
            x = torch.zeros((), device=device, dtype=torch.int64)
            with torch.cuda.stream(side), torch.inference_mode():
                for _ in range(3):
                    y = x + 5
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, stream=side, capture_error_mode="thread_local"
            ), torch.inference_mode():
                output = x + 5
            for value in (0, 2, 11):
                x.fill_(value)
                graph.replay()
                torch.cuda.synchronize()
                actual = int(output.item())
                report["cases"].append({
                    "input": value, "actual": actual, "expected": value + 5,
                    "pass": actual == value + 5,
                })
            if not all(c["pass"] for c in report["cases"]):
                raise AssertionError("plain CUDA Graph did not update its input")
        else:
            # Mirrors the concrete upstream PyTorch 2.14 CUDAGraphTrees path,
            # not the invalid plain-graph torch.while_loop call in P6.0.
            from torch._higher_order_ops.cudagraph_conditional_nodes import (
                CUDAGraphCaptureControlFlowOpDispatchMode,
            )
            from torch._higher_order_ops.while_loop import while_loop_op

            initial = torch.zeros((), device=device, dtype=torch.int64)
            if stage == "while_minimal":
                limit = torch.full((), 3, device=device, dtype=torch.int64)

                def condition(iteration, bound):
                    return iteration < bound

                def body(iteration, bound):
                    return (iteration + 1,)

                def operation():
                    return while_loop_op(
                        condition, body, (initial,), (limit,)
                    )

                values = (0, 3, 7)
            elif stage == "depth_k2":
                state = torch.zeros(16, dtype=torch.bool, device=device)
                depth0 = torch.ones((), device=device, dtype=torch.int64)
                zero = torch.zeros((), device=device, dtype=torch.int64)

                def operation():
                    return device_depth_hop(
                        while_loop_op, state, depth0, zero
                    )

                values = tuple(masks())
            else:
                raise ValueError(f"unexpected stage {stage!r}")

            # Eager warmup is allowed to synchronize. The captured while
            # control path must use the explicit PyTorch conditional mode.
            with torch.cuda.stream(side), torch.inference_mode():
                for _ in range(2):
                    operation()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.graph(
                    graph, stream=side, capture_error_mode="thread_local"
                ),
                CUDAGraphCaptureControlFlowOpDispatchMode(),
                torch.inference_mode(),
            ):
                output = operation()
            torch.cuda.synchronize()

            for value in values:
                if stage == "while_minimal":
                    limit.fill_(value)
                    expected = {"iteration": value}
                else:
                    state.copy_(torch.tensor(
                        masks()[value], dtype=torch.bool, device=device
                    ))
                    expected = cpu_depth_reference(masks()[value])
                graph.replay()
                torch.cuda.synchronize()
                if stage == "while_minimal":
                    actual = {"iteration": int(output[0].item())}
                else:
                    actual = dict(zip(
                        ("depth", "rounds", "accepted", "bonus"),
                        [int(x.item()) for x in output],
                    ))
                passed = actual == expected
                report["cases"].append({
                    "input": value,
                    "actual": actual,
                    "expected": expected,
                    "pass": passed,
                })
                if not passed:
                    raise AssertionError(
                        f"graph replay was not dynamic or correct: {report['cases'][-1]}"
                    )
            # The same dynamic graph should have different execution work for
            # different on-device loop trip counts. Event timings remain
            # launch envelopes; this is only a control-flow overhead screen.
            report["dynamic_replay_costs"] = []
            for value in values:
                if stage == "while_minimal":
                    limit.fill_(value)
                    iteration_count = value
                else:
                    state.copy_(torch.tensor(
                        masks()[value], dtype=torch.bool, device=device
                    ))
                    iteration_count = cpu_depth_reference(masks()[value])["rounds"]
                for _ in range(4):
                    graph.replay()
                torch.cuda.synchronize()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                import time

                wall_start = time.perf_counter()
                begin.record()
                for _ in range(32):
                    graph.replay()
                end.record()
                end.synchronize()
                report["dynamic_replay_costs"].append({
                    "mask": str(value),
                    "device_rounds": iteration_count,
                    "cuda_event_ms_per_graph": begin.elapsed_time(end) / 32,
                    "host_wall_ms_per_graph": (
                        (time.perf_counter() - wall_start) * 1000.0 / 32
                    ),
                })
        report["status"] = "PASS"
    except BaseException as exc:
        # Do NOT attempt another capture in this process: a CUDA error may
        # poison the capture stream. Preserve Python's chained exception.
        report["status"] = "FAILED"
        report["first_error_type"] = type(exc).__name__
        report["first_error_message"] = str(exc)
        report["first_error_traceback"] = traceback.format_exc()
        (stage_dir / "traceback.txt").write_text(
            report["first_error_traceback"]
        )
    (stage_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"P61_STAGE={stage} status={report['status']}", flush=True)
    return 0 if report["status"] == "PASS" else 2


def coordinator(out: Path, timeout: int) -> int:
    if out.exists():
        raise FileExistsError(f"output exists, refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "status": FAIL,
        "stages": [],
        "limitations": [
            "Uses an internal, prototype PyTorch 2.14 conditional-node dispatch mode",
            "No model weights, dynamic Predictor KV pointers or sampling RNG are tested",
            "Three fresh subprocesses avoid a poisoned capture and known conditional-node allocator reuse issues",
            "A PASS does not establish GPU-resident K2 integration or speedup",
        ],
    }
    for stage in STAGES:
        folder = out / stage
        cmd = [
            sys.executable, "-m",
            "experiments.qwen3_tts_cp_spec_k2.p6_1_conditional_isolation",
            "--stage", stage, "--stage-out", str(folder),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=False,
            )
            (out / (stage + ".stdout.txt")).write_text(proc.stdout)
            (out / (stage + ".stderr.txt")).write_text(proc.stderr)
            stage_report_file = folder / "report.json"
            stage_report = (
                json.loads(stage_report_file.read_text())
                if stage_report_file.is_file() else {}
            )
            row = {
                "stage": stage, "exit_code": proc.returncode,
                "status": stage_report.get("status", "NO_REPORT"),
                "first_error_type": stage_report.get("first_error_type"),
                "first_error_message": stage_report.get("first_error_message"),
                "cases": stage_report.get("cases", []),
                "dynamic_replay_costs": stage_report.get(
                    "dynamic_replay_costs", []
                ),
            }
        except subprocess.TimeoutExpired:
            row = {
                "stage": stage, "status": "TIMEOUT",
                "exit_code": None, "first_error_type": "TimeoutExpired",
                "first_error_message": f"stage timed out after {timeout}s",
                "cases": [],
            }
        result["stages"].append(row)
        (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"P61_STAGE_DONE={stage} status={row['status']}"
            f" exit_code={row['exit_code']}", flush=True
        )
        if row["status"] != "PASS" or row["exit_code"] != 0:
            break
    if len(result["stages"]) == len(STAGES) and all(
        x["status"] == "PASS" and x["exit_code"] == 0
        for x in result["stages"]
    ):
        result["status"] = PASS
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# P6.1 — isolated H200 CUDA Graph conditional WHILE compatibility",
        "",
        "**" + result["status"] + "**", "",
        "| Stage | Status | Exit | First error |",
        "|---|---|---:|---|",
    ]
    for stage in result["stages"]:
        err = stage.get("first_error_message") or "—"
        err = str(err).splitlines()[0].replace("|", "/")[:180]
        lines.append(
            f"| {stage['stage']} | {stage['status']} | "
            f"{stage['exit_code']} | {err} |"
        )
    for row in result["stages"]:
        if row.get("dynamic_replay_costs"):
            lines += [
                "", "## " + row["stage"] + " dynamic replay envelopes",
                "",
                "| GPU loop rounds | CUDA event ms/graph | Host wall ms/graph |",
                "|---:|---:|---:|",
            ]
            for cost in row["dynamic_replay_costs"]:
                lines.append(
                    f"| {cost['device_rounds']} | "
                    f"{cost['cuda_event_ms_per_graph']:.5f} | "
                    f"{cost['host_wall_ms_per_graph']:.5f} |"
                )
    lines += [
        "", "Each stage lives in a separate fresh CUDA process.",
        "Read the stage traceback.txt for the full first exception and chain.",
        "CUDA-event time measures launch-to-completion envelope, NOT kernel busy.",
        "", "## Non-claims", "",
    ] + ["- " + item for item in result["limitations"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"P61_RESULT={result['status']} REPORT={out / 'summary.md'}", flush=True)
    return 0 if result["status"] == PASS else 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--stage-out", type=Path)
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()
    if args.stage:
        if args.stage_out is None:
            p.error("--stage-out required when --stage is specified")
        code = stage_main(args.stage, args.stage_out.expanduser().resolve())
    else:
        if args.out is None:
            p.error("--out required")
        if args.timeout < 30 or args.timeout > 900:
            p.error("--timeout must be 30..900")
        code = coordinator(args.out.expanduser().resolve(), args.timeout)
    if code:
        p.exit(code, "P6.1 not PASS; inspect report.json and per-stage traceback\n")


if __name__ == "__main__":
    main()

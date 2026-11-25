# SPDX-License-Identifier: Apache-2.0
"""P6.0: GPU-resident dynamic residual-depth loop CUDA Graph compatibility gate.

PyTorch 2.14 introduced CUDA Graph capture for torch.while_loop via CUDA
conditional WHILE nodes. Before porting the real 5-layer Qwen3-TTS
CodePredictor plus fused sample/accept to device control, verify that:
1. the installed *actual* Torch/CUDA/H200 stack supports this feature;
2. the SAME captured graph replays different GPU decision masks correctly;
3. a dynamic accepted depth advances by 2 (proposal+bonus), a rejection by 1;
4. all branching happens on-device without CPU .item() per iteration;
5. output tensors and graph/capture memory stay live across replays.

This does NOT run any model weights or prove SpecDec speedup, quality or
probabilistic sampling correctness. No root or additional packages needed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

STATUS_PASS = "P6_DYNAMIC_DEPTH_CUDA_GRAPH_WHILE_CAPTURE_AND_REPLAY_PASS_NOT_MODEL_SPEEDUP"
STATUS_FAIL = "P6_DYNAMIC_DEPTH_CUDA_GRAPH_UNSUPPORTED_OR_INCORRECT"
STATUS_PRECHECK = "P6_DYNAMIC_DEPTH_GRAPH_PRECHECK_FAIL"


def reference(decisions: list[int]) -> dict[str, int]:
    """CPU oracle: positions are 1-based residual codebook depths (1..15).

    Decision array has 16 entries. Index d holds whether a speculative
    proposal at depth d is accepted; code1 at index1 is always committed.
    """
    if len(decisions) != 16 or any(v not in (0, 1) for v in decisions):
        raise ValueError("expected 16 Boolean integers for codec decisions")
    current = 1
    rounds = accepted = bonus = 0
    while current < 15:
        next_depth = current + 1
        accept = decisions[next_depth]
        bonus_here = int(bool(accept) and next_depth < 15)
        current = next_depth + bonus_here
        accepted += int(accept)
        bonus += bonus_here
        rounds += 1
        if rounds > 14:
            raise RuntimeError("invalid reference nonterminating path")
    if rounds + bonus != 14 or accepted > rounds:
        raise RuntimeError("invalid reference residual code accounting")
    return {"depth": current, "rounds": rounds, "accepted": accepted, "bonus": bonus}


def decision_masks() -> dict[str, list[int]]:
    masks = {
        "reject_all": [0] * 16,
        "accept_all": [0] + [1] * 15,
        "alternating": [0] + [(i % 2) for i in range(1, 16)],
        "early_fail_late_accept": [0] + [int(i >= 6) for i in range(1, 16)],
        "early_accept_late_fail": [0] + [int(i <= 6) for i in range(1, 16)],
    }
    return masks


def conditional_loop(
    decisions: torch.Tensor,
    initial_depth: torch.Tensor,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single torch.while_loop expressing the actual K2 depth transition.

    Control flow is driven entirely by a device array; no Python branching
    based on sampled token acceptance, no in-place carried-input mutation.
    """
    if not hasattr(torch, "while_loop"):
        raise RuntimeError("installed PyTorch has no torch.while_loop")
    if decisions.shape != (16,) or initial_depth.ndim != 0 or zero.ndim != 0:
        raise ValueError("invalid static tensor shapes")

    def cond(depth: torch.Tensor, rounds: torch.Tensor,
             accepted: torch.Tensor, bonus: torch.Tensor) -> torch.Tensor:
        return depth < 15

    def body(depth: torch.Tensor, rounds: torch.Tensor,
             accepted: torch.Tensor, bonus: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_depth = depth + 1
        # GPU-resident gather; no .item, no Python indexing of device scalar.
        decision = torch.gather(decisions, 0, next_depth.view(1)).squeeze(0)
        bonus_here = decision.logical_and(next_depth < 15).to(torch.int64)
        return (
            next_depth + bonus_here,
            rounds + 1,
            accepted + decision.to(torch.int64),
            bonus + bonus_here,
        )

    return torch.while_loop(
        cond, body, (initial_depth, zero, zero.clone(), zero.clone())
    )


def run(repeats: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": STATUS_PRECHECK,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_while_loop": hasattr(torch, "while_loop"),
        "cuda_available": torch.cuda.is_available(),
        "cases": [],
        "limitations": [
            "This GPU while loop models K2 codebook-depth scheduling only; no neural network",
            "Decisions are precomputed static device input, not actual GPU sampler decisions",
            "No Predictor kernels, KV ownership, rejection/residual math, bonus output or RNG",
            "Capture and replay proves dynamic loop feasibility, not a serving throughput gain",
            "P5 2.2266ms full Graph-ON baseline remains the actual performance target",
        ],
    }
    if not torch.cuda.is_available() or not hasattr(torch, "while_loop"):
        result["reason"] = "missing CUDA device or torch.while_loop"
        return result
    try:
        dev = torch.device("cuda:0")
        result["gpu"] = torch.cuda.get_device_name(dev)
        state = torch.zeros(16, device=dev, dtype=torch.bool)
        depth0 = torch.ones((), device=dev, dtype=torch.int64)
        zero = torch.zeros((), device=dev, dtype=torch.int64)
        masks = decision_masks()

        # Warmup on a side stream to prepare CUDA kernels without capture.
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side), torch.inference_mode():
            state.copy_(torch.tensor(masks["reject_all"], device=dev, dtype=torch.bool))
            for _ in range(2):
                conditional_loop(state, depth0, zero)
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side), torch.inference_mode():
            captured = conditional_loop(state, depth0, zero)
        torch.cuda.synchronize(dev)

        for name, mask in masks.items():
            state.copy_(torch.tensor(mask, device=dev, dtype=torch.bool))
            graph.replay()
            torch.cuda.synchronize(dev)
            got = [int(v.item()) for v in captured]
            expected = reference(mask)
            record = {
                "name": name,
                "expected": expected,
                "actual": dict(zip(
                    ("depth", "rounds", "accepted", "bonus"), got
                )),
            }
            record["pass"] = record["actual"] == expected
            result["cases"].append(record)
            if not record["pass"]:
                raise RuntimeError(
                    f"dynamic graph did not follow device mask for {name}: {record}"
                )

        # Same captured dynamic graph, three different actual GPU loop
        # iteration counts. Distinguish host launch cost from a kernel-heavy
        # device-scheduled while-node body.
        replay_costs = {}
        for name in ("reject_all", "accept_all", "alternating"):
            state.copy_(torch.tensor(masks[name], device=dev, dtype=torch.bool))
            for _ in range(8):
                graph.replay()
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            cuda_begin = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_begin.record()
            for _ in range(repeats):
                graph.replay()
            cuda_end.record()
            cuda_end.synchronize()
            event_ms = cuda_begin.elapsed_time(cuda_end) / repeats
            wall_ms = (time.perf_counter() - t0) * 1e3 / repeats
            if any(not math.isfinite(x) or x <= 0 for x in (event_ms, wall_ms)):
                raise RuntimeError("invalid P6 dynamic loop graph replay timing")
            replay_costs[name] = {
                "gpu_iterations": reference(masks[name])["rounds"],
                "cuda_event_ms_per_replay": event_ms,
                "host_wall_ms_per_replay": wall_ms,
            }
        result["replay_costs_by_dynamic_iterations"] = replay_costs
        result["status"] = STATUS_PASS
    except Exception as exc:
        result["status"] = STATUS_FAIL
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not 16 <= args.repeats <= 2048:
        p.error("repeats must be 16..2048")
    target = args.out.expanduser().resolve()
    if target.exists():
        p.error(f"output exists, do not overwrite: {target}")
    target.mkdir(parents=True)
    result = run(args.repeats)
    (target / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# P6.0 — CUDA Graph GPU-resident dynamic K2 depth loop",
        "",
        "**" + result["status"] + "**",
        "",
        "PyTorch: " + str(result["torch_version"]),
        "CUDA: " + str(result["torch_cuda_version"]),
        "GPU: " + str(result.get("gpu", "NONE")),
        "",
        "| Decision mask | Expected rounds | Actual rounds | Pass |",
        "|---|---:|---:|---|",
    ]
    for c in result["cases"]:
        lines.append(
            f"| {c['name']} | {c['expected']['rounds']} | "
            f"{c['actual']['rounds']} | {c['pass']} |"
        )
    if "replay_costs_by_dynamic_iterations" in result:
        lines += [
            "", "## Conditional graph replay cost at different device loop counts",
            "", "| Decision mask | GPU loop rounds | CUDA event ms/replay | Host wall ms/replay |",
            "|---|---:|---:|---:|",
        ]
        for label, values in result["replay_costs_by_dynamic_iterations"].items():
            lines.append(
                f"| {label} | {values['gpu_iterations']} | "
                f"{values['cuda_event_ms_per_replay']:.5f} | "
                f"{values['host_wall_ms_per_replay']:.5f} |"
            )
    if "error" in result:
        lines += ["", "Capture/replay failure: " + result["error"]]
    lines.extend(["", "## Non-claims", ""] + [
        "- " + x for x in result["limitations"]
    ])
    (target / "summary.md").write_text("\n".join(lines) + "\n")
    print(result["status"] + ": " + str(target / "summary.md"))
    if result["status"] != STATUS_PASS:
        p.exit(2, "P6.0 graph condition gate did not pass; inspect report.json\n")


if __name__ == "__main__":
    main()

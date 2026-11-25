# SPDX-License-Identifier: Apache-2.0
"""P5 opt-in CodePredictor CUDA Graph cost screen (C1 H200 experiment only).

Measure the *existing* full sampled Predictor graph replay on baseline frames
and separately capture/replay the real model's one-token vs packed-two-token
verification stack at frozen on-policy K2 states. No model-weight copying,
training, alternate checkpoint, or production perf claim.

All probes require a post-health marker and are OFF by default. Preserves the
model K/V cache, does not consume actual speculative RNG, and fails closed
if native CUDA Graph capture is incompatible with selected backends.

Caution: individual two-token graph replay is NOT an optimized whole K2
control graph, and the baseline whole-Predictor graph contains sampling,
embeddings and the initial pair while packed graph captures stack ONLY.
The resulting cost screen is deliberately an optimistic lower bound.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

SCHEMA = "qwen3tts-cpspec-p5-graph-cost-v1"
P5_DIR = "SGLANG_OMNI_QTTS_CP_P5_DIR"
BASE_ARM = "SGLANG_OMNI_QTTS_CP_P5_BASE_ARM_FILE"
GRAPH_DEPTHS = "SGLANG_OMNI_QTTS_CP_P5_GRAPH_DEPTHS"
REPEATS = "SGLANG_OMNI_QTTS_CP_P5_REPEATS"
BASE_FRAMES = "SGLANG_OMNI_QTTS_CP_P5_BASE_FRAMES"
P9_BASE_DIR = "SGLANG_OMNI_QTTS_CP_P9_BASE_DIR"
P9_BASE_ARM = "SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"


def config() -> dict | None:
    p5_root = os.environ.get(P5_DIR)
    p9_root = os.environ.get(P9_BASE_DIR)
    if not p5_root and not p9_root:
        return None
    if p5_root and p9_root:
        raise RuntimeError("P5 and P9 Graph baseline environments cannot coexist")
    root = p9_root or p5_root
    if os.environ.get("SGLANG_OMNI_QTTS_CP_P4_DIR") or os.environ.get(
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR"
    ):
        raise RuntimeError("P5/P9 baseline needs P4/P1 instrumentation OFF")
    depths = os.environ.get(GRAPH_DEPTHS, "2,6,10,14")
    try:
        picked = tuple(int(x) for x in depths.split(","))
    except ValueError as exc:
        raise ValueError("P5 depth indices invalid") from exc
    if not picked or len(picked) != len(set(picked)) or any(not 2 <= x <= 15 for x in picked):
        raise ValueError("P5 depths must be unique residual codebook 2..15")
    n = int(os.environ.get(REPEATS, "64"))
    m = int(os.environ.get(BASE_FRAMES, "32"))
    if not 8 <= n <= 1024 or not 8 <= m <= 128:
        raise ValueError("P5 repeats=8..1024 and baseline frames=8..128")
    return {
        "root": Path(root).expanduser().resolve(),
        "depths": picked,
        "repeats": n,
        "baseline_frames": m,
    }


def write_once(root: Path, name: str, report: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = root / name
    with target.open("x", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, "pid": os.getpid(), **report}, handle, indent=2)
        handle.write("\n")
    return target


def baseline_begin(model: "Qwen3TTSTalker") -> torch.cuda.Event | None:
    cfg = config()
    if cfg is None:
        return None
    arm = os.environ.get(
        P9_BASE_ARM if os.environ.get(P9_BASE_DIR) else BASE_ARM
    )
    if not arm or not os.path.isfile(arm):
        return None
    if model.predictor_device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("P5 baseline requires live C1 CUDA outside capture")
    state = getattr(model, "_qt_cp_p5_base", None)
    if state is None:
        state = {"events": [], "replays": 0, "completed": False}
        model._qt_cp_p5_base = state
    if state["completed"]:
        return None
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    return start


def baseline_end(
    model: "Qwen3TTSTalker",
    start: torch.cuda.Event | None,
    *,
    graph_replayed: bool,
) -> None:
    if start is None:
        return
    if not graph_replayed:
        raise RuntimeError(
            "P5_GRAPH_ON_BASELINE_GATE_FAIL: Predictor did not replay CUDA Graph"
        )
    state = model._qt_cp_p5_base
    stop = torch.cuda.Event(enable_timing=True)
    stop.record()
    state["events"].append((start, stop))
    state["replays"] += 1
    cfg = config()
    assert cfg is not None
    if len(state["events"]) < cfg["baseline_frames"]:
        return
    stop.synchronize()
    durations = [a.elapsed_time(b) for a, b in state["events"]]
    if any(not math.isfinite(x) or x <= 0 for x in durations):
        raise RuntimeError("P5 baseline CUDA Graph event timings invalid")
    report = {
        "status": "P5_EXISTING_FULL_PREDICTOR_GRAPH_ON_REPLAY_MEASURED_NOT_HTTP_SPEEDUP",
        "kind": "baseline_full_predictor_graph_on",
        "source_sha": os.environ.get("SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"),
        "device": torch.cuda.get_device_name(model.predictor_device),
        "frames": len(durations),
        "graph_replays": state["replays"],
        "cuda_event_ms_per_frame": durations,
        "median_cuda_event_ms": statistics.median(durations),
        "mean_cuda_event_ms": statistics.mean(durations),
        "min_cuda_event_ms": min(durations),
        "max_cuda_event_ms": max(durations),
        "limitations": [
            "Existing full Predictor CUDA Graph includes pair, all codebooks, samplers and embedding work",
            "Event includes launch/stream envelope, not CUPTI GPU busy time",
            "32 selected baseline semantic frames are correlated and do not prove serving QPS",
            "Must compare to actual optimized no-spec full serving before any release",
        ],
    }
    write_once(cfg["root"], f"baseline_graph_pid{os.getpid()}.json", report)
    state["completed"] = True


def _capture_fixed_graph(
    model: "Qwen3TTSTalker", embeds: torch.Tensor, cache_len: int
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    device = embeds.device
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            model.predictor_forward_tokens(
                token_embeds=embeds, batch_size=1, cache_len=cache_len
            )
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    captured = torch.cuda.CUDAGraph()
    # Capture on the dedicated stream just as the existing predictor-graph
    # startup uses a capture stream; preserve the model's K/V afterwards.
    with torch.cuda.graph(captured, stream=stream), torch.inference_mode():
        output = model.predictor_forward_tokens(
            token_embeds=embeds, batch_size=1, cache_len=cache_len
        )
    torch.cuda.synchronize(device)
    # Keep outputs alive for the full graph replay lifetime, including
    # any capture-private allocation owners.
    return captured, output


def _replay_ms(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    stop.record()
    stop.synchronize()
    val = start.elapsed_time(stop) / repeats
    if not math.isfinite(val) or val <= 0:
        raise RuntimeError("P5 invalid captured-graph replay timing")
    return val


def maybe_profile_packed(
    model: "Qwen3TTSTalker",
    *,
    frame: int,
    depth: int,
    cache_len: int,
    last_embedding: torch.Tensor,
    proposed_embedding: torch.Tensor,
) -> None:
    cfg = config()
    if cfg is None:
        return
    proposal_depth = depth + 1
    seen = getattr(model, "_qt_cp_p5_packed_depths", None)
    if seen is None:
        seen = set()
        model._qt_cp_p5_packed_depths = seen
    if proposal_depth not in cfg["depths"] or proposal_depth in seen:
        return
    if (
        model.predictor_device.type != "cuda"
        or torch.cuda.is_current_stream_capturing()
        or last_embedding.shape[0:2] != (1, 1)
        or proposed_embedding.shape[0:2] != (1, 1)
        or cache_len < 2
    ):
        raise RuntimeError("P5 packed graph only supports C1 CUDA on a real K2 state")
    end = cache_len + 2
    if end > model.predictor_k_cache.shape[2]:
        raise RuntimeError("P5 packed graph exceeds KV window")
    # These same two inputs have already passed P1's real packed-vs-serial
    # correctness gate. Graph capture adds another, separate execution path.
    k_cache = model.predictor_k_cache
    v_cache = model.predictor_v_cache
    old_k = k_cache[:, :1, :end].clone()
    old_v = v_cache[:, :1, :end].clone()
    embed_one = last_embedding.detach().clone()
    embed_pair = torch.cat((last_embedding, proposed_embedding), dim=1).detach().clone()
    try:
        # Before each capture, restore the committed KV. A captured graph will
        # write its own suffix; it must not depend on a prior capture's writes.
        k_cache[:, :1, :end].copy_(old_k)
        v_cache[:, :1, :end].copy_(old_v)
        graph_one, out_one = _capture_fixed_graph(model, embed_one, cache_len)
        k_cache[:, :1, :end].copy_(old_k)
        v_cache[:, :1, :end].copy_(old_v)
        graph_two, out_two = _capture_fixed_graph(model, embed_pair, cache_len)
        # Hold captured outputs for the entire replay window.
        keep_alive = (out_one, out_two)
        assert all(value.numel() > 0 for value in keep_alive)
        observations = []
        for name, graph in (
            ("graph1", graph_one), ("graph2", graph_two),
            ("graph2", graph_two), ("graph1", graph_one),
        ):
            # State reset before each cell; no same-token state drift.
            k_cache[:, :1, :end].copy_(old_k)
            v_cache[:, :1, :end].copy_(old_v)
            observations.append({"cell": name, "cuda_event_ms": _replay_ms(graph, cfg["repeats"])})
        single = statistics.median(o["cuda_event_ms"] for o in observations if o["cell"] == "graph1")
        packed = statistics.median(o["cuda_event_ms"] for o in observations if o["cell"] == "graph2")
        if packed <= 0 or single <= 0:
            raise RuntimeError("P5 invalid graph replay costs")
        record = {
            "status": "P5_INDIVIDUAL_VERIFIER_GRAPHS_CAPTURED_NOT_FULL_K2_SPEEDUP",
            "kind": "single_vs_packed_verifier_cuda_graph",
            "source_sha": os.environ.get("SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"),
            "device": torch.cuda.get_device_name(last_embedding.device),
            "frame": frame,
            "proposal_depth": proposal_depth,
            "cache_len": cache_len,
            "replays_per_cell": cfg["repeats"],
            "serial_one_graph_ms": single,
            "packed_two_graph_ms": packed,
            "packed_to_one_ratio": packed / single,
            "observations": observations,
            "limitations": [
                "Two independent captured subgraphs, not an entire GPU-resident speculative loop",
                "No GPU-side rejection control or bonus branch, no draft head/sampler",
                "CUDA-event replay intervals include graph-launch gaps; not pure GPU kernel busy time",
                "Only one on-policy state at this depth; cannot infer throughput or production equivalence",
            ],
        }
        write_once(
            cfg["root"],
            f"packed_graph_pid{os.getpid()}_frame{frame:05d}_depth{proposal_depth:02d}.json",
            record,
        )
        seen.add(proposal_depth)
    finally:
        torch.cuda.synchronize(last_embedding.device)
        k_cache[:, :1, :end].copy_(old_k)
        v_cache[:, :1, :end].copy_(old_v)
        torch.cuda.synchronize(last_embedding.device)

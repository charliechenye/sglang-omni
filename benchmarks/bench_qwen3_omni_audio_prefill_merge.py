# SPDX-License-Identifier: Apache-2.0
"""Measure Qwen3-Omni audio embedding prologue cost at two separate SHAs.

This benchmark contains only a synthetic embedding/scatter harness. It does not
inspect or predict CUDA-graph eligibility. Run the same file separately at the
PR #1161 baseline and at the model-owned path, for example:

    python benchmarks/bench_qwen3_omni_audio_prefill_merge.py --arm A
    python benchmarks/bench_qwen3_omni_audio_prefill_merge.py --arm B

The default matrix covers 256, 2912, and 8192 audio tokens; batch sizes 1 and
8; contiguous and disjoint spans; and one or all requests carrying audio. The
default hidden size is an explicitly synthetic 2048 BF16 control. No result is
reported when CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ScatterPart:
    destination_cpu: torch.Tensor
    source: torch.Tensor


@dataclass(frozen=True)
class SyntheticCase:
    input_ids: torch.Tensor
    embed_tokens: torch.nn.Embedding
    parts: tuple[ScatterPart, ...]
    total_rows: int
    hidden_size: int
    audio_tokens: int
    batch_size: int
    span_layout: str
    audio_requests: str


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _positions(audio_tokens: int, span_layout: str) -> tuple[torch.Tensor, ...]:
    prefix = 16
    if span_layout == "contiguous":
        return (torch.arange(prefix, prefix + audio_tokens, dtype=torch.long),)

    first_count = audio_tokens // 2
    second_count = audio_tokens - first_count
    first = torch.arange(prefix, prefix + first_count, dtype=torch.long)
    second_start = prefix + first_count + 8
    second = torch.arange(second_start, second_start + second_count, dtype=torch.long)
    return first, second


def _build_case(
    *,
    audio_tokens: int,
    batch_size: int,
    span_layout: str,
    audio_requests: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SyntheticCase:
    per_request_positions = _positions(audio_tokens, span_layout)
    request_length = max(int(positions[-1]) for positions in per_request_positions) + 16
    total_rows = request_length * batch_size
    vocab_size = 32768
    embed_tokens = torch.nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
    input_ids = torch.arange(total_rows, device=device, dtype=torch.long).remainder(
        vocab_size
    )
    audio_request_count = 1 if audio_requests == "one" else batch_size
    parts: list[ScatterPart] = []
    for request_index in range(audio_request_count):
        flat_start = request_index * request_length
        source = torch.randn(
            audio_tokens, hidden_size, device=device, dtype=dtype
        )
        source_offset = 0
        for positions in per_request_positions:
            count = positions.numel()
            destination = positions + flat_start
            parts.append(
                ScatterPart(
                    destination_cpu=destination,
                    source=source[source_offset : source_offset + count],
                )
            )
            source_offset += count
    return SyntheticCase(
        input_ids=input_ids,
        embed_tokens=embed_tokens,
        parts=tuple(parts),
        total_rows=total_rows,
        hidden_size=hidden_size,
        audio_tokens=audio_tokens,
        batch_size=batch_size,
        span_layout=span_layout,
        audio_requests=audio_requests,
    )


def _run_arm(case: SyntheticCase, arm: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run one synthetic prologue; A is per-span, B is one batch scatter."""
    device = case.input_ids.device
    dtype = case.embed_tokens.weight.dtype
    text_start = torch.cuda.Event(enable_timing=True)
    text_end = torch.cuda.Event(enable_timing=True)
    prep_start = torch.cuda.Event(enable_timing=True)
    prep_end = torch.cuda.Event(enable_timing=True)
    source_start = torch.cuda.Event(enable_timing=True)
    source_end = torch.cuda.Event(enable_timing=True)
    index_start = torch.cuda.Event(enable_timing=True)
    index_end = torch.cuda.Event(enable_timing=True)
    text_start.record()
    text_embeds = case.embed_tokens(case.input_ids)
    text_end.record()

    prep_start.record()
    if arm == "A":
        destination_parts = [
            part.destination_cpu.to(device=device, non_blocking=True)
            for part in case.parts
        ]
        destination = None
        destination_cat_count = 0
    else:
        destination_cat_count = int(len(case.parts) > 1)
        destination_cpu = (
            case.parts[0].destination_cpu
            if len(case.parts) == 1
            else torch.cat([part.destination_cpu for part in case.parts], dim=0)
        )
        destination = destination_cpu.to(device=device, non_blocking=True)
        destination_parts = []
    prep_end.record()

    source_start.record()
    source_cat_count = 0
    source_cat_bytes = 0
    if arm == "A":
        source_parts = [
            part.source.to(device=device, dtype=dtype, non_blocking=True)
            for part in case.parts
        ]
        source = None
    else:
        source_cat_count = int(len(case.parts) > 1)
        source_cat_bytes = sum(part.source.numel() * part.source.element_size() for part in case.parts)
        source_parts = []
        source = (
            case.parts[0].source.to(device=device, dtype=dtype, non_blocking=True)
            if len(case.parts) == 1
            else torch.cat(
                [
                    part.source.to(device=device, dtype=dtype, non_blocking=True)
                    for part in case.parts
                ],
                dim=0,
            )
        )
    source_end.record()

    index_copy_count = 0
    index_start.record()
    if arm == "A":
        for destination_part, source_part in zip(destination_parts, source_parts):
            text_embeds.index_copy_(0, destination_part, source_part)
            index_copy_count += 1
    else:
        text_embeds.index_copy_(0, destination, source)
        index_copy_count = 1
    index_end.record()

    metrics = {
        "gpu_events": {
            "text_embedding_ms": (text_start, text_end),
            "destination_prepare_ms": (prep_start, prep_end),
            "source_concat_ms": (source_start, source_end),
            "index_copy_ms": (index_start, index_end),
        },
        "source_cat_count": source_cat_count,
        "source_cat_bytes": source_cat_bytes,
        "destination_cat_count": destination_cat_count,
        "index_copy_count": index_copy_count,
        "torch_cat_count": source_cat_count + destination_cat_count,
    }
    return text_embeds, metrics


def _measure_case(
    case: SyntheticCase, *, arm: str, warmup: int, iterations: int
) -> dict[str, Any]:
    for _ in range(warmup):
        torch.cuda.synchronize()
        output, _ = _run_arm(case, arm)
        torch.cuda.synchronize()
        del output

    samples: list[dict[str, Any]] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        sync_count = 0
        original_synchronize = torch.cuda.synchronize

        def counted_synchronize(*args: Any, **kwargs: Any) -> None:
            nonlocal sync_count
            sync_count += 1
            original_synchronize(*args, **kwargs)

        torch.cuda.synchronize = counted_synchronize
        started = time.perf_counter()
        try:
            output, metrics = _run_arm(case, arm)
        finally:
            torch.cuda.synchronize = original_synchronize
        torch.cuda.synchronize()
        gpu_events = metrics.pop("gpu_events")
        metrics.update(
            {
                name: start_event.elapsed_time(end_event)
                for name, (start_event, end_event) in gpu_events.items()
            }
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        metrics["total_prologue_ms"] = total_ms
        metrics["peak_temp_allocated_bytes"] = max(
            0, torch.cuda.max_memory_allocated() - baseline_allocated
        )
        metrics["cuda_sync_count"] = sync_count
        samples.append(metrics)
        del output

    timed_keys = (
        "text_embedding_ms",
        "destination_prepare_ms",
        "source_concat_ms",
        "index_copy_ms",
        "total_prologue_ms",
        "peak_temp_allocated_bytes",
    )
    summary: dict[str, Any] = {
        "arm": arm,
        "audio_tokens": case.audio_tokens,
        "batch_size": case.batch_size,
        "span_layout": case.span_layout,
        "audio_requests": case.audio_requests,
        "hidden_size": case.hidden_size,
        "dtype": str(case.embed_tokens.weight.dtype).removeprefix("torch."),
        "iterations": iterations,
        "warmup": warmup,
        "metrics": {},
    }
    for key in timed_keys:
        values = [float(sample[key]) for sample in samples]
        summary["metrics"][key] = {
            "median": statistics.median(values),
            "p95": _percentile(values, 95),
        }
    for key in (
        "source_cat_count",
        "source_cat_bytes",
        "destination_cat_count",
        "torch_cat_count",
        "index_copy_count",
        "cuda_sync_count",
    ):
        values = [int(sample[key]) for sample in samples]
        summary["metrics"][key] = {
            "median": int(statistics.median(values)),
            "p95": int(_percentile([float(value) for value in values], 95)),
        }
    return summary


def run_benchmark(
    *,
    arm: str,
    audio_tokens: list[int],
    batch_sizes: list[int],
    span_layouts: list[str],
    audio_requests: list[str],
    hidden_size: int,
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "CUDA is unavailable; no performance result was collected.",
        }
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device")
    results = []
    for token_count in audio_tokens:
        for batch_size in batch_sizes:
            for span_layout in span_layouts:
                for request_mode in audio_requests:
                    case = _build_case(
                        audio_tokens=token_count,
                        batch_size=batch_size,
                        span_layout=span_layout,
                        audio_requests=request_mode,
                        hidden_size=hidden_size,
                        device=torch_device,
                        dtype=torch.bfloat16,
                    )
                    results.append(
                        _measure_case(
                            case,
                            arm=arm,
                            warmup=warmup,
                            iterations=iterations,
                        )
                    )
                    del case
                    torch.cuda.synchronize()
    return {
        "available": True,
        "arm": arm,
        "device": str(torch_device),
        "hidden_size": hidden_size,
        "dtype": "bfloat16",
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument(
        "--audio-tokens", type=int, nargs="+", default=[256, 2912, 8192]
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--span-layout",
        choices=("contiguous", "disjoint"),
        nargs="+",
        default=["contiguous", "disjoint"],
    )
    parser.add_argument(
        "--audio-requests",
        choices=("one", "all"),
        nargs="+",
        default=["one", "all"],
    )
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations positive")
    report = run_benchmark(
        arm=args.arm,
        audio_tokens=args.audio_tokens,
        batch_sizes=args.batch_size,
        span_layouts=args.span_layout,
        audio_requests=args.audio_requests,
        hidden_size=args.hidden_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report.get("available") else 2


if __name__ == "__main__":
    raise SystemExit(main())

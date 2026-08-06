# SPDX-License-Identifier: Apache-2.0
"""A/B/C Thinker prefill benchmark for Qwen3-Omni.

This benchmark is deliberately independent of graph dispatch. The server is
configured externally for each arm; this script records that configuration and
measures client-visible text TTFT, total latency, and request throughput.

Arms:

``A``
    PR #1161 optimized legacy eager baseline, prefill graphs disabled.
``B``
    Model-owned low-copy request/forward path, prefill graphs disabled.
``C``
    Exactly B with upstream breakable prefill graphs enabled.

The script never predicts or reports graph eligibility. Any capture/replay
counts or kernel accounting are supplied by an optional benchmark-only JSON
file produced by the server probe.

Example::

    python benchmarks/eval/benchmark_qwen3_omni_prefill_ab.py \
        --base-url http://127.0.0.1:8000 --arm B --requests 96 \
        --concurrency 16 --prompt-file prompts.txt --output results/qwen-prefill-b.json

For audio-to-text, add ``--audio-path /path/to/sample.wav``. The request
shape follows the existing Omni ``audios`` API used by MMSU benchmarks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp


ARM_DEFAULT_BACKENDS = {
    "A": "disabled",
    "B": "disabled",
    "C": "breakable",
}


@dataclass(frozen=True)
class ArmConfig:
    label: str
    description: str
    prefill_backend: str


@dataclass
class RequestMeasurement:
    arm: str
    request_id: str
    success: bool
    ttft_seconds: float | None = None
    total_seconds: float = 0.0
    output_tokens: int = 0
    status_code: int = 0
    error: str = ""


@dataclass
class BenchmarkSummary:
    arm: ArmConfig
    model: str
    base_url: str
    code_sha: str
    runtime_config: dict[str, Any]
    requests: list[RequestMeasurement] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    accounting: dict[str, Any] | None = None


def _arm_config(label: str, backend: str | None) -> ArmConfig:
    descriptions = {
        "A": "PR #1161 optimized legacy eager baseline",
        "B": "model-owned low-copy path with upstream prefill graphs disabled",
        "C": "B with upstream breakable prefill graphs enabled",
    }
    return ArmConfig(
        label=label,
        description=descriptions[label],
        prefill_backend=backend or ARM_DEFAULT_BACKENDS[label],
    )


def _git_sha() -> str:
    explicit = os.environ.get("QWEN_PREFILL_CODE_SHA")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file is not None:
        prompts = [
            line.strip()
            for line in args.prompt_file.read_text().splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError(f"prompt file is empty: {args.prompt_file}")
        return prompts
    return [args.prompt]


def _build_payload(
    *, model: str, prompt: str, audio_path: Path | None, seed: int, max_tokens: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text"],
        "stream": True,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if audio_path is not None:
        payload["audios"] = [str(audio_path)]
    return payload


async def _measure_one(
    session: aiohttp.ClientSession,
    *,
    args: argparse.Namespace,
    arm: ArmConfig,
    request_id: str,
    prompt: str,
    seed: int,
) -> RequestMeasurement:
    start = time.perf_counter()
    result = RequestMeasurement(arm=arm.label, request_id=request_id, success=False)
    payload = _build_payload(
        model=args.model,
        prompt=prompt,
        audio_path=args.audio_path,
        seed=seed,
        max_tokens=args.max_tokens,
    )
    try:
        async with session.post(
            f"{args.base_url.rstrip('/')}/v1/chat/completions", json=payload
        ) as response:
            result.status_code = response.status
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:512]}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    continue
                event = json.loads(body)
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content") and result.ttft_seconds is None:
                        result.ttft_seconds = time.perf_counter() - start
                usage = event.get("usage") or {}
                result.output_tokens = max(
                    result.output_tokens, int(usage.get("completion_tokens", 0))
                )
        result.success = result.ttft_seconds is not None
        if not result.success:
            result.error = "no text delta received"
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        result.error = str(exc)
    result.total_seconds = time.perf_counter() - start
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate(results: list[RequestMeasurement], wall_clock: float) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    ttft = [result.ttft_seconds for result in successful if result.ttft_seconds is not None]
    totals = [result.total_seconds for result in successful]
    return {
        "attempted_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "ttft_p50_s": _percentile(ttft, 50),
        "ttft_p95_s": _percentile(ttft, 95),
        "ttft_p99_s": _percentile(ttft, 99),
        "total_latency_p50_s": _percentile(totals, 50),
        "total_latency_p95_s": _percentile(totals, 95),
        "requests_per_second": len(successful) / wall_clock if wall_clock else 0.0,
        "output_tokens_per_second": (
            sum(result.output_tokens for result in successful) / wall_clock
            if wall_clock
            else 0.0
        ),
        "ttft_mean_s": statistics.fmean(ttft) if ttft else None,
    }


async def _run(args: argparse.Namespace) -> BenchmarkSummary:
    arm = _arm_config(args.arm, args.prefill_backend)
    prompts = _read_prompts(args)
    runtime_config = {
        "arm": arm.label,
        "prefill_backend": arm.prefill_backend,
        "server_config": args.server_config,
        "benchmark_compatibility_injection": args.compatibility_injection,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "audio_path": str(args.audio_path) if args.audio_path else None,
    }
    summary = BenchmarkSummary(
        arm=arm,
        model=args.model,
        base_url=args.base_url,
        code_sha=_git_sha(),
        runtime_config=runtime_config,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for warmup in range(args.warmup):
            await _measure_one(
                session,
                args=args,
                arm=arm,
                request_id=f"{arm.label}-warmup-{warmup}",
                prompt=prompts[warmup % len(prompts)],
                seed=warmup,
            )

        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_one(index: int) -> RequestMeasurement:
            async with semaphore:
                return await _measure_one(
                    session,
                    args=args,
                    arm=arm,
                    request_id=f"{arm.label}-{index}",
                    prompt=prompts[index % len(prompts)],
                    seed=1000 + index,
                )

        started = time.perf_counter()
        summary.requests = await asyncio.gather(
            *(run_one(index) for index in range(args.requests))
        )
        wall_clock = time.perf_counter() - started

    summary.aggregate = _aggregate(summary.requests, wall_clock)
    if args.accounting_json is not None:
        summary.accounting = json.loads(args.accounting_json.read_text())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--prefill-backend", default=None)
    parser.add_argument("--server-config", default="")
    parser.add_argument("--compatibility-injection", default="none")
    parser.add_argument("--code-sha", default=None)
    parser.add_argument("--prompt", default="Reply with exactly one short sentence.")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--accounting-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("--requests and --concurrency must be positive")
    if args.code_sha:
        os.environ["QWEN_PREFILL_CODE_SHA"] = args.code_sha
    summary = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(json.dumps(summary.aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

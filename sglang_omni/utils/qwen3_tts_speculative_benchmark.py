# SPDX-License-Identifier: Apache-2.0
"""Single-GPU sequential Qwen3-TTS serving benchmark phases."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

from sglang_omni.utils.json import JsonValue


def request_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def request_json(base_url: str, api_key: str, payload: dict[str, JsonValue]) -> bytes:
    request = Request(
        f"{base_url.rstrip('/')}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(api_key),
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        response_body = response.read()
    if not response_body:
        raise RuntimeError("speech endpoint returned an empty response body")
    else:
        pass
    return response_body


def timed_speech_request(
    base_url: str, api_key: str, payload: dict[str, JsonValue]
) -> float:
    started = time.perf_counter()
    request_json(base_url, api_key, payload)
    return time.perf_counter() - started


def run_speech_requests(
    base_url: str,
    api_key: str,
    payload: dict[str, JsonValue],
    request_count: int,
    concurrency: int,
) -> tuple[list[float], float]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(timed_speech_request, base_url, api_key, payload)
            for _ in range(request_count)
        ]
        latencies = [future.result() for future in futures]
    return latencies, time.perf_counter() - started


def fetch_model_info(base_url: str, api_key: str) -> JsonValue:
    request = Request(
        f"{base_url.rstrip('/')}/model_info",
        headers=request_headers(api_key),
        method="GET",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def find_speculative_stats(payload: JsonValue) -> dict[str, JsonValue] | None:
    if isinstance(payload, dict):
        stats = payload.get("predictor_speculative")
        if isinstance(stats, dict):
            return stats
        else:
            pass
        for value in payload.values():
            found = find_speculative_stats(value)
            if found is not None:
                return found
            else:
                pass
    elif isinstance(payload, list):
        for value in payload:
            found = find_speculative_stats(value)
            if found is not None:
                return found
            else:
                pass
    else:
        pass
    return None


def find_predictor_graph_stats(payload: JsonValue) -> dict[str, JsonValue] | None:
    if isinstance(payload, dict):
        stats = payload.get("predictor_cuda_graph")
        if isinstance(stats, dict):
            return stats
        else:
            pass
        for value in payload.values():
            found = find_predictor_graph_stats(value)
            if found is not None:
                return found
            else:
                pass
    elif isinstance(payload, list):
        for value in payload:
            found = find_predictor_graph_stats(value)
            if found is not None:
                return found
            else:
                pass
    else:
        pass
    return None


def build_speech_payload(arguments: argparse.Namespace) -> dict[str, JsonValue]:
    return {
        "model": arguments.model,
        "input": arguments.input,
        "voice": arguments.voice,
        "language": arguments.language,
    }


def run_warmup_requests(arguments: argparse.Namespace, base_url: str) -> None:
    if arguments.warmup_requests > 0:
        run_speech_requests(
            base_url,
            arguments.api_key,
            build_speech_payload(arguments),
            arguments.warmup_requests,
            arguments.concurrency,
        )
    else:
        pass


def run_native_phase(arguments: argparse.Namespace) -> None:
    baseline_path = Path(arguments.baseline_file)
    run_warmup_requests(arguments, arguments.base_url)
    graph_before = find_predictor_graph_stats(
        fetch_model_info(arguments.base_url, arguments.api_key)
    )
    if graph_before is None:
        raise RuntimeError("native server model_info has no Predictor graph metrics")
    else:
        pass
    native_speculative_stats = find_speculative_stats(
        fetch_model_info(arguments.base_url, arguments.api_key)
    )
    if native_speculative_stats is None:
        raise RuntimeError("native server model_info has no speculative setup metrics")
    else:
        pass
    latencies, wall_seconds = run_speech_requests(
        arguments.base_url,
        arguments.api_key,
        build_speech_payload(arguments),
        arguments.request_count,
        arguments.concurrency,
    )
    graph_after = find_predictor_graph_stats(
        fetch_model_info(arguments.base_url, arguments.api_key)
    )
    if graph_after is None:
        raise RuntimeError("native server model_info has no Predictor graph metrics")
    else:
        pass
    graph_replays = integer_stat(graph_after, "replays") - integer_stat(
        graph_before, "replays"
    )
    startup_graphs = integer_stat(graph_after, "startup_graph_count")
    if startup_graphs <= 0 or graph_replays <= 0:
        raise RuntimeError(
            "native baseline did not replay a startup full-Predictor CUDA graph"
        )
    else:
        pass
    baseline = {
        "model": arguments.model,
        "input": arguments.input,
        "voice": arguments.voice,
        "language": arguments.language,
        "request_count": arguments.request_count,
        "concurrency": arguments.concurrency,
        "wall_seconds": wall_seconds,
        "latency_p50_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": percentile(latencies, 0.95) * 1000,
        "predictor_graph_replays": graph_replays,
        "predictor_graph_startup_captures": startup_graphs,
        "speculative_configuration_time_ms": numeric_stat(
            native_speculative_stats, "configuration_time_ms"
        ),
    }
    baseline_path.write_text(json.dumps(baseline, sort_keys=True), encoding="utf-8")
    print(json.dumps({"phase": "native", **baseline}, sort_keys=True))


def run_speculative_phase(arguments: argparse.Namespace) -> None:
    baseline_path = Path(arguments.baseline_file)
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline_payload, dict):
        raise RuntimeError("native baseline file must contain a JSON object")
    elif (
        baseline_payload.get("model") != arguments.model
        or baseline_payload.get("input") != arguments.input
        or baseline_payload.get("voice") != arguments.voice
        or baseline_payload.get("language") != arguments.language
        or baseline_payload.get("request_count") != arguments.request_count
        or baseline_payload.get("concurrency") != arguments.concurrency
    ):
        raise RuntimeError("native and speculative phases must use identical inputs")
    else:
        pass
    run_warmup_requests(arguments, arguments.base_url)
    speculative_stats_before = find_speculative_stats(
        fetch_model_info(arguments.base_url, arguments.api_key)
    )
    if speculative_stats_before is None:
        raise RuntimeError("speculative server model_info has no Predictor metrics")
    else:
        pass
    latencies, wall_seconds = run_speech_requests(
        arguments.base_url,
        arguments.api_key,
        build_speech_payload(arguments),
        arguments.request_count,
        arguments.concurrency,
    )
    speculative_stats_after = find_speculative_stats(
        fetch_model_info(arguments.base_url, arguments.api_key)
    )
    if speculative_stats_after is None:
        raise RuntimeError("speculative server model_info has no Predictor metrics")
    else:
        pass

    speculative_frames = integer_stat(speculative_stats_after, "frames") - integer_stat(
        speculative_stats_before, "frames"
    )
    native_fallbacks = integer_stat(
        speculative_stats_after, "native_fallback_count"
    ) - integer_stat(speculative_stats_before, "native_fallback_count")
    packed_forwards = integer_stat(
        speculative_stats_after, "packed_transformer_forwards"
    ) - integer_stat(speculative_stats_before, "packed_transformer_forwards")
    graph_replays = integer_stat(
        speculative_stats_after, "graph_replays"
    ) - integer_stat(speculative_stats_before, "graph_replays")
    graph_captures = integer_stat(speculative_stats_after, "graph_captures")
    if speculative_frames <= 0:
        raise RuntimeError("performance gate failed: speculative frame count is zero")
    elif native_fallbacks != 0:
        raise RuntimeError("performance gate failed: speculative workload fell back")
    elif packed_forwards <= 0 or graph_replays <= 0 or graph_captures <= 0:
        raise RuntimeError("performance gate failed: speculative graph did not replay")
    else:
        pass

    native_wall_seconds = float(baseline_payload["wall_seconds"])
    speedup = native_wall_seconds / wall_seconds
    if speedup < arguments.minimum_speedup:
        raise RuntimeError(
            "performance gate failed: measured serving speedup "
            f"{speedup:.3f} is below {arguments.minimum_speedup:.3f}"
        )
    else:
        pass
    attempted_frames = integer_stat(
        speculative_stats_after, "attempted_frames"
    ) - integer_stat(speculative_stats_before, "attempted_frames")
    fused_stack_traversals = integer_stat(
        speculative_stats_after, "fused_predictor_forwards"
    ) - integer_stat(speculative_stats_before, "fused_predictor_forwards")
    valid_query_slots = integer_stat(
        speculative_stats_after, "packed_query_slots"
    ) - integer_stat(speculative_stats_before, "packed_query_slots")
    padded_query_slots = integer_stat(
        speculative_stats_after, "padded_query_slots"
    ) - integer_stat(speculative_stats_before, "padded_query_slots")
    total_query_slots = valid_query_slots + padded_query_slots
    summary = {
        "phase": "speculative",
        "requests": arguments.request_count,
        "concurrency": arguments.concurrency,
        "speculative_frames": speculative_frames,
        "native_fallbacks": native_fallbacks,
        "native_fallback_fraction": (
            native_fallbacks / attempted_frames if attempted_frames > 0 else 0.0
        ),
        "packed_transformer_forwards": packed_forwards,
        "packed_forwards_per_request_frame": packed_forwards / speculative_frames,
        "fused_predictor_stack_traversals": fused_stack_traversals,
        "stack_traversals_per_request_frame": (
            fused_stack_traversals / speculative_frames
        ),
        "speculative_graph_replays": graph_replays,
        "speculative_graph_captures": graph_captures,
        "speculative_graph_capture_memory_bytes": integer_stat(
            speculative_stats_after, "graph_capture_memory_delta_bytes"
        ),
        "speculative_graph_capture_time_ms": numeric_stat(
            speculative_stats_after, "graph_capture_time_ms"
        ),
        "speculative_configuration_time_ms": numeric_stat(
            speculative_stats_after, "configuration_time_ms"
        ),
        "enabled_minus_disabled_setup_time_ms": numeric_stat(
            speculative_stats_after, "configuration_time_ms"
        )
        - float(baseline_payload["speculative_configuration_time_ms"]),
        "serving_speedup": speedup,
        "native_wall_seconds": native_wall_seconds,
        "speculative_wall_seconds": wall_seconds,
        "native_latency_p50_ms": baseline_payload["latency_p50_ms"],
        "native_latency_p95_ms": baseline_payload["latency_p95_ms"],
        "speculative_latency_p50_ms": statistics.median(latencies) * 1000,
        "speculative_latency_p95_ms": percentile(latencies, 0.95) * 1000,
        "generic_predictor_forwards_delta": integer_stat(
            speculative_stats_after, "generic_predictor_forwards"
        )
        - integer_stat(speculative_stats_before, "generic_predictor_forwards"),
        "standalone_correction_forward_count_delta": integer_stat(
            speculative_stats_after, "standalone_correction_forward_count"
        )
        - integer_stat(speculative_stats_before, "standalone_correction_forward_count"),
        "committed_codes_delta": integer_stat(
            speculative_stats_after, "committed_codes"
        )
        - integer_stat(speculative_stats_before, "committed_codes"),
        "accepted_draft_tokens_delta": integer_stat(
            speculative_stats_after, "accepted_draft_tokens"
        )
        - integer_stat(speculative_stats_before, "accepted_draft_tokens"),
        "bonus_tokens_delta": integer_stat(speculative_stats_after, "bonus_tokens")
        - integer_stat(speculative_stats_before, "bonus_tokens"),
        "draft_head_evaluations_delta": integer_stat(
            speculative_stats_after, "draft_head_evaluations"
        )
        - integer_stat(speculative_stats_before, "draft_head_evaluations"),
        "target_head_evaluations_delta": integer_stat(
            speculative_stats_after, "target_head_evaluations"
        )
        - integer_stat(speculative_stats_before, "target_head_evaluations"),
        "effective_width_histogram": histogram_delta(
            speculative_stats_before, speculative_stats_after
        ),
        "valid_query_slots_delta": valid_query_slots,
        "padded_query_slots_delta": padded_query_slots,
        "query_padding_fraction": (
            padded_query_slots / total_query_slots if total_query_slots > 0 else 0.0
        ),
    }
    print(json.dumps(summary, sort_keys=True))


def integer_stat(stats: dict[str, JsonValue], name: str) -> int:
    value = stats.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"model_info is missing integer predictor metric {name!r}")
    else:
        pass
    return value


def numeric_stat(stats: dict[str, JsonValue], name: str) -> float:
    value = stats.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"model_info is missing numeric predictor metric {name!r}")
    else:
        pass
    return float(value)


def histogram_delta(
    before: dict[str, JsonValue], after: dict[str, JsonValue]
) -> dict[str, int]:
    before_value = before.get("effective_width_histogram")
    after_value = after.get("effective_width_histogram")
    if not isinstance(before_value, dict) or not isinstance(after_value, dict):
        raise RuntimeError("model_info is missing speculative width histograms")
    else:
        pass
    delta: dict[str, int] = {}
    for width, count in after_value.items():
        before_count = before_value.get(width, 0)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or isinstance(before_count, bool)
            or not isinstance(before_count, int)
        ):
            raise RuntimeError("model_info contains a malformed width histogram")
        else:
            pass
        width_delta = count - before_count
        if width_delta > 0:
            delta[width] = width_delta
        else:
            pass
    return delta


def percentile(latencies: list[float], quantile: float) -> float:
    ordered = sorted(latencies)
    index = min(math.ceil(quantile * len(ordered)) - 1, len(ordered) - 1)
    return ordered[index]

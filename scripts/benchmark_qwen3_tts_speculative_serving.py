# SPDX-License-Identifier: Apache-2.0
"""Compare native and speculative Qwen3-TTS serving with an execution gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from urllib.request import Request, urlopen

from sglang_omni.utils.json import JsonValue


def positive_integer(value: str) -> int:
    """Parse a command line integer that must be positive."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    else:
        pass
    return parsed


def nonnegative_integer(value: str) -> int:
    """Parse a command line integer that may be zero."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    else:
        pass
    return parsed


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


def run_benchmark(arguments: argparse.Namespace) -> None:
    payload: dict[str, JsonValue] = {
        "model": arguments.model,
        "input": arguments.input,
        "voice": arguments.voice,
        "language": arguments.language,
    }
    if arguments.warmup_requests > 0:
        run_speech_requests(
            arguments.native_url,
            arguments.api_key,
            payload,
            arguments.warmup_requests,
            arguments.concurrency,
        )
        run_speech_requests(
            arguments.speculative_url,
            arguments.api_key,
            payload,
            arguments.warmup_requests,
            arguments.concurrency,
        )
    else:
        pass

    native_latencies, native_wall_seconds = run_speech_requests(
        arguments.native_url,
        arguments.api_key,
        payload,
        arguments.request_count,
        arguments.concurrency,
    )
    native_stats = find_speculative_stats(
        fetch_model_info(arguments.native_url, arguments.api_key)
    )
    if native_stats is None:
        raise RuntimeError("native server model_info has no predictor metrics")
    else:
        pass
    speculative_stats_before = find_speculative_stats(
        fetch_model_info(arguments.speculative_url, arguments.api_key)
    )
    if speculative_stats_before is None:
        raise RuntimeError("speculative server model_info has no predictor metrics")
    else:
        pass
    speculative_latencies, speculative_wall_seconds = run_speech_requests(
        arguments.speculative_url,
        arguments.api_key,
        payload,
        arguments.request_count,
        arguments.concurrency,
    )
    speculative_stats_after = find_speculative_stats(
        fetch_model_info(arguments.speculative_url, arguments.api_key)
    )
    if speculative_stats_after is None:
        raise RuntimeError("speculative server model_info has no predictor metrics")
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
    packed_query_slots = integer_stat(
        speculative_stats_after, "packed_query_slots"
    ) - integer_stat(speculative_stats_before, "packed_query_slots")
    padded_query_slots = integer_stat(
        speculative_stats_after, "padded_query_slots"
    ) - integer_stat(speculative_stats_before, "padded_query_slots")
    fused_predictor_forwards = integer_stat(
        speculative_stats_after, "fused_predictor_forwards"
    ) - integer_stat(speculative_stats_before, "fused_predictor_forwards")
    if speculative_frames <= 0:
        raise RuntimeError("performance gate failed: speculative frame count is zero")
    else:
        pass
    if native_fallbacks != 0:
        raise RuntimeError(
            "performance gate failed: speculative workload used native fallback "
            f"for {native_fallbacks} frames"
        )
    else:
        pass
    if packed_forwards <= 0:
        raise RuntimeError("performance gate failed: no packed verifier forwards ran")
    else:
        pass

    speculative_ms_per_frame = speculative_wall_seconds * 1000 / speculative_frames
    native_speedup = native_wall_seconds / speculative_wall_seconds
    if native_speedup < arguments.minimum_speedup:
        raise RuntimeError(
            "performance gate failed: measured serving speedup "
            f"{native_speedup:.3f} is below {arguments.minimum_speedup:.3f}"
        )
    else:
        pass
    speculative_summary = {
        "frames": speculative_frames,
        "native_fallbacks": native_fallbacks,
        "native_fallback_fraction": 0.0,
        "packed_transformer_forwards": packed_forwards,
        "packed_forwards_per_frame": packed_forwards / speculative_frames,
        "fused_predictor_forwards_delta": fused_predictor_forwards,
        "fused_predictor_forwards_per_frame": (
            fused_predictor_forwards / speculative_frames
        ),
        "generic_predictor_forwards": integer_stat(
            speculative_stats_after, "generic_predictor_forwards"
        ),
        "standalone_correction_forward_count": integer_stat(
            speculative_stats_after, "standalone_correction_forward_count"
        ),
        "draft_head_evaluations_delta": integer_stat(
            speculative_stats_after, "draft_head_evaluations"
        )
        - integer_stat(speculative_stats_before, "draft_head_evaluations"),
        "target_head_evaluations_delta": integer_stat(
            speculative_stats_after, "target_head_evaluations"
        )
        - integer_stat(speculative_stats_before, "target_head_evaluations"),
        "rounds_delta": integer_stat(speculative_stats_after, "rounds")
        - integer_stat(speculative_stats_before, "rounds"),
        "effective_width_histogram": histogram_delta(
            speculative_stats_before, speculative_stats_after
        ),
        "committed_codes_delta": integer_stat(
            speculative_stats_after, "committed_codes"
        )
        - integer_stat(speculative_stats_before, "committed_codes"),
        "round_committed_codes_delta": integer_stat(
            speculative_stats_after, "round_committed_codes"
        )
        - integer_stat(speculative_stats_before, "round_committed_codes"),
        "proposed_draft_tokens_delta": integer_stat(
            speculative_stats_after, "proposed_draft_tokens"
        )
        - integer_stat(speculative_stats_before, "proposed_draft_tokens"),
        "accepted_draft_tokens_delta": integer_stat(
            speculative_stats_after, "accepted_draft_tokens"
        )
        - integer_stat(speculative_stats_before, "accepted_draft_tokens"),
        "bonus_tokens_delta": integer_stat(speculative_stats_after, "bonus_tokens")
        - integer_stat(speculative_stats_before, "bonus_tokens"),
        "packed_query_slots_delta": packed_query_slots,
        "padded_query_slots_delta": padded_query_slots,
        "padded_query_fraction": (
            padded_query_slots / packed_query_slots if packed_query_slots > 0 else 0.0
        ),
        "graph_capture_memory_delta_bytes": integer_stat(
            speculative_stats_after, "graph_capture_memory_delta_bytes"
        ),
        "graph_capture_time_ms": speculative_stats_after.get("graph_capture_time_ms"),
        "graph_captures": integer_stat(speculative_stats_after, "graph_captures"),
        "graph_replays_delta": integer_stat(speculative_stats_after, "graph_replays")
        - integer_stat(speculative_stats_before, "graph_replays"),
    }
    print(
        json.dumps(
            {
                "requests": arguments.request_count,
                "concurrency": arguments.concurrency,
                "unseeded_requests": True,
                "native_wall_seconds": native_wall_seconds,
                "speculative_wall_seconds": speculative_wall_seconds,
                "serving_speedup": native_speedup,
                "native_latency_p50_ms": statistics.median(native_latencies) * 1000,
                "native_latency_p95_ms": percentile(native_latencies, 0.95) * 1000,
                "speculative_latency_p50_ms": (
                    statistics.median(speculative_latencies) * 1000
                ),
                "speculative_latency_p95_ms": (
                    percentile(speculative_latencies, 0.95) * 1000
                ),
                "speculative_wall_ms_per_frame": speculative_ms_per_frame,
                "native_speculative_setup_time_ms": numeric_stat(
                    native_stats, "configuration_time_ms"
                ),
                "speculative_setup_time_ms": numeric_stat(
                    speculative_stats_after, "configuration_time_ms"
                ),
                "predictor_speculative": speculative_summary,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--speculative-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument(
        "--input", default="Please speak a short sentence for this benchmark."
    )
    parser.add_argument("--voice", default="Ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-count", type=positive_integer, default=32)
    parser.add_argument("--warmup-requests", type=nonnegative_integer, default=2)
    parser.add_argument("--concurrency", type=positive_integer, default=4)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.minimum_speedup) or arguments.minimum_speedup < 0.0:
        parser.error("--minimum-speedup must be finite and nonnegative")
    else:
        pass
    run_benchmark(arguments)


if __name__ == "__main__":
    main()

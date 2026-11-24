# SPDX-License-Identifier: Apache-2.0
"""Single-GPU sequential Qwen3-TTS serving benchmark phases."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from sglang_omni.utils.json import JsonValue


@dataclass(frozen=True)
class SpeechResult:
    """Measured latency and decoded audio facts for one complete response."""

    request_index: int
    latency_seconds: float
    audio_bytes: int
    audio_duration_seconds: float
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    rms: float
    peak: float
    clipping_fraction: float


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


def parse_wav_audio(audio: bytes) -> tuple[float, int, int, int, float, float, float]:
    """Validate a PCM WAV response and calculate duration and basic signal checks."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate_hz = wav_file.getframerate()
            sample_width_bytes = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            sample_bytes = wav_file.readframes(frame_count)
    except (wave.Error, EOFError) as error:
        raise RuntimeError("speech endpoint did not return a valid PCM WAV") from error
    if channels <= 0 or sample_rate_hz <= 0 or frame_count <= 0:
        raise RuntimeError("speech endpoint returned an empty WAV stream")
    else:
        pass
    if len(sample_bytes) != frame_count * channels * sample_width_bytes:
        raise RuntimeError("speech endpoint returned a truncated WAV stream")
    else:
        pass
    if sample_width_bytes == 1:
        samples = (
            np.frombuffer(sample_bytes, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif sample_width_bytes == 2:
        samples = np.frombuffer(sample_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width_bytes == 3:
        triplets = np.frombuffer(sample_bytes, dtype=np.uint8).reshape(-1, 3)
        values = (
            triplets[:, 0].astype(np.int32)
            | (triplets[:, 1].astype(np.int32) << 8)
            | (triplets[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608.0
    elif sample_width_bytes == 4:
        samples = (
            np.frombuffer(sample_bytes, dtype="<i4").astype(np.float32) / 2147483648.0
        )
    else:
        raise RuntimeError(
            f"speech endpoint returned unsupported PCM width {sample_width_bytes}"
        )
    if samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError("speech endpoint returned invalid PCM samples")
    else:
        pass
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    clipping_fraction = float(np.mean(np.abs(samples) >= 0.999))
    if peak <= 0.0 or rms <= 1.0e-6:
        raise RuntimeError("speech endpoint returned silent audio")
    else:
        pass
    return (
        frame_count / sample_rate_hz,
        sample_rate_hz,
        channels,
        sample_width_bytes,
        rms,
        peak,
        clipping_fraction,
    )


def timed_speech_request(
    base_url: str,
    api_key: str,
    payload: dict[str, JsonValue],
    request_index: int,
) -> SpeechResult:
    started = time.perf_counter()
    audio = request_json(base_url, api_key, payload)
    latency_seconds = time.perf_counter() - started
    (
        audio_duration_seconds,
        sample_rate_hz,
        channels,
        sample_width_bytes,
        rms,
        peak,
        clipping_fraction,
    ) = parse_wav_audio(audio)
    return SpeechResult(
        request_index=request_index,
        latency_seconds=latency_seconds,
        audio_bytes=len(audio),
        audio_duration_seconds=audio_duration_seconds,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        rms=rms,
        peak=peak,
        clipping_fraction=clipping_fraction,
    )


def run_speech_requests(
    base_url: str,
    api_key: str,
    payload: dict[str, JsonValue],
    request_count: int,
    concurrency: int,
) -> tuple[list[SpeechResult], float]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                timed_speech_request, base_url, api_key, payload, request_index
            )
            for request_index in range(request_count)
        ]
        results = [future.result() for future in futures]
    return results, time.perf_counter() - started


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
        "response_format": "wav",
        "stream": False,
        "max_new_tokens": arguments.max_new_tokens,
    }


def summarize_speech_results(results: list[SpeechResult]) -> dict[str, JsonValue]:
    """Summarize complete audio responses and retain each measured request."""
    latencies = [result.latency_seconds for result in results]
    durations = [result.audio_duration_seconds for result in results]
    real_time_factors = [
        result.latency_seconds / result.audio_duration_seconds for result in results
    ]
    if not results:
        raise ValueError("speech benchmark produced no request results")
    else:
        pass
    return {
        "response_count": len(results),
        "audio_bytes_total": sum(result.audio_bytes for result in results),
        "audio_duration_seconds_total": sum(durations),
        "audio_duration_p50_seconds": statistics.median(durations),
        "audio_duration_p95_seconds": percentile(durations, 0.95),
        "real_time_factor_p50": statistics.median(real_time_factors),
        "real_time_factor_p95": percentile(real_time_factors, 0.95),
        "pcm_rms_mean": statistics.fmean(result.rms for result in results),
        "pcm_peak_max": max(result.peak for result in results),
        "pcm_clipping_fraction_max": max(
            result.clipping_fraction for result in results
        ),
        "ttfa_measured": False,
        "requests": [asdict(result) for result in results],
        "latency_p50_ms": statistics.median(latencies) * 1000.0,
        "latency_p95_ms": percentile(latencies, 0.95) * 1000.0,
    }


def write_json_new(path: Path, value: dict[str, JsonValue]) -> None:
    """Write a fresh benchmark artifact without overwriting prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


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
    model_info_before = fetch_model_info(arguments.base_url, arguments.api_key)
    graph_before = find_predictor_graph_stats(model_info_before)
    if graph_before is None:
        raise RuntimeError("native server model_info has no Predictor graph metrics")
    else:
        pass
    stats_before = find_speculative_stats(model_info_before)
    if stats_before is None:
        raise RuntimeError("native server model_info has no speculative setup metrics")
    else:
        pass
    requested_width = integer_stat(stats_before, "requested_width")
    if requested_width != 0:
        raise RuntimeError("native baseline server must have speculation disabled")
    else:
        pass
    results, wall_seconds = run_speech_requests(
        arguments.base_url,
        arguments.api_key,
        build_speech_payload(arguments),
        arguments.request_count,
        arguments.concurrency,
    )
    model_info_after = fetch_model_info(arguments.base_url, arguments.api_key)
    graph_after = find_predictor_graph_stats(model_info_after)
    if graph_after is None:
        raise RuntimeError("native server model_info has no Predictor graph metrics")
    else:
        pass
    stats_after = find_speculative_stats(model_info_after)
    if stats_after is None:
        raise RuntimeError("native server model_info has no Predictor frame metrics")
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
    output_code_frames = integer_stat(stats_after, "output_code_frames") - integer_stat(
        stats_before, "output_code_frames"
    )
    if output_code_frames <= 0:
        raise RuntimeError("native baseline produced no Predictor codec frames")
    else:
        pass
    audio_summary = summarize_speech_results(results)
    baseline = {
        "model": arguments.model,
        "input": arguments.input,
        "voice": arguments.voice,
        "language": arguments.language,
        "max_new_tokens": arguments.max_new_tokens,
        "request_count": arguments.request_count,
        "concurrency": arguments.concurrency,
        "wall_seconds": wall_seconds,
        "requests_per_second": arguments.request_count / wall_seconds,
        "output_code_frames": output_code_frames,
        "output_code_frames_per_second": output_code_frames / wall_seconds,
        "predictor_graph_replays": graph_replays,
        "predictor_graph_startup_captures": startup_graphs,
        "speculative_configuration_time_ms": numeric_stat(
            stats_after, "configuration_time_ms"
        ),
        "audio": audio_summary,
    }
    write_json_new(baseline_path, baseline)
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
        or baseline_payload.get("max_new_tokens") != arguments.max_new_tokens
        or baseline_payload.get("request_count") != arguments.request_count
        or baseline_payload.get("concurrency") != arguments.concurrency
    ):
        raise RuntimeError("native and speculative phases must use identical inputs")
    else:
        pass
    run_warmup_requests(arguments, arguments.base_url)
    model_info_before = fetch_model_info(arguments.base_url, arguments.api_key)
    speculative_stats_before = find_speculative_stats(model_info_before)
    if speculative_stats_before is None:
        raise RuntimeError("speculative server model_info has no Predictor metrics")
    else:
        pass
    requested_width = integer_stat(speculative_stats_before, "requested_width")
    if requested_width <= 0:
        raise RuntimeError("speculative arm is configured with width 0")
    elif (
        arguments.expected_width is not None
        and requested_width != arguments.expected_width
    ):
        raise RuntimeError(
            "speculative server width does not match the requested arm: "
            f"expected {arguments.expected_width}, got {requested_width}"
        )
    else:
        pass
    results, wall_seconds = run_speech_requests(
        arguments.base_url,
        arguments.api_key,
        build_speech_payload(arguments),
        arguments.request_count,
        arguments.concurrency,
    )
    model_info_after = fetch_model_info(arguments.base_url, arguments.api_key)
    speculative_stats_after = find_speculative_stats(model_info_after)
    if speculative_stats_after is None:
        raise RuntimeError("speculative server model_info has no Predictor metrics")
    else:
        pass

    speculative_frames = integer_stat(speculative_stats_after, "frames") - integer_stat(
        speculative_stats_before, "frames"
    )
    output_code_frames = integer_stat(
        speculative_stats_after, "output_code_frames"
    ) - integer_stat(speculative_stats_before, "output_code_frames")
    native_fallbacks = integer_stat(
        speculative_stats_after, "native_fallback_count"
    ) - integer_stat(speculative_stats_before, "native_fallback_count")
    packed_forwards = integer_stat(
        speculative_stats_after, "packed_transformer_forwards"
    ) - integer_stat(speculative_stats_before, "packed_transformer_forwards")
    verifier_graph_replays = integer_stat(
        speculative_stats_after, "verifier_graph_replays"
    ) - integer_stat(speculative_stats_before, "verifier_graph_replays")
    graph_mode = speculative_stats_after.get("graph_mode")
    if not isinstance(graph_mode, str):
        raise RuntimeError("speculative server has no graph execution mode metric")
    else:
        pass
    graph_captures = integer_stat(speculative_stats_after, "graph_captures")
    if speculative_frames <= 0:
        raise RuntimeError("performance gate failed: speculative frame count is zero")
    elif output_code_frames != speculative_frames:
        raise RuntimeError(
            "performance gate failed: output frame count differs from successful "
            "speculative frame rows"
        )
    elif native_fallbacks != 0:
        raise RuntimeError("performance gate failed: speculative workload fell back")
    elif packed_forwards <= 0 or verifier_graph_replays <= 0 or graph_captures <= 0:
        raise RuntimeError("performance gate failed: verifier graph did not replay")
    elif (
        integer_stat(speculative_stats_after, "standalone_correction_forward_count")
        - integer_stat(speculative_stats_before, "standalone_correction_forward_count")
        != 0
    ):
        raise RuntimeError(
            "performance gate failed: standalone correction forward was observed"
        )
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
    audio_summary = summarize_speech_results(results)
    baseline_requests = baseline_payload.get("audio")
    if not isinstance(baseline_requests, dict) or not isinstance(
        baseline_requests.get("requests"), list
    ):
        raise RuntimeError("native baseline has no per-request audio measurements")
    else:
        pass
    baseline_request_rows = baseline_requests["requests"]
    if len(baseline_request_rows) != len(results):
        raise RuntimeError("native and speculative response counts do not match")
    else:
        pass
    matched_latency_ratios: list[float] = []
    for request_index, result in enumerate(results):
        baseline_row = baseline_request_rows[request_index]
        if not isinstance(baseline_row, dict):
            raise RuntimeError("native baseline has a malformed request measurement")
        else:
            pass
        baseline_latency = baseline_row.get("latency_seconds")
        if (
            isinstance(baseline_latency, bool)
            or not isinstance(baseline_latency, (int, float))
            or baseline_latency <= 0.0
        ):
            raise RuntimeError("native baseline has an invalid request latency")
        else:
            pass
        matched_latency_ratios.append(float(baseline_latency) / result.latency_seconds)
    summary = {
        "phase": "speculative",
        "requested_width": requested_width,
        "requests": arguments.request_count,
        "concurrency": arguments.concurrency,
        "speculative_frames": speculative_frames,
        "output_code_frames": output_code_frames,
        "output_code_frames_per_second": output_code_frames / wall_seconds,
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
        "verifier_graph_replays": verifier_graph_replays,
        "speculative_graph_mode": graph_mode,
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
        "requests_per_second": arguments.request_count / wall_seconds,
        "matched_request_latency_ratio_p50": statistics.median(matched_latency_ratios),
        "matched_request_latency_ratio_p95": percentile(matched_latency_ratios, 0.95),
        "native_wall_seconds": native_wall_seconds,
        "speculative_wall_seconds": wall_seconds,
        "native_latency_p50_ms": baseline_requests.get("latency_p50_ms"),
        "native_latency_p95_ms": baseline_requests.get("latency_p95_ms"),
        "speculative_latency_p50_ms": audio_summary["latency_p50_ms"],
        "speculative_latency_p95_ms": audio_summary["latency_p95_ms"],
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
        "audio": audio_summary,
    }
    if arguments.result_file is not None:
        write_json_new(Path(arguments.result_file), summary)
    else:
        pass
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

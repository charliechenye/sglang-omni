# SPDX-License-Identifier: Apache-2.0
"""Small, local-only screen for Qwen3-ASR authoritative Final SSE candidates."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import math
import platform
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

from sglang_omni_mlx.native.tests.native_server import SupervisedServer, multipart_form

ARMS = ("off", "shadow", "shadow_dense")
FIXTURES = (
    ("en-2", Path("Voxt/testdata/eval/en-2.wav"), "English"),
    ("en-1", Path("Voxt/testdata/eval/en-1.wav"), "English"),
    ("zh-filler-1", Path("Voxt/testdata/eval/zh-filler-1.wav"), "Chinese"),
)
REQUEST_FIELDS = {
    "stream": "true",
    "include_generation_metadata": "true",
    "stop_on_token_loop": "true",
    "stop_at_end_of_text": "true",
    "audio_layout": "voxt_swift",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def common_prefix_length(left: str, right: str) -> int:
    for index, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return index
    return min(len(left), len(right))


def candidate_metrics(
    candidates: list[dict[str, Any]],
    final_text: str,
    final_ns: int | None,
    debug_text: bool,
) -> dict[str, Any]:
    final_length = len(final_text)
    measured: list[dict[str, Any]] = []
    for candidate in candidates:
        text = candidate["_text"]
        prefix_length = common_prefix_length(text, final_text)
        item = {
                **{key: value for key, value in candidate.items() if key != "_text"},
                "exact_transcript_match": text == final_text,
                "length_ratio": len(text) / final_length if final_length else None,
                "common_prefix_length": prefix_length,
                "common_prefix_completeness": prefix_length / final_length
                if final_length
                else None,
                "client_lead_ms": (
                    (final_ns - candidate["arrival_monotonic_ns"]) / 1_000_000
                    if final_ns is not None
                    else None
                ),
                "transcript_sha256": sha256(text.encode("utf-8")),
                "transcript_chars": len(text),
            }
        if debug_text:
            item["fixture_debug_text"] = text
        measured.append(item)
    valid = [
        item
        for item in measured
        if item["common_prefix_completeness"] is not None
    ]
    closest = max(
        valid,
        key=lambda item: (item["common_prefix_completeness"], item["client_lead_ms"]),
        default=None,
    )
    lead_at_threshold = {}
    for threshold in (0.90, 0.95, 1.0):
        candidates_at_threshold = [
            item for item in valid if item["common_prefix_completeness"] >= threshold
        ]
        earliest = min(
            candidates_at_threshold,
            key=lambda item: item["arrival_monotonic_ns"],
            default=None,
        )
        lead_at_threshold[f"{int(threshold * 100)}_percent_ms"] = (
            earliest["client_lead_ms"] if earliest is not None else None
        )
    exact = [item for item in measured if item["exact_transcript_match"]]
    earliest_exact = min(exact, key=lambda item: item["arrival_monotonic_ns"], default=None)
    return {
        "candidates": measured,
        "earliest_exact_match_lead_ms": (
            earliest_exact["client_lead_ms"] if earliest_exact is not None else None
        ),
        "closest_candidate_lead_ms": closest["client_lead_ms"] if closest else None,
        "closest_candidate_prefix_completeness": (
            closest["common_prefix_completeness"] if closest else None
        ),
        "lead_at_prefix_completeness": lead_at_threshold,
    }


def transcribe(
    server: SupervisedServer,
    audio: bytes,
    fixture: str,
    audio_sha256: str,
    language_mode: str,
    explicit_language: str,
    arm: str,
    cadence: str | None,
    round_number: int,
    warmup: bool,
    timeout_seconds: float,
    max_new_tokens: int | None,
    audio_seconds: float,
    oracle_guess: str | None = None,
    debug_text: bool = False,
) -> tuple[dict[str, Any], str | None]:
    fields = dict(REQUEST_FIELDS)
    if language_mode == "explicit":
        fields["language"] = explicit_language
    if max_new_tokens is not None:
        fields["max_new_tokens"] = str(max_new_tokens)
    if arm != "off":
        fields["include_transcript_candidates"] = "true"
        fields["candidate_cadence"] = cadence or "bounded"
    if oracle_guess is not None:
        fields["guessed_text"] = oracle_guess

    boundary, body = multipart_form(fields, audio)
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=timeout_seconds)
    request_started_ns = time.monotonic_ns()
    response_headers_ns: int | None = None
    protocol_supported: bool | None = None
    protocol_cadence = ""
    candidate_summary_observed = False
    summary: dict[str, Any] = {}
    candidate_events: list[dict[str, Any]] = []
    final_text: str | None = None
    final_arrival_ns: int | None = None
    done_arrival_ns: int | None = None
    terminal_error: str | None = None
    http_status: int | None = None
    failure_kind = ""
    try:
        connection.request(
            "POST",
            "/v1/audio/transcriptions",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        response_headers_ns = time.monotonic_ns()
        http_status = response.status
        if response.status != 200:
            response.read()
            failure_kind = f"http_{response.status}"
        else:
            for line in response:
                if time.monotonic() - request_started_ns / 1_000_000_000 > timeout_seconds:
                    raise TimeoutError("SSE request deadline exceeded")
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                arrival_ns = time.monotonic_ns()
                if payload == b"[DONE]":
                    done_arrival_ns = arrival_ns
                    break
                event = json.loads(payload)
                event_type = event.get("type")
                if event_type == "transcript.text.candidate_protocol":
                    protocol_supported = event.get("supported")
                    protocol_cadence = event.get("candidate_cadence", "")
                elif event_type == "transcript.text.candidate":
                    text = event.get("text", "")
                    candidate_events.append(
                        {
                            "sequence": event.get("sequence"),
                            "generated_token_count": event.get("generated_token_count"),
                            "native_elapsed_ms": event.get("elapsed_ms"),
                            "first_transcript_token_elapsed_ms": event.get(
                                "first_transcript_token_elapsed_ms"
                            ),
                            "snapshot_build_ms": event.get("snapshot_build_ms"),
                            "snapshot_build_ms_total": event.get("snapshot_build_ms_total"),
                            "published_candidate_count": event.get(
                                "published_candidate_count"
                            ),
                            "language": event.get("language"),
                            "arrival_monotonic_ns": arrival_ns,
                            "arrival_ms_since_request_start": (
                                arrival_ns - request_started_ns
                            )
                            / 1_000_000,
                            "transcript_sha256": sha256(text.encode("utf-8")),
                            "transcript_chars": len(text),
                            "_text": text,
                        }
                    )
                elif event_type == "transcript.text.candidate_summary":
                    candidate_summary_observed = True
                    summary = {
                        "native_published_candidate_count": event.get(
                            "published_candidate_count"
                        ),
                        "native_coalesced_candidate_count": event.get(
                            "coalesced_candidate_count"
                        ),
                    }
                elif event_type == "transcript.text.done":
                    final_text = event.get("text", "")
                    final_arrival_ns = arrival_ns
                    metadata = event.get("generation_metadata", {})
                    summary["native_generation_metadata"] = {
                        key: metadata.get(key)
                        for key in (
                            "generated_token_count", "language", "finish_reason",
                            "verified_guess_token_count", "accepted_guess_token_count",
                            "draft_decoder_used", "verify_round_count",
                            "proposed_draft_token_count", "accepted_draft_token_count",
                            "reused_prompt_token_count", "reused_audio_token_count",
                            "decode_paths", "batch_decode_used", "native_chunked",
                        )
                    }
                elif event_type == "error":
                    terminal_error = event.get("error", {}).get("code", "error")
                    failure_kind = "server_error"
            if done_arrival_ns is None and not failure_kind:
                failure_kind = "missing_done"
    except Exception as error:  # Keep failed samples in the denominator.
        failure_kind = type(error).__name__
        terminal_error = str(error)
    finally:
        connection.close()

    status = "measured" if final_text is not None and failure_kind == "" else "failed"
    summary.update(
        candidate_metrics(candidate_events, final_text or "", final_arrival_ns, debug_text)
    )
    summary["native_snapshot_build_ms_total"] = max(
        (candidate.get("snapshot_build_ms_total", 0.0) or 0.0 for candidate in candidate_events),
        default=0.0,
    )
    safe_final_sha = sha256((final_text or "").encode("utf-8")) if final_text is not None else None
    metadata = summary.get("native_generation_metadata", {})
    record: dict[str, Any] = {
        "event": "native_final_candidate_request",
        "fixture": fixture,
        "audio_seconds": audio_seconds,
        "audio_sha256": audio_sha256,
        "language_mode": language_mode,
        "requested_language": explicit_language if language_mode == "explicit" else "auto",
        "effective_language": metadata.get("language"),
        "arm": arm,
        "requested_candidate_cadence": cadence,
        "observed_candidate_cadence": protocol_cadence or None,
        "path": "oracle_verified_guess_http" if oracle_guess is not None else "standalone_http",
        "oracle_derived_guess": oracle_guess is not None,
        "oracle_guess_sha256": sha256(oracle_guess.encode("utf-8")) if oracle_guess else None,
        "warmup": warmup,
        "round": round_number,
        "status": status,
        "failure_kind": failure_kind or None,
        "terminal_error_code": terminal_error,
        "http_status": http_status,
        "candidate_protocol_supported": protocol_supported,
        "candidate_summary_observed": candidate_summary_observed,
        "candidate_transport_status": (
            "supported" if protocol_supported is True else
            "unsupported" if protocol_supported is False else
            "unresolved" if arm != "off" else
            "not_requested"
        ),
        "request_start_monotonic_ns": request_started_ns,
        "response_headers_monotonic_ns": response_headers_ns,
        "first_candidate_monotonic_ns": (
            candidate_events[0]["arrival_monotonic_ns"] if candidate_events else None
        ),
        "authoritative_final_monotonic_ns": final_arrival_ns,
        "terminal_done_monotonic_ns": done_arrival_ns,
        "authoritative_final_ms": (
            (final_arrival_ns - request_started_ns) / 1_000_000
            if final_arrival_ns is not None
            else None
        ),
        "terminal_done_ms": (
            (done_arrival_ns - request_started_ns) / 1_000_000
            if done_arrival_ns is not None
            else None
        ),
        "final_transcript_sha256": safe_final_sha,
        "final_transcript_chars": len(final_text) if final_text is not None else None,
        "native_generated_token_count": metadata.get("generated_token_count"),
        "accepted_guess_token_count": metadata.get("accepted_guess_token_count"),
        "verified_guess_token_count": metadata.get("verified_guess_token_count"),
        "draft_decoder_used": metadata.get("draft_decoder_used"),
        "draft_verify_round_count": metadata.get("verify_round_count"),
        "draft_proposed_token_count": metadata.get("proposed_draft_token_count"),
        "draft_accepted_token_count": metadata.get("accepted_draft_token_count"),
        "reused_prompt_token_count": metadata.get("reused_prompt_token_count"),
        "reused_audio_token_count": metadata.get("reused_audio_token_count"),
        "decode_paths": metadata.get("decode_paths"),
        "batch_decode_used": metadata.get("batch_decode_used"),
        "native_chunked": metadata.get("native_chunked"),
        "finish_reason": metadata.get("finish_reason"),
        "native_sse_candidate_received_count": len(candidate_events),
        **summary,
        "go_candidate_suppressed_count": None,
        "session_candidates_processed": None,
        "dispatch_eligible_candidates": None,
        "retrospective_cleanup_compatible_candidates": None,
    }
    if debug_text:
        record["fixture_debug_final_text"] = final_text
    return record, final_text


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "sample_count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "values_ms": values,
    }


def summarize(records: list[dict[str, Any]], rounds: int, model_id: str) -> dict[str, Any]:
    measured = [
        record
        for record in records
        if record.get("event") == "native_final_candidate_request"
        and not record.get("warmup", False)
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in measured:
        groups.setdefault((record["path"], record["fixture"], record["language_mode"]), []).append(record)
    result_groups: list[dict[str, Any]] = []
    for (path, fixture, language_mode), rows in sorted(groups.items()):
        by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
        off_by_round = {row["round"]: row for row in by_arm["off"]}
        arm_results = {}
        for arm, arm_rows in by_arm.items():
            successful = [row for row in arm_rows if row["status"] == "measured"]
            latencies = [row["authoritative_final_ms"] for row in successful]
            latency_values = [value for value in latencies if value is not None]
            supported = [row for row in arm_rows if row["candidate_protocol_supported"] is True]
            completed_without_support = [
                row
                for row in successful
                if row["candidate_protocol_supported"] is not True
            ]
            candidate_signal_rows = supported
            paired = []
            paired_round_count = 0
            transcript_parity = {"exact_matches": 0, "comparable_pairs": 0}
            metadata_parity = {"exact_matches": 0, "comparable_pairs": 0}
            if arm != "off":
                for row in arm_rows:
                    baseline = off_by_round.get(row["round"])
                    if not baseline:
                        continue
                    paired_round_count += 1
                    if row["status"] != "measured" or baseline["status"] != "measured":
                        continue
                    if row["authoritative_final_ms"] is not None and baseline["authoritative_final_ms"] is not None:
                        paired.append(row["authoritative_final_ms"] - baseline["authoritative_final_ms"])
                    if row["final_transcript_sha256"] is not None and baseline["final_transcript_sha256"] is not None:
                        transcript_parity["comparable_pairs"] += 1
                        transcript_parity["exact_matches"] += int(
                            row["final_transcript_sha256"] == baseline["final_transcript_sha256"]
                        )
                    metric_keys = (
                        "native_generated_token_count", "effective_language", "finish_reason",
                        "verified_guess_token_count", "accepted_guess_token_count",
                        "draft_decoder_used", "draft_verify_round_count",
                        "draft_proposed_token_count", "draft_accepted_token_count",
                        "reused_prompt_token_count", "reused_audio_token_count",
                        "decode_paths", "batch_decode_used", "native_chunked",
                    )
                    if all(row.get(key) is not None and baseline.get(key) is not None for key in metric_keys):
                        metadata_parity["comparable_pairs"] += 1
                        metadata_parity["exact_matches"] += int(
                            all(row.get(key) == baseline.get(key) for key in metric_keys)
                        )
            candidate_summary_rows = [
                row for row in candidate_signal_rows
                if isinstance(row.get("native_published_candidate_count"), int)
            ]
            coalesced_summary_rows = [
                row for row in candidate_signal_rows
                if row.get("native_coalesced_candidate_count") is not None
            ]
            candidate_count = (
                sum(row.get("native_published_candidate_count") or 0 for row in candidate_summary_rows)
                if candidate_summary_rows
                else None
            )
            candidate_arrivals = [
                candidate
                for row in candidate_signal_rows
                for candidate in row.get("candidates", [])
            ]
            exact_candidates = [candidate for candidate in candidate_arrivals if candidate["exact_transcript_match"]]
            exact_leads = [
                row["earliest_exact_match_lead_ms"]
                for row in candidate_signal_rows
                if row.get("earliest_exact_match_lead_ms") is not None
            ]
            closest_leads = [
                row["closest_candidate_lead_ms"]
                for row in arm_rows
                if row.get("closest_candidate_lead_ms") is not None
            ]
            prefix_leads = {
                name: [
                    row["lead_at_prefix_completeness"][name]
                    for row in arm_rows
                    if row.get("lead_at_prefix_completeness", {}).get(name) is not None
                ]
                for name in ("90_percent_ms", "95_percent_ms", "100_percent_ms")
            }
            length_ratios = [
                candidate["length_ratio"]
                for candidate in candidate_arrivals
                if candidate["length_ratio"] is not None
            ]
            arm_results[arm] = {
                "measured_session_count": len(arm_rows),
                "successful_session_count": len(successful),
                "failed_session_count": len(arm_rows) - len(successful),
                "candidate_protocol_supported_sessions": len(supported),
                "candidate_protocol_unsupported_or_unknown_sessions": (
                    len(arm_rows) - len(supported) if arm != "off" else 0
                ),
                "candidate_summary_observed_sessions": len(candidate_summary_rows),
                "candidate_summary_missing_sessions": len(supported) - len(candidate_summary_rows),
                "coalescing_observed_sessions": len(coalesced_summary_rows),
                "candidate_published_session_count": (
                    sum(
                        1
                        for row in candidate_summary_rows
                        if (row.get("native_published_candidate_count") or 0) > 0
                    )
                    if candidate_summary_rows
                    else None
                ),
                "native_published_candidate_count": candidate_count,
                "native_sse_candidate_received_count": (
                    sum(row["native_sse_candidate_received_count"] for row in candidate_signal_rows)
                    if candidate_signal_rows
                    else None
                ),
                "native_coalesced_candidate_count": (
                    sum(row["native_coalesced_candidate_count"] for row in coalesced_summary_rows)
                    if coalesced_summary_rows
                    else None
                ),
                "authoritative_final_latency": distribution(latency_values),
                "paired_latency_arm_minus_off": distribution(paired),
                "transcript_parity_vs_off": transcript_parity if arm != "off" else None,
                "generation_metadata_parity_vs_off": metadata_parity if arm != "off" else None,
                "oracle_exact_candidate_count": len(exact_candidates) if candidate_signal_rows else None,
                "oracle_exact_candidate_lead": distribution(exact_leads) if candidate_signal_rows else None,
                "candidate_length_ratio": distribution(length_ratios) if candidate_signal_rows else None,
                "closest_candidate_lead": distribution(closest_leads) if candidate_signal_rows else None,
                "lead_at_prefix_completeness": (
                    {name: distribution(values) for name, values in prefix_leads.items()}
                    if candidate_signal_rows
                    else None
                ),
                "candidate_coverage_denominator": len(arm_rows),
                "candidate_supported_denominator": len(candidate_signal_rows),
                "paired_round_count": paired_round_count if arm != "off" else 0,
                "paired_latency_comparable_count": len(paired),
                "candidate_observation_status": (
                    "not_requested" if arm == "off" else
                    "no_successful_response" if not successful else
                    "unsupported_or_unresolved_transport" if completed_without_support else
                    "supported_transport_with_unresolved_candidate_summary"
                    if len(candidate_summary_rows) != len(supported)
                    else
                    "supported_transport_no_publishable_candidate"
                    if candidate_count == 0
                    else "supported_transport_with_candidates"
                ),
            }
        result_groups.append(
            {
                "path": path,
                "fixture": fixture,
                "language_mode": language_mode,
                "measured_rounds_requested": rounds,
                "arms": arm_results,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "qwen3_asr_authoritative_final_candidate_feasibility",
        "model_id": model_id,
        "model_weights_downloaded": False,
        "native_realtime_prefill_reuse": {
            "status": "unvalidated",
            "reason": "This runner measures standalone HTTP Final only; defer reuse to the full Voxt benchmark.",
        },
        "oracle_guess_interpretation": "Oracle-derived guessed_text runs are diagnostic only, not a production preview estimate.",
        "groups": result_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("Voxt/build/native-final-candidate-screen"))
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--draft-model-dir", type=Path)
    parser.add_argument("--draft-bits", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--oracle-guess-diagnostic", action="store_true")
    parser.add_argument("--fixture-debug-text", action="store_true")
    arguments = parser.parse_args()
    runtime_bin = arguments.runtime_bin.expanduser().resolve()
    model_dir = arguments.model_dir.expanduser().resolve()
    if not runtime_bin.is_file() or not model_dir.is_dir():
        parser.error("--runtime-bin must be a local file and --model-dir a local directory")
    if arguments.rounds < 1 or arguments.warmups < 0 or arguments.timeout_seconds <= 0:
        parser.error("rounds, warmups, and timeout must be positive (warmups may be zero)")
    if arguments.draft_model_dir is not None and not arguments.draft_model_dir.is_dir():
        parser.error("--draft-model-dir must already exist; weights are never downloaded")
    if arguments.draft_bits not in (0, 4, 6, 8):
        parser.error("--draft-bits must be 0, 4, 6, or 8")
    if arguments.draft_bits and arguments.draft_model_dir is None:
        parser.error("--draft-bits requires --draft-model-dir")

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "requests.jsonl"
    summary_path = output_dir / "summary.json"
    git_root = Path(git_value("rev-parse", "--show-toplevel"))
    if str(git_root) == "unknown":
        git_root = Path.cwd()
    runtime_hash = sha256(runtime_bin.read_bytes())
    config_path = model_dir / "config.json"
    provenance_path = runtime_bin.parent.parent / "build-provenance.json"
    provenance = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
    model_id = arguments.model_id or model_dir.name
    draft_flags: list[str] = []
    if arguments.draft_model_dir is not None:
        draft_flags = ["--draft-model-path", str(arguments.draft_model_dir.resolve())]
        if arguments.draft_bits:
            draft_flags += ["--draft-bits", str(arguments.draft_bits)]
    server: SupervisedServer | None = None
    records: list[dict[str, Any]] = []
    oracle_guesses: dict[tuple[str, str], str] = {}
    server_shutdown_error: str | None = None
    run_error: str | None = None

    def interleaved_paths(
        path: str, fixture_name: str, fixture_path: Path, explicit_language: str,
        language_mode: str, guess: str | None,
    ) -> None:
        audio = fixture_path.read_bytes()
        with wave.open(io.BytesIO(audio), "rb") as reader:
            audio_seconds = reader.getnframes() / reader.getframerate()
        audio_hash = sha256(audio)
        arm_warmups: dict[str, str | None] = {}
        for arm in ARMS:
            for _ in range(arguments.warmups):
                record, text = transcribe(
                    server, audio, fixture_name, audio_hash, language_mode,
                    explicit_language, arm,
                    "bounded" if arm == "shadow" else "dense" if arm == "shadow_dense" else None,
                    -1, True, arguments.timeout_seconds, arguments.max_new_tokens,
                    audio_seconds,
                    guess, arguments.fixture_debug_text,
                )
                record["path"] = path
                records.append(record)
                if arm == "off" and guess is None and text is not None:
                    arm_warmups[arm] = text
        if path == "standalone_http" and guess is None and arm_warmups.get("off"):
            oracle_guesses[(fixture_name, language_mode)] = arm_warmups["off"]
        for round_number in range(arguments.rounds):
            order = ARMS[round_number % len(ARMS) :] + ARMS[: round_number % len(ARMS)]
            for arm in order:
                record, _ = transcribe(
                    server, audio, fixture_name, audio_hash, language_mode,
                    explicit_language, arm,
                    "bounded" if arm == "shadow" else "dense" if arm == "shadow_dense" else None,
                    round_number, False, arguments.timeout_seconds,
                    arguments.max_new_tokens, audio_seconds, guess, arguments.fixture_debug_text,
                )
                record["path"] = path
                records.append(record)

    try:
        server = SupervisedServer(runtime_bin, "qwen3_asr", model_dir, tuple(draft_flags))
        for fixture_name, relative_path, explicit_language in FIXTURES:
            fixture_path = git_root / relative_path
            if not fixture_path.is_file():
                raise FileNotFoundError(f"local fixture is missing: {fixture_path}")
            for language_mode in ("automatic", "explicit"):
                interleaved_paths(
                    "standalone_http", fixture_name, fixture_path,
                    explicit_language, language_mode, None,
                )
                if arguments.oracle_guess_diagnostic:
                    guess = oracle_guesses.get((fixture_name, language_mode))
                    if guess:
                        interleaved_paths(
                            "oracle_verified_guess_http", fixture_name, fixture_path,
                            explicit_language, language_mode, guess,
                        )
                    else:
                        records.append(
                            {
                                "event": "native_final_candidate_path_skipped",
                                "path": "oracle_verified_guess_http",
                                "fixture": fixture_name,
                                "language_mode": language_mode,
                                "status": "failed",
                                "reason": "the OFF warmup did not produce an authoritative transcript",
                            }
                        )
    except Exception as error:
        run_error = f"{type(error).__name__}: {error}"
        records.append(
            {
                "event": "native_final_candidate_run_error",
                "status": "failed",
                "error": run_error,
            }
        )
    finally:
        try:
            if server is not None:
                server.stop()
        except Exception as error:
            server_shutdown_error = f"{type(error).__name__}: {error}"

    safe_records = []
    for record in records:
        safe = dict(record)
        if not arguments.fixture_debug_text:
            safe.pop("fixture_debug_final_text", None)
            for candidate in safe.get("candidates", []):
                candidate.pop("fixture_debug_text", None)
        safe_records.append(safe)
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in safe_records))
    summary = summarize(records, arguments.rounds, model_id)
    summary["runtime_provenance"] = {
        "runtime_binary": str(runtime_bin),
        "runtime_binary_sha256": runtime_hash,
        "build_provenance_path": str(provenance_path) if provenance else None,
        "build_provenance": provenance,
        "model_directory": str(model_dir),
        "model_config_sha256": sha256(config_path.read_bytes()) if config_path.is_file() else None,
        "experiment_commit_sha": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    summary["request_configuration"] = {
        **REQUEST_FIELDS,
        "language_modes": ["automatic", "explicit"],
        "arms": list(ARMS),
        "rounds": arguments.rounds,
        "warmups": arguments.warmups,
        "max_new_tokens": arguments.max_new_tokens,
        "draft_model_directory": str(arguments.draft_model_dir.resolve()) if arguments.draft_model_dir else None,
        "draft_bits": arguments.draft_bits,
        "oracle_guess_diagnostic": arguments.oracle_guess_diagnostic,
        "candidate_flag_by_arm": {
            "off": False,
            "shadow": "true with bounded cadence",
            "shadow_dense": "true with dense cadence",
        },
    }
    summary["artifacts"] = {
        "request_jsonl": str(jsonl_path),
        "summary_json": str(summary_path),
    }
    successful_shadow_rows = [
        row for row in records
        if row.get("arm") in ("shadow", "shadow_dense")
        and not row.get("warmup", False)
        and row.get("status") == "measured"
    ]
    protocol_observations = {
        row.get("candidate_protocol_supported") for row in successful_shadow_rows
    }
    summary["runtime_candidate_protocol_supported"] = (
        True
        if protocol_observations == {True}
        else False
        if protocol_observations == {False}
        else None
    )
    summary["runtime_candidate_protocol_status"] = (
        "supported"
        if protocol_observations == {True}
        else "unsupported"
        if protocol_observations == {False}
        else "unresolved_or_mixed"
        if protocol_observations
        else "not_observed"
    )
    summary["server_shutdown_error"] = server_shutdown_error
    summary["run_error"] = run_error
    summary["run_status"] = (
        "failed"
        if run_error or server_shutdown_error or any(
            row.get("status") == "failed" for row in records
        )
        else "complete"
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"request_jsonl": str(jsonl_path), "summary_json": str(summary_path)}))
    return 1 if summary["run_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

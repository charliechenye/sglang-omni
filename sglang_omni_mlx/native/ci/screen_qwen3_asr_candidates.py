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
ALWAYS_PRESENT_METADATA_FIELDS = {
    "generated_token_count",
    "language",
    "finish_reason",
    "verified_guess_token_count",
    "accepted_guess_token_count",
    "decode_paths",
    "draft_decoder_used",
    "batch_decode_used",
    "native_chunked",
}
OMITTED_ZERO_METADATA_FIELDS = {
    "verify_round_count",
    "proposed_draft_token_count",
    "accepted_draft_token_count",
    "reused_prompt_token_count",
    "reused_audio_token_count",
}
METADATA_PARITY_FIELDS = {
    "generated_token_count": "native_generated_token_count",
    "language": "effective_language",
    "finish_reason": "finish_reason",
    "verified_guess_token_count": "verified_guess_token_count",
    "accepted_guess_token_count": "accepted_guess_token_count",
    "draft_decoder_used": "draft_decoder_used",
    "verify_round_count": "draft_verify_round_count",
    "proposed_draft_token_count": "draft_proposed_token_count",
    "accepted_draft_token_count": "draft_accepted_token_count",
    "reused_prompt_token_count": "reused_prompt_token_count",
    "reused_audio_token_count": "reused_audio_token_count",
    "decode_paths": "decode_paths",
    "batch_decode_used": "batch_decode_used",
    "native_chunked": "native_chunked",
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


def validate_runtime_provenance(runtime_bin: Path, git_root: Path) -> dict[str, Any]:
    provenance_path = runtime_bin.parent.parent / "build-provenance.json"
    if not provenance_path.is_file():
        raise ValueError(f"missing runtime provenance: {provenance_path}")
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid runtime provenance at {provenance_path}: {error}"
        ) from error
    if not isinstance(provenance, dict):
        raise ValueError(
            f"invalid runtime provenance at {provenance_path}: expected a JSON object"
        )

    experiment_commit = git_value("-C", str(git_root), "rev-parse", "HEAD")
    native_source_path = provenance.get("native_source_path")
    runtime_binary_path = provenance.get("runtime_binary_path")
    expected_source = (git_root / "sglang_omni_mlx" / "native").resolve()
    current_source_commit = git_value("-C", str(expected_source), "rev-parse", "HEAD")
    current_source_status = git_value(
        "-C", str(expected_source), "status", "--porcelain", "--untracked-files=all"
    )
    checks = {
        "experiment commit": provenance.get("experiment_commit") == experiment_commit,
        "native source commit": (
            provenance.get("native_source_commit") == experiment_commit
        ),
        "current native source commit": current_source_commit == experiment_commit,
        "clean native source": provenance.get("native_worktree_clean") is True,
        "current native source clean": current_source_status == "",
        "native source path": isinstance(native_source_path, str)
        and Path(native_source_path).expanduser().resolve() == expected_source,
        "runtime binary path": isinstance(runtime_binary_path, str)
        and Path(runtime_binary_path).expanduser().resolve() == runtime_bin.resolve(),
        "runtime binary SHA-256": (
            provenance.get("runtime_binary_sha256") == sha256(runtime_bin.read_bytes())
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError("runtime provenance check failed: " + ", ".join(failed_checks))
    return {"path": str(provenance_path), "record": provenance, "checks": checks}


def normalize_generation_metadata(
    metadata: dict[str, Any], metadata_observed: bool = True
) -> dict[str, dict[str, Any]]:
    normalized = {}
    for field in METADATA_PARITY_FIELDS:
        if not metadata_observed:
            normalized[field] = {"state": "unknown", "value": None}
        elif field in metadata and (metadata[field] is not None or field == "language"):
            normalized[field] = {"state": "present", "value": metadata[field]}
        elif field in OMITTED_ZERO_METADATA_FIELDS:
            normalized[field] = {"state": "known_zero", "value": 0}
        elif field in ALWAYS_PRESENT_METADATA_FIELDS:
            normalized[field] = {"state": "unknown", "value": None}
        else:
            normalized[field] = {"state": "unknown", "value": None}
    return normalized


def metadata_parity(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    counts = {
        field_name: {
            "comparable_pair_count": 0,
            "exact_match_count": 0,
            "mismatch_count": 0,
            "unresolved_pair_count": 0,
            "not_applicable_pair_count": 0,
        }
        for field_name in METADATA_PARITY_FIELDS.values()
    }
    baseline_states = baseline.get("native_generation_metadata_states", {})
    candidate_states = candidate.get("native_generation_metadata_states", {})
    pair_mismatches = 0
    pair_unresolved = 0
    pair_not_applicable = 0
    comparable_pairs = 0
    for metadata_field, report_field in METADATA_PARITY_FIELDS.items():
        left = baseline_states.get(metadata_field, {"state": "unknown", "value": None})
        right = candidate_states.get(
            metadata_field, {"state": "unknown", "value": None}
        )
        if metadata_field in {
            "verify_round_count",
            "proposed_draft_token_count",
            "accepted_draft_token_count",
        } and (
            baseline.get("draft_decoder_used") is False
            and candidate.get("draft_decoder_used") is False
        ):
            counts[report_field]["not_applicable_pair_count"] += 1
            pair_not_applicable += 1
        elif (
            metadata_field in {"reused_prompt_token_count", "reused_audio_token_count"}
            and isinstance(baseline.get("decode_paths"), list)
            and isinstance(candidate.get("decode_paths"), list)
            and "final_prefill_reuse" not in baseline["decode_paths"]
            and "final_prefill_reuse" not in candidate["decode_paths"]
        ):
            counts[report_field]["not_applicable_pair_count"] += 1
            pair_not_applicable += 1
        elif left["state"] == "unknown" or right["state"] == "unknown":
            counts[report_field]["unresolved_pair_count"] += 1
            pair_unresolved += 1
        else:
            left_value = left["value"]
            right_value = right["value"]
            if (
                metadata_field == "decode_paths"
                and isinstance(left_value, list)
                and isinstance(right_value, list)
            ):
                left_value = sorted(left_value)
                right_value = sorted(right_value)
            counts[report_field]["comparable_pair_count"] += 1
            comparable_pairs += 1
            if left_value == right_value:
                counts[report_field]["exact_match_count"] += 1
            else:
                counts[report_field]["mismatch_count"] += 1
                pair_mismatches += 1
    return {
        "fields": counts,
        "overall": {
            "comparable_pair_count": int(comparable_pairs > 0),
            "exact_match_count": int(
                comparable_pairs > 0 and pair_mismatches == 0 and pair_unresolved == 0
            ),
            "mismatch_count": int(pair_mismatches > 0),
            "unresolved_pair_count": int(pair_unresolved > 0),
            "not_applicable_pair_count": int(
                comparable_pairs == 0 and pair_not_applicable > 0
            ),
        },
    }


def aggregate_metadata_parity(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {
        "fields": {
            field_name: {
                "comparable_pair_count": 0,
                "exact_match_count": 0,
                "mismatch_count": 0,
                "unresolved_pair_count": 0,
                "not_applicable_pair_count": 0,
            }
            for field_name in METADATA_PARITY_FIELDS.values()
        },
        "overall": {
            "comparable_pair_count": 0,
            "exact_match_count": 0,
            "mismatch_count": 0,
            "unresolved_pair_count": 0,
            "not_applicable_pair_count": 0,
        },
    }
    for pair in pairs:
        for field_name, field_counts in pair["fields"].items():
            for count_name, count in field_counts.items():
                aggregate["fields"][field_name][count_name] += count
        for count_name, count in pair["overall"].items():
            aggregate["overall"][count_name] += count
    return aggregate


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
        completion_ms = candidate.get("authoritative_completion_elapsed_ms")
        native_elapsed_ms = candidate.get("native_elapsed_ms")
        native_lead_ms = (
            completion_ms - native_elapsed_ms
            if isinstance(completion_ms, (int, float))
            and isinstance(native_elapsed_ms, (int, float))
            else None
        )
        if native_lead_ms is None:
            timing_classification = "unresolved_timing"
        elif native_lead_ms < 0:
            timing_classification = "generated_too_late"
        elif candidate.get("delivery_during_terminal_drain"):
            timing_classification = (
                "generated_before_completion_delivered_during_terminal_drain"
            )
        else:
            timing_classification = "generated_before_completion_delivered_while_active"
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
                "native_candidate_to_authoritative_completion_ms": native_lead_ms,
                "timing_classification": timing_classification,
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
    candidate_summary_arrival_ns: int | None = None
    summary: dict[str, Any] = {}
    candidate_events: list[dict[str, Any]] = []
    final_text: str | None = None
    final_arrival_ns: int | None = None
    done_arrival_ns: int | None = None
    terminal_error: str | None = None
    http_status: int | None = None
    failure_kind = ""
    authoritative_completion_elapsed_ms: float | None = None
    event_types: list[str] = []
    final_event_count = 0
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
                event_types.append(event_type)
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
                            "delivery_during_terminal_drain": event.get(
                                "delivery_during_terminal_drain"
                            ),
                            "authoritative_completion_elapsed_ms": (
                                authoritative_completion_elapsed_ms
                            ),
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
                    candidate_summary_arrival_ns = arrival_ns
                    summary.update(
                        {
                            "native_published_candidate_count": event.get(
                                "published_candidate_count"
                            ),
                            "native_coalesced_candidate_count": event.get(
                                "coalesced_candidate_count"
                            ),
                        }
                    )
                elif event_type == "transcript.text.done":
                    final_event_count += 1
                    final_text = event.get("text", "")
                    final_arrival_ns = arrival_ns
                    metadata = event.get("generation_metadata")
                    summary["native_generation_metadata_observed"] = isinstance(
                        metadata, dict
                    )
                    if not isinstance(metadata, dict):
                        metadata = {}
                    authoritative_completion_elapsed_ms = metadata.get(
                        "authoritative_completion_elapsed_ms"
                    )
                    summary["native_generation_metadata"] = {
                        key: metadata[key]
                        for key in (
                            "generated_token_count", "language", "finish_reason",
                            "verified_guess_token_count", "accepted_guess_token_count",
                            "draft_decoder_used", "verify_round_count",
                            "proposed_draft_token_count", "accepted_draft_token_count",
                            "reused_prompt_token_count", "reused_audio_token_count",
                            "decode_paths", "batch_decode_used", "native_chunked",
                            "authoritative_completion_elapsed_ms",
                        )
                        if key in metadata
                    }
                    summary["native_metadata_present_fields"] = sorted(metadata)
                    for candidate in candidate_events:
                        candidate["authoritative_completion_elapsed_ms"] = metadata.get(
                            "authoritative_completion_elapsed_ms"
                        )
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

    if final_text is None and not failure_kind:
        failure_kind = "missing_final"
    elif final_text == "" and not failure_kind:
        failure_kind = "empty_final_transcript"
    elif http_status != 200 and not failure_kind:
        failure_kind = "http_status_unresolved"
    elif final_event_count != 1 and not failure_kind:
        failure_kind = "invalid_final_event_count"
    else:
        pass
    status = (
        "measured"
        if http_status == 200
        and final_text
        and done_arrival_ns is not None
        and final_event_count == 1
        and failure_kind == ""
        else "failed"
    )
    summary.update(
        candidate_metrics(candidate_events, final_text or "", final_arrival_ns, debug_text)
    )
    summary["native_snapshot_build_ms_total"] = max(
        (candidate.get("snapshot_build_ms_total", 0.0) or 0.0 for candidate in candidate_events),
        default=0.0,
    )
    safe_final_sha = sha256((final_text or "").encode("utf-8")) if final_text is not None else None
    metadata = summary.get("native_generation_metadata", {})
    metadata_states = normalize_generation_metadata(
        metadata, summary.get("native_generation_metadata_observed") is True
    )
    validation_errors = []
    expected_cadence = cadence if arm != "off" else None
    if arm != "off":
        if protocol_supported is not True:
            validation_errors.append("candidate_protocol_unsupported_or_unresolved")
        if protocol_cadence != expected_cadence:
            validation_errors.append("candidate_cadence_mismatch_or_unresolved")
        if not candidate_summary_observed:
            validation_errors.append("candidate_summary_missing")
        elif not all(
            isinstance(summary.get(field), int)
            and not isinstance(summary.get(field), bool)
            and summary[field] >= 0
            for field in (
                "native_published_candidate_count",
                "native_coalesced_candidate_count",
            )
        ):
            validation_errors.append("candidate_summary_invalid")
        if not isinstance(
            metadata.get("authoritative_completion_elapsed_ms"), (int, float)
        ):
            validation_errors.append("native_completion_timing_unavailable")
    candidate_valid = arm == "off" or not validation_errors
    if arm != "off" and status != "measured":
        candidate_valid = False
    if oracle_guess is None:
        guess_status = "not_requested"
    elif metadata.get("verified_guess_token_count") is None:
        guess_status = "requested_but_actual_verification_unknown"
    elif metadata.get("verified_guess_token_count", 0) == 0:
        guess_status = "requested_but_not_verified"
    elif metadata.get("accepted_guess_token_count", 0) == 0:
        guess_status = "verified_but_rejected"
    elif metadata.get("accepted_guess_token_count") < metadata.get(
        "verified_guess_token_count"
    ):
        guess_status = "partially_accepted"
    else:
        guess_status = "accepted"
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
        "guess_source": "oracle_derived" if oracle_guess is not None else "none",
        "real_preview_derived_guess": False,
        "real_preview_guess_status": "unvalidated_by_standalone_http",
        "requested_guess_character_count": (
            len(oracle_guess) if oracle_guess is not None else 0
        ),
        "guess_verification_status": guess_status,
        "oracle_guess_sha256": sha256(oracle_guess.encode("utf-8")) if oracle_guess else None,
        "warmup": warmup,
        "round": round_number,
        "status": status,
        "failure_kind": failure_kind or None,
        "terminal_error_code": terminal_error,
        "http_status": http_status,
        "candidate_protocol_supported": protocol_supported,
        "candidate_summary_observed": candidate_summary_observed,
        "candidate_summary_arrival_monotonic_ns": candidate_summary_arrival_ns,
        "candidate_summary_after_final_ms": (
            (candidate_summary_arrival_ns - final_arrival_ns) / 1_000_000
            if candidate_summary_arrival_ns is not None and final_arrival_ns is not None
            else None
        ),
        "candidate_valid_for_screening": candidate_valid,
        "candidate_validation_errors": validation_errors,
        "native_generation_metadata_states": metadata_states,
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
        "final_to_done_ms": (
            (done_arrival_ns - final_arrival_ns) / 1_000_000
            if done_arrival_ns is not None and final_arrival_ns is not None
            else None
        ),
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
        "authoritative_completion_elapsed_ms": metadata.get(
            "authoritative_completion_elapsed_ms"
        ),
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


def summarize(
    records: list[dict[str, Any]],
    rounds: int,
    model_id: str,
    oracle_guess_diagnostic: bool = False,
) -> dict[str, Any]:
    measured = [
        record
        for record in records
        if record.get("event") == "native_final_candidate_request"
        and not record.get("warmup", False)
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in measured:
        groups.setdefault(
            (record["path"], record["fixture"], record["language_mode"]), []
        ).append(record)
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
            supported_successful = [
                row for row in supported if row["status"] == "measured"
            ]
            completed_without_support = [
                row
                for row in successful
                if row["candidate_protocol_supported"] is not True
            ]
            candidate_signal_rows = [
                row for row in supported
                if row["status"] == "measured"
                and row.get("candidate_valid_for_screening") is True
                and isinstance(
                    row.get("authoritative_completion_elapsed_ms"), (int, float)
                )
            ]
            paired = []
            paired_round_count = 0
            missing_pair_count = 0
            unresolved_pair_count = 0
            transcript_parity = {
                "exact_match_count": 0,
                "comparable_pair_count": 0,
                "unresolved_pair_count": 0,
            }
            metadata_parity_pairs = []
            if arm != "off":
                rows_by_round = {row["round"]: row for row in arm_rows}
                for round_number in range(rounds):
                    row = rows_by_round.get(round_number)
                    baseline = off_by_round.get(round_number)
                    if row is None or baseline is None:
                        missing_pair_count += 1
                        continue
                    paired_round_count += 1
                    pair_valid = (
                        row["status"] == "measured"
                        and baseline["status"] == "measured"
                        and row.get("final_transcript_chars", 0) > 0
                        and baseline.get("final_transcript_chars", 0) > 0
                        and row.get("audio_sha256") == baseline.get("audio_sha256")
                    )
                    if not pair_valid:
                        unresolved_pair_count += 1
                        transcript_parity["unresolved_pair_count"] += 1
                        continue
                    if row["authoritative_final_ms"] is not None and baseline["authoritative_final_ms"] is not None:
                        paired.append(row["authoritative_final_ms"] - baseline["authoritative_final_ms"])
                    if row["final_transcript_sha256"] and baseline[
                        "final_transcript_sha256"
                    ]:
                        transcript_parity["comparable_pair_count"] += 1
                        transcript_parity["exact_match_count"] += int(
                            row["final_transcript_sha256"] == baseline["final_transcript_sha256"]
                        )
                    metadata_parity_pairs.append(metadata_parity(baseline, row))
            metadata_parity_result = aggregate_metadata_parity(metadata_parity_pairs)
            candidate_summary_rows = [
                row
                for row in supported_successful
                if row.get("candidate_summary_observed") is True
            ]
            candidate_summary_valid_rows = [
                row
                for row in candidate_signal_rows
                if isinstance(row.get("native_published_candidate_count"), int)
            ]
            coalesced_summary_rows = [
                row for row in candidate_signal_rows
                if row.get("native_coalesced_candidate_count") is not None
            ]
            candidate_count = (
                sum(
                    row.get("native_published_candidate_count") or 0
                    for row in candidate_summary_valid_rows
                )
                if candidate_summary_valid_rows
                else None
            )
            candidate_arrivals = [
                candidate
                for row in candidate_signal_rows
                for candidate in row.get("candidates", [])
            ]
            exact_candidates = [
                candidate
                for candidate in candidate_arrivals
                if candidate["exact_transcript_match"]
            ]
            exact_leads = [
                row["earliest_exact_match_lead_ms"]
                for row in candidate_signal_rows
                if row.get("earliest_exact_match_lead_ms") is not None
            ]
            closest_leads = [
                row["closest_candidate_lead_ms"]
                for row in candidate_signal_rows
                if row.get("closest_candidate_lead_ms") is not None
            ]
            prefix_leads = {
                name: [
                    row["lead_at_prefix_completeness"][name]
                    for row in candidate_signal_rows
                    if row.get("lead_at_prefix_completeness", {}).get(name) is not None
                ]
                for name in ("90_percent_ms", "95_percent_ms", "100_percent_ms")
            }
            length_ratios = [
                candidate["length_ratio"]
                for candidate in candidate_arrivals
                if candidate["length_ratio"] is not None
            ]
            native_completion_leads = [
                candidate["native_candidate_to_authoritative_completion_ms"]
                for candidate in candidate_arrivals
                if candidate[
                    "native_candidate_to_authoritative_completion_ms"
                ]
                is not None
            ]
            client_candidate_to_final = [
                candidate["client_lead_ms"]
                for candidate in candidate_arrivals
                if candidate["client_lead_ms"] is not None
            ]
            summary_write_after_completion = [
                row["candidate_summary_after_final_ms"]
                for row in candidate_signal_rows
                if row.get("candidate_summary_after_final_ms") is not None
            ]
            final_to_done = [
                row["final_to_done_ms"]
                for row in candidate_signal_rows
                if row.get("final_to_done_ms") is not None
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
                "candidate_summary_missing_sessions": len(supported_successful)
                - len(candidate_summary_rows),
                "coalescing_observed_sessions": len(coalesced_summary_rows),
                "candidate_published_session_count": (
                    sum(
                        1
                        for row in candidate_summary_rows
                        if (row.get("native_published_candidate_count") or 0) > 0
                    )
                    if candidate_summary_valid_rows
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
                "generation_metadata_parity_vs_off": (
                    metadata_parity_result if arm != "off" else None
                ),
                "oracle_exact_candidate_count": len(exact_candidates) if candidate_signal_rows else None,
                "oracle_exact_candidate_lead": distribution(exact_leads) if candidate_signal_rows else None,
                "candidate_length_ratio": distribution(length_ratios) if candidate_signal_rows else None,
                "native_candidate_to_authoritative_completion": (
                    distribution(native_completion_leads)
                    if candidate_signal_rows
                    else None
                ),
                "client_candidate_to_final_event": (
                    distribution(client_candidate_to_final)
                    if candidate_signal_rows
                    else None
                ),
                "authoritative_final_to_done": (
                    distribution(final_to_done) if candidate_signal_rows else None
                ),
                "candidate_summary_write_after_final": (
                    distribution(summary_write_after_completion)
                    if candidate_signal_rows
                    else None
                ),
                "closest_candidate_lead": distribution(closest_leads) if candidate_signal_rows else None,
                "lead_at_prefix_completeness": (
                    {name: distribution(values) for name, values in prefix_leads.items()}
                    if candidate_signal_rows
                    else None
                ),
                "candidate_coverage_denominator": len(arm_rows),
                "candidate_supported_denominator": len(candidate_signal_rows),
                "candidate_timing_unresolved_session_count": sum(
                    1 for row in supported_successful
                    if not isinstance(
                        row.get("authoritative_completion_elapsed_ms"), (int, float)
                    )
                ),
                "candidate_no_published_session_count": sum(
                    1 for row in candidate_summary_valid_rows
                    if row.get("native_published_candidate_count") == 0
                ),
                "candidate_published_but_coalesced_session_count": sum(
                    1 for row in candidate_signal_rows
                    if (row.get("native_coalesced_candidate_count") or 0) > 0
                    and (row.get("native_published_candidate_count") or 0)
                    > (row.get("native_sse_candidate_received_count") or 0)
                ),
                "candidate_generated_before_completion_count": sum(
                    1 for candidate in candidate_arrivals
                    if candidate["timing_classification"] in (
                        "generated_before_completion_delivered_while_active",
                        "generated_before_completion_delivered_during_terminal_drain",
                    )
                ),
                "candidate_terminal_drain_delivery_count": sum(
                    1 for candidate in candidate_arrivals
                    if candidate["timing_classification"]
                    == "generated_before_completion_delivered_during_terminal_drain"
                ),
                "candidate_too_late_count": sum(
                    1 for candidate in candidate_arrivals
                    if candidate["timing_classification"] == "generated_too_late"
                ),
                "candidate_timing_unresolved_count": sum(
                    1 for candidate in candidate_arrivals
                    if candidate["timing_classification"] == "unresolved_timing"
                ),
                "candidate_exact_transcript_match_count": sum(
                    1
                    for candidate in candidate_arrivals
                    if candidate["exact_transcript_match"]
                ),
                "candidate_prefix_completeness_counts": {
                    threshold: sum(
                        1 for candidate in candidate_arrivals
                        if candidate["common_prefix_completeness"] is not None
                        and candidate["common_prefix_completeness"] >= fraction
                    )
                    for threshold, fraction in (
                        ("90_percent", 0.90),
                        ("95_percent", 0.95),
                        ("100_percent", 1.0),
                    )
                },
                "paired_round_count": paired_round_count if arm != "off" else 0,
                "paired_missing_round_count": missing_pair_count if arm != "off" else 0,
                "paired_unresolved_round_count": (
                    unresolved_pair_count if arm != "off" else 0
                ),
                "paired_latency_comparable_count": len(paired),
                "candidate_observation_status": (
                    "not_requested" if arm == "off" else
                    "no_successful_response" if not successful else
                    "unsupported_or_unresolved_transport" if completed_without_support else
                    "supported_transport_with_unresolved_candidate_summary"
                    if len(candidate_summary_rows) != len(supported_successful)
                    else
                    "supported_transport_no_publishable_candidate"
                    if candidate_signal_rows and candidate_count == 0
                    else "supported_transport_with_unresolved_timing"
                    if supported_successful and not candidate_signal_rows
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
        "automatic_language_oracle_guess_status": (
            "unsupported" if oracle_guess_diagnostic else "not_requested"
        ),
        "automatic_language_oracle_guess_reason": (
            "the standalone guessed_text field cannot represent automatic-language "
            "header and ASR marker tokens"
            if oracle_guess_diagnostic
            else None
        ),
        "real_preview_derived_guess_status": "unvalidated_by_standalone_http",
        "groups": result_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Voxt/build/native-final-candidate-screen"),
    )
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
    if arguments.rounds < 1 or arguments.warmups < 0 or arguments.timeout_seconds <= 0:
        parser.error("rounds, warmups, and timeout must be positive (warmups may be zero)")
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
    runtime_hash = sha256(runtime_bin.read_bytes()) if runtime_bin.is_file() else None
    config_path = model_dir / "config.json"
    provenance_path = runtime_bin.parent.parent / "build-provenance.json"
    provenance_validation: dict[str, Any] | None = None
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
    if not runtime_bin.is_file():
        run_error = f"FileNotFoundError: runtime binary is missing: {runtime_bin}"
    elif not model_dir.is_dir():
        run_error = f"FileNotFoundError: model directory is missing: {model_dir}"
    elif (
        arguments.draft_model_dir is not None
        and not arguments.draft_model_dir.is_dir()
    ):
        run_error = (
            "FileNotFoundError: draft model directory is missing: "
            f"{arguments.draft_model_dir}"
        )
    else:
        try:
            provenance_validation = validate_runtime_provenance(runtime_bin, git_root)
        except (OSError, ValueError) as error:
            run_error = f"{type(error).__name__}: {error}"

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
        if (
            arm == "off"
            and guess is None
            and record["status"] == "measured"
            and text
        ):
                    arm_warmups[arm] = text
        if (
            path == "standalone_http"
            and guess is None
            and language_mode == "explicit"
            and arm_warmups.get("off")
        ):
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
        if run_error is not None:
            raise RuntimeError(run_error)
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
                    if language_mode == "automatic":
                        records.append(
                            {
                                "event": "native_final_candidate_path_skipped",
                                "path": "oracle_verified_guess_http",
                                "fixture": fixture_name,
                                "language_mode": language_mode,
                                "status": "unsupported",
                                "reason": (
                                    "the standalone guessed_text field "
                                    "cannot represent "
                                    "automatic-language header and ASR marker tokens"
                                ),
                            }
                        )
                    elif oracle_guesses.get((fixture_name, language_mode)):
                        guess = oracle_guesses[(fixture_name, language_mode)]
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
                                "status": "unavailable",
                                "reason": (
                                    "the explicit-language OFF warmup did not produce "
                                    "an authoritative transcript"
                                ),
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
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in safe_records)
    )
    summary = summarize(
        records, arguments.rounds, model_id, arguments.oracle_guess_diagnostic
    )
    summary["runtime_provenance"] = {
        "runtime_binary": str(runtime_bin),
        "runtime_binary_sha256": runtime_hash,
        "build_provenance_path": (
            str(provenance_path) if provenance_path.is_file() else None
        ),
        "build_provenance": (
            provenance_validation.get("record") if provenance_validation else None
        ),
        "validation_checks": (
            provenance_validation.get("checks") if provenance_validation else None
        ),
        "validated": provenance_validation is not None,
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
    shadow_rows = [
        row for row in records
        if row.get("arm") in ("shadow", "shadow_dense")
        and not row.get("warmup", False)
    ]
    protocol_observations = {
        row.get("candidate_protocol_supported") for row in shadow_rows
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
    validation_errors = [
        error
        for row in records
        for error in row.get("candidate_validation_errors", [])
    ]
    failed_requests = [
        row
        for row in records
        if row.get("event") == "native_final_candidate_request"
        and row.get("status") != "measured"
    ]
    summary["validation_errors"] = validation_errors
    summary["failed_request_count"] = len(failed_requests)
    summary["run_status"] = "validation_blocked" if (
        run_error
        or server_shutdown_error
        or failed_requests
        or validation_errors
        or not records
    ) else "complete"
    summary["screening_validation_status"] = (
        "blocked" if summary["run_status"] != "complete" else "valid"
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"request_jsonl": str(jsonl_path), "summary_json": str(summary_path)}))
    return 1 if summary["run_status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())

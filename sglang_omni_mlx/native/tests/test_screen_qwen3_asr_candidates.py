# SPDX-License-Identifier: Apache-2.0
"""Validation rules for the local authoritative Final screening runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni_mlx.native.ci import screen_qwen3_asr_candidates as screen


def generation_metadata(**overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "generated_token_count": 18,
        "language": "English",
        "finish_reason": "stop",
        "verified_guess_token_count": 0,
        "accepted_guess_token_count": 0,
        "decode_paths": ["autoregressive"],
        "draft_decoder_used": False,
        "batch_decode_used": False,
        "native_chunked": False,
    }
    metadata.update(overrides)
    return metadata


def parity_row(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "native_generation_metadata_states": screen.normalize_generation_metadata(
            metadata
        ),
        "draft_decoder_used": metadata.get("draft_decoder_used"),
        "decode_paths": metadata.get("decode_paths"),
    }


def request_row(arm: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event": "native_final_candidate_request",
        "path": "standalone_http",
        "fixture": "en-short",
        "language_mode": "explicit",
        "audio_sha256": "same-audio",
        "arm": arm,
        "round": 0,
        "warmup": False,
        "status": "measured",
        "final_transcript_sha256": "transcript",
        "final_transcript_chars": 12,
        "authoritative_final_ms": 100.0,
        "authoritative_completion_elapsed_ms": 85.0,
        "candidate_protocol_supported": True if arm != "off" else None,
        "candidate_valid_for_screening": True,
        "candidate_summary_observed": True if arm != "off" else False,
        "native_published_candidate_count": 0 if arm != "off" else None,
        "native_coalesced_candidate_count": 0 if arm != "off" else None,
        "native_sse_candidate_received_count": 0,
        "candidates": [],
        "lead_at_prefix_completeness": {},
        "closest_candidate_lead_ms": None,
        "earliest_exact_match_lead_ms": None,
        "native_generation_metadata_states": screen.normalize_generation_metadata(
            generation_metadata()
        ),
        "draft_decoder_used": False,
        "decode_paths": ["autoregressive"],
        "final_to_done_ms": 0.5,
        "candidate_summary_after_final_ms": 0.1,
    }
    row.update(overrides)
    return row


def transcribe_sse_payloads(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict[str, Any] | str],
) -> dict[str, Any]:
    lines = []
    for payload in payloads:
        encoded = payload if isinstance(payload, str) else json.dumps(payload)
        lines.extend((f"data: {encoded}\n".encode(), b"\n"))

    class FakeResponse:
        status = 200

        def __iter__(self) -> Any:
            return iter(lines)

    class FakeConnection:
        def request(self, *_: Any, **__: Any) -> None:
            return

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            return

    monkeypatch.setattr(
        screen.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: FakeConnection(),
    )
    record, _ = screen.transcribe(
        SimpleNamespace(port=0),
        b"audio",
        "fixture",
        "audio-hash",
        "explicit",
        "English",
        "shadow",
        "bounded",
        0,
        False,
        2,
        None,
        1.0,
    )
    return record


def valid_runtime(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    runtime_bin = tmp_path / "build" / "bin" / "qwen3_asr_server"
    runtime_bin.parent.mkdir(parents=True)
    runtime_bin.write_bytes(b"native-runtime")
    source_path = tmp_path / "sglang_omni_mlx" / "native"
    source_path.mkdir(parents=True)
    provenance_path = runtime_bin.parent.parent / "build-provenance.json"
    provenance = {
        "experiment_commit": "commit-a",
        "native_source_commit": "commit-a",
        "native_source_path": str(source_path),
        "runtime_binary_path": str(runtime_bin),
        "runtime_binary_sha256": screen.sha256(runtime_bin.read_bytes()),
        "native_worktree_clean": True,
    }
    provenance_path.write_text(json.dumps(provenance))
    return runtime_bin, tmp_path, provenance_path, provenance


def test_standalone_omitted_draft_and_reuse_counters_are_known_zero() -> None:
    normalized = screen.normalize_generation_metadata(generation_metadata())
    for field in (
        "verify_round_count",
        "proposed_draft_token_count",
        "accepted_draft_token_count",
        "reused_prompt_token_count",
        "reused_audio_token_count",
    ):
        assert normalized[field] == {"state": "known_zero", "value": 0}
    parity = screen.metadata_parity(
        parity_row(generation_metadata()), parity_row(generation_metadata())
    )
    assert parity["overall"]["exact_match_count"] == 1
    assert (
        parity["fields"]["draft_verify_round_count"]["not_applicable_pair_count"]
        == 1
    )
    assert (
        parity["fields"]["reused_prompt_token_count"]["not_applicable_pair_count"]
        == 1
    )


def test_absent_generation_metadata_block_does_not_imply_zero_counters() -> None:
    normalized = screen.normalize_generation_metadata({}, metadata_observed=False)
    assert normalized["reused_prompt_token_count"] == {
        "state": "unknown",
        "value": None,
    }
    assert normalized["generated_token_count"] == {"state": "unknown", "value": None}


def test_explicit_zero_matches_a_protocol_omission_that_means_zero() -> None:
    omitted = generation_metadata(
        decode_paths=["autoregressive", "final_prefill_reuse"]
    )
    explicit = generation_metadata(
        decode_paths=["autoregressive", "final_prefill_reuse"],
        reused_prompt_token_count=0,
    )
    parity = screen.metadata_parity(parity_row(omitted), parity_row(explicit))
    field = parity["fields"]["reused_prompt_token_count"]
    assert field["comparable_pair_count"] == 1
    assert field["exact_match_count"] == 1


def test_draft_metadata_and_actual_decoder_usage_are_compared() -> None:
    metadata = generation_metadata(
        draft_decoder_used=True,
        verify_round_count=2,
        proposed_draft_token_count=7,
        accepted_draft_token_count=5,
        decode_paths=["draft_speculative"],
    )
    parity = screen.metadata_parity(parity_row(metadata), parity_row(metadata))
    assert parity["fields"]["draft_decoder_used"]["exact_match_count"] == 1
    assert parity["fields"]["draft_proposed_token_count"]["exact_match_count"] == 1


def test_actual_metadata_mismatch_and_genuinely_missing_metadata_are_separate() -> None:
    baseline = generation_metadata()
    mismatch = generation_metadata(generated_token_count=19)
    parity = screen.metadata_parity(parity_row(baseline), parity_row(mismatch))
    assert parity["fields"]["native_generated_token_count"]["mismatch_count"] == 1

    missing = generation_metadata()
    missing.pop("generated_token_count")
    parity = screen.metadata_parity(parity_row(baseline), parity_row(missing))
    assert (
        parity["fields"]["native_generated_token_count"]["unresolved_pair_count"]
        == 1
    )


@pytest.mark.parametrize(
    ("change", "expected_name"),
    [
        ("experiment_commit", "experiment commit"),
        ("native_source_commit", "native source commit"),
        ("native_source_path", "native source path"),
        ("native_worktree_clean", "clean native source"),
        ("runtime_binary_path", "runtime binary path"),
        ("runtime_binary_sha256", "runtime binary SHA-256"),
    ],
)
def test_runtime_provenance_rejects_stale_or_mismatched_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    expected_name: str,
) -> None:
    runtime_bin, git_root, provenance_path, provenance = valid_runtime(tmp_path)
    monkeypatch.setattr(
        screen,
        "git_value",
        lambda *arguments: "" if "status" in arguments else "commit-a",
    )
    provenance[change] = {
        "experiment_commit": "old-commit",
        "native_source_commit": "old-commit",
        "native_source_path": str(tmp_path / "other-native"),
        "native_worktree_clean": False,
        "runtime_binary_path": str(tmp_path / "other-runtime"),
        "runtime_binary_sha256": "old-hash",
    }[change]
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match=expected_name):
        screen.validate_runtime_provenance(runtime_bin, git_root)


def test_runtime_provenance_requires_a_valid_file_and_records_verified_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_bin, git_root, provenance_path, _ = valid_runtime(tmp_path)
    monkeypatch.setattr(
        screen,
        "git_value",
        lambda *arguments: "" if "status" in arguments else "commit-a",
    )
    verified = screen.validate_runtime_provenance(runtime_bin, git_root)
    assert all(verified["checks"].values())
    provenance_path.unlink()
    with pytest.raises(ValueError, match="missing runtime provenance"):
        screen.validate_runtime_provenance(runtime_bin, git_root)
    provenance_path.write_text("{")
    with pytest.raises(ValueError, match="invalid runtime provenance"):
        screen.validate_runtime_provenance(runtime_bin, git_root)


def test_runtime_provenance_rejects_a_dirty_current_native_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_bin, git_root, _, _ = valid_runtime(tmp_path)
    monkeypatch.setattr(
        screen,
        "git_value",
        lambda *arguments: " M native.cpp" if "status" in arguments else "commit-a",
    )
    with pytest.raises(ValueError, match="current native source clean"):
        screen.validate_runtime_provenance(runtime_bin, git_root)


@pytest.mark.parametrize(
    ("off_status", "off_chars"),
    [("failed", 0), ("measured", 0)],
)
def test_failed_or_empty_transcript_is_not_a_successful_parity_pair(
    off_status: str, off_chars: int
) -> None:
    records = [
        request_row(
            "off",
            status=off_status,
            final_transcript_chars=off_chars,
            final_transcript_sha256=None,
        ),
        request_row("shadow"),
    ]
    arms = screen.summarize(records, 1, "stub")["groups"][0]["arms"]
    shadow = arms["shadow"]
    assert shadow["transcript_parity_vs_off"]["comparable_pair_count"] == 0
    assert shadow["paired_unresolved_round_count"] == 1
    assert (
        shadow["generation_metadata_parity_vs_off"]["overall"][
            "comparable_pair_count"
        ]
        == 0
    )


def test_unsupported_candidate_protocol_is_not_reported_as_no_opportunity() -> None:
    records = [
        request_row("off"),
        request_row(
            "shadow",
            candidate_protocol_supported=False,
            candidate_valid_for_screening=False,
            candidate_validation_errors=[
                "candidate_protocol_unsupported_or_unresolved"
            ],
        ),
    ]
    arm = screen.summarize(records, 1, "stub")["groups"][0]["arms"]["shadow"]
    assert arm["candidate_observation_status"] == "unsupported_or_unresolved_transport"
    assert arm["candidate_supported_denominator"] == 0
    assert arm["closest_candidate_lead"] is None


def test_screening_accepts_final_before_terminal_candidate_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = transcribe_sse_payloads(
        monkeypatch,
        [
            {
                "type": "transcript.text.candidate_protocol",
                "supported": True,
                "candidate_cadence": "bounded",
            },
            {
                "type": "transcript.text.done",
                "text": "authoritative",
                "generation_metadata": {"authoritative_completion_elapsed_ms": 35.0},
            },
            {
                "type": "transcript.text.candidate",
                "text": "authoritative",
                "sequence": 1,
                "generated_token_count": 12,
                "elapsed_ms": 20.0,
                "delivery_during_terminal_drain": True,
            },
            {
                "type": "transcript.text.candidate_summary",
                "published_candidate_count": 1,
                "coalesced_candidate_count": 0,
            },
            "[DONE]",
        ]
    )
    assert record["status"] == "measured"
    assert record["candidate_valid_for_screening"] is True
    candidate = record["candidates"][0]
    assert candidate["native_candidate_to_authoritative_completion_ms"] == 15
    assert (
        candidate["timing_classification"]
        == "generated_before_completion_delivered_during_terminal_drain"
    )
    assert record["final_to_done_ms"] >= 0


def test_unsupported_protocol_is_invalid_even_when_http_transcription_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = transcribe_sse_payloads(
        monkeypatch,
        [
            {
                "type": "transcript.text.candidate_protocol",
                "supported": False,
                "candidate_cadence": "bounded",
            },
            {
                "type": "transcript.text.done",
                "text": "authoritative",
                "generation_metadata": {
                    "authoritative_completion_elapsed_ms": 35.0
                },
            },
            {
                "type": "transcript.text.candidate_summary",
                "published_candidate_count": 0,
                "coalesced_candidate_count": 0,
            },
            "[DONE]",
        ]
    )
    assert record["status"] == "measured"
    assert record["candidate_valid_for_screening"] is False
    assert (
        "candidate_protocol_unsupported_or_unresolved"
        in record["candidate_validation_errors"]
    )


@pytest.mark.parametrize(
    ("payloads", "failure_kind"),
    [
        ([{"type": "transcript.text.done", "text": "text"}], "missing_done"),
        (
            [{"type": "transcript.text.done", "text": ""}, "[DONE]"],
            "empty_final_transcript",
        ),
        ([{"type": "error", "error": {"code": "failed"}}, "[DONE]"], "server_error"),
    ],
)
def test_screening_rejects_missing_done_empty_final_and_server_error(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict[str, Any] | str],
    failure_kind: str,
) -> None:
    record = transcribe_sse_payloads(monkeypatch, payloads)
    assert record["status"] == "failed"
    assert record["failure_kind"] == failure_kind


def test_main_preserves_artifacts_when_provenance_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin" / "qwen3_asr_server"
    runtime_bin.parent.mkdir(parents=True)
    runtime_bin.write_bytes(b"stale-runtime")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "screen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_qwen3_asr_candidates",
            "--runtime-bin",
            str(runtime_bin),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert screen.main() == 1
    assert "summary_json" in capsys.readouterr().out
    assert (output_dir / "requests.jsonl").is_file()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["run_status"] == "validation_blocked"
    assert summary["runtime_provenance"]["validated"] is False
    assert "missing runtime provenance" in summary["run_error"]

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.models.minicpm_o.profiling import (
    main,
    summarize_preprocessing_intervals,
)

Event = dict[str, str | int | float | dict[str, str | int | float]]


def make_event(
    request_id: str,
    stage: str,
    event_name: str,
    timestamp_ns: int,
) -> Event:
    return {
        "request_id": request_id,
        "stage": stage,
        "event_name": event_name,
        "timestamp_ns": timestamp_ns,
        "metadata": {},
    }


def add_pair(
    events: list[Event],
    request_id: str,
    stage: str,
    opener: str,
    closer: str,
    open_ns: int,
    close_ns: int,
) -> None:
    events.extend(
        [
            make_event(request_id, stage, opener, open_ns),
            make_event(request_id, stage, closer, close_ns),
        ]
    )


def write_events(
    path: Path,
    events: list[Event],
) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def add_diagnostic_event(
    events: list[Event],
    request_id: str,
    event_name: str,
    timestamp_ns: int,
    video_index: int,
    duration_ms: float,
) -> None:
    events.append(
        {
            "request_id": request_id,
            "stage": "preprocessing",
            "event_name": event_name,
            "timestamp_ns": timestamp_ns,
            "metadata": {
                "video_index": video_index,
                "duration_ms": duration_ms,
            },
        }
    )


def test_summarize_preprocessing_intervals_preserves_request_and_stage_scope(
    tmp_path: Path,
) -> None:
    events: list[Event] = []
    add_pair(
        events,
        "r1",
        "preprocessing",
        "minicpmo_preprocess_cache_key_start",
        "minicpmo_preprocess_cache_key_end",
        0,
        1_000_000,
    )
    add_pair(
        events,
        "r1",
        "preprocessing",
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
        2_000_000,
        5_000_000,
    )
    add_pair(
        events,
        "r1",
        "preprocessing",
        "minicpmo_preprocess_video_to_images_start",
        "minicpmo_preprocess_video_to_images_end",
        6_000_000,
        8_000_000,
    )
    add_pair(
        events,
        "r1",
        "preprocessing",
        "minicpmo_preprocess_processor_start",
        "minicpmo_preprocess_processor_end",
        9_000_000,
        13_000_000,
    )
    add_pair(
        events,
        "r1",
        "preprocessing",
        "minicpmo_preprocess_payload_start",
        "minicpmo_preprocess_payload_end",
        14_000_000,
        15_000_000,
    )
    add_pair(
        events,
        "r2",
        "preprocessing",
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
        20_000_000,
        24_000_000,
    )
    add_pair(
        events,
        "r2",
        "preprocessing",
        "minicpmo_preprocess_video_to_images_start",
        "minicpmo_preprocess_video_to_images_end",
        25_000_000,
        26_000_000,
    )
    add_pair(
        events,
        "r2",
        "preprocessing",
        "minicpmo_preprocess_processor_start",
        "minicpmo_preprocess_processor_end",
        27_000_000,
        29_000_000,
    )
    add_pair(
        events,
        "r2",
        "preprocessing",
        "minicpmo_preprocess_payload_start",
        "minicpmo_preprocess_payload_end",
        30_000_000,
        32_000_000,
    )
    add_pair(
        events,
        "r3",
        "preprocessing",
        "minicpmo_preprocess_prompt_start",
        "minicpmo_preprocess_prompt_end",
        40_000_000,
        41_000_000,
    )
    add_pair(
        events,
        "r1",
        "encoder",
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
        50_000_000,
        150_000_000,
    )
    event_path = tmp_path / "events_synthetic.jsonl"
    write_events(event_path, events)

    summaries = summarize_preprocessing_intervals(event_path)
    by_phase = {summary.phase: summary for summary in summaries}

    assert set(by_phase) == {
        "cache_key",
        "video_decode",
        "video_to_images",
        "prompt",
        "processor",
        "payload",
    }
    decode = by_phase["video_decode"]
    assert decode.count == 2
    assert decode.total_ms == pytest.approx(7.0)
    assert decode.avg_ms == pytest.approx(3.5)
    assert decode.p50_ms == pytest.approx(3.5)
    assert decode.p95_ms == pytest.approx(3.95)
    assert decode.max_ms == pytest.approx(4.0)
    video_to_images = by_phase["video_to_images"]
    assert video_to_images.total_ms == pytest.approx(3.0)
    assert video_to_images.avg_ms == pytest.approx(1.5)
    assert video_to_images.p50_ms == pytest.approx(1.5)
    assert video_to_images.p95_ms == pytest.approx(1.95)
    assert video_to_images.max_ms == pytest.approx(2.0)
    processor = by_phase["processor"]
    assert processor.count == 2
    assert processor.total_ms == pytest.approx(6.0)
    assert processor.avg_ms == pytest.approx(3.0)
    assert processor.p50_ms == pytest.approx(3.0)
    assert processor.p95_ms == pytest.approx(3.9)
    assert processor.max_ms == pytest.approx(4.0)
    payload = by_phase["payload"]
    assert payload.count == 2
    assert payload.total_ms == pytest.approx(3.0)
    assert payload.avg_ms == pytest.approx(1.5)
    assert payload.p50_ms == pytest.approx(1.5)
    assert payload.p95_ms == pytest.approx(1.95)
    assert payload.max_ms == pytest.approx(2.0)


def test_summarize_omits_unobserved_media_phases(tmp_path: Path) -> None:
    events: list[Event] = []
    add_pair(
        events,
        "no-video",
        "preprocessing",
        "minicpmo_preprocess_prompt_start",
        "minicpmo_preprocess_prompt_end",
        0,
        1_000_000,
    )
    event_path = tmp_path / "events_no_video.jsonl"
    write_events(event_path, events)

    phases = {
        summary.phase for summary in summarize_preprocessing_intervals(event_path)
    }

    assert phases == {"prompt"}


def test_profiling_cli_defaults_to_table_and_supports_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[Event] = []
    add_pair(
        events,
        "cli-test",
        "preprocessing",
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
        0,
        2_000_000,
    )
    event_path = tmp_path / "events_cli.jsonl"
    write_events(event_path, events)

    assert main([str(event_path)]) == 0
    table_output = capsys.readouterr().out
    assert "phase" in table_output
    assert "video_decode" in table_output

    assert main([str(event_path), "--format", "json"]) == 0
    json_output = capsys.readouterr().out
    rows = json.loads(json_output)
    assert rows == [
        {
            "phase": "video_decode",
            "count": 1,
            "total_ms": 2.0,
            "avg_ms": 2.0,
            "p50_ms": 2.0,
            "p95_ms": 2.0,
            "max_ms": 2.0,
        }
    ]


def test_summarize_video_diagnostics_counts_completed_records_per_video(
    tmp_path: Path,
) -> None:
    events: list[Event] = []
    add_pair(
        events,
        "two-videos",
        "preprocessing",
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
        0,
        30_000_000,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_backend_decode",
        10_000_000,
        0,
        11.0,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_backend_decode",
        11_000_000,
        1,
        22.0,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_resize_convert",
        12_000_000,
        1,
        5.0,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_resize_convert",
        13_000_000,
        0,
        4.0,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_tensor_prepare",
        14_000_000,
        0,
        3.0,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_tensor_prepare",
        15_000_000,
        1,
        7.0,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_pil_materialize",
        16_000_000,
        0,
        2.0,
    )
    add_diagnostic_event(
        events,
        "two-videos",
        "minicpmo_preprocess_video_pil_materialize",
        17_000_000,
        1,
        6.0,
    )
    event_path = tmp_path / "events_two_videos.jsonl"
    write_events(event_path, events)

    summaries = {
        summary.phase: summary
        for summary in summarize_preprocessing_intervals(event_path)
    }

    assert summaries["video_decode"].count == 1
    assert summaries["video_backend_decode"].count == 2
    assert summaries["video_backend_decode"].total_ms == pytest.approx(33.0)
    assert summaries["video_resize_convert"].count == 2
    assert summaries["video_resize_convert"].total_ms == pytest.approx(9.0)
    assert summaries["video_tensor_prepare"].count == 2
    assert summaries["video_tensor_prepare"].total_ms == pytest.approx(10.0)
    assert summaries["video_pil_materialize"].count == 2
    assert summaries["video_pil_materialize"].total_ms == pytest.approx(8.0)

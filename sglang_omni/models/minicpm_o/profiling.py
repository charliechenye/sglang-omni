# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o preprocessing profiler views and CLI.

The phase summaries below cover measured, request-local preprocessing intervals
only. Their sum is not stage residence: stage residence may also contain
uninstrumented work and scheduler waiting. This module intentionally does not
label that remainder as queue time. Use the generic profiler stage breakdown
when stage residence is needed.

The cache-key phase retains its existing name and covers image/video cache-key
work before media loading. Audio cache-key work remains inside the audio phase.

Parent phases are request-scoped start/end intervals. Video diagnostic phases
are completed, per-video duration records; they are not paired by this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sglang_omni.profiler.views import (
    compute_stage_intervals,
    format_table,
    percentile,
    reconstruct_timelines,
)

MINICPMO_PREPROCESS_STAGE = "preprocessing"
MINICPMO_PREPROCESS_INTERVAL_EVENTS = (
    (
        "minicpmo_preprocess_cache_key_start",
        "minicpmo_preprocess_cache_key_end",
    ),
    (
        "minicpmo_preprocess_image_load_start",
        "minicpmo_preprocess_image_load_end",
    ),
    (
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
    ),
    (
        "minicpmo_preprocess_video_to_images_start",
        "minicpmo_preprocess_video_to_images_end",
    ),
    ("minicpmo_preprocess_audio_start", "minicpmo_preprocess_audio_end"),
    ("minicpmo_preprocess_prompt_start", "minicpmo_preprocess_prompt_end"),
    (
        "minicpmo_preprocess_processor_start",
        "minicpmo_preprocess_processor_end",
    ),
    ("minicpmo_preprocess_payload_start", "minicpmo_preprocess_payload_end"),
)

MINICPMO_PREPROCESS_PHASE_NAMES = {
    (
        "minicpmo_preprocess_cache_key_start",
        "minicpmo_preprocess_cache_key_end",
    ): "cache_key",
    (
        "minicpmo_preprocess_image_load_start",
        "minicpmo_preprocess_image_load_end",
    ): "image_load",
    (
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
    ): "video_decode",
    (
        "minicpmo_preprocess_video_to_images_start",
        "minicpmo_preprocess_video_to_images_end",
    ): "video_to_images",
    ("minicpmo_preprocess_audio_start", "minicpmo_preprocess_audio_end"): "audio",
    ("minicpmo_preprocess_prompt_start", "minicpmo_preprocess_prompt_end"): "prompt",
    (
        "minicpmo_preprocess_processor_start",
        "minicpmo_preprocess_processor_end",
    ): "processor",
    (
        "minicpmo_preprocess_payload_start",
        "minicpmo_preprocess_payload_end",
    ): "payload",
}

MINICPMO_PREPROCESS_DIAGNOSTIC_EVENTS = {
    "minicpmo_preprocess_video_backend_decode": "video_backend_decode",
    "minicpmo_preprocess_video_resize_convert": "video_resize_convert",
    "minicpmo_preprocess_video_tensor_prepare": "video_tensor_prepare",
    "minicpmo_preprocess_video_pil_materialize": "video_pil_materialize",
}

MINICPMO_PREPROCESS_PHASE_ORDER = (
    "cache_key",
    "image_load",
    "video_decode",
    "video_backend_decode",
    "video_resize_convert",
    "video_to_images",
    "video_tensor_prepare",
    "video_pil_materialize",
    "audio",
    "prompt",
    "processor",
    "payload",
)


@dataclass(frozen=True, kw_only=True)
class PreprocessingPhaseSummary:
    """Aggregate measured durations for one MiniCPM-o preprocessing phase."""

    phase: str
    count: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "phase": self.phase,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def summarize_preprocessing_intervals(
    source: str | Path | Iterable[str | Path],
) -> list[PreprocessingPhaseSummary]:
    """Summarize observed MiniCPM-o preprocessing intervals by phase.

    Matching remains scoped by request and stage through
    :func:`compute_stage_intervals`; intervals from other stages are ignored.
    Parent phases count one request interval. Diagnostic child phases count
    one completed per-video duration record, so concurrent videos cannot be
    cross-paired. Phases with no matching record are omitted rather than
    reported as zero.
    """
    timelines = reconstruct_timelines(source)
    intervals = compute_stage_intervals(
        timelines,
        interval_events=MINICPMO_PREPROCESS_INTERVAL_EVENTS,
    )
    durations_by_phase: dict[str, list[float]] = defaultdict(list)
    for interval in intervals:
        if interval.stage != MINICPMO_PREPROCESS_STAGE:
            continue
        else:
            pass
        phase = MINICPMO_PREPROCESS_PHASE_NAMES[
            (interval.open_event, interval.close_event)
        ]
        durations_by_phase[phase].append(interval.duration_ms)

    for timeline in timelines.values():
        for event in timeline.events:
            if event.get("stage") != MINICPMO_PREPROCESS_STAGE:
                continue
            else:
                pass
            event_name = event.get("event_name")
            if not isinstance(event_name, str):
                continue
            else:
                pass
            phase = MINICPMO_PREPROCESS_DIAGNOSTIC_EVENTS.get(event_name)
            if phase is None:
                continue
            else:
                pass
            metadata = event.get("metadata")
            if not isinstance(metadata, dict):
                continue
            else:
                pass
            video_index = metadata.get("video_index")
            if isinstance(video_index, bool) or not isinstance(video_index, int):
                continue
            else:
                pass
            duration_ms = metadata.get("duration_ms")
            if isinstance(duration_ms, bool) or not isinstance(
                duration_ms, (int, float)
            ):
                continue
            else:
                pass
            if duration_ms < 0:
                continue
            else:
                pass
            durations_by_phase[phase].append(float(duration_ms))

    summaries: list[PreprocessingPhaseSummary] = []
    for phase in MINICPMO_PREPROCESS_PHASE_ORDER:
        durations = durations_by_phase.get(phase, [])
        if not durations:
            continue
        else:
            pass
        durations.sort()
        total_ms = sum(durations)
        summaries.append(
            PreprocessingPhaseSummary(
                phase=phase,
                count=len(durations),
                total_ms=total_ms,
                avg_ms=total_ms / len(durations),
                p50_ms=percentile(durations, 0.50),
                p95_ms=percentile(durations, 0.95),
                max_ms=durations[-1],
            )
        )
    return summaries


def main(argv: list[str] | None = None) -> int:
    """Run the MiniCPM-o preprocessing profiling CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m sglang_omni.models.minicpm_o.profiling",
        description="Summarize MiniCPM-o preprocessing phase intervals",
        epilog=(
            "Phase totals are measured preprocessing intervals, not stage "
            "residence or queue time."
        ),
    )
    parser.add_argument(
        "source",
        help="Event JSONL file or directory of events_*.jsonl files",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output path; '-' writes to stdout (default)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Report format (default: table)",
    )
    args = parser.parse_args(argv)

    summaries = summarize_preprocessing_intervals(args.source)
    rows = [summary.to_dict() for summary in summaries]
    if args.format == "json":
        text = json.dumps(rows, indent=2)
    else:
        text = format_table(
            rows,
            ["phase", "count", "total_ms", "avg_ms", "p50_ms", "p95_ms", "max_ms"],
        )

    if args.out == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        else:
            pass
    else:
        with open(args.out, "w", encoding="utf-8") as output_file:
            output_file.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
else:
    pass

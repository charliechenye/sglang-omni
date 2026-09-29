# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o preprocessing profiler views and CLI.

The phase summaries below cover measured, request-local preprocessing intervals
only. Their sum is not stage residence: stage residence may also contain
uninstrumented work and scheduler waiting. This module intentionally does not
label that remainder as queue time. Use the generic profiler stage breakdown
when stage residence is needed.

The cache-key phase retains its existing name and covers image/video cache-key
work before media loading. Audio cache-key work remains inside the audio phase.
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
    Phases with no matching interval are omitted rather than reported as zero.
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

    summaries: list[PreprocessingPhaseSummary] = []
    for interval_events in MINICPMO_PREPROCESS_INTERVAL_EVENTS:
        phase = MINICPMO_PREPROCESS_PHASE_NAMES[interval_events]
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

# SPDX-License-Identifier: Apache-2.0
"""Frozen Video-MME requests and production-generated vision input census."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from benchmarks.eval.minicpmo_vision_metrics import Distribution, summarize

FROZEN_SAMPLE_IDS = (
    "002-1",
    "003-1",
    "005-1",
    "006-1",
    "007-1",
    "008-1",
    "009-1",
    "010-1",
    "011-1",
    "012-1",
    "014-1",
    "015-1",
    "017-1",
    "018-1",
    "019-1",
    "020-1",
)
VISION_BATCH_SIZES = (16, 32, 64)
VIDEO_FPS = 2
VIDEO_MAX_FRAMES = 128
VIDEO_MAX_PIXELS = 401408


class FrozenRequest(BaseModel):
    sample_id: str
    video_path: str | None = None
    prompt: str | None = None


class EncoderInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pixel_values: list[torch.Tensor]
    tgt_sizes: torch.Tensor


@dataclass(kw_only=True)
class CensusEntry:
    sample_id: str
    video_path: str
    video_size_bytes: int
    video_mtime_ns: int
    num_slices: int
    tgt_sizes: list[list[int]]
    patch_counts: list[int]
    total_patches: int
    max_patches: int
    attention_work_proxy: int
    vpm_chunks: dict[int, int]


@dataclass(kw_only=True)
class CachedRequest:
    sample_id: str
    inputs: EncoderInputs
    census: CensusEntry


@dataclass(kw_only=True)
class Geometry:
    chunk_slices: int
    padded_patches_per_slice: int
    packed_tokens: int
    segmentation_pattern: list[int]
    count: int
    share: float


@dataclass(kw_only=True)
class ChunkSummary:
    batch_size: int
    mean: float
    maximum: int
    unique_geometries: int
    chunks: int
    top_geometries: list[Geometry]


@dataclass(kw_only=True)
class CensusSummary:
    samples: int
    num_slices: Distribution
    total_patches: Distribution
    attention_work_proxy: Distribution
    batches: list[ChunkSummary]


@dataclass(kw_only=True)
class Selection:
    sample_id: str
    reason: str
    metric: str
    target_attention_work_proxy: float
    actual_attention_work_proxy: int
    actual_total_patches: int


class CheckpointIdentity(BaseModel):
    checkpoint_config_sha256: str


class CensusSnapshot(BaseModel):
    mode: Literal["census"]
    model_id: str
    videomme_repo: str
    sample_ids: list[str]
    video_fps: float
    video_max_frames: int
    video_max_pixels: int
    model: CheckpointIdentity
    census: list[CensusEntry]


def validate_frozen_ids(sample_ids: list[str]) -> None:
    if len(sample_ids) != len(FROZEN_SAMPLE_IDS) or set(sample_ids) != set(
        FROZEN_SAMPLE_IDS
    ):
        raise ValueError(
            f"Cohort must contain exactly the frozen 16 IDs: {list(FROZEN_SAMPLE_IDS)}; "
            f"received {sample_ids}"
        )
    else:
        pass


def summarize_census(entries: list[CensusEntry]) -> CensusSummary:
    batches = []
    for batch_size in VISION_BATCH_SIZES:
        geometries = Counter(
            (len(patch_counts), entry.max_patches, tuple(patch_counts))
            for entry in entries
            for start in range(0, entry.num_slices, batch_size)
            for patch_counts in [entry.patch_counts[start : start + batch_size]]
        )
        chunk_count = sum(geometries.values())
        batches.append(
            ChunkSummary(
                batch_size=batch_size,
                mean=sum(entry.vpm_chunks[batch_size] for entry in entries)
                / len(entries),
                maximum=max(entry.vpm_chunks[batch_size] for entry in entries),
                unique_geometries=len(geometries),
                chunks=chunk_count,
                top_geometries=[
                    Geometry(
                        chunk_slices=geometry[0],
                        padded_patches_per_slice=geometry[1],
                        packed_tokens=sum(geometry[2]),
                        segmentation_pattern=list(geometry[2]),
                        count=count,
                        share=count / chunk_count,
                    )
                    for geometry, count in geometries.most_common(5)
                ],
            )
        )
    return CensusSummary(
        samples=len(entries),
        num_slices=summarize([entry.num_slices for entry in entries]),
        total_patches=summarize([entry.total_patches for entry in entries]),
        attention_work_proxy=summarize(
            [entry.attention_work_proxy for entry in entries]
        ),
        batches=batches,
    )


def select_request(
    entries: list[CensusEntry],
    sample_id: str | None,
    selection: Literal["median", "high"],
) -> Selection:
    work = summarize([entry.attention_work_proxy for entry in entries])
    target = work.median if selection == "median" else work.p95
    if sample_id is None:
        selected = min(
            entries,
            key=lambda entry: (
                abs(entry.attention_work_proxy - target),
                entry.sample_id,
            ),
        )
        reason = (
            f"closest to {'p50' if selection == 'median' else 'p95'} "
            "attention_work_proxy; ties use sample ID"
        )
    else:
        matching = [entry for entry in entries if entry.sample_id == sample_id]
        if not matching:
            raise ValueError(f"Sample {sample_id} is not in the frozen cohort")
        else:
            selected = matching[0]
            target = selected.attention_work_proxy
            reason = "explicit --sample-id"
    return Selection(
        sample_id=selected.sample_id,
        reason=reason,
        metric="attention_work_proxy",
        target_attention_work_proxy=target,
        actual_attention_work_proxy=selected.attention_work_proxy,
        actual_total_patches=selected.total_patches,
    )


def load_census(
    census_file: Path,
    requests: list[FrozenRequest],
    model_id: str,
    videomme_repo: str,
    checkpoint_config_sha256: str,
) -> list[CensusEntry]:
    snapshot = CensusSnapshot.model_validate_json(census_file.read_text())
    validate_frozen_ids(snapshot.sample_ids)
    validate_frozen_ids([entry.sample_id for entry in snapshot.census])
    if (
        snapshot.model_id,
        snapshot.videomme_repo,
        snapshot.video_fps,
        snapshot.video_max_frames,
        snapshot.video_max_pixels,
        snapshot.model.checkpoint_config_sha256,
    ) != (
        model_id,
        videomme_repo,
        VIDEO_FPS,
        VIDEO_MAX_FRAMES,
        VIDEO_MAX_PIXELS,
        checkpoint_config_sha256,
    ):
        raise ValueError(
            "Census configuration differs; rerun census with this model/dataset"
        )
    else:
        pass
    indexed_requests = {request.sample_id: request for request in requests}
    for entry in snapshot.census:
        video_path = Path(indexed_requests[entry.sample_id].video_path)
        video_stat = video_path.stat()
        if (str(video_path), video_stat.st_size, video_stat.st_mtime_ns) != (
            entry.video_path,
            entry.video_size_bytes,
            entry.video_mtime_ns,
        ):
            raise ValueError(
                f"Video identity changed for {entry.sample_id}; rerun census"
            )
        else:
            pass
    return snapshot.census


def print_census(entries: list[CensusEntry], summary: CensusSummary) -> None:
    print("=== CENSUS ===")
    for entry in entries:
        print(json.dumps(asdict(entry), separators=(",", ":")))
    print(f"samples={summary.samples}")
    for label, distribution in (
        ("num_slices", summary.num_slices),
        ("total_patches", summary.total_patches),
        ("attention_work_proxy", summary.attention_work_proxy),
    ):
        print(
            f"{label}: min={distribution.minimum:.0f} p50={distribution.median:.1f} p95={distribution.p95:.1f} max={distribution.maximum:.0f}"
        )
    print("run_vpm calls/request: batch  mean  max  unique_geometries  total_chunks")
    for batch in summary.batches:
        print(
            f"batch={batch.batch_size:2d} mean={batch.mean:.2f} max={batch.maximum} unique={batch.unique_geometries} chunks={batch.chunks}"
        )
        for geometry in batch.top_geometries:
            print(
                f"  chunk_slices={geometry.chunk_slices} packed_tokens={geometry.packed_tokens} padded_patches={geometry.padded_patches_per_slice} count={geometry.count} share={geometry.share:.3%}"
            )

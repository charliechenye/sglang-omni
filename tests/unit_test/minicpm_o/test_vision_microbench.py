# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for experiment selection, parity, and reversible MHA hooks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from benchmarks.eval.minicpmo_vision_metrics import (
    ParityLimits,
    compare_outputs,
    temporary_need_weights,
)
from benchmarks.eval.minicpmo_vision_workload import (
    FROZEN_SAMPLE_IDS,
    VIDEO_FPS,
    VIDEO_MAX_FRAMES,
    VIDEO_MAX_PIXELS,
    CensusEntry,
    FrozenRequest,
    load_census,
    select_request,
    summarize_census,
    validate_frozen_ids,
)


def test_parity_accepts_small_nonbitwise_change_and_rejects_drift() -> None:
    baseline = torch.tensor([1.0, 2.0, 3.0])
    small_change = compare_outputs(baseline, baseline + 1e-6, ParityLimits())
    assert not small_change.torch_equal
    assert not small_change.substantial_drift
    assert small_change.max_abs is not None and small_change.max_abs < 1e-5
    assert compare_outputs(baseline, baseline + 1.0, ParityLimits()).substantial_drift


def test_parity_handles_zero_vectors_dtype_shape_and_nonfinite() -> None:
    baseline = torch.zeros(3)
    equality = compare_outputs(baseline, baseline, ParityLimits())
    assert equality.rel_l2 == 0.0 and equality.cosine_similarity == 1.0
    assert not equality.substantial_drift
    assert compare_outputs(baseline, torch.ones(3), ParityLimits()).substantial_drift
    assert compare_outputs(
        baseline, baseline.to(torch.float64), ParityLimits()
    ).substantial_drift
    assert compare_outputs(baseline, torch.zeros(4), ParityLimits()).substantial_drift
    assert compare_outputs(
        baseline, torch.tensor([math.nan, 0.0, 0.0]), ParityLimits()
    ).substantial_drift


def test_frozen_cohort_cannot_silently_replace_or_duplicate_ids() -> None:
    validate_frozen_ids(list(FROZEN_SAMPLE_IDS))
    with pytest.raises(ValueError, match="exactly the frozen 16"):
        validate_frozen_ids([*FROZEN_SAMPLE_IDS[:-1], FROZEN_SAMPLE_IDS[0]])
    with pytest.raises(ValueError, match="exactly the frozen 16"):
        validate_frozen_ids(list(FROZEN_SAMPLE_IDS[:-1]))


def make_census_entry(
    sample_id: str, slices: int, patches_per_slice: int
) -> CensusEntry:
    return CensusEntry(
        sample_id=sample_id,
        video_path="unused",
        video_size_bytes=0,
        video_mtime_ns=0,
        num_slices=slices,
        tgt_sizes=[[1, patches_per_slice]] * slices,
        patch_counts=[patches_per_slice] * slices,
        total_patches=slices * patches_per_slice,
        max_patches=patches_per_slice,
        vpm_chunks={
            batch_size: math.ceil(slices / batch_size) for batch_size in (16, 32, 64)
        },
    )


def test_real_slice_counts_drive_chunk_census_and_geometry_shares() -> None:
    summary = summarize_census([make_census_entry("002-1", 17, 2)])
    assert [batch.maximum for batch in summary.batches] == [2, 1, 1]
    assert summary.total_patches.maximum == 34
    assert sum(geometry.share for geometry in summary.batches[0].top_geometries) == 1.0
    assert summary.batches[0].unique_geometries == 2


def test_selection_uses_visual_work_with_deterministic_ties() -> None:
    entries = [
        make_census_entry("020-1", 20, 2),
        make_census_entry("003-1", 2, 2),
        make_census_entry("002-1", 2, 2),
    ]
    assert select_request(entries, None, "median").sample_id == "002-1"
    assert select_request(entries, None, "high").sample_id == "020-1"
    assert select_request(entries, "003-1", "high").sample_id == "003-1"
    with pytest.raises(ValueError, match="not in the frozen cohort"):
        select_request(entries, "missing", "median")


def test_saved_census_roundtrip_rejects_changed_checkpoint_or_media(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"unit-test media identity")
    video_stat = video_path.stat()
    entries = [make_census_entry(sample_id, 17, 2) for sample_id in FROZEN_SAMPLE_IDS]
    for entry in entries:
        entry.video_path = str(video_path)
        entry.video_size_bytes = video_stat.st_size
        entry.video_mtime_ns = video_stat.st_mtime_ns
    requests = [
        FrozenRequest(
            sample_id=entry.sample_id, video_path=entry.video_path, prompt="unused"
        )
        for entry in entries
    ]
    census_path = tmp_path / "census.json"
    census_path.write_text(
        json.dumps(
            {
                "mode": "census",
                "model_id": "checkpoint",
                "videomme_repo": "dataset",
                "sample_ids": list(FROZEN_SAMPLE_IDS),
                "video_fps": VIDEO_FPS,
                "video_max_frames": VIDEO_MAX_FRAMES,
                "video_max_pixels": VIDEO_MAX_PIXELS,
                "model": {"checkpoint_config_sha256": "configuration"},
                "census": [asdict(entry) for entry in entries],
            }
        )
    )
    restored = load_census(
        census_path, requests, "checkpoint", "dataset", "configuration"
    )
    assert restored[0].vpm_chunks[16] == 2
    with pytest.raises(ValueError, match="configuration differs"):
        load_census(census_path, requests, "checkpoint", "dataset", "changed")
    video_path.write_bytes(b"different media")
    with pytest.raises(ValueError, match="Video identity changed"):
        load_census(census_path, requests, "checkpoint", "dataset", "configuration")


def test_need_weights_change_preserves_weights_and_restores_on_exception() -> None:
    attention = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2)
    query = torch.ones(3, 1, 8)
    value = torch.ones(5, 1, 8)
    original_forward = attention.forward
    original_weights = {
        name: parameter.detach().clone()
        for name, parameter in attention.named_parameters()
    }
    with torch.inference_mode():
        baseline, weights = attention(query, value, value)
        assert weights is not None
        with pytest.raises(RuntimeError, match="deliberate"):
            with temporary_need_weights(attention, False):
                candidate, discarded_weights = attention(query, value, value)
                assert discarded_weights is None
                assert not compare_outputs(
                    baseline, candidate, ParityLimits()
                ).substantial_drift
                raise RuntimeError("deliberate")
        _, restored_weights = attention(query, value, value)
        assert restored_weights is not None
    assert attention.forward == original_forward
    assert all(
        torch.equal(parameter, original_weights[name])
        for name, parameter in attention.named_parameters()
    )

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.dataset.videomme import VideoMMESample
from benchmarks.eval.minicpmo_video_resize_parity_helpers import (
    compare_processor_outputs,
    read_cohort_sample_ids,
    select_cohort_samples,
)


def make_processor_outputs(
    *,
    pixel_values: list[torch.Tensor],
    image_bound: torch.Tensor | None = None,
    tgt_sizes: list[torch.Tensor] | None = None,
) -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "image_bound": [
            [
                (
                    image_bound
                    if image_bound is not None
                    else torch.tensor([0, 1], dtype=torch.long)
                )
            ]
        ],
        "tgt_sizes": [
            [
                tgt_size
                for tgt_size in (
                    tgt_sizes
                    if tgt_sizes is not None
                    else [torch.tensor([1, 1], dtype=torch.long)]
                )
            ]
        ],
        "pixel_values": [pixel_values],
    }


def test_identical_processor_outputs_pass_exact_parity() -> None:
    outputs = make_processor_outputs(
        pixel_values=[torch.tensor([[1.0, 2.0]], dtype=torch.float32)]
    )

    report = compare_processor_outputs(outputs, outputs)

    assert report.parity_pass is True
    assert report.input_ids_equal is True
    assert report.image_bound_equal is True
    assert report.tgt_sizes_equal is True
    assert report.pixel_exact_equal is True
    assert report.pixel_max_abs == pytest.approx(0.0)
    assert report.pixel_mean_abs == pytest.approx(0.0)
    assert report.pixel_rel_l2 == pytest.approx(0.0)


def test_numerical_pixel_mismatch_reports_metrics() -> None:
    baseline = make_processor_outputs(
        pixel_values=[torch.tensor([[1.0, 2.0]], dtype=torch.float32)]
    )
    candidate = make_processor_outputs(
        pixel_values=[torch.tensor([[1.0, 4.0]], dtype=torch.float32)]
    )

    report = compare_processor_outputs(baseline, candidate)

    assert report.parity_pass is False
    assert report.pixel_shapes_equal is True
    assert report.pixel_exact_equal is False
    assert report.pixel_max_abs == pytest.approx(2.0)
    assert report.pixel_mean_abs == pytest.approx(1.0)
    assert report.pixel_rel_l2 == pytest.approx(2.0 / (5.0**0.5))


def test_structural_mismatch_fails_without_fake_pixel_parity() -> None:
    baseline = make_processor_outputs(
        pixel_values=[torch.tensor([[1.0, 2.0]], dtype=torch.float32)],
        image_bound=torch.tensor([0, 1], dtype=torch.long),
        tgt_sizes=[torch.tensor([1, 1], dtype=torch.long)],
    )
    candidate = make_processor_outputs(
        pixel_values=[
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        ],
        image_bound=torch.tensor([0, 2], dtype=torch.long),
        tgt_sizes=[torch.tensor([2, 1], dtype=torch.long)],
    )

    report = compare_processor_outputs(baseline, candidate)

    assert report.parity_pass is False
    assert report.image_bound_equal is False
    assert report.tgt_sizes_equal is False
    assert report.pixel_tensor_count_equal is False
    assert report.pixel_exact_equal is False
    assert report.pixel_max_abs is None
    assert report.pixel_mean_abs is None
    assert report.pixel_rel_l2 is None


def test_frozen_cohort_filter_preserves_requested_order(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(
        json.dumps({"sample_ids": ["sample-2", "sample-1"]}), encoding="utf-8"
    )
    samples = [
        VideoMMESample(
            sample_id="sample-1",
            video_path="one.mp4",
            question="",
            options=[],
            answer="",
        ),
        VideoMMESample(
            sample_id="sample-2",
            video_path="two.mp4",
            question="",
            options=[],
            answer="",
        ),
    ]

    sample_ids = read_cohort_sample_ids(cohort_path)
    selected = select_cohort_samples(samples, sample_ids)

    assert sample_ids == ["sample-2", "sample-1"]
    assert [sample.sample_id for sample in selected] == sample_ids

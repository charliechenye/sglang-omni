from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from benchmarks.dataset.videomme import VideoMMESample
from benchmarks.eval.minicpmo_video_resize_parity_helpers import (
    compare_processor_outputs,
    read_cohort_sample_ids,
    select_cohort_samples,
)
from sglang_omni.preprocessing.video import (
    VideoResizeGeometry,
    resize_video_tensor,
    resize_video_tensor_parallel,
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


def make_resize_input(frame_count: int, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(frame_count * 3 * 4 * 6, dtype=torch.float32)
    if dtype is torch.uint8:
        values = values.remainder(251)
    else:
        pass
    return values.to(dtype=dtype).reshape(frame_count, 3, 4, 6)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize(
    ("frame_count", "chunks"),
    [(1, 1), (3, 8), (5, 2), (7, 3)],
)
def test_parallel_video_resize_is_bitwise_exact(
    dtype: torch.dtype,
    frame_count: int,
    chunks: int,
) -> None:
    video = make_resize_input(frame_count, dtype)
    geometry = VideoResizeGeometry(
        frame_count=frame_count,
        source_height=4,
        source_width=6,
        resized_height=8,
        resized_width=12,
    )

    serial = resize_video_tensor(video, geometry)
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = resize_video_tensor_parallel(
            video,
            geometry,
            executor=executor,
            chunks=chunks,
        )

    assert parallel.shape == serial.shape
    assert parallel.dtype == serial.dtype
    assert torch.equal(parallel, serial)


def test_parallel_video_resize_chunks_one_uses_serial_helper() -> None:
    video = make_resize_input(3, torch.uint8)
    geometry = VideoResizeGeometry(
        frame_count=3,
        source_height=4,
        source_width=6,
        resized_height=8,
        resized_width=12,
    )
    executor = Mock()

    parallel = resize_video_tensor_parallel(
        video,
        geometry,
        executor=executor,
        chunks=1,
    )

    assert torch.equal(parallel, resize_video_tensor(video, geometry))
    executor.submit.assert_not_called()


def test_shared_resize_executor_handles_concurrent_callers() -> None:
    geometry = VideoResizeGeometry(
        frame_count=5,
        source_height=4,
        source_width=6,
        resized_height=8,
        resized_width=12,
    )
    videos = [make_resize_input(5, torch.float32) + index for index in range(4)]
    serial_outputs = [resize_video_tensor(video, geometry) for video in videos]

    with ThreadPoolExecutor(max_workers=8) as resize_executor:
        with ThreadPoolExecutor(max_workers=4) as request_executor:
            futures = [
                request_executor.submit(
                    resize_video_tensor_parallel,
                    video,
                    geometry,
                    executor=resize_executor,
                    chunks=8,
                )
                for video in videos
            ]
            parallel_outputs = [future.result(timeout=5.0) for future in futures]

    assert all(
        torch.equal(parallel, serial)
        for parallel, serial in zip(parallel_outputs, serial_outputs)
    )


def test_parallel_video_resize_propagates_worker_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = make_resize_input(4, torch.uint8)
    geometry = VideoResizeGeometry(
        frame_count=4,
        source_height=4,
        source_width=6,
        resized_height=8,
        resized_width=12,
    )
    original_resize = resize_video_tensor

    def fail_for_chunk(
        video_piece: torch.Tensor,
        video_geometry: VideoResizeGeometry,
    ) -> torch.Tensor:
        if video_piece.shape[0] == 2:
            raise RuntimeError("resize worker failed")
        else:
            pass
        return original_resize(video_piece, video_geometry)

    monkeypatch.setattr(
        "sglang_omni.preprocessing.video.resize_video_tensor", fail_for_chunk
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(RuntimeError, match="resize worker failed"):
            resize_video_tensor_parallel(
                video,
                geometry,
                executor=executor,
                chunks=2,
            )

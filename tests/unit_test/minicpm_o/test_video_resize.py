# SPDX-License-Identifier: Apache-2.0
"""Exactness and concurrency contracts for MiniCPM-o video resizing."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from sglang_omni.preprocessing import video as video_module
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.preprocessing.video import (
    VideoMediaIO,
    ensure_video_list_async,
    resize_video_tensor,
    resize_video_tensor_parallel,
)


def make_video_tensor(
    frame_count: int,
    dtype: torch.dtype,
    non_contiguous: bool,
) -> torch.Tensor:
    source = torch.arange(frame_count * 3 * 9 * 11, dtype=torch.int64)
    source = source.remainder(251).reshape(frame_count, 3, 9, 11).to(dtype)
    if dtype.is_floating_point:
        source = source / 17.0
    else:
        pass
    if non_contiguous:
        return source.transpose(2, 3)
    else:
        pass
    return source


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("non_contiguous", [False, True])
@pytest.mark.parametrize(
    ("frame_count", "chunks"),
    [(1, 1), (5, 1), (5, 3), (5, 8)],
)
def test_parallel_resize_is_bitwise_equal_to_serial_resize(
    dtype: torch.dtype,
    non_contiguous: bool,
    frame_count: int,
    chunks: int,
) -> None:
    video = make_video_tensor(frame_count, dtype, non_contiguous)
    serial = resize_video_tensor(video, 7, 5)
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = resize_video_tensor_parallel(
            video,
            7,
            5,
            executor=executor,
            chunks=chunks,
        )
    assert torch.equal(serial, parallel)


def test_parallel_resize_propagates_worker_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = make_video_tensor(5, torch.float32, False)

    def fail_for_two_frame_chunk(
        video_chunk: torch.Tensor,
        resized_height: int,
        resized_width: int,
    ) -> torch.Tensor:
        if video_chunk.shape[0] == 2:
            raise RuntimeError("resize worker failed")
        else:
            pass
        return video_chunk

    monkeypatch.setattr(video_module, "resize_video_tensor", fail_for_two_frame_chunk)
    with ThreadPoolExecutor(max_workers=3) as executor:
        with pytest.raises(RuntimeError, match="resize worker failed"):
            resize_video_tensor_parallel(
                video,
                7,
                5,
                executor=executor,
                chunks=3,
            )


def test_multiple_callers_can_share_one_resize_executor() -> None:
    videos = [
        make_video_tensor(frame_count, torch.float32, frame_count % 2 == 0)
        for frame_count in (3, 5, 7)
    ]
    with ThreadPoolExecutor(max_workers=6) as shared_executor:
        with ThreadPoolExecutor(max_workers=3) as caller_executor:
            futures = [
                caller_executor.submit(
                    resize_video_tensor_parallel,
                    video,
                    7,
                    5,
                    executor=shared_executor,
                    chunks=4,
                )
                for video in videos
            ]
            parallel_videos = [future.result() for future in futures]
    serial_videos = [resize_video_tensor(video, 7, 5) for video in videos]
    assert all(
        torch.equal(serial_video, parallel_video)
        for serial_video, parallel_video in zip(
            serial_videos, parallel_videos, strict=True
        )
    )


def test_load_video_path_keeps_float_conversion_outside_parallel_resize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = make_video_tensor(5, torch.uint8, False)
    for constant_name, constant_value in {
        "VIDEO_MIN_PIXELS": 8,
        "VIDEO_TOTAL_PIXELS": 256,
        "VIDEO_MAX_PIXELS": 128,
        "FRAME_FACTOR": 1,
        "IMAGE_FACTOR": 2,
    }.items():
        monkeypatch.setattr(
            video_module.qwen_vision,
            constant_name,
            constant_value,
            raising=False,
        )
    monkeypatch.setattr(
        video_module.qwen_vision, "get_video_reader_backend", lambda: "fake"
    )
    monkeypatch.setitem(
        video_module.qwen_vision.VIDEO_READER_BACKENDS,
        "fake",
        lambda _element: (source, 8.0),
    )
    monkeypatch.setattr(
        video_module.qwen_vision,
        "smart_resize",
        lambda *_arguments, **_options: (7, 5),
    )
    monkeypatch.setattr(
        video_module,
        "resize_video_tensor",
        lambda video, _resized_height, _resized_width: video,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        video, sample_fps = video_module.load_video_path(
            tmp_path / "clip.mp4",
            resize_executor=executor,
            resize_chunks=8,
        )

    assert video.dtype == torch.float32
    assert torch.equal(video, source.float())
    assert sample_fps == 8.0


def test_local_video_loading_forwards_the_shared_resize_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.touch()
    captured: list[tuple[Executor | None, int]] = []

    def fake_load_video_path(
        path: str | Path,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        *,
        resize_executor: Executor | None = None,
        resize_chunks: int = 1,
    ) -> tuple[torch.Tensor, float]:
        captured.append((resize_executor, resize_chunks))
        return torch.zeros((2, 3, 2, 2)), 1.0

    monkeypatch.setattr(video_module, "load_video_path", fake_load_video_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        videos, sample_fps, audios = asyncio.run(
            ensure_video_list_async(
                [video_path],
                resource_connector=object(),
                resize_executor=executor,
                resize_chunks=8,
            )
        )

    assert len(videos) == 1
    assert sample_fps == [1.0]
    assert audios is None
    assert captured == [(executor, 8)]


def test_url_video_loading_forwards_the_shared_resize_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = MultiModalResourceConnector()
    captured: list[tuple[Executor | None, int]] = []

    async def fake_load_resource_async(
        resource_url: str,
        media_io: VideoMediaIO,
        *,
        timeout: float,
    ) -> tuple[torch.Tensor, float, None]:
        captured.append((media_io.resize_executor, media_io.resize_chunks))
        return torch.zeros((2, 3, 2, 2)), 1.0, None

    monkeypatch.setattr(connector, "load_resource_async", fake_load_resource_async)
    with ThreadPoolExecutor(max_workers=2) as executor:
        video, sample_fps, audio = asyncio.run(
            connector.fetch_video_async(
                "https://example.com/clip.mp4",
                resize_executor=executor,
                resize_chunks=8,
            )
        )

    assert video.shape == (2, 3, 2, 2)
    assert sample_fps == 1.0
    assert audio is None
    assert captured == [(executor, 8)]


def test_data_backed_video_media_io_forwards_the_shared_resize_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[Executor | None, int]] = []

    def fake_load_video_path(
        path: str | Path,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        *,
        resize_executor: Executor | None = None,
        resize_chunks: int = 1,
    ) -> tuple[torch.Tensor, float]:
        captured.append((resize_executor, resize_chunks))
        return torch.zeros((2, 3, 2, 2)), 1.0

    monkeypatch.setattr(video_module, "load_video_path", fake_load_video_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        video, sample_fps, audio = VideoMediaIO(
            resize_executor=executor,
            resize_chunks=8,
        ).load_bytes(b"video-bytes")

    assert video.shape == (2, 3, 2, 2)
    assert sample_fps == 1.0
    assert audio is None
    assert captured == [(executor, 8)]

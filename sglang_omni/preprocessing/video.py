# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic video preprocessing utilities."""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import av
import librosa
import numpy as np
import torch
from qwen_vl_utils import vision_process as qwen_vision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_f

from .base import MediaIO, is_url
from .cache_key import compute_media_cache_key
from .resource_connector import global_thread_pool

logger = logging.getLogger(__name__)


class VideoDecodeError(RuntimeError):
    """Raised when video decoding fails."""


VideoDiagnosticValue = str | int | float | bool


class VideoDiagnosticHook(Protocol):
    """Receive completed, model-agnostic video preprocessing subphases."""

    def __call__(
        self,
        phase: str,
        metadata: Mapping[str, VideoDiagnosticValue],
    ) -> None: ...


@dataclass(frozen=True, kw_only=True)
class VideoResizeGeometry:
    """Resolved frame geometry for the production video resize path."""

    frame_count: int
    source_height: int
    source_width: int
    resized_height: int
    resized_width: int


def decode_video_path(
    path: str | Path,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
    profile_hook: VideoDiagnosticHook | None = None,
) -> tuple[torch.Tensor, float]:
    """Decode a local video with the same backend fallback as serving."""
    path = Path(path)
    element: dict[str, str | int | float] = {"video": str(path)}
    if fps is not None:
        element["fps"] = float(fps)
    else:
        pass
    if max_frames is not None:
        element["max_frames"] = int(max_frames)
    else:
        pass
    if min_pixels is not None:
        element["min_pixels"] = int(min_pixels)
    else:
        pass
    if max_pixels is not None:
        element["max_pixels"] = int(max_pixels)
    else:
        pass
    if total_pixels is not None:
        element["total_pixels"] = int(total_pixels)
    else:
        pass

    backend = qwen_vision.get_video_reader_backend()
    backend_started_ns = time.perf_counter_ns() if profile_hook is not None else 0
    fallback_used = False
    try:
        video, sample_fps = qwen_vision.VIDEO_READER_BACKENDS[backend](element)
        selected_backend = backend
    except Exception as backend_exc:
        if backend == "torchvision":
            raise VideoDecodeError(
                f"Failed to decode video path={path}; torchvision failed with "
                f"{type(backend_exc).__name__}: {backend_exc}"
            ) from backend_exc
        else:
            pass
        logger.warning(f"Video reader {backend} failed, falling back to torchvision")
        fallback_used = True
        try:
            video, sample_fps = qwen_vision.VIDEO_READER_BACKENDS["torchvision"](
                element
            )
            selected_backend = "torchvision"
        except Exception as fallback_exc:
            raise VideoDecodeError(
                f"Failed to decode video path={path}; {backend} failed with "
                f"{type(backend_exc).__name__}: {backend_exc}; "
                f"torchvision failed with {type(fallback_exc).__name__}: "
                f"{fallback_exc}"
            ) from fallback_exc
    else:
        pass

    if profile_hook is not None:
        backend_metadata: dict[str, VideoDiagnosticValue] = {
            "backend": selected_backend,
            "fallback_used": fallback_used,
            "sampled_frame_count": int(video.shape[0]),
            "source_height": int(video.shape[-2]),
            "source_width": int(video.shape[-1]),
            "duration_ms": (time.perf_counter_ns() - backend_started_ns) / 1_000_000.0,
        }
        if sample_fps is not None:
            backend_metadata["sample_fps"] = float(sample_fps)
        else:
            pass
        profile_hook("backend_decode", backend_metadata)
    else:
        pass
    return video, sample_fps


def compute_video_resize_geometry(
    video: torch.Tensor,
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
) -> VideoResizeGeometry:
    """Resolve the serving resize dimensions without touching tensor values."""
    frame_count, _, source_height, source_width = video.shape
    effective_min_pixels = (
        qwen_vision.VIDEO_MIN_PIXELS if min_pixels is None else min_pixels
    )
    effective_total_pixels = (
        qwen_vision.VIDEO_TOTAL_PIXELS if total_pixels is None else total_pixels
    )
    computed_max_pixels = max(
        min(
            qwen_vision.VIDEO_MAX_PIXELS,
            effective_total_pixels / frame_count * qwen_vision.FRAME_FACTOR,
        ),
        int(effective_min_pixels * 1.05),
    )
    effective_max_pixels = computed_max_pixels if max_pixels is None else max_pixels
    effective_max_pixels = min(effective_max_pixels, computed_max_pixels)
    resized_height, resized_width = qwen_vision.smart_resize(
        source_height,
        source_width,
        factor=qwen_vision.IMAGE_FACTOR,
        min_pixels=effective_min_pixels,
        max_pixels=effective_max_pixels,
    )
    return VideoResizeGeometry(
        frame_count=int(frame_count),
        source_height=int(source_height),
        source_width=int(source_width),
        resized_height=int(resized_height),
        resized_width=int(resized_width),
    )


def resize_video_tensor(
    video: torch.Tensor,
    geometry: VideoResizeGeometry,
) -> torch.Tensor:
    """Apply the production torchvision resize without dtype conversion."""
    return tv_f.resize(
        video,
        [geometry.resized_height, geometry.resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def resize_video_tensor_parallel(
    video: torch.Tensor,
    geometry: VideoResizeGeometry,
    *,
    executor: Executor,
    chunks: int,
) -> torch.Tensor:
    """Resize frame chunks with the existing production tensor operation."""
    frame_count = int(video.shape[0])
    if frame_count <= 1 or chunks <= 1:
        return resize_video_tensor(video, geometry)
    else:
        pass

    effective_chunks = min(chunks, frame_count)
    resized_futures: list[Future[torch.Tensor]] = []
    for video_piece in torch.tensor_split(video, effective_chunks, dim=0):
        if video_piece.shape[0] > 0:
            resized_futures.append(
                executor.submit(resize_video_tensor, video_piece, geometry)
            )
        else:
            pass
    resized_pieces = [future.result() for future in resized_futures]
    return torch.cat(resized_pieces, dim=0)


class VideoMediaIO(MediaIO[tuple[torch.Tensor, float, Any | None]]):
    """MediaIO implementation for video files with optional audio extraction."""

    def __init__(
        self,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        image_mode: str = "RGB",
        extract_audio: bool = False,
        audio_target_sr: int = 16000,
        **kwargs,
    ) -> None:
        """Initialize VideoMediaIO.

        Args:
            fps: Target FPS for video loading.
            max_frames: Optional frame cap passed to the video reader backend.
            min_pixels: Optional lower resize budget per frame.
            max_pixels: Optional upper resize budget per frame.
            total_pixels: Optional total video pixel budget.
            image_mode: Target image mode (default: "RGB").
            extract_audio: If True, extract audio from video and return as third element.
            audio_target_sr: Target sample rate for audio extraction (default: 16000).
            **kwargs: Additional arguments (for compatibility with MultiModalResourceConnector).
        """
        super().__init__()
        self.fps = fps
        self.max_frames = max_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.image_mode = image_mode
        self.extract_audio = extract_audio
        self.audio_target_sr = audio_target_sr
        self.kwargs = kwargs

    def load_path(self, filepath: Path) -> tuple[torch.Tensor, float]:
        return load_video_path(
            filepath,
            fps=self.fps,
            max_frames=self.max_frames,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            total_pixels=self.total_pixels,
        )

    def load_bytes(self, data: bytes) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from raw bytes, optionally extracting audio.

        Returns:
            Tuple of (video_tensor, sample_fps, audio_or_None).
            If extract_audio is False, the third element is None.
        """
        # qwen_vision._read_video_torchvision requires a file path,
        # so we need to write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(data)

        try:
            if self.extract_audio:
                # Load video and extract audio from the same file
                video, sample_fps = self.load_path(tmp_path)
                audio = extract_audio_from_path(tmp_path, self.audio_target_sr)
                return video, sample_fps, audio
            else:
                video, sample_fps = self.load_path(tmp_path)
                return video, sample_fps, None
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from base64-encoded data, optionally extracting audio."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from a local file path, optionally extracting audio."""
        if self.extract_audio:
            # Load video and extract audio from the same file
            video, sample_fps = self.load_path(filepath)
            audio = extract_audio_from_path(filepath, self.audio_target_sr)
            return video, sample_fps, audio
        else:
            video, sample_fps = self.load_path(filepath)
            return video, sample_fps, None


async def ensure_video_list_async(
    videos: Any,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
    image_mode: str = "RGB",
    resource_connector: Any | None = None,
    extract_audio: bool = False,
    audio_target_sr: int = 16000,
    profile_hook: VideoDiagnosticHook | None = None,
    resize_executor: Executor | None = None,
    resize_workers: int = 0,
    resize_chunks: int = 1,
) -> tuple[list[Any], list[float] | None, list[Any] | None]:
    """Asynchronously normalize video inputs into a list.

    Args:
        videos: Video input(s) - can be a path, URL, torch Tensor, or list.
        fps: Target FPS for video loading.
        max_frames: Optional frame cap passed to the video reader backend.
        min_pixels: Optional lower resize budget per frame.
        max_pixels: Optional upper resize budget per frame.
        total_pixels: Optional total video pixel budget.
        image_mode: Target image mode (default: "RGB").
        resource_connector: Optional MultiModalResourceConnector instance. If None, uses
                        the global connector.
        extract_audio: If True, extract audio from videos and return as third element.
        audio_target_sr: Target sample rate for audio extraction (default: 16000).
        profile_hook: Optional callback for completed local video subphase durations.
        resize_executor: Optional dedicated executor for frame tensor resize.
        resize_workers: Configured worker count for resize diagnostics.
        resize_chunks: Requested frame chunks for each video resize.

    Returns:
        Tuple of (normalized video list, sample_fps_list or None, extracted_audio_list or None).
        If extract_audio is False, the third element is None.
    """
    if videos is None:
        return [], None, None
    else:
        pass
    if isinstance(videos, list):
        items = videos
    else:
        items = [videos]
    normalized: list[Any] = []
    sample_fps_list: list[float] = []
    extracted_audios: list[Any] = [] if extract_audio else []
    all_paths = True

    # Import here to avoid circular dependency
    if resource_connector is None:
        from .resource_connector import get_global_resource_connector

        resource_connector = get_global_resource_connector()
    else:
        pass

    async def _load_video_with_audio(
        video_item: str | Path, is_url: bool, video_index: int
    ) -> tuple[Any, float, Any | None]:
        """Load video and optionally extract audio."""
        loop = asyncio.get_running_loop()
        indexed_profile_hook: VideoDiagnosticHook | None = None
        if profile_hook is not None:

            def indexed_profile_hook(
                phase: str,
                metadata: Mapping[str, VideoDiagnosticValue],
            ) -> None:
                profile_hook(
                    phase,
                    {"video_index": video_index, **dict(metadata)},
                )

        else:
            pass

        if is_url:
            # Use fetch_video_async for URL videos, similar to fetch_image_async
            return await resource_connector.fetch_video_async(
                str(video_item),
                fps=fps,
                max_frames=max_frames,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                total_pixels=total_pixels,
                image_mode=image_mode,
                extract_audio=extract_audio,
                audio_target_sr=audio_target_sr,
            )
        else:
            # Local file path
            video_path = Path(video_item)
            video_loader = partial(
                load_video_path,
                video_path,
                fps,
                max_frames,
                min_pixels,
                max_pixels,
                total_pixels,
                indexed_profile_hook,
            )
            if resize_executor is not None:
                video_loader = partial(
                    video_loader,
                    resize_executor=resize_executor,
                    resize_workers=resize_workers,
                    resize_chunks=resize_chunks,
                )
            else:
                pass
            if extract_audio:
                video_task = loop.run_in_executor(global_thread_pool, video_loader)
                audio_task = loop.run_in_executor(
                    global_thread_pool,
                    extract_audio_from_path,
                    video_path,
                    audio_target_sr,
                )
                (video, sample_fps), audio = await asyncio.gather(
                    video_task, audio_task
                )
                return video, sample_fps, audio
            else:
                video, sample_fps = await loop.run_in_executor(
                    global_thread_pool, video_loader
                )
                return video, sample_fps, None

    # Collect coroutines for URL and local file items
    coroutines: list[asyncio.Task[tuple[Any, float, Any | None]] | None] = []
    url_indices: list[int] = []

    # First pass: identify items that need loading
    for idx, video_item in enumerate(items):
        if isinstance(video_item, (str, Path)):
            if is_url(video_item):
                # Create coroutine for async URL fetching with optional audio extraction
                coro = _load_video_with_audio(video_item, is_url=True, video_index=idx)
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder for video
                sample_fps_list.append(0.0)  # Placeholder for fps
                if extract_audio:
                    extracted_audios.append(None)  # Placeholder for audio
                else:
                    pass
            elif Path(video_item).exists():
                # Load from local path with optional audio extraction
                coro = _load_video_with_audio(video_item, is_url=False, video_index=idx)
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder for video
                sample_fps_list.append(0.0)  # Placeholder for fps
                if extract_audio:
                    extracted_audios.append(None)  # Placeholder for audio
                else:
                    pass
            else:
                # Path doesn't exist, treat as already processed
                normalized.append(video_item)
                all_paths = False
                if extract_audio:
                    extracted_audios.append(None)
                else:
                    pass
        else:
            # Already processed (torch Tensor, etc.)
            normalized.append(video_item)
            all_paths = False
            if extract_audio:
                extracted_audios.append(None)
            else:
                pass

    # Wait for all loads to complete
    if coroutines:
        results = await asyncio.gather(*coroutines)
        # Fill in the results at the correct indices
        for url_idx, (video, sample_fps, audio) in zip(url_indices, results):
            normalized[url_idx] = video
            sample_fps_list[url_idx] = sample_fps
            if extract_audio:
                extracted_audios[url_idx] = audio
            else:
                pass
    else:
        pass

    if all_paths:
        return (
            normalized,
            sample_fps_list,
            extracted_audios if extract_audio else None,
        )
    else:
        pass
    return normalized, None, extracted_audios if extract_audio else None


def extract_audio_from_path(video_path: Path, target_sr: int) -> np.ndarray | None:
    """Decode the first audio stream to mono float32 at the target sample rate."""
    try:
        with av.open(str(video_path)) as container:
            if not container.streams.audio:
                return None
            else:
                pass
            stream = container.streams.audio[0]
            sample_rate = stream.rate
            # note (MayDomine): convert packed/integer PCM before channel averaging.
            converter = av.AudioResampler(
                format="fltp", layout=stream.layout, rate=sample_rate
            )
            frames = []
            for frame in container.decode(stream):
                frames.extend(
                    output.to_ndarray() for output in converter.resample(frame)
                )
            frames.extend(output.to_ndarray() for output in converter.resample(None))
        if not frames:
            return None
        else:
            pass
        audio = librosa.to_mono(np.concatenate(frames, axis=1))
        return librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    except (av.FFmpegError, ValueError) as exc:
        logger.warning(f"Failed to extract audio from {video_path}: {exc}")
        return None


def load_video_path(
    path: str | Path,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
    profile_hook: VideoDiagnosticHook | None = None,
    *,
    resize_executor: Executor | None = None,
    resize_workers: int = 0,
    resize_chunks: int = 1,
) -> tuple[torch.Tensor, float]:
    """Load a local video into a torch tensor (T, C, H, W) on CPU."""
    video, sample_fps = decode_video_path(
        path,
        fps=fps,
        max_frames=max_frames,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
        profile_hook=profile_hook,
    )

    resize_started_ns = time.perf_counter_ns() if profile_hook is not None else 0
    geometry_started_ns = time.perf_counter_ns() if profile_hook is not None else 0
    geometry = compute_video_resize_geometry(
        video,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
    )
    geometry_duration_ms = (
        (time.perf_counter_ns() - geometry_started_ns) / 1_000_000.0
        if profile_hook is not None
        else 0.0
    )
    if profile_hook is not None:
        profile_hook(
            "resize_geometry",
            {
                "frame_count": geometry.frame_count,
                "source_height": geometry.source_height,
                "source_width": geometry.source_width,
                "resized_height": geometry.resized_height,
                "resized_width": geometry.resized_width,
                "duration_ms": geometry_duration_ms,
            },
        )
    else:
        pass

    input_dtype = str(video.dtype) if profile_hook is not None else ""
    effective_resize_chunks = min(max(resize_chunks, 1), geometry.frame_count)
    parallel_resize_enabled = (
        resize_executor is not None and effective_resize_chunks > 1
    )
    tensor_resize_started_ns = time.perf_counter_ns() if profile_hook is not None else 0
    if parallel_resize_enabled:
        assert resize_executor is not None
        resized_video = resize_video_tensor_parallel(
            video,
            geometry,
            executor=resize_executor,
            chunks=resize_chunks,
        )
    else:
        resized_video = resize_video_tensor(video, geometry)

    tensor_resize_duration_ms = (
        (time.perf_counter_ns() - tensor_resize_started_ns) / 1_000_000.0
        if profile_hook is not None
        else 0.0
    )
    if profile_hook is not None:
        profile_hook(
            "tensor_resize",
            {
                "frame_count": geometry.frame_count,
                "source_height": geometry.source_height,
                "source_width": geometry.source_width,
                "resized_height": geometry.resized_height,
                "resized_width": geometry.resized_width,
                "input_dtype": input_dtype,
                "resize_output_dtype": str(resized_video.dtype),
                "parallel_resize": parallel_resize_enabled,
                "resize_workers": resize_workers,
                "resize_chunks": resize_chunks,
                "effective_chunks": effective_resize_chunks,
                "duration_ms": tensor_resize_duration_ms,
            },
        )
    else:
        pass

    dtype_convert_started_ns = time.perf_counter_ns() if profile_hook is not None else 0
    video = resized_video.float()
    dtype_convert_duration_ms = (
        (time.perf_counter_ns() - dtype_convert_started_ns) / 1_000_000.0
        if profile_hook is not None
        else 0.0
    )
    if profile_hook is not None:
        profile_hook(
            "dtype_convert",
            {
                "input_dtype": str(resized_video.dtype),
                "output_dtype": str(video.dtype),
                "dtype_changed": resized_video.dtype != video.dtype,
                "duration_ms": dtype_convert_duration_ms,
            },
        )
    else:
        pass

    if profile_hook is not None:
        profile_hook(
            "resize_convert",
            {
                "frame_count": geometry.frame_count,
                "source_height": geometry.source_height,
                "source_width": geometry.source_width,
                "resized_height": geometry.resized_height,
                "resized_width": geometry.resized_width,
                "output_dtype": str(video.dtype),
                "parallel_resize": parallel_resize_enabled,
                "resize_workers": resize_workers,
                "resize_chunks": resize_chunks,
                "effective_chunks": effective_resize_chunks,
                "duration_ms": (time.perf_counter_ns() - resize_started_ns)
                / 1_000_000.0,
            },
        )
    else:
        pass
    return video, sample_fps


def build_video_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "pixel_values_videos": hf_inputs.get("pixel_values_videos"),
        "video_grid_thw": hf_inputs.get("video_grid_thw"),
        "video_second_per_grid": hf_inputs.get("video_second_per_grid"),
    }


def compute_video_cache_key(
    videos: Any,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
) -> str | None:
    """Compute cache key from raw video inputs + effective decode params.

    Decode params change the resulting frame count and thus the encoder
    output length. They must be part of the cache key — otherwise an entry
    produced under one (fps, max_frames, pixel-limit) tuple could be
    returned for a request with different params, yielding video_embeds
    whose length no longer matches the prompt placeholders.
    """
    base = compute_media_cache_key(videos, prefix="video")
    if base is None:
        return None
    else:
        pass
    decode_sig = (
        f"|fps={fps}|max_frames={max_frames}"
        f"|min_px={min_pixels}|max_px={max_pixels}|total_px={total_pixels}"
    )
    return base + decode_sig

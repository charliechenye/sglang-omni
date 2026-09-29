from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.preprocessing import video as video_mod
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(inputs: dict) -> StagePayload:
    return StagePayload(
        request_id="video-test",
        request=OmniRequest(inputs=inputs),
        data=None,
    )


def patch_video_resize_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "VIDEO_MIN_PIXELS": 8,
        "VIDEO_TOTAL_PIXELS": 256,
        "VIDEO_MAX_PIXELS": 128,
        "IMAGE_FACTOR": 2,
    }.items():
        monkeypatch.setattr(video_mod.qwen_vision, name, value, raising=False)


class FakeProcessor:
    def __init__(self) -> None:
        self.images = None
        self.audios = None
        self.options = None

    def __call__(self, prompt_text, *, images, audios, return_tensors, **options):
        self.images = images
        self.audios = audios
        self.options = options
        image_count = len(images[0]) if images else 0
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "image_bound": [[torch.tensor([0, 1])] * image_count],
            "pixel_values": [[torch.zeros(1, 2) for _ in range(image_count)]],
            "tgt_sizes": [[torch.tensor([1, 1]) for _ in range(image_count)]],
            "audio_bounds": [[]],
            "audio_feature_lens": [[]],
            "audio_features": [],
        }


async def empty_images(images):
    return []


async def explicit_audios(_audios, *, target_sr):
    return [np.array([0.25, 0.5], dtype=np.float32)] if _audios else []


@pytest.mark.parametrize("use_audio_in_video", [None, False, True])
@pytest.mark.parametrize("explicit_audio", [False, True])
def test_minicpm_preprocessor_uses_only_requested_video_audio(
    monkeypatch,
    use_audio_in_video,
    explicit_audio,
) -> None:
    fake_processor = FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        fake_processor  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    preprocessor.tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        preprocessor,
        "render_chat_template",
        lambda messages, **_: str(messages),
    )

    video = torch.stack(
        [
            torch.zeros((3, 2, 2)),
            torch.ones((3, 2, 2)) * 0.5,
        ]
    )
    captured_video_kwargs = {}

    async def videos(videos, **kwargs):
        captured_video_kwargs.update(kwargs)
        audio = (
            [np.array([1.0, 2.0], dtype=np.float32)]
            if kwargs["extract_audio"]
            else None
        )
        return [video], [2.0], audio

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", empty_images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", explicit_audios)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)
    monkeypatch.setattr(
        preprocessor_mod,
        "compute_video_cache_key",
        lambda *args, **_kwargs: "video-cache",
    )

    payload = make_payload(
        {
            "messages": [{"role": "user", "content": "What happens?"}],
            "videos": ["clip.mp4"],
            **({"audios": ["question.wav"]} if explicit_audio else {}),
            **(
                {"use_audio_in_video": use_audio_in_video}
                if use_audio_in_video is not None
                else {}
            ),
            "video_fps": 2,
            "video_max_frames": 8,
            "video_min_pixels": 128,
            "video_max_pixels": 4096,
            "video_total_pixels": 8192,
        }
    )

    result = asyncio.run(preprocessor(payload))

    assert callable(captured_video_kwargs.pop("profile_hook"))
    assert captured_video_kwargs == {
        "fps": 2,
        "max_frames": 8,
        "min_pixels": 128,
        "max_pixels": 4096,
        "total_pixels": 8192,
        "extract_audio": bool(use_audio_in_video),
        "audio_target_sr": 16000,
    }
    assert len(fake_processor.images[0]) == 2
    assert fake_processor.options == {"max_slice_nums": 1, "use_image_id": False}
    expected_audio_count = int(explicit_audio) + int(bool(use_audio_in_video))
    if expected_audio_count:
        assert len(fake_processor.audios[0]) == expected_audio_count
        if explicit_audio:
            np.testing.assert_array_equal(
                fake_processor.audios[0][0],
                np.array([0.25, 0.5], dtype=np.float32),
            )
        if use_audio_in_video:
            np.testing.assert_array_equal(
                fake_processor.audios[0][-1],
                np.array([1.0, 2.0], dtype=np.float32),
            )
    else:
        assert fake_processor.audios is None
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.count("<image>./</image>") == 2
    assert prompt_text.count("<audio>./</audio>") == expected_audio_count
    assert payload.request.inputs is None


@pytest.mark.parametrize(
    ("with_image", "with_audio", "with_video"),
    [(True, False, False), (False, True, False), (True, True, True)],
)
def test_minicpm_video_options_preserve_other_media(
    monkeypatch, with_image, with_audio, with_video
) -> None:
    fake_processor = FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        fake_processor  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    monkeypatch.setattr(
        preprocessor, "render_chat_template", lambda messages, **_: str(messages)
    )
    image = Image.new("RGB", (2, 2), color="red")
    frame = Image.new("RGB", (2, 2), color="blue")

    async def images(raw_images):
        return [image] if raw_images else []

    async def videos(raw_videos, **kwargs):
        return [[frame]], [1.0], None

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", explicit_audios)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)
    result = asyncio.run(
        preprocessor(
            make_payload(
                {
                    "messages": [{"role": "user", "content": "Describe this."}],
                    "images": [image] if with_image else None,
                    "audios": ["question.wav"] if with_audio else None,
                    "videos": ["clip.mp4"] if with_video else None,
                }
            )
        )
    )

    # The processor has one policy for the whole image list, including mixed inputs.
    assert fake_processor.options == (
        {"max_slice_nums": 1, "use_image_id": False} if with_video else {}
    )
    expected_images = ([image] if with_image else []) + ([frame] if with_video else [])
    if expected_images:
        assert len(fake_processor.images[0]) == len(expected_images)
        for actual, expected in zip(fake_processor.images[0], expected_images):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    else:
        assert fake_processor.images is None
    if with_audio:
        np.testing.assert_array_equal(
            fake_processor.audios[0][0], np.array([0.25, 0.5], dtype=np.float32)
        )
    else:
        assert fake_processor.audios is None
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.count("<image>./</image>") == len(expected_images)
    assert prompt_text.count("<audio>./</audio>") == int(with_audio)


def make_test_preprocessor(
    fake_processor: FakeProcessor, monkeypatch: pytest.MonkeyPatch
) -> MiniCPMOPreprocessor:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = fake_processor  # noqa: leading-underscore
    preprocessor.speech_enabled = False
    monkeypatch.setattr(
        preprocessor, "render_chat_template", lambda messages, **_: str(messages)
    )
    return preprocessor


def test_minicpm_preprocessor_events_cover_video_request(monkeypatch) -> None:
    fake_processor = FakeProcessor()
    preprocessor = make_test_preprocessor(fake_processor, monkeypatch)
    events = []
    monkeypatch.setattr(
        preprocessor_mod, "_emit_event", lambda **event: events.append(event)
    )

    image = Image.new("RGB", (2, 2), color="red")
    frame = Image.new("RGB", (2, 2), color="blue")

    async def images(raw_images):
        return [image] if raw_images else []

    async def videos(raw_videos, **kwargs):
        return [[frame]], [1.0], None

    async def audios(_audios, *, target_sr):
        return []

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", audios)
    monkeypatch.setattr(
        preprocessor_mod, "compute_image_cache_key", lambda _images: "image-cache"
    )
    monkeypatch.setattr(
        preprocessor_mod,
        "compute_video_cache_key",
        lambda _videos, **_kwargs: "video-cache",
    )
    monkeypatch.setattr(
        preprocessor_mod, "compute_audio_cache_key", lambda _audios: "audio-cache"
    )

    payload = make_payload(
        {
            "messages": [{"role": "user", "content": "Describe this."}],
            "images": [image],
            "videos": ["clip.mp4"],
            "video_fps": 2,
            "video_max_frames": 8,
            "video_min_pixels": 128,
            "video_max_pixels": 4096,
            "video_total_pixels": 8192,
        }
    )
    result = asyncio.run(preprocessor(payload))

    assert result.request_id == "video-test"
    assert [event["event_name"] for event in events] == [
        "minicpmo_preprocess_cache_key_start",
        "minicpmo_preprocess_cache_key_end",
        "minicpmo_preprocess_image_load_start",
        "minicpmo_preprocess_image_load_end",
        "minicpmo_preprocess_video_decode_start",
        "minicpmo_preprocess_video_decode_end",
        "minicpmo_preprocess_video_to_images_start",
        "minicpmo_preprocess_video_pil_materialize",
        "minicpmo_preprocess_video_to_images_end",
        "minicpmo_preprocess_prompt_start",
        "minicpmo_preprocess_prompt_end",
        "minicpmo_preprocess_processor_start",
        "minicpmo_preprocess_processor_end",
        "minicpmo_preprocess_payload_start",
        "minicpmo_preprocess_payload_end",
    ]
    assert all(
        event["request_id"] == "video-test" and event["stage"] == "preprocessing"
        for event in events
    )
    assert events[0]["metadata"] == {"has_images": True, "has_videos": True}
    assert events[4]["metadata"] == {
        "video_count": 1,
        "video_fps": 2,
        "video_max_frames": 8,
        "video_min_pixels": 128,
        "video_max_pixels": 4096,
        "video_total_pixels": 8192,
    }
    assert events[7]["metadata"]["video_index"] == 0
    assert events[7]["metadata"]["frame_count"] == 1
    assert events[8]["metadata"] == {"decoded_frame_count": 1}
    assert events[10]["metadata"] == {"num_images": 2, "num_audios": 0}
    assert events[12]["metadata"] == {"num_images": 2}
    assert events[14]["metadata"] == {
        "input_token_count": 3,
        "image_slice_count": 2,
    }
    for event in events:
        assert all(
            isinstance(value, (bool, float, int, str))
            for value in event.get("metadata", {}).values()
        )


def test_minicpm_preprocessor_audio_events_cover_audio_work(monkeypatch) -> None:
    fake_processor = FakeProcessor()
    preprocessor = make_test_preprocessor(fake_processor, monkeypatch)
    events = []
    monkeypatch.setattr(
        preprocessor_mod, "_emit_event", lambda **event: events.append(event)
    )

    async def images(_raw_images):
        return []

    async def audios(raw_audios, *, target_sr):
        return [np.array([0.25, 0.5], dtype=np.float32)] if raw_audios else []

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", audios)
    monkeypatch.setattr(
        preprocessor_mod, "compute_audio_cache_key", lambda _audios: "audio-cache"
    )

    result = asyncio.run(
        preprocessor(
            make_payload(
                {
                    "messages": [{"role": "user", "content": "Transcribe this."}],
                    "audios": ["question.wav"],
                }
            )
        )
    )

    assert result.request_id == "video-test"
    assert [event["event_name"] for event in events] == [
        "minicpmo_preprocess_cache_key_start",
        "minicpmo_preprocess_cache_key_end",
        "minicpmo_preprocess_audio_start",
        "minicpmo_preprocess_audio_end",
        "minicpmo_preprocess_prompt_start",
        "minicpmo_preprocess_prompt_end",
        "minicpmo_preprocess_processor_start",
        "minicpmo_preprocess_processor_end",
        "minicpmo_preprocess_payload_start",
        "minicpmo_preprocess_payload_end",
    ]
    assert all(event["stage"] == "preprocessing" for event in events)
    assert events[3]["metadata"] == {"num_audios": 1}
    assert events[5]["metadata"] == {"num_images": 0, "num_audios": 1}


def test_minicpm_image_only_preprocessing_needs_no_profiler_setup(monkeypatch) -> None:
    fake_processor = FakeProcessor()
    preprocessor = make_test_preprocessor(fake_processor, monkeypatch)
    image = Image.new("RGB", (2, 2), color="red")

    async def images(raw_images):
        return [image] if raw_images else []

    async def audios(_audios, *, target_sr):
        return []

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", audios)
    monkeypatch.setattr(
        preprocessor_mod, "compute_image_cache_key", lambda _images: "image-cache"
    )
    monkeypatch.setattr(
        preprocessor_mod, "compute_audio_cache_key", lambda _audios: "audio-cache"
    )

    result = asyncio.run(
        preprocessor(
            make_payload(
                {
                    "messages": [{"role": "user", "content": "Describe this."}],
                    "images": [image],
                }
            )
        )
    )

    assert fake_processor.options == {}
    assert len(fake_processor.images[0]) == 1
    assert result.data["prompt"]["prompt_text"].count("<image>./</image>") == 1
    assert result.request.inputs is None


def test_load_video_path_profiles_backend_and_resize_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, video_mod.VideoDiagnosticValue]]] = []
    source = torch.zeros((2, 3, 4, 6), dtype=torch.uint8)
    patch_video_resize_constants(monkeypatch)

    def backend(_element: dict[str, object]) -> tuple[torch.Tensor, float]:
        return source, 12.5

    def smart_resize(
        height: int,
        width: int,
        **_kwargs: object,
    ) -> tuple[int, int]:
        assert (height, width) == (4, 6)
        return 8, 12

    def resize(
        video: torch.Tensor,
        size: list[int],
        **_kwargs: object,
    ) -> torch.Tensor:
        return torch.zeros(
            (video.shape[0], video.shape[1], size[0], size[1]),
            dtype=video.dtype,
        )

    monkeypatch.setattr(
        video_mod.qwen_vision, "get_video_reader_backend", lambda: "fake"
    )
    monkeypatch.setitem(video_mod.qwen_vision.VIDEO_READER_BACKENDS, "fake", backend)
    monkeypatch.setattr(video_mod.qwen_vision, "smart_resize", smart_resize)
    monkeypatch.setattr(video_mod.tv_f, "resize", resize)

    def collect_profile(
        phase: str,
        metadata: dict[str, video_mod.VideoDiagnosticValue],
    ) -> None:
        events.append((phase, dict(metadata)))

    video, sample_fps = video_mod.load_video_path(
        tmp_path / "clip.mp4",
        min_pixels=8,
        max_pixels=128,
        total_pixels=256,
        profile_hook=collect_profile,
    )

    assert video.shape == (2, 3, 8, 12)
    assert video.dtype == torch.float32
    assert sample_fps == 12.5
    assert [phase for phase, _metadata in events] == [
        "backend_decode",
        "resize_geometry",
        "tensor_resize",
        "dtype_convert",
        "resize_convert",
    ]
    backend_metadata = events[0][1]
    assert backend_metadata["backend"] == "fake"
    assert backend_metadata["sampled_frame_count"] == 2
    assert backend_metadata["source_height"] == 4
    assert backend_metadata["source_width"] == 6
    assert backend_metadata["sample_fps"] == 12.5
    assert isinstance(backend_metadata["duration_ms"], float)
    geometry_metadata = events[1][1]
    assert geometry_metadata["frame_count"] == 2
    assert geometry_metadata["source_height"] == 4
    assert geometry_metadata["source_width"] == 6
    assert geometry_metadata["resized_height"] == 8
    assert geometry_metadata["resized_width"] == 12
    tensor_resize_metadata = events[2][1]
    assert tensor_resize_metadata["input_dtype"] == "torch.uint8"
    assert tensor_resize_metadata["resize_output_dtype"] == "torch.uint8"
    dtype_metadata = events[3][1]
    assert dtype_metadata["input_dtype"] == "torch.uint8"
    assert dtype_metadata["output_dtype"] == "torch.float32"
    assert dtype_metadata["dtype_changed"] is True
    resize_metadata = events[4][1]
    assert resize_metadata["frame_count"] == 2
    assert resize_metadata["resized_height"] == 8
    assert resize_metadata["resized_width"] == 12
    assert resize_metadata["output_dtype"] == "torch.float32"
    assert all(
        isinstance(metadata["duration_ms"], float) and metadata["duration_ms"] >= 0
        for _phase, metadata in events
    )
    assert resize_metadata["duration_ms"] >= max(
        metadata["duration_ms"] for _phase, metadata in events[1:4]
    )


def test_load_video_path_dtype_conversion_is_a_noop_for_float32(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, video_mod.VideoDiagnosticValue]]] = []
    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    patch_video_resize_constants(monkeypatch)

    monkeypatch.setattr(
        video_mod.qwen_vision, "get_video_reader_backend", lambda: "fake"
    )
    monkeypatch.setitem(
        video_mod.qwen_vision.VIDEO_READER_BACKENDS,
        "fake",
        lambda _element: (source, 8.0),
    )
    monkeypatch.setattr(
        video_mod.qwen_vision,
        "smart_resize",
        lambda _height, _width, **_kwargs: (2, 2),
    )
    monkeypatch.setattr(video_mod.tv_f, "resize", lambda video, _size, **_kwargs: video)

    def collect_profile(
        phase: str,
        metadata: dict[str, video_mod.VideoDiagnosticValue],
    ) -> None:
        events.append((phase, dict(metadata)))

    video, sample_fps = video_mod.load_video_path(
        tmp_path / "float32.mp4", profile_hook=collect_profile
    )
    unprofiled_video, unprofiled_sample_fps = video_mod.load_video_path(
        tmp_path / "float32-unprofiled.mp4"
    )

    assert torch.equal(video, source)
    assert sample_fps == 8.0
    assert torch.equal(unprofiled_video, source)
    assert unprofiled_sample_fps == 8.0
    dtype_metadata = next(
        metadata for phase, metadata in events if phase == "dtype_convert"
    )
    assert dtype_metadata == {
        "input_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "dtype_changed": False,
        "duration_ms": dtype_metadata["duration_ms"],
    }


def test_load_video_path_profiles_successful_torchvision_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, video_mod.VideoDiagnosticValue]]] = []
    source = torch.zeros((1, 3, 4, 4), dtype=torch.uint8)
    patch_video_resize_constants(monkeypatch)

    def primary_backend(_element: dict[str, object]) -> tuple[torch.Tensor, float]:
        raise RuntimeError("primary backend failed")

    def fallback_backend(_element: dict[str, object]) -> tuple[torch.Tensor, float]:
        return source, 6.0

    monkeypatch.setattr(
        video_mod.qwen_vision, "get_video_reader_backend", lambda: "fake"
    )
    monkeypatch.setitem(
        video_mod.qwen_vision.VIDEO_READER_BACKENDS,
        "fake",
        primary_backend,
    )
    monkeypatch.setitem(
        video_mod.qwen_vision.VIDEO_READER_BACKENDS,
        "torchvision",
        fallback_backend,
    )
    monkeypatch.setattr(
        video_mod.qwen_vision,
        "smart_resize",
        lambda _height, _width, **_kwargs: (4, 4),
    )
    monkeypatch.setattr(
        video_mod.tv_f,
        "resize",
        lambda video, _size, **_kwargs: video,
    )

    def collect_profile(
        phase: str,
        metadata: dict[str, video_mod.VideoDiagnosticValue],
    ) -> None:
        events.append((phase, dict(metadata)))

    video_mod.load_video_path(tmp_path / "fallback.mp4", profile_hook=collect_profile)

    backend_events = [event for event in events if event[0] == "backend_decode"]
    assert len(backend_events) == 1
    assert backend_events[0][1]["backend"] == "torchvision"
    assert backend_events[0][1]["fallback_used"] is True

    events.clear()

    def broken_fallback(_element: dict[str, object]) -> tuple[torch.Tensor, float]:
        raise RuntimeError("fallback failed")

    monkeypatch.setitem(
        video_mod.qwen_vision.VIDEO_READER_BACKENDS,
        "torchvision",
        broken_fallback,
    )
    with pytest.raises(video_mod.VideoDecodeError):
        video_mod.load_video_path(tmp_path / "failed.mp4", profile_hook=collect_profile)
    assert events == []


def test_video_to_images_profiles_tensor_and_pil_materialization() -> None:
    events: list[tuple[str, dict[str, video_mod.VideoDiagnosticValue]]] = []
    video = torch.stack(
        [
            torch.zeros((3, 2, 2), dtype=torch.float32),
            torch.ones((3, 2, 2), dtype=torch.float32),
        ]
    )

    def collect_profile(
        phase: str,
        metadata: dict[str, video_mod.VideoDiagnosticValue],
    ) -> None:
        events.append((phase, dict(metadata)))

    images = preprocessor_mod.video_to_images(
        video,
        profile_hook=collect_profile,
        video_index=3,
    )

    assert len(images) == 2
    assert images[0].mode == "RGB"
    assert images[1].mode == "RGB"
    assert [phase for phase, _metadata in events] == [
        "tensor_prepare",
        "pil_materialize",
    ]
    assert events[0][1]["video_index"] == 3
    assert events[0][1]["frame_count"] == 2
    assert events[0][1]["output_dtype"] == "torch.uint8"
    assert events[1][1]["video_index"] == 3
    assert events[1][1]["output_dtype"] == "PIL.RGB"
    assert all(
        isinstance(metadata["duration_ms"], float) for _phase, metadata in events
    )

    assert len(preprocessor_mod.video_to_images(video)) == 2


def test_video_to_images_existing_pil_input_skips_tensor_prepare() -> None:
    events: list[tuple[str, dict[str, video_mod.VideoDiagnosticValue]]] = []
    frame = Image.new("L", (3, 2), color=128)

    def collect_profile(
        phase: str,
        metadata: dict[str, video_mod.VideoDiagnosticValue],
    ) -> None:
        events.append((phase, dict(metadata)))

    images = preprocessor_mod.video_to_images(
        [frame], profile_hook=collect_profile, video_index=1
    )

    assert len(images) == 1
    assert images[0].mode == "RGB"
    assert [phase for phase, _metadata in events] == ["pil_materialize"]
    assert events[0][1]["video_index"] == 1
    assert events[0][1]["input_kind"] == "pil"


def test_ensure_video_list_profiles_multiple_local_videos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, video_mod.VideoDiagnosticValue]]] = []
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    first_path.touch()
    second_path.touch()

    def fake_load_video_path(
        path: str | Path,
        _fps: float | None = None,
        _max_frames: int | None = None,
        _min_pixels: int | None = None,
        _max_pixels: int | None = None,
        _total_pixels: int | None = None,
        profile_hook: video_mod.VideoDiagnosticHook | None = None,
    ) -> tuple[torch.Tensor, float]:
        if profile_hook is not None:
            first_video = Path(path).name == "first.mp4"
            phase_durations = {
                "backend_decode": 10.0 if first_video else 20.0,
                "resize_geometry": 1.0 if first_video else 2.0,
                "tensor_resize": 3.0 if first_video else 4.0,
                "dtype_convert": 5.0 if first_video else 6.0,
                "resize_convert": 9.0 if first_video else 12.0,
            }
            for phase, duration_ms in phase_durations.items():
                profile_hook(
                    phase,
                    {
                        "backend": "fake",
                        "sampled_frame_count": 1,
                        "source_height": 2,
                        "source_width": 2,
                        "sample_fps": 1.0,
                        "duration_ms": duration_ms,
                    },
                )
        else:
            pass
        return torch.zeros((1, 3, 2, 2)), 1.0

    monkeypatch.setattr(video_mod, "load_video_path", fake_load_video_path)

    def collect_profile(
        phase: str,
        metadata: dict[str, video_mod.VideoDiagnosticValue],
    ) -> None:
        events.append((phase, dict(metadata)))

    videos, sample_fps, audios = asyncio.run(
        video_mod.ensure_video_list_async(
            [first_path, second_path],
            resource_connector=SimpleNamespace(),
            profile_hook=collect_profile,
        )
    )

    assert len(videos) == 2
    assert sample_fps == [1.0, 1.0]
    assert audios is None
    for phase, expected_durations in {
        "backend_decode": (10.0, 20.0),
        "resize_geometry": (1.0, 2.0),
        "tensor_resize": (3.0, 4.0),
        "dtype_convert": (5.0, 6.0),
        "resize_convert": (9.0, 12.0),
    }.items():
        assert sorted(
            (metadata["video_index"], metadata["duration_ms"])
            for event_phase, metadata in events
            if event_phase == phase
        ) == [(0, expected_durations[0]), (1, expected_durations[1])]

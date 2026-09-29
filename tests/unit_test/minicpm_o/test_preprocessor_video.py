from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(inputs: dict) -> StagePayload:
    return StagePayload(
        request_id="video-test",
        request=OmniRequest(inputs=inputs),
        data=None,
    )


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
    assert events[7]["metadata"] == {"decoded_frame_count": 1}
    assert events[9]["metadata"] == {"num_images": 2, "num_audios": 0}
    assert events[11]["metadata"] == {"num_images": 2}
    assert events[13]["metadata"] == {
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

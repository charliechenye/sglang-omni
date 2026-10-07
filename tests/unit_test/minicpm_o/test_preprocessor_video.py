from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchFeature

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


class FakeCheckpointImageProcessor:
    def __init__(
        self,
        *,
        include_unknown_field: bool = False,
        failure_frame: int | None = None,
        blocking_frame: int | None = None,
        blocking_started: threading.Event | None = None,
        blocking_release: threading.Event | None = None,
        blocking_finished: threading.Event | None = None,
    ) -> None:
        self.include_unknown_field = include_unknown_field
        self.failure_frame = failure_frame
        self.blocking_frame = blocking_frame
        self.blocking_started = blocking_started
        self.blocking_release = blocking_release
        self.blocking_finished = blocking_finished
        self.calls: list[list[int]] = []
        self.call_thread_names: list[str] = []
        self.call_arguments: list[
            tuple[bool, str | None, dict[str, bool | float | int | str | None]]
        ] = []

    def preprocess(
        self,
        images: list[list[int]] | list[int],
        do_pad: bool = True,
        return_tensors: str | None = None,
        **options: bool | float | int | str | None,
    ) -> BatchFeature:
        if isinstance(images[0], list):
            frame_ids = list(images[0])
        else:
            frame_ids = list(images)

        self.calls.append(frame_ids)
        self.call_thread_names.append(threading.current_thread().name)
        self.call_arguments.append((do_pad, return_tensors, options))
        if self.failure_frame is not None and self.failure_frame in frame_ids:
            raise RuntimeError("checkpoint preprocessing failed")
        else:
            pass
        if (
            self.blocking_frame is not None
            and self.blocking_frame in frame_ids
            and self.blocking_started is not None
            and self.blocking_release is not None
            and self.blocking_finished is not None
        ):
            self.blocking_started.set()
            self.blocking_release.wait(timeout=5)
            self.blocking_finished.set()
        else:
            pass

        data = {
            "pixel_values": [
                [
                    torch.tensor([[frame_id]], dtype=torch.float32)
                    for frame_id in frame_ids
                ]
            ],
            "image_sizes": [
                [
                    torch.tensor([frame_id, frame_id], dtype=torch.long)
                    for frame_id in frame_ids
                ]
            ],
            "tgt_sizes": [
                torch.tensor(
                    [[frame_id, frame_id] for frame_id in frame_ids],
                    dtype=torch.long,
                )
            ],
        }
        if self.include_unknown_field:
            data["future_field"] = [torch.tensor([1], dtype=torch.long)]
        else:
            pass
        return BatchFeature(data=data)


class FakeCheckpointProcessor:
    def __init__(self, image_processor: FakeCheckpointImageProcessor) -> None:
        self.image_processor = image_processor


def make_processor_with_parallel_image_processor(
    monkeypatch: pytest.MonkeyPatch,
    image_processor: FakeCheckpointImageProcessor,
    video_frame_executor: ThreadPoolExecutor | None,
    video_frame_workers: int,
) -> MiniCPMOPreprocessor:
    checkpoint_processor = FakeCheckpointProcessor(image_processor)
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = None  # noqa: leading-underscore  # production name
    preprocessor.model_dir = "unused"
    preprocessor.video_frame_executor = video_frame_executor
    preprocessor.video_frame_workers = video_frame_workers
    monkeypatch.setattr(
        preprocessor_mod.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: checkpoint_processor,
    )
    assert preprocessor.processor is checkpoint_processor
    return preprocessor


def assert_processor_outputs_equal(
    expected: BatchFeature, actual: BatchFeature
) -> None:
    assert expected.keys() == actual.keys()
    expected_pixel_values = expected["pixel_values"][0]
    actual_pixel_values = actual["pixel_values"][0]
    assert len(expected_pixel_values) == len(actual_pixel_values)
    for expected_value, actual_value in zip(expected_pixel_values, actual_pixel_values):
        assert torch.equal(expected_value, actual_value)

    expected_image_sizes = expected["image_sizes"][0]
    actual_image_sizes = actual["image_sizes"][0]
    assert len(expected_image_sizes) == len(actual_image_sizes)
    for expected_value, actual_value in zip(expected_image_sizes, actual_image_sizes):
        assert torch.equal(expected_value, actual_value)
    assert torch.equal(expected["tgt_sizes"][0], actual["tgt_sizes"][0])


def test_parallel_video_image_processor_is_exact_ordered_and_reuses_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_processor = FakeCheckpointImageProcessor()
    with ThreadPoolExecutor(
        max_workers=3, thread_name_prefix="shared-video-pool"
    ) as executor:
        serial_preprocess = image_processor.preprocess
        serial = serial_preprocess(
            [[0, 1, 2, 3, 4]],
            do_pad=False,
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )
        image_processor.calls.clear()
        image_processor.call_thread_names.clear()
        image_processor.call_arguments.clear()
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=3,
        )

        parallel = preprocessor.processor.image_processor.preprocess(
            [[0, 1, 2, 3, 4]],
            do_pad=False,
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )

    assert sorted(image_processor.calls, key=lambda frame_ids: frame_ids[0]) == [
        [0, 1],
        [2, 3],
        [4],
    ]
    assert all(
        thread_name.startswith("shared-video-pool")
        for thread_name in image_processor.call_thread_names
    )
    assert len(image_processor.call_arguments) == 3
    assert all(
        arguments == (False, "pt", {"max_slice_nums": 1, "use_image_id": False})
        for arguments in image_processor.call_arguments
    )
    assert_processor_outputs_equal(serial, parallel)


def test_parallel_video_image_processor_is_exact_for_mixed_image_and_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_processor = FakeCheckpointImageProcessor()
    mixed_images = [[100, 0, 1, 2, 3]]
    with ThreadPoolExecutor(
        max_workers=3, thread_name_prefix="shared-video-pool"
    ) as executor:
        serial = image_processor.preprocess(
            mixed_images,
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )
        image_processor.calls.clear()
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=3,
        )
        parallel = preprocessor.processor.image_processor.preprocess(
            mixed_images,
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )

    assert sorted(image_processor.calls, key=lambda frame_ids: frame_ids[0]) == [
        [100, 0],
        [1, 2],
        [3],
    ]
    assert_processor_outputs_equal(serial, parallel)
    assert [size[0].item() for size in parallel["image_sizes"][0]] == [
        100,
        0,
        1,
        2,
        3,
    ]


def test_parallel_video_image_processor_submits_no_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_processor = FakeCheckpointImageProcessor()
    with ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="shared-video-pool"
    ) as executor:
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=8,
        )
        preprocessor.processor.image_processor.preprocess(
            [[0, 1, 2]],
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )

    assert sorted(image_processor.calls, key=lambda frame_ids: frame_ids[0]) == [
        [0],
        [1],
        [2],
    ]


@pytest.mark.parametrize(
    ("executor_enabled", "video_frame_workers", "images"),
    [
        (True, 8, [[0]]),
        (False, 8, [[0, 1]]),
        (True, 1, [[0, 1]]),
        (True, 8, [0, 1]),
    ],
)
def test_parallel_video_image_processor_serial_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    executor_enabled: bool,
    video_frame_workers: int,
    images: list[list[int]] | list[int],
) -> None:
    image_processor = FakeCheckpointImageProcessor()
    executor = (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="shared-video-pool")
        if executor_enabled
        else None
    )
    try:
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=video_frame_workers,
        )
        preprocessor.processor.image_processor.preprocess(
            images,
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )
    finally:
        if executor is not None:
            executor.shutdown()
        else:
            pass

    expected_frames = images[0] if isinstance(images[0], list) else images
    assert image_processor.calls == [expected_frames]


def test_parallel_video_image_processor_falls_back_for_unknown_output_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_processor = FakeCheckpointImageProcessor(include_unknown_field=True)
    with ThreadPoolExecutor(
        max_workers=3, thread_name_prefix="shared-video-pool"
    ) as executor:
        serial_preprocess = image_processor.preprocess
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=3,
        )
        result = preprocessor.processor.image_processor.preprocess(
            [[0, 1, 2]],
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )
        expected = serial_preprocess(
            [[0, 1, 2]],
            return_tensors="pt",
            max_slice_nums=1,
            use_image_id=False,
        )

    assert [0, 1, 2] in image_processor.calls
    assert_processor_outputs_equal(expected, result)
    assert "future_field" in result
    assert torch.equal(result["future_field"][0], torch.tensor([1]))


def test_parallel_video_image_processor_drains_futures_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking_started = threading.Event()
    blocking_release = threading.Event()
    blocking_finished = threading.Event()
    image_processor = FakeCheckpointImageProcessor(
        failure_frame=0,
        blocking_frame=1,
        blocking_started=blocking_started,
        blocking_release=blocking_release,
        blocking_finished=blocking_finished,
    )
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="shared-video-pool"
    ) as executor:
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=2,
        )
        errors: list[Exception] = []

        def run_processor() -> None:
            try:
                preprocessor.processor.image_processor.preprocess(
                    [[0, 1]],
                    return_tensors="pt",
                    max_slice_nums=1,
                    use_image_id=False,
                )
            except Exception as exception:
                errors.append(exception)

        processing_thread = threading.Thread(target=run_processor)
        processing_thread.start()
        assert blocking_started.wait(timeout=2)
        assert not blocking_finished.is_set()
        blocking_release.set()
        processing_thread.join(timeout=2)

    assert not processing_thread.is_alive()
    assert len(errors) == 1
    assert str(errors[0]) == "checkpoint preprocessing failed"
    assert blocking_finished.is_set()


@pytest.mark.parametrize(
    ("executor_enabled", "video_frame_workers"),
    [(False, 8), (True, 1)],
)
def test_processor_property_leaves_checkpoint_processor_unwrapped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    executor_enabled: bool,
    video_frame_workers: int,
) -> None:
    image_processor = FakeCheckpointImageProcessor()
    executor = (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="shared-video-pool")
        if executor_enabled
        else None
    )
    original_preprocess = image_processor.preprocess.__func__
    try:
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=video_frame_workers,
        )
        assert (
            preprocessor.processor.image_processor.preprocess.__func__
            is original_preprocess
        )
    finally:
        if executor is not None:
            executor.shutdown()
        else:
            pass


def test_processor_property_wraps_checkpoint_image_processor_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_processor = FakeCheckpointImageProcessor()
    with ThreadPoolExecutor(max_workers=2) as executor:
        preprocessor = make_processor_with_parallel_image_processor(
            monkeypatch,
            image_processor,
            video_frame_executor=executor,
            video_frame_workers=2,
        )
        wrapped_preprocess = preprocessor.processor.image_processor.preprocess
        assert preprocessor.processor.image_processor.preprocess is wrapped_preprocess


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
    preprocessor.video_frame_executor = None
    preprocessor.video_frame_workers = 1
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
        "resize_executor": None,
        "resize_workers": 1,
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
    preprocessor.video_frame_executor = None
    preprocessor.video_frame_workers = 1
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


@pytest.mark.parametrize("changed", ["image", "video"])
def test_minicpm_visual_cache_key_tracks_decoded_content(monkeypatch, changed) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        FakeProcessor()  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    preprocessor.video_frame_executor = None
    preprocessor.video_frame_workers = 1
    monkeypatch.setattr(
        preprocessor, "render_chat_template", lambda messages, **_: str(messages)
    )
    media = {"image": Image.new("RGB", (2, 2), "red"), "video": torch.zeros(2, 3, 2, 2)}

    async def images(raw_images):
        return [media["image"]]

    async def videos(raw_videos, **kwargs):
        return [media["video"]], [1.0], None

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)

    def cache_key(name: str = "same") -> str:
        inputs = {
            "messages": [{"role": "user", "content": "Describe this."}],
            "images": [f"https://media.invalid/{name}.png"],
            "videos": [f"https://media.invalid/{name}.mp4"],
        }
        data = asyncio.run(preprocessor(make_payload(inputs))).data
        key = data["encoder_inputs"]["image_encoder"]["cache_key"]
        assert data["mm_inputs"]["image"]["cache_key"] == key
        return key

    before = cache_key()
    assert cache_key() == before
    # New content behind the same URL must not reuse the previous entry.
    media[changed] = (
        Image.new("RGB", (2, 2), "blue")
        if changed == "image"
        else torch.ones(2, 3, 2, 2)
    )
    after = cache_key()
    assert after != before
    # Identical content at another address shares the entry.
    assert cache_key("other") == after

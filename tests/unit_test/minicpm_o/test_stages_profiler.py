from __future__ import annotations

from concurrent.futures import Executor

import pytest
import torch

from sglang_omni.models.minicpm_o import stages as stages_mod
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class FakeEncoder:
    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, **model_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.call_count += 1
        return {"features": torch.ones(1)}


class FakePreprocessor:
    def __init__(
        self,
        _model_path: str,
        *,
        speech_enabled: bool,
        resize_executor: Executor | None,
        resize_workers: int,
        resize_chunks: int,
    ) -> None:
        self.speech_enabled = speech_enabled
        self.resize_executor = resize_executor
        self.resize_workers = resize_workers
        self.resize_chunks = resize_chunks
        self.shutdown_count = 0

    async def __call__(self, payload: StagePayload) -> StagePayload:
        return payload

    def shutdown(self) -> None:
        self.shutdown_count += 1
        if self.resize_executor is not None:
            self.resize_executor.shutdown(wait=False, cancel_futures=True)
        else:
            pass


def make_encoder_payload(stage_name: str, request_id: str) -> StagePayload:
    state = MiniCPMOPipelineState(
        encoder_inputs={
            stage_name: {
                "model_input": torch.ones(1),
                "cache_key": "shared-cache-key",
            }
        }
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None),
        data=state.to_dict(),
    )


@pytest.mark.parametrize(
    ("stage_name", "modality"),
    [("image_encoder", "image"), ("audio_encoder", "audio")],
)
def test_minicpm_encoder_events_only_cover_uncached_execution(
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
    modality: str,
) -> None:
    events = []
    monkeypatch.setattr(stages_mod, "emit_event", lambda **event: events.append(event))
    encoder = FakeEncoder()
    scheduler = stages_mod.create_encoder_executor(encoder, stage_name=stage_name)

    scheduler.fn(make_encoder_payload(stage_name, "uncached-request"))
    assert encoder.call_count == 1
    assert [event["event_name"] for event in events] == [
        "encoder_start",
        "encoder_end",
    ]
    assert [event["request_id"] for event in events] == [
        "uncached-request",
        "uncached-request",
    ]
    assert all(event["stage"] is None for event in events)
    assert all(
        event["metadata"] == {"modality": modality, "batch_size": 1} for event in events
    )

    scheduler.fn(make_encoder_payload(stage_name, "cached-request"))
    assert encoder.call_count == 1
    assert [event["request_id"] for event in events] == [
        "uncached-request",
        "uncached-request",
    ]


@pytest.mark.parametrize(
    ("resize_workers", "has_resize_executor"),
    [(0, False), (1, False), (32, True)],
)
def test_minicpm_preprocessing_resize_pool_is_shared_and_lifecycle_bound(
    monkeypatch: pytest.MonkeyPatch,
    resize_workers: int,
    has_resize_executor: bool,
) -> None:
    created: list[FakePreprocessor] = []

    def make_preprocessor(
        _model_path: str,
        *,
        speech_enabled: bool,
        resize_executor: Executor | None,
        resize_workers: int,
        resize_chunks: int,
    ) -> FakePreprocessor:
        preprocessor = FakePreprocessor(
            _model_path,
            speech_enabled=speech_enabled,
            resize_executor=resize_executor,
            resize_workers=resize_workers,
            resize_chunks=resize_chunks,
        )
        created.append(preprocessor)
        return preprocessor

    monkeypatch.setattr(stages_mod, "MiniCPMOPreprocessor", make_preprocessor)
    scheduler = stages_mod.create_preprocessing_executor(
        "model-path",
        max_concurrency=4,
        video_resize_workers=resize_workers,
        video_resize_chunks=8,
    )

    assert len(created) == 1
    preprocessor = created[0]
    assert (preprocessor.resize_executor is not None) is has_resize_executor
    assert preprocessor.resize_workers == resize_workers
    assert preprocessor.resize_chunks == 8
    if has_resize_executor:
        assert (
            preprocessor.resize_executor._max_workers == 32
        )  # noqa: leading-underscore  # executor implementation detail
    else:
        pass

    scheduler.stop()
    assert preprocessor.shutdown_count == 1

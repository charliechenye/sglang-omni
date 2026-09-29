from __future__ import annotations

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
    monkeypatch.setattr(stages_mod, "_emit_event", lambda **event: events.append(event))
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

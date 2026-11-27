# SPDX-License-Identifier: Apache-2.0
"""Request state and tracking."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.realtime import RealtimeTranscriptionContext


class RequestState(Enum):
    """State of a request in the pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RequestInfo:
    """Tracking info for a request in the coordinator."""

    request_id: str
    state: RequestState = RequestState.PENDING
    current_stage: str | None = None
    terminal_stages: set[str] | None = None
    # Note (Junnan Li): A session's close and its persistent read bypass admission and hold no in-flight slot.
    counts_toward_admission: bool = True
    result: object = None
    error: str | None = None


@dataclass
class OmniRequest:
    """User-facing request with inputs and parameters."""

    inputs: object
    params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    realtime_transcription_context: RealtimeTranscriptionContext | None = None

    def to_dict(self) -> dict[str, object]:
        data = {
            "_type": "OmniRequest",
            "inputs": self.inputs,
            "params": self.params,
            "metadata": self.metadata,
        }
        if self.realtime_transcription_context is not None:
            data["realtime_transcription_context"] = (
                self.realtime_transcription_context.to_dict()
            )
        else:
            pass
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OmniRequest":
        context = data.get("realtime_transcription_context")
        if isinstance(context, Mapping):
            realtime_context = RealtimeTranscriptionContext.from_dict(context)
        else:
            realtime_context = None
        return cls(
            inputs=data.get("inputs"),
            params=data.get("params", {}),
            metadata=data.get("metadata", {}),
            realtime_transcription_context=realtime_context,
        )


@dataclass
class StagePayload:
    """Payload passed between stages with request context."""

    request_id: str
    request: OmniRequest
    data: object
    # Scheduler-local stream ingress state. These fields intentionally stay
    # out of to_dict(); they are rebuilt by the receiving scheduler and never
    # form part of the inter-stage wire contract.
    prefetched_chunks: list[StreamItem] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    prefetched_stream_done: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "_type": "StagePayload",
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StagePayload":
        request = data.get("request", {})
        if isinstance(request, dict) and request.get("_type") == "OmniRequest":
            request_obj = OmniRequest.from_dict(request)
        else:
            request_obj = OmniRequest.from_dict(request)
        return cls(
            request_id=data.get("request_id", ""),
            request=request_obj,
            data=data.get("data"),
        )


@dataclass(frozen=True)
class RoutedPayloads:
    """A stage result whose route sends each named stage its own payload.

    keep_request_state keeps the request live at the routing stage, whose
    other work for it may already be queued there.
    """

    payloads: dict[str, StagePayload]
    keep_request_state: bool = False

# SPDX-License-Identifier: Apache-2.0
"""Trusted request context created by the realtime transcription server."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeTranscriptionContext:
    request_id: str
    is_final: bool
    previous_hypothesis: str
    language: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "is_final": self.is_final,
            "previous_hypothesis": self.previous_hypothesis,
            "language": self.language,
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, object]
    ) -> RealtimeTranscriptionContext | None:
        request_id = data.get("request_id")
        is_final = data.get("is_final")
        previous_hypothesis = data.get("previous_hypothesis")
        language = data.get("language")
        if (
            not isinstance(request_id, str)
            or not isinstance(is_final, bool)
            or not isinstance(previous_hypothesis, str)
            or (language is not None and not isinstance(language, str))
        ):
            return None
        else:
            pass
        return cls(
            request_id=request_id,
            is_final=is_final,
            previous_hypothesis=previous_hypothesis,
            language=language,
        )

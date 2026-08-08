# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni's model-local prefill sidecar adopter."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner

_PREFILL_AUDIO_INPUT_KEYS = frozenset(
    {
        "audio_embeds",
        "audio_feature_lengths",
        "feature_attention_mask",
        "pad_values",
    }
)

_STANDARD = "standard"
_SIDECAR = "sidecar"
_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class _PrefillDisposition:
    kind: str
    has_live_audio: bool = False


class Qwen3OmniThinkerModelRunner(ThinkerModelRunner):
    """Adopt the shared #1364 sidecar for supported audio prefill.

    SGLang remains responsible for graph admission, bucket selection, padding,
    capture, replay, attention metadata, and eager fallback. This runner only
    decides whether Qwen's existing eager audio composition can be represented
    by the shared late-bound ``input_embeds`` contract.
    """

    @staticmethod
    def _is_integer(value: Any) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    @staticmethod
    def _coerce_cpu_int(value: Any) -> int | None:
        if Qwen3OmniThinkerModelRunner._is_integer(value):
            return int(value)
        if (
            isinstance(value, torch.Tensor)
            and value.numel() == 1
            and value.device.type == "cpu"
            and value.dtype != torch.bool
            and not torch.is_floating_point(value)
            and not torch.is_complex(value)
        ):
            return int(value)
        return None

    @staticmethod
    def _as_cpu_ids(value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        try:
            ids = torch.as_tensor(value, dtype=torch.long, device="cpu")
        except (TypeError, ValueError, RuntimeError):
            return None
        return ids if ids.ndim == 1 else None

    def _audio_positions(
        self, req: Any, pad_values: dict[str, Any]
    ) -> torch.Tensor | None:
        metadata = getattr(req, "_omni_mm_positions", None)
        if metadata is not None:
            if not isinstance(metadata, dict) or "audio" not in metadata:
                return None
            positions = metadata["audio"]
            if (
                not isinstance(positions, torch.Tensor)
                or positions.ndim != 1
                or positions.device.type != "cpu"
                or positions.dtype == torch.bool
                or torch.is_floating_point(positions)
                or torch.is_complex(positions)
            ):
                return None
            values = positions.tolist()
            if any(left >= right for left, right in zip(values, values[1:])):
                return None
            if values and values[0] < 0:
                return None
            origin_ids = self._as_cpu_ids(getattr(req, "origin_input_ids", None))
            if origin_ids is not None and values and values[-1] >= origin_ids.numel():
                return None
            return positions.to(dtype=torch.long)

        origin_ids = self._as_cpu_ids(getattr(req, "origin_input_ids", None))
        if origin_ids is None:
            return None
        audio_token_id = pad_values.get("audio", self._audio_token_id)
        if not self._is_integer(audio_token_id):
            return None
        return (origin_ids == int(audio_token_id)).nonzero(as_tuple=True)[0]

    @staticmethod
    def _chunk_span(
        req: Any, forward_batch: Any, index: int
    ) -> tuple[int, int] | None:
        extend_range = getattr(req, "extend_range", None)
        start = getattr(extend_range, "start", None)
        length = getattr(extend_range, "length", None)

        if start is None or length is None:
            prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
            extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
            try:
                start = prefix_lens[index]
                length = extend_lens[index]
            except (IndexError, TypeError):
                return None

        start = Qwen3OmniThinkerModelRunner._coerce_cpu_int(start)
        length = Qwen3OmniThinkerModelRunner._coerce_cpu_int(length)
        if start is None or length is None:
            return None
        if start < 0 or length <= 0:
            return None

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            try:
                batch_length = extend_lens[index]
            except (IndexError, TypeError):
                return None
            batch_length = Qwen3OmniThinkerModelRunner._coerce_cpu_int(batch_length)
            if batch_length is None:
                return None
            if batch_length != length:
                return None
        return start, length

    @staticmethod
    def _audio_inputs_are_well_formed(model_inputs: Any) -> bool:
        if not isinstance(model_inputs, dict) or not model_inputs:
            return False
        if set(model_inputs) - _PREFILL_AUDIO_INPUT_KEYS:
            return False

        audio_embeds = model_inputs.get("audio_embeds")
        if (
            not isinstance(audio_embeds, torch.Tensor)
            or audio_embeds.ndim != 2
            or audio_embeds.shape[0] <= 0
            or audio_embeds.shape[1] <= 0
        ):
            return False

        lengths = model_inputs.get("audio_feature_lengths")
        if lengths is not None and (
            not isinstance(lengths, torch.Tensor) or lengths.ndim != 1
        ):
            return False
        feature_mask = model_inputs.get("feature_attention_mask")
        if feature_mask is not None and (
            not isinstance(feature_mask, torch.Tensor) or feature_mask.ndim != 2
        ):
            return False

        pad_values = model_inputs.get("pad_values")
        if pad_values is not None:
            if not isinstance(pad_values, dict) or set(pad_values) - {"audio"}:
                return False
            if "audio" in pad_values and not Qwen3OmniThinkerModelRunner._is_integer(
                pad_values["audio"]
            ):
                return False
        return True

    def _classify_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> _PrefillDisposition:
        schedule_reqs = getattr(schedule_batch, "reqs", None)
        if schedule_reqs is None or len(schedule_reqs) != len(requests):
            return _PrefillDisposition(_UNSUPPORTED)
        if getattr(forward_batch, "input_embeds", None) is not None:
            return _PrefillDisposition(_UNSUPPORTED)
        if getattr(forward_batch, "replace_embeds", None) is not None:
            return _PrefillDisposition(_UNSUPPORTED)
        if get_omni_prefill_inputs(forward_batch) is not None:
            return _PrefillDisposition(_UNSUPPORTED)

        has_live_audio = False
        for index, req in enumerate(schedule_reqs):
            model_inputs = getattr(req, "omni_model_inputs", None)
            if model_inputs is None or (
                isinstance(model_inputs, dict) and not model_inputs
            ):
                continue
            if not self._audio_inputs_are_well_formed(model_inputs):
                return _PrefillDisposition(_UNSUPPORTED)

            pad_values = model_inputs.get("pad_values", {})
            chunk_span = self._chunk_span(req, forward_batch, index)
            if chunk_span is None:
                return _PrefillDisposition(_UNSUPPORTED)
            start, length = chunk_span

            origin_ids = self._as_cpu_ids(getattr(req, "origin_input_ids", None))
            if origin_ids is not None and start + length > origin_ids.numel():
                return _PrefillDisposition(_UNSUPPORTED)

            positions = self._audio_positions(req, pad_values)
            if positions is None:
                return _PrefillDisposition(_UNSUPPORTED)
            live_positions = positions[
                (positions >= start) & (positions < start + length)
            ]

            consumed = getattr(req, "_omni_consumed", None)
            if consumed is None:
                audio_offset = 0
            elif isinstance(consumed, dict):
                audio_offset = consumed.get("audio", 0)
            else:
                return _PrefillDisposition(_UNSUPPORTED)
            audio_offset = self._coerce_cpu_int(audio_offset)
            if audio_offset is None or audio_offset < 0:
                return _PrefillDisposition(_UNSUPPORTED)

            audio_rows = int(model_inputs["audio_embeds"].shape[0])
            live_count = int(live_positions.numel())
            if audio_offset > audio_rows or audio_offset + live_count > audio_rows:
                return _PrefillDisposition(_UNSUPPORTED)
            has_live_audio |= live_count > 0

        if has_live_audio:
            return _PrefillDisposition(_SIDECAR, has_live_audio=True)
        return _PrefillDisposition(_STANDARD)

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> None:
        disposition = self._classify_prefill(forward_batch, schedule_batch, requests)
        if disposition.kind != _SIDECAR:
            return

        omni_result = self._inject_multimodal_embeds(
            forward_batch, schedule_batch
        )
        if omni_result is None:
            raise RuntimeError(
                "Qwen audio prefill was classified as sidecar-compatible, "
                "but multimodal embedding composition returned no result"
            )
        input_embeds, deepstack_embeds, visual_masks = omni_result
        if input_embeds is None:
            raise RuntimeError(
                "Qwen audio prefill composition returned no input embeddings"
            )
        if deepstack_embeds is not None or visual_masks is not None:
            raise RuntimeError(
                "Qwen audio prefill sidecar cannot carry visual deepstack embeddings"
            )
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(input_embeds=input_embeds),
        )

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> Any | None:
        if get_omni_prefill_inputs(forward_batch) is not None:
            return None

        disposition = self._classify_prefill(forward_batch, schedule_batch, requests)
        if disposition.kind == _STANDARD:
            return None
        if disposition.kind == _SIDECAR:
            raise RuntimeError(
                "Qwen audio prefill sidecar was not attached before forward"
            )
        return super().custom_prefill_forward(
            forward_batch, schedule_batch, requests
        )


__all__ = ["Qwen3OmniThinkerModelRunner"]

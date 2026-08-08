# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni's model-local prefill sidecar adopter."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner

logger = logging.getLogger(__name__)

_PREFILL_AUDIO_INPUT_KEYS = frozenset(
    {
        "audio_embeds",
        "audio_feature_lengths",
        "feature_attention_mask",
        "pad_values",
    }
)


class Qwen3OmniThinkerModelRunner(ThinkerModelRunner):
    """Adopt the shared prefill sidecar for supported audio conditioning.

    SGLang remains responsible for graph admission, bucket selection, padding,
    capture, replay, attention metadata, and eager fallback. This runner only
    decides whether Qwen's eager audio embedding composition can be represented
    by the shared late-bound ``input_embeds`` contract.
    """

    def _batch_is_text_only(self, schedule_batch: Any) -> bool:
        reqs = getattr(schedule_batch, "reqs", None)
        if reqs is None:
            return False
        return all(
            (model_inputs := getattr(req, "omni_model_inputs", None)) is None
            or (isinstance(model_inputs, dict) and not model_inputs)
            for req in reqs
        )

    def _audio_token_count(self, req: Any, model_inputs: dict[str, Any]) -> int:
        """Count live audio placeholders using request-owned CPU metadata."""
        origin_input_ids = getattr(req, "origin_input_ids", None)
        extend_range = getattr(req, "extend_range", None)
        if origin_input_ids is None or extend_range is None:
            return -1

        start = getattr(extend_range, "start", None)
        length = getattr(extend_range, "length", None)
        if start is None or length is None:
            return -1

        if isinstance(origin_input_ids, torch.Tensor):
            origin_input_ids = origin_input_ids.tolist()
        try:
            start = int(start)
            length = int(length)
            if start < 0 or length <= 0 or start + length > len(origin_input_ids):
                return -1
            live_ids = origin_input_ids[start : start + length]
            if len(live_ids) != length:
                return -1
            pad_values = model_inputs.get("pad_values")
            audio_token_id = self._audio_token_id
            if isinstance(pad_values, dict) and "audio" in pad_values:
                audio_token_id = pad_values["audio"]
            return sum(
                int(token_id) == int(audio_token_id) for token_id in live_ids
            )
        except (TypeError, ValueError, IndexError):
            return -1

    @staticmethod
    def _audio_inputs_are_well_formed(model_inputs: Any) -> bool:
        if not isinstance(model_inputs, dict) or not model_inputs:
            return False
        if any(key not in _PREFILL_AUDIO_INPUT_KEYS for key in model_inputs):
            return False

        audio_embeds = model_inputs.get("audio_embeds")
        if (
            not isinstance(audio_embeds, torch.Tensor)
            or audio_embeds.ndim != 2
            or audio_embeds.shape[0] <= 0
            or audio_embeds.shape[1] <= 0
        ):
            return False

        for key in ("audio_feature_lengths", "feature_attention_mask"):
            value = model_inputs.get(key)
            if value is not None and not isinstance(value, torch.Tensor):
                return False

        pad_values = model_inputs.get("pad_values")
        if pad_values is not None:
            if not isinstance(pad_values, dict) or set(pad_values) - {"audio"}:
                return False
            if "audio" in pad_values and (
                isinstance(pad_values["audio"], bool)
                or not isinstance(pad_values["audio"], int)
            ):
                return False
        return True

    def _prefill_payload_preflight(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> bool:
        schedule_reqs = getattr(schedule_batch, "reqs", None)
        if schedule_reqs is None or len(schedule_reqs) != len(requests):
            return False
        try:
            batch_size = int(getattr(forward_batch, "batch_size", -1))
        except (TypeError, ValueError):
            return False
        if len(requests) != batch_size:
            return False
        if getattr(forward_batch, "input_embeds", None) is not None:
            return False
        if get_omni_prefill_inputs(forward_batch) is not None:
            return False
        if getattr(forward_batch, "replace_embeds", None) is not None:
            return False

        has_audio = False
        for req in schedule_reqs:
            model_inputs = getattr(req, "omni_model_inputs", None)
            if model_inputs is None or (
                isinstance(model_inputs, dict) and not model_inputs
            ):
                continue
            if not self._audio_inputs_are_well_formed(model_inputs):
                return False

            live_audio_tokens = self._audio_token_count(req, model_inputs)
            if live_audio_tokens < 0:
                return False
            consumed = getattr(req, "_omni_consumed", None)
            if consumed is None:
                consumed = {}
            if not isinstance(consumed, dict):
                return False
            try:
                audio_offset = int(consumed.get("audio", 0))
                audio_rows = int(model_inputs["audio_embeds"].shape[0])
            except (TypeError, ValueError):
                return False
            if audio_offset < 0:
                return False
            if audio_offset + live_audio_tokens > audio_rows:
                return False
            has_audio |= live_audio_tokens > 0

        return has_audio

    @staticmethod
    def _snapshot_request_inputs(reqs: list[Any]) -> list[tuple[Any, Any, Any]]:
        snapshots = []
        for req in reqs:
            consumed = getattr(req, "_omni_consumed", None)
            snapshots.append(
                (
                    req,
                    getattr(req, "omni_model_inputs", None),
                    dict(consumed) if isinstance(consumed, dict) else consumed,
                )
            )
        return snapshots

    @staticmethod
    def _restore_request_inputs(snapshots: list[tuple[Any, Any, Any]]) -> None:
        for req, model_inputs, consumed in snapshots:
            req.omni_model_inputs = model_inputs
            req._omni_consumed = consumed

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> None:
        if not self._prefill_payload_preflight(
            forward_batch, schedule_batch, requests
        ):
            return

        schedule_reqs = list(schedule_batch.reqs)
        snapshots = self._snapshot_request_inputs(schedule_reqs)
        try:
            omni_result = self._inject_multimodal_embeds(
                forward_batch, schedule_batch
            )
            if omni_result is None:
                return
            input_embeds, deepstack_embeds, visual_masks = omni_result
            if (
                input_embeds is None
                or deepstack_embeds is not None
                or visual_masks is not None
            ):
                self._restore_request_inputs(snapshots)
                return
            attach_omni_prefill_inputs(
                forward_batch,
                OmniPrefillInputs(input_embeds=input_embeds),
            )
        except (IndexError, TypeError, ValueError, RuntimeError):
            self._restore_request_inputs(snapshots)
            logger.debug(
                "Qwen audio prefill sidecar rejected; using eager fallback",
                exc_info=True,
            )

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> Any | None:
        """Keep sidecar and pure-text batches on SGLang's standard path."""
        if get_omni_prefill_inputs(forward_batch) is not None:
            return None
        if self._batch_is_text_only(schedule_batch):
            return None
        return super().custom_prefill_forward(forward_batch, schedule_batch, requests)


__all__ = ["Qwen3OmniThinkerModelRunner"]

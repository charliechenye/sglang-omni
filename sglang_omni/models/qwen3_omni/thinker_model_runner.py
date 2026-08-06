# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni-specific prefill adaptation.

The shared ``ThinkerModelRunner`` remains responsible for the existing legacy
embedding path. Qwen3-Omni additionally carries native audio embeddings in
``MultimodalDataItem`` objects. When a legacy request shares a batch with a
native request, this adapter temporarily presents the native item through the
legacy runner contract so the existing eager scatter handles the whole batch.

Native-only batches do not enter this path: the adapter leaves their request
contract untouched, so the shared runner returns ``None`` and upstream SGLang
performs the normal model-forward dispatch and graph/eager decision.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner

_MISSING = object()


class Qwen3OmniThinkerModelRunner(ThinkerModelRunner):
    """Qwen-owned bridge for mixed native-audio and legacy batches."""

    def _native_audio_item(self, req: Any) -> Any | None:
        """Return a representable native audio item, if ``req`` has one.

        This is a semantic contract check only. It does not inspect captured
        buckets, graph state, padding, or any upstream replay condition.
        """
        if getattr(req, "omni_model_inputs", None) is not None:
            return None

        multimodal_inputs = getattr(req, "multimodal_inputs", None)
        items = getattr(multimodal_inputs, "mm_items", None) or ()
        if not items:
            return None
        if len(items) != 1:
            raise RuntimeError(
                "Qwen3-Omni mixed eager batches only support one native audio item"
            )

        item = items[0]
        modality = getattr(item, "modality", None)
        modality_name = getattr(modality, "name", modality)
        item_format = getattr(item, "format", None)
        format_name = getattr(item_format, "name", item_format)
        if (
            modality_name not in ("AUDIO", "audio")
            or format_name != "PRECOMPUTED_EMBEDDING"
        ):
            raise RuntimeError(
                "Qwen3-Omni mixed eager batches only support native "
                "PRECOMPUTED_EMBEDDING audio items"
            )

        source = getattr(item, "precomputed_embeddings", None)
        if not isinstance(source, torch.Tensor) or source.ndim != 2:
            raise RuntimeError(
                "Qwen3-Omni native audio items must carry a 2-D "
                "precomputed_embeddings tensor"
            )

        model_specific_data = getattr(item, "model_specific_data", None) or {}
        positions_cpu = model_specific_data.get("positions_cpu")
        if (
            not isinstance(positions_cpu, torch.Tensor)
            or positions_cpu.device.type != "cpu"
            or positions_cpu.ndim != 1
            or positions_cpu.dtype != torch.long
        ):
            raise RuntimeError(
                "Qwen3-Omni native audio items must carry 1-D CPU int64 "
                "positions_cpu metadata"
            )
        if positions_cpu.numel() != source.shape[0]:
            raise ValueError(
                "Qwen3-Omni native audio rows do not match positions_cpu"
            )
        if positions_cpu.numel() > 1 and not torch.equal(
            positions_cpu, positions_cpu.sort().values
        ):
            raise RuntimeError(
                "Qwen3-Omni native audio positions_cpu must be sorted"
            )
        return item

    def _legacy_inputs_for_native_audio(self, item: Any) -> dict[str, Any]:
        """Build a temporary legacy view without copying encoder output."""
        pad_value = getattr(item, "pad_value", None)
        if pad_value is None:
            pad_value = self._audio_token_id
        return {
            "audio_embeds": item.precomputed_embeddings,
            "pad_values": {"audio": int(pad_value)},
        }

    @contextlib.contextmanager
    def _temporary_native_audio_legacy_views(
        self, native_requests: list[tuple[Any, Any]]
    ) -> Iterator[None]:
        """Temporarily adapt native requests for the existing eager scatter."""
        saved: list[tuple[Any, dict[str, Any]]] = []
        for req, item in native_requests:
            state = {
                name: getattr(req, name, _MISSING)
                for name in (
                    "omni_model_inputs",
                    "_omni_consumed",
                    "_omni_mm_positions",
                )
            }
            saved.append((req, state))
            req.omni_model_inputs = self._legacy_inputs_for_native_audio(item)
            # The shared runner owns these temporary cursors for this eager
            # call. Restore the native request state after it returns.
            req._omni_consumed = None
            req._omni_mm_positions = None

        try:
            yield
        finally:
            for req, state in reversed(saved):
                for name, value in state.items():
                    if value is _MISSING:
                        try:
                            delattr(req, name)
                        except AttributeError:
                            pass
                    else:
                        setattr(req, name, value)

    def _inject_multimodal_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> tuple[torch.Tensor | None, list | None, torch.Tensor | None] | None:
        """Merge native audio only when a legacy request already needs eager."""
        requests = schedule_batch.reqs
        has_legacy_request = any(
            getattr(req, "omni_model_inputs", None) is not None for req in requests
        )
        if not has_legacy_request:
            # Native-only requests stay representable by normal model.forward;
            # the shared runner returns None and upstream owns dispatch.
            return super()._inject_multimodal_embeds(forward_batch, schedule_batch)

        native_requests = []
        for req in requests:
            item = self._native_audio_item(req)
            if item is not None:
                native_requests.append((req, item))
        if not native_requests:
            return super()._inject_multimodal_embeds(forward_batch, schedule_batch)

        with self._temporary_native_audio_legacy_views(native_requests):
            return super()._inject_multimodal_embeds(forward_batch, schedule_batch)

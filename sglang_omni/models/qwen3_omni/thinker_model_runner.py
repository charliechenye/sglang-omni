# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni thinker runner and its qualified prefill payload contract."""

from __future__ import annotations

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


def _request_generates_speech(request: Any) -> bool:
    from sglang_omni.models.qwen3_omni.request_builders import (
        should_generate_audio_output,
    )

    data = getattr(request, "data", None)
    return should_generate_audio_output(getattr(data, "stage_payload", None))


class Qwen3OmniThinkerModelRunner(ThinkerModelRunner):
    """Thinker runner with Qwen3-Omni's qualified prefill payload path."""

    def _can_use_prefill_payload(
        self, schedule_batch: Any, requests: list[Any]
    ) -> bool:
        """Return whether the whole batch has Qwen-compatible semantics.

        This is deliberately a model-semantic check only. Upstream SGLang owns
        all graph eligibility, bucket selection, padding, and fallback policy.
        """
        schedule_reqs = getattr(schedule_batch, "reqs", None)
        if schedule_reqs is None or len(schedule_reqs) != len(requests):
            return False
        if self._batch_should_capture_hidden(requests):
            return False

        for req, request in zip(schedule_reqs, requests):
            if _request_generates_speech(request):
                return False

            model_inputs = getattr(req, "omni_model_inputs", None)
            if model_inputs is None:
                continue
            if not isinstance(model_inputs, dict):
                return False

            # Visual/deepstack tensors and video audio are intentionally kept on
            # the legacy eager path until they have their own qualified payload
            # contract. This is a Qwen capability decision, not graph routing.
            if any(
                key.startswith(("image_", "video_"))
                or key
                in {
                    "image_embeds",
                    "video_embeds",
                    "deepstack_visual_embeds",
                    "use_audio_in_video",
                }
                for key in model_inputs
            ):
                return False
            if any(key not in _PREFILL_AUDIO_INPUT_KEYS for key in model_inputs):
                return False

            audio_keys = _PREFILL_AUDIO_INPUT_KEYS.intersection(model_inputs)
            if audio_keys:
                audio_embeds = model_inputs.get("audio_embeds")
                if not isinstance(audio_embeds, torch.Tensor) or audio_embeds.ndim != 2:
                    return False

            pad_values = model_inputs.get("pad_values")
            if pad_values is not None and (
                not isinstance(pad_values, dict)
                or any(key != "audio" for key in pad_values)
            ):
                return False

        return True

    def _prefill_payload_preflight(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> bool:
        """Check attachment invariants before composing request embeddings."""
        if not self._can_use_prefill_payload(schedule_batch, requests):
            return False

        # The semantic classifier already establishes schedule/request
        # alignment; this check ties that batch to the live ForwardBatch.
        if len(requests) != forward_batch.batch_size:
            return False
        if getattr(forward_batch, "input_embeds", None) is not None:
            return False
        if get_omni_prefill_inputs(forward_batch) is not None:
            return False
        return self._prefill_payload_mm_shell_is_replaceable(forward_batch)

    @staticmethod
    def _prefill_payload_mm_shell_is_replaceable(forward_batch: Any) -> bool:
        """Check whether SGLang's MM shell can be replaced by our payload.

        This check must remain pure. ``_inject_multimodal_embeds`` consumes
        request-side multimodal state at the end of a prefill chunk, so doing
        composition before this check would make a genuine SGLang multimodal
        batch impossible to recover from after failing closed.
        """
        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if mm_inputs is None:
            return True
        if not isinstance(mm_inputs, (list, tuple)):
            return False
        if len(mm_inputs) != forward_batch.batch_size:
            return False

        for item in mm_inputs:
            if item is None:
                continue
            mm_items = getattr(item, "mm_items", None)
            if mm_items is None or len(mm_items) != 0:
                return False
            shell_has_mrope = (
                getattr(item, "mrope_positions", None) is not None
                or getattr(item, "mrope_position_delta", None) is not None
            )
            if (
                shell_has_mrope
                and getattr(forward_batch, "mrope_positions", None) is None
            ):
                # note (chenye): SGLang materializes only current-prefill
                # M-RoPE positions on ForwardBatch; the delta stays on the
                # request-side MultimodalInputs for future decode positions.
                return False

        return True

    @staticmethod
    def _clear_prefill_mm_shell(forward_batch: Any) -> None:
        """Clear an already-validated, item-free SGLang MM shell."""
        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if mm_inputs is None:
            return

        # note (chenye): Replace only the ForwardBatch-local shell. The
        # request's MultimodalInputs retains mrope_position_delta for decode.
        forward_batch.mm_inputs = [None] * forward_batch.batch_size

    def _build_prefill_input_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> torch.Tensor | None:
        omni_result = self._inject_multimodal_embeds(forward_batch, schedule_batch)
        if omni_result is None:
            embed_input_ids = forward_batch.input_ids.clamp(
                0, self._embed_tokens.num_embeddings - 1
            )
            return self._embed_tokens(embed_input_ids)

        input_embeds, deepstack_embeds, visual_masks = omni_result
        if deepstack_embeds is not None or visual_masks is not None:
            return None
        return input_embeds

    def before_prefill(self, forward_batch, schedule_batch, requests):
        if not self._prefill_payload_preflight(
            forward_batch, schedule_batch, requests
        ):
            return
        composed_input_embeds = self._build_prefill_input_embeds(
            forward_batch, schedule_batch
        )
        if composed_input_embeds is None:
            return
        self._clear_prefill_mm_shell(forward_batch)

        # note (chenye): Compose the normal eager prefill embeddings
        # first, then carry them through the platform OmniPrefillInputs channel.
        # The live ForwardBatch.input_embeds remains None so upstream SGLang owns
        # graph eligibility and graph-static storage.
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=composed_input_embeds,
                rids=tuple(request.request_id for request in requests),
            ),
        )

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        """Delegate payload-compatible batches to normal SGLang execution."""
        if get_omni_prefill_inputs(forward_batch) is not None:
            # The upstream runner selects eager versus Breakable Prefill CG for
            # a payload-compatible batch.
            return None
        return super().custom_prefill_forward(forward_batch, schedule_batch, requests)


__all__ = ["Qwen3OmniThinkerModelRunner"]

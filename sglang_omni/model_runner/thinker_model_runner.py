# SPDX-License-Identifier: Apache-2.0
"""Thinker model runner — injects multimodal embeddings before forward.

Handles image/video/audio token → embedding replacement and deepstack
visual embeddings for Qwen3-Omni's thinker stage.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)

logger = logging.getLogger(__name__)

_QWEN_PREFILL_AUDIO_INPUT_KEYS = frozenset(
    {
        "audio_embeds",
        "audio_feature_lengths",
        "feature_attention_mask",
        "pad_values",
    }
)


def _qwen_request_generates_speech(request: Any) -> bool:
    from sglang_omni.models.qwen3_omni.request_builders import (
        should_generate_audio_output,
    )

    data = getattr(request, "data", None)
    return should_generate_audio_output(getattr(data, "stage_payload", None))


class ThinkerModelRunner(ModelRunner):
    """Thinker: injects multimodal embeddings in the prefill phase."""

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        should_capture_hidden: Callable[[Any], bool] | None = None,
    ):
        super().__init__(tp_worker, output_processor)
        self._should_capture_hidden = should_capture_hidden

        model = self.model
        self._outer_model = model.thinker
        self._text_model = self._outer_model.model
        self._embed_tokens = self._text_model.embed_tokens
        self._th_host_bufs = None
        self._th_slot = 0

        thinker_cfg = tp_worker.model_runner.model_config.hf_config.thinker_config
        self._image_token_id = thinker_cfg.image_token_id
        self._video_token_id = thinker_cfg.video_token_id
        self._audio_token_id = thinker_cfg.audio_token_id

    @contextlib.contextmanager
    def _text_only_capture_guard(self, requests: list[Any]):
        # note (jiaxin deng): drop hidden-capture for an all-text batch, shared by
        # sync execute() and async execute_launch so both take the same path.
        capture_layers = self._text_model.layers_to_capture
        if not (capture_layers and not self._batch_should_capture_hidden(requests)):
            yield
            return
        saved_capture_layers = list(capture_layers)
        self._text_model.layers_to_capture = []
        try:
            yield
        finally:
            self._text_model.layers_to_capture = saved_capture_layers

    def execute(self, scheduler_output: Any):
        with self._text_only_capture_guard(scheduler_output.requests):
            return super().execute(scheduler_output)

    def execute_launch(self, scheduler_output: Any):
        with self._text_only_capture_guard(scheduler_output.requests):
            return super().execute_launch(scheduler_output)

    def _batch_should_capture_hidden(self, requests: list[Any]) -> bool:
        if self._should_capture_hidden is None:
            return True
        for request in requests:
            if self._should_capture_hidden(request):
                return True
        return False

    def _can_use_qwen_prefill_payload(
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
            if _qwen_request_generates_speech(request):
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
                or key in {
                    "image_embeds",
                    "video_embeds",
                    "deepstack_visual_embeds",
                    "use_audio_in_video",
                }
                for key in model_inputs
            ):
                return False
            if any(key not in _QWEN_PREFILL_AUDIO_INPUT_KEYS for key in model_inputs):
                return False

            pad_values = model_inputs.get("pad_values")
            if pad_values is not None and (
                not isinstance(pad_values, dict)
                or any(key != "audio" for key in pad_values)
            ):
                return False

        return True

    @staticmethod
    def _replace_consumed_mrope_shell(forward_batch: Any) -> bool:
        """Clear only item-free Qwen ``MultimodalInputs`` shells."""
        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if mm_inputs is None:
            return True
        if not isinstance(mm_inputs, (list, tuple)):
            return False

        for item in mm_inputs:
            if item is None:
                continue
            mm_items = getattr(item, "mm_items", None)
            if mm_items is None or len(mm_items) != 0:
                return False
            for field in ("mrope_positions", "mrope_position_delta"):
                if getattr(item, field, None) is not None and getattr(
                    forward_batch, field, None
                ) is None:
                    return False

        # NOTE(qwen3-omni-pcg): Qwen uses an empty MultimodalInputs shell to
        # move M-RoPE metadata onto ForwardBatch. At this point that metadata
        # has already been materialized; only an item-free shell may be replaced
        # with None entries before the official platform payload is attached.
        forward_batch.mm_inputs = [None] * forward_batch.batch_size
        return True

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
        if not self._can_use_qwen_prefill_payload(schedule_batch, requests):
            return
        composed_input_embeds = self._build_prefill_input_embeds(
            forward_batch, schedule_batch
        )
        if composed_input_embeds is None:
            return
        if not self._replace_consumed_mrope_shell(forward_batch):
            return

        # NOTE(qwen3-omni-pcg): Compose the normal eager prefill embeddings
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
        """Run custom prefill when multimodal embeddings must be injected."""
        if not schedule_batch.forward_mode.is_extend():
            return None
        if get_omni_prefill_inputs(forward_batch) is not None:
            # A payload-compatible batch delegates to normal SGLang execution;
            # the upstream runner selects eager versus Breakable Prefill CG.
            return None

        omni_result = self._inject_multimodal_embeds(forward_batch, schedule_batch)
        if omni_result is not None and omni_result[0] is not None:
            input_embeds, ds_embeds, vis_masks = omni_result
            return self._forward_with_omni_embeds(
                forward_batch, input_embeds, ds_embeds, vis_masks
            )
        return None

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ):
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        # Hidden capture for thinker streaming comes from our local forward hooks,
        # not from SGLang's logits-output hidden-state path. Requesting LAST here
        # causes CUDA-graph mode mismatches and can silently disable replay.
        return CaptureHiddenMode.NULL

    def requested_capture_hidden_mode_decode(self, schedule_batch: Any, requests: list):
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        # Hidden capture for thinker streaming comes from our local forward hooks,
        # not from SGLang's logits-output hidden-state path. Requesting LAST here
        # causes CUDA-graph mode mismatches and can silently disable replay.
        return CaptureHiddenMode.NULL

    @staticmethod
    def _is_final_prefill_chunk(req: Any) -> bool:
        """Support the current Req chunk marker and older SGLang snapshots."""
        is_chunked = getattr(req, "is_chunked", None)
        if is_chunked is not None:
            return not bool(is_chunked)
        return int(getattr(req, "inflight_middle_chunks", 0) or 0) == 0

    # ------------------------------------------------------------------
    # Multimodal embedding injection
    # ------------------------------------------------------------------

    def _req_mm_token_positions(
        self, req: Any, pad_values: dict
    ) -> dict[str, torch.Tensor]:
        """Prompt-absolute placeholder positions per modality, as CPU int64
        tensors so the merge never reads placement off a GPU mask."""
        positions = getattr(req, "_omni_mm_positions", None)
        if positions is not None:
            return positions
        prompt_ids = torch.as_tensor(req.origin_input_ids, dtype=torch.long)
        positions = {
            modality: (prompt_ids == pad_values.get(modality, default_id)).nonzero(
                as_tuple=True
            )[0]
            for modality, default_id in (
                ("image", self._image_token_id),
                ("video", self._video_token_id),
                ("audio", self._audio_token_id),
            )
        }
        req._omni_mm_positions = positions
        return positions

    def _inject_multimodal_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> tuple[torch.Tensor | None, list | None, torch.Tensor | None] | None:
        if not any(req.omni_model_inputs is not None for req in schedule_batch.reqs):
            return None

        device = forward_batch.input_ids.device

        embed_input_ids = forward_batch.input_ids.clamp(
            0, self._embed_tokens.num_embeddings - 1
        )
        input_embeds = self._embed_tokens(embed_input_ids)

        # note (chenrui): these arrive as CPU tensors on some sglang paths, where
        # int(tensor[i]) per request would put a .item() on the hot path.
        extend_lens = forward_batch.extend_seq_lens_cpu
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        if isinstance(extend_lens, torch.Tensor):
            extend_lens = extend_lens.tolist()
        if isinstance(prefix_lens, torch.Tensor):
            prefix_lens = prefix_lens.tolist()
        offsets = []
        pos = 0
        for length in extend_lens:
            offsets.append(pos)
            pos += length

        scatter_rows: list[torch.Tensor] = []
        scatter_srcs: list[torch.Tensor] = []
        deepstack_visual_embeds_list = []
        visual_rows: list[torch.Tensor] = []

        for i, req in enumerate(schedule_batch.reqs):
            omni_inputs = req.omni_model_inputs
            if omni_inputs is None:
                continue

            start = offsets[i]
            length = extend_lens[i]
            prefix = 0 if prefix_lens is None else int(prefix_lens[i])
            consumed = req._omni_consumed or {}
            req._omni_consumed = consumed
            chunk_offsets: dict[str, tuple[int, int]] = {}
            pad_values = omni_inputs.get("pad_values", {})

            positions = self._req_mm_token_positions(req, pad_values)
            chunk_positions: dict[str, torch.Tensor] = {}
            for modality in ("image", "video", "audio"):
                mod_positions = positions[modality]
                # note (chenrui): dispatching the mask ops for an absent modality
                # costs more than this shortcut saves.
                if mod_positions.numel() == 0:
                    chunk_positions[modality] = mod_positions
                    continue
                in_chunk = (mod_positions >= prefix) & (mod_positions < prefix + length)
                rel = mod_positions[in_chunk] - prefix
                chunk_positions[modality] = rel
                n_tokens = rel.numel()
                embeds = omni_inputs.get(f"{modality}_embeds")
                if embeds is None or n_tokens == 0:
                    continue
                offset = consumed.get(modality, 0)
                chunk_offsets[modality] = (offset, n_tokens)
                scatter_rows.append(rel + start)
                scatter_srcs.append(embeds[offset : offset + n_tokens])
                consumed[modality] = offset + n_tokens

            ds_embeds = omni_inputs.get("deepstack_visual_embeds")
            image_ds = omni_inputs.get("image_deepstack_visual_embeds")
            video_ds = omni_inputs.get("video_deepstack_visual_embeds")

            if ds_embeds is not None or image_ds is not None or video_ds is not None:
                img_pos = chunk_positions["image"]
                vid_pos = chunk_positions["video"]
                # note (chenrui): positions are unique across modalities, so the
                # sort is tie-free and its permutation identifies each slot.
                visual_pos, visual_order = torch.sort(torch.cat([img_pos, vid_pos]))
                visual_count = visual_pos.numel()

                if ds_embeds is None:
                    if image_ds and video_ds:
                        image_offset, image_count = chunk_offsets.get("image", (0, 0))
                        video_offset, video_count = chunk_offsets.get("video", (0, 0))
                        # note (chenrui): the equivalent mask plus nonzero would
                        # sync the moment its input is device-resident.
                        slots = torch.empty_like(visual_order)
                        slots[visual_order] = torch.arange(
                            visual_count, device=slots.device
                        )
                        n_image = img_pos.numel()
                        img_idx = slots[:n_image].to(device)
                        vid_idx = slots[n_image:].to(device)
                        merged = []
                        for img_e, vid_e in zip(image_ds, video_ds):
                            img_e = img_e[image_offset : image_offset + image_count]
                            vid_e = vid_e[video_offset : video_offset + video_count]
                            joint = img_e.new_zeros(
                                visual_count, img_e.shape[-1], device=device
                            )
                            joint[img_idx] = img_e.to(device=device)
                            joint[vid_idx] = vid_e.to(device=device)
                            merged.append(joint)
                        ds_embeds = merged
                    elif image_ds:
                        image_offset, image_count = chunk_offsets.get("image", (0, 0))
                        ds_embeds = [
                            layer[image_offset : image_offset + image_count]
                            for layer in image_ds
                        ]
                    elif video_ds:
                        video_offset, video_count = chunk_offsets.get("video", (0, 0))
                        ds_embeds = [
                            layer[video_offset : video_offset + video_count]
                            for layer in video_ds
                        ]
                elif visual_count > 0:
                    if not img_pos.numel():
                        visual_offset = chunk_offsets.get("video", (0, 0))[0]
                    elif not vid_pos.numel():
                        visual_offset = chunk_offsets.get("image", (0, 0))[0]
                    else:
                        visual_offset = consumed.get("_visual", 0)
                    ds_embeds = [
                        layer[visual_offset : visual_offset + visual_count]
                        for layer in ds_embeds
                    ]
                    consumed["_visual"] = visual_offset + visual_count
                else:
                    ds_embeds = None

                if ds_embeds is not None:
                    deepstack_visual_embeds_list.append(ds_embeds)
                    visual_rows.append(visual_pos + start)

            if self._is_final_prefill_chunk(req):
                req.omni_model_inputs = None
                req._omni_consumed = None
                req._omni_mm_positions = None

        if scatter_rows:
            # note (chenrui): one index_copy_ keeps the kernel count independent
            # of batch composition; the cat it costs is pointless for one source.
            row_idx = torch.cat(scatter_rows).to(device=device)
            srcs = [
                s.to(device=device, dtype=input_embeds.dtype, non_blocking=True)
                for s in scatter_srcs
            ]
            src = srcs[0] if len(srcs) == 1 else torch.cat(srcs, dim=0)
            input_embeds.index_copy_(0, row_idx, src)

        ds_embeds_out = None
        visual_masks_out = None
        if deepstack_visual_embeds_list:
            combined_mask = torch.zeros(
                len(forward_batch.input_ids), dtype=torch.bool, device=device
            )
            combined_mask[torch.cat(visual_rows).to(device=device)] = True
            visual_masks_out = combined_mask
            if len(deepstack_visual_embeds_list) == 1:
                ds_embeds_out = deepstack_visual_embeds_list[0]
            else:
                num_layers = len(deepstack_visual_embeds_list[0])
                merged_ds = []
                for layer_idx in range(num_layers):
                    parts = [
                        req_ds[layer_idx].to(device=device, dtype=input_embeds.dtype)
                        for req_ds in deepstack_visual_embeds_list
                    ]
                    merged_ds.append(torch.cat(parts, dim=0))
                ds_embeds_out = merged_ds

        return input_embeds, ds_embeds_out, visual_masks_out

    # ------------------------------------------------------------------
    # Custom forward with multimodal embeddings + deepstack
    # ------------------------------------------------------------------

    def _forward_with_omni_embeds(
        self,
        forward_batch,
        input_embeds,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
    ):
        model_runner = self.tp_worker.model_runner
        outer = self._outer_model

        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        ds_input = None
        if deepstack_visual_embeds is not None and visual_pos_masks is not None:
            device = input_embeds.device
            dtype = input_embeds.dtype
            layer_tensors = [
                t.to(device=device, dtype=dtype) for t in deepstack_visual_embeds
            ]
            ds_input = torch.cat(layer_tensors, dim=-1)
            full_ds = torch.zeros(
                input_embeds.shape[0], ds_input.shape[-1], device=device, dtype=dtype
            )
            full_ds[visual_pos_masks] = ds_input
            ds_input = full_ds

        hidden_states = outer.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_deepstack_embeds=ds_input,
        )

        logits_output = outer.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            outer.lm_head,
            forward_batch,
        )

        return GenerationBatchResult(
            logits_output=logits_output, can_run_cuda_graph=False
        )

    def lookahead_eligible(self, batch: Any) -> bool:
        """Route to sync where the one-step lag would diverge from sync. A request
        that emits audio captures hidden states for the talker; the per-forward
        _captured_aux_hidden_states side channel would be overwritten by a lookahead
        launch(N) before resolve(N-1) collects it, so those requests route to sync
        per batch. Sampling that reads the lagged output history (repetition /
        presence / frequency penalty, min_new_tokens), a fixed seed, or
        return_logprob (the lookahead sampler skips the base logprob path) also
        diverges; logit_bias / custom_params are routed conservatively.
        """
        from sglang_omni.models.qwen3_omni.request_builders import (
            should_generate_audio_output,
        )

        for req in batch.reqs:
            # note (jiaxin deng): fail closed if the request data is missing or None
            # so a hidden-capture batch can never slip onto the async path.
            try:
                data = req._omni_data
            except AttributeError:
                data = None
            if data is None or should_generate_audio_output(data.stage_payload):
                return False
            try:
                needs_logprob = data.return_logprob
            except AttributeError:
                needs_logprob = False
            if needs_logprob:
                return False
            sp = req.sampling_params
            if (
                sp.repetition_penalty != 1.0
                or sp.presence_penalty != 0.0
                or sp.frequency_penalty != 0.0
                or sp.min_new_tokens > 0
                or sp.sampling_seed is not None
                or sp.logit_bias is not None
                or sp.custom_params
            ):
                return False
        return True

    def _async_host_buf(self, like: torch.Tensor, n: int) -> torch.Tensor:
        # note (jiaxin deng): two pinned buffers ping-ponged so resolve(N) reads
        # one while launch(N+1) writes the other.
        if self._th_host_bufs is None or self._th_host_bufs[0].shape[0] < n:
            self._th_host_bufs = [
                torch.empty(n, dtype=like.dtype, device="cpu", pin_memory=True)
                for _ in range(2)
            ]
            self._th_slot = 0
        buf = self._th_host_bufs[self._th_slot]
        self._th_slot ^= 1
        return buf

    def _sample_lookahead(self, logits_output, forward_batch, requests):
        # note (jiaxin deng): penalties never reach here (lookahead_eligible routes
        # those batches to sync); only static suppress tokens are lag-safe.
        self._apply_codec_suppress_tokens(logits_output, requests)
        return self.tp_worker.model_runner.sample(logits_output, forward_batch)

    def post_decode_launch(self, result, forward_batch, requests):
        n = len(requests)
        if n == 0:
            return None
        # note (jiaxin deng): the decode forward leaves next_token_ids None (sync
        # samples in _finalize); set it here for the next-step input chain.
        if result.next_token_ids is None:
            result.next_token_ids = self._sample_lookahead(
                result.logits_output, forward_batch, requests
            )
        nt = result.next_token_ids
        host_buf = self._async_host_buf(nt, n)
        host_buf[:n].copy_(nt[:n], non_blocking=True)
        return host_buf

    def post_decode_resolve(
        self, launch_buf, result, forward_batch, schedule_batch, requests
    ):
        del forward_batch, schedule_batch
        if len(requests) == 0 or launch_buf is None:
            return
        n = len(requests)
        result.next_token_ids = launch_buf[:n].to(torch.long).clone()

# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

import os
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from sglang_omni.utils.device import has_device_streams

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
    from sglang_omni.models.qwen3_tts.sglang_model import (
        PredictorGraphStats,
        Qwen3TTSTalker,
    )
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
else:
    pass

# note (ratish): official qwen-tts generates the talker with min_new_tokens=2; an EOS on the first frame ended a request with no audio.
QWEN3_TTS_MIN_NEW_FRAMES = 2


class Qwen3TTSModelInfo(TypedDict):
    predictor_cuda_graph: PredictorGraphStats


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    model: Qwen3TTSTalker
    tp_worker: ModelWorker
    # Resolved in __init__: post_process_outputs records a readiness event per
    # decode step, and the device cannot change under the runner.
    codes_event_module: ModuleType | None = None

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        leading_silence_mask_frames: int,
        silence_codec_ids: torch.Tensor,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        if os.environ.get("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR"):
            from .code_predictor_spec_p17_activation import record

            record("model_runner_initialized", caller_module=__file__)
        self.has_pending_code_step = False
        self.row_ids_cache: torch.Tensor | None = None
        self.leading_silence_mask_frames = leading_silence_mask_frames
        self.silence_codec_ids = silence_codec_ids
        self.codes_event_module = (
            torch.get_device_module(self.device)
            if has_device_streams(self.device)
            else None
        )

    def model_info(self) -> Qwen3TTSModelInfo:
        return {"predictor_cuda_graph": self.model.predictor_graph_stats()}

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=self.build_prefill_input_embeds(
                    forward_batch,
                    requests,
                ),
            ),
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        self.write_feedback_buffers(forward_batch, requests)

    def custom_decode_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        # Base prepare_and_forward runs before_decode (including feedback
        # staging) and then this hook IMMEDIATELY before the main Talker
        # tp_worker.forward_batch_generation. Source-locked P19-only.
        if forward_batch is not None:
            self._p19_native_tail_mark(
                "MAIN_TALKER_FORWARD_BEGIN",
                device=forward_batch.input_ids.device,
                requests=requests,
            )
        # Preserve normal Qwen3-TTS backend ownership. Returning None
        # makes base dispatch the existing tp_worker CUDA Graph/eager path.
        return None

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        # Main tp_worker forward is complete on the HOST, while the GPU
        # may still have enqueued work. An opt-in full-device fence here
        # separates its kernels from the native semantic sampler.
        self._p19_native_tail_mark(
            "MAIN_TALKER_FORWARD_RETURNED_BEFORE_SAMPLE",
            device=forward_batch.input_ids.device,
            requests=requests,
        )
        sampled = super().sample_next_token_ids(
            logits_output, forward_batch, schedule_batch, requests
        )
        self._p19_native_tail_mark(
            "MAIN_SEMANTIC_SAMPLER_RETURNED",
            device=forward_batch.input_ids.device,
            requests=requests,
        )
        return sampled

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_codes(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        # Always reached after base prepare_and_forward returns. A backend
        # may prepopulate next_token_ids and skip sample_next_token_ids,
        # so this stage captures that alternate path as well.
        if forward_batch is not None:
            self._p19_native_tail_mark(
                "MAIN_TALKER_STEP_RETURNED_BEFORE_COLLECT",
                device=forward_batch.input_ids.device,
                requests=requests,
            )
        self.collect_codes(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def lookahead_eligible(self, batch: ScheduleBatch | None) -> bool:
        # note(ratish): the lookahead's launch and resolve hooks do not run the
        # codec collect, they would feed token embeddings back.
        del batch
        return False

    # ------------------------------------------------------------------
    # Qwen3-TTS logit shaping
    # ------------------------------------------------------------------

    def apply_codec_suppress_tokens(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or not requests:
            return
        else:
            pass

        # Qwen3-TTS reserves the final 1024 configured token IDs for codec
        # control and suppresses that range except for codec EOS.
        configured_vocab = int(self.model.config.vocab_size)
        suppress_start = max(0, configured_vocab - 1024)
        suppress_stop = min(configured_vocab, logits.shape[1])
        if suppress_start >= suppress_stop:
            return
        else:
            pass

        active_logits = logits[: len(requests)]
        codec_eos = int(self.model.config.codec_eos_token_id)
        if suppress_start <= codec_eos < suppress_stop:
            active_logits[:, suppress_start:codec_eos] = float("-inf")
            active_logits[:, codec_eos + 1 : suppress_stop] = float("-inf")
        else:
            active_logits[:, suppress_start:suppress_stop] = float("-inf")
        for row_index, scheduled_request in enumerate(requests):
            if len(scheduled_request.data.output_codes) < QWEN3_TTS_MIN_NEW_FRAMES:
                # note (ratish): a slice fills on the device; assigning one element copies a host scalar and waits for the stream.
                active_logits[row_index, codec_eos : codec_eos + 1] = float("-inf")
            else:
                pass

        # note (luojiaxuan): a cold-start clone that samples a silence id first
        # tends to stay silent for several frames, so silence ids are excluded
        # from its opening frames.
        if self.leading_silence_mask_frames > 0:
            for row_index, scheduled_request in enumerate(requests):
                if (
                    scheduled_request.data.mask_leading_silence
                    and len(scheduled_request.data.output_codes)
                    < self.leading_silence_mask_frames
                ):
                    active_logits[row_index].index_fill_(
                        0, self.silence_codec_ids, float("-inf")
                    )
                else:
                    pass
        else:
            pass

    def install_sampling_seeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Install the staged semantic seeds when a request in the batch carries a
        seed, or deterministic inference has already seeded every row."""
        sampling_info = forward_batch.sampling_info
        if sampling_info.sampling_seed is None and all(
            scheduled_request.data.req.sampling_params.sampling_seed is None
            for scheduled_request in requests
        ):
            return
        else:
            pass
        self.validate_seeded_sampling_supported(sampling_info)
        sampling_info.sampling_seed = self.model.semantic_sampling_seed_tensor[
            : len(requests)
        ]

    def _p19_native_tail_mark(
        self, stage: str, *, device: torch.device,
        requests: list[SchedulerRequest] | None = None,
    ) -> None:
        """Research-only: attribute post-code15 scheduler GPU work."""
        if os.environ.get("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM") not in (
            "ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER",
        ):
            return
        pending = getattr(self.model, "_p19_native_boundary_next_req", None)
        if pending is None:
            return
        if (requests is not None
                and (len(requests) != 1
                     or str(requests[0].request_id) != pending)):
            raise RuntimeError("P19_NATIVE_TAIL_SCHEDULER_REQUEST_DRIFT")
        from .code_predictor_spec_p19_diag import maybe_tail_runner_event

        maybe_tail_runner_event(
            self.model, stage, device=device, request_id=pending,
        )

    def collect_codes(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        if os.environ.get("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR"):
            from .code_predictor_spec_p17_activation import record

            record(
                "collect_codes_called_with_tokens"
                if result.next_token_ids is not None
                else "collect_codes_called_without_tokens",
                caller_module=__file__,
            )
        self.has_pending_code_step = False
        if result.next_token_ids is None:
            return
        else:
            pass
        layer0_codes = result.next_token_ids
        self._p19_native_tail_mark(
            "COLLECT_CODES_BEGIN",
            device=layer0_codes.device, requests=requests,
        )
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        # note(ratish): the layer 0 id is the step's only host read, staged ahead
        # of the predictor so the finalize wait covers the sample, not the predictor.
        self.stage_token_ids(result, result.next_token_ids)
        self._p19_native_tail_mark(
            "NEXT_FRAME_TOKEN_IDS_STAGED",
            device=layer0_codes.device, requests=requests,
        )

        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        else:
            pass
        semantic_positions = self.sample_positions(
            forward_batch, layer0_codes.device
        )
        self._p19_native_tail_mark(
            "NEXT_FRAME_SAMPLE_POSITIONS_READY",
            device=layer0_codes.device, requests=requests,
        )
        if os.environ.get("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR"):
            from .code_predictor_spec_p17_activation import record

            record("model_runner_collect_before_predictor", caller_module=__file__)
        # P18A is C1-only: bind an actual scheduler request identity
        # to the on-policy depth observations, not an inferred HTTP frame.
        p18_arm = os.environ.get("SGLANG_OMNI_QTTS_CP_P18_ARM_FILE")
        p18_on = bool(
            os.environ.get("SGLANG_OMNI_QTTS_CP_P18_DIR")
            and p18_arm and os.path.isfile(p18_arm)
        )
        if p18_on:
            if len(requests) != 1 or not requests[0].request_id:
                raise RuntimeError("P18_REQUIRES_EXACTLY_ONE_SCHEDULER_REQUEST")
            self.model._p18_current_request_id = str(requests[0].request_id)
        # P21 request-scoped serving observability, NOT a per-frame log.
        # The same binary runs A (normal Predictor Graph-ON) and B (K2),
        # with the only difference an explicit post-health ARM marker.
        p21_on = os.environ.get(
            "SGLANG_OMNI_QTTS_CP_P21_SERVING_K2"
        ) == "1"
        if p21_on:
            if len(requests) != 1 or not requests[0].request_id:
                raise RuntimeError("P21_SERVING_OBSERVABILITY_C1_ONLY")
            rid = str(requests[0].request_id)
            graph_before = self.model.predictor_graph_replay_count
            eager_before = self.model.predictor_eager_count
            self.model._p21_last_frame_stats = None
        try:
            self._p19_native_tail_mark(
                "NEXT_FRAME_BEFORE_CODE_PREDICTOR",
                device=layer0_codes.device, requests=requests,
            )
            self.model.code_predictor_forward(
                layer0_codes,
                hidden,
                semantic_positions=semantic_positions,
            )
            if p18_on:
                self._p19_native_tail_mark(
                    "COLLECT_CODES_PREDICTOR_RETURNED",
                    device=layer0_codes.device, requests=requests,
                )
            if p21_on:
                stats = self.__dict__.setdefault(
                    "_p21_request_stats", {}
                ).setdefault(rid, {
                    "predictor_calls": 0, "native_graph_replays": 0,
                    "native_eager_calls": 0, "candidate_frames": 0,
                    "k2_rounds": 0, "accepted_drafts": 0,
                    "rejected_drafts": 0,
                })
                stats["predictor_calls"] += 1
                stats["native_graph_replays"] += (
                    self.model.predictor_graph_replay_count - graph_before
                )
                stats["native_eager_calls"] += (
                    self.model.predictor_eager_count - eager_before
                )
                frame = getattr(self.model, "_p21_last_frame_stats", None)
                if frame is not None:
                    stats["candidate_frames"] += 1
                    stats["k2_rounds"] += frame["rounds"]
                    stats["accepted_drafts"] += frame["accepted_drafts"]
                    stats["rejected_drafts"] += frame["rejected_drafts"]
        finally:
            if p18_on:
                del self.model._p18_current_request_id
        self.has_pending_code_step = True

    def post_process_outputs(
        self,
        result: GenerationBatchResult | None,
        scheduler_output: SchedulerOutput | None,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        if not self.has_pending_code_step:
            return
        else:
            pass
        self.has_pending_code_step = False
        self._p19_native_tail_mark(
            "POST_PROCESS_BEGIN",
            device=self.model.output_codes.device,
            requests=scheduler_output.requests,
        )
        eos_id = int(self.model.config.codec_eos_token_id)
        # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
        # rows must stay views of a snapshot, never of the reused graph buffers.
        batch_size = len(scheduler_output.requests)
        codes_snap = self.model.output_codes[:batch_size].detach().clone()
        self._p19_native_tail_mark(
            "POST_PROCESS_CODES_SNAPSHOT",
            device=codes_snap.device,
            requests=scheduler_output.requests,
        )
        embeds_snap = self.model.output_embeds[:batch_size].detach().clone()
        self._p19_native_tail_mark(
            "POST_PROCESS_EMBEDS_SNAPSHOT",
            device=embeds_snap.device,
            requests=scheduler_output.requests,
        )
        codes_ready = None
        if self.codes_event_module is not None:
            codes_ready = self.codes_event_module.Event()
            codes_ready.record()
        else:
            pass
        self._p19_native_tail_mark(
            "POST_PROCESS_CODES_EVENT_RECORDED",
            device=embeds_snap.device,
            requests=scheduler_output.requests,
        )
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            # note (ratish): a middle prefill chunk samples at a prompt row; SGLang drops
            # that token, so it gives no frame, feedback or stream chunk either.
            if sched_req.data.req.inflight_middle_chunks > 0 or req_output.data is None:
                continue
            else:
                pass
            sched_req.data.speech_output_tokens += 1
            if int(req_output.data) == eos_id:
                continue
            else:
                pass
            code_chunk = codes_snap[row_idx]
            sched_req.data.output_codes.append(code_chunk)
            sched_req.data.latest_stream_code_chunk = code_chunk
            sched_req.data.codes_ready_event = codes_ready
            sched_req.data.pending_feedback_queue.append(embeds_snap[row_idx])
        self._p19_native_tail_mark(
            "POST_PROCESS_FEEDBACK_QUEUED",
            device=embeds_snap.device,
            requests=scheduler_output.requests,
        )

    def on_request_finished(self, request_id: str, req_data) -> None:
        if os.environ.get("SGLANG_OMNI_QTTS_CP_P21_SERVING_K2") != "1":
            return
        import hashlib
        stats = self.__dict__.setdefault(
            "_p21_request_stats", {}
        ).pop(str(request_id), None)
        if stats is None:
            raise RuntimeError("P21_REQUEST_FINISHED_WITHOUT_CODE_STATISTICS")
        request_hash = hashlib.sha256(str(request_id).encode()).hexdigest()[:12]
        print(
            "P21_REAL_SERVING_REQUEST "
            f"request_hash={request_hash} "
            + " ".join(f"{k}={v}" for k, v in sorted(stats.items())),
            flush=True,
        )

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if sched_req.data.req.inflight_middle_chunks > 0
        }

    def sample_positions(
        self, forward_batch: ForwardBatch, device: torch.device
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_decode():
            return forward_batch.positions.to(device=device, dtype=torch.long)
        else:
            seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)
            return (seq_lens - 1).clamp_min(0)

    def write_feedback_buffers(
        self, forward_batch: ForwardBatch | None, requests: list[SchedulerRequest]
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass
        self._p19_native_tail_mark(
            "WRITE_FEEDBACK_ENTER",
            device=forward_batch.input_ids.device, requests=requests,
        )
        decode_feedback_embedding = self.model.decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        else:
            pass
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        else:
            pass
        row_ids = self.decode_row_ids(batch_size, input_ids)
        weight = decode_feedback_embedding.weight
        device = weight.device
        dtype = weight.dtype
        feedback_rows: list[torch.Tensor] = []
        text_rows: list[torch.Tensor] = []
        for sched_req in requests:
            data: Qwen3TTSSGLangRequestData = sched_req.data
            inputs = QwenTalkerModelRunner.peek_next_decode_inputs(data)
            if inputs is None:
                raise RuntimeError(
                    f"Qwen3-TTS decode for {sched_req.request_id} has no queued "
                    "feedback embedding"
                )
            else:
                pass
            feedback, text = inputs
            feedback_rows.append(
                QwenTalkerModelRunner.decode_row(feedback, device=device, dtype=dtype)
            )
            text_rows.append(
                QwenTalkerModelRunner.decode_row(text, device=device, dtype=dtype)
            )
            QwenTalkerModelRunner.pop_next_decode_inputs(data)

        with torch.no_grad():
            target = weight[:batch_size]
            torch.stack(feedback_rows, dim=0, out=target)
            target.add_(torch.stack(text_rows, dim=0))
            # note(ratish): the history outlives the buffer, a retracted request
            # replays it in its re-prefill.
            history = target.detach().clone()
        self._p19_native_tail_mark(
            "WRITE_FEEDBACK_TARGET_HISTORY_READY",
            device=target.device, requests=requests,
        )
        for row_idx, sched_req in enumerate(requests):
            QwenTalkerModelRunner.append_decode_input_history(
                sched_req.data, history[row_idx]
            )
        # During graph decode, input_ids carries staged embedding row ids.
        input_ids[:batch_size].copy_(row_ids)
        self._p19_native_tail_mark(
            "WRITE_FEEDBACK_ROW_IDS_READY",
            device=input_ids.device, requests=requests,
        )

    def decode_row_ids(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        cached = self.row_ids_cache
        if (
            cached is None
            or cached.numel() < batch_size
            or cached.dtype != input_ids.dtype
            or cached.device != input_ids.device
        ):
            cached = torch.arange(
                max(batch_size, 64),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            self.row_ids_cache = cached
        else:
            pass
        return cached[:batch_size]

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for sched_req in requests:
            data: Qwen3TTSSGLangRequestData = sched_req.data
            req = data.req
            req_len = req.extend_len
            prefix_len = req.prefix_len
            if data.prefill_input_embeds is None:
                data.prefill_input_embeds = data.prompt_input_embeds
            else:
                pass
            if data.prefill_input_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
            else:
                pass
            piece = QwenTalkerModelRunner.projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=req_len,
                device=forward_batch.input_ids.device,
            )
            if piece is None or int(piece.shape[0]) != req_len:
                have = 0 if piece is None else int(piece.shape[0])
                raise RuntimeError(
                    f"Qwen3-TTS prefill embed mismatch for {req.rid}: "
                    f"have {have} rows, need {req_len}"
                )
            else:
                pass
            pieces.append(piece)
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

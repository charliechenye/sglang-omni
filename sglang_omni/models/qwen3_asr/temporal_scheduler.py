# SPDX-License-Identifier: Apache-2.0
"""Commit temporal verification into SGLang's ordinary request lifecycle."""

from __future__ import annotations

from array import array
from collections.abc import Iterable

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
)
from sglang_omni.models.qwen3_asr.temporal_prefill import Qwen3ASRTemporalPrefillAdder
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalBatchResult,
    Qwen3ASRTemporalRunnerOutput,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput, SchedulerOutput


class Qwen3ASRTemporalScheduler(OmniScheduler[Qwen3ASRRequestData]):
    def build_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        with self.request_admission_lock:
            plan = super().build_new_batch_prefill(running_batch)
            if plan.batch_to_run is not None:
                self.cur_batch = plan.batch_to_run
            else:
                pass
            return plan

    def get_prefill_adder_class(self) -> type[PrefillAdder]:
        return Qwen3ASRTemporalPrefillAdder

    def emit_stream_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
        skip_rids: Iterable[str] = (),
    ) -> None:
        if isinstance(model_output, Qwen3ASRTemporalRunnerOutput):
            return
        else:
            super().emit_stream_output(scheduler_output, model_output, skip_rids)

    def stream_output(
        self,
        requests: list[Req],
        return_logprob: bool = False,
        skip_req: Req | None = None,
    ) -> None:
        streamable = [
            request
            for request in requests
            if not (
                request is not skip_req
                and isinstance(request.omni_data, Qwen3ASRRequestData)
                and request.omni_data.temporal_request is not None
                and request.omni_data.temporal_request.admitted
            )
        ]
        if streamable:
            super().stream_output(
                streamable, return_logprob=return_logprob, skip_req=skip_req
            )
        else:
            pass

    @staticmethod
    def make_batch_result(
        mr_output: ModelRunnerOutput,
    ) -> GenerationBatchResult | Qwen3ASRTemporalBatchResult:
        native_result = OmniScheduler.make_batch_result(mr_output)
        if isinstance(mr_output, Qwen3ASRTemporalRunnerOutput):
            verification = mr_output.temporal_verification
            return Qwen3ASRTemporalBatchResult(
                logits_output=native_result.logits_output,
                next_token_ids=native_result.next_token_ids,
                can_run_cuda_graph=native_result.can_run_cuda_graph,
                temporal_verification=verification,
            )
        else:
            return native_result

    def process_batch_result(
        self,
        schedule_batch: ScheduleBatch,
        batch_result: GenerationBatchResult | Qwen3ASRTemporalBatchResult,
    ) -> None:
        if not isinstance(batch_result, Qwen3ASRTemporalBatchResult):
            super().process_batch_result(schedule_batch, batch_result)
            return
        else:
            pass
        with self.request_admission_lock:
            temporal: Qwen3ASRTemporalRequest | None = None
            try:
                if len(schedule_batch.reqs) != 1:
                    raise RuntimeError("Temporal verification requires one request")
                else:
                    pass
                request = schedule_batch.reqs[0]
                request_data = request.omni_data
                if not isinstance(request_data, Qwen3ASRRequestData):
                    raise RuntimeError("Temporal verification lost request data")
                else:
                    pass
                temporal = request_data.temporal_request
                if temporal is None or self.model_runner is None:
                    raise RuntimeError("Temporal verification lost request ownership")
                elif self.model_runner.execution_bridge is None:
                    raise RuntimeError("Temporal request has no FutureMap bridge")
                else:
                    pass
                cancelled = self.precommit_temporal_result(
                    schedule_batch, request, temporal, batch_result
                )
                try:
                    super().process_batch_result(schedule_batch, batch_result)
                finally:
                    temporal.admitted = False

                can_continue = (
                    not cancelled
                    and request.rid not in self.aborted_request_ids
                    and request.to_finish is None
                    and not request.is_retracted
                    and request.discard_output_reason is None
                )
                if can_continue:
                    self.emit_speculative_tokens(request, request_data)
                else:
                    pass
                if request.finished():
                    super().stream_output(
                        [request], return_logprob=schedule_batch.return_logprob
                    )
                elif can_continue:
                    self.model_runner.execution_bridge.publish_next_tokens(
                        schedule_batch, batch_result.next_token_ids
                    )
                else:
                    pass
            except Exception as exc:
                if temporal is not None:
                    temporal.admitted = False
                else:
                    pass
                self.handle_batch_failure(schedule_batch, exc)

    def precommit_temporal_result(
        self,
        schedule_batch: ScheduleBatch,
        request: Req,
        temporal: Qwen3ASRTemporalRequest,
        batch_result: Qwen3ASRTemporalBatchResult,
    ) -> bool:
        verification = batch_result.temporal_verification
        prompt_len = temporal.prompt_boundary
        candidate_end = prompt_len + len(temporal.candidate_token_ids)
        request_pool_index = request.kv.req_pool_idx
        if (
            not schedule_batch.forward_mode.is_extend()
            or len(schedule_batch.reqs) != 1
            or not temporal.admitted
            or request.rid != verification.request_id
            or request_pool_index is None
            or request.kv.kv_committed_len != candidate_end
            or request.kv.kv_allocated_len != candidate_end
            or request.kv.cache_protected_len != 0
            or request.kv.cache_inserted_len != 0
        ):
            raise RuntimeError("Temporal KV ownership changed before commit")
        else:
            pass

        cancelled = (
            request.finished()
            or request.is_retracted
            or request.to_finish is not None
            or request.discard_output_reason is not None
            or request.rid in self.aborted_request_ids
        )
        accepted = () if cancelled else verification.accepted_prefix
        owned_boundary = prompt_len + len(accepted)
        token_row = self.req_to_token_pool.req_to_token[request_pool_index]
        rejected_locations = token_row[owned_boundary:candidate_end].clone()
        output_ids = array("q", request.output_ids)
        output_ids.extend(accepted)
        fill_ids = request.origin_input_ids + output_ids
        mrope_positions = request.multimodal_inputs.mrope_positions[
            :, :prompt_len
        ].clone()

        request.output_ids = output_ids
        request.full_untruncated_fill_ids = fill_ids
        request.kv.kv_committed_len = owned_boundary
        request.kv.kv_allocated_len = owned_boundary
        request.extend_end = owned_boundary
        schedule_batch.seq_lens[0] = owned_boundary
        schedule_batch.seq_lens_cpu[0] = owned_boundary
        schedule_batch.orig_seq_lens[0] = owned_boundary
        schedule_batch.seq_lens_sum = None
        request.skip_radix_cache_insert = True
        request.multimodal_inputs.mrope_positions = mrope_positions
        token_row[owned_boundary:candidate_end] = 0
        if cancelled and not request.finished() and not request.is_retracted:
            if request.to_finish is None:
                request.to_finish = FINISH_ABORT("cancelled before temporal commit")
            else:
                pass
            if request.discard_output_reason is None:
                request.discard_output_reason = request.to_finish
            else:
                pass
        else:
            pass
        if rejected_locations.numel() > 0:
            self.token_to_kv_pool_allocator.free(rejected_locations)
        else:
            pass
        return cancelled


__all__ = ["Qwen3ASRTemporalScheduler"]

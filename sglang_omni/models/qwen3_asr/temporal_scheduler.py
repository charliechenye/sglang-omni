# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR temporal verification result commit adapter."""

from __future__ import annotations

import logging

from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRRequestData
from sglang_omni.models.qwen3_asr.temporal_prefill import Qwen3ASRTemporalPrefillAdder
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalBatchResult,
    Qwen3ASRTemporalRunnerOutput,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput

logger = logging.getLogger(__name__)


class Qwen3ASRTemporalScheduler(OmniScheduler[Qwen3ASRRequestData]):
    def get_prefill_adder_class(self) -> type[PrefillAdder]:
        return Qwen3ASRTemporalPrefillAdder

    def get_coalesced_batch_prefill(
        self, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        with self.request_admission_lock:
            return super().get_coalesced_batch_prefill(running_batch)

    @staticmethod
    def make_batch_result(
        mr_output: ModelRunnerOutput,
    ) -> GenerationBatchResult | Qwen3ASRTemporalBatchResult:
        native_result = OmniScheduler.make_batch_result(mr_output)
        if isinstance(mr_output, Qwen3ASRTemporalRunnerOutput):
            verification = mr_output.temporal_verification
            if verification is not None:
                return Qwen3ASRTemporalBatchResult(
                    native_result=native_result,
                    temporal_verification=verification,
                )
            else:
                pass
        else:
            pass
        return native_result

    def process_batch_result(
        self,
        schedule_batch: ScheduleBatch,
        batch_result: GenerationBatchResult | Qwen3ASRTemporalBatchResult,
    ) -> None:
        if not isinstance(batch_result, Qwen3ASRTemporalBatchResult):
            tracked_requests = []
            if not schedule_batch.forward_mode.is_extend():
                for request in schedule_batch.reqs:
                    request_data = request.omni_data
                    if isinstance(request_data, Qwen3ASRRequestData):
                        temporal_request = request_data.temporal_request
                        if (
                            temporal_request is not None
                            and temporal_request.verification_committed
                        ):
                            tracked_requests.append(
                                (
                                    request,
                                    temporal_request,
                                    not temporal_request.cancelled
                                    and request.rid not in self.aborted_request_ids
                                    and not request.is_retracted
                                    and request.discard_output_reason is None,
                                )
                            )
                        else:
                            pass
                    else:
                        pass
            else:
                pass
            super().process_batch_result(schedule_batch, batch_result)
            for request, temporal_request, count_round in tracked_requests:
                if count_round:
                    temporal_request.ordinary_decode_rounds += 1
                else:
                    temporal_request.cancelled = True
                if request.finished() or temporal_request.cancelled:
                    logger.info(
                        f"qwen3_asr_temporal_complete request_id={request.rid} "
                        f"candidate_tokens={len(temporal_request.candidate_token_ids)} "
                        f"accepted_tokens={temporal_request.accepted_token_count} "
                        f"rejected_tokens={temporal_request.rejected_token_count} "
                        f"zero_decode={temporal_request.zero_decode} "
                        f"ordinary_decode_rounds={temporal_request.ordinary_decode_rounds} "
                        f"cancelled={temporal_request.cancelled}"
                    )
                else:
                    pass
            return
        else:
            pass
        with self.request_admission_lock:
            try:
                if len(schedule_batch.reqs) != 1:
                    raise RuntimeError(
                        "Qwen3-ASR temporal diagnostics lost its isolated request"
                    )
                else:
                    pass
                request = schedule_batch.reqs[0]
                request_data = request.omni_data
                if not isinstance(request_data, Qwen3ASRRequestData):
                    raise RuntimeError(
                        "Qwen3-ASR temporal diagnostics lost request data"
                    )
                else:
                    pass
                temporal_request = request_data.temporal_request
                if temporal_request is None:
                    raise RuntimeError(
                        "Qwen3-ASR temporal diagnostics lost request state"
                    )
                else:
                    pass
                self.precommit_temporal_result(schedule_batch, batch_result)
            except Exception as exc:
                self.handle_batch_failure(schedule_batch, exc)
                return
            else:
                pass
            super().process_batch_result(schedule_batch, batch_result.native_result)
            temporal_request.admitted = False
            temporal_request.verification_committed = True
        if (
            not temporal_request.cancelled
            and request.rid not in self.aborted_request_ids
            and request.to_finish is None
            and not request.is_retracted
            and request.discard_output_reason is None
        ):
            temporal_request.zero_decode = request.finished() or (
                len(request.output_ids) >= request.sampling_params.max_new_tokens
            )
        else:
            temporal_request.cancelled = True
        logger.info(
            f"qwen3_asr_temporal_verification request_id={request.rid} "
            f"candidate_tokens={len(temporal_request.candidate_token_ids)} "
            f"accepted_tokens={temporal_request.accepted_token_count} "
            f"rejected_tokens={temporal_request.rejected_token_count} "
            f"zero_decode={temporal_request.zero_decode} "
            f"ordinary_decode_rounds={temporal_request.ordinary_decode_rounds} "
            f"cancelled={temporal_request.cancelled} "
            f"prefill_graph_replay={batch_result.native_result.can_run_cuda_graph}"
        )

    def precommit_temporal_result(
        self,
        schedule_batch: ScheduleBatch,
        batch_result: Qwen3ASRTemporalBatchResult,
    ) -> None:
        verification = batch_result.temporal_verification
        if (
            not schedule_batch.forward_mode.is_extend()
            or len(schedule_batch.reqs) != 1
            or verification.row_index != 0
        ):
            raise RuntimeError("Qwen3-ASR temporal result has an unsupported batch")
        else:
            pass
        request = schedule_batch.reqs[0]
        request_data = request.omni_data
        if not isinstance(request_data, Qwen3ASRRequestData):
            raise RuntimeError("Qwen3-ASR temporal result lost its request data")
        else:
            pass
        temporal_request = request_data.temporal_request
        if (
            temporal_request is None
            or not temporal_request.admitted
            or request.rid != verification.request_id
            or temporal_request.context.request_id != verification.request_id
            or temporal_request.prompt_boundary != verification.prompt_boundary
            or temporal_request.candidate_token_ids != verification.candidate_token_ids
            or len(request.origin_input_ids) != verification.prompt_boundary
            or len(request.output_ids) != 0
            or verification.accepted_prefix
            != verification.candidate_token_ids[: len(verification.accepted_prefix)]
            or verification.owned_kv_boundary
            != verification.prompt_boundary + len(verification.accepted_prefix)
            or len(request.full_untruncated_fill_ids)
            != verification.prompt_boundary + len(verification.candidate_token_ids)
            or tuple(request.full_untruncated_fill_ids[verification.prompt_boundary :])
            != verification.candidate_token_ids
            or request.extend_end != len(request.full_untruncated_fill_ids)
            or request.kv.req_pool_idx is None
            or request.kv.cache_protected_len != 0
            or request.kv.cache_inserted_len != 0
            or request.kv.kv_committed_len != len(request.full_untruncated_fill_ids)
            or request.kv.kv_allocated_len != len(request.full_untruncated_fill_ids)
            or schedule_batch.seq_lens_cpu[0] != len(request.full_untruncated_fill_ids)
            or request.multimodal_inputs is None
            or request.multimodal_inputs.mrope_positions is None
            or request.multimodal_inputs.mrope_positions.shape[1]
            != len(request.full_untruncated_fill_ids)
        ):
            raise RuntimeError("Qwen3-ASR temporal KV ownership changed before commit")
        else:
            pass
        next_token_ids = batch_result.native_result.next_token_ids
        if next_token_ids is None or next_token_ids.numel() != 1:
            raise RuntimeError("Qwen3-ASR temporal result is missing its pending token")
        else:
            pass

        prompt_boundary = verification.prompt_boundary
        candidate_end = prompt_boundary + len(verification.candidate_token_ids)
        owned_boundary = verification.owned_kv_boundary
        request_pool_index = request.kv.req_pool_idx
        if (
            request.finished()
            or request.is_retracted
            or request.to_finish is not None
            or request.discard_output_reason is not None
            or request.rid in self.aborted_request_ids
        ):
            accepted_prefix = ()
            owned_boundary = prompt_boundary
            temporal_request.cancelled = True
        else:
            accepted_prefix = verification.accepted_prefix
            temporal_request.cancelled = False
        token_row = self.req_to_token_pool.req_to_token[request_pool_index]
        rejected_locations = token_row[owned_boundary:candidate_end].clone()
        if rejected_locations.numel() > 0:
            self.token_to_kv_pool_allocator.free(rejected_locations)
        else:
            pass
        temporal_request.accepted_token_count = len(accepted_prefix)
        temporal_request.rejected_token_count = len(
            verification.candidate_token_ids
        ) - len(accepted_prefix)
        del request.full_untruncated_fill_ids[prompt_boundary:]
        request.output_ids.extend(accepted_prefix)
        request.refresh_fill_ids()
        request.multimodal_inputs.mrope_positions = (
            request.multimodal_inputs.mrope_positions[:, :prompt_boundary].clone()
        )
        request.kv.kv_committed_len = owned_boundary
        request.kv.kv_allocated_len = owned_boundary
        request.extend_end = owned_boundary
        request.skip_radix_cache_insert = True
        token_row[owned_boundary:candidate_end] = 0
        schedule_batch.seq_lens[0] = owned_boundary
        schedule_batch.seq_lens_cpu[0] = owned_boundary


__all__ = ["Qwen3ASRTemporalScheduler"]

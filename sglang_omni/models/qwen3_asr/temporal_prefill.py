# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR temporal prefill admission and graph selection."""

from __future__ import annotations

import logging

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS,
    AddReqResult,
    PrefillAdder,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_exec

from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
)
from sglang_omni.models.qwen3_asr.temporal_cuda_graph_runner import (
    Qwen3ASRTemporalPrefillCudaGraphRunner,
)
from sglang_omni.models.qwen3_asr.temporal_verification import (
    get_temporal_forward_inputs,
)

logger = logging.getLogger(__name__)


def install_temporal_prefill_graph_gate(
    graph_runner: PrefillCudaGraphRunner,
    *,
    enable_temporal_cuda_graph: bool = False,
) -> None:
    graph_runner.qwen3_asr_temporal_cuda_graph_enabled = enable_temporal_cuda_graph
    native_can_run_graph = getattr(
        graph_runner, "qwen3_asr_temporal_native_graph_check", None
    )
    if native_can_run_graph is None:
        native_can_run_graph = graph_runner.can_run_graph
        graph_runner.qwen3_asr_temporal_native_graph_check = native_can_run_graph
    else:
        pass
    if hasattr(graph_runner, "qwen3_asr_temporal_graph_gate_installed"):
        return
    else:
        pass

    def can_run_graph(forward_batch: ForwardBatch) -> bool:
        temporal_forward = get_temporal_forward_inputs(forward_batch)
        if temporal_forward is not None:
            supports_temporal_cuda_graph = isinstance(
                graph_runner, Qwen3ASRTemporalPrefillCudaGraphRunner
            )
            if (
                graph_runner.qwen3_asr_temporal_cuda_graph_enabled
                and supports_temporal_cuda_graph
            ):
                can_replay = native_can_run_graph(forward_batch)
                dispatch = "graph" if can_replay else "eager"
                reason = "native_eligibility" if not can_replay else "native_bcg"
            else:
                can_replay = False
                dispatch = "eager"
                reason = (
                    "unsupported_runner"
                    if graph_runner.qwen3_asr_temporal_cuda_graph_enabled
                    else "temporal_graph_disabled"
                )
            logger.info(
                f"qwen3_asr_temporal_dispatch request_id={temporal_forward.request_id} "
                f"graph_runner={type(graph_runner).__name__} dispatch={dispatch} "
                f"reason={reason}"
            )
            return can_replay
        else:
            return native_can_run_graph(forward_batch)

    graph_runner.can_run_graph = can_run_graph
    graph_runner.qwen3_asr_temporal_graph_gate_installed = True


class Qwen3ASRTemporalPrefillAdder(PrefillAdder):
    def add_one_req(
        self,
        req: Req,
        has_chunked_req: bool,
        truncation_align_size: int | None,
    ) -> AddReqResult:
        request = req
        request_data = request.omni_data
        if not isinstance(request_data, Qwen3ASRRequestData):
            return super().add_one_req(request, has_chunked_req, truncation_align_size)
        else:
            pass
        temporal_request = request_data.temporal_request
        if temporal_request is None or not self.can_admit_temporal(
            request,
            temporal_request,
            has_chunked_req=has_chunked_req,
        ):
            if temporal_request is not None:
                self.log_temporal_fallback(request, temporal_request)
            else:
                if (
                    request_data.temporal_fallback_reason is not None
                    and not request_data.temporal_fallback_logged
                ):
                    logger.info(
                        f"qwen3_asr_temporal_fallback request_id={request.rid} "
                        "candidate_tokens=0 "
                        f"reason={request_data.temporal_fallback_reason}"
                    )
                    request_data.temporal_fallback_logged = True
                else:
                    pass
            return super().add_one_req(request, has_chunked_req, truncation_align_size)
        else:
            pass

        canonical_prompt_token_count = len(request.full_untruncated_fill_ids)
        canonical_full_fill_ids = request.full_untruncated_fill_ids[:]
        multimodal_inputs = request.multimodal_inputs
        if multimodal_inputs is None or multimodal_inputs.mrope_positions is None:
            temporal_request.fallback_reason = "missing_mrope_positions"
            self.log_temporal_fallback(request, temporal_request)
            return super().add_one_req(request, has_chunked_req, truncation_align_size)
        else:
            pass
        canonical_mrope_positions = multimodal_inputs.mrope_positions
        try:
            speculative_fill_ids = canonical_full_fill_ids[:]
            speculative_fill_ids.extend(temporal_request.candidate_token_ids)
            request.full_untruncated_fill_ids = speculative_fill_ids
            extend_token_count = (
                len(request.full_untruncated_fill_ids) - request.prefix_len
            )
            remaining_new_token_count = min(
                max(
                    request.sampling_params.max_new_tokens - len(request.output_ids),
                    0,
                ),
                CLIP_MAX_NEW_TOKENS,
            )
            mamba_gap_token_budget = self._mamba_gap_budget_for_req(
                request, include_host_hit=True
            )  # noqa: leading-underscore  # pinned SGLang 0.5.23 admission contract
            request_slot_token_reserve = self._request_slot_reserve_for_req(
                request
            )  # noqa: leading-underscore  # pinned SGLang 0.5.23 admission contract
            required_token_count = (
                extend_token_count
                + remaining_new_token_count
                + self.per_req_token_overhead
                + mamba_gap_token_budget
                + request_slot_token_reserve
            )
            admission = self._select_prefill_admission(  # noqa: leading-underscore  # pinned SGLang 0.5.23 pure preflight selector
                request,
                total_tokens=required_token_count,
                host_hit_length=request.host_hit_length,
                swa_host_hit_length=request.swa_host_hit_length,
                truncation_align_size=truncation_align_size,
                has_chunked_req=has_chunked_req,
            )
            candidate_positions = torch.arange(
                canonical_prompt_token_count,
                canonical_prompt_token_count
                + len(temporal_request.candidate_token_ids),
                dtype=canonical_mrope_positions.dtype,
                device=canonical_mrope_positions.device,
            )
            candidate_positions = candidate_positions.unsqueeze(0).expand(3, -1)
            temporal_mrope_positions = torch.cat(
                [canonical_mrope_positions, candidate_positions], dim=1
            )
        except Exception:
            request.full_untruncated_fill_ids = canonical_full_fill_ids
            multimodal_inputs.mrope_positions = canonical_mrope_positions
            raise
        if (
            isinstance(admission, AddReqResult)
            or admission.is_chunked
            or admission.prefix_len != request.prefix_len
            or admission.extend_len != extend_token_count
        ):
            request.full_untruncated_fill_ids = canonical_full_fill_ids
            temporal_request.fallback_reason = "prefill_budget_or_shape"
            self.log_temporal_fallback(request, temporal_request)
            return super().add_one_req(request, has_chunked_req, truncation_align_size)
        else:
            pass
        multimodal_inputs.mrope_positions = temporal_mrope_positions
        try:
            admission_result = super().add_one_req(
                request,
                has_chunked_req,
                truncation_align_size,
            )
        except Exception:
            if (
                not any(
                    admitted_request is request
                    for admitted_request in self.can_run_list
                )
                and not request.kv.holds_kv
                and request.kv.kv_allocated_len == 0
            ):
                request.full_untruncated_fill_ids = canonical_full_fill_ids
                multimodal_inputs.mrope_positions = canonical_mrope_positions
            else:
                pass
            raise
        admitted_temporal_request = any(
            admitted_request is request for admitted_request in self.can_run_list
        )
        if not admitted_temporal_request:
            if not request.kv.holds_kv and request.kv.kv_allocated_len == 0:
                request.full_untruncated_fill_ids = canonical_full_fill_ids
                multimodal_inputs.mrope_positions = canonical_mrope_positions
            else:
                raise RuntimeError(
                    "SGLang rejected a temporal request after acquiring KV ownership"
                )
            temporal_request.fallback_reason = "native_admission_rejected"
            self.log_temporal_fallback(request, temporal_request)
            return admission_result
        elif request.extend_end != len(request.full_untruncated_fill_ids):
            raise RuntimeError("SGLang changed a temporal request into chunked prefill")
        else:
            pass
        temporal_request.admitted = True
        temporal_request.attempted = True
        logger.info(
            f"qwen3_asr_temporal_eligible request_id={request.rid} "
            f"candidate_tokens={len(temporal_request.candidate_token_ids)}"
        )
        logger.info(
            f"qwen3_asr_temporal_attempt request_id={request.rid} "
            f"candidate_tokens={len(temporal_request.candidate_token_ids)}"
        )
        return admission_result

    @staticmethod
    def log_temporal_fallback(
        request: Req, temporal_request: Qwen3ASRTemporalRequest
    ) -> None:
        if temporal_request.fallback_logged:
            return
        else:
            pass
        logger.info(
            f"qwen3_asr_temporal_fallback request_id={request.rid} "
            f"candidate_tokens={len(temporal_request.candidate_token_ids)} "
            f"reason={temporal_request.fallback_reason}"
        )
        temporal_request.fallback_logged = True

    def can_admit_temporal(
        self,
        request: Req,
        temporal_request: Qwen3ASRTemporalRequest,
        *,
        has_chunked_req: bool,
    ) -> bool:
        candidate_token_ids = temporal_request.candidate_token_ids
        graph_backend = get_exec().graph.cuda_graph_config.prefill.backend
        if graph_backend not in (Backend.DISABLED, Backend.BREAKABLE):
            temporal_request.fallback_reason = "unsupported_prefill_graph_backend"
            return False
        else:
            pass
        if (
            not candidate_token_ids
            or temporal_request.admitted
            or temporal_request.attempted
            or temporal_request.prompt_boundary != len(request.origin_input_ids)
            or temporal_request.context.request_id != request.rid
            or not temporal_request.context.is_final
            or len(self.can_run_list) != 0
            or self.running_batch is None
            or self.running_batch.reqs
            or not self.running_batch.spec_algorithm.is_none()
            or self.waiting_queue_len != 1
            or self.page_size != 1
            or not isinstance(self.token_to_kv_pool_allocator, TokenToKVPoolAllocator)
            or self.is_hybrid_swa
            or self.is_hybrid_ssm_cache
            or self.dllm_config is not None
            or has_chunked_req
            or request.finished()
            or request.to_finish is not None
            or request.discard_output_reason is not None
            or request.session is not None
            or request.session_id is not None
            or request.is_retracted
            or request.prefix_len != 0
            or request.host_hit_length != 0
            or request.swa_host_hit_length != 0
            or request.needs_host_load_back()
            or request.output_ids
            or request.kv.cache_protected_len != 0
            or request.kv.cache_inserted_len != 0
            or request.return_logprob
            or request.return_sampling_mask
            or request.custom_logit_processor is not None
            or request.grammar is not None
            or request.multimodal_inputs is None
            or request.multimodal_inputs.mrope_positions is None
            or request.multimodal_inputs.mrope_positions.shape[0] != 3
            or request.multimodal_inputs.mrope_positions.shape[1]
            != len(request.origin_input_ids)
            or len(request.full_untruncated_fill_ids) != len(request.origin_input_ids)
            or request.full_untruncated_fill_ids is request.origin_input_ids
            or request.full_untruncated_fill_ids != request.origin_input_ids
            or request.sampling_params.temperature != 1
            or request.sampling_params.top_p != 1
            or request.sampling_params.top_k != 1
            or request.sampling_params.min_p != 0
            or request.sampling_params.min_new_tokens != 0
            or request.sampling_params.ignore_eos
            or request.sampling_params.repetition_penalty != 1
            or request.sampling_params.frequency_penalty != 0
            or request.sampling_params.presence_penalty != 0
            or request.sampling_params.custom_params
            or request.sampling_params.stop_strs
            or request.sampling_params.stop_regex_strs
            or request.sampling_params.max_new_tokens - len(request.output_ids)
            < len(candidate_token_ids) + 1
        ):
            temporal_request.fallback_reason = "unsupported_admission_state"
            return False
        else:
            pass
        temporal_request.fallback_reason = None
        return True


__all__ = ["Qwen3ASRTemporalPrefillAdder", "install_temporal_prefill_graph_gate"]

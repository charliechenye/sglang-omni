# SPDX-License-Identifier: Apache-2.0
"""Admission for isolated Qwen3-ASR temporal final requests."""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS,
    AddReqResult,
    PrefillAdder,
)

from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
)


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
        temporal = request_data.temporal_request
        if temporal is None or not self.can_admit_temporal(
            request, temporal, has_chunked_req
        ):
            return super().add_one_req(request, has_chunked_req, truncation_align_size)
        else:
            pass

        multimodal_inputs = request.multimodal_inputs
        if multimodal_inputs is None or multimodal_inputs.mrope_positions is None:
            raise RuntimeError("Qwen3-ASR request builder omitted temporal mRoPE")
        else:
            pass
        prompt_len = len(request.full_untruncated_fill_ids)
        assert temporal.prompt_boundary == prompt_len
        canonical_fill_ids = request.full_untruncated_fill_ids[:]
        canonical_mrope_positions = multimodal_inputs.mrope_positions
        candidate_end = prompt_len + len(temporal.candidate_token_ids)
        try:
            speculative_fill_ids = canonical_fill_ids[:]
            speculative_fill_ids.extend(temporal.candidate_token_ids)
            request.full_untruncated_fill_ids = speculative_fill_ids
            candidate_positions = torch.arange(
                prompt_len,
                candidate_end,
                dtype=canonical_mrope_positions.dtype,
                device=canonical_mrope_positions.device,
            ).expand(3, -1)
            temporal_mrope_positions = torch.cat(
                (canonical_mrope_positions, candidate_positions), dim=1
            )
            extend_token_count = candidate_end - request.prefix_len
            remaining_new_tokens = min(
                request.sampling_params.max_new_tokens - len(request.output_ids),
                CLIP_MAX_NEW_TOKENS,
            )
            required_tokens = (
                extend_token_count
                + remaining_new_tokens
                + self.per_req_token_overhead
                + self._mamba_gap_budget_for_req(request, include_host_hit=True)
                + self._request_slot_reserve_for_req(request)
            )  # noqa: leading-underscore  # pinned SGLang 0.5.23 admission contract
            admission = self._select_prefill_admission(  # noqa: leading-underscore  # pinned SGLang 0.5.23 pure preflight selector
                request,
                total_tokens=required_tokens,
                host_hit_length=request.host_hit_length,
                swa_host_hit_length=request.swa_host_hit_length,
                truncation_align_size=truncation_align_size,
                has_chunked_req=has_chunked_req,
            )
        except Exception:
            request.full_untruncated_fill_ids = canonical_fill_ids
            raise

        if (
            isinstance(admission, AddReqResult)
            or admission.is_chunked
            or admission.prefix_len != request.prefix_len
            or admission.extend_len != extend_token_count
        ):
            request.full_untruncated_fill_ids = canonical_fill_ids
            return super().add_one_req(request, has_chunked_req, truncation_align_size)
        else:
            pass

        multimodal_inputs.mrope_positions = temporal_mrope_positions
        try:
            admission_result = super().add_one_req(
                request, has_chunked_req, truncation_align_size
            )
        except Exception:
            is_admitted = any(item is request for item in self.can_run_list)
            if (
                not is_admitted
                and not request.kv.holds_kv
                and request.kv.kv_allocated_len == 0
            ):
                request.full_untruncated_fill_ids = canonical_fill_ids
                multimodal_inputs.mrope_positions = canonical_mrope_positions
            else:
                pass
            raise
        if not any(item is request for item in self.can_run_list):
            if request.kv.holds_kv or request.kv.kv_allocated_len != 0:
                raise RuntimeError(
                    "Native admission acquired KV for a rejected request"
                )
            else:
                pass
            request.full_untruncated_fill_ids = canonical_fill_ids
            multimodal_inputs.mrope_positions = canonical_mrope_positions
            return admission_result
        elif request.extend_end != len(request.full_untruncated_fill_ids):
            raise RuntimeError("Temporal request was changed into chunked prefill")
        else:
            temporal.admitted = True
            return admission_result

    def can_admit_temporal(
        self,
        request: Req,
        temporal: Qwen3ASRTemporalRequest,
        has_chunked_req: bool,
    ) -> bool:
        candidate_token_ids = temporal.candidate_token_ids
        if (
            not candidate_token_ids
            or temporal.admitted
            or temporal.prompt_boundary != len(request.origin_input_ids)
            or self.can_run_list
            or self.running_batch.reqs
            or self.waiting_queue_len != 1
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
            or len(request.full_untruncated_fill_ids) != len(request.origin_input_ids)
            or request.sampling_params.temperature != 1
            or request.sampling_params.repetition_penalty != 1
            or request.sampling_params.max_new_tokens - len(request.output_ids)
            < len(candidate_token_ids) + 1
        ):
            return False
        else:
            return True


__all__ = ["Qwen3ASRTemporalPrefillAdder"]

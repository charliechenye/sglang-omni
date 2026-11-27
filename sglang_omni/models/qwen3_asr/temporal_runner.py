# SPDX-License-Identifier: Apache-2.0
"""Opt-in model runner hooks for Qwen3-ASR temporal verification."""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.qwen3_asr.engine_builder import Qwen3ASRModelRunner
from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRRequestData
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalForward,
    Qwen3ASRTemporalLogitsProcessorOutput,
    Qwen3ASRTemporalRunnerOutput,
    attach_temporal_forward_inputs,
    clear_temporal_forward_inputs,
)
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class Qwen3ASRTemporalModelRunner(Qwen3ASRModelRunner):
    supports_speculative_async_decode: bool = False

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        if not requests or not isinstance(requests[0].data, Qwen3ASRRequestData):
            return
        else:
            pass
        request_data = requests[0].data
        temporal_request = request_data.temporal_request
        if temporal_request is None:
            return
        else:
            pass
        if not temporal_request.admitted:
            return
        elif (
            forward_batch is None or len(requests) != 1 or len(schedule_batch.reqs) != 1
        ):
            raise RuntimeError("Qwen3-ASR temporal forward lost its isolated row")
        else:
            pass
        if temporal_request.prompt_boundary + len(
            temporal_request.candidate_token_ids
        ) != len(schedule_batch.input_ids):
            raise RuntimeError(
                "Qwen3-ASR temporal input boundary changed before forward"
            )
        else:
            pass
        attach_temporal_forward_inputs(
            forward_batch,
            Qwen3ASRTemporalForward(
                request_id=requests[0].request_id,
                prompt_boundary=temporal_request.prompt_boundary,
                candidate_token_ids=temporal_request.candidate_token_ids,
            ),
        )

    def cleanup_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch, requests
        clear_temporal_forward_inputs(forward_batch)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        if not requests or not isinstance(requests[0].data, Qwen3ASRRequestData):
            return
        else:
            pass
        temporal_request = requests[0].data.temporal_request
        if temporal_request is None or not temporal_request.admitted:
            return
        else:
            pass
        logits_output = result.logits_output
        if not isinstance(logits_output, Qwen3ASRTemporalLogitsProcessorOutput):
            raise RuntimeError(
                "Qwen3-ASR admitted temporal prefill did not return a verification"
            )
        else:
            pass
        verification = logits_output.temporal_verification
        pending_token_ids = logits_output.pending_token_ids
        if (
            verification is None
            or pending_token_ids is None
            or pending_token_ids.numel() != 1
            or len(schedule_batch.reqs) != 1
            or len(schedule_batch.input_ids)
            != temporal_request.prompt_boundary
            + len(temporal_request.candidate_token_ids)
        ):
            raise RuntimeError(
                "Qwen3-ASR temporal target rows differ from the admitted prefill"
            )
        elif requests[0].request_id != temporal_request.context.request_id:
            raise RuntimeError("Qwen3-ASR temporal result lost request ownership")
        else:
            pass
        if (
            verification.request_id != requests[0].request_id
            or verification.prompt_boundary != temporal_request.prompt_boundary
            or verification.candidate_token_ids != temporal_request.candidate_token_ids
        ):
            raise RuntimeError("Qwen3-ASR temporal result lost request ownership")
        else:
            pass
        logits_output.temporal_verification = verification
        result.next_token_ids = pending_token_ids

    def next_input_token_ids(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        if (
            len(requests) == 1
            and isinstance(requests[0].data, Qwen3ASRRequestData)
            and requests[0].data.temporal_request is not None
            and requests[0].data.temporal_request.admitted
        ):
            return None
        else:
            return super().next_input_token_ids(result, forward_batch, requests)

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        return {
            request.request_id
            for request in scheduler_output.requests
            if isinstance(request.data, Qwen3ASRRequestData)
            and request.data.temporal_request is not None
            and request.data.temporal_request.admitted
        }

    def finalize(
        self,
        batch_result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        scheduler_output: SchedulerOutput,
        skip_rids: set[str] | None = None,
    ) -> ModelRunnerOutput:
        output = super().finalize(
            batch_result,
            forward_batch,
            schedule_batch,
            scheduler_output,
            skip_rids=skip_rids,
        )
        logits_output = batch_result.logits_output
        if not isinstance(logits_output, Qwen3ASRTemporalLogitsProcessorOutput):
            return output
        else:
            pass
        verification = logits_output.temporal_verification
        if verification is None:
            return output
        else:
            pass
        output.outputs[verification.request_id].data = None
        return Qwen3ASRTemporalRunnerOutput(
            outputs=output.outputs,
            req_ids=output.req_ids,
            req_id_to_index=output.req_id_to_index,
            can_run_cuda_graph=output.can_run_cuda_graph,
            next_token_ids=output.next_token_ids,
            host_token_ids=output.host_token_ids,
            temporal_verification=verification,
        )


__all__ = ["Qwen3ASRTemporalModelRunner"]

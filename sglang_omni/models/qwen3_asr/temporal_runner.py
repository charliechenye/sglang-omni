# SPDX-License-Identifier: Apache-2.0
"""Model runner hooks for Qwen3-ASR temporal verification."""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.qwen3_asr.engine_builder import Qwen3ASRModelRunner
from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRRequestData
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalForwardContext,
    Qwen3ASRTemporalLogitsProcessorOutput,
    Qwen3ASRTemporalRunnerOutput,
    attach_temporal_forward_inputs,
    clear_temporal_forward_inputs,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class Qwen3ASRTemporalModelRunner(Qwen3ASRModelRunner):
    supports_speculative_async_decode: bool = False

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.temporal_context: Qwen3ASRTemporalForwardContext | None = None

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
        temporal = requests[0].data.temporal_request
        if temporal is None or not temporal.admitted:
            return
        elif (
            forward_batch is None or len(requests) != 1 or len(schedule_batch.reqs) != 1
        ):
            raise RuntimeError("Qwen3-ASR temporal prefill must contain one request")
        elif self.temporal_context is not None:
            raise RuntimeError("Qwen3-ASR temporal forward is already active")
        else:
            pass
        if len(schedule_batch.input_ids) != temporal.prompt_boundary + len(
            temporal.candidate_token_ids
        ):
            raise RuntimeError("Qwen3-ASR temporal prefill boundary changed")
        else:
            pass
        self.temporal_context = attach_temporal_forward_inputs(
            requests[0].request_id,
            temporal.prompt_boundary,
            temporal.candidate_token_ids,
        )

    def cleanup_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch, requests
        context = self.temporal_context
        if context is not None:
            clear_temporal_forward_inputs(context)
            self.temporal_context = None
        else:
            pass

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
        temporal = requests[0].data.temporal_request
        if temporal is None or not temporal.admitted:
            return
        elif not isinstance(
            result.logits_output, Qwen3ASRTemporalLogitsProcessorOutput
        ):
            raise RuntimeError("Qwen3-ASR temporal prefill returned no verification")
        else:
            pass
        logits_output = result.logits_output
        verification = logits_output.temporal_verification
        pending_token_ids = logits_output.pending_token_ids
        if (
            verification is None
            or verification.request_id != requests[0].request_id
            or pending_token_ids is None
            or pending_token_ids.numel() != 1
        ):
            raise RuntimeError("Qwen3-ASR temporal verification lost request ownership")
        elif result.can_run_cuda_graph != (logits_output.graph_bucket_size is not None):
            raise RuntimeError("Qwen3-ASR temporal graph metadata was lost")
        else:
            pass
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

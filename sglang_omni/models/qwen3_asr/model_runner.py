# SPDX-License-Identifier: Apache-2.0
"""Model runners for Qwen3-ASR."""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRRequestData
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalForwardContext,
    Qwen3ASRTemporalLogitsProcessorOutput,
    Qwen3ASRTemporalRunnerOutput,
    attach_temporal_forward_inputs,
    clear_temporal_forward_inputs,
)
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class Qwen3ASRModelRunner(ModelRunner[Qwen3ASRRequestData]):
    """Shared autoregressive runner for Qwen3-ASR."""

    supports_speculative_async_decode: bool = True
    speculative_repetition_penalty_owner = (
        "verify_logits" if current_platform.is_cuda() else None
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
        del forward_batch
        if not requests:
            return
        else:
            pass
        temporal = requests[0].data.temporal_request
        if temporal is None or not temporal.admitted:
            return
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
        del forward_batch, schedule_batch
        if not requests:
            return
        else:
            pass
        temporal = requests[0].data.temporal_request
        if temporal is None or not temporal.admitted:
            return
        else:
            pass
        logits_output = result.logits_output
        assert isinstance(logits_output, Qwen3ASRTemporalLogitsProcessorOutput)
        result.next_token_ids = logits_output.pending_token_ids

    def next_input_token_ids(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        if (
            len(requests) == 1
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
            if request.data.temporal_request is not None
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
        if not isinstance(
            batch_result.logits_output, Qwen3ASRTemporalLogitsProcessorOutput
        ):
            return output
        else:
            pass
        verification = batch_result.logits_output.temporal_verification
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


__all__ = ["Qwen3ASRModelRunner", "Qwen3ASRTemporalModelRunner"]

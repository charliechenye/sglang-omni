# SPDX-License-Identifier: Apache-2.0
"""Opt-in model runner hooks for Qwen3-ASR temporal verification."""

from __future__ import annotations

import logging

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.qwen3_asr.engine_builder import Qwen3ASRModelRunner
from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRRequestData
from sglang_omni.models.qwen3_asr.temporal_prefill import (
    install_temporal_prefill_graph_gate,
)
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalForward,
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

logger = logging.getLogger(__name__)


class Qwen3ASRTemporalModelRunner(Qwen3ASRModelRunner):
    supports_speculative_async_decode: bool = False

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.temporal_forward_context_stack: list[
            Qwen3ASRTemporalForwardContext | None
        ] = []

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.temporal_forward_context_stack.append(None)
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
        temporal_context = attach_temporal_forward_inputs(
            forward_batch,
            Qwen3ASRTemporalForward(
                request_id=requests[0].request_id,
                prompt_boundary=temporal_request.prompt_boundary,
                candidate_token_ids=temporal_request.candidate_token_ids,
            ),
        )
        self.temporal_forward_context_stack[-1] = temporal_context
        graph_runner = self.tp_worker.model_runner.prefill_cuda_graph_runner
        if isinstance(graph_runner, PrefillCudaGraphRunner):
            install_temporal_prefill_graph_gate(graph_runner)
        else:
            pass

    def cleanup_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch, requests
        if not self.temporal_forward_context_stack:
            return
        else:
            pass
        temporal_context = self.temporal_forward_context_stack[-1]
        if temporal_context is not None:
            if forward_batch is None:
                raise RuntimeError("Qwen3-ASR cleanup lost its owning forward batch")
            else:
                clear_temporal_forward_inputs(forward_batch, temporal_context)
        else:
            pass
        self.temporal_forward_context_stack.pop()

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
        if result.can_run_cuda_graph:
            logger.error(
                f"qwen3_asr_temporal_verification_failed "
                f"request_id={requests[0].request_id} dispatch=graph "
                f"logits_output_type={type(logits_output).__name__} "
                "reason=admitted_prefill_used_cuda_graph"
            )
            raise RuntimeError(
                "Qwen3-ASR admitted temporal prefill unexpectedly used a CUDA graph"
            )
        else:
            pass
        if not isinstance(logits_output, Qwen3ASRTemporalLogitsProcessorOutput):
            logger.error(
                f"qwen3_asr_temporal_verification_failed "
                f"request_id={requests[0].request_id} dispatch=eager "
                f"logits_output_type={type(logits_output).__name__} "
                "reason=missing_temporal_verification"
            )
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
        logger.info(
            f"qwen3_asr_temporal_verified request_id={requests[0].request_id} "
            f"dispatch=eager logits_output_type={type(logits_output).__name__} "
            f"accepted_candidate_tokens={len(verification.accepted_prefix)} "
            f"rejected_candidate_tokens="
            f"{len(verification.candidate_token_ids) - len(verification.accepted_prefix)}"
        )

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

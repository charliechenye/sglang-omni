# SPDX-License-Identifier: Apache-2.0
"""Breakable prefill graph adapter for Qwen3-ASR temporal verification."""

from __future__ import annotations

from dataclasses import fields

from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

from sglang_omni.models.qwen3_asr.temporal_verification import (
    QWEN3_ASR_TEMPORAL_FORWARD_ATTR,
    Qwen3ASRTemporalGraphReplay,
    Qwen3ASRTemporalLogitsProcessorOutput,
    get_temporal_forward_context,
)


class Qwen3ASRTemporalPrefillCudaGraphRunner(PrefillCudaGraphRunner):
    def load_batch(
        self, forward_batch: ForwardBatch, **kwargs
    ) -> tuple[ForwardBatch, SharedReadEnds]:
        temporal_context = get_temporal_forward_context(forward_batch)
        if temporal_context is None:
            return super().load_batch(forward_batch, **kwargs)
        else:
            pass
        temporal_forward = temporal_context.temporal_forward
        raw_num_tokens = len(forward_batch.input_ids)
        if (
            getattr(forward_batch, QWEN3_ASR_TEMPORAL_FORWARD_ATTR, None)
            is not temporal_context
            or temporal_context.owner_forward_batch_id != id(forward_batch)
            or forward_batch.rids != [temporal_forward.request_id]
            or forward_batch.batch_size != 1
            or raw_num_tokens
            != temporal_forward.prompt_boundary
            + len(temporal_forward.candidate_token_ids)
            or forward_batch.extend_seq_lens_cpu != [raw_num_tokens]
            or forward_batch.extend_prefix_lens_cpu != [0]
            or temporal_context.graph_replay is not None
        ):
            raise RuntimeError(
                "Qwen3-ASR temporal graph replay lost its owning raw request"
            )
        else:
            pass
        static_forward_batch, shared_read_ends = super().load_batch(
            forward_batch, **kwargs
        )
        if (
            static_forward_batch is forward_batch
            or self.raw_num_tokens != raw_num_tokens
            or len(static_forward_batch.input_ids) < raw_num_tokens
        ):
            raise RuntimeError("Qwen3-ASR temporal graph replay has an invalid bucket")
        else:
            pass
        static_forward_batch.rids = list(forward_batch.rids)
        temporal_context.graph_replay = Qwen3ASRTemporalGraphReplay(
            request_id=temporal_forward.request_id,
            raw_num_tokens=self.raw_num_tokens,
            graph_bucket_size=len(static_forward_batch.input_ids),
        )
        return static_forward_batch, shared_read_ends

    def _trim_logits_output(  # noqa: leading-underscore  # pinned SGLang trim override
        self, output: LogitsProcessorOutput
    ) -> LogitsProcessorOutput:
        trimmed_output = super()._trim_logits_output(  # noqa: leading-underscore  # pinned SGLang trim hook
            output
        )
        if not isinstance(output, Qwen3ASRTemporalLogitsProcessorOutput):
            return trimmed_output
        else:
            pass
        return Qwen3ASRTemporalLogitsProcessorOutput(
            **{
                field.name: getattr(trimmed_output, field.name)
                for field in fields(LogitsProcessorOutput)
            },
            temporal_verification=output.temporal_verification,
            pending_token_ids=output.pending_token_ids,
            graph_replay=output.graph_replay,
        )


__all__ = ["Qwen3ASRTemporalPrefillCudaGraphRunner"]

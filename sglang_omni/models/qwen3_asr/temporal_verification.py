# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR final-request greedy verification."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, fields, replace

import torch
from sglang.srt.layers.aux_hidden_states import AuxHiddenStates
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from sglang_omni.scheduling.types import ModelRunnerOutput


@dataclass(slots=True)
class Qwen3ASRTemporalForwardContext:
    request_id: str
    prompt_boundary: int
    candidate_token_ids: tuple[int, ...]
    context_token: Token[Qwen3ASRTemporalForwardContext | None] | None = None
    graph_bucket_size: int | None = None


_TEMPORAL_FORWARD: ContextVar[Qwen3ASRTemporalForwardContext | None] = ContextVar(
    "qwen3_asr_temporal_forward", default=None
)


@dataclass(frozen=True, slots=True)
class Qwen3ASRTemporalVerification:
    request_id: str
    accepted_prefix: tuple[int, ...]
    pending_token_id: int


@dataclass(slots=True)
class Qwen3ASRTemporalLogitsProcessorOutput(LogitsProcessorOutput):
    temporal_verification: Qwen3ASRTemporalVerification | None = None
    pending_token_ids: torch.Tensor | None = None
    graph_bucket_size: int | None = None


@dataclass(slots=True)
class Qwen3ASRTemporalRunnerOutput(ModelRunnerOutput):
    temporal_verification: Qwen3ASRTemporalVerification | None = None


@dataclass(slots=True, kw_only=True)
class Qwen3ASRTemporalBatchResult(GenerationBatchResult):
    temporal_verification: Qwen3ASRTemporalVerification


def attach_temporal_forward_inputs(
    request_id: str,
    prompt_boundary: int,
    candidate_token_ids: tuple[int, ...],
) -> Qwen3ASRTemporalForwardContext:
    if _TEMPORAL_FORWARD.get() is not None:
        raise RuntimeError("Qwen3-ASR temporal forward context is already active")
    else:
        pass
    context = Qwen3ASRTemporalForwardContext(
        request_id=request_id,
        prompt_boundary=prompt_boundary,
        candidate_token_ids=candidate_token_ids,
    )
    context.context_token = _TEMPORAL_FORWARD.set(context)
    return context


def get_temporal_forward_context(
    forward_batch: ForwardBatch,
) -> Qwen3ASRTemporalForwardContext | None:
    context = _TEMPORAL_FORWARD.get()
    if context is None:
        return None
    elif forward_batch.rids != [context.request_id]:
        raise RuntimeError("Qwen3-ASR temporal context belongs to another request")
    else:
        return context


def clear_temporal_forward_inputs(
    context: Qwen3ASRTemporalForwardContext,
) -> None:
    if context.context_token is None or _TEMPORAL_FORWARD.get() is not context:
        raise RuntimeError("Qwen3-ASR temporal cleanup does not own the active context")
    else:
        pass
    _TEMPORAL_FORWARD.reset(context.context_token)
    context.context_token = None


def verify_greedy_candidates(
    request_id: str,
    candidate_token_ids: tuple[int, ...],
    target_token_ids: list[int],
) -> Qwen3ASRTemporalVerification:
    if len(target_token_ids) != len(candidate_token_ids) + 1:
        raise ValueError("Qwen3-ASR temporal verification needs one bonus prediction")
    else:
        pass
    accepted_count = 0
    for candidate, target in zip(candidate_token_ids, target_token_ids):
        if candidate != target:
            break
        else:
            accepted_count += 1
    return Qwen3ASRTemporalVerification(
        request_id=request_id,
        accepted_prefix=candidate_token_ids[:accepted_count],
        pending_token_id=target_token_ids[accepted_count],
    )


class Qwen3ASRTemporalLogitsProcessor(nn.Module):
    def __init__(self, native_processor: LogitsProcessor) -> None:
        super().__init__()
        self.native_processor = native_processor

    def forward(
        self,
        input_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata | ForwardBatch,
        aux_hidden_states: AuxHiddenStates | None = None,
        hidden_states_before_norm: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        if not isinstance(logits_metadata, ForwardBatch):
            return self.native_processor(
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                aux_hidden_states,
                hidden_states_before_norm,
            )
        else:
            pass
        context = get_temporal_forward_context(logits_metadata)
        if context is None:
            return self.native_processor(
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                aux_hidden_states,
                hidden_states_before_norm,
            )
        else:
            pass
        if not logits_metadata.forward_mode.is_extend():
            raise RuntimeError("Qwen3-ASR temporal verification only supports prefill")
        else:
            pass

        expected_rows = context.prompt_boundary + len(context.candidate_token_ids)
        if context.graph_bucket_size is not None and (
            context.graph_bucket_size < expected_rows
            or hidden_states.shape[0] != context.graph_bucket_size
        ):
            raise RuntimeError("Qwen3-ASR temporal graph bucket changed during replay")
        else:
            pass
        first_prediction_row = context.prompt_boundary - 1
        prediction_states = hidden_states[first_prediction_row:expected_rows]
        if hidden_states_before_norm is not None:
            selected_hidden_states_before_norm = hidden_states_before_norm[
                first_prediction_row:expected_rows
            ]
        else:
            selected_hidden_states_before_norm = None
        if isinstance(aux_hidden_states, torch.Tensor):
            selected_aux_hidden_states = aux_hidden_states[
                first_prediction_row:expected_rows
            ]
        elif aux_hidden_states is not None:
            selected_aux_hidden_states = [
                auxiliary_hidden_states[first_prediction_row:expected_rows]
                for auxiliary_hidden_states in aux_hidden_states
            ]
        else:
            selected_aux_hidden_states = None

        logits_metadata_view = LogitsMetadata.from_forward_batch(logits_metadata)
        assert logits_metadata_view.extend_seq_lens is not None
        prediction_count = len(context.candidate_token_ids) + 1
        temporal_metadata = replace(
            logits_metadata_view,
            extend_seq_lens=torch.ones(
                prediction_count,
                dtype=logits_metadata_view.extend_seq_lens.dtype,
                device=logits_metadata_view.extend_seq_lens.device,
            ),
            extend_seq_lens_cpu=[1] * prediction_count,
            next_token_logits_buffer=None,
        )
        temporal_output = self.native_processor(
            input_ids=None,
            hidden_states=prediction_states,
            lm_head=lm_head,
            logits_metadata=temporal_metadata,
            aux_hidden_states=selected_aux_hidden_states,
            hidden_states_before_norm=selected_hidden_states_before_norm,
        )
        temporal_logits = temporal_output.next_token_logits
        if temporal_logits is None:
            raise RuntimeError("Native Qwen3-ASR logits processor returned no logits")
        else:
            pass
        temporal_token_ids = temporal_logits.argmax(-1)
        verification = verify_greedy_candidates(
            context.request_id,
            context.candidate_token_ids,
            [int(token_id) for token_id in temporal_token_ids.tolist()],
        )
        pending_row = len(verification.accepted_prefix)
        native_values = {
            field.name: getattr(temporal_output, field.name)
            for field in fields(LogitsProcessorOutput)
        }
        native_values["next_token_logits"] = temporal_logits[
            pending_row : pending_row + 1
        ].clone()
        return Qwen3ASRTemporalLogitsProcessorOutput(
            **native_values,
            temporal_verification=verification,
            pending_token_ids=temporal_token_ids[pending_row : pending_row + 1].clone(),
            graph_bucket_size=context.graph_bucket_size,
        )


__all__ = [
    "Qwen3ASRTemporalBatchResult",
    "Qwen3ASRTemporalForwardContext",
    "Qwen3ASRTemporalLogitsProcessor",
    "Qwen3ASRTemporalLogitsProcessorOutput",
    "Qwen3ASRTemporalRunnerOutput",
    "Qwen3ASRTemporalVerification",
    "attach_temporal_forward_inputs",
    "clear_temporal_forward_inputs",
    "get_temporal_forward_context",
    "verify_greedy_candidates",
]

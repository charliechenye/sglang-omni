# SPDX-License-Identifier: Apache-2.0
"""Request-scoped Qwen3-ASR temporal verification types and logits adapter."""

from __future__ import annotations

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

from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRTemporalRequest
from sglang_omni.scheduling.types import ModelRunnerOutput

QWEN3_ASR_TEMPORAL_FORWARD_ATTR = "qwen3_asr_temporal_forward"


@dataclass(frozen=True, slots=True, kw_only=True)
class Qwen3ASRTemporalForward:
    request_id: str
    prompt_boundary: int
    candidate_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Qwen3ASRTemporalVerification:
    request_id: str
    row_index: int
    prompt_boundary: int
    candidate_token_ids: tuple[int, ...]
    accepted_prefix: tuple[int, ...]
    pending_token_id: int
    owned_kv_boundary: int


@dataclass(slots=True)
class Qwen3ASRTemporalLogitsProcessorOutput(LogitsProcessorOutput):
    temporal_verification: Qwen3ASRTemporalVerification | None = None
    pending_token_ids: torch.Tensor | None = None


@dataclass(slots=True)
class Qwen3ASRTemporalRunnerOutput(ModelRunnerOutput):
    temporal_verification: Qwen3ASRTemporalVerification | None = None


@dataclass(slots=True, kw_only=True)
class Qwen3ASRTemporalBatchResult:
    native_result: GenerationBatchResult
    temporal_verification: Qwen3ASRTemporalVerification


def attach_temporal_forward_inputs(
    forward_batch: ForwardBatch, temporal_forward: Qwen3ASRTemporalForward
) -> None:
    if hasattr(forward_batch, QWEN3_ASR_TEMPORAL_FORWARD_ATTR):
        raise RuntimeError("Qwen3-ASR temporal forward state was already attached")
    else:
        pass
    setattr(forward_batch, QWEN3_ASR_TEMPORAL_FORWARD_ATTR, temporal_forward)


def get_temporal_forward_inputs(
    forward_batch: ForwardBatch,
) -> Qwen3ASRTemporalForward | None:
    temporal_forward = getattr(forward_batch, QWEN3_ASR_TEMPORAL_FORWARD_ATTR, None)
    if isinstance(temporal_forward, Qwen3ASRTemporalForward):
        return temporal_forward
    else:
        return None


def clear_temporal_forward_inputs(forward_batch: ForwardBatch | None) -> None:
    if forward_batch is not None and hasattr(
        forward_batch, QWEN3_ASR_TEMPORAL_FORWARD_ATTR
    ):
        delattr(forward_batch, QWEN3_ASR_TEMPORAL_FORWARD_ATTR)
    else:
        pass


def verify_greedy_candidates(
    request_id: str,
    row_index: int,
    prompt_boundary: int,
    candidate_token_ids: tuple[int, ...],
    target_token_ids: list[int],
) -> Qwen3ASRTemporalVerification:
    if len(target_token_ids) != len(candidate_token_ids) + 1:
        raise ValueError("Qwen3-ASR temporal verification needs one bonus prediction")
    else:
        pass
    accepted_token_count = 0
    for candidate_token_id, target_token_id in zip(
        candidate_token_ids, target_token_ids, strict=False
    ):
        if candidate_token_id != target_token_id:
            break
        else:
            accepted_token_count += 1
    return Qwen3ASRTemporalVerification(
        request_id=request_id,
        row_index=row_index,
        prompt_boundary=prompt_boundary,
        candidate_token_ids=candidate_token_ids,
        accepted_prefix=candidate_token_ids[:accepted_token_count],
        pending_token_id=target_token_ids[accepted_token_count],
        owned_kv_boundary=prompt_boundary + accepted_token_count,
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
        temporal_forward = get_temporal_forward_inputs(logits_metadata)
        if temporal_forward is None:
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
        if len(logits_metadata.rids) != 1 or logits_metadata.rids[0] != (
            temporal_forward.request_id
        ):
            raise RuntimeError(
                "Qwen3-ASR temporal result row does not match its request"
            )
        else:
            pass

        candidate_token_count = len(temporal_forward.candidate_token_ids)
        expected_input_rows = temporal_forward.prompt_boundary + candidate_token_count
        forward_input_ids = logits_metadata.input_ids.reshape(-1)
        expected_candidate_ids = torch.tensor(
            temporal_forward.candidate_token_ids,
            dtype=forward_input_ids.dtype,
            device=forward_input_ids.device,
        )
        expected_positions = torch.arange(
            expected_input_rows,
            dtype=logits_metadata.positions.dtype,
            device=logits_metadata.positions.device,
        )
        if (
            temporal_forward.prompt_boundary <= 0
            or candidate_token_count <= 0
            or forward_input_ids.numel() != expected_input_rows
            or (
                input_ids is not None
                and not torch.equal(input_ids.reshape(-1), forward_input_ids)
            )
            or not torch.equal(
                forward_input_ids[-candidate_token_count:], expected_candidate_ids
            )
            or logits_metadata.positions.numel() != expected_input_rows
            or not torch.equal(
                logits_metadata.positions.reshape(-1), expected_positions
            )
            or hidden_states.shape[0] != expected_input_rows
            or logits_metadata.extend_seq_lens_cpu != [expected_input_rows]
            or logits_metadata.extend_prefix_lens_cpu != [0]
        ):
            raise RuntimeError(
                "Qwen3-ASR temporal prefill does not contain its exact unpadded rows"
            )
        else:
            pass
        first_prediction_row = temporal_forward.prompt_boundary - 1
        prediction_row_end = expected_input_rows
        prediction_indices = torch.arange(
            first_prediction_row,
            prediction_row_end,
            dtype=torch.long,
            device=hidden_states.device,
        )
        prediction_states = hidden_states.index_select(0, prediction_indices)
        if hidden_states_before_norm is not None:
            selected_hidden_states_before_norm = hidden_states_before_norm.index_select(
                0, prediction_indices
            )
        else:
            selected_hidden_states_before_norm = None
        if isinstance(aux_hidden_states, torch.Tensor):
            selected_aux_hidden_states = aux_hidden_states.index_select(
                0, prediction_indices
            )
        elif aux_hidden_states is not None:
            selected_aux_hidden_states = [
                auxiliary_hidden_states.index_select(0, prediction_indices)
                for auxiliary_hidden_states in aux_hidden_states
            ]
        else:
            selected_aux_hidden_states = None
        logits_metadata_view = LogitsMetadata.from_forward_batch(logits_metadata)
        assert logits_metadata_view.extend_seq_lens is not None
        temporal_metadata = replace(
            logits_metadata_view,
            extend_seq_lens=torch.ones(
                candidate_token_count + 1,
                dtype=logits_metadata_view.extend_seq_lens.dtype,
                device=logits_metadata_view.extend_seq_lens.device,
            ),
            extend_seq_lens_cpu=[1] * (candidate_token_count + 1),
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
            request_id=temporal_forward.request_id,
            row_index=0,
            prompt_boundary=temporal_forward.prompt_boundary,
            candidate_token_ids=temporal_forward.candidate_token_ids,
            target_token_ids=[
                int(token_id) for token_id in temporal_token_ids.tolist()
            ],
        )
        pending_row_index = len(verification.accepted_prefix)
        native_values = {
            field.name: getattr(temporal_output, field.name)
            for field in fields(LogitsProcessorOutput)
        }
        native_values["next_token_logits"] = temporal_logits[
            pending_row_index : pending_row_index + 1
        ].clone()
        return Qwen3ASRTemporalLogitsProcessorOutput(
            **native_values,
            temporal_verification=verification,
            pending_token_ids=temporal_token_ids[
                pending_row_index : pending_row_index + 1
            ].clone(),
        )


__all__ = [
    "Qwen3ASRTemporalBatchResult",
    "Qwen3ASRTemporalForward",
    "Qwen3ASRTemporalLogitsProcessor",
    "Qwen3ASRTemporalLogitsProcessorOutput",
    "Qwen3ASRTemporalRequest",
    "Qwen3ASRTemporalRunnerOutput",
    "Qwen3ASRTemporalVerification",
    "attach_temporal_forward_inputs",
    "clear_temporal_forward_inputs",
    "get_temporal_forward_inputs",
    "verify_greedy_candidates",
]

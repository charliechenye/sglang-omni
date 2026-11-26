# SPDX-License-Identifier: Apache-2.0
"""Native sequential replay helpers for frozen Qwen3-TTS Predictor frames."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

import torch
from torch import nn

from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.utils.qwen3_tts_speculative_frozen import FrozenPredictorFrame


class PredictorTokenOperation(Protocol):
    def __call__(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor: ...


class PredictorSampleOperation(Protocol):
    def __call__(
        self,
        logits: torch.Tensor,
        codec_table: torch.Tensor,
        accumulated: torch.Tensor,
        output_codes: torch.Tensor,
        projected_table: torch.Tensor | None,
        embedding_buffer: torch.Tensor | None,
        projected_buffer: torch.Tensor | None,
        sub_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class PredictorHeadHook(Protocol):
    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None: ...


@dataclass(frozen=True, kw_only=True)
class NativePredictorReference:
    codes: torch.Tensor
    summed_embedding: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    projected_embeddings: torch.Tensor
    keys_by_length: dict[int, torch.Tensor]
    values_by_length: dict[int, torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class TeacherForcedPredictorReference:
    hidden_by_depth: dict[int, torch.Tensor]
    logits_by_depth: dict[int, torch.Tensor]
    keys_by_length: dict[int, torch.Tensor]
    values_by_length: dict[int, torch.Tensor]


def head_output_logits(
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    elif isinstance(output, torch.Tensor):
        return output
    else:
        raise TypeError("Predictor LM head returned an unsupported output")


def reset_single_frame_state(talker: Qwen3TTSTalker) -> None:
    talker.predictor_k_cache[:, :1].zero_()
    talker.predictor_v_cache[:, :1].zero_()
    talker.output_codes[:1].fill_(-1)
    talker.output_embeds[:1].zero_()


def prepare_frozen_frame_request(
    talker: Qwen3TTSTalker,
    frame: FrozenPredictorFrame,
    *,
    request_id: str,
) -> None:
    """Stage the artifact's sampling settings as an internally seeded request."""
    sampling = frame.sampling
    data = SimpleNamespace(
        semantic_sampling_seed=int(sampling["semantic_sampling_seed_tensor"].item()),
        subtalker_dosample=bool(sampling["sub_do_sample_tensor"].item()),
        subtalker_temperature=float(sampling["sub_temperature_tensor"].item()),
        subtalker_top_p=float(sampling["sub_top_p_tensor"].item()),
        subtalker_top_k=int(sampling["sub_top_k_tensor"].item()),
        subtalker_sampling_seed=int(sampling["sub_sampling_seed_tensor"].item()),
        user_seed_provided=False,
        decode_prep_epoch=None,
    )
    talker.prepare_decode_buffers([SimpleNamespace(request_id=request_id, data=data)])


def capture_native_predictor_reference(
    talker: Qwen3TTSTalker,
    frame: FrozenPredictorFrame,
    *,
    request_id: str,
) -> NativePredictorReference:
    """Independently replay the native token-at-a-time Predictor for one frame."""
    prepare_frozen_frame_request(talker, frame, request_id=request_id)
    reset_single_frame_state(talker)
    head_count = len(talker.code_predictor.lm_head)
    hidden_rows: list[torch.Tensor | None] = [None] * head_count
    logit_rows: list[torch.Tensor | None] = [None] * head_count
    projected_rows: list[torch.Tensor] = []
    keys_by_length: dict[int, torch.Tensor] = {}
    values_by_length: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_head_hook(head_index: int) -> PredictorHeadHook:
        def capture_head(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            if not inputs or inputs[0].ndim != 3:
                raise RuntimeError("native Predictor head input has an invalid shape")
            else:
                pass
            hidden_rows[head_index] = inputs[0][0, -1].detach().clone()
            logit_rows[head_index] = head_output_logits(output)[0, -1].detach().clone()

        return capture_head

    for head_index, head in enumerate(talker.code_predictor.lm_head):
        hooks.append(head.register_forward_hook(make_head_hook(head_index)))
    else:
        pass

    original_forward: PredictorTokenOperation = talker.predictor_forward_tokens
    original_sample: PredictorSampleOperation = talker.sample_subtalker_embedding

    def record_predictor_tokens(
        *, token_embeds: torch.Tensor, batch_size: int, cache_len: int
    ) -> torch.Tensor:
        result = original_forward(
            token_embeds=token_embeds, batch_size=batch_size, cache_len=cache_len
        )
        valid_length = cache_len + token_embeds.shape[1]
        keys_by_length[valid_length] = (
            talker.predictor_k_cache[:, :1, :valid_length].detach().clone()
        )
        values_by_length[valid_length] = (
            talker.predictor_v_cache[:, :1, :valid_length].detach().clone()
        )
        return result

    def record_sample(
        logits: torch.Tensor,
        codec_table: torch.Tensor,
        accumulated: torch.Tensor,
        output_codes: torch.Tensor,
        projected_table: torch.Tensor | None,
        embedding_buffer: torch.Tensor | None,
        projected_buffer: torch.Tensor | None,
        sub_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding, projected = original_sample(
            logits,
            codec_table,
            accumulated,
            output_codes,
            projected_table,
            embedding_buffer,
            projected_buffer,
            sub_positions,
        )
        if projected is not None:
            projected_rows.append(projected[0].detach().clone())
        else:
            pass
        return embedding, projected

    talker.predictor_forward_tokens = record_predictor_tokens
    talker.sample_subtalker_embedding = record_sample
    try:
        with torch.inference_mode():
            codes, summed_embedding = talker.code_predictor_forward_incremental(
                frame.layer0_code.reshape(1, 1).to(device=talker.predictor_device),
                frame.talker_hidden.reshape(1, 1, -1).to(
                    device=talker.predictor_device, dtype=talker.predictor_k_cache.dtype
                ),
                frame.semantic_positions.reshape(1, -1).to(
                    device=talker.predictor_device, dtype=torch.long
                ),
            )
    finally:
        talker.predictor_forward_tokens = original_forward
        talker.sample_subtalker_embedding = original_sample
        for hook in hooks:
            hook.remove()
        else:
            pass

    if any(row is None for row in hidden_rows) or any(
        row is None for row in logit_rows
    ):
        raise RuntimeError("native replay did not evaluate every Predictor LM head")
    else:
        pass
    if len(projected_rows) != frame.projected_embed.shape[0]:
        raise RuntimeError(
            "native replay projected embedding count differs from artifact"
        )
    else:
        pass
    return NativePredictorReference(
        codes=codes[0, :, 0].detach().clone(),
        summed_embedding=summed_embedding[0, 0].detach().clone(),
        hidden=torch.stack([row for row in hidden_rows if row is not None]),
        logits=torch.stack([row for row in logit_rows if row is not None]),
        projected_embeddings=torch.stack(projected_rows),
        keys_by_length=keys_by_length,
        values_by_length=values_by_length,
    )


def capture_teacher_forced_predictor_reference(
    talker: Qwen3TTSTalker,
    frame: FrozenPredictorFrame,
    *,
    code_tokens: torch.Tensor,
    max_depth: int,
    request_id: str,
) -> TeacherForcedPredictorReference:
    """Replay an observed committed prefix and draft trajectory sequentially."""
    if code_tokens.ndim != 1 or code_tokens.numel() <= max_depth:
        raise ValueError("teacher-forced codes must cover the requested depth")
    elif max_depth < 1 or max_depth >= len(talker.code_predictor.lm_head):
        raise ValueError("teacher-forced depth is outside the Predictor heads")
    else:
        pass
    prepare_frozen_frame_request(talker, frame, request_id=request_id)
    reset_single_frame_state(talker)
    device = talker.predictor_device
    predictor_dtype = talker.predictor_k_cache.dtype
    layer0_code = frame.layer0_code.reshape(1, 1).to(device=device, dtype=torch.long)
    layer0_embed = talker.get_input_embeddings()(layer0_code).to(dtype=predictor_dtype)
    talker_hidden = frame.talker_hidden.reshape(1, 1, -1).to(
        device=device, dtype=predictor_dtype
    )
    pair_embeddings = talker.code_predictor.project_input(
        torch.cat((talker_hidden, layer0_embed), dim=1)
    )
    hidden_by_depth: dict[int, torch.Tensor] = {}
    logits_by_depth: dict[int, torch.Tensor] = {}
    keys_by_length: dict[int, torch.Tensor] = {}
    values_by_length: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        pair_hidden = talker.predictor_forward_tokens(
            token_embeds=pair_embeddings, batch_size=1, cache_len=0
        )
        hidden_by_depth[0] = pair_hidden[0, 1].detach().clone()
        pair_logits = head_output_logits(
            talker.code_predictor.lm_head[0](pair_hidden[:, 1:, :])
        )
        logits_by_depth[0] = pair_logits[0, -1].detach().clone()
        keys_by_length[2] = talker.predictor_k_cache[:, :1, :2].detach().clone()
        values_by_length[2] = talker.predictor_v_cache[:, :1, :2].detach().clone()
        projected_tables = talker.predictor_projected_embeddings
        codec_tables = talker.code_predictor.model.codec_embedding.weight
        for depth in range(1, max_depth + 1):
            token = code_tokens[depth].reshape(1).to(device=device, dtype=torch.long)
            if projected_tables is None:
                codec_embedding = codec_tables[depth - 1, token].reshape(1, 1, -1)
                token_embedding = talker.code_predictor.project_input(codec_embedding)
            else:
                token_embedding = projected_tables[depth - 1, token].reshape(1, 1, -1)
            hidden = talker.predictor_forward_tokens(
                token_embeds=token_embedding,
                batch_size=1,
                cache_len=depth + 1,
            )
            hidden_by_depth[depth] = hidden[0, 0].detach().clone()
            logits = head_output_logits(talker.code_predictor.lm_head[depth](hidden))
            logits_by_depth[depth] = logits[0, 0].detach().clone()
            valid_length = depth + 2
            keys_by_length[valid_length] = (
                talker.predictor_k_cache[:, :1, :valid_length].detach().clone()
            )
            values_by_length[valid_length] = (
                talker.predictor_v_cache[:, :1, :valid_length].detach().clone()
            )
        else:
            pass
    return TeacherForcedPredictorReference(
        hidden_by_depth=hidden_by_depth,
        logits_by_depth=logits_by_depth,
        keys_by_length=keys_by_length,
        values_by_length=values_by_length,
    )


def compare_native_reference(
    frame: FrozenPredictorFrame, reference: NativePredictorReference
) -> None:
    torch.testing.assert_close(
        reference.codes.cpu(), frame.native_codes, atol=0, rtol=0
    )
    torch.testing.assert_close(
        reference.summed_embedding.cpu(), frame.native_embeddings, atol=0, rtol=0
    )
    torch.testing.assert_close(
        reference.hidden.cpu(), frame.hidden, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        reference.logits.cpu(), frame.logits, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        reference.projected_embeddings.cpu(),
        frame.projected_embed,
        atol=2e-2,
        rtol=2e-2,
    )

# SPDX-License-Identifier: Apache-2.0
"""Run and time Qwen3-TTS Predictor inference on frozen frames."""

from __future__ import annotations

import statistics
import time
from typing import Protocol

import torch

from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.utils.qwen3_tts_speculative_frame_observation import (
    SpeculativeRoundObservation,
    observe_speculative_rounds,
)
from sglang_omni.utils.qwen3_tts_speculative_frozen import FrozenPredictorFrame
from sglang_omni.utils.qwen3_tts_speculative_native_replay import (
    prepare_frozen_frame_request,
    reset_single_frame_state,
)


class FrameOperation(Protocol):
    def __call__(self) -> None: ...


def run_predictor_frame(
    talker: Qwen3TTSTalker,
    frame: FrozenPredictorFrame,
    *,
    request_id: str,
    speculate: bool,
    trace_rounds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[SpeculativeRoundObservation]]:
    prepare_frozen_frame_request(talker, frame, request_id=request_id)
    reset_single_frame_state(talker)
    layer0_codes = frame.layer0_code.reshape(1, 1).to(device=talker.predictor_device)
    talker_hidden = frame.talker_hidden.reshape(1, 1, -1).to(
        device=talker.predictor_device, dtype=talker.predictor_k_cache.dtype
    )
    semantic_positions = frame.semantic_positions.reshape(1, -1).to(
        device=talker.predictor_device, dtype=torch.long
    )
    if speculate and trace_rounds:
        with observe_speculative_rounds(talker) as observations:
            codes, embeddings = talker.code_predictor_forward(
                layer0_codes, talker_hidden, semantic_positions
            )
    else:
        codes, embeddings = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        observations = []
    return (
        codes[0, :, 0].detach().clone(),
        embeddings[0, 0].detach().clone(),
        observations,
    )


def assert_checkpoint_lm_heads(
    talker: Qwen3TTSTalker, frozen_head_weights: torch.Tensor
) -> None:
    model_weights = torch.stack(
        [head.weight.detach() for head in talker.code_predictor.lm_head]
    ).cpu()
    if not torch.equal(model_weights, frozen_head_weights):
        raise AssertionError("loaded checkpoint LM heads differ from frozen artifacts")
    else:
        pass


def assert_codec_embedding_sum(
    talker: Qwen3TTSTalker, codes: torch.Tensor, summed_embedding: torch.Tensor
) -> None:
    codec_tables = talker.code_predictor.model.codec_embedding.weight
    with torch.no_grad():
        expected = talker.get_input_embeddings()(codes[:1]).reshape(-1).clone()
        for codebook_index in range(1, codes.numel()):
            expected.add_(codec_tables[codebook_index - 1, codes[codebook_index]])
        else:
            pass
    torch.testing.assert_close(summed_embedding, expected, atol=0, rtol=0)


def median_frame_latency_ms(samples: list[float]) -> float:
    if not samples:
        raise ValueError("at least one latency measurement is required")
    else:
        pass
    return statistics.median(samples)


def timed_predictor_frame(operation: FrameOperation) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0

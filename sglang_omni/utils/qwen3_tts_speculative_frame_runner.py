# SPDX-License-Identifier: Apache-2.0
"""Run and time Qwen3-TTS Predictor inference on frozen frames."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

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


@dataclass(frozen=True, kw_only=True)
class PredictorFrameTiming:
    predictor_only_ms: float
    request_preparation_host_ms: float
    request_preparation_cuda_ms: float
    cache_reset_host_ms: float
    cache_reset_cuda_ms: float
    input_transfer_host_ms: float
    input_transfer_cuda_ms: float
    staging_synchronization_ms: float
    predictor_host_call_ms: float
    predictor_completion_wait_ms: float
    output_clone_ms: float
    output_clone_synchronization_ms: float


def stage_predictor_frame_inputs(
    talker: Qwen3TTSTalker, frame: FrozenPredictorFrame
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer0_codes = frame.layer0_code.reshape(1, 1).to(device=talker.predictor_device)
    talker_hidden = frame.talker_hidden.reshape(1, 1, -1).to(
        device=talker.predictor_device, dtype=talker.predictor_k_cache.dtype
    )
    semantic_positions = frame.semantic_positions.reshape(1, -1).to(
        device=talker.predictor_device, dtype=torch.long
    )
    return layer0_codes, talker_hidden, semantic_positions


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
    layer0_codes, talker_hidden, semantic_positions = stage_predictor_frame_inputs(
        talker, frame
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


def timed_predictor_frame(
    talker: Qwen3TTSTalker,
    frame: FrozenPredictorFrame,
    *,
    request_id: str,
) -> PredictorFrameTiming:
    stage_stream = torch.cuda.current_stream(talker.predictor_device)
    request_start = torch.cuda.Event(enable_timing=True)
    request_end = torch.cuda.Event(enable_timing=True)
    request_start.record(stage_stream)
    started = time.perf_counter()
    prepare_frozen_frame_request(talker, frame, request_id=request_id)
    request_preparation_host_ms = (time.perf_counter() - started) * 1000.0
    request_end.record(stage_stream)

    reset_start = torch.cuda.Event(enable_timing=True)
    reset_end = torch.cuda.Event(enable_timing=True)
    reset_start.record(stage_stream)
    started = time.perf_counter()
    reset_single_frame_state(talker)
    cache_reset_host_ms = (time.perf_counter() - started) * 1000.0
    reset_end.record(stage_stream)

    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    transfer_start.record(stage_stream)
    started = time.perf_counter()
    layer0_codes, talker_hidden, semantic_positions = stage_predictor_frame_inputs(
        talker, frame
    )
    input_transfer_host_ms = (time.perf_counter() - started) * 1000.0
    transfer_end.record(stage_stream)

    started = time.perf_counter()
    torch.cuda.synchronize(talker.predictor_device)
    staging_synchronization_ms = (time.perf_counter() - started) * 1000.0

    predictor_start = torch.cuda.Event(enable_timing=True)
    predictor_end = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    predictor_start.record(stage_stream)
    codes, embeddings = talker.code_predictor_forward(
        layer0_codes, talker_hidden, semantic_positions
    )
    predictor_end.record(stage_stream)
    predictor_host_call_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    predictor_end.synchronize()
    predictor_completion_wait_ms = (time.perf_counter() - started) * 1000.0
    predictor_only_ms = float(predictor_start.elapsed_time(predictor_end))

    clone_start = torch.cuda.Event(enable_timing=True)
    clone_end = torch.cuda.Event(enable_timing=True)
    clone_start.record(stage_stream)
    codes[0, :, 0].detach().clone()
    embeddings[0, 0].detach().clone()
    clone_end.record(stage_stream)
    started = time.perf_counter()
    clone_end.synchronize()
    output_clone_synchronization_ms = (time.perf_counter() - started) * 1000.0
    output_clone_ms = float(clone_start.elapsed_time(clone_end))
    return PredictorFrameTiming(
        predictor_only_ms=predictor_only_ms,
        request_preparation_host_ms=request_preparation_host_ms,
        request_preparation_cuda_ms=float(request_start.elapsed_time(request_end)),
        cache_reset_host_ms=cache_reset_host_ms,
        cache_reset_cuda_ms=float(reset_start.elapsed_time(reset_end)),
        input_transfer_host_ms=input_transfer_host_ms,
        input_transfer_cuda_ms=float(transfer_start.elapsed_time(transfer_end)),
        staging_synchronization_ms=staging_synchronization_ms,
        predictor_host_call_ms=predictor_host_call_ms,
        predictor_completion_wait_ms=predictor_completion_wait_ms,
        output_clone_ms=output_clone_ms,
        output_clone_synchronization_ms=output_clone_synchronization_ms,
    )

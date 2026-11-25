#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and benchmark Predictor backends on real frozen Qwen3-TTS frames."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Literal, TypedDict

import torch

from sglang_omni.models.qwen3_tts.sglang_model import (
    PredictorGraphStats,
    PredictorSpeculativeStats,
    Qwen3TTSTalker,
)
from sglang_omni.models.qwen3_tts.stages import create_sglang_tts_engine_executor
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.utils.qwen3_tts_speculative_frame_observation import (
    validate_speculative_observations,
)
from sglang_omni.utils.qwen3_tts_speculative_frame_runner import (
    assert_checkpoint_lm_heads,
    assert_codec_embedding_sum,
    median_frame_latency_ms,
    run_predictor_frame,
    timed_predictor_frame,
)
from sglang_omni.utils.qwen3_tts_speculative_frozen import (
    FrozenPredictorArtifacts,
    load_frozen_predictor_artifacts,
)
from sglang_omni.utils.qwen3_tts_speculative_native_replay import (
    capture_native_predictor_reference,
    compare_native_reference,
)

PredictorArm = Literal[
    "native_graph", "spec_eager", "spec_verifier_graph", "spec_full_round_graph"
]
ProbabilityBackend = Literal["split_top_k", "radix_compact", "radix_compact_4bit"]
LMHeadBackend = Literal["grouped_triton", "warp_gemv", "warp_accumulate"]
RNGBackend = Literal["per_round", "frame_pool"]
AcceptanceBackend = Literal["compact", "dense_indexed"]


class FrozenFrameBenchmarkResult(TypedDict):
    schema: str
    arm: PredictorArm
    model_path: str
    model_snapshot: str
    capture_sha256: str
    dataset_sha256: str
    device: str
    verify_width: int
    probability_backend: ProbabilityBackend
    lm_head_backend: LMHeadBackend
    rng_backend: RNGBackend
    acceptance_backend: AcceptanceBackend
    opening_pair_graph_requested: bool
    correctness_preflight: dict[str, int]
    frame_latency_median_ms: float
    frame_latency_p95_ms: float
    measured_frames: int
    warmup_frames: int
    speculative_stats: PredictorSpeculativeStats
    native_predictor_graph_stats: PredictorGraphStats


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    else:
        pass
    return parsed


def finite_fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be finite and between 0 and 1")
    else:
        pass
    return parsed


def build_talker(
    model_path: str,
    *,
    arm: PredictorArm,
    gpu_id: int,
    verify_width: int,
    mem_fraction_static: float,
    probability_backend: ProbabilityBackend,
    lm_head_backend: LMHeadBackend,
    rng_backend: RNGBackend,
    acceptance_backend: AcceptanceBackend,
    opening_pair_graph: bool,
) -> tuple[OmniScheduler, Qwen3TTSTalker]:
    if arm == "native_graph":
        speculative_width = 0
        speculative_backend = "cuda_graph"
    elif arm == "spec_eager":
        speculative_width = verify_width
        speculative_backend = "eager"
    elif arm == "spec_verifier_graph":
        speculative_width = verify_width
        speculative_backend = "cuda_graph"
    else:
        speculative_width = verify_width
        speculative_backend = "full_round_graph"
    scheduler = create_sglang_tts_engine_executor(
        model_path,
        device="cuda",
        gpu_id=gpu_id,
        dtype="bfloat16",
        code_predictor_spec_verify_size=speculative_width,
        code_predictor_spec_backend=speculative_backend,
        code_predictor_spec_probability_backend=probability_backend,
        code_predictor_spec_lm_head_backend=lm_head_backend,
        code_predictor_spec_rng_backend=rng_backend,
        code_predictor_spec_acceptance_backend=acceptance_backend,
        code_predictor_spec_opening_pair_graph=opening_pair_graph,
        server_args_overrides={
            "max_running_requests": 1,
            "mem_fraction_static": mem_fraction_static,
            "disable_cuda_graph": False,
        },
    )
    model = scheduler.tp_worker.model_runner.model
    if not isinstance(model, Qwen3TTSTalker):
        scheduler.stop()
        raise TypeError("the scheduler did not load a Qwen3TTSTalker model")
    else:
        pass
    return scheduler, model


def check_speculative_output(
    talker: Qwen3TTSTalker,
    frame_codes: torch.Tensor,
    frame_embedding: torch.Tensor,
) -> None:
    vocab_size = int(talker.config.code_predictor_config.vocab_size)
    if frame_codes.shape != (16,):
        raise AssertionError("Predictor output did not contain all 16 codebooks")
    elif not bool(((frame_codes >= 0) & (frame_codes < vocab_size)).all()):
        raise AssertionError(
            "Predictor returned a code outside the codebook vocabulary"
        )
    else:
        pass
    assert_codec_embedding_sum(talker, frame_codes, frame_embedding)


def run_correctness_preflight(
    talker: Qwen3TTSTalker,
    artifacts: FrozenPredictorArtifacts,
    *,
    arm: PredictorArm,
    allow_mixed_graph: bool,
) -> dict[str, int | float]:
    speculative = arm.startswith("spec_")
    logit_checks = 0
    probability_checks = 0
    carried_hidden_checks = 0
    valid_kv_prefix_checks = 0
    for frame_index, frame in enumerate(artifacts.frames):
        request_id = f"frozen-{arm}-{frame_index}-{frame.ordinal}"
        reference = capture_native_predictor_reference(
            talker, frame, request_id=f"native-reference-{request_id}"
        )
        compare_native_reference(frame, reference)
        if frame_index == 0:
            assert_checkpoint_lm_heads(talker, artifacts.lm_head_weights)
        else:
            pass
        codes, summed_embedding, observations = run_predictor_frame(
            talker,
            frame,
            request_id=request_id,
            speculate=speculative,
        )
        check_speculative_output(talker, codes, summed_embedding)
        if speculative:
            checks = validate_speculative_observations(
                talker, frame, reference, observations
            )
            logit_checks += checks[0]
            probability_checks += checks[1]
            carried_hidden_checks += checks[2]
            valid_kv_prefix_checks += checks[3]
        else:
            pass
        if codes[0].item() != frame.native_codes[0].item():
            raise AssertionError("frozen Predictor changed the conditioned codebook")
        else:
            pass
    else:
        pass

    stats = talker.predictor_speculative_stats()
    if speculative and stats["frames"] != len(artifacts.frames):
        raise AssertionError("speculative frame preflight fell back to native decode")
    elif speculative and stats["native_fallback_count"] != 0:
        raise AssertionError("speculative frame preflight included native fallback")
    else:
        pass
    if speculative and not allow_mixed_graph:
        if arm == "spec_verifier_graph":
            if stats["verifier_graph_replay_coverage"] != 1.0:
                raise AssertionError(
                    "verifier graph preflight had eager round fallback"
                )
            elif (
                talker.code_predictor_spec_opening_pair_graph_enabled
                and stats["opening_pair_graph_replay_coverage"] != 1.0
            ):
                raise AssertionError(
                    "opening-pair graph did not replay for every frame"
                )
            else:
                pass
        elif arm == "spec_full_round_graph":
            if stats["full_round_graph_replay_coverage"] != 1.0:
                raise AssertionError("full-round preflight had eager round fallback")
            elif stats["opening_pair_graph_replay_coverage"] != 1.0:
                raise AssertionError(
                    "opening-pair graph did not replay for every frame"
                )
            else:
                pass
        else:
            pass
    else:
        pass
    return {
        "frames": len(artifacts.frames),
        "target_logit_context_checks": logit_checks,
        "target_probability_support_checks": probability_checks,
        "carried_hidden_checks": carried_hidden_checks,
        "valid_kv_prefix_checks": valid_kv_prefix_checks,
    }


def run_timed_pass(
    talker: Qwen3TTSTalker,
    artifacts: FrozenPredictorArtifacts,
    *,
    arm: PredictorArm,
    warmup_frames: int,
) -> list[float]:
    speculative = arm.startswith("spec_")
    for frame_index in range(warmup_frames):
        frame = artifacts.frames[frame_index % len(artifacts.frames)]
        run_predictor_frame(
            talker,
            frame,
            request_id=f"timed-warmup-{arm}-{frame_index}-{frame.ordinal}",
            speculate=speculative,
            trace_rounds=False,
        )
    else:
        pass
    torch.cuda.synchronize()
    latencies: list[float] = []
    for frame_index, frame in enumerate(artifacts.frames):
        latencies.append(
            timed_predictor_frame(
                lambda current_frame=frame, current_index=frame_index: run_predictor_frame(
                    talker,
                    current_frame,
                    request_id=(f"timed-{arm}-{current_index}-{current_frame.ordinal}"),
                    speculate=speculative,
                    trace_rounds=False,
                )
            )
        )
    else:
        pass
    return latencies


def run_benchmark(
    artifacts: FrozenPredictorArtifacts,
    *,
    model_path: str,
    arm: PredictorArm,
    gpu_id: int,
    verify_width: int,
    warmup_frames: int,
    mem_fraction_static: float,
    probability_backend: ProbabilityBackend,
    lm_head_backend: LMHeadBackend,
    rng_backend: RNGBackend,
    acceptance_backend: AcceptanceBackend,
    opening_pair_graph: bool,
    allow_mixed_graph: bool,
) -> FrozenFrameBenchmarkResult:
    if arm == "spec_full_round_graph" and acceptance_backend != "compact":
        raise ValueError("full-round CUDA Graph requires compact acceptance")
    else:
        pass
    scheduler, talker = build_talker(
        model_path,
        arm=arm,
        gpu_id=gpu_id,
        verify_width=verify_width,
        mem_fraction_static=mem_fraction_static,
        probability_backend=probability_backend,
        lm_head_backend=lm_head_backend,
        rng_backend=rng_backend,
        acceptance_backend=acceptance_backend,
        opening_pair_graph=opening_pair_graph,
    )
    try:
        preflight = run_correctness_preflight(
            talker,
            artifacts,
            arm=arm,
            allow_mixed_graph=allow_mixed_graph,
        )
        latencies = run_timed_pass(
            talker, artifacts, arm=arm, warmup_frames=warmup_frames
        )
        stats = talker.predictor_speculative_stats()
        graph_stats = talker.predictor_graph_stats()
        return {
            "schema": "qwen3-tts-speculative-frozen-frame-benchmark-v1",
            "arm": arm,
            "model_path": model_path,
            "model_snapshot": artifacts.model_snapshot,
            "capture_sha256": artifacts.capture_sha256,
            "dataset_sha256": artifacts.dataset_sha256,
            "device": torch.cuda.get_device_name(talker.predictor_device),
            "verify_width": verify_width if arm.startswith("spec_") else 0,
            "probability_backend": probability_backend,
            "lm_head_backend": lm_head_backend,
            "rng_backend": rng_backend,
            "acceptance_backend": acceptance_backend,
            "opening_pair_graph_requested": opening_pair_graph,
            "correctness_preflight": preflight,
            "frame_latency_median_ms": median_frame_latency_ms(latencies),
            "frame_latency_p95_ms": (
                statistics.quantiles(latencies, n=20)[18]
                if len(latencies) >= 2
                else latencies[0]
            ),
            "measured_frames": len(latencies),
            "warmup_frames": warmup_frames,
            "speculative_stats": stats,
            "native_predictor_graph_stats": graph_stats,
        }
    finally:
        scheduler.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "native_graph",
            "spec_eager",
            "spec_verifier_graph",
            "spec_full_round_graph",
        ),
        required=True,
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--verify-width", type=positive_integer, default=4)
    parser.add_argument("--frame-limit", type=positive_integer, default=16)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--mem-fraction-static", type=finite_fraction, default=0.8)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="all"
    )
    parser.add_argument(
        "--probability-backend",
        choices=("split_top_k", "radix_compact", "radix_compact_4bit"),
        default="split_top_k",
    )
    parser.add_argument(
        "--lm-head-backend",
        choices=("grouped_triton", "warp_gemv", "warp_accumulate"),
        default="grouped_triton",
    )
    parser.add_argument(
        "--rng-backend", choices=("per_round", "frame_pool"), default="per_round"
    )
    parser.add_argument(
        "--acceptance-backend",
        choices=("compact", "dense_indexed"),
        default="compact",
    )
    parser.add_argument("--opening-pair-graph", action="store_true")
    parser.add_argument("--allow-mixed-graph", action="store_true")
    parser.add_argument("--expected-model-snapshot")
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be nonnegative")
    else:
        pass
    artifacts = load_frozen_predictor_artifacts(
        args.capture_root,
        args.dataset_root,
        frame_limit=args.frame_limit,
        split=args.split,
        expected_model_snapshot=args.expected_model_snapshot,
    )
    result = run_benchmark(
        artifacts,
        model_path=args.model_path,
        arm=args.arm,
        gpu_id=args.gpu_id,
        verify_width=args.verify_width,
        warmup_frames=args.warmup_frames,
        mem_fraction_static=args.mem_fraction_static,
        probability_backend=args.probability_backend,
        lm_head_backend=args.lm_head_backend,
        rng_backend=args.rng_backend,
        acceptance_backend=args.acceptance_backend,
        opening_pair_graph=args.opening_pair_graph,
        allow_mixed_graph=args.allow_mixed_graph,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

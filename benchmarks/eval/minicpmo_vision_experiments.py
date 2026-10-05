# SPDX-License-Identifier: Apache-2.0
"""Independent batch-size and resampler ablations with numerical gates."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass

import torch

from benchmarks.eval.minicpmo_vision_measurement import (
    EncoderForward,
    ResamplerForward,
    Trial,
    benchmark_forward,
    capture_resampler_inputs,
)
from benchmarks.eval.minicpmo_vision_metrics import (
    ParityLimits,
    compare_outputs,
    temporary_need_weights,
)

BASELINE_BATCH_SIZE = 16


def print_trial(trial: Trial) -> None:
    if trial.status == "OOM":
        print(
            f"arm={trial.arm} batch_size={trial.batch_size} OOM: {trial.error}",
            flush=True,
        )
    else:
        assert trial.gpu_ms is not None and trial.cpu_wall_ms is not None
        print(
            f"arm={trial.arm} batch_size={trial.batch_size} "
            f"mean_ms={trial.gpu_ms.mean:.3f} median_ms={trial.gpu_ms.median:.3f} "
            f"p95_ms={trial.gpu_ms.p95:.3f} min_ms={trial.gpu_ms.minimum:.3f} "
            f"peak_allocated_mb={trial.peak_allocated_mb:.1f} peak_reserved_mb={trial.peak_reserved_mb:.1f} "
            f"cpu_wall_median_ms={trial.cpu_wall_ms.median:.3f}",
            flush=True,
        )
    if trial.parity is not None:
        print(f"parity={asdict(trial.parity)}", flush=True)
    else:
        pass
    if trial.median_speedup is not None:
        print(
            f"median_speedup={trial.median_speedup:.4f} median_delta_percent={trial.median_delta_percent:.3f}",
            flush=True,
        )
    else:
        pass


def add_delta(baseline: Trial, candidate: Trial) -> None:
    assert baseline.gpu_ms is not None and candidate.gpu_ms is not None
    candidate.median_speedup = baseline.gpu_ms.median / candidate.gpu_ms.median
    candidate.median_delta_percent = 100.0 * (
        candidate.gpu_ms.median / baseline.gpu_ms.median - 1.0
    )


def vision_batch_sweep(
    operation: EncoderForward,
    batch_sizes: list[int],
    warmup: int,
    iters: int,
    limits: ParityLimits,
) -> list[Trial]:
    trials = []
    baseline_output: torch.Tensor | None = None
    original_batch_size = operation.encoder.vision_batch_size
    try:
        for batch_size in dict.fromkeys([BASELINE_BATCH_SIZE, *batch_sizes]):
            operation.encoder.vision_batch_size = batch_size
            trial, output = benchmark_forward(
                operation,
                arm=f"bs{batch_size}",
                batch_size=batch_size,
                device=operation.encoder.device,
                warmup=warmup,
                iters=iters,
            )
            trials.append(trial)
            if trial.status == "OOM":
                print_trial(trial)
                if batch_size == BASELINE_BATCH_SIZE:
                    break
                else:
                    continue
            else:
                assert output is not None
                if baseline_output is None:
                    baseline_output = output
                    trial.parity = compare_outputs(output, output, limits)
                else:
                    trial.parity = compare_outputs(baseline_output, output, limits)
                    add_delta(trials[0], trial)
                print_trial(trial)
                if trial.parity.substantial_drift:
                    print(
                        "SUBSTANTIAL DRIFT: stopping sweep; investigate before further experiments",
                        flush=True,
                    )
                    break
                else:
                    pass
    finally:
        operation.encoder.vision_batch_size = original_batch_size
    return trials


@dataclass(kw_only=True)
class ResamplerExperiment:
    isolated: list[Trial]
    full_encoder: list[Trial]
    full_encoder_status: str


def resampler_experiment(
    operation: EncoderForward,
    warmup: int,
    iters: int,
    resampler_warmup: int,
    resampler_iters: int,
    limits: ParityLimits,
) -> ResamplerExperiment:
    encoder = operation.encoder
    assert encoder.vision_batch_size == BASELINE_BATCH_SIZE
    attention = encoder.resampler.attn
    if not isinstance(attention, torch.nn.MultiheadAttention):
        raise TypeError("Expected Resampler2_5.attn to be nn.MultiheadAttention")
    else:
        pass
    chunks = capture_resampler_inputs(operation)
    isolated_operation = ResamplerForward(resampler=encoder.resampler, chunks=chunks)
    print(
        f"isolated resampler: all {len(chunks)} production bs16 chunks per request; VPM outputs frozen",
        flush=True,
    )
    isolated_trials: list[Trial] = []
    baseline_output: torch.Tensor | None = None
    for need_weights in (None, False):
        with temporary_need_weights(attention, need_weights):
            trial, output = benchmark_forward(
                isolated_operation,
                arm="R0/current" if need_weights is None else "R1/need_weights=False",
                batch_size=BASELINE_BATCH_SIZE,
                device=encoder.device,
                warmup=resampler_warmup,
                iters=resampler_iters,
            )
        isolated_trials.append(trial)
        if trial.status == "OOM":
            print_trial(trial)
            return ResamplerExperiment(
                isolated=isolated_trials,
                full_encoder=[],
                full_encoder_status="skipped: isolated resampler OOM",
            )
        else:
            assert output is not None
            if baseline_output is None:
                baseline_output = output
                trial.parity = compare_outputs(output, output, limits)
            else:
                trial.parity = compare_outputs(baseline_output, output, limits)
                add_delta(isolated_trials[0], trial)
            print_trial(trial)
    candidate = isolated_trials[-1]
    assert candidate.parity is not None and candidate.median_speedup is not None
    if candidate.parity.substantial_drift:
        return ResamplerExperiment(
            isolated=isolated_trials,
            full_encoder=[],
            full_encoder_status="skipped: SUBSTANTIAL DRIFT",
        )
    elif candidate.median_speedup <= 1.0:
        return ResamplerExperiment(
            isolated=isolated_trials,
            full_encoder=[],
            full_encoder_status="skipped: isolated candidate did not improve median latency",
        )
    else:
        pass
    chunks.clear()
    gc.collect()
    torch.cuda.empty_cache()
    full_trials: list[Trial] = []
    baseline_output = None
    print("=== FULL ENCODER RESAMPLER A/B vision_batch_size=16 ===", flush=True)
    for need_weights in (None, False):
        with temporary_need_weights(attention, need_weights):
            trial, output = benchmark_forward(
                operation,
                arm="V0/current" if need_weights is None else "V1/need_weights=False",
                batch_size=BASELINE_BATCH_SIZE,
                device=encoder.device,
                warmup=warmup,
                iters=iters,
            )
        full_trials.append(trial)
        if trial.status == "OOM":
            print_trial(trial)
            return ResamplerExperiment(
                isolated=isolated_trials,
                full_encoder=full_trials,
                full_encoder_status="OOM",
            )
        else:
            assert output is not None
            if baseline_output is None:
                baseline_output = output
                trial.parity = compare_outputs(output, output, limits)
            else:
                trial.parity = compare_outputs(baseline_output, output, limits)
                add_delta(full_trials[0], trial)
            print_trial(trial)
    return ResamplerExperiment(
        isolated=isolated_trials,
        full_encoder=full_trials,
        full_encoder_status="measured",
    )

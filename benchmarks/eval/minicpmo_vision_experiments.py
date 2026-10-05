# SPDX-License-Identifier: Apache-2.0
"""Independent batch-size and resampler ablations with numerical gates."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import torch

from benchmarks.eval.minicpmo_vision_metrics import (
    ParityLimits,
    compare_outputs,
    temporary_need_weights,
)

if TYPE_CHECKING:
    from benchmarks.eval.minicpmo_vision_measurement import EncoderForward, Trial

BASELINE_BATCH_SIZE = 16
VISION_BATCH_VISITS = (
    (16, "bs16-A"),
    (32, "bs32-A"),
    (64, "bs64-A"),
    (64, "bs64-B"),
    (32, "bs32-B"),
    (16, "bs16-B"),
)
RESAMPLER_VISITS = (
    (None, "R0-A"),
    (False, "R1-A"),
    (False, "R1-B"),
    (None, "R0-B"),
)
FULL_ENCODER_VISITS = (
    (None, "V0-A"),
    (False, "V1-A"),
    (False, "V1-B"),
    (None, "V0-B"),
)


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


@dataclass(kw_only=True)
class PairSummary:
    label: str
    visits: list[str]
    status: Literal["ok", "OOM", "incomplete"]
    mean_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    peak_allocated_mb: float | None = None
    peak_reserved_mb: float | None = None
    repeat_spread_percent: float | None = None
    median_speedup: float | None = None
    median_delta_percent: float | None = None


def pair_summary(first: Trial, second: Trial, label: str) -> PairSummary:
    status: Literal["ok", "OOM", "incomplete"]
    if first.status == "OOM" or second.status == "OOM":
        status = "OOM"
    elif first.gpu_ms is None or second.gpu_ms is None:
        status = "incomplete"
    else:
        status = "ok"
    result = PairSummary(
        label=label,
        visits=[first.arm, second.arm],
        status=status,
    )
    if status != "ok":
        return result
    assert first.gpu_ms is not None and second.gpu_ms is not None
    result.mean_ms = (first.gpu_ms.mean + second.gpu_ms.mean) / 2.0
    result.median_ms = (first.gpu_ms.median + second.gpu_ms.median) / 2.0
    result.p95_ms = (first.gpu_ms.p95 + second.gpu_ms.p95) / 2.0
    result.peak_allocated_mb = (
        first.peak_allocated_mb + second.peak_allocated_mb
    ) / 2.0
    result.peak_reserved_mb = (first.peak_reserved_mb + second.peak_reserved_mb) / 2.0
    midpoint = (first.gpu_ms.median + second.gpu_ms.median) / 2.0
    result.repeat_spread_percent = (
        0.0
        if midpoint == 0.0
        else abs(first.gpu_ms.median - second.gpu_ms.median) / midpoint * 100.0
    )
    return result


def add_pair_delta(baseline: PairSummary, candidate: PairSummary) -> None:
    if (
        baseline.status != "ok"
        or candidate.status != "ok"
        or baseline.median_ms is None
        or candidate.median_ms is None
    ):
        return
    candidate.median_speedup = baseline.median_ms / candidate.median_ms
    candidate.median_delta_percent = 100.0 * (
        candidate.median_ms / baseline.median_ms - 1.0
    )


@dataclass(kw_only=True)
class VisionBatchExperiment:
    visits: list[Trial]
    pair_summary: list[PairSummary]


def print_pair_summary(summary: PairSummary) -> None:
    if summary.status != "ok":
        print(
            f"pair={summary.label} visits={summary.visits} status={summary.status}",
            flush=True,
        )
        return
    assert (
        summary.mean_ms is not None
        and summary.median_ms is not None
        and summary.p95_ms is not None
        and summary.peak_allocated_mb is not None
        and summary.peak_reserved_mb is not None
        and summary.repeat_spread_percent is not None
    )
    line = (
        f"pair={summary.label} visits={summary.visits} status=ok "
        f"mean_ms={summary.mean_ms:.3f} median_ms={summary.median_ms:.3f} "
        f"p95_ms={summary.p95_ms:.3f} "
        f"peak_allocated_mb={summary.peak_allocated_mb:.1f} "
        f"peak_reserved_mb={summary.peak_reserved_mb:.1f} "
        f"repeat_spread_percent={summary.repeat_spread_percent:.3f}"
    )
    if summary.median_speedup is not None:
        line += (
            f" median_speedup={summary.median_speedup:.4f}"
            f" median_delta_percent={summary.median_delta_percent:.3f}"
        )
    print(line, flush=True)


def _find_trial(trials: list[Trial], arm: str) -> Trial | None:
    return next((trial for trial in trials if trial.arm == arm), None)


def _pair_summaries(
    trials: list[Trial], pairs: tuple[tuple[str, str, str], ...]
) -> list[PairSummary]:
    summaries = []
    for label, first_arm, second_arm in pairs:
        first = _find_trial(trials, first_arm)
        second = _find_trial(trials, second_arm)
        if first is None or second is None:
            status: Literal["OOM", "incomplete"] = (
                "OOM"
                if (first is not None and first.status == "OOM")
                or (second is not None and second.status == "OOM")
                else "incomplete"
            )
            summaries.append(
                PairSummary(
                    label=label,
                    visits=[first_arm, second_arm],
                    status=status,
                )
            )
        else:
            summaries.append(pair_summary(first, second, label))
    return summaries


def vision_batch_sweep(
    operation: EncoderForward,
    batch_sizes: list[int],
    warmup: int,
    iters: int,
    limits: ParityLimits,
) -> VisionBatchExperiment:
    from benchmarks.eval.minicpmo_vision_measurement import benchmark_forward

    trials: list[Trial] = []
    baseline_output: torch.Tensor | None = None
    original_batch_size = operation.encoder.vision_batch_size
    requested_sizes = set(batch_sizes) | {BASELINE_BATCH_SIZE}
    visits = list(VISION_BATCH_VISITS)
    if 8 in requested_sizes:
        visits.append((8, "bs8-diagnostic"))
    try:
        for batch_size, arm in visits:
            if batch_size not in requested_sizes:
                continue
            operation.encoder.vision_batch_size = batch_size
            trial, output = benchmark_forward(
                operation,
                arm=arm,
                batch_size=batch_size,
                device=operation.encoder.device,
                warmup=warmup,
                iters=iters,
            )
            trials.append(trial)
            if trial.status == "OOM":
                print_trial(trial)
                continue
            else:
                assert output is not None
                if baseline_output is None and batch_size == BASELINE_BATCH_SIZE:
                    baseline_output = output
                    trial.parity = compare_outputs(output, output, limits)
                elif baseline_output is None:
                    trial.parity = None
                elif batch_size == BASELINE_BATCH_SIZE:
                    trial.parity = None
                else:
                    trial.parity = compare_outputs(baseline_output, output, limits)
                print_trial(trial)
                if trial.parity is not None and trial.parity.substantial_drift:
                    print(
                        "SUBSTANTIAL DRIFT: retaining scheduled visits; investigate before using the sweep",
                        flush=True,
                    )
                else:
                    pass
    finally:
        operation.encoder.vision_batch_size = original_batch_size
    pairs = _pair_summaries(
        trials,
        (
            ("bs16", "bs16-A", "bs16-B"),
            ("bs32", "bs32-A", "bs32-B"),
            ("bs64", "bs64-A", "bs64-B"),
        ),
    )
    baseline = pairs[0]
    for candidate in pairs[1:]:
        add_pair_delta(baseline, candidate)
    print("=== VISION BATCH PAIR SUMMARY ===", flush=True)
    for summary in pairs:
        print_pair_summary(summary)
    return VisionBatchExperiment(visits=trials, pair_summary=pairs)


@dataclass(kw_only=True)
class ResamplerExperiment:
    isolated_visits: list[Trial]
    isolated_pair_summary: list[PairSummary]
    full_encoder_visits: list[Trial]
    full_encoder_pair_summary: list[PairSummary]
    full_encoder_status: str


def resampler_experiment(
    operation: EncoderForward,
    warmup: int,
    iters: int,
    resampler_warmup: int,
    resampler_iters: int,
    limits: ParityLimits,
) -> ResamplerExperiment:
    from benchmarks.eval.minicpmo_vision_measurement import (
        ResamplerForward,
        benchmark_forward,
        capture_resampler_inputs,
    )

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
    for need_weights, arm in RESAMPLER_VISITS:
        with temporary_need_weights(attention, need_weights):
            trial, output = benchmark_forward(
                isolated_operation,
                arm=arm,
                batch_size=BASELINE_BATCH_SIZE,
                device=encoder.device,
                warmup=resampler_warmup,
                iters=resampler_iters,
            )
        isolated_trials.append(trial)
        if trial.status == "OOM":
            print_trial(trial)
            status = (
                "skipped: isolated resampler baseline OOM"
                if arm.startswith("R0")
                else "skipped: isolated resampler candidate OOM"
            )
            return ResamplerExperiment(
                isolated_visits=isolated_trials,
                isolated_pair_summary=_pair_summaries(
                    isolated_trials, (("R0", "R0-A", "R0-B"), ("R1", "R1-A", "R1-B"))
                ),
                full_encoder_visits=[],
                full_encoder_pair_summary=[],
                full_encoder_status=status,
            )
        else:
            assert output is not None
            if baseline_output is None:
                baseline_output = output
                trial.parity = compare_outputs(output, output, limits)
            elif arm.startswith("R0"):
                trial.parity = None
            else:
                trial.parity = compare_outputs(baseline_output, output, limits)
            print_trial(trial)
    isolated_pairs = _pair_summaries(
        isolated_trials, (("R0", "R0-A", "R0-B"), ("R1", "R1-A", "R1-B"))
    )
    add_pair_delta(isolated_pairs[0], isolated_pairs[1])
    print("=== ISOLATED RESAMPLER PAIR SUMMARY ===", flush=True)
    for summary in isolated_pairs:
        print_pair_summary(summary)
    candidate_visits = [
        _find_trial(isolated_trials, "R1-A"),
        _find_trial(isolated_trials, "R1-B"),
    ]
    if any(trial is None or trial.parity is None for trial in candidate_visits):
        return ResamplerExperiment(
            isolated_visits=isolated_trials,
            isolated_pair_summary=isolated_pairs,
            full_encoder_visits=[],
            full_encoder_pair_summary=[],
            full_encoder_status="skipped: isolated candidate parity unavailable",
        )
    if any(trial.parity.substantial_drift for trial in candidate_visits if trial):
        return ResamplerExperiment(
            isolated_visits=isolated_trials,
            isolated_pair_summary=isolated_pairs,
            full_encoder_visits=[],
            full_encoder_pair_summary=[],
            full_encoder_status="skipped: SUBSTANTIAL DRIFT",
        )
    elif (
        isolated_pairs[1].median_speedup is None
        or isolated_pairs[1].median_speedup <= 1.0
    ):
        return ResamplerExperiment(
            isolated_visits=isolated_trials,
            isolated_pair_summary=isolated_pairs,
            full_encoder_visits=[],
            full_encoder_pair_summary=[],
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
    for need_weights, arm in FULL_ENCODER_VISITS:
        with temporary_need_weights(attention, need_weights):
            trial, output = benchmark_forward(
                operation,
                arm=arm,
                batch_size=BASELINE_BATCH_SIZE,
                device=encoder.device,
                warmup=warmup,
                iters=iters,
            )
        full_trials.append(trial)
        if trial.status == "OOM":
            print_trial(trial)
            return ResamplerExperiment(
                isolated_visits=isolated_trials,
                isolated_pair_summary=isolated_pairs,
                full_encoder_visits=full_trials,
                full_encoder_pair_summary=_pair_summaries(
                    full_trials, (("V0", "V0-A", "V0-B"), ("V1", "V1-A", "V1-B"))
                ),
                full_encoder_status=(
                    "OOM: full encoder baseline"
                    if arm.startswith("V0")
                    else "skipped: full encoder candidate OOM"
                ),
            )
        else:
            assert output is not None
            if baseline_output is None:
                baseline_output = output
                trial.parity = compare_outputs(output, output, limits)
            elif arm.startswith("V0"):
                trial.parity = None
            else:
                trial.parity = compare_outputs(baseline_output, output, limits)
            print_trial(trial)
    full_pairs = _pair_summaries(
        full_trials, (("V0", "V0-A", "V0-B"), ("V1", "V1-A", "V1-B"))
    )
    add_pair_delta(full_pairs[0], full_pairs[1])
    print("=== FULL ENCODER RESAMPLER PAIR SUMMARY ===", flush=True)
    for summary in full_pairs:
        print_pair_summary(summary)
    return ResamplerExperiment(
        isolated_visits=isolated_trials,
        isolated_pair_summary=isolated_pairs,
        full_encoder_visits=full_trials,
        full_encoder_pair_summary=full_pairs,
        full_encoder_status="measured",
    )

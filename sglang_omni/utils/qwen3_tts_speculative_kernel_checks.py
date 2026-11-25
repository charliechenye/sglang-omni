# SPDX-License-Identifier: Apache-2.0
"""Parity checks and timing helpers for Predictor kernel microbenchmarks."""

from __future__ import annotations

import statistics
from typing import Protocol

import torch

from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    SpeculativeAcceptanceBatch,
)
from sglang_omni.utils.qwen3_tts_speculative_frozen import FrozenPredictorArtifacts


class KernelOperation(Protocol):
    def __call__(self) -> None: ...


def cuda_median_ms(
    operation: KernelOperation, *, warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
    else:
        pass
    torch.cuda.synchronize()
    elapsed_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        finish.record()
        finish.synchronize()
        elapsed_ms.append(start.elapsed_time(finish))
    return statistics.median(elapsed_ms)


def assert_probability_parity(
    actual_probabilities: torch.Tensor,
    actual_token_ids: torch.Tensor,
    reference_probabilities: torch.Tensor,
    reference_token_ids: torch.Tensor,
) -> None:
    if actual_probabilities.shape != reference_probabilities.shape:
        raise AssertionError("probability kernels returned different compact widths")
    else:
        pass
    if not torch.equal(actual_probabilities > 0.0, reference_probabilities > 0.0):
        raise AssertionError("probability kernels produced different support masks")
    else:
        pass
    positive = reference_probabilities > 0.0
    if not torch.equal(actual_token_ids[positive], reference_token_ids[positive]):
        raise AssertionError("probability kernels produced different token support")
    else:
        pass
    torch.testing.assert_close(
        actual_probabilities.double().sum(dim=-1),
        torch.ones(
            actual_probabilities.shape[0],
            device=actual_probabilities.device,
            dtype=torch.float64,
        ),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual_probabilities,
        reference_probabilities,
        atol=1e-6,
        rtol=1e-5,
    )


def build_microbenchmark_tensors(
    artifacts: FrozenPredictorArtifacts,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = artifacts.frames
    hidden_states = torch.stack([frame.hidden for frame in frames]).to(device=device)
    logits = torch.stack([frame.logits for frame in frames]).to(device=device)
    temperatures = torch.stack(
        [frame.sampling["sub_temperature_tensor"] for frame in frames]
    ).to(device=device, dtype=torch.float32)
    top_ks = torch.stack([frame.sampling["sub_top_k_tensor"] for frame in frames]).to(
        device=device, dtype=torch.long
    )
    top_ps = torch.stack([frame.sampling["sub_top_p_tensor"] for frame in frames]).to(
        device=device, dtype=torch.float32
    )
    return hidden_states, logits, temperatures, top_ks, top_ps


def assert_grouped_lm_head_rounding_bound(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    packed_weights: torch.Tensor,
    head_indices: torch.Tensor,
) -> None:
    selected_weights = packed_weights[head_indices].double()
    expanded_hidden = hidden_states.double().unsqueeze(1)
    products = selected_weights * expanded_hidden
    reference = products.sum(dim=-1)
    absolute_product_sums = products.abs().sum(dim=-1)
    rounded_reference = reference.to(torch.bfloat16)
    unit_roundoff = 2.0**-24
    gamma = 64 * unit_roundoff / (1 - 64 * unit_roundoff)
    fp32_error_bound = gamma * absolute_product_sums
    previous = torch.nextafter(logits, torch.full_like(logits, -float("inf"))).double()
    following = torch.nextafter(logits, torch.full_like(logits, float("inf"))).double()
    values = logits.double()
    lower = (previous + values) * 0.5
    upper = (following + values) * 0.5
    outside_interval = torch.maximum(lower - reference, reference - upper).clamp_min(
        0.0
    )
    if bool((outside_interval > fp32_error_bound).any()):
        raise AssertionError("grouped LM-head output exceeded the BF16 rounding bound")
    else:
        pass


def allocate_acceptance_outputs(
    batch_size: int, verify_width: int, device: torch.device
) -> SpeculativeAcceptanceBatch:
    return SpeculativeAcceptanceBatch(
        committed_tokens=torch.empty(
            (batch_size, verify_width), device=device, dtype=torch.long
        ),
        accepted_draft_counts=torch.empty(batch_size, device=device, dtype=torch.long),
        committed_token_counts=torch.empty(batch_size, device=device, dtype=torch.long),
        rejected_rows=torch.empty(batch_size, device=device, dtype=torch.bool),
        bonus_rows=torch.empty(batch_size, device=device, dtype=torch.bool),
        invalid_rows=torch.empty(batch_size, device=device, dtype=torch.bool),
    )

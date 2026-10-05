# SPDX-License-Identifier: Apache-2.0
"""Timing summaries and output comparisons for the vision experiments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(kw_only=True)
class Distribution:
    minimum: float
    mean: float
    median: float
    p95: float
    maximum: float


def summarize(values: list[float] | list[int]) -> Distribution:
    if not values:
        raise ValueError("Cannot summarize an empty measurement list")
    else:
        return Distribution(
            minimum=float(np.min(values)),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            maximum=float(np.max(values)),
        )


@dataclass(kw_only=True)
class ParityLimits:
    max_absolute_error: float = 0.1
    max_relative_l2: float = 0.01
    min_cosine_similarity: float = 0.9999


@dataclass(kw_only=True)
class Parity:
    baseline_shape: list[int]
    candidate_shape: list[int]
    baseline_dtype: str
    candidate_dtype: str
    shape_equal: bool
    dtype_equal: bool
    torch_equal: bool
    finite: bool
    max_abs: float | None = None
    mean_abs: float | None = None
    rel_l2: float | None = None
    cosine_similarity: float | None = None
    substantial_drift: bool = True


def compare_outputs(
    baseline: torch.Tensor, candidate: torch.Tensor, limits: ParityLimits
) -> Parity:
    baseline = baseline.detach().cpu()
    candidate = candidate.detach().cpu()
    parity = Parity(
        baseline_shape=list(baseline.shape),
        candidate_shape=list(candidate.shape),
        baseline_dtype=str(baseline.dtype),
        candidate_dtype=str(candidate.dtype),
        shape_equal=baseline.shape == candidate.shape,
        dtype_equal=baseline.dtype == candidate.dtype,
        torch_equal=torch.equal(baseline, candidate),
        finite=bool(torch.isfinite(baseline).all() and torch.isfinite(candidate).all()),
    )
    if not parity.shape_equal or not parity.finite or baseline.numel() == 0:
        return parity
    else:
        reference = baseline.to(torch.float64).flatten()
        measured = candidate.to(torch.float64).flatten()
        difference = measured - reference
        reference_norm = float(torch.linalg.vector_norm(reference))
        measured_norm = float(torch.linalg.vector_norm(measured))
        difference_norm = float(torch.linalg.vector_norm(difference))
        parity.max_abs = float(difference.abs().max())
        parity.mean_abs = float(difference.abs().mean())
        parity.rel_l2 = difference_norm / max(
            reference_norm, torch.finfo(torch.float64).eps
        )
        if reference_norm == 0.0 and measured_norm == 0.0:
            parity.cosine_similarity = 1.0
        else:
            parity.cosine_similarity = float(torch.dot(reference, measured)) / max(
                reference_norm * measured_norm, torch.finfo(torch.float64).tiny
            )
        parity.substantial_drift = (
            not parity.dtype_equal
            or parity.max_abs > limits.max_absolute_error
            or parity.rel_l2 > limits.max_relative_l2
            or parity.cosine_similarity < limits.min_cosine_similarity
        )
        return parity


@contextmanager
def temporary_need_weights(
    attention: nn.MultiheadAttention, need_weights: bool | None
) -> Iterator[None]:
    if need_weights is None:
        yield
    else:

        def replace_option(
            module: nn.Module,
            attention_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            attention_options: dict[str, torch.Tensor | bool | None],
        ) -> tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            dict[str, torch.Tensor | bool | None],
        ]:
            return attention_inputs, {**attention_options, "need_weights": need_weights}

        hook = attention.register_forward_pre_hook(replace_option, with_kwargs=True)
        try:
            yield
        finally:
            hook.remove()

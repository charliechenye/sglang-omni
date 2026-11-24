# SPDX-License-Identifier: Apache-2.0
"""CPU and CUDA parity tests for row-selected CodePredictor LM heads."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    build_speculative_probability_candidates,
)
from sglang_omni.utils.predictor_grouped_heads import (
    grouped_lm_head_cuda_available,
    grouped_lm_head_linear,
)


def reference_grouped_logits(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    hidden_indices: torch.Tensor,
    head_indices: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for hidden_index, head_index, active in zip(
        hidden_indices, head_indices, active_rows
    ):
        if bool(active):
            rows.append(F.linear(hidden[hidden_index], weights[head_index]))
        else:
            rows.append(torch.zeros(weights.shape[1], dtype=hidden.dtype))
    if not rows:
        return torch.empty((0, weights.shape[1]), dtype=hidden.dtype)
    else:
        pass
    return torch.stack(rows)


def test_grouped_lm_head_cpu_selects_rows_heads_and_masks_padding() -> None:
    torch.manual_seed(5)
    hidden = torch.randn(3, 7)
    weights = torch.randn(4, 11, 7)
    hidden_indices = torch.tensor([2, 0, 1, 99])
    head_indices = torch.tensor([1, 3, 0, 99])
    active_rows = torch.tensor([True, True, False, False])

    actual = grouped_lm_head_linear(
        hidden, weights, hidden_indices, head_indices, active_rows
    )
    expected = reference_grouped_logits(
        hidden, weights, hidden_indices, head_indices, active_rows
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not grouped_lm_head_cuda_available(), reason="Triton grouped kernel is required"
)
@pytest.mark.parametrize("verify_width", [2, 3, 4, 8])
def test_grouped_lm_head_cuda_matches_native_linear(verify_width: int) -> None:
    torch.manual_seed(17)
    batch_size = 3
    head_count = 16
    vocab_size = 2048
    hidden_size = 1024
    row_count = batch_size * verify_width
    hidden = torch.randn(row_count, hidden_size, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        head_count,
        vocab_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_indices = torch.arange(row_count, device="cuda")
    head_indices = torch.arange(row_count, device="cuda") % head_count
    active_rows = torch.ones(row_count, device="cuda", dtype=torch.bool)
    active_rows[-1] = False

    actual = grouped_lm_head_linear(
        hidden, weights, hidden_indices, head_indices, active_rows
    )
    expected = torch.stack(
        [F.linear(hidden[row], weights[head_indices[row]]) for row in range(row_count)]
    )
    expected[-1].zero_()

    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.02)
    temperatures = torch.tensor(
        [0.7 + 0.1 * (row % 3) for row in range(row_count)], device="cuda"
    )
    top_ks = torch.full((row_count,), 4, device="cuda", dtype=torch.long)
    top_ps = torch.tensor(
        [0.6 if row % 2 == 0 else 1.0 for row in range(row_count)], device="cuda"
    )
    grouped_candidates = build_speculative_probability_candidates(
        actual.contiguous(),
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=4,
        has_top_p=True,
        has_unbounded_top_k=False,
    )
    native_candidates = build_speculative_probability_candidates(
        expected.contiguous(),
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=4,
        has_top_p=True,
        has_unbounded_top_k=False,
    )
    assert grouped_candidates is not None
    assert native_candidates is not None
    grouped_probabilities = torch.zeros((row_count, vocab_size), device="cuda")
    native_probabilities = torch.zeros_like(grouped_probabilities)
    grouped_probabilities.scatter_add_(
        1, grouped_candidates[1].long(), grouped_candidates[0]
    )
    native_probabilities.scatter_add_(
        1, native_candidates[1].long(), native_candidates[0]
    )
    torch.testing.assert_close(
        grouped_probabilities, native_probabilities, atol=3e-5, rtol=3e-4
    )

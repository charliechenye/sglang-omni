# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks for the fixed P16 runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.speculative_execution import PredictorSpeculativeMixin
from sglang_omni.models.qwen3_tts.speculative_runtime import (
    is_fatal_cuda_capture_error,
    speculative_opening_code_output_view,
    speculative_sampling_fallback_reason,
)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_opening_pair_code_output_view_supports_batched_copies(
    batch_size: int,
) -> None:
    output_codes = torch.full((batch_size, 16), -1, dtype=torch.long)
    layer0_codes = torch.arange(10, 10 + batch_size, dtype=torch.long).unsqueeze(1)
    layer1_codes = torch.arange(30, 30 + batch_size, dtype=torch.long)
    pending_tokens = torch.full((batch_size,), -1, dtype=torch.long)

    code_outputs = speculative_opening_code_output_view(output_codes, batch_size)
    code_outputs[:, 0].copy_(layer0_codes[:, 0])
    code_outputs[:, 1].copy_(layer1_codes)
    pending_tokens.copy_(code_outputs[:, 1])

    assert code_outputs.shape == (batch_size, 16)
    assert code_outputs[:, 0].shape == (batch_size,)
    assert code_outputs[:, 1].shape == (batch_size,)
    assert pending_tokens.shape == (batch_size,)
    torch.testing.assert_close(code_outputs[:, 0], layer0_codes[:, 0])
    torch.testing.assert_close(code_outputs[:, 1], layer1_codes)
    torch.testing.assert_close(pending_tokens, layer1_codes)


def test_p16_sampling_eligibility_keeps_native_fallbacks() -> None:
    assert (
        speculative_sampling_fallback_reason(
            ("request",),
            (True,),
            has_unbounded_top_k=False,
        )
        is None
    )
    assert (
        speculative_sampling_fallback_reason(
            ("request-a", "request-b"),
            (True, False),
            has_unbounded_top_k=False,
        )
        == "greedy-or-mixed-sampling-requires-native-decoding"
    )
    assert (
        speculative_sampling_fallback_reason(
            ("request",),
            (True,),
            has_unbounded_top_k=True,
        )
        == "unsupported-sampling-configuration"
    )


@pytest.mark.parametrize(
    "message",
    ["CUDA error 719: unspecified launch failure", "device-side assert triggered"],
)
def test_fatal_cuda_capture_errors_are_not_retryable(message: str) -> None:
    assert is_fatal_cuda_capture_error(RuntimeError(message))


def test_ordinary_capture_error_is_not_fatal() -> None:
    assert not is_fatal_cuda_capture_error(RuntimeError("capture shape mismatch"))


def test_speculative_audit_observer_cannot_reuse_graphs_with_stale_buffers() -> None:
    talker = SimpleNamespace(
        code_predictor_spec_audit_observer=None,
        code_predictor_spec_effective_width=4,
        code_predictor_spec_round_graphs={"captured": object()},
        code_predictor_spec_opening_graphs={},
    )

    with pytest.raises(RuntimeError, match="before graph capture"):
        PredictorSpeculativeMixin.set_code_predictor_speculative_audit_observer(
            talker, lambda event, data: None
        )

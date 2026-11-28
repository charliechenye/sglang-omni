# SPDX-License-Identifier: Apache-2.0
"""Small runtime helpers for the fixed P16 speculative path."""

from __future__ import annotations

import torch


def speculative_opening_code_output_view(
    output_codes: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Return the persistent 2D code output slice used by opening-pair copies."""
    return output_codes[:batch_size]


def speculative_sampling_fallback_reason(
    request_ids: tuple[str, ...],
    sample_rows: tuple[bool, ...],
    *,
    has_unbounded_top_k: bool,
) -> str | None:
    """Resolve request-level sampling eligibility without reading device state."""
    if len(request_ids) != len(sample_rows):
        return "sampling-request-metadata-mismatch"
    elif not request_ids or any(not request_id for request_id in request_ids):
        return "missing-request-identity"
    elif not all(sample_rows):
        return "greedy-or-mixed-sampling-requires-native-decoding"
    elif has_unbounded_top_k:
        return "unsupported-sampling-configuration"
    else:
        return None


def is_fatal_cuda_capture_error(error: Exception) -> bool:
    """Identify CUDA capture failures that make eager reuse unsafe."""
    error_message = f"{type(error).__name__}: {error}".lower()
    fatal_markers = (
        "illegal memory access",
        "illegal address",
        "misaligned address",
        "device-side assert",
        "device assert",
        "unspecified launch failure",
        "cudaerrorillegaladdress",
        "cudaerrorlaunchfailure",
        "error code 700",
        "error code 719",
        "error 700",
        "error 719",
        "cuda error 700",
        "cuda error 719",
        "context is destroyed",
        "context has been destroyed",
    )
    return any(marker in error_message for marker in fatal_markers)

# SPDX-License-Identifier: Apache-2.0
"""Small shape checks shared by frozen-frame observation and CPU tests."""

from __future__ import annotations

import torch


def target_logits_vector_for_query(
    logits_by_query: torch.Tensor, query_offset: int
) -> torch.Tensor:
    """Return one complete vocabulary vector from a [query, vocabulary] tensor."""
    if logits_by_query.ndim != 2:
        raise ValueError("observed target logits must have [query, vocabulary] shape")
    elif query_offset < 0 or query_offset >= logits_by_query.shape[0]:
        raise IndexError("target query offset is outside the observed logits")
    else:
        pass
    return logits_by_query[query_offset]

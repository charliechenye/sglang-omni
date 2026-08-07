# SPDX-License-Identifier: Apache-2.0
"""Batch-carried prefill inputs for breakable prefill CUDA graphs.

Runner-composed prefill conditioning (embeddings plus request identity)
crosses the graph boundary as an OmniPrefillInputs payload stored on
ForwardBatch.mm_inputs: the prefill graph gate rejects batches carrying
input_embeds, and the replay-time static batch preserves mm_inputs but not
rids. The model's forward reads the payload on both the graph and eager
paths. On a real prefill batch mm_inputs is never None: prepare_for_extend
fills it with one entry per request (None for text-only requests), and
attaching replaces that placeholder list on the ForwardBatch only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OmniPrefillInputs:
    """Per-forward prefill conditioning composed by an omni model runner.

    input_embeds covers exactly the extend-window tokens of the batch, in
    model dtype. rids carries request identity for the eager tail.
    """

    input_embeds: torch.Tensor
    rids: tuple[str, ...]


def attach_omni_prefill_inputs(
    forward_batch: Any, prefill_inputs: OmniPrefillInputs
) -> None:
    """Stow prefill_inputs on forward_batch.mm_inputs."""
    if forward_batch.input_embeds is not None:
        raise RuntimeError(
            "OmniPrefillInputs requires forward_batch.input_embeds to stay "
            "None; the model forward consumes the payload instead"
        )
    mm_inputs = forward_batch.mm_inputs
    if isinstance(mm_inputs, OmniPrefillInputs):
        raise RuntimeError(
            "forward_batch.mm_inputs already carries OmniPrefillInputs; "
            "refusing to attach twice to the same batch"
        )
    if mm_inputs is not None and any(item is not None for item in mm_inputs):
        raise RuntimeError(
            "forward_batch.mm_inputs carries SGLang multimodal inputs; "
            "refusing to overwrite them with OmniPrefillInputs"
        )
    num_tokens = len(forward_batch.input_ids)
    if prefill_inputs.input_embeds.shape[0] != num_tokens:
        raise RuntimeError(
            "OmniPrefillInputs embeddings must cover the extend-window tokens: "
            f"embeds rows={prefill_inputs.input_embeds.shape[0]}, "
            f"batch tokens={num_tokens}"
        )
    if len(prefill_inputs.rids) != forward_batch.batch_size:
        raise RuntimeError(
            "OmniPrefillInputs rids must cover the batch: "
            f"rids={len(prefill_inputs.rids)}, "
            f"batch_size={forward_batch.batch_size}"
        )
    forward_batch.mm_inputs = prefill_inputs


def get_omni_prefill_inputs(forward_batch: Any) -> OmniPrefillInputs | None:
    """Return the attached payload, or None for batches without one,
    including batches whose mm_inputs carries genuine multimodal inputs."""
    payload = forward_batch.mm_inputs
    if isinstance(payload, OmniPrefillInputs):
        return payload
    return None


__all__ = [
    "OmniPrefillInputs",
    "attach_omni_prefill_inputs",
    "get_omni_prefill_inputs",
]

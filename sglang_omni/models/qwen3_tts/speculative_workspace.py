# SPDX-License-Identifier: Apache-2.0
"""Persistent scratch storage for Qwen3-TTS speculative frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    SpeculativeAcceptanceBatch,
)
from sglang_omni.models.qwen3_tts.speculative_prepare import SpeculativeVerifierInputs

PredictorGraphSignature: TypeAlias = tuple[
    Literal["argmax", "sampled"], int, bool, bool, bool
]
PredictorSpeculativeRoundGraphKey: TypeAlias = tuple[
    int, int, int, PredictorGraphSignature
]


def compact_candidate_view(
    storage: torch.Tensor, row_count: int, candidate_count: int
) -> torch.Tensor:
    """View a reserved candidate arena as tightly packed rows for one Top-K shape."""
    if storage.ndim != 2 or row_count < 0 or candidate_count <= 0:
        raise ValueError("candidate storage and requested view dimensions are invalid")
    elif candidate_count > storage.shape[1] or row_count > storage.shape[0]:
        raise ValueError("candidate view exceeds its reserved capacity")
    else:
        pass
    return storage.view(-1)[: row_count * candidate_count].view(
        row_count, candidate_count
    )


def compact_scratch_view(
    storage: torch.Tensor, row_count: int, columns_per_row: int
) -> torch.Tensor:
    """View wider persistent scratch as contiguous rows with a smaller signature."""
    if storage.ndim != 2 or row_count < 0 or columns_per_row <= 0:
        raise ValueError("scratch storage and requested view dimensions are invalid")
    elif row_count > storage.shape[0] or columns_per_row > storage.shape[1]:
        raise ValueError("scratch view exceeds its reserved capacity")
    else:
        pass
    return storage.view(-1)[: row_count * columns_per_row].view(
        row_count, columns_per_row
    )


@dataclass(kw_only=True)
class PredictorSpeculativeRoundWorkspace:
    verify_width: int
    proposal_width: int
    proposal_hidden_rows: torch.Tensor
    proposal_head_offsets: torch.Tensor
    proposal_head_indices: torch.Tensor
    proposal_active_rows: torch.Tensor
    proposal_counts: torch.Tensor
    valid_query_counts: torch.Tensor
    proposal_temperatures: torch.Tensor
    proposal_top_ks: torch.Tensor
    proposal_top_ps: torch.Tensor
    target_temperatures: torch.Tensor
    target_top_ks: torch.Tensor
    target_top_ps: torch.Tensor
    proposal_logits: torch.Tensor
    target_logits: torch.Tensor
    proposal_probabilities: torch.Tensor
    proposal_token_ids: torch.Tensor
    target_probabilities: torch.Tensor
    target_token_ids: torch.Tensor
    proposal_tokens: torch.Tensor
    acceptance_outputs: SpeculativeAcceptanceBatch
    verifier_inputs: SpeculativeVerifierInputs
    split_candidate_keys: torch.Tensor | None


class PredictorSpeculativeWorkspace:
    """Own fixed-capacity frame and per-width round buffers for one Talker."""

    def __init__(
        self,
        max_batch_size: int,
        max_verify_width: int,
        vocab_size: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        device: torch.device,
        max_top_k: int,
        candidate_width: int,
        split_top_k_chunks: int,
        *,
        seed_rows: torch.Tensor,
        uniforms: torch.Tensor | None,
        rng_pool: torch.Tensor | None,
        metadata_device: torch.Tensor,
        metadata_host: torch.Tensor,
        verifier_input_buffers: dict[int, SpeculativeVerifierInputs],
    ) -> None:
        if max_batch_size != 1:
            raise ValueError("P16 speculative workspace supports batch size one")
        elif candidate_width <= 0 or split_top_k_chunks <= 0:
            raise ValueError("workspace candidate dimensions must be positive")
        else:
            pass
        if not 1 <= max_verify_width <= 4:
            raise ValueError("workspace verification width must be in [1, 4]")
        else:
            pass
        self.max_batch_size = max_batch_size
        self.max_verify_width = max_verify_width
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_dtype = hidden_dtype
        self.device = device
        self.max_top_k = max_top_k
        self.candidate_width = candidate_width
        self.seed_rows = seed_rows
        self.uniforms = (
            torch.empty(
                (max_batch_size, 4, max_verify_width),
                device=device,
                dtype=torch.float32,
            )
            if uniforms is None
            else uniforms
        )
        self.rng_pool = rng_pool
        self.metadata_device = metadata_device
        self.metadata_host = metadata_host
        self.committed_depths = torch.empty(
            max_batch_size, device=device, dtype=torch.long
        )
        self.logical_kv_lengths = torch.empty_like(self.committed_depths)
        self.pending_tokens = torch.empty_like(self.committed_depths)
        self.semantic_positions = torch.empty_like(self.committed_depths)
        self.preceding_hidden = torch.empty(
            max_batch_size, hidden_size, device=device, dtype=hidden_dtype
        )
        self.active_rows = torch.empty(max_batch_size, device=device, dtype=torch.bool)
        self.rounds: dict[int, PredictorSpeculativeRoundWorkspace] = {}
        for verify_width in range(1, max_verify_width + 1):
            proposal_width = verify_width - 1
            proposal_row_count = max_batch_size * proposal_width
            target_row_count = max_batch_size * verify_width
            proposal_hidden_rows = torch.arange(
                proposal_row_count, device=device, dtype=torch.long
            )
            if proposal_width > 0:
                proposal_hidden_rows.div_(proposal_width, rounding_mode="floor")
                proposal_head_offsets = torch.arange(
                    proposal_row_count, device=device, dtype=torch.long
                ).remainder_(proposal_width)
            else:
                proposal_head_offsets = torch.empty(
                    (0,), device=device, dtype=torch.long
                )
            proposal_active_rows = torch.empty(
                proposal_row_count, device=device, dtype=torch.bool
            )
            proposal_head_indices = torch.empty(
                proposal_row_count, device=device, dtype=torch.long
            )
            proposal_counts = torch.empty(
                max_batch_size, device=device, dtype=torch.long
            )
            valid_query_counts = torch.empty_like(proposal_counts)
            proposal_temperatures = torch.empty(
                proposal_row_count, device=device, dtype=torch.float32
            )
            proposal_top_ks = torch.empty(
                proposal_row_count, device=device, dtype=torch.long
            )
            proposal_top_ps = torch.empty_like(proposal_temperatures)
            target_temperatures = torch.empty(
                target_row_count, device=device, dtype=torch.float32
            )
            target_top_ks = torch.empty(
                target_row_count, device=device, dtype=torch.long
            )
            target_top_ps = torch.empty_like(target_temperatures)
            proposal_logits = torch.empty(
                proposal_row_count,
                vocab_size,
                device=device,
                dtype=hidden_dtype,
            )
            target_logits = torch.empty(
                target_row_count,
                vocab_size,
                device=device,
                dtype=hidden_dtype,
            )
            proposal_probabilities = torch.empty(
                proposal_row_count,
                candidate_width,
                device=device,
                dtype=torch.float32,
            )
            proposal_token_ids = torch.empty(
                proposal_row_count,
                candidate_width,
                device=device,
                dtype=torch.int32,
            )
            target_probabilities = torch.empty(
                target_row_count,
                candidate_width,
                device=device,
                dtype=torch.float32,
            )
            target_token_ids = torch.empty(
                target_row_count,
                candidate_width,
                device=device,
                dtype=torch.int32,
            )
            proposal_tokens = torch.empty(
                max_batch_size,
                proposal_width,
                device=device,
                dtype=torch.long,
            )
            committed_tokens = torch.empty(
                max_batch_size, verify_width, device=device, dtype=torch.long
            )
            accepted_draft_counts = torch.empty_like(proposal_counts)
            committed_token_counts = torch.empty_like(proposal_counts)
            rejected_rows = torch.empty(max_batch_size, device=device, dtype=torch.bool)
            bonus_rows = torch.empty_like(rejected_rows)
            invalid_rows = torch.empty_like(rejected_rows)
            split_candidate_keys = None
            if max_top_k > 32:
                split_candidate_keys = torch.empty(
                    proposal_row_count + target_row_count,
                    split_top_k_chunks * candidate_width,
                    device=device,
                    dtype=torch.int64,
                )
            else:
                pass
            self.rounds[verify_width] = PredictorSpeculativeRoundWorkspace(
                verify_width=verify_width,
                proposal_width=proposal_width,
                proposal_hidden_rows=proposal_hidden_rows,
                proposal_head_offsets=proposal_head_offsets,
                proposal_head_indices=proposal_head_indices,
                proposal_active_rows=proposal_active_rows,
                proposal_counts=proposal_counts,
                valid_query_counts=valid_query_counts,
                proposal_temperatures=proposal_temperatures,
                proposal_top_ks=proposal_top_ks,
                proposal_top_ps=proposal_top_ps,
                target_temperatures=target_temperatures,
                target_top_ks=target_top_ks,
                target_top_ps=target_top_ps,
                proposal_logits=proposal_logits,
                target_logits=target_logits,
                proposal_probabilities=proposal_probabilities,
                proposal_token_ids=proposal_token_ids,
                target_probabilities=target_probabilities,
                target_token_ids=target_token_ids,
                proposal_tokens=proposal_tokens,
                acceptance_outputs=SpeculativeAcceptanceBatch(
                    committed_tokens=committed_tokens,
                    accepted_draft_counts=accepted_draft_counts,
                    committed_token_counts=committed_token_counts,
                    rejected_rows=rejected_rows,
                    bonus_rows=bonus_rows,
                    invalid_rows=invalid_rows,
                ),
                verifier_inputs=verifier_input_buffers[verify_width],
                split_candidate_keys=split_candidate_keys,
            )

    @property
    def tensor_storage(self) -> list[torch.Tensor]:
        tensors = [
            self.seed_rows,
            self.metadata_host,
            self.committed_depths,
            self.logical_kv_lengths,
            self.pending_tokens,
            self.semantic_positions,
            self.preceding_hidden,
            self.active_rows,
            self.metadata_device,
        ]
        if self.uniforms is not None:
            tensors.append(self.uniforms)
        else:
            pass
        if self.rng_pool is not None:
            tensors.append(self.rng_pool)
        else:
            pass
        for round_workspace in self.rounds.values():
            tensors.extend(
                (
                    round_workspace.proposal_hidden_rows,
                    round_workspace.proposal_head_offsets,
                    round_workspace.proposal_head_indices,
                    round_workspace.proposal_active_rows,
                    round_workspace.proposal_counts,
                    round_workspace.valid_query_counts,
                    round_workspace.proposal_temperatures,
                    round_workspace.proposal_top_ks,
                    round_workspace.proposal_top_ps,
                    round_workspace.target_temperatures,
                    round_workspace.target_top_ks,
                    round_workspace.target_top_ps,
                    round_workspace.proposal_logits,
                    round_workspace.target_logits,
                    round_workspace.proposal_probabilities,
                    round_workspace.proposal_token_ids,
                    round_workspace.target_probabilities,
                    round_workspace.target_token_ids,
                    round_workspace.proposal_tokens,
                    round_workspace.acceptance_outputs.committed_tokens,
                    round_workspace.acceptance_outputs.accepted_draft_counts,
                    round_workspace.acceptance_outputs.committed_token_counts,
                    round_workspace.acceptance_outputs.rejected_rows,
                    round_workspace.acceptance_outputs.bonus_rows,
                    round_workspace.acceptance_outputs.invalid_rows,
                    round_workspace.verifier_inputs.token_embeddings,
                    round_workspace.verifier_inputs.positions,
                    round_workspace.verifier_inputs.target_head_indices,
                    round_workspace.verifier_inputs.valid_queries,
                )
            )
            if round_workspace.split_candidate_keys is not None:
                tensors.append(round_workspace.split_candidate_keys)
            else:
                pass
        return tensors

    @property
    def tensor_count(self) -> int:
        return len(self.tensor_storage)

    @property
    def memory_bytes(self) -> int:
        unique_storage = {tensor.data_ptr(): tensor for tensor in self.tensor_storage}
        return sum(
            tensor.numel() * tensor.element_size() for tensor in unique_storage.values()
        )

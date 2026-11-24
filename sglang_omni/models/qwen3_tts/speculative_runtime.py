# SPDX-License-Identifier: Apache-2.0
"""Tensor operations for Qwen3-TTS speculative rounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpeculativeAcceptanceBatch:
    committed_tokens: torch.Tensor
    accepted_draft_counts: torch.Tensor
    committed_token_counts: torch.Tensor
    rejected_rows: torch.Tensor
    bonus_rows: torch.Tensor


def speculative_sampling_fallback_reason(
    request_ids: tuple[str, ...],
    explicit_seed_rows: tuple[bool, ...],
    sample_rows: tuple[bool, ...],
    *,
    has_unbounded_top_k: bool,
    deterministic_mode: bool,
) -> str | None:
    """Resolve request-level sampling eligibility without inspecting device data."""
    if len(request_ids) != len(explicit_seed_rows) or len(request_ids) != len(
        sample_rows
    ):
        return "sampling-request-metadata-mismatch"
    else:
        pass
    if deterministic_mode:
        return "deterministic-mode-requires-native-decoding"
    else:
        pass
    if any(explicit_seed_rows):
        return "explicit-seed-requires-native-decoding"
    else:
        pass
    if not request_ids or any(not request_id for request_id in request_ids):
        return "missing-request-identity"
    else:
        pass
    if not all(sample_rows):
        return "greedy-or-mixed-sampling-requires-native-decoding"
    else:
        pass
    if has_unbounded_top_k:
        return "unsupported-sampling-configuration"
    else:
        pass
    return None


def sample_compact_probability_rows(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    uniforms: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Sample compact normalized rows with inverse-CDF uniforms in [0, 1)."""
    if probabilities.ndim != 2 or token_ids.shape != probabilities.shape:
        raise ValueError(
            "compact probabilities and token IDs must have equal 2D shapes"
        )
    else:
        pass
    row_count, candidate_count = probabilities.shape
    if uniforms.shape != (row_count,) or active_rows.shape != (row_count,):
        raise ValueError("uniforms and active mask must contain one value per row")
    else:
        pass
    if candidate_count == 0:
        raise ValueError("compact probability rows must contain candidates")
    else:
        pass
    cumulative = probabilities.cumsum(dim=-1)
    selected_ranks = (cumulative <= uniforms[:, None]).sum(dim=-1)
    candidate_ranks = torch.arange(candidate_count, device=probabilities.device)
    last_positive_ranks = torch.where(
        probabilities > 0,
        candidate_ranks[None, :],
        torch.zeros_like(candidate_ranks)[None, :],
    ).amax(dim=-1)
    selected_ranks = torch.minimum(
        selected_ranks.clamp(max=candidate_count - 1), last_positive_ranks
    )
    selected_tokens = token_ids.gather(1, selected_ranks[:, None]).squeeze(1).long()
    return torch.where(active_rows, selected_tokens, torch.zeros_like(selected_tokens))


def expand_compact_probability_rows(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Scatter unique compact candidates into dense vocabulary distributions."""
    if probabilities.ndim != 2 or token_ids.shape != probabilities.shape:
        raise ValueError(
            "compact probabilities and token IDs must have equal 2D shapes"
        )
    else:
        pass
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    else:
        pass
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compact token IDs must use an integer dtype")
    else:
        pass
    if (
        token_ids.device.type == "cpu"
        and token_ids.numel()
        and ((token_ids < 0).any() or (token_ids >= vocab_size).any())
    ):
        raise ValueError("compact token IDs are outside the vocabulary")
    else:
        pass
    dense = torch.zeros(
        (probabilities.shape[0], vocab_size),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    dense.scatter_add_(1, token_ids.long(), probabilities)
    return dense


def accept_speculative_candidates(
    draft_tokens: torch.Tensor,
    draft_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    proposal_counts: torch.Tensor,
    active_rows: torch.Tensor,
    acceptance_uniforms: torch.Tensor,
    correction_uniforms: torch.Tensor,
    bonus_uniforms: torch.Tensor,
) -> SpeculativeAcceptanceBatch:
    """Accept a draft prefix and sample one residual correction or trailing bonus."""
    if draft_tokens.ndim != 2 or draft_probabilities.ndim != 3:
        raise ValueError("draft tokens and probabilities have invalid ranks")
    else:
        pass
    batch_size, draft_width = draft_tokens.shape
    if draft_width == 0:
        raise ValueError("speculative acceptance requires at least one draft slot")
    else:
        pass
    if draft_probabilities.shape[:2] != (batch_size, draft_width):
        raise ValueError("draft probability rows do not match draft token rows")
    else:
        pass
    vocab_size = int(draft_probabilities.shape[2])
    if target_probabilities.shape != (batch_size, draft_width + 1, vocab_size):
        raise ValueError("target probabilities must include the trailing bonus row")
    else:
        pass
    if proposal_counts.shape != (batch_size,) or active_rows.shape != (batch_size,):
        raise ValueError(
            "proposal counts and active mask must contain one value per row"
        )
    else:
        pass
    if acceptance_uniforms.shape != (batch_size, draft_width):
        raise ValueError("acceptance uniforms must match the draft window")
    else:
        pass
    if correction_uniforms.shape != (
        batch_size,
        draft_width + 1,
    ) or bonus_uniforms.shape != (batch_size, draft_width + 1):
        raise ValueError("correction and bonus uniforms must match the verify window")
    else:
        pass
    tensors = (
        draft_probabilities,
        target_probabilities,
        proposal_counts,
        active_rows,
        acceptance_uniforms,
        correction_uniforms,
        bonus_uniforms,
    )
    if any(tensor.device != draft_tokens.device for tensor in tensors):
        raise ValueError("speculative round tensors must share one device")
    else:
        pass
    token_ids = draft_tokens.long().unsqueeze(-1)
    draft_token_probabilities = draft_probabilities.gather(2, token_ids).squeeze(-1)
    target_token_probabilities = (
        target_probabilities[:, :draft_width].gather(2, token_ids).squeeze(-1)
    )
    safe_draft_probabilities = torch.where(
        draft_token_probabilities > 0,
        draft_token_probabilities,
        torch.ones_like(draft_token_probabilities),
    )
    acceptance_ratio = torch.minimum(
        torch.ones_like(target_token_probabilities),
        target_token_probabilities / safe_draft_probabilities,
    )
    valid_drafts = (
        torch.arange(draft_width, device=draft_tokens.device)[None, :]
        < proposal_counts[:, None]
    ) & active_rows[:, None]
    accepted = (acceptance_uniforms < acceptance_ratio) & valid_drafts
    rejected = valid_drafts & ~accepted
    has_rejection = rejected.any(dim=1)
    first_rejection = rejected.to(torch.int32).argmax(dim=1)
    accepted_draft_counts = torch.where(has_rejection, first_rejection, proposal_counts)
    accepted_draft_counts = torch.where(
        active_rows, accepted_draft_counts, torch.zeros_like(accepted_draft_counts)
    )
    selected_query = accepted_draft_counts.clamp(max=draft_width)
    batch_indices = torch.arange(batch_size, device=draft_tokens.device)
    target_selected = target_probabilities[batch_indices, selected_query]
    safe_draft_query = selected_query.clamp(max=draft_width - 1)
    draft_selected = draft_probabilities[batch_indices, safe_draft_query]
    draft_selected = torch.where(
        has_rejection[:, None], draft_selected, torch.zeros_like(draft_selected)
    )
    residual = torch.clamp(target_selected - draft_selected, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    use_residual = has_rejection[:, None] & (residual_mass > 0.0)
    selected_distribution = torch.where(use_residual, residual, target_selected)
    selected_mass = selected_distribution.sum(dim=-1, keepdim=True)
    selected_distribution = selected_distribution / selected_mass.clamp_min(
        torch.finfo(selected_distribution.dtype).tiny
    )
    correction_draws = correction_uniforms.gather(1, selected_query[:, None]).squeeze(1)
    bonus_draws = bonus_uniforms.gather(1, selected_query[:, None]).squeeze(1)
    final_uniforms = torch.where(has_rejection, correction_draws, bonus_draws)
    cumulative = selected_distribution.cumsum(dim=-1)
    final_token_indices = (cumulative <= final_uniforms[:, None]).sum(dim=-1)
    final_token_indices = final_token_indices.clamp(max=vocab_size - 1)
    token_vocab = torch.arange(vocab_size, device=draft_tokens.device)
    last_positive_token = torch.where(
        selected_distribution > 0,
        token_vocab[None, :],
        torch.zeros_like(token_vocab)[None, :],
    ).amax(dim=-1)
    final_token_indices = torch.where(
        active_rows, final_token_indices, torch.zeros_like(final_token_indices)
    )
    final_token_indices = torch.where(
        active_rows,
        torch.where(
            selected_distribution[batch_indices, final_token_indices] > 0,
            final_token_indices,
            last_positive_token,
        ),
        torch.zeros_like(final_token_indices),
    )
    committed_tokens = torch.zeros(
        (batch_size, draft_width + 1), device=draft_tokens.device, dtype=torch.long
    )
    draft_positions = torch.arange(draft_width, device=draft_tokens.device)[None, :]
    committed_tokens[:, :draft_width] = torch.where(
        draft_positions < accepted_draft_counts[:, None],
        draft_tokens.long(),
        torch.zeros_like(draft_tokens, dtype=torch.long),
    )
    committed_tokens.scatter_(
        1, accepted_draft_counts[:, None], final_token_indices[:, None]
    )
    committed_token_counts = torch.where(
        active_rows, accepted_draft_counts + 1, torch.zeros_like(accepted_draft_counts)
    )
    return SpeculativeAcceptanceBatch(
        committed_tokens=committed_tokens,
        accepted_draft_counts=accepted_draft_counts,
        committed_token_counts=committed_token_counts,
        rejected_rows=has_rejection & active_rows,
        bonus_rows=~has_rejection & active_rows,
    )

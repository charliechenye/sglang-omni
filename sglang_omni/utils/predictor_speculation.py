# SPDX-License-Identifier: Apache-2.0
"""CPU reference for CodePredictor speculative decoding.

This module is deliberately independent of CUDA and model classes. It defines the
sampling contract and the per-row state transitions used by the packed Predictor
path, so tests can exercise rejection and KV-prefix commitment without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class SpeculativeRowState:
    """Committed codes and the one token that the next Predictor round must consume."""

    committed_codes: list[int]
    pending_token: int
    preceding_hidden: torch.Tensor
    valid_kv_length: int
    accumulated_embedding: torch.Tensor
    finished: bool = False
    valid_proposal_count: int = 0
    valid_query_count: int = 0

    @property
    def committed_predictor_code_count(self) -> int:
        """Count committed codebooks after codebook 0's opening input."""
        return len(self.committed_codes) - 1


@dataclass(frozen=True)
class SpeculativeRoundDecision:
    """Tokens committed by one row in one speculative round."""

    committed_tokens: tuple[int, ...]
    accepted_draft_count: int
    proposal_count: int
    rejected: bool
    bonus_sampled: bool

    @property
    def valid_query_count(self) -> int:
        """The pending input plus each proposal occupies one verifier row."""
        return self.proposal_count + 1


@dataclass(frozen=True)
class SpeculativeFrameResult:
    """Final CPU reference output and counters for a batch of frames."""

    codes: tuple[tuple[int, ...], ...]
    summed_embeddings: torch.Tensor
    rounds: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    bonus_tokens: int
    rejections: int


@dataclass(frozen=True)
class SpeculativeCapabilityResolution:
    """Requested and effective verify width plus a stable fallback reason."""

    requested_width: int
    effective_width: int
    fallback_reason: str | None


def resolve_speculative_capability(
    requested_width: int,
    *,
    is_cuda: bool,
    is_bfloat16: bool,
    tensor_parallel_size: int,
    fused_predictor_available: bool,
    batch_size: int,
    packed_row_capacity: int,
    query_heads: int,
    kv_heads: int,
    attention_tile_capacity: int,
    vocab_size: int,
    graph_available: bool,
    sampling_supported: bool,
    has_explicit_seed: bool,
    has_greedy_rows: bool,
    deterministic_mode: bool,
) -> SpeculativeCapabilityResolution:
    """Resolve opt-in width before a frame mutates Predictor state."""
    if (
        isinstance(requested_width, bool)
        or not isinstance(requested_width, int)
        or requested_width not in range(0, 9)
    ):
        raise ValueError("code_predictor_spec_verify_size must be 0 or in [2, 8]")
    else:
        pass
    if requested_width == 1:
        raise ValueError(
            "code_predictor_spec_verify_size=1 is invalid; use 0 or [2, 8]"
        )
    else:
        pass
    if requested_width == 0:
        return SpeculativeCapabilityResolution(0, 0, None)
    else:
        pass
    if not is_cuda:
        reason = "unsupported-device-backend"
    elif not is_bfloat16:
        reason = "unsupported-dtype"
    elif tensor_parallel_size != 1:
        reason = "unsupported-tensor-parallel-size"
    elif not fused_predictor_available:
        reason = "fused-predictor-unavailable"
    elif batch_size <= 0 or requested_width * batch_size > packed_row_capacity:
        reason = "packed-row-capacity-exceeded"
    elif query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads != 0:
        reason = "invalid-grouped-query-head-shape"
    elif requested_width * (query_heads // kv_heads) > attention_tile_capacity:
        reason = "attention-tile-capacity-exceeded"
    elif vocab_size != 2048:
        reason = "unsupported-codebook-vocabulary"
    elif not graph_available:
        reason = "speculative-graph-unavailable"
    elif not sampling_supported:
        reason = "unsupported-sampling-configuration"
    elif has_explicit_seed:
        reason = "seeded-request-requires-native-decoding"
    elif has_greedy_rows:
        reason = "greedy-request-requires-native-decoding"
    elif deterministic_mode:
        reason = "deterministic-mode-requires-native-decoding"
    else:
        reason = None
    if reason is not None:
        return SpeculativeCapabilityResolution(requested_width, 0, reason)
    else:
        pass
    return SpeculativeCapabilityResolution(requested_width, requested_width, None)


def speculative_resources_requested(requested_width: int) -> bool:
    """Whether a nonzero configured width permits allocating speculative buffers."""
    if (
        isinstance(requested_width, bool)
        or not isinstance(requested_width, int)
        or requested_width not in range(0, 9)
    ):
        raise ValueError("code_predictor_spec_verify_size must be 0 or in [2, 8]")
    else:
        pass
    if requested_width == 1:
        raise ValueError(
            "code_predictor_spec_verify_size=1 is invalid; use 0 or [2, 8]"
        )
    else:
        pass
    return requested_width > 0


class DraftProvider(Protocol):
    def __call__(
        self,
        states: list[SpeculativeRowState],
        proposal_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class VerifyProvider(Protocol):
    def __call__(
        self,
        pending_tokens: torch.Tensor,
        padded_draft_tokens: torch.Tensor,
        valid_proposal_counts: torch.Tensor,
        valid_query_counts: torch.Tensor,
        active_row_mask: torch.Tensor,
        logical_kv_lengths: torch.Tensor,
        preceding_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class UniformProvider(Protocol):
    def __call__(
        self,
        round_index: int,
        proposal_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def probabilities_from_logits_cpu(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    ordered_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the FP64 reference distribution for one Qwen3-TTS codebook row.

    If ``ordered_token_ids`` is provided, it supplies the already-selected native
    candidate order, including backend-specific tie handling. Without it, top-k
    boundary ties use the lower token ID as a deterministic reference policy.
    Top-p follows SGLang's nucleus rule: remove a token only after the probability
    mass before it has reached top_p. Values of top_p equal to zero or one disable
    nucleus filtering.
    """
    if logits.device.type != "cpu" or logits.ndim != 1:
        raise ValueError("CPU reference logits must be a one-dimensional CPU tensor")
    else:
        pass
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    else:
        pass
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be non-negative")
    else:
        pass
    if not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be finite and in [0, 1]")
    else:
        pass
    values = logits.to(torch.float64) / temperature
    if torch.isnan(values).any() or torch.isposinf(values).any():
        raise ValueError("logits may not contain NaN or positive infinity")
    else:
        pass
    if ordered_token_ids is None:
        ordered_ids = [
            index
            for index, value in enumerate(values.tolist())
            if value != -float("inf")
        ]
        ordered_ids.sort(key=lambda index: (-float(values[index]), index))
    elif (
        ordered_token_ids.device.type != "cpu"
        or ordered_token_ids.ndim != 1
        or ordered_token_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "ordered_token_ids must be a one-dimensional CPU integer tensor"
        )
    else:
        ordered_ids = [int(token) for token in ordered_token_ids]
        if (
            not ordered_ids
            or any(token < 0 or token >= values.numel() for token in ordered_ids)
            or len(set(ordered_ids)) != len(ordered_ids)
        ):
            raise ValueError("ordered_token_ids contains invalid or repeated token IDs")
        else:
            pass
    if not ordered_ids:
        raise ValueError("logits have no finite token")
    else:
        pass
    if not any(float(values[token]) != -float("inf") for token in ordered_ids):
        raise ValueError("logits have no finite token")
    else:
        pass
    if ordered_token_ids is not None:
        minimum_candidates = top_k if 0 < top_k < values.numel() else values.numel()
        if len(ordered_ids) < minimum_candidates:
            raise ValueError(
                "ordered_token_ids does not cover the native candidate set"
            )
        else:
            pass
    else:
        pass
    if 0 < top_k < len(ordered_ids):
        ordered_ids = ordered_ids[:top_k]
    else:
        pass
    selected = values[torch.tensor(ordered_ids, dtype=torch.long)]
    selected_probabilities = torch.softmax(selected, dim=0)
    if 0.0 < top_p < 1.0:
        previous_mass = (
            torch.cumsum(selected_probabilities, dim=0) - selected_probabilities
        )
        keep = (previous_mass < top_p) | (torch.arange(len(ordered_ids)) == 0)
        selected_probabilities = selected_probabilities * keep.to(torch.float64)
    else:
        pass
    mass = selected_probabilities.sum()
    if not torch.isfinite(mass) or mass <= 0.0:
        raise ValueError("sampling filters produced a zero-mass distribution")
    else:
        pass
    selected_probabilities = selected_probabilities / mass
    result = torch.zeros_like(values)
    result[torch.tensor(ordered_ids, dtype=torch.long)] = selected_probabilities
    return result


def validate_probability_distribution(probabilities: torch.Tensor) -> torch.Tensor:
    """Return a normalized FP64 view after checking a finite nonnegative distribution."""
    if probabilities.ndim != 1 or probabilities.device.type != "cpu":
        raise ValueError("probabilities must be a one-dimensional CPU tensor")
    else:
        pass
    values = probabilities.to(torch.float64)
    if not torch.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    else:
        pass
    mass = values.sum()
    if not torch.isfinite(mass) or mass <= 0.0:
        raise ValueError("probabilities must have positive finite mass")
    else:
        pass
    return values / mass


def sample_categorical_cpu(probabilities: torch.Tensor, uniform: float) -> int:
    """Draw with inverse CDF using a uniform value in [0, 1)."""
    values = validate_probability_distribution(probabilities)
    if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("uniform must be finite and in [0, 1)")
    else:
        pass
    cumulative = torch.cumsum(values, dim=0)
    token = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(uniform, dtype=torch.float64),
            right=True,
        )
    )
    if token >= values.numel():
        token = int(torch.nonzero(values > 0.0, as_tuple=False)[-1, 0])
    else:
        pass
    if values[token] <= 0.0:
        positive_ids = torch.nonzero(values > 0.0, as_tuple=False).flatten()
        token = int(
            positive_ids[
                torch.searchsorted(
                    cumulative[positive_ids],
                    torch.tensor(uniform, dtype=torch.float64),
                    right=True,
                )
            ]
        )
    else:
        pass
    return token


def acceptance_probability(
    target_probability: float, draft_probability: float
) -> float:
    """Compute min(1, p(x)/q(x)), rejecting an invalid proposal distribution."""
    if not math.isfinite(target_probability) or target_probability < 0.0:
        raise ValueError("target probability must be finite and nonnegative")
    else:
        pass
    if not math.isfinite(draft_probability) or draft_probability <= 0.0:
        raise ValueError("draft probability for a proposed token must be positive")
    else:
        pass
    return min(1.0, target_probability / draft_probability)


def residual_distribution_cpu(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Normalize max(p-q, 0), falling back to p for a numerically empty residual."""
    target = validate_probability_distribution(target_probabilities)
    draft = validate_probability_distribution(draft_probabilities)
    if target.shape != draft.shape:
        raise ValueError("target and draft distributions must have equal shape")
    else:
        pass
    residual = torch.clamp(target - draft, min=0.0)
    mass = residual.sum()
    if not torch.isfinite(mass) or mass <= torch.finfo(torch.float64).eps:
        return target
    else:
        pass
    return residual / mass


def speculative_acceptance_cpu(
    draft_tokens: torch.Tensor,
    draft_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    acceptance_uniforms: torch.Tensor,
    sampling_uniforms: torch.Tensor,
) -> SpeculativeRoundDecision:
    """Apply first-rejection acceptance, residual correction, or trailing bonus."""
    if draft_tokens.device.type != "cpu" or draft_tokens.ndim != 1:
        raise ValueError("draft_tokens must be a one-dimensional CPU tensor")
    else:
        pass
    proposal_count = int(draft_tokens.numel())
    if draft_probabilities.ndim != 2 or draft_probabilities.shape[0] != proposal_count:
        raise ValueError("draft probability count does not match draft tokens")
    else:
        pass
    if (
        draft_probabilities.ndim != 2
        or target_probabilities.ndim != 2
        or target_probabilities.shape[0] != proposal_count + 1
        or target_probabilities.shape[1] != draft_probabilities.shape[1]
    ):
        raise ValueError("target distributions must include one trailing bonus row")
    else:
        pass
    if acceptance_uniforms.shape != (proposal_count,) or sampling_uniforms.shape != (
        proposal_count + 1,
    ):
        raise ValueError("uniform arrays do not match the proposal count")
    else:
        pass
    cpu_inputs = (
        draft_probabilities,
        target_probabilities,
        acceptance_uniforms,
        sampling_uniforms,
    )
    if any(tensor.device.type != "cpu" for tensor in cpu_inputs):
        raise ValueError("the CPU reference accepts CPU tensors only")
    else:
        pass
    if draft_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("draft tokens must use an integer dtype")
    else:
        pass
    committed: list[int] = []
    for proposal_index in range(proposal_count):
        draft = validate_probability_distribution(draft_probabilities[proposal_index])
        target = validate_probability_distribution(target_probabilities[proposal_index])
        token = int(draft_tokens[proposal_index])
        if token < 0 or token >= draft.numel():
            raise ValueError("draft token is outside the vocabulary")
        else:
            pass
        q_token = float(draft[token])
        p_token = float(target[token])
        accept_probability = acceptance_probability(p_token, q_token)
        uniform = float(acceptance_uniforms[proposal_index])
        if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
            raise ValueError("acceptance uniforms must be finite and in [0, 1)")
        else:
            pass
        if uniform < accept_probability:
            committed.append(token)
        else:
            residual = residual_distribution_cpu(target, draft)
            correction = sample_categorical_cpu(
                residual, float(sampling_uniforms[proposal_index])
            )
            committed.append(correction)
            return SpeculativeRoundDecision(
                committed_tokens=tuple(committed),
                accepted_draft_count=proposal_index,
                proposal_count=proposal_count,
                rejected=True,
                bonus_sampled=False,
            )
    else:
        pass
    bonus_distribution = validate_probability_distribution(
        target_probabilities[proposal_count]
    )
    bonus = sample_categorical_cpu(
        bonus_distribution, float(sampling_uniforms[proposal_count])
    )
    committed.append(bonus)
    return SpeculativeRoundDecision(
        committed_tokens=tuple(committed),
        accepted_draft_count=proposal_count,
        proposal_count=proposal_count,
        rejected=False,
        bonus_sampled=True,
    )


def run_speculative_frame_reference(
    *,
    initial_codes: torch.Tensor,
    initial_hidden_before_pending: torch.Tensor,
    initial_embedding_sum: torch.Tensor,
    codec_embedding_tables: torch.Tensor,
    num_code_groups: int,
    verify_width: int,
    draft_provider: DraftProvider,
    verify_provider: VerifyProvider,
    uniform_provider: UniformProvider,
) -> SpeculativeFrameResult:
    """Run a complete mocked frame using the production round state transitions.

    The initial codes contain codebook 0 and the natively sampled codebook 1;
    ``initial_embedding_sum`` already includes both embeddings. ``verify_provider``
    receives pending tokens, padded drafts, proposal and query counts, the active-row
    mask, valid KV lengths, and the hidden state preceding each pending token.
    Its target probabilities and verified hidden rows have packed shape
    ``[batch, verify_width, ...]``. The cache length advances by the consumed
    pending token plus accepted drafts; correction and bonus tokens stay pending.
    """
    tensors = (
        initial_codes,
        initial_hidden_before_pending,
        initial_embedding_sum,
        codec_embedding_tables,
    )
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("the state-machine reference runs on CPU only")
    else:
        pass
    if initial_codes.ndim != 2 or initial_codes.shape[1] != 2:
        raise ValueError("initial_codes must contain codebook 0 and codebook 1")
    else:
        pass
    batch_size = int(initial_codes.shape[0])
    if batch_size <= 0:
        raise ValueError("the speculative frame must contain at least one row")
    else:
        pass
    if (
        initial_hidden_before_pending.ndim != 2
        or initial_hidden_before_pending.shape[0] != batch_size
    ):
        raise ValueError("initial hidden-state shape does not match the batch")
    else:
        pass
    if initial_embedding_sum.ndim != 2 or initial_embedding_sum.shape[0] != batch_size:
        raise ValueError("initial embedding-sum shape does not match the batch")
    else:
        pass
    if (
        isinstance(num_code_groups, bool)
        or not isinstance(num_code_groups, int)
        or isinstance(verify_width, bool)
        or not isinstance(verify_width, int)
        or num_code_groups < 2
        or verify_width < 2
        or verify_width > 8
    ):
        raise ValueError(
            "num_code_groups must be at least 2 and verify_width in [2, 8]"
        )
    else:
        pass
    if initial_codes.dtype not in (torch.int32, torch.int64):
        raise ValueError("initial codes must use an integer dtype")
    else:
        pass
    if (
        codec_embedding_tables.ndim != 3
        or codec_embedding_tables.shape[0] != num_code_groups - 1
    ):
        raise ValueError("codec tables must have one table for each nonzero codebook")
    else:
        pass
    if codec_embedding_tables.shape[1] <= 0:
        raise ValueError("codec tables must contain at least one vocabulary row")
    else:
        pass
    if codec_embedding_tables.shape[2] != initial_embedding_sum.shape[1]:
        raise ValueError("codec embedding width does not match the accumulated output")
    else:
        pass
    vocab_size = int(codec_embedding_tables.shape[1])
    states = [
        SpeculativeRowState(
            committed_codes=[int(initial_codes[row, 0]), int(initial_codes[row, 1])],
            pending_token=int(initial_codes[row, 1]),
            preceding_hidden=initial_hidden_before_pending[row].clone(),
            valid_kv_length=2,
            accumulated_embedding=initial_embedding_sum[row].clone(),
            finished=num_code_groups == 2,
        )
        for row in range(batch_size)
    ]
    for state in states:
        if state.committed_codes[0] < 0 or state.committed_codes[0] >= vocab_size:
            raise ValueError("initial codebook-0 token is outside the vocabulary")
        else:
            pass
        if state.pending_token < 0 or state.pending_token >= vocab_size:
            raise ValueError("initial pending token is outside the vocabulary")
        else:
            pass
    summed_embeddings = initial_embedding_sum.clone()
    rounds = 0
    proposed_count = 0
    accepted_count = 0
    bonus_count = 0
    rejection_count = 0
    while any(not state.finished for state in states):
        proposal_counts = torch.tensor(
            [
                (
                    min(
                        verify_width - 1,
                        num_code_groups - len(state.committed_codes) - 1,
                    )
                    if not state.finished
                    else 0
                )
                for state in states
            ],
            dtype=torch.long,
        )
        active_rows = torch.tensor(
            [not state.finished for state in states], dtype=torch.bool
        )
        valid_query_counts = torch.where(
            active_rows, proposal_counts + 1, torch.zeros_like(proposal_counts)
        )
        for row, state in enumerate(states):
            state.valid_proposal_count = int(proposal_counts[row])
            state.valid_query_count = int(valid_query_counts[row])
        else:
            pass
        draft_probabilities, draft_tokens = draft_provider(states, proposal_counts)
        expected_draft_shape = (batch_size, verify_width - 1, vocab_size)
        if draft_probabilities.shape != expected_draft_shape or draft_tokens.shape != (
            batch_size,
            verify_width - 1,
        ):
            raise ValueError("draft provider returned an invalid packed shape")
        else:
            pass
        if (
            draft_probabilities.device.type != "cpu"
            or draft_tokens.device.type != "cpu"
        ):
            raise ValueError("draft provider must return CPU tensors")
        else:
            pass
        target_probabilities, verified_hidden = verify_provider(
            torch.tensor([state.pending_token for state in states], dtype=torch.long),
            draft_tokens,
            proposal_counts,
            valid_query_counts,
            active_rows,
            torch.tensor([state.valid_kv_length for state in states], dtype=torch.long),
            torch.stack([state.preceding_hidden for state in states]),
        )
        expected_target_shape = (batch_size, verify_width, vocab_size)
        if target_probabilities.shape != expected_target_shape:
            raise ValueError("verify provider returned an invalid probability shape")
        else:
            pass
        if (
            target_probabilities.device.type != "cpu"
            or verified_hidden.device.type != "cpu"
        ):
            raise ValueError("verify provider must return CPU tensors")
        else:
            pass
        if (
            verified_hidden.ndim != 3
            or verified_hidden.shape[:2] != (batch_size, verify_width)
            or verified_hidden.shape[2] != initial_hidden_before_pending.shape[1]
        ):
            raise ValueError("verify provider returned an invalid hidden-state shape")
        else:
            pass
        acceptance_uniforms, sampling_uniforms = uniform_provider(
            rounds, proposal_counts
        )
        if acceptance_uniforms.shape != (
            batch_size,
            verify_width - 1,
        ) or sampling_uniforms.shape != (batch_size, verify_width):
            raise ValueError("uniform provider returned invalid packed shapes")
        else:
            pass
        if (
            acceptance_uniforms.device.type != "cpu"
            or sampling_uniforms.device.type != "cpu"
        ):
            raise ValueError("uniform provider must return CPU tensors")
        else:
            pass
        rounds += 1
        for row, state in enumerate(states):
            if state.finished:
                continue
            else:
                pass
            proposal_count = int(proposal_counts[row])
            decision = speculative_acceptance_cpu(
                draft_tokens[row, :proposal_count],
                draft_probabilities[row, :proposal_count],
                target_probabilities[row, : proposal_count + 1],
                acceptance_uniforms[row, :proposal_count],
                sampling_uniforms[row, : proposal_count + 1],
            )
            previous_code_count = len(state.committed_codes)
            accepted_count += decision.accepted_draft_count
            proposed_count += proposal_count
            bonus_count += int(decision.bonus_sampled)
            rejection_count += int(decision.rejected)
            for token in decision.committed_tokens:
                code_index = len(state.committed_codes)
                if code_index >= num_code_groups:
                    raise RuntimeError("speculative decoding emitted beyond the frame")
                else:
                    pass
                if code_index == 0:
                    raise RuntimeError("codebook 0 is committed during initialization")
                else:
                    pass
                if token < 0 or token >= vocab_size:
                    raise RuntimeError("speculative sampler emitted an invalid token")
                else:
                    pass
                state.committed_codes.append(token)
                state.accumulated_embedding.add_(
                    codec_embedding_tables[code_index - 1, token]
                )
            else:
                pass
            consumed_rows = decision.accepted_draft_count
            state.pending_token = decision.committed_tokens[-1]
            state.preceding_hidden = verified_hidden[row, consumed_rows].clone()
            state.valid_kv_length += consumed_rows + 1
            state.finished = len(state.committed_codes) == num_code_groups
            if state.valid_kv_length != state.committed_predictor_code_count + 1:
                raise RuntimeError(
                    "logical KV prefix and committed-code count diverged"
                )
            else:
                pass
            if state.pending_token != state.committed_codes[-1]:
                raise RuntimeError("pending token is not the final committed code")
            else:
                pass
            if len(state.committed_codes) != previous_code_count + len(
                decision.committed_tokens
            ):
                raise RuntimeError(
                    "the committed code count does not match the decision"
                )
            else:
                pass
        else:
            pass
        for row, state in enumerate(states):
            summed_embeddings[row].copy_(state.accumulated_embedding)
        else:
            pass
    else:
        pass
    return SpeculativeFrameResult(
        codes=tuple(tuple(state.committed_codes) for state in states),
        summed_embeddings=summed_embeddings,
        rounds=rounds,
        proposed_draft_tokens=proposed_count,
        accepted_draft_tokens=accepted_count,
        bonus_tokens=bonus_count,
        rejections=rejection_count,
    )

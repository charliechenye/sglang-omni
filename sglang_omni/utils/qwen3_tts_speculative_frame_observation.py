# SPDX-License-Identifier: Apache-2.0
"""Speculative round probes and native-context correctness checks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Protocol

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    build_speculative_probability_candidates,
    fused_raw_logit_block_k,
)
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.models.qwen3_tts.speculative_workspace import (
    PredictorSpeculativeRoundWorkspace,
    compact_candidate_view,
)
from sglang_omni.utils.qwen3_tts_speculative_frozen import FrozenPredictorFrame
from sglang_omni.utils.qwen3_tts_speculative_native_replay import (
    NativePredictorReference,
)


class VerifierGraphOutput(Protocol):
    target_logits: torch.Tensor | None


class RoundResolverOperation(Protocol):
    def __call__(
        self, batch_size: int, requested_width: int
    ) -> tuple[int, VerifierGraphOutput | None]: ...


@dataclass(frozen=True, kw_only=True)
class SpeculativeRoundObservation:
    verify_width: int
    logits: torch.Tensor
    probabilities: torch.Tensor
    token_ids: torch.Tensor
    valid_queries: torch.Tensor
    head_indices: torch.Tensor
    draft_tokens: torch.Tensor
    committed_tokens: torch.Tensor
    committed_count: int
    accepted_count: int
    starting_depth: int
    ending_depth: int
    preceding_hidden: torch.Tensor
    logical_kv_length: int
    keys: torch.Tensor
    values: torch.Tensor
    output_codes: torch.Tensor


def compare_probability_row(
    talker: Qwen3TTSTalker,
    frame: FrozenPredictorFrame,
    observation: SpeculativeRoundObservation,
    *,
    batch_row: int,
    query_offset: int,
    head_index: int,
) -> None:
    row_index = batch_row * observation.verify_width + query_offset
    temperature = talker.sub_temperature_tensor[batch_row : batch_row + 1]
    top_k = talker.sub_top_k_tensor[batch_row : batch_row + 1]
    top_p = talker.sub_top_p_tensor[batch_row : batch_row + 1]
    active = torch.ones(1, device=talker.predictor_device, dtype=torch.bool)
    expected = build_speculative_probability_candidates(
        frame.logits[head_index].reshape(1, -1).to(device=talker.predictor_device),
        temperature,
        top_k,
        top_p,
        active,
        max_top_k=int(talker.sub_sampled_max_top_k),
        has_top_p=bool(talker.sub_sampled_has_top_p),
        has_unbounded_top_k=False,
    )
    if expected is None:
        raise RuntimeError("native probability reference rejected a validated frame")
    else:
        pass
    expected_probabilities, expected_token_ids = expected
    actual_probabilities = observation.probabilities[row_index]
    actual_token_ids = observation.token_ids[row_index]
    if not torch.equal(actual_probabilities > 0.0, expected_probabilities[0] > 0.0):
        raise AssertionError(
            "speculative target probability support differs from native"
        )
    else:
        pass
    positive = expected_probabilities[0] > 0.0
    if not torch.equal(actual_token_ids[positive], expected_token_ids[0][positive]):
        raise AssertionError("speculative target token IDs differ from native support")
    else:
        pass
    torch.testing.assert_close(
        actual_probabilities, expected_probabilities[0], atol=1e-6, rtol=1e-6
    )


def validate_speculative_observations(
    talker: Qwen3TTSTalker,
    frame: FrozenPredictorFrame,
    reference: NativePredictorReference,
    observations: list[SpeculativeRoundObservation],
) -> tuple[int, int, int, int]:
    target_logit_checks = 0
    target_probability_checks = 0
    carried_hidden_checks = 0
    kv_prefix_checks = 0
    for observation in observations:
        for query_offset in range(observation.verify_width):
            if not bool(observation.valid_queries[0, query_offset]):
                continue
            else:
                pass
            start_depth = observation.starting_depth
            head_index = int(observation.head_indices[0, query_offset].item())
            context_matches = torch.equal(
                observation.output_codes[0, 1 : start_depth + 1].cpu(),
                reference.codes[1 : start_depth + 1].cpu(),
            )
            if context_matches and query_offset > 0:
                context_matches = torch.equal(
                    observation.draft_tokens[0, :query_offset].cpu(),
                    reference.codes[
                        start_depth + 1 : start_depth + query_offset + 1
                    ].cpu(),
                )
            else:
                pass
            if context_matches:
                torch.testing.assert_close(
                    observation.logits[0, query_offset].cpu(),
                    reference.logits[head_index].cpu(),
                    atol=2e-2,
                    rtol=2e-2,
                )
                target_logit_checks += 1
                compare_probability_row(
                    talker,
                    frame,
                    observation,
                    batch_row=0,
                    query_offset=query_offset,
                    head_index=head_index,
                )
                target_probability_checks += 1
            else:
                pass

        accepted_context_matches = torch.equal(
            observation.output_codes[0, 1 : observation.starting_depth + 1].cpu(),
            reference.codes[1 : observation.starting_depth + 1].cpu(),
        )
        if observation.accepted_count > 0:
            accepted_context_matches = accepted_context_matches and torch.equal(
                observation.draft_tokens[0, : observation.accepted_count].cpu(),
                reference.codes[
                    observation.starting_depth
                    + 1 : observation.starting_depth
                    + observation.accepted_count
                    + 1
                ].cpu(),
            )
        else:
            pass
        hidden_index = observation.ending_depth - 1
        if accepted_context_matches and hidden_index < reference.hidden.shape[0]:
            torch.testing.assert_close(
                observation.preceding_hidden.cpu(),
                reference.hidden[hidden_index].cpu(),
                atol=2e-2,
                rtol=2e-2,
            )
            carried_hidden_checks += 1
            expected_length = observation.logical_kv_length
            expected_keys = reference.keys_by_length.get(expected_length)
            expected_values = reference.values_by_length.get(expected_length)
            if expected_keys is None or expected_values is None:
                raise AssertionError(
                    "native replay lacks the committed valid KV prefix"
                )
            else:
                pass
            torch.testing.assert_close(
                observation.keys, expected_keys, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                observation.values, expected_values, atol=2e-2, rtol=2e-2
            )
            kv_prefix_checks += 1
        else:
            pass
    return (
        target_logit_checks,
        target_probability_checks,
        carried_hidden_checks,
        kv_prefix_checks,
    )


@contextmanager
def observe_speculative_rounds(
    talker: Qwen3TTSTalker,
) -> Generator[list[SpeculativeRoundObservation], None, None]:
    observations: list[SpeculativeRoundObservation] = []
    original_resolver: RoundResolverOperation = (
        talker.resolve_code_predictor_speculative_round_path
    )
    original_range = talker.code_predictor_spec_nvtx_range
    paths: list[tuple[int, VerifierGraphOutput | None]] = []

    def record_resolved_path(
        batch_size: int, requested_width: int
    ) -> tuple[int, VerifierGraphOutput | None]:
        path = original_resolver(batch_size, requested_width)
        paths.append(path)
        return path

    @contextmanager
    def record_range(label: str) -> Generator[None, None, None]:
        with original_range(label):
            yield
        if label == "CPU orchestration":
            if len(paths) != len(observations) + 1:
                raise RuntimeError("speculative round observation lost its width")
            else:
                pass
            verify_width, verify_graph = paths[-1]
            workspace = talker.code_predictor_spec_active_workspace
            if workspace is None:
                raise RuntimeError("speculative frame workspace was not retained")
            else:
                pass
            round_workspace: PredictorSpeculativeRoundWorkspace = workspace.rounds[
                verify_width
            ]
            metadata = talker.code_predictor_spec_metadata_host[0].tolist()
            committed_count = int(metadata[3])
            accepted_count = int(metadata[0])
            ending_depth = int(workspace.committed_depths[0].item())
            starting_depth = ending_depth - committed_count
            if ending_depth != starting_depth + committed_count:
                raise AssertionError(
                    "committed Predictor depth did not advance by its count"
                )
            elif not torch.equal(
                talker.output_codes[0, starting_depth + 1 : ending_depth + 1],
                round_workspace.acceptance_outputs.committed_tokens[
                    0, :committed_count
                ],
            ):
                raise AssertionError(
                    "committed Predictor prefix differs from acceptance"
                )
            elif int(workspace.logical_kv_lengths[0].item()) != ending_depth + 1:
                raise AssertionError("logical KV length differs from committed depth")
            else:
                pass
            candidate_width = fused_raw_logit_block_k(int(talker.sub_sampled_max_top_k))
            if candidate_width is None:
                raise RuntimeError("frozen-frame candidate width is unavailable")
            else:
                pass
            if verify_graph is None:
                target_logits = round_workspace.target_logits[:verify_width]
            elif verify_graph.target_logits is None:
                raise RuntimeError("verifier graph omitted target logits")
            else:
                target_logits = verify_graph.target_logits[:verify_width]
            probabilities = compact_candidate_view(
                round_workspace.target_probabilities,
                verify_width,
                candidate_width,
            )
            token_ids = compact_candidate_view(
                round_workspace.target_token_ids,
                verify_width,
                candidate_width,
            )
            valid_length = int(workspace.logical_kv_lengths[0].item())
            observations.append(
                SpeculativeRoundObservation(
                    verify_width=verify_width,
                    logits=target_logits.detach().view(1, verify_width, -1)[0].clone(),
                    probabilities=probabilities.detach().clone(),
                    token_ids=token_ids.detach().clone(),
                    valid_queries=round_workspace.verifier_inputs.valid_queries[:1]
                    .detach()
                    .clone(),
                    head_indices=round_workspace.verifier_inputs.target_head_indices[:1]
                    .detach()
                    .clone(),
                    draft_tokens=round_workspace.proposal_tokens[:1].detach().clone(),
                    committed_tokens=(
                        round_workspace.acceptance_outputs.committed_tokens[:1]
                        .detach()
                        .clone()
                    ),
                    committed_count=committed_count,
                    accepted_count=accepted_count,
                    starting_depth=starting_depth,
                    ending_depth=ending_depth,
                    preceding_hidden=workspace.preceding_hidden[0].detach().clone(),
                    logical_kv_length=valid_length,
                    keys=talker.predictor_k_cache[:, :1, :valid_length]
                    .detach()
                    .clone(),
                    values=talker.predictor_v_cache[:, :1, :valid_length]
                    .detach()
                    .clone(),
                    output_codes=talker.output_codes[:1].detach().clone(),
                )
            )
        else:
            pass

    talker.resolve_code_predictor_speculative_round_path = record_resolved_path
    talker.code_predictor_spec_nvtx_range = record_range
    try:
        yield observations
    finally:
        talker.resolve_code_predictor_speculative_round_path = original_resolver
        talker.code_predictor_spec_nvtx_range = original_range

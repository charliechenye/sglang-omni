# SPDX-License-Identifier: Apache-2.0
"""CPU checks for request-isolated speculative random streams."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_tts.speculative_rng import (
    derive_speculative_request_seed,
    speculative_uniforms,
)


def test_speculative_uniforms_follow_request_identity_after_row_reordering() -> None:
    first_seed = derive_speculative_request_seed(104, "request-first")
    second_seed = derive_speculative_request_seed(205, "request-second")
    seeds = torch.tensor([first_seed, second_seed], dtype=torch.long)
    positions = torch.tensor([31, 7], dtype=torch.long)
    first = speculative_uniforms(seeds, positions, 2, 16, 4, "proposal")
    reordered = speculative_uniforms(
        seeds.flip(0), positions.flip(0), 2, 16, 4, "proposal"
    )

    torch.testing.assert_close(reordered, first.flip(0), atol=0.0, rtol=0.0)


def test_speculative_draw_namespace_separates_purpose_round_and_depth() -> None:
    seeds = torch.tensor(
        [derive_speculative_request_seed(912, "request-one")], dtype=torch.long
    )
    positions = torch.tensor([12], dtype=torch.long)
    proposal = speculative_uniforms(seeds, positions, 0, 16, 4, "proposal")
    acceptance = speculative_uniforms(seeds, positions, 0, 16, 4, "acceptance")
    next_round = speculative_uniforms(seeds, positions, 1, 16, 4, "proposal")

    assert not torch.equal(proposal, acceptance)
    assert not torch.equal(proposal, next_round)
    assert torch.all((proposal > 0.0) & (proposal < 1.0))

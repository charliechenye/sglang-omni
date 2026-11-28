# SPDX-License-Identifier: Apache-2.0
"""CPU checks for stable speculative random streams."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_tts.speculative_rng import speculative_uniforms


def test_speculative_uniforms_depend_on_seed_and_semantic_position() -> None:
    seeds = torch.tensor([104, 205], dtype=torch.long)
    positions = torch.tensor([31, 7], dtype=torch.long)
    first = speculative_uniforms(seeds, positions, 2, 16, 4, "proposal")
    repeated = speculative_uniforms(seeds, positions, 2, 16, 4, "proposal")
    reordered = speculative_uniforms(
        seeds.flip(0), positions.flip(0), 2, 16, 4, "proposal"
    )
    other_position = speculative_uniforms(seeds, positions + 1, 2, 16, 4, "proposal")

    torch.testing.assert_close(repeated, first, atol=0.0, rtol=0.0)
    torch.testing.assert_close(reordered, first.flip(0), atol=0.0, rtol=0.0)
    assert not torch.equal(first, other_position)
    assert torch.all((first > 0.0) & (first < 1.0))


def test_speculative_draw_namespace_separates_purpose_and_round() -> None:
    seeds = torch.tensor([912], dtype=torch.long)
    positions = torch.tensor([12], dtype=torch.long)
    proposal = speculative_uniforms(seeds, positions, 0, 16, 4, "proposal")
    acceptance = speculative_uniforms(seeds, positions, 0, 16, 4, "acceptance")
    next_round = speculative_uniforms(seeds, positions, 1, 16, 4, "proposal")

    assert not torch.equal(proposal, acceptance)
    assert not torch.equal(proposal, next_round)
    assert torch.all((proposal > 0.0) & (proposal < 1.0))

# SPDX-License-Identifier: Apache-2.0
"""CPU-only P6 dynamic-depth semantics and opt-in runtime guards.

The H200 CUDA condition test is intentionally in the standalone P6 script;
running these CPU tests is not a substitute for CUDA Graph capture/replay.
"""
from __future__ import annotations

import pytest

from experiments.qwen3_tts_cp_spec_k2.p6_device_conditional_gate import (
    decision_masks,
    reference,
)


def test_all_reject_and_all_accept_have_correct_14_depth_coverage() -> None:
    masks = decision_masks()
    a = reference(masks["reject_all"])
    b = reference(masks["accept_all"])
    assert a == {"depth": 15, "rounds": 14, "accepted": 0, "bonus": 0}
    assert b == {"depth": 15, "rounds": 7, "accepted": 7, "bonus": 7}


def test_mixed_masks_are_bounded_and_preserve_accounting() -> None:
    masks = decision_masks()
    assert len(masks) == 5
    for name, mask in masks.items():
        result = reference(mask)
        assert result["depth"] == 15, name
        assert 7 <= result["rounds"] <= 14
        assert result["rounds"] + result["bonus"] == 14
        assert 0 <= result["accepted"] <= result["rounds"]


def test_invalid_decisions_fail_closed() -> None:
    with pytest.raises(ValueError, match="16 Boolean integers"):
        reference([0] * 15)
    with pytest.raises(ValueError, match="16 Boolean integers"):
        reference([0] * 15 + [2])
    with pytest.raises(ValueError, match="16 Boolean integers"):
        reference([-1] + [0] * 15)


def test_first_depth_is_never_a_proposal() -> None:
    decisions = [0] * 16
    decisions[1] = 1  # First residual code is already committed.
    assert reference(decisions) == reference([0] * 16)

# SPDX-License-Identifier: Apache-2.0
"""Experimental WholeHistory-only Graph tier override (no GPU required)."""
from __future__ import annotations

import pytest

from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.fun_cosyvoice3 import stages


def test_whole_history_override_factory_config_round_trip() -> None:
    parsed = FactoryArgs(flow_whole_history_tier_frames="4,8,12,16")
    assert parsed.flow_whole_history_tier_frames == "4,8,12,16"
    assert parsed.model_dump(exclude_none=True)["flow_whole_history_tier_frames"] == (
        "4,8,12,16"
    )


def test_whole_history_tiers_default_is_exact_source_ladder() -> None:
    full = stages.generate_prefill_cuda_graph_batch_sizes(8000)
    assert len(full) == 57
    assert stages.resolve_whole_history_graph_tiers(full, None) == tuple(full)


def test_whole_history_tier_subset_and_guard() -> None:
    # All previously-covered source geometries remain within 1.5x capacity.
    source = (4, 8, 12, 16, 24, 32)
    assert stages.resolve_whole_history_graph_tiers(
        source, "4,8,12,16,24,32"
    ) == source
    assert stages.resolve_whole_history_graph_tiers(
        source, "4,8,12,16,32"
    ) == (4, 8, 12, 16, 32)


@pytest.mark.parametrize(
    "override_csv,match",
    [
        ("", "positive comma-separated"),
        ("4,,8,12,16", "positive comma-separated"),
        ("4,8,8,12,16", "sorted unique"),
        ("4,8,16,12", "sorted unique"),
        ("4,8,12", "source maximum"),
        ("4,8,12,16,20,32", "source tiers"),
        (" 4,8,12,16,24,32", "positive comma-separated"),
        ("4,8,12,32", "1.5x"),
    ],
)
def test_whole_history_tier_override_rejects_unsafe(
    override_csv: str, match: str
) -> None:
    source = (4, 8, 12, 16, 24, 32)
    with pytest.raises(ValueError, match=match):
        stages.resolve_whole_history_graph_tiers(source, override_csv)


def test_whole_history_tier_override_rejects_disabled_graph_before_model_load() -> None:
    with pytest.raises(ValueError, match="override requires"):
        stages.create_vocoder_executor(
            "unused-model",
            flow_prefix_cache_gb=0.0,
            enable_flow_prefix_cuda_graph=False,
            enable_flow_whole_history_cuda_graph=False,
            enable_hift_cuda_graph=False,
            enable_dit_torch_compile=False,
            flow_whole_history_tier_frames="4,8,12,16",
        )


def test_whole_history_override_rejected_before_any_model_loading() -> None:
    with pytest.raises(ValueError, match="source maximum"):
        stages.create_vocoder_executor(
            "unused-model",
            flow_prefix_cache_gb=0.0,
            enable_flow_prefix_cuda_graph=False,
            enable_flow_whole_history_cuda_graph=True,
            enable_hift_cuda_graph=False,
            enable_dit_torch_compile=False,
            flow_whole_history_tier_frames="4,8,12,16",
        )

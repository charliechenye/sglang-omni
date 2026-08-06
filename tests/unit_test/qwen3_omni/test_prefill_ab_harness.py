# SPDX-License-Identifier: Apache-2.0
"""Independent parity helpers for the Qwen3-Omni prefill A/B/C arms."""

from __future__ import annotations

import torch

from benchmarks.eval.benchmark_qwen3_omni_prefill_ab import _arm_config


def _oracle_text_audio_embeddings(
    input_ids: torch.Tensor,
    embed_tokens: torch.nn.Module,
    audio_positions_cpu: torch.Tensor,
    audio_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Build the expected text/audio rows without calling production helpers."""
    vocab_size = int(embed_tokens.num_embeddings)
    output = embed_tokens(input_ids.clamp(min=0, max=vocab_size - 1)).detach().clone()
    if audio_positions_cpu.numel():
        positions = audio_positions_cpu.to(device=output.device)
        output.index_copy_(0, positions, audio_embeddings.to(output))
    return output


def test_prefill_oracle_handles_text_and_disjoint_audio_spans() -> None:
    torch.manual_seed(0)
    embed_tokens = torch.nn.Embedding(32, 4)
    input_ids = torch.tensor([1, 31, 2, 31, 3], dtype=torch.long)
    audio = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    expected = _oracle_text_audio_embeddings(
        input_ids,
        embed_tokens,
        torch.tensor([1, 3], dtype=torch.long),
        audio,
    )

    assert torch.equal(expected[[1, 3]], audio)
    assert torch.equal(expected[[0, 2, 4]], embed_tokens(input_ids[[0, 2, 4]]))


def test_prefill_oracle_handles_current_chunk_positions() -> None:
    torch.manual_seed(1)
    embed_tokens = torch.nn.Embedding(32, 4)
    chunk_ids = torch.tensor([31, 4, 31], dtype=torch.long)
    audio = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    expected = _oracle_text_audio_embeddings(
        chunk_ids,
        embed_tokens,
        torch.tensor([0, 2], dtype=torch.long),
        audio[[1, 2]],
    )

    assert torch.equal(expected[0], audio[1])
    assert torch.equal(expected[2], audio[2])


def test_benchmark_arms_keep_graph_configuration_external() -> None:
    assert _arm_config("A", None).prefill_backend == "disabled"
    assert _arm_config("B", None).prefill_backend == "disabled"
    assert _arm_config("C", None).prefill_backend == "breakable"
    assert _arm_config("C", "disabled").prefill_backend == "disabled"

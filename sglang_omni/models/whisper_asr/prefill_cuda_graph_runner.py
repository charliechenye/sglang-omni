# SPDX-License-Identifier: Apache-2.0
"""Whisper-specific metadata adapter for breakable prefill CUDA graphs."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)


class WhisperPrefillCudaGraphRunner(PrefillCudaGraphRunner):
    """Keep Whisper encoder-decoder metadata live during prefill graph replay."""

    def __init__(self, model_runner: Any) -> None:
        decoder = model_runner.model.model.decoder
        self_attention_layers = [layer.self_attn.attn for layer in decoder.layers]
        cross_attention_layers = [layer.encoder_attn.attn for layer in decoder.layers]
        original_attention_layers = model_runner.attention_layers
        model_runner.attention_layers = self_attention_layers + cross_attention_layers
        try:
            super().__init__(model_runner)
        finally:
            # note(chenye): decode and eager paths expect the original attention registry,
            # so the temporary self+cross-attention view must not escape BCG setup.
            model_runner.attention_layers = original_attention_layers

    def capture_prepare(self, num_tokens: int) -> tuple[ForwardBatch, Any]:
        forward_batch, attn_backend = super().capture_prepare(num_tokens)

        # note(chenye): capture must exercise cross-attention without capturing
        # request-specific encoder K/V writes, so use a cached one-token context.
        encoder_lens_cpu = [1] * forward_batch.batch_size
        forward_batch.encoder_lens = torch.ones(
            forward_batch.batch_size,
            dtype=torch.int64,
            device=self.device,
        )
        forward_batch.encoder_lens_cpu = encoder_lens_cpu
        forward_batch.encoder_cached = [True] * forward_batch.batch_size
        return forward_batch, attn_backend

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        encoder_lens_cpu = forward_batch.encoder_lens_cpu
        if not encoder_lens_cpu or not all(encoder_lens_cpu):
            return False
        return super().can_run_graph(forward_batch)

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> ForwardBatch:
        static_forward_batch = super().load_batch(forward_batch, **kwargs)
        # note(chenye): cross-attention planning must follow the live request, so
        # restore encoder metadata that SGLang 0.5.16 omits from its static batch.
        static_forward_batch.encoder_lens_cpu = forward_batch.encoder_lens_cpu
        static_forward_batch.encoder_cached = forward_batch.encoder_cached
        static_forward_batch.encoder_out_cache_loc = forward_batch.encoder_out_cache_loc
        return static_forward_batch

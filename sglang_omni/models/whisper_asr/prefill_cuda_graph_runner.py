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
    def __init__(self, model_runner: Any) -> None:
        decoder = model_runner.model.model.decoder
        self_attention_layers = [layer.self_attn.attn for layer in decoder.layers]
        cross_attention_layers = [layer.encoder_attn.attn for layer in decoder.layers]
        original_attention_layers = model_runner.attention_layers
        model_runner.attention_layers = self_attention_layers + cross_attention_layers
        try:
            super().__init__(model_runner)
        finally:
            model_runner.attention_layers = original_attention_layers

    def capture_prepare(self, num_tokens: int) -> tuple[ForwardBatch, Any]:
        forward_batch, attn_backend = super().capture_prepare(num_tokens)

        # Capture reads cached cross-attention K/V, not request-specific writes.
        forward_batch.encoder_lens = torch.ones(
            forward_batch.batch_size,
            dtype=torch.int64,
            device=self.device,
        )
        forward_batch.encoder_lens_cpu = [1] * forward_batch.batch_size
        forward_batch.encoder_cached = [True] * forward_batch.batch_size
        return forward_batch, attn_backend

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> ForwardBatch:
        static_forward_batch = super().load_batch(forward_batch, **kwargs)
        # SGLang 0.5.16 omits these Whisper encoder-decoder fields from replay.
        static_forward_batch.encoder_lens_cpu = forward_batch.encoder_lens_cpu
        static_forward_batch.encoder_cached = forward_batch.encoder_cached
        static_forward_batch.encoder_out_cache_loc = forward_batch.encoder_out_cache_loc
        return static_forward_batch

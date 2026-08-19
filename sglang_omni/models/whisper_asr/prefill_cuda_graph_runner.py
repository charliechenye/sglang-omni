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
    """Keep encoder-decoder metadata live while capturing the decoder body.

    SGLang's generic prefill runner is written for decoder-only models. It
    copies the token-side batch fields into its static batch but, in 0.5.16,
    omits the host encoder mirrors and the encoder KV write locations. Whisper
    also has two attention modules per decoder block, while generic layer
    discovery only sees ``self_attn``.
    """

    def __init__(self, model_runner: Any) -> None:
        decoder = model_runner.model.model.decoder
        self_attention_layers = [layer.self_attn.attn for layer in decoder.layers]
        cross_attention_layers = [layer.encoder_attn.attn for layer in decoder.layers]
        if len(self_attention_layers) != len(cross_attention_layers):
            raise RuntimeError(
                "Whisper prefill CUDA graph requires one cross-attention layer "
                "for every self-attention layer"
            )
        original_attention_layers = model_runner.attention_layers
        model_runner.attention_layers = self_attention_layers + cross_attention_layers
        try:
            super().__init__(model_runner)
        finally:
            # note(chenye): decode and eager paths expect the original attention registry,
            # so the temporary self+cross-attention view must not escape BCG setup.
            model_runner.attention_layers = original_attention_layers

    @staticmethod
    def _has_encoder_inputs(forward_batch: ForwardBatch) -> bool:
        """Return whether every uncached encoder request has source inputs."""
        if forward_batch.mm_inputs is None:
            return False
        if len(forward_batch.mm_inputs) != forward_batch.batch_size:
            return False
        for index, cached in enumerate(forward_batch.encoder_cached or []):
            if cached:
                continue
            mm_input = forward_batch.mm_inputs[index]
            if mm_input is None or not any(
                item.feature is not None or item.precomputed_embeddings is not None
                for item in mm_input.mm_items
            ):
                return False
        return True

    @classmethod
    def _encoder_metadata_is_usable(cls, forward_batch: ForwardBatch) -> bool:
        """Check the fields required by the eager encoder/KV-cache prefix."""
        encoder_lens = forward_batch.encoder_lens
        encoder_lens_cpu = forward_batch.encoder_lens_cpu
        encoder_cached = forward_batch.encoder_cached
        if encoder_lens is None or encoder_lens.numel() != forward_batch.batch_size:
            return False
        if (
            encoder_lens_cpu is None
            or len(encoder_lens_cpu) != forward_batch.batch_size
        ):
            return False
        if encoder_cached is None or len(encoder_cached) != forward_batch.batch_size:
            return False
        # note(chenye): BCG admission must not synchronize on GPU metadata, so inspect
        # encoder lengths through SGLang's host mirror.
        if any(int(length) <= 0 for length in encoder_lens_cpu):
            return False

        uncached_length = sum(
            int(encoder_lens_cpu[index])
            for index, cached in enumerate(encoder_cached)
            if not cached
        )
        if uncached_length:
            cache_loc = forward_batch.encoder_out_cache_loc
            if (
                cache_loc is None
                or cache_loc.ndim != 1
                or cache_loc.numel() != uncached_length
            ):
                return False
            if not cls._has_encoder_inputs(forward_batch):
                return False
        return True

    def capture_prepare(self, num_tokens: int) -> tuple[ForwardBatch, Any]:
        forward_batch, attn_backend = super().capture_prepare(num_tokens)

        # note(chenye): capture must exercise cross-attention without capturing
        # request-specific encoder K/V writes, so use a cached one-token context.
        encoder_lens_cpu = [1] * forward_batch.batch_size
        forward_batch.encoder_lens = torch.tensor(
            encoder_lens_cpu,
            dtype=torch.int64,
            device=self.device,
        )
        forward_batch.encoder_lens_cpu = encoder_lens_cpu
        forward_batch.encoder_cached = [True] * forward_batch.batch_size
        forward_batch.encoder_out_cache_loc = None
        return forward_batch, attn_backend

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        # note(chenye): incomplete encoder context must stay eager because graph replay
        # could otherwise read stale cross-attention KV.
        if not self._encoder_metadata_is_usable(forward_batch):
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
        static_forward_batch.encoder_lens = forward_batch.encoder_lens
        static_forward_batch.encoder_lens_cpu = forward_batch.encoder_lens_cpu
        static_forward_batch.encoder_cached = forward_batch.encoder_cached
        static_forward_batch.encoder_out_cache_loc = forward_batch.encoder_out_cache_loc
        return static_forward_batch


__all__ = ["WhisperPrefillCudaGraphRunner"]

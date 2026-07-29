# SPDX-License-Identifier: Apache-2.0
"""Per-chunk-count CUDA graph for the MOSS-TD Whisper encoder.

The Whisper encoder is a fixed-shape, stateless pure function: input mel
[num_chunks, num_mel_bins, input_feature_len] -> [num_chunks, encoder_len, d_model].
Nothing survives a call, so the persistent-state registry stays empty here.

Only the first dim (chunk count) varies, so we bucket over chunk count and pad
up to the nearest captured bucket on replay.
"""

from __future__ import annotations

import logging

import torch

from sglang_omni.cuda_graph import (
    DEFAULT_MAX_FAILURES,
    DEFAULT_MAX_KEYS,
    KeyedGraphCache,
    PersistentStateRegistry,
)

logger = logging.getLogger(__name__)

ENCODER_CUDA_GRAPH_ENV = "SGLANG_OMNI_MOSS_TD_ENCODER_GRAPH"


class WhisperEncoderCudaGraphRunner:
    # Note: (Jiaxin Deng) class-level default so a runner built without __init__
    # still reads as ungraphed instead of raising on the encode path.
    _cache: KeyedGraphCache | None = None

    def __init__(
        self,
        encoder,
        num_mel_bins: int,
        input_feature_len: int,
        min_free_gb: float = 3.0,
        warmup_iters: int = 3,
        *,
        max_keys: int = DEFAULT_MAX_KEYS,
        max_failures: int = DEFAULT_MAX_FAILURES,
    ) -> None:
        self._encoder = encoder
        self._num_mel_bins = int(num_mel_bins)
        self._input_feature_len = int(input_feature_len)
        self._device = next(encoder.parameters()).device
        self._dtype = next(encoder.parameters()).dtype
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = int(warmup_iters)
        self._max_keys = int(max_keys)
        self._max_failures = int(max_failures)
        self._cache = None
        self._persistent_state = PersistentStateRegistry()
        self._forward_batch = None

    @property
    def persistent_state(self) -> PersistentStateRegistry:
        """Empty: the encoder carries nothing across calls."""
        return self._persistent_state

    def captured_buckets(self) -> list[int]:
        """Chunk counts that have a captured graph, ascending."""
        if self._cache is None:
            return []
        return sorted(key[0] for key in self._cache.graphs)

    def _enough_free_vram(self) -> bool:
        free, _ = torch.cuda.mem_get_info(self._device)
        if free >= self._min_free_bytes:
            return True
        logger.warning(
            "MOSS-TD encoder CUDA graph: free VRAM %.1fGB < %.1fGB headroom; "
            "leaving the remaining chunk counts on eager",
            free / 1024**3,
            self._min_free_bytes / 1024**3,
        )
        return False

    def _warmup(self, static_feat, static_pos, forward_batch) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._encoder(static_feat, static_pos, forward_batch)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def _capture_bucket(self, c: int, encoder_len: int, forward_batch) -> tuple:
        static_feat = torch.zeros(
            c,
            self._num_mel_bins,
            self._input_feature_len,
            device=self._device,
            dtype=self._dtype,
        )
        static_pos = torch.arange(encoder_len, device=self._device, dtype=torch.long)
        self._warmup(static_feat, static_pos, forward_batch)

        self._persistent_state.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._cache.memory_pool(), capture_error_mode="thread_local"
        ):
            static_out = self._encoder(static_feat, static_pos, forward_batch)
        self._persistent_state.snapshot_addresses()
        logger.info(
            "Captured MOSS-TD encoder CUDA graph chunks=%d -> out %s",
            c,
            tuple(static_out.shape),
        )
        return graph, static_feat, static_pos, static_out

    @torch.no_grad()
    def capture(self, chunk_buckets, forward_batch=None) -> None:
        """Capture one graph per chunk-count bucket, once, at warmup."""
        self._forward_batch = forward_batch
        buckets = sorted({int(x) for x in chunk_buckets if int(x) >= 1})
        if not buckets:
            return
        if self._cache is None:
            self._cache = KeyedGraphCache(
                name="MOSS-TD encoder",
                batch_sizes=buckets,
                env_var=ENCODER_CUDA_GRAPH_ENV,
                max_keys=self._max_keys,
                max_failures=self._max_failures,
            )
        if not self._cache.enabled:
            logger.info(
                "MOSS-TD encoder CUDA graphs disabled by %s", ENCODER_CUDA_GRAPH_ENV
            )
            return
        encoder_len = (self._input_feature_len - 1) // 2 + 1
        with torch.cuda.device(self._device):
            # Note: (Jiaxin Deng) a declined headroom check stops the pass rather
            # than skipping one bucket: free VRAM does not grow between buckets.
            self._cache.warmup(
                [(c,) for c in buckets],
                lambda key: self._capture_bucket(key[0], encoder_len, forward_batch),
                precheck=self._enough_free_vram,
            )

    @torch.no_grad()
    def run(self, input_features, encoder_position_ids, forward_batch):
        """Replay the graph for [n, num_mel_bins, input_feature_len] features,
        padding up to the nearest captured bucket. Falls back to eager if no
        bucket fits or the input_feature_len differs from capture."""
        n = input_features.shape[0]
        graphs = {} if self._cache is None else self._cache.graphs
        chunk_bucket = min((key[0] for key in graphs if key[0] >= n), default=None)
        if chunk_bucket is None or input_features.shape[-1] != self._input_feature_len:
            return self._encoder(input_features, encoder_position_ids, forward_batch)
        graph, static_feat, _static_pos, static_out = graphs[(chunk_bucket,)]
        static_feat[:n].copy_(input_features)
        if n < chunk_bucket:
            static_feat[n:].zero_()
        graph.replay()
        return static_out[:n].clone()

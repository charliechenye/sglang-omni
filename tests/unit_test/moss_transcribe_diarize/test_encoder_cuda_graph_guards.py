# SPDX-License-Identifier: Apache-2.0
"""Shared-cache guards on the MOSS-TD Whisper encoder graph runner.

A tiny capturable stand-in replaces sglang's WhisperEncoder so the kill switch,
the key ceiling and the capture fuse are exercised without the checkpoint. The
real-encoder bit-identity gate lives in ``test_encoder_cuda_graph.py``.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_transcribe_diarize.encoder_cuda_graph import (
    ENCODER_CUDA_GRAPH_ENV,
    WhisperEncoderCudaGraphRunner,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-graph capture requires a GPU"
)

_MEL = 4
_FEAT_LEN = 32


class _FakeEncoder(torch.nn.Module):
    """``(features [n, mel, T], position_ids, forward_batch) -> [n, T//2, d]``."""

    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(_MEL, d_model)

    def forward(self, features, position_ids, forward_batch):
        del position_ids, forward_batch
        return self.proj(features.transpose(1, 2)[:, ::2])


def _pos(device: str = "cpu") -> torch.Tensor:
    return torch.arange((_FEAT_LEN - 1) // 2 + 1, device=device, dtype=torch.long)


def _feats(n: int, device: str = "cpu") -> torch.Tensor:
    return torch.randn(n, _MEL, _FEAT_LEN, device=device)


def _runner(device: str = "cpu", **kwargs):
    encoder = _FakeEncoder().to(device).eval()
    return encoder, WhisperEncoderCudaGraphRunner(encoder, _MEL, _FEAT_LEN, **kwargs)


@torch.no_grad()
def test_env_kill_switch_skips_capture_entirely(monkeypatch):
    monkeypatch.setenv(ENCODER_CUDA_GRAPH_ENV, "0")
    encoder, runner = _runner()
    attempts = []
    runner._capture_bucket = lambda *args, **kwargs: attempts.append(1)

    runner.capture([1, 2, 4])

    assert attempts == [], "the kill switch must skip capture, not just replay"
    assert runner.captured_buckets() == []
    features, positions = _feats(2), _pos()
    assert torch.equal(
        runner.run(features, positions, None), encoder(features, positions, None)
    )


@cuda_only
@torch.no_grad()
def test_key_ceiling_declines_without_evicting_hot_keys():
    encoder, runner = _runner("cuda", max_keys=2)
    runner.capture([1, 2, 4, 8])

    assert runner.captured_buckets() == [4, 8]

    features = _feats(3, "cuda")
    padded = torch.zeros(4, _MEL, _FEAT_LEN, device="cuda")
    padded[:3] = features
    graphed = runner.run(features, _pos("cuda"), None)

    assert torch.equal(graphed, encoder(padded, _pos("cuda"), None)[:3])
    assert runner.captured_buckets() == [4, 8], "a replay must not evict a hot key"


@cuda_only
@torch.no_grad()
def test_repeated_capture_failures_fuse_the_capture_pass_off():
    _encoder, runner = _runner("cuda", max_failures=2)
    attempts = []

    def boom(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("simulated capture OOM")

    runner._capture_bucket = boom
    runner.capture([1, 2, 4, 8, 16, 32])

    assert len(attempts) == 2, "the fuse must stop the pass after max_failures"
    assert runner.captured_buckets() == []


@cuda_only
@torch.no_grad()
def test_failed_bucket_is_not_recaptured():
    encoder, runner = _runner("cuda")
    capture_bucket = runner._capture_bucket
    attempts = []

    def flaky(chunks, *args, **kwargs):
        attempts.append(chunks)
        if chunks == 2:
            raise RuntimeError("simulated capture OOM")
        return capture_bucket(chunks, *args, **kwargs)

    runner._capture_bucket = flaky
    runner.capture([1, 2])
    runner.capture([1, 2])

    assert attempts == [2, 1], "a failed bucket must not be captured again"
    assert runner.captured_buckets() == [1]

    features = _feats(2, "cuda")
    assert torch.equal(
        runner.run(features, _pos("cuda"), None),
        encoder(features, _pos("cuda"), None),
    )


@cuda_only
@torch.no_grad()
def test_no_persistent_state_is_declared():
    """The encoder is a pure function of the mel input, so nothing is declared."""
    _encoder, runner = _runner("cuda")
    runner.capture([1])

    assert runner.persistent_state.is_empty()
    assert runner.persistent_state.declared_names() == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

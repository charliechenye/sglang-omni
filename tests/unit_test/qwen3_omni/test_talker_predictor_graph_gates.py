# SPDX-License-Identifier: Apache-2.0
"""Operational gates on the Qwen3-Omni talker predictor CUDA graph.

The capture itself was already correct; what it lacked was every guard the
Qwen3-TTS predictor graph carries: an env kill switch, the server-arg gates, a
key ceiling, a global fuse after repeated capture failures, and a shared graph
pool. These pin that behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.cuda_graph import KeyedGraphCache
from sglang_omni.models.qwen3_omni.components import talker as talker_module
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker

_HAS_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA, reason="predictor graph dispatch requires CUDA"
)


def _talker(*, batch_sizes=(1, 2, 4), max_keys=32, max_failures=8) -> Qwen3OmniTalker:
    talker = object.__new__(Qwen3OmniTalker)
    talker._predictor_graph_cache = KeyedGraphCache(
        name="Qwen3-Omni predictor",
        batch_sizes=batch_sizes,
        env_var=talker_module.QWEN3_OMNI_PREDICTOR_GRAPH_ENV,
        max_keys=max_keys,
        max_failures=max_failures,
    )
    talker._predictor_graph_runtime_checked = True
    return talker


def _inputs(batch_size: int):
    device = torch.device("cuda")
    layer0 = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    hidden = torch.zeros(batch_size, 1, 8, dtype=torch.bfloat16, device=device)
    return layer0, hidden


def _dispatch(talker, batch_size: int):
    layer0, hidden = _inputs(batch_size)
    return talker._code_predictor_forward_single_token_graph(
        layer0_codes=layer0,
        talker_hidden=hidden,
        batch_size=batch_size,
        code_dtype=layer0.dtype,
    )


def test_env_kill_switch_skips_the_graph_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(talker_module.QWEN3_OMNI_PREDICTOR_GRAPH_ENV, "0")
    talker = _talker()
    captured = []
    monkeypatch.setattr(
        talker_module,
        "_PredictorDecodeGraph",
        lambda *a, **k: captured.append(1),
    )
    assert _dispatch(talker, 2) is None
    assert captured == [], "kill switch must prevent capture entirely"


@pytest.mark.parametrize(
    "server_args, expected_enabled",
    [
        (dict(disable_cuda_graph=False, tp_size=1), True),
        (dict(disable_cuda_graph=True, tp_size=1), False),
        (dict(disable_cuda_graph=False, tp_size=2), False),
    ],
)
def test_server_arg_gates(
    monkeypatch: pytest.MonkeyPatch, server_args, expected_enabled
):
    monkeypatch.delenv(talker_module.QWEN3_OMNI_PREDICTOR_GRAPH_ENV, raising=False)
    monkeypatch.setattr(
        talker_module,
        "get_global_server_args",
        lambda: SimpleNamespace(**server_args),
    )
    talker = _talker()
    talker._predictor_graph_runtime_checked = False

    talker._check_predictor_graph_runtime()

    assert talker._predictor_graph_cache.enabled is expected_enabled


def test_repeated_capture_failures_fuse_the_path_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(talker_module.QWEN3_OMNI_PREDICTOR_GRAPH_ENV, raising=False)
    talker = _talker(batch_sizes=(1, 2, 4), max_failures=2)
    attempts = []

    def _boom(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(talker_module, "_PredictorDecodeGraph", _boom)

    assert _dispatch(talker, 1) is None
    assert talker._predictor_graph_cache.enabled is True
    assert _dispatch(talker, 2) is None
    assert (
        talker._predictor_graph_cache.enabled is False
    ), "the graph path must fuse off after repeated capture failures"

    before = len(attempts)
    assert _dispatch(talker, 4) is None
    assert len(attempts) == before, "a fused path must not attempt capture again"


def test_key_ceiling_declines_new_signatures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(talker_module.QWEN3_OMNI_PREDICTOR_GRAPH_ENV, raising=False)
    talker = _talker(batch_sizes=(1, 2, 4), max_keys=1)
    monkeypatch.setattr(
        talker_module,
        "_PredictorDecodeGraph",
        lambda *a, **k: SimpleNamespace(replay=lambda *_: ("codes", "embeds")),
    )
    assert _dispatch(talker, 1) == ("codes", "embeds")
    assert _dispatch(talker, 2) is None, "second key must be declined at the ceiling"
    assert _dispatch(talker, 1) == ("codes", "embeds"), "hot key must survive"


def test_failed_capture_releases_its_graph_pool(monkeypatch: pytest.MonkeyPatch):
    """A capture that raises must reset() its graph so the pool is freed."""
    resets = []
    real_graph_cls = torch.cuda.CUDAGraph

    class _SpyGraph(real_graph_cls):
        def reset(self):
            resets.append(1)
            return super().reset()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _SpyGraph)
    monkeypatch.setattr(
        talker_module.Qwen3OmniTalker,
        "_code_predictor_forward_incremental_eager",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    talker = _talker()
    talker._predictor_input_buffer = torch.zeros(4, 1, 8, device="cuda")

    with pytest.raises(RuntimeError):
        talker_module._PredictorDecodeGraph(talker, 2, torch.long)

    assert resets, "failed capture must release the graph"

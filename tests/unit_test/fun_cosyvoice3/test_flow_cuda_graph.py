# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.stages as stages


@pytest.fixture(autouse=True)
def _cpu_cuda_contexts(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda, "device", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()
    )


def _flow(*, channels: int = 4, max_frames: int = 512) -> SimpleNamespace:
    parameter = torch.nn.Parameter(torch.zeros(1))
    return SimpleNamespace(
        parameters=lambda: iter((parameter,)),
        decoder=SimpleNamespace(
            t_scheduler="linear",
            inference_cfg_rate=0.0,
            rand_noise=torch.zeros(1, channels, max_frames),
            estimator=torch.nn.Identity(),
        ),
        output_size=channels,
        token_mel_ratio=1,
        spk_embed_affine_layer=torch.nn.Linear(3, 5),
        input_embedding=lambda token: torch.ones(*token.shape, channels),
        pre_lookahead_layer=lambda x, context=None: x,
    )


class _ReplayGraph:
    def __init__(
        self,
        static_inputs: tuple[torch.Tensor, ...],
        static_output: torch.Tensor,
        *,
        fail: bool = False,
    ) -> None:
        self._static_inputs = static_inputs
        self._static_output = static_output
        self._fail = fail

    def replay(self) -> None:
        if self._fail:
            raise RuntimeError("replay failed")
        self._static_output.copy_(
            self._static_inputs[0] + self._static_inputs[2] + self._static_inputs[5]
        )


def _runner() -> stages.FlowCudaGraphRunner:
    return stages.FlowCudaGraphRunner(
        _flow(), device=torch.device("cpu"), compute_dtype=None
    )


def _install(
    runner: stages.FlowCudaGraphRunner, key: tuple[int, int], *, fail: bool = False
) -> None:
    static_inputs = runner.capture_inputs(*key)
    static_output = torch.empty_like(static_inputs[0])
    runner.graphs[key] = stages.CapturedFlowCudaGraph(
        _ReplayGraph(static_inputs, static_output, fail=fail),
        static_inputs,
        static_output,
    )


def _solver_inputs(
    batch_size: int, frames: int, channels: int = 4
) -> tuple[torch.Tensor, ...]:
    x = torch.ones(batch_size, channels, frames)
    return (
        x,
        torch.linspace(0, 1, 11),
        torch.full_like(x, 2),
        torch.ones(batch_size, 1, frames),
        torch.zeros(batch_size, 5),
        torch.full_like(x, 3),
    )


def _packed_tokens(length: int = 17) -> SimpleNamespace:
    return SimpleNamespace(
        token=torch.ones(1, length, dtype=torch.int32),
        token_mask=torch.ones(1, length, 1, dtype=torch.bool),
        combined_token_lengths=(length,),
        prompt_mel_lengths=(0,),
        total_mel_lengths_tensor=torch.tensor([length]),
        prompt_feat=torch.zeros(1, 0, 4),
        embedding=torch.ones(1, 3),
    )


def test_verify_capture_shapes_rejects_unaligned_frames() -> None:
    with pytest.raises(ValueError, match="multiples"):
        stages.verify_flow_cuda_graph_capture_shapes(((1, 495),))


def test_resident_replay_crops_to_actual_frames() -> None:
    runner = _runner()
    _install(runner, (2, 496))
    x, t_span, mu, mask, spks, cond = _solver_inputs(2, 489)
    output = runner.run(x, t_span, mu, mask, spks, cond)

    assert output is not None
    assert output.shape == (2, 4, 489)
    assert torch.equal(output, x + mu + cond)


def test_nonresident_shape_returns_none() -> None:
    runner = _runner()
    _install(runner, (2, 496))
    assert runner.run(*_solver_inputs(2, 1)) is None


def test_replay_failure_clears_resident_graphs() -> None:
    runner = _runner()
    _install(runner, (1, 464), fail=True)
    with pytest.raises(RuntimeError, match="replay failed"):
        runner.run(*_solver_inputs(1, 449))
    assert runner.graphs == {}


def test_generate_flow_does_not_retry_eager_after_replay_failure(monkeypatch) -> None:
    eager_calls: list[object] = []
    monkeypatch.setattr(
        stages, "solve_flow_euler", lambda *args, **kwargs: eager_calls.append(args)
    )

    class _FailingRunner:
        def run(self, *args, **kwargs):
            raise RuntimeError("replay failed")

    with pytest.raises(RuntimeError, match="replay failed"):
        stages.generate_flow(
            _flow(max_frames=64),
            _packed_tokens(),
            cuda_graph_runner=_FailingRunner(),
        )
    assert eager_calls == []

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages


class _FakeGraph:
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
        self.replay_calls = 0

    def replay(self) -> None:
        self.replay_calls += 1
        if self._fail:
            raise RuntimeError("synthetic replay failure")
        self._static_output.copy_(
            self._static_inputs[0].float()
            + self._static_inputs[2].float()
            + self._static_inputs[5].float()
        )


def _runner(
    *, compute_dtype: torch.dtype | None = torch.float32
) -> stages._CosyVoice3FlowCudaGraphRunner:
    return stages._CosyVoice3FlowCudaGraphRunner(
        SimpleNamespace(t_scheduler="linear"),
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
        capture_shapes=(),
    )


def _inputs(
    batch_size: int, frames: int, *, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    x = torch.arange(batch_size * 80 * frames, dtype=dtype).reshape(
        batch_size, 80, frames
    )
    t_span = torch.linspace(0, 1, 11, dtype=dtype)
    mu = torch.full_like(x, 2)
    mask = torch.ones(batch_size, 1, frames, dtype=dtype)
    spks = torch.arange(batch_size * 80, dtype=dtype).reshape(batch_size, 80)
    cond = torch.full_like(x, 3)
    return x, t_span, mu, mask, spks, cond


def _install_graph(
    runner: stages._CosyVoice3FlowCudaGraphRunner,
    key: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
    fail: bool = False,
) -> _FakeGraph:
    static_inputs = _inputs(key[0], key[1], dtype=dtype)
    static_output = torch.zeros_like(static_inputs[0], dtype=torch.float32)
    graph = _FakeGraph(static_inputs, static_output, fail=fail)
    runner._graphs[key] = stages._CapturedFlowCudaGraph(
        graph, static_inputs, static_output
    )
    return graph


def test_runner_routes_exact_shapes_and_copies_all_solver_inputs(monkeypatch) -> None:
    runner = _runner(compute_dtype=torch.bfloat16)
    graph = _install_graph(runner, (2, 4), dtype=torch.bfloat16)

    def _unexpected_capture(*args, **kwargs):
        del args, kwargs
        raise AssertionError("request-path capture is forbidden")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _unexpected_capture)
    inputs = _inputs(2, 4, dtype=torch.bfloat16)
    output = runner.run(*inputs)

    assert output is not None
    expected = inputs[0].float() + inputs[2].float() + inputs[5].float()
    torch.testing.assert_close(output, expected)
    assert graph.replay_calls == 1
    for static, value in zip(runner._graphs[(2, 4)].static_inputs, inputs, strict=True):
        torch.testing.assert_close(static, value)

    changed_inputs = tuple(value.clone() for value in inputs)
    changed_inputs[0].add_(1)
    previous_output = output.clone()
    changed_output = runner.run(*changed_inputs)
    assert changed_output is not None
    assert not torch.equal(changed_output, previous_output)
    torch.testing.assert_close(output, previous_output)

    assert runner.run(*_inputs(1, 4, dtype=torch.bfloat16)) is None
    assert graph.replay_calls == 2


def test_runner_declines_dtype_mismatch_and_missing_exact_shape() -> None:
    runner = _runner(compute_dtype=torch.bfloat16)
    graph = _install_graph(runner, (2, 4), dtype=torch.bfloat16)

    assert runner.run(*_inputs(2, 4)) is None
    assert runner.run(*_inputs(2, 5, dtype=torch.bfloat16)) is None
    assert graph.replay_calls == 0


def test_runner_disables_a_shape_after_replay_failure() -> None:
    runner = _runner()
    graph = _install_graph(runner, (1, 3), fail=True)
    inputs = _inputs(1, 3)

    assert runner.run(*inputs) is None
    assert runner.run(*inputs) is None
    assert graph.replay_calls == 1
    assert (1, 3) not in runner._graphs
    assert (1, 3) in runner._failed


def test_runner_build_rejects_unsupported_dtype_without_capture() -> None:
    assert (
        stages._CosyVoice3FlowCudaGraphRunner.build(
            SimpleNamespace(),
            device=torch.device("cuda"),
            compute_dtype=torch.float16,
            capture_shapes=((1, 3),),
        )
        is None
    )


def test_runner_orders_requested_shapes_largest_first() -> None:
    runner = stages._CosyVoice3FlowCudaGraphRunner(
        SimpleNamespace(t_scheduler="linear"),
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
        capture_shapes=((1, 20), (4, 4), (2, 12), (1, 20)),
    )

    assert runner._capture_shapes == ((2, 12), (1, 20), (4, 4))


def test_graph_safe_mask_matches_all_false_row_fallback() -> None:
    masks = torch.tensor([[[True, False, False]], [[False, False, False]]])

    result = stages._graph_safe_nonstreaming_chunk_mask(
        torch.zeros(2, 3, 1),
        masks,
        False,
        False,
        0,
        0,
        -1,
    )

    assert result.tolist() == [
        [[True, False, False]],
        [[True, True, True]],
    ]


@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.float64])
def test_unsupported_dtype_stays_eager(compute_dtype) -> None:
    runner = _runner(compute_dtype=compute_dtype)
    graph = _install_graph(runner, (1, 3))

    assert runner.run(*_inputs(1, 3)) is None
    assert graph.replay_calls == 0

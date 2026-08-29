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


class _TinyCudaDecoder:
    t_scheduler = "linear"
    inference_cfg_rate = 0.5

    def forward_estimator(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        streaming: bool,
    ) -> torch.Tensor:
        if streaming:
            raise AssertionError("the CUDA graph fixture is buffered-only")
        return (
            0.125 * x
            + 0.25 * mu
            + 0.5 * cond
            + 0.01 * spks.unsqueeze(-1)
            + 0.02 * t[:, None, None]
        ) * mask


def _runner(
    *, compute_dtype: torch.dtype | None = torch.float32
) -> stages._CosyVoice3FlowCudaGraphRunner:
    return stages._CosyVoice3FlowCudaGraphRunner(
        SimpleNamespace(t_scheduler="linear"),
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
        capture_shapes=(),
    )


def _runner_with_shapes(
    shapes: tuple[tuple[int, int], ...],
) -> stages._CosyVoice3FlowCudaGraphRunner:
    return stages._CosyVoice3FlowCudaGraphRunner(
        SimpleNamespace(t_scheduler="linear"),
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
        capture_shapes=shapes,
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
    assert runner.failed_shapes == ((1, 3),)


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

    assert runner._capture_shapes == ((1, 20), (2, 12), (4, 4))


@pytest.mark.parametrize("shape", [(0, 16), (2, 0), (-1, 16), (2, -1)])
def test_runner_rejects_invalid_capture_shapes(shape) -> None:
    with pytest.raises(ValueError, match="batch_size >= 1 and frames >= 1"):
        _runner_with_shapes((shape,))


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


def _real_inputs(
    runner: stages._CosyVoice3FlowCudaGraphRunner,
    batch_size: int,
    frames: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(value.clone() for value in runner._capture_inputs(batch_size, frames))


def _assert_cuda_close(eager: torch.Tensor, graph: torch.Tensor | None) -> None:
    assert graph is not None
    assert eager.shape == graph.shape
    assert torch.isfinite(eager).all()
    assert torch.isfinite(graph).all()
    # The tiny fixture is intentionally BF16 end-to-end; these tolerances allow
    # normal BF16 accumulation differences without hiding a bad replay.
    torch.testing.assert_close(graph, eager, rtol=1e-2, atol=1e-2)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_bf16_cuda_graph_replay_matches_eager_and_updates_inputs() -> None:
    device = torch.device("cuda")
    decoder = _TinyCudaDecoder()
    runner = stages._CosyVoice3FlowCudaGraphRunner(
        decoder,
        device=device,
        compute_dtype=torch.bfloat16,
        capture_shapes=((2, 16),),
    )
    runner.capture()

    assert runner.captured_shapes == ((2, 16),)
    assert runner.failed_shapes == ()
    assert isinstance(runner._graphs[(2, 16)].graph, torch.cuda.CUDAGraph)
    inputs_a = _real_inputs(runner, 2, 16)
    assert all(value.dtype == torch.bfloat16 for value in inputs_a)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eager_a = stages._solve_flow_euler(decoder, *inputs_a).detach().clone()
        graph_a = runner.run(*inputs_a)
    _assert_cuda_close(eager_a, graph_a)
    assert graph_a is not None
    saved_a = graph_a.clone()

    inputs_b = tuple(value.clone() for value in inputs_a)
    inputs_b[0].add_(1)
    inputs_b[2].add_(0.5)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eager_b = stages._solve_flow_euler(decoder, *inputs_b).detach().clone()
        graph_b = runner.run(*inputs_b)
    _assert_cuda_close(eager_b, graph_b)
    assert graph_b is not None
    assert not torch.equal(graph_a, graph_b)
    torch.testing.assert_close(graph_a, saved_a)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_graphs_isolate_interleaved_exact_shapes() -> None:
    device = torch.device("cuda")
    decoder = _TinyCudaDecoder()
    runner = stages._CosyVoice3FlowCudaGraphRunner(
        decoder,
        device=device,
        compute_dtype=torch.bfloat16,
        capture_shapes=((1, 16), (2, 16)),
    )
    assert runner._capture_shapes == ((2, 16), (1, 16))
    runner.capture()

    assert runner.captured_shapes == ((1, 16), (2, 16))
    assert runner.failed_shapes == ()
    inputs_a = _real_inputs(runner, 1, 16)
    inputs_b = _real_inputs(runner, 2, 16)
    inputs_a[0].add_(0.25)
    inputs_b[0].add_(0.75)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eager_a = stages._solve_flow_euler(decoder, *inputs_a).detach().clone()
        eager_b = stages._solve_flow_euler(decoder, *inputs_b).detach().clone()
        graph_a_1 = runner.run(*inputs_a)
        graph_b_1 = runner.run(*inputs_b)
        graph_a_2 = runner.run(*inputs_a)
        graph_b_2 = runner.run(*inputs_b)

    _assert_cuda_close(eager_a, graph_a_1)
    _assert_cuda_close(eager_b, graph_b_1)
    _assert_cuda_close(eager_a, graph_a_2)
    _assert_cuda_close(eager_b, graph_b_2)
    assert graph_a_1 is not None and graph_a_2 is not None
    assert graph_b_1 is not None and graph_b_2 is not None
    torch.testing.assert_close(graph_a_1, graph_a_2)
    torch.testing.assert_close(graph_b_1, graph_b_2)

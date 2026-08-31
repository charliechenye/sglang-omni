# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.stages as stages


class _TokenEmbedding:
    def __init__(self, channels: int) -> None:
        self._channels = channels

    def __call__(self, token: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            *token.shape,
            self._channels,
            device=token.device,
            dtype=torch.float32,
        )


class _SpeakerAffine:
    in_features = 3
    out_features = 5

    def __call__(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            embedding.shape[0],
            self.out_features,
            device=embedding.device,
            dtype=embedding.dtype,
        )


class _ReplayGraph:
    def __init__(
        self,
        static_inputs: tuple[torch.Tensor, ...],
        static_output: torch.Tensor,
        *,
        fail: bool,
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
            self._static_inputs[0] + self._static_inputs[2] + self._static_inputs[5]
        )


@pytest.fixture(autouse=True)
def _patch_cuda_contexts(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "device", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()
    )


def _flow(*, channels: int = 4, max_frames: int = 512):
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    decoder = SimpleNamespace(
        t_scheduler="linear",
        inference_cfg_rate=0.0,
        rand_noise=torch.zeros(1, channels, max_frames, dtype=torch.bfloat16),
        estimator=torch.nn.Identity(),
    )
    return SimpleNamespace(
        parameters=lambda: iter((parameter,)),
        decoder=decoder,
        output_size=channels,
        token_mel_ratio=1,
        spk_embed_affine_layer=_SpeakerAffine(),
        input_embedding=_TokenEmbedding(channels),
        pre_lookahead_layer=torch.nn.Identity(),
    )


def _runner(flow=None) -> stages._FlowCudaGraphRunner:
    return stages._FlowCudaGraphRunner(
        flow or _flow(),
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )


def _inputs(
    batch_size: int, frames: int, *, channels: int = 4
) -> tuple[torch.Tensor, ...]:
    x = (
        torch.arange(batch_size * channels * frames, dtype=torch.float32)
        .reshape(batch_size, channels, frames)
        .to(torch.bfloat16)
    )
    t_span = torch.linspace(0, 1, 11, dtype=torch.bfloat16)
    mu = torch.full_like(x, 2)
    mask = torch.ones(batch_size, 1, frames, dtype=torch.bfloat16)
    spks = (
        torch.arange(batch_size * 5, dtype=torch.float32)
        .reshape(batch_size, 5)
        .to(torch.bfloat16)
    )
    cond = torch.full_like(x, 3)
    return x, t_span, mu, mask, spks, cond


def _install_graph(
    runner: stages._FlowCudaGraphRunner,
    key: tuple[int, int],
    *,
    fail: bool = False,
) -> _ReplayGraph:
    static_inputs = runner._capture_inputs(*key)
    static_output = torch.empty_like(static_inputs[0])
    graph = _ReplayGraph(static_inputs, static_output, fail=fail)
    runner._graphs[key] = stages._CapturedFlowCudaGraph(
        graph, static_inputs, static_output
    )
    return graph


def _generation_case():
    flow = _flow(max_frames=64)
    packed = SimpleNamespace(
        token=torch.ones(1, 17, dtype=torch.int32),
        token_mask=torch.ones(1, 17, 1, dtype=torch.bool),
        combined_token_lengths=(17,),
        target_token_lengths=(17,),
        prompt_mel_lengths=(0,),
        total_mel_lengths_tensor=torch.tensor([17]),
        prompt_feat=torch.zeros(1, 0, 4),
        embedding=torch.ones(1, 3),
    )
    return flow, packed


def test_resident_replay_and_nonresident_fallback(monkeypatch) -> None:
    flow = _flow(channels=6)
    runner = _runner(flow)
    resident = _install_graph(runner, (2, 496))
    nonresident = _install_graph(runner, (1, 464))

    inputs = _inputs(2, 489, channels=6)
    original_x, _, original_mu, _, _, original_cond = inputs
    output = runner.run(*inputs)

    assert output is not None
    assert output.shape == (2, 6, 489)
    assert torch.equal(output, original_x + original_mu + original_cond)
    assert resident.replay_calls == 1
    captured_inputs = runner._graphs[(2, 496)].static_inputs
    assert torch.count_nonzero(captured_inputs[0][..., 489:]) == 0
    assert torch.count_nonzero(captured_inputs[2][..., 489:]) == 0
    assert torch.count_nonzero(captured_inputs[3][..., 489:]) == 0
    assert torch.count_nonzero(captured_inputs[5][..., 489:]) == 0
    assert torch.equal(captured_inputs[1], inputs[1])
    assert torch.equal(captured_inputs[4], inputs[4])

    capture_calls = []
    monkeypatch.setattr(
        runner,
        "capture",
        lambda *args, **kwargs: capture_calls.append((args, kwargs)),
    )
    miss_inputs = _inputs(1, 1, channels=6)
    assert runner.run(*miss_inputs) is None
    assert capture_calls == []
    assert nonresident.replay_calls == 0

    fallback_flow, packed = _generation_case()
    eager_calls = []

    def _eager(*args, **kwargs):
        eager_calls.append((args, kwargs))
        return args[1]

    monkeypatch.setattr(stages, "_solve_flow_euler", _eager)
    generated = stages._generate_flow(fallback_flow, packed, cuda_graph_runner=runner)
    assert generated.shape == (1, 4, 17)
    assert len(eager_calls) == 1


def test_replay_failure_disables_all_graphs_without_retry(monkeypatch) -> None:
    runner = _runner()
    failing = _install_graph(runner, (1, 464), fail=True)
    surviving = _install_graph(runner, (1, 480))

    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        runner.run(*_inputs(1, 449))
    assert failing.replay_calls == 1
    assert runner._graphs == {}

    assert runner.run(*_inputs(1, 465)) is None
    assert surviving.replay_calls == 0

    flow, packed = _generation_case()
    eager_calls = []

    class _ReplayFailure:
        def run(self, *args, **kwargs):
            raise RuntimeError("synthetic graph failure")

    monkeypatch.setattr(
        stages,
        "_solve_flow_euler",
        lambda *args, **kwargs: eager_calls.append((args, kwargs)),
    )
    with pytest.raises(RuntimeError, match="synthetic graph failure"):
        stages._generate_flow(flow, packed, cuda_graph_runner=_ReplayFailure())
    assert eager_calls == []


@pytest.mark.parametrize("capture_fails", [False, True])
def test_factory_lifecycle_keeps_compile_and_serving_independent(
    monkeypatch, capture_fails: bool
) -> None:
    events = []
    flow = stages.FunCosyVoice3Flow(_flow())

    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: "cuda:0"
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, **kwargs: (flow, object()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        stages,
        "_enable_flow_cuda_graph_dit_compat",
        lambda: events.append("compat"),
    )
    monkeypatch.setattr(
        stages,
        "_compile_dit_backbone",
        lambda flow, *, compute_dtype: events.append("compile"),
    )

    class _FakeRunner:
        def __init__(self, flow, *, device, compute_dtype):
            events.append("runner")

        def capture(self, capture_shapes):
            events.append("capture")
            if capture_fails:
                raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(stages, "_FlowCudaGraphRunner", _FakeRunner)
    monkeypatch.setattr(
        flow,
        "attach_cuda_graph_runner",
        lambda runner: events.append("attach"),
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=True,
    )

    assert scheduler is not None
    expected = ["compat", "compile", "runner", "capture"]
    if not capture_fails:
        expected.append("attach")
    assert events == expected


def test_graph_safe_mask_preserves_buffered_behavior(monkeypatch) -> None:
    monkeypatch.setattr(torch._dynamo, "graph_break", lambda: None)
    masks = torch.tensor([[[1, 1, 0]], [[0, 0, 0]]], dtype=torch.float32)
    xs = torch.ones(2, 1, 3)

    result = stages._graph_safe_nonstreaming_chunk_mask(
        xs, masks, False, False, 0, 0, 0
    )

    assert result is masks
    assert torch.equal(result[0], torch.tensor([[1, 1, 0]], dtype=torch.float32))
    assert torch.equal(result[1], torch.ones(1, 3))

    with pytest.raises(RuntimeError, match="buffered non-streaming"):
        stages._graph_safe_nonstreaming_chunk_mask(xs, masks, True, False, 0, 0, 0)
    with pytest.raises(RuntimeError, match="buffered non-streaming"):
        stages._graph_safe_nonstreaming_chunk_mask(xs, masks, False, False, 0, 1, 0)

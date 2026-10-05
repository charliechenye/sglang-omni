# SPDX-License-Identifier: Apache-2.0
"""Scoped instrumentation around unmodified production encoder forwards."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol
from unittest.mock import patch

import torch
from torch import nn

from benchmarks.eval.minicpmo_vision_metrics import Distribution, Parity, summarize
from benchmarks.eval.minicpmo_vision_workload import EncoderInputs
from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder

PhaseName = Literal[
    "input_prepare", "vpm_embeddings", "vpm_encoder", "vpm_unpack", "resampler"
]


class VisionForward(Protocol):
    def __call__(self) -> torch.Tensor: ...


@dataclass(kw_only=True)
class EncoderForward:
    encoder: MiniCPMOImageEncoder
    inputs: EncoderInputs

    def __call__(self) -> torch.Tensor:
        return self.encoder(
            pixel_values=self.inputs.pixel_values, tgt_sizes=self.inputs.tgt_sizes
        )["image_embeds"]


@dataclass(kw_only=True)
class ResamplerChunk:
    vision_embeddings: torch.Tensor
    target_sizes: torch.Tensor


@dataclass(kw_only=True)
class ResamplerForward:
    resampler: nn.Module
    chunks: list[ResamplerChunk]

    def __call__(self) -> torch.Tensor:
        return torch.cat(
            [
                self.resampler(chunk.vision_embeddings, chunk.target_sizes)
                for chunk in self.chunks
            ],
            dim=0,
        ).flatten(0, 1)


@dataclass(kw_only=True)
class Trial:
    arm: str
    batch_size: int
    warmup: int
    iters: int
    status: Literal["ok", "OOM"]
    gpu_ms: Distribution | None = None
    cpu_wall_ms: Distribution | None = None
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    phases_ms: dict[PhaseName, Distribution] = field(default_factory=dict)
    parity: Parity | None = None
    median_speedup: float | None = None
    median_delta_percent: float | None = None
    error: str | None = None


@dataclass(kw_only=True)
class CudaPhases:
    spans: list[tuple[PhaseName, torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    current_phase: PhaseName = "input_prepare"
    current_event: torch.cuda.Event | None = None

    def begin(self) -> None:
        self.spans.clear()
        self.current_phase = "input_prepare"
        self.current_event = torch.cuda.Event(enable_timing=True)
        self.current_event.record()

    def switch(self, phase: PhaseName) -> None:
        if self.current_event is None:
            pass
        elif phase == self.current_phase:
            pass
        else:
            boundary = torch.cuda.Event(enable_timing=True)
            boundary.record()
            self.spans.append((self.current_phase, self.current_event, boundary))
            self.current_event = boundary
            self.current_phase = phase

    def finish(self) -> None:
        assert self.current_event is not None
        boundary = torch.cuda.Event(enable_timing=True)
        boundary.record()
        self.spans.append((self.current_phase, self.current_event, boundary))
        self.current_event = None

    def elapsed_ms(self) -> dict[PhaseName, float]:
        milliseconds: dict[PhaseName, float] = {
            "input_prepare": 0.0,
            "vpm_embeddings": 0.0,
            "vpm_encoder": 0.0,
            "vpm_unpack": 0.0,
            "resampler": 0.0,
        }
        for phase, start, end in self.spans:
            milliseconds[phase] += start.elapsed_time(end)
        return milliseconds


@contextmanager
def phase_hooks(encoder: MiniCPMOImageEncoder, phases: CudaPhases) -> Iterator[None]:
    original_run_vpm = encoder.run_vpm

    def finish_chunk(
        pixel_values: torch.Tensor,
        patch_attn_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
        patch_counts_cpu: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = original_run_vpm(
            pixel_values, patch_attn_mask, tgt_sizes, patch_counts_cpu
        )
        phases.switch("input_prepare")
        return embeddings

    def begin_embeddings(
        module: nn.Module, module_inputs: tuple[torch.Tensor, ...]
    ) -> None:
        phases.switch("vpm_embeddings")

    def end_embeddings(
        module: nn.Module, module_inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        phases.switch("input_prepare")

    def begin_encoder(
        module: nn.Module, module_inputs: tuple[torch.Tensor, ...]
    ) -> None:
        phases.switch("vpm_encoder")

    def end_encoder(
        module: nn.Module, module_inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        phases.switch("vpm_unpack")

    def begin_resampler(
        module: nn.Module, module_inputs: tuple[torch.Tensor, ...]
    ) -> None:
        phases.switch("resampler")

    with ExitStack() as cleanup:
        cleanup.callback(
            encoder.vpm.embeddings.register_forward_pre_hook(begin_embeddings).remove
        )
        cleanup.callback(
            encoder.vpm.embeddings.register_forward_hook(end_embeddings).remove
        )
        cleanup.callback(
            encoder.vpm.encoder.register_forward_pre_hook(begin_encoder).remove
        )
        cleanup.callback(encoder.vpm.encoder.register_forward_hook(end_encoder).remove)
        cleanup.callback(
            encoder.resampler.register_forward_pre_hook(begin_resampler).remove
        )
        cleanup.enter_context(patch.object(encoder, "run_vpm", finish_chunk))
        yield


def capture_resampler_inputs(operation: EncoderForward) -> list[ResamplerChunk]:
    chunks: list[ResamplerChunk] = []
    observed_need_weights: list[bool] = []

    def observe_attention(
        module: nn.Module,
        attention_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_options: dict[str, torch.Tensor | bool | None],
    ) -> None:
        option = attention_options.get("need_weights", True)
        assert isinstance(option, bool)
        observed_need_weights.append(option)

    def capture(
        module: nn.Module, module_inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        vision_embeddings, target_sizes = module_inputs
        chunks.append(
            ResamplerChunk(
                vision_embeddings=vision_embeddings.detach(),
                target_sizes=target_sizes.detach(),
            )
        )

    hook = operation.encoder.resampler.register_forward_pre_hook(capture)
    attention_hook = operation.encoder.resampler.attn.register_forward_pre_hook(
        observe_attention, with_kwargs=True
    )
    try:
        with torch.inference_mode():
            operation()
        torch.cuda.synchronize(operation.encoder.device)
    finally:
        hook.remove()
        attention_hook.remove()
    assert chunks
    if not observed_need_weights or not all(observed_need_weights):
        raise RuntimeError(
            "Current resampler already requests need_weights=False; the requested R0/R1 baseline changed"
        )
    else:
        print("observed current resampler need_weights=True", flush=True)
    return chunks


def benchmark_forward(
    operation: VisionForward,
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
    phases: CudaPhases | None = None,
) -> tuple[Trial, torch.Tensor | None]:
    trial = Trial(
        arm=arm, batch_size=batch_size, warmup=warmup, iters=iters, status="ok"
    )
    output: torch.Tensor | None = None
    cpu_output: torch.Tensor | None = None
    gpu_times: list[float] = []
    wall_times: list[float] = []
    phase_times: dict[PhaseName, list[float]] = {}
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for i in range(iters)
    ]
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for i in range(warmup):
                operation()
            torch.cuda.synchronize(device)
            for start, end in events:
                wall_start = time.perf_counter()
                start.record()
                if phases is not None:
                    phases.begin()
                else:
                    pass
                output = operation()
                if phases is not None:
                    phases.finish()
                else:
                    pass
                end.record()
                end.synchronize()
                wall_times.append((time.perf_counter() - wall_start) * 1000.0)
                gpu_times.append(start.elapsed_time(end))
                if phases is not None:
                    for phase, milliseconds in phases.elapsed_ms().items():
                        phase_times.setdefault(phase, []).append(milliseconds)
                else:
                    pass
                if len(gpu_times) == iters:
                    cpu_output = output.detach().cpu()
                else:
                    pass
                output = None
        trial.gpu_ms = summarize(gpu_times)
        trial.cpu_wall_ms = summarize(wall_times)
        trial.phases_ms = {
            phase: summarize(milliseconds)
            for phase, milliseconds in phase_times.items()
        }
    except torch.cuda.OutOfMemoryError as error:
        trial.status = "OOM"
        trial.error = str(error)
        output = None
        cpu_output = None
    finally:
        trial.peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        trial.peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    gc.collect()
    torch.cuda.empty_cache()
    return trial, cpu_output

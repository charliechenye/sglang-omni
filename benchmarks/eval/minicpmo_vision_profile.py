# SPDX-License-Identifier: Apache-2.0
"""Model introspection and CUDA contributor summaries for real-input traces."""

from __future__ import annotations

import hashlib
import inspect
from collections import defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import ProfilerActivity, profile, record_function

from benchmarks.eval.minicpmo_vision_experiments import BASELINE_BATCH_SIZE
from benchmarks.eval.minicpmo_vision_measurement import EncoderForward
from sglang_omni.models.minicpm_o.components.image_encoder import (
    STACKED_QKV,
    MiniCPMOImageEncoder,
    vision_config_object,
)
from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig


@dataclass(kw_only=True)
class ModelDetails:
    model_directory: str
    checkpoint_config_sha256: str
    hidden_size: int
    intermediate_size: int
    configured_vision_layers: int
    vision_layers: int
    num_attention_heads: int
    patch_size: int
    query_num: int
    configured_vision_batch_size: int
    vision_batch_size: int
    dtype: str
    device: str
    gpu_name: str | None
    qkv_backend_name: str | None
    pass_strided_qkv: bool | None
    fused_qkv: bool | None
    qkv_projection_shape: list[int] | None
    checkpoint_qkv_mapping: list[tuple[str, str, str]]
    resampler_type: str | None
    multihead_attention_default_need_weights: bool | None
    torch_version: str
    sglang_version: str


def inspect_model(
    model_dir: str, encoder: MiniCPMOImageEncoder | None, dtype: str, device: str
) -> ModelDetails:
    config = MiniCPMOConfig.from_pretrained(model_dir)
    vision_config = vision_config_object(config)
    config_values = config.to_dict()
    configured_batch_size = int(
        config_values.get("vision_batch_size", BASELINE_BATCH_SIZE)
    )
    configured_layers = vision_config.num_hidden_layers
    if encoder is None:
        backend = None
        strided_qkv = None
        fused_qkv = None
        projection_shape = None
        resampler_type = None
        default_need_weights = None
        layer_count = configured_layers - int(
            bool(config_values.get("drop_vision_last_layer", False))
        )
        dtype = f"{dtype} (requested; weights not loaded)"
        device = "cpu (preprocessing only)"
        gpu_name = None
        batch_size = configured_batch_size
    else:
        attention = encoder.vpm.encoder.layers[0].self_attn
        backend = attention.qkv_backend_name
        strided_qkv = attention.pass_strided_qkv
        fused_qkv = attention.use_qkv_parallel
        projection_shape = list(attention.qkv_proj.weight.shape)
        resampler_type = type(encoder.resampler).__name__
        default_need_weights = (
            inspect.signature(encoder.resampler.attn.forward)
            .parameters["need_weights"]
            .default
        )
        layer_count = len(encoder.vpm.encoder.layers)
        dtype = str(next(encoder.vpm.parameters()).dtype)
        device = str(encoder.device)
        gpu_name = torch.cuda.get_device_name(encoder.device)
        batch_size = encoder.vision_batch_size
    return ModelDetails(
        model_directory=model_dir,
        checkpoint_config_sha256=hashlib.sha256(
            (Path(model_dir) / "config.json").read_bytes()
        ).hexdigest(),
        hidden_size=vision_config.hidden_size,
        intermediate_size=vision_config.intermediate_size,
        configured_vision_layers=configured_layers,
        vision_layers=layer_count,
        num_attention_heads=vision_config.num_attention_heads,
        patch_size=vision_config.patch_size,
        query_num=config.query_num,
        configured_vision_batch_size=configured_batch_size,
        vision_batch_size=batch_size,
        dtype=dtype,
        device=device,
        gpu_name=gpu_name,
        qkv_backend_name=backend,
        pass_strided_qkv=strided_qkv,
        fused_qkv=fused_qkv,
        qkv_projection_shape=projection_shape,
        checkpoint_qkv_mapping=STACKED_QKV,
        resampler_type=resampler_type,
        multihead_attention_default_need_weights=default_need_weights,
        torch_version=torch.__version__,
        sglang_version=version("sglang"),
    )


@dataclass(kw_only=True)
class CudaContributor:
    category: str
    cuda_ms: float
    share: float


@dataclass(kw_only=True)
class ProfileArtifacts:
    table_path: str
    trace_path: str | None
    contributors: list[CudaContributor]


@contextmanager
def module_range(module: nn.Module, label: str) -> Iterator[None]:
    active_ranges: list[record_function] = []

    def begin(module: nn.Module, module_inputs: tuple[torch.Tensor, ...]) -> None:
        scope = record_function(label)
        scope.__enter__()
        active_ranges.append(scope)

    def end(
        module: nn.Module,
        module_inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        active_ranges.pop().__exit__(None, None, None)

    begin_hook = module.register_forward_pre_hook(begin)
    end_hook = module.register_forward_hook(end)
    try:
        yield
    finally:
        begin_hook.remove()
        end_hook.remove()
        for scope in reversed(active_ranges):
            scope.__exit__(None, None, None)


def contributor_category(event: FunctionEvent) -> str:
    ancestor = event
    while ancestor is not None:
        if ancestor.name.startswith("vision/attention_backend"):
            return "attention backend"
        elif ancestor.name == "resampler/attention":
            if event.name in (
                "aten::mm",
                "aten::addmm",
                "aten::linear",
                "aten::matmul",
            ):
                return "resampler projections/norms"
            else:
                return "resampler attention"
        elif ancestor.name == "vision/transformer":
            if "gelu" in event.name.lower():
                return "GELU"
            else:
                return "residual/pointwise"
        elif ancestor.name.startswith("vision/") or ancestor.name.startswith(
            "resampler/"
        ):
            return ancestor.name.split("/", 1)[1]
        else:
            ancestor = ancestor.cpu_parent
    return "input preparation / other"


def profile_encoder(
    operation: EncoderForward,
    profile_dir: Path,
    warmup: int,
    iters: int,
    chrome_trace: bool,
) -> ProfileArtifacts:
    encoder = operation.encoder
    with torch.inference_mode():
        for i in range(warmup):
            operation()
        torch.cuda.synchronize(encoder.device)
        with ExitStack() as cleanup:
            for module_name, module in encoder.named_modules():
                if module_name == "resampler":
                    label = "resampler/resampler projections/norms"
                elif module_name == "resampler.attn":
                    label = "resampler/attention"
                elif module_name.startswith("resampler."):
                    label = "resampler/resampler projections/norms"
                elif module_name == "vpm.embeddings":
                    label = "vision/patch embedding"
                elif module_name == "vpm.encoder":
                    label = "vision/transformer"
                elif module_name.endswith("self_attn.qkv_proj"):
                    label = "vision/QKV GEMM"
                elif module_name.endswith("self_attn.proj"):
                    label = "vision/attention output projection"
                elif module_name.endswith("self_attn"):
                    label = "vision/attention_backend"
                elif module_name.endswith(
                    ("layer_norm1", "layer_norm2", "post_layernorm")
                ):
                    label = "vision/LayerNorm"
                elif module_name.endswith("mlp.fc1"):
                    label = "vision/fc1"
                elif module_name.endswith("mlp.fc2"):
                    label = "vision/fc2"
                elif module_name.endswith("mlp.activation_fn"):
                    label = "vision/GELU"
                else:
                    continue
                cleanup.enter_context(module_range(module, label))
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as profiler:
                for i in range(iters):
                    with record_function(
                        f"image_encoder/{operation.inputs.tgt_sizes.shape[0]}_slices"
                    ):
                        operation()
                    profiler.step()
                torch.cuda.synchronize(encoder.device)
    table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by="self_device_time_total", row_limit=50
    )
    table_path = profile_dir / "profiler-table.txt"
    table_path.write_text(table)
    if chrome_trace:
        trace_path = profile_dir / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
    else:
        trace_path = None
    cuda_times: dict[str, float] = defaultdict(float)
    for event in profiler.events():
        if (
            event.device_type == torch.autograd.DeviceType.CPU
            and event.self_device_time_total > 0
        ):
            cuda_times[contributor_category(event)] += (
                event.self_device_time_total / 1000.0
            )
        else:
            pass
    total_cuda_ms = sum(cuda_times.values())
    if total_cuda_ms <= 0:
        raise RuntimeError(
            "Profiler recorded no CUDA kernel times; inspect CUPTI/CUDA profiler support"
        )
    else:
        contributors = [
            CudaContributor(
                category=category,
                cuda_ms=milliseconds / iters,
                share=milliseconds / total_cuda_ms,
            )
            for category, milliseconds in sorted(
                cuda_times.items(), key=lambda pair: -pair[1]
            )
        ]
    print(table)
    return ProfileArtifacts(
        table_path=str(table_path),
        trace_path=str(trace_path) if trace_path is not None else None,
        contributors=contributors,
    )

# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
    execute_flow_estimator,
    is_flow_estimator_trt,
)
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    cleanup_prepared_cosyvoice3_request,
    preprocess_cosyvoice3_payload,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_device_spec

# Note (xinran): This is an admission budget, not a maximum supported request
# length. The scheduler admits a request that exceeds it as a singleton Flow
# batch and defers following requests to the next batch.

_DEFAULT_FLOW_BATCH_ADMISSION_FRAMES = 8000

_AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "float32": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_COSYVOICE_INSTALL_HINT = (
    "Fun-CosyVoice3 support requires the `cosyvoice` package. "
    "Clone the official repository and set PYTHONPATH, or install it "
    "in the serving environment before launching Fun-CosyVoice3."
)


@dataclass(frozen=True)
class FlowBatchInput:
    token: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


@dataclass(frozen=True)
class _PackedFlowBatch:
    token: torch.Tensor
    token_mask: torch.Tensor
    combined_token_lengths: tuple[int, ...]
    prompt_token_lengths: tuple[int, ...]
    target_token_lengths: tuple[int, ...]
    prompt_mel_lengths: tuple[int, ...]
    total_mel_lengths: tuple[int, ...]
    combined_token_lengths_tensor: torch.Tensor
    total_mel_lengths_tensor: torch.Tensor
    prompt_mel_lengths_tensor: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


logger = logging.getLogger(__name__)


def _flow_device_and_dtype(flow: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(flow.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ValueError("Flow must expose at least one parameter") from exc
    return parameter.device, parameter.dtype


def _validate_flow_input(flow: Any, item: FlowBatchInput, index: int) -> None:
    if item.token.ndim != 2 or item.token.shape[0] != 1 or item.token.shape[1] <= 0:
        raise ValueError(f"input {index} token must have shape [1, target_tokens]")
    if item.prompt_token.ndim != 2 or item.prompt_token.shape[0] != 1:
        raise ValueError(
            f"input {index} prompt_token must have shape [1, prompt_tokens]"
        )
    if item.prompt_feat.ndim != 3 or item.prompt_feat.shape[0] != 1:
        raise ValueError(
            f"input {index} prompt_feat must have shape [1, prompt_frames, channels]"
        )
    if item.prompt_feat.shape[2] != flow.output_size:
        raise ValueError(
            f"input {index} prompt feature width must equal Flow output_size"
        )
    expected_frames = item.prompt_token.shape[1] * flow.token_mel_ratio
    if item.prompt_feat.shape[1] != expected_frames:
        raise ValueError(
            f"input {index} prompt feature length must equal prompt token length "
            f"times token_mel_ratio ({item.prompt_feat.shape[1]} != {expected_frames})"
        )
    if item.embedding.ndim != 2 or item.embedding.shape[0] != 1:
        raise ValueError(f"input {index} embedding must have shape [1, speaker_dim]")
    expected_embedding_size = getattr(flow.spk_embed_affine_layer, "in_features", None)
    if (
        expected_embedding_size is not None
        and item.embedding.shape[1] != expected_embedding_size
    ):
        raise ValueError(
            f"input {index} embedding width must be {expected_embedding_size}"
        )


def _pack_flow_inputs(flow: Any, inputs: Sequence[FlowBatchInput]) -> _PackedFlowBatch:
    if not inputs:
        raise ValueError("Flow batch must contain at least one input")
    for index, item in enumerate(inputs):
        _validate_flow_input(flow, item, index)

    device, dtype = _flow_device_and_dtype(flow)
    prompt_lengths = tuple(int(item.prompt_token.shape[1]) for item in inputs)
    target_lengths = tuple(int(item.token.shape[1]) for item in inputs)
    combined_lengths = tuple(
        p + t for p, t in zip(prompt_lengths, target_lengths, strict=True)
    )
    prompt_mel_lengths = tuple(int(item.prompt_feat.shape[1]) for item in inputs)
    total_mel_lengths = tuple(
        length * flow.token_mel_ratio for length in combined_lengths
    )
    combined_token_lengths_tensor = torch.tensor(
        combined_lengths, dtype=torch.int64, device=device
    )
    total_mel_lengths_tensor = torch.tensor(
        total_mel_lengths, dtype=torch.int64, device=device
    )
    prompt_mel_lengths_tensor = torch.tensor(
        prompt_mel_lengths, dtype=torch.int64, device=device
    )

    max_tokens = max(combined_lengths)
    token = torch.zeros(len(inputs), max_tokens, dtype=torch.int32, device=device)
    for index, item in enumerate(inputs):
        prompt_length = prompt_lengths[index]
        token[index, :prompt_length] = item.prompt_token[0].to(
            device=device, dtype=torch.int32
        )
        token[index, prompt_length : combined_lengths[index]] = item.token[0].to(
            device=device, dtype=torch.int32
        )
    token_mask = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < combined_token_lengths_tensor.unsqueeze(1)
    ).unsqueeze(-1)

    max_prompt_frames = max(prompt_mel_lengths)
    prompt_feat = torch.zeros(
        len(inputs), max_prompt_frames, flow.output_size, device=device, dtype=dtype
    )
    for index, item in enumerate(inputs):
        prompt_feat[index, : prompt_mel_lengths[index]] = item.prompt_feat[0].to(
            device=device, dtype=dtype
        )
    embedding = torch.cat(
        [item.embedding.to(device=device, dtype=dtype) for item in inputs], dim=0
    )
    return _PackedFlowBatch(
        token=token,
        token_mask=token_mask,
        combined_token_lengths=combined_lengths,
        prompt_token_lengths=prompt_lengths,
        target_token_lengths=target_lengths,
        prompt_mel_lengths=prompt_mel_lengths,
        total_mel_lengths=total_mel_lengths,
        combined_token_lengths_tensor=combined_token_lengths_tensor,
        total_mel_lengths_tensor=total_mel_lengths_tensor,
        prompt_mel_lengths_tensor=prompt_mel_lengths_tensor,
        prompt_feat=prompt_feat,
        embedding=embedding,
    )


def _solve_flow_euler(
    decoder: Any,
    x: torch.Tensor,
    t_span: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    batch_size, channels, frames = x.shape
    dtype = spks.dtype
    x_in = torch.zeros(2 * batch_size, channels, frames, device=x.device, dtype=dtype)
    mask_in = torch.zeros(2 * batch_size, 1, frames, device=x.device, dtype=dtype)
    mu_in = torch.zeros_like(x_in)
    t_in = torch.zeros(1, device=x.device, dtype=dtype)
    spks_in = torch.zeros(2 * batch_size, spks.shape[1], device=x.device, dtype=dtype)
    cond_in = torch.zeros_like(x_in)
    t, dt = t_span[0], t_span[1] - t_span[0]
    for step in range(1, len(t_span)):
        x_in[:batch_size] = x
        x_in[batch_size:] = x
        mask_in[:batch_size] = mask
        mask_in[batch_size:] = mask
        mu_in[:batch_size] = mu
        t_in[:] = t
        spks_in[:batch_size] = spks
        cond_in[:batch_size] = cond
        derivative = _forward_flow_estimator(
            decoder, x_in, mask_in, mu_in, t_in, spks_in, cond_in
        )
        conditional, unconditional = derivative[:batch_size], derivative[batch_size:]
        x = x + dt * (
            (1.0 + decoder.inference_cfg_rate) * conditional
            - decoder.inference_cfg_rate * unconditional
        )
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return x.float()


@torch.inference_mode()
def _generate_flow(flow: Any, packed: _PackedFlowBatch) -> torch.Tensor:
    embedding = flow.spk_embed_affine_layer(F.normalize(packed.embedding, dim=1))
    token_embedding = flow.input_embedding(torch.clamp(packed.token, min=0))
    h = flow.pre_lookahead_layer(
        token_embedding * packed.token_mask.to(token_embedding.dtype)
    )
    mu = h.repeat_interleave(flow.token_mel_ratio, dim=1).transpose(1, 2).contiguous()
    batch_size, channels, max_mel = mu.shape
    if channels != flow.output_size:
        raise ValueError("Flow pre-lookahead output width does not match output_size")
    mask = (
        (
            torch.arange(max_mel, device=mu.device).unsqueeze(0)
            < packed.total_mel_lengths_tensor.unsqueeze(1)
        )
        .unsqueeze(1)
        .to(mu.dtype)
    )
    cond = torch.zeros_like(mu)
    for index, prompt_frames in enumerate(packed.prompt_mel_lengths):
        cond[index, :, :prompt_frames] = packed.prompt_feat[
            index, :prompt_frames
        ].transpose(0, 1)
    decoder = flow.decoder
    if max_mel > decoder.rand_noise.shape[2]:
        raise ValueError(
            f"decoder.rand_noise supports {decoder.rand_noise.shape[2]} frames, "
            f"but batch requires {max_mel}"
        )
    z = (
        decoder.rand_noise[:, :, :max_mel]
        .to(device=mu.device, dtype=mu.dtype)
        .expand(batch_size, -1, -1)
        .clone()
    )
    t_span = torch.linspace(0, 1, 11, device=mu.device, dtype=mu.dtype)
    if decoder.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    return _solve_flow_euler(decoder, z, t_span, mu, mask, embedding, cond)


def _forward_flow_estimator(
    decoder: Any,
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    # note (guozhihao-224): CosyVoice forward_estimator hardcodes TRT shapes to
    # (2, 80, T). Packed Flow is CFG=2N, so TRT uses execute_flow_estimator.
    estimator = decoder.estimator
    if isinstance(estimator, torch.nn.Module):
        return decoder.forward_estimator(x, mask, mu, t, spks, cond, streaming=False)
    return execute_flow_estimator(estimator, x, mask, mu, t, spks, cond)


class FunCosyVoice3Flow:
    """CosyVoice3 Flow with batch inference enabled as its default API."""

    def __init__(self, flow: Any) -> None:
        self._flow = flow

    def __getattr__(self, name: str) -> Any:
        return getattr(self._flow, name)

    def parameters(self):
        return self._flow.parameters()

    def to(self, *args: Any, **kwargs: Any) -> "FunCosyVoice3Flow":
        self._flow.to(*args, **kwargs)
        return self

    def eval(self) -> "FunCosyVoice3Flow":
        self._flow.eval()
        return self

    @torch.inference_mode()
    def inference(self, inputs: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        packed = _pack_flow_inputs(self._flow, inputs)
        generated = _generate_flow(self._flow, packed)
        outputs: list[torch.Tensor] = []
        for index, prompt_frames in enumerate(packed.prompt_mel_lengths):
            mel = generated[
                index : index + 1,
                :,
                prompt_frames : packed.total_mel_lengths[index],
            ]
            expected_frames = (
                packed.target_token_lengths[index] * self._flow.token_mel_ratio
            )
            if mel.shape != (1, self._flow.output_size, expected_frames):
                raise RuntimeError(
                    f"Flow output {index} has unexpected shape {tuple(mel.shape)}"
                )
            outputs.append(mel)
        return outputs


def load_state(payload: StagePayload) -> FunCosyVoice3State:
    return _load_pipeline_state(payload, FunCosyVoice3State)


def store_state(payload: StagePayload, state: FunCosyVoice3State) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _attach_flow_estimator_trt(
    flow: Any,
    checkpoint_dir: str,
    device: str,
) -> None:
    from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
        build_flow_estimator_trt,
        resolve_flow_estimator_onnx,
    )

    if str(device).split(":", 1)[0].lower() != "cuda":
        raise RuntimeError(
            "enable_flow_estimator_trt requires a CUDA vocoder device, "
            f"got {device!r}"
        )
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        raise RuntimeError(
            "enable_flow_estimator_trt requires NVIDIA CUDA, "
            f"got platform {current_platform.device_type!r}"
        )

    onnx_path = resolve_flow_estimator_onnx(checkpoint_dir)
    wrapper = build_flow_estimator_trt(onnx_path, device)
    # note (guozhihao-224): CosyVoice registers estimator as an nn.Module child;
    # delete first so assigning the TRT wrapper does not raise TypeError.
    del flow.decoder.estimator
    flow.decoder.estimator = wrapper
    logger.info(
        "Fun-CosyVoice3 Flow DiT estimator is TensorRT (%s, max_cfg_batch=%d)",
        onnx_path,
        wrapper.max_batch,
    )


def _fold_weight_norm(module: torch.nn.Module) -> int:
    folded = 0
    for submodule in list(module.modules()):
        while is_parametrized(submodule):
            name = next(iter(submodule.parametrizations.keys()))
            remove_parametrizations(submodule, name, leave_parametrized=True)
            folded += 1
    return folded


def _hift_samples_per_mel_frame(hift: Any) -> int:
    rates = getattr(hift, "upsample_rates", None)
    hop_len = (
        getattr(hift, "istft_params", {}).get("hop_len")
        if hasattr(hift, "istft_params")
        else None
    )
    if not rates or not hop_len:
        raise RuntimeError(
            "Fun-CosyVoice3 HiFT generator is missing upsample_rates / "
            "istft_params; refusing to guess the mel->wave stride"
        )
    stride = int(hop_len)
    for rate in rates:
        stride *= int(rate)
    return stride


def _prepare_hift_for_inference(hift: Any) -> None:
    # note (Dayuxiaoshui): folding weight_norm is the only load-time step
    # batched decode needs. The pinned CausalHiFTGenerator already squeezes
    # the source to [B, T] before its STFT and casts f0_predictor to float64
    # inside inference(), and right-zero-padded mels reproduce per-request
    # output bit-for-bit except in the final mel frame of padded requests.
    folded = _fold_weight_norm(hift)
    logger.info(
        "Prepared Fun-CosyVoice3 HiFT for inference (folded %d weight_norm "
        "parametrizations)",
        folded,
    )


def _load_cosyvoice3_flow_hift(
    checkpoint_dir: str,
    device: str,
    fp16: bool = False,
    *,
    enable_flow_estimator_trt: bool = False,
) -> tuple[Any, Any]:
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        raise RuntimeError(_COSYVOICE_INSTALL_HINT) from exc

    cv = CosyVoice3(checkpoint_dir, fp16=fp16)
    flow = cv.model.flow
    hift = cv.model.hift
    flow.to(device).eval()
    hift.to(device).eval()
    _prepare_hift_for_inference(hift)
    del cv.model.llm
    wrapped = FunCosyVoice3Flow(flow)
    if enable_flow_estimator_trt:
        _attach_flow_estimator_trt(wrapped, checkpoint_dir, device)
    return wrapped, hift


def _configure_dit_torch_compile() -> None:
    """Enable the Inductor/Dynamo flags the DiT graph wants, without pulling in
    the full sglang.srt stack (the vocoder is a plain pipeline process)."""
    torch._inductor.config.fx_graph_cache = True
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = 1024


def _run_dit_estimator(
    estimator: Any,
    mel_frames: int,
    *,
    compute_dtype: torch.dtype | None = None,
) -> None:

    param = next(estimator.parameters())
    device, dtype = param.device, param.dtype
    t = int(mel_frames)
    # note (guozhihao-224): CFG batch 2; mel dim 80 matches pinned checkpoint proj_out.
    x = torch.randn(2, 80, t, device=device, dtype=dtype)
    mask = torch.ones(2, 1, t, device=device, dtype=dtype)
    mu = torch.randn(2, 80, t, device=device, dtype=dtype)
    timestep = torch.zeros(1, device=device, dtype=dtype)
    spks = torch.randn(2, 80, device=device, dtype=dtype)
    cond = torch.randn(2, 80, t, device=device, dtype=dtype)
    with torch.autocast(
        device_type=current_platform.device_type,
        dtype=compute_dtype,
        enabled=compute_dtype is not None,
    ):
        estimator(x, mask, mu, timestep, spks, cond, streaming=False)


def _compile_dit_backbone(
    flow: Any,
    *,
    warmup_mel_frames: int = 128,
    warmup_steps: int = 3,
    compute_dtype: torch.dtype | None = None,
) -> bool:

    estimator = getattr(getattr(flow, "decoder", None), "estimator", None)
    if not isinstance(estimator, torch.nn.Module):
        logger.warning(
            "Fun-CosyVoice3 DiT estimator is not a PyTorch module (%s); "
            "skipping torch.compile",
            type(estimator).__name__,
        )
        return False
    if warmup_mel_frames < 2:
        raise ValueError(f"warmup_mel_frames must be >= 2, got {warmup_mel_frames}")

    original_forward = estimator.forward
    _configure_dit_torch_compile()
    try:
        estimator.forward = torch.compile(original_forward, dynamic=True)
        with torch.inference_mode():
            for _ in range(warmup_steps):
                _run_dit_estimator(
                    estimator,
                    warmup_mel_frames,
                    compute_dtype=compute_dtype,
                )
    except Exception as exc:
        estimator.forward = original_forward
        logger.warning(
            "torch.compile for the Fun-CosyVoice3 DiT backbone failed "
            "(%s: %s); the flow decoder will run eager",
            type(exc).__name__,
            exc,
        )
        return False
    logger.info(
        "Compiled Fun-CosyVoice3 DiT backbone (dynamic=True, compute_dtype=%s, "
        "warmup_mel_frames=%d, warmup_steps=%d)",
        compute_dtype,
        warmup_mel_frames,
        warmup_steps,
    )
    return True


def create_preprocessing_executor(
    model_path: str,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")
    del model_path
    # note(chenye): Reference conditioning supports concurrent calls;
    # model prompt finalization is serialized.
    return SimpleScheduler(
        preprocess_cosyvoice3_payload,
        max_concurrency=max_concurrency,
        abort_callback=cleanup_prepared_cosyvoice3_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    onnx_intra_op_threads: int = 16,
) -> Any:
    from sglang_omni.models.fun_cosyvoice3.engine_builder import (
        FunCosyVoice3EngineBuilder,
    )

    return FunCosyVoice3EngineBuilder(
        onnx_intra_op_threads=onnx_intra_op_threads,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


@dataclass(frozen=True)
class _PreparedFlowRequest:
    index: int
    sample_rate: int
    flow_input: FlowBatchInput
    total_mel_frames: int
    # Set while preparing a scheduler batch.  This is the immutable provenance
    # key used to keep HiFT grouping inside the original Flow bucket.
    baseline_bucket_key: int | None = None


@dataclass(frozen=True)
class _FlowSegment:
    start: int
    end: int
    first_bucket_key: int
    last_bucket_key: int
    requests: tuple[_PreparedFlowRequest, ...]
    request_count: int
    padded_work: int
    newly_merged_span: int
    within_coalesce_span: bool


@dataclass(frozen=True)
class _FlowPartitionCandidate:
    padded_work: int
    bucket_range_signature: tuple[tuple[int, int], ...]
    segments: tuple[_FlowSegment, ...]


def _validate_flow_batch_coalescing_config(
    span_frames: int,
    max_added_padding_pct: float,
) -> None:
    if span_frames < 0:
        raise ValueError(
            "flow_batch_coalesce_span_frames must be greater than or equal to zero"
        )
    if not math.isfinite(max_added_padding_pct) or max_added_padding_pct < 0:
        raise ValueError(
            "flow_batch_coalesce_max_added_padding_pct must be finite and "
            "greater than or equal to zero"
        )
    if span_frames == 0 and max_added_padding_pct != 0:
        raise ValueError(
            "flow_batch_coalesce_max_added_padding_pct must be zero when "
            "flow_batch_coalesce_span_frames is zero"
        )


def _precompute_flow_segments(
    buckets: dict[int, list[_PreparedFlowRequest]],
    *,
    coalesce_span_frames: int,
) -> tuple[
    tuple[tuple[int, tuple[_PreparedFlowRequest, ...]], ...],
    list[list[_FlowSegment | None]],
]:
    atomic_groups = tuple(
        (bucket_key, tuple(requests))
        for bucket_key, requests in sorted(buckets.items())
    )
    segment_matrix: list[list[_FlowSegment | None]] = [
        [None] * (len(atomic_groups) + 1) for _ in range(len(atomic_groups) + 1)
    ]

    for start in range(len(atomic_groups)):
        flattened: list[_PreparedFlowRequest] = []
        minimum_total: int | None = None
        maximum_total: int | None = None
        for end in range(start + 1, len(atomic_groups) + 1):
            _, atomic_requests = atomic_groups[end - 1]
            flattened.extend(atomic_requests)
            atomic_minimum = min(
                request.total_mel_frames for request in atomic_requests
            )
            atomic_maximum = max(
                request.total_mel_frames for request in atomic_requests
            )
            minimum_total = (
                atomic_minimum
                if minimum_total is None
                else min(minimum_total, atomic_minimum)
            )
            maximum_total = (
                atomic_maximum
                if maximum_total is None
                else max(maximum_total, atomic_maximum)
            )
            assert minimum_total is not None
            assert maximum_total is not None
            newly_merged_span = 0 if end - start == 1 else maximum_total - minimum_total
            segment_matrix[start][end] = _FlowSegment(
                start=start,
                end=end,
                first_bucket_key=atomic_groups[start][0],
                last_bucket_key=atomic_groups[end - 1][0],
                requests=tuple(flattened),
                request_count=len(flattened),
                padded_work=len(flattened) * maximum_total,
                newly_merged_span=newly_merged_span,
                within_coalesce_span=(
                    end - start == 1 or newly_merged_span <= coalesce_span_frames
                ),
            )

    return atomic_groups, segment_matrix


def _minimum_flow_work_by_group_count(
    segments: Sequence[Sequence[_FlowSegment | None]],
    *,
    max_merged_span: int,
) -> list[int | None]:
    """Return minimum padded work for every exact partition size.

    The state covers a prefix of the sorted atomic buckets and every
    transition adds one contiguous segment.  Thus no non-contiguous or
    partially split atomic grouping can enter the production planner.
    """
    atomic_count = len(segments) - 1
    minimum_work: list[list[int | None]] = [
        [None] * (atomic_count + 1) for _ in range(atomic_count + 1)
    ]
    minimum_work[0][0] = 0

    for end in range(1, atomic_count + 1):
        for group_count in range(1, end + 1):
            best: int | None = None
            for start in range(group_count - 1, end):
                previous = minimum_work[start][group_count - 1]
                segment = segments[start][end]
                if (
                    previous is None
                    or segment is None
                    or not segment.within_coalesce_span
                    or segment.newly_merged_span > max_merged_span
                ):
                    continue
                candidate = previous + segment.padded_work
                if best is None or candidate < best:
                    best = candidate
            minimum_work[end][group_count] = best

    return minimum_work[atomic_count]


def _minimum_flow_partition(
    segments: Sequence[Sequence[_FlowSegment | None]],
    *,
    max_merged_span: int,
) -> list[_FlowPartitionCandidate | None]:
    """Return minimum-work partitions, lexicographically tie-broken by range."""
    atomic_count = len(segments) - 1
    candidates: list[list[_FlowPartitionCandidate | None]] = [
        [None] * (atomic_count + 1) for _ in range(atomic_count + 1)
    ]
    candidates[0][0] = _FlowPartitionCandidate(0, (), ())

    for end in range(1, atomic_count + 1):
        for group_count in range(1, end + 1):
            best: _FlowPartitionCandidate | None = None
            for start in range(group_count - 1, end):
                previous = candidates[start][group_count - 1]
                segment = segments[start][end]
                if (
                    previous is None
                    or segment is None
                    or not segment.within_coalesce_span
                    or segment.newly_merged_span > max_merged_span
                ):
                    continue
                candidate = _FlowPartitionCandidate(
                    padded_work=previous.padded_work + segment.padded_work,
                    bucket_range_signature=(
                        *previous.bucket_range_signature,
                        (segment.first_bucket_key, segment.last_bucket_key),
                    ),
                    segments=(*previous.segments, segment),
                )
                if best is None or (
                    candidate.padded_work,
                    candidate.bucket_range_signature,
                ) < (
                    best.padded_work,
                    best.bucket_range_signature,
                ):
                    best = candidate
            candidates[end][group_count] = best

    return candidates[atomic_count]


def _within_flow_padding_cap(
    padded_work: int,
    baseline_work: int,
    max_added_padding_pct: float,
) -> bool:
    # Keep the validated policy's floating-point comparison and tolerance.
    return (padded_work / baseline_work - 1) * 100 <= max_added_padding_pct + 1e-9


def _group_flow_requests(
    buckets: dict[int, list[_PreparedFlowRequest]],
    *,
    coalesce_span_frames: int,
    coalesce_max_added_padding_pct: float,
) -> list[list[_PreparedFlowRequest]]:
    """Choose an exact objective-optimal contiguous coarsening of Flow buckets.

    The adaptive path first minimizes solve count subject to the global
    whole-batch padding cap, then padded work, then maximum newly merged raw
    mel span, and finally the deterministic bucket-range signature.  The
    disabled path intentionally returns the existing insertion-order groups.
    """
    _validate_flow_batch_coalescing_config(
        coalesce_span_frames, coalesce_max_added_padding_pct
    )
    if coalesce_span_frames == 0:
        # Preserve current upstream grouping and insertion order when disabled.
        return list(buckets.values())
    if not buckets:
        return []

    atomic_groups, segment_matrix = _precompute_flow_segments(
        buckets, coalesce_span_frames=coalesce_span_frames
    )
    baseline_groups = [list(requests) for _, requests in atomic_groups]
    baseline_work = sum(
        len(requests) * max(request.total_mel_frames for request in requests)
        for _, requests in atomic_groups
    )
    if baseline_work == 0:
        # Let the existing Flow validation handle malformed zero-length inputs.
        return baseline_groups

    # Stage A: exact minimum work for every possible solve count.
    minimum_work_by_count = _minimum_flow_work_by_group_count(
        segment_matrix, max_merged_span=coalesce_span_frames
    )
    solve_count: int | None = None
    optimal_work: int | None = None
    for count, minimum_work in enumerate(minimum_work_by_count):
        if (
            count > 0
            and minimum_work is not None
            and _within_flow_padding_cap(
                minimum_work,
                baseline_work,
                coalesce_max_added_padding_pct,
            )
        ):
            solve_count = count
            optimal_work = minimum_work
            break

    # The atomic partition is always valid, so adaptive planning must never
    # turn an unexpected arithmetic edge case into a runtime failure.
    if solve_count is None or optimal_work is None:
        return baseline_groups

    # Stage B: binary-search the finite set of achievable segment spans for the
    # smallest maximum newly merged span that still attains Stage A's work.
    candidate_spans = sorted(
        {
            segment.newly_merged_span
            for row in segment_matrix
            for segment in row
            if segment is not None and segment.within_coalesce_span
        }
    )
    if not candidate_spans:
        return baseline_groups

    low = 0
    high = len(candidate_spans) - 1
    if (
        _minimum_flow_work_by_group_count(
            segment_matrix, max_merged_span=candidate_spans[high]
        )[solve_count]
        != optimal_work
    ):
        return baseline_groups
    while low < high:
        middle = (low + high) // 2
        work = _minimum_flow_work_by_group_count(
            segment_matrix, max_merged_span=candidate_spans[middle]
        )[solve_count]
        if work == optimal_work:
            high = middle
        else:
            low = middle + 1
    optimal_max_merged_span = candidate_spans[low]

    # Stage C: with the first three objective components fixed, reconstruct the
    # lexicographically smallest complete bucket-range signature.
    final = _minimum_flow_partition(
        segment_matrix, max_merged_span=optimal_max_merged_span
    )[solve_count]
    if final is None or final.padded_work != optimal_work:
        return baseline_groups
    return [list(segment.requests) for segment in final.segments]


def _group_by_padding_waste(
    items: Sequence[tuple[Any, torch.Tensor]],
    *,
    max_waste: float,
) -> Iterator[list[tuple[Any, torch.Tensor]]]:
    ordered = sorted(items, key=lambda pair: int(pair[1].shape[-1]))
    group: list[tuple[Any, torch.Tensor]] = []
    total = 0
    longest = 0
    for pair in ordered:
        length = int(pair[1].shape[-1])
        candidate_longest = max(longest, length)
        candidate_total = total + length
        if group and candidate_longest * (len(group) + 1) > max_waste * candidate_total:
            yield group
            group, total, longest = [], 0, 0
            candidate_longest = length
            candidate_total = length
        group.append(pair)
        total, longest = candidate_total, candidate_longest
    if group:
        yield group


class _CosyVoice3Vocoder(BatchVocoderBase):
    def __init__(
        self,
        flow: Any,
        hift: Any,
        compute_dtype: torch.dtype | None = None,
        flow_batch_bucket_frames: int = 50,
        hift_compute_dtype: str = "float32",
        hift_max_padding_waste: float = 1.5,
        flow_batch_coalesce_span_frames: int = 0,
        flow_batch_coalesce_max_added_padding_pct: float = 0.0,
    ) -> None:
        if flow_batch_bucket_frames <= 0:
            raise ValueError("flow_batch_bucket_frames must be greater than zero")
        _validate_flow_batch_coalescing_config(
            flow_batch_coalesce_span_frames,
            flow_batch_coalesce_max_added_padding_pct,
        )
        if hift_max_padding_waste < 1.0:
            raise ValueError("hift_max_padding_waste must be at least 1.0")
        if hift_compute_dtype not in _AUTOCAST_DTYPES:
            raise ValueError(
                f"Unsupported Fun-CosyVoice3 HiFT dtype {hift_compute_dtype!r}; "
                f"expected one of {sorted(_AUTOCAST_DTYPES)}"
            )
        estimator = flow.decoder.estimator
        if not isinstance(estimator, torch.nn.Module) and not is_flow_estimator_trt(
            estimator
        ):
            raise RuntimeError(
                "Fun-CosyVoice3 Flow estimator must be a PyTorch module or a "
                "TensorRT wrapper exposing acquire_estimator / execute"
            )
        self._flow = (
            flow if isinstance(flow, FunCosyVoice3Flow) else FunCosyVoice3Flow(flow)
        )
        self._hift = hift
        self._compute_dtype = compute_dtype
        self._flow_batch_bucket_frames = flow_batch_bucket_frames
        self._flow_batch_coalesce_span_frames = flow_batch_coalesce_span_frames
        self._flow_batch_coalesce_max_added_padding_pct = (
            flow_batch_coalesce_max_added_padding_pct
        )
        self._hift_compute_dtype = _AUTOCAST_DTYPES[hift_compute_dtype]
        self._hift_max_padding_waste = hift_max_padding_waste
        self._hift_samples_per_mel_frame: int | None = None

    def _mel_stride(self) -> int:
        if self._hift_samples_per_mel_frame is None:
            self._hift_samples_per_mel_frame = _hift_samples_per_mel_frame(self._hift)
        return self._hift_samples_per_mel_frame

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )

        codes = torch.as_tensor(state.audio_codes, dtype=torch.long).reshape(-1)
        return state, codes

    async def decode_batch(
        self, items: list[tuple[FunCosyVoice3State, torch.Tensor]]
    ) -> list[tuple[Any, int]]:
        prepared: list[_PreparedFlowRequest] = []
        buckets: dict[int, list[_PreparedFlowRequest]] = defaultdict(list)
        for index, (state, codes) in enumerate(items):
            flow_input = self._make_flow_input(state, codes)
            total_mel_frames = self._flow_total_mel_frames(flow_input)
            baseline_bucket_key = self._flow_bucket_key_for_total(total_mel_frames)
            request = _PreparedFlowRequest(
                index=index,
                sample_rate=state.sample_rate,
                flow_input=flow_input,
                total_mel_frames=total_mel_frames,
                baseline_bucket_key=baseline_bucket_key,
            )
            prepared.append(request)
            buckets[baseline_bucket_key].append(request)

        results: list[tuple[Any, int] | None] = [None] * len(prepared)
        baseline_bucket_order = tuple(buckets)
        flow_groups = _group_flow_requests(
            buckets,
            coalesce_span_frames=self._flow_batch_coalesce_span_frames,
            coalesce_max_added_padding_pct=(
                self._flow_batch_coalesce_max_added_padding_pct
            ),
        )

        for flow_group in flow_groups:
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=self._compute_dtype,
                enabled=self._compute_dtype is not None,
            ):
                mel_list = self._flow.inference(
                    [request.flow_input for request in flow_group]
                )

                pairs = list(zip(flow_group, mel_list, strict=True))
                # A coalesced Flow result still carries the original atomic
                # bucket key.  Apply the unchanged HiFT policy independently
                # per key, using an explicit baseline order rather than dict
                # iteration to recover provenance.
                pairs_by_baseline_bucket: dict[
                    int, list[tuple[_PreparedFlowRequest, torch.Tensor]]
                ] = defaultdict(list)
                for request, mel in pairs:
                    if request.baseline_bucket_key is None:
                        raise RuntimeError(
                            "Fun-CosyVoice3 request is missing Flow bucket provenance"
                        )
                    pairs_by_baseline_bucket[request.baseline_bucket_key].append(
                        (request, mel)
                    )

                for bucket_key in baseline_bucket_order:
                    original_bucket_pairs = pairs_by_baseline_bucket.get(bucket_key)
                    if not original_bucket_pairs:
                        continue
                    for group in _group_by_padding_waste(
                        original_bucket_pairs, max_waste=self._hift_max_padding_waste
                    ):
                        wavs = self._mel2wav_batch([mel for _, mel in group])
                        for (request, _), wav in zip(group, wavs, strict=True):
                            results[request.index] = (wav, request.sample_rate)

        if any(result is None for result in results):
            raise RuntimeError("Fun-CosyVoice3 vocoder did not decode every request")
        return [cast(tuple[Any, int], result) for result in results]

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self.decode_payloads([payload])
        if len(results) != 1:
            raise RuntimeError(
                f"Fun-CosyVoice3 vocoder returned {len(results)} results for 1 input"
            )
        return results[0]

    def _make_flow_input(
        self,
        state: FunCosyVoice3State,
        codes: torch.Tensor,
    ) -> FlowBatchInput:
        prompt_token = (
            torch.as_tensor(state.flow_prompt_speech_token, dtype=torch.int32).reshape(
                1, -1
            )
            if state.flow_prompt_speech_token is not None
            else torch.zeros(1, 0, dtype=torch.int32)
        )
        prompt_feat = (
            torch.as_tensor(state.flow_prompt_speech_feat).reshape(1, -1, 80)
            if state.flow_prompt_speech_feat is not None
            else torch.zeros(1, 0, 80)
        )
        embedding = (
            torch.as_tensor(state.flow_embedding).reshape(1, -1)
            if state.flow_embedding is not None
            else torch.zeros(1, 192)
        )
        return FlowBatchInput(
            token=codes.reshape(1, -1).to(torch.int32),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )

    def _flow_bucket_key(self, item: FlowBatchInput) -> int:
        return self._flow_bucket_key_for_total(self._flow_total_mel_frames(item))

    def _flow_bucket_key_for_total(self, total_mel: int) -> int:
        return (
            total_mel + self._flow_batch_bucket_frames - 1
        ) // self._flow_batch_bucket_frames

    def _flow_total_mel_frames(self, item: FlowBatchInput) -> int:
        total_tokens = item.prompt_token.shape[1] + item.token.shape[1]
        return total_tokens * self._flow.token_mel_ratio

    def _flow_scheduler_cost(self, payload: StagePayload) -> int:
        state, codes = self.prepare_item(payload)
        total_mel = self._flow_total_mel_frames(self._make_flow_input(state, codes))
        return (
            (total_mel + self._flow_batch_bucket_frames - 1)
            // self._flow_batch_bucket_frames
            * self._flow_batch_bucket_frames
        )

    def _mel2wav(self, tts_mel: torch.Tensor) -> torch.Tensor:
        with self._hift_autocast():
            tts_speech, _ = self._hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech.detach().cpu()

    def _hift_autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=current_platform.device_type,
            dtype=self._hift_compute_dtype,
            enabled=self._hift_compute_dtype is not None,
        )

    def _mel2wav_batch(self, mels: list[torch.Tensor]) -> list[torch.Tensor]:
        if not mels:
            return []
        if len(mels) == 1:
            return [self._mel2wav(mels[0])]
        lengths = [int(mel.shape[2]) for mel in mels]
        longest = max(lengths)
        if min(lengths) == longest:
            padded = torch.cat(mels, dim=0)
        else:
            padded = torch.cat(
                [
                    F.pad(mel, (0, longest - length))
                    for mel, length in zip(mels, lengths)
                ],
                dim=0,
            )
        with self._hift_autocast():
            wav, _ = self._hift.inference(speech_feat=padded, finalize=True)
        wav = wav.detach()
        samples_per_frame = self._mel_stride()
        return [
            wav[index : index + 1, : length * samples_per_frame].cpu()
            for index, length in enumerate(lengths)
        ]

    def store_result(
        self,
        payload: StagePayload,
        state: FunCosyVoice3State,
        wav: Any,
        sample_rate: int,
    ) -> StagePayload:
        if wav is None:
            raise RuntimeError("Fun-CosyVoice3 vocoder did not return audio")
        audio_payload = audio_waveform_payload(wav, source_hint="Fun-CosyVoice3")
        state.audio_samples = None
        state.sample_rate = int(sample_rate)
        state.audio_codes = None

        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_batch_size: int = 16,
    max_batch_wait_ms: int = 30,
    flow_batch_bucket_frames: int = 50,
    flow_batch_admission_frames: int = _DEFAULT_FLOW_BATCH_ADMISSION_FRAMES,
    flow_batch_coalesce_span_frames: int = 0,
    flow_batch_coalesce_max_added_padding_pct: float = 0.0,
    enable_dit_torch_compile: bool = False,
    enable_flow_estimator_trt: bool = False,
    hift_dtype: str = "float32",
    hift_max_padding_waste: float = 1.5,
) -> SimpleScheduler:
    if flow_batch_admission_frames <= 0:
        raise ValueError("flow_batch_admission_frames must be greater than zero")
    _validate_flow_batch_coalescing_config(
        flow_batch_coalesce_span_frames,
        flow_batch_coalesce_max_added_padding_pct,
    )
    if enable_flow_estimator_trt and enable_dit_torch_compile:
        raise ValueError(
            "enable_flow_estimator_trt and enable_dit_torch_compile both "
            "target flow.decoder.estimator; enable only one"
        )
    device = resolve_device_spec(device, gpu_id)
    checkpoint_dir = resolve_checkpoint(model_path)
    if dtype not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported Fun-CosyVoice3 vocoder dtype {dtype!r}; "
            f"expected one of {sorted(_AUTOCAST_DTYPES)}"
        )
    compute_dtype = _AUTOCAST_DTYPES[dtype]
    flow, hift = _load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
        enable_flow_estimator_trt=enable_flow_estimator_trt,
    )
    if enable_dit_torch_compile:
        _compile_dit_backbone(flow, compute_dtype=compute_dtype)

    vocoder = _CosyVoice3Vocoder(
        flow,
        hift,
        compute_dtype=compute_dtype,
        flow_batch_bucket_frames=flow_batch_bucket_frames,
        flow_batch_coalesce_span_frames=flow_batch_coalesce_span_frames,
        flow_batch_coalesce_max_added_padding_pct=(
            flow_batch_coalesce_max_added_padding_pct
        ),
        hift_compute_dtype=hift_dtype,
        hift_max_padding_waste=hift_max_padding_waste,
    )

    return SimpleScheduler(
        vocoder.decode_payload,
        batch_compute_fn=vocoder.decode_payloads,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=vocoder._flow_scheduler_cost,
        max_batch_cost=flow_batch_admission_frames,
    )

# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import logging
from typing import Any

import torch

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

logger = logging.getLogger(__name__)

# note(chenye): Buffered decoder T is even, so the reachable graph domains are
# q16 at T=256..544 and q32 at T=546..1024.
# Unexpected shapes, including odd T, fall back to eager.
_FLOW_CUDA_GRAPH_BUCKETS = tuple(range(256, 545, 16)) + tuple(range(576, 1025, 32))


def _flow_cuda_graph_bucket(decoder_t: int) -> int | None:
    if 256 <= decoder_t <= 544:
        return ((decoder_t + 15) // 16) * 16
    if 546 <= decoder_t <= 1024:
        return ((decoder_t + 31) // 32) * 32
    return None


def _graph_safe_nonstreaming_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
) -> torch.Tensor:
    del (
        xs,
        use_dynamic_left_chunk,
        decoding_chunk_size,
        num_decoding_left_chunks,
        enable_full_context,
    )
    if use_dynamic_chunk or static_chunk_size != 0:
        raise RuntimeError(
            "Fun-CosyVoice3 CUDA graph mask is only valid for buffered Flow"
        )
    chunk_masks = masks
    empty_rows = chunk_masks.sum(dim=-1, keepdim=True) == 0
    return chunk_masks.masked_fill(empty_rows, True)


class _CosyVoice3FlowCudaGraphRunner:
    """Startup captured CUDA graphs for the buffered 10 step Flow decoder."""

    def __init__(
        self,
        decoder: Any,
        *,
        device: torch.device,
        warmup_iters: int = 1,
    ) -> None:
        self._decoder = decoder
        self._eager_forward = decoder.forward
        self._device = device
        self._warmup_iters = int(warmup_iters)
        self._graphs: dict[int, tuple[Any, dict[str, torch.Tensor], torch.Tensor]] = {}
        self._graph_rand_noise: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        decoder: Any,
        *,
        device: torch.device,
    ) -> _CosyVoice3FlowCudaGraphRunner | None:
        device = torch.device(device)
        if device.type != "cuda":
            return None

        runner = cls(decoder, device=device)
        try:
            runner._capture_all()
            decoder.forward = runner.forward
        except Exception as exc:
            for graph, _, _ in runner._graphs.values():
                reset = getattr(graph, "reset", None)
                if reset is not None:
                    reset()
            runner._graphs.clear()
            runner._graph_rand_noise = None
            logger.warning(
                "Fun-CosyVoice3 Flow CUDA graph initialization failed; "
                "using eager decoder (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return None
        logger.info(
            "Fun-CosyVoice3 Flow CUDA graphs initialized: %d buckets",
            len(runner._graphs),
        )
        return runner

    @staticmethod
    def _call_decoder(
        decoder_forward: Any,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Any]:
        return decoder_forward(
            mu=inputs["mu"],
            mask=inputs["mask"],
            n_timesteps=10,
            spks=inputs["spks"],
            cond=inputs["cond"],
            streaming=False,
        )

    def _new_inputs(self, bucket_t: int) -> dict[str, torch.Tensor]:
        return {
            "mu": torch.zeros((1, 80, bucket_t), device=self._device),
            "mask": torch.ones((1, 1, bucket_t), device=self._device),
            "spks": torch.zeros((1, 80), device=self._device),
            "cond": torch.zeros((1, 80, bucket_t), device=self._device),
        }

    def _capture_one(
        self,
        *,
        bucket_t: int,
        capture_stream: Any,
        dit_module: Any,
    ) -> tuple[Any, dict[str, torch.Tensor], torch.Tensor]:
        inputs = self._new_inputs(bucket_t)
        current_stream = torch.cuda.current_stream(self._device)
        original_mask = dit_module.add_optional_chunk_mask
        graph = None
        try:
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream), torch.inference_mode():
                for _ in range(self._warmup_iters):
                    self._call_decoder(self._eager_forward, inputs)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(self._device)

            capture_stream.wait_stream(current_stream)
            graph = torch.cuda.CUDAGraph()
            dit_module.add_optional_chunk_mask = _graph_safe_nonstreaming_mask
            try:
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(
                        graph,
                        stream=capture_stream,
                        capture_error_mode="thread_local",
                    ),
                ):
                    output, _ = self._call_decoder(self._eager_forward, inputs)
            finally:
                dit_module.add_optional_chunk_mask = original_mask
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(self._device)

            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Flow decoder graph returned a non-tensor output")
            if output.shape != (1, 80, bucket_t):
                raise RuntimeError(
                    "unexpected Flow decoder graph output shape: "
                    f"{tuple(output.shape)} for T={bucket_t}"
                )
            return graph, inputs, output
        except Exception:
            if graph is not None:
                reset = getattr(graph, "reset", None)
                if reset is not None:
                    reset()
            raise

    def _capture_all(self) -> None:
        rand_noise = getattr(self._decoder, "rand_noise", None)
        if not isinstance(rand_noise, torch.Tensor):
            raise RuntimeError("expected CausalConditionalCFM.rand_noise tensor")
        if rand_noise.device.type != "cpu":
            raise RuntimeError("expected the official CFM rand_noise tensor on CPU")
        if rand_noise.dtype != torch.float32:
            raise RuntimeError("expected the official CFM rand_noise tensor in FP32")
        if rand_noise.ndim != 3 or tuple(rand_noise.shape[:2]) != (1, 80):
            raise RuntimeError(
                f"unexpected CFM rand_noise shape: {tuple(rand_noise.shape)}"
            )
        max_bucket_t = _FLOW_CUDA_GRAPH_BUCKETS[-1]
        if rand_noise.shape[2] < max_bucket_t:
            raise RuntimeError(
                "CFM rand_noise is shorter than the largest Flow graph bucket: "
                f"noise={rand_noise.shape[2]} bucket={max_bucket_t}"
            )

        import cosyvoice.flow.DiT.dit as dit_module

        noise_cpu = rand_noise[:, :, :max_bucket_t].detach()
        self._graph_rand_noise = noise_cpu.to(device=self._device).contiguous()
        if not torch.equal(self._graph_rand_noise.cpu(), noise_cpu):
            raise RuntimeError("pre-staged CFM rand_noise changed deterministic values")

        with torch.cuda.device(self._device):
            capture_stream = torch.cuda.Stream(device=self._device)
            try:
                self._decoder.rand_noise = self._graph_rand_noise
                for bucket_t in _FLOW_CUDA_GRAPH_BUCKETS:
                    self._graphs[bucket_t] = self._capture_one(
                        bucket_t=bucket_t,
                        capture_stream=capture_stream,
                        dit_module=dit_module,
                    )
            finally:
                self._decoder.rand_noise = rand_noise

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if (
            args
            or set(kwargs).difference(
                {
                    "mu",
                    "mask",
                    "n_timesteps",
                    "spks",
                    "cond",
                    "streaming",
                    "temperature",
                }
            )
            or kwargs.get("n_timesteps") != 10
            or kwargs.get("streaming", False)
        ):
            return self._eager_forward(*args, **kwargs)
        if kwargs.get("temperature", 1.0) != 1.0:
            return self._eager_forward(*args, **kwargs)

        mu = kwargs.get("mu")
        mask = kwargs.get("mask")
        spks = kwargs.get("spks")
        cond = kwargs.get("cond")
        if not all(isinstance(value, torch.Tensor) for value in (mu, mask, spks, cond)):
            return self._eager_forward(*args, **kwargs)

        actual_t = int(mu.shape[-1]) if mu.ndim else 0
        bucket_t = _flow_cuda_graph_bucket(actual_t)
        entry = self._graphs.get(bucket_t) if bucket_t is not None else None
        if entry is None:
            return self._eager_forward(*args, **kwargs)

        graph, static, output = entry
        if (
            any(value.device != self._device for value in (mu, mask, spks, cond))
            or mu.shape != (1, 80, actual_t)
            or mask.shape != (1, 1, actual_t)
            or spks.shape != (1, 80)
            or cond.shape != (1, 80, actual_t)
            or any(value.dtype != torch.float32 for value in (mu, mask, spks, cond))
        ):
            return self._eager_forward(*args, **kwargs)

        static["mu"].zero_()
        static["mu"][..., :actual_t].copy_(mu)
        static["mask"].zero_()
        static["mask"][..., :actual_t].copy_(mask)
        static["spks"].copy_(spks)
        static["cond"].zero_()
        static["cond"][..., :actual_t].copy_(cond)
        graph.replay()
        return output[..., :actual_t], None


_COSYVOICE_INSTALL_HINT = (
    "Fun-CosyVoice3 support requires the `cosyvoice` package. "
    "Clone the official repository and set PYTHONPATH, or install it "
    "in the serving environment before launching Fun-CosyVoice3."
)


def load_state(payload: StagePayload) -> FunCosyVoice3State:
    return _load_pipeline_state(payload, FunCosyVoice3State)


def store_state(payload: StagePayload, state: FunCosyVoice3State) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _load_cosyvoice3_flow_hift(
    checkpoint_dir: str,
    device: str,
    fp16: bool = False,
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
    del cv.model.llm
    return flow, hift


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    del model_path
    # note(chenye): Reference conditioning supports concurrent calls;
    # model prompt finalization is serialized.
    return SimpleScheduler(
        preprocess_cosyvoice3_payload,
        max_concurrency=4,
        abort_callback=cleanup_prepared_cosyvoice3_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.fun_cosyvoice3.engine_builder import (
        FunCosyVoice3EngineBuilder,
    )

    return FunCosyVoice3EngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


class _CosyVoice3Vocoder(BatchVocoderBase):
    def __init__(
        self,
        flow: Any,
        hift: Any,
        fp16: bool = False,
    ) -> None:
        self._flow = flow
        self._hift = hift
        self._fp16 = fp16
        decoder = getattr(flow, "decoder", None)
        flow_parameter = next(flow.parameters(), None)
        if not fp16 and decoder is not None and flow_parameter is not None:
            _CosyVoice3FlowCudaGraphRunner.build(
                decoder,
                device=flow_parameter.device,
            )

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )
        # The AR runner stores one-element tensors per step, which serialize as
        # ``[num_tokens, 1]``. Flow consumes one unbatched token sequence here.
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long).reshape(-1)
        return state, codes

    async def decode_batch(
        self, items: list[tuple[FunCosyVoice3State, torch.Tensor]]
    ) -> list[tuple[Any, int]]:
        results = []
        for state, codes in items:
            prompt_token = (
                torch.as_tensor(state.flow_prompt_speech_token, dtype=torch.int32)
                if state.flow_prompt_speech_token is not None
                else torch.zeros(1, 0, dtype=torch.int32)
            )
            prompt_feat = (
                torch.as_tensor(state.flow_prompt_speech_feat)
                if state.flow_prompt_speech_feat is not None
                else torch.zeros(1, 0, 80)
            )
            embedding = (
                torch.as_tensor(state.flow_embedding)
                if state.flow_embedding is not None
                else torch.zeros(1, 192)
            )
            wav = self._token2wav(
                token=codes.unsqueeze(0),
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
            )
            results.append((wav, state.sample_rate))
        return results

    def _token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if token.shape[1] == 0:
            raise RuntimeError(
                "Fun-CosyVoice3 generation produced no usable speech tokens"
            )
        device = next(self._flow.parameters()).device

        with torch.autocast(
            device_type=current_platform.device_type, enabled=self._fp16
        ):
            tts_mel, _ = self._flow.inference(
                token=token.to(device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(device),
                prompt_token=prompt_token.to(device),
                prompt_token_len=torch.tensor(
                    [prompt_token.shape[1]], dtype=torch.int32
                ).to(device),
                prompt_feat=prompt_feat.to(device),
                prompt_feat_len=torch.tensor(
                    [prompt_feat.shape[1]], dtype=torch.int32
                ).to(device),
                embedding=embedding.to(device),
                streaming=False,
                finalize=True,
            )
        tts_speech, _ = self._hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech.detach().cpu()

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
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    device = resolve_device_spec(device, gpu_id)
    checkpoint_dir = resolve_checkpoint(model_path)
    flow, hift = _load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
    )

    return _CosyVoice3Vocoder(flow, hift, fp16=(dtype == "float16")).build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )

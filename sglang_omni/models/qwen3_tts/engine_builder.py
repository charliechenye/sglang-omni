# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import numpy.typing as npt
import torch
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.model_executor.cuda_graph_config import Phase, with_phase
from sglang.srt.runtime_context import (
    get_exec,
    get_model,
    get_parallel,
    get_platform,
    get_schedule,
)
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.qwen3_tts import CAPABILITIES, request_builders
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.config import (
    QWEN3_TTS_MAX_QUEUED_REQUESTS,
    QWEN3_TTS_MAX_RUNNING_REQUESTS,
    load_qwen3_tts_checkpoint_config,
    normalize_qwen3_tts_model_type,
)
from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
    DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES,
)
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.models.qwen3_tts.speculative_draft import (
    P16DraftRuntime,
    load_p16_draft_checkpoint,
)
from sglang_omni.models.qwen3_tts.speculative_telemetry import P16WorkerTelemetry
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    nested_prefill_overrides,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import StreamOutputBuilder
from sglang_omni.utils.predictor_speculation import speculative_resources_requested
from sglang_omni.vendor.sglang.server_args import override_server_args

if current_platform.is_npu():
    import triton
    from sgl_kernel_npu.mem_cache import allocator as allocator_kernels
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.layers.sampler import multinomial_with_seed
else:
    pass

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel

    from sglang_omni.models.qwen3_tts.prompt_frontend import Qwen3TTSPromptFrontend
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass


logger = logging.getLogger(__name__)

P16_ENABLE_ENV = "SGLANG_OMNI_QTTS_P16_ENABLE"
P16_DRAFT_PATH_ENV = "SGLANG_OMNI_QTTS_P16_DRAFT_PATH"
P16_DRAFT_SHA256_ENV = "SGLANG_OMNI_QTTS_P16_DRAFT_SHA256"
P16_MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
P16_TALKER_HIDDEN_SIZE = 2048
P16_PREDICTOR_HIDDEN_SIZE = 1024
P16_NUM_CODE_GROUPS = 16
P16_VOCAB_SIZE = 2048


def p16_model_fallback_reason(
    *,
    model_type: str,
    model_size: str,
    checkpoint_dir: str | Path,
    talker_hidden_size: int,
    predictor_hidden_size: int,
    num_code_groups: int,
    vocab_size: int,
) -> str | None:
    """Check the pinned P16 model shape without touching device state."""
    if (
        model_type != "custom_voice"
        or model_size != "1b7"
        or P16_MODEL_REVISION not in Path(checkpoint_dir).resolve().parts
        or talker_hidden_size != P16_TALKER_HIDDEN_SIZE
        or predictor_hidden_size != P16_PREDICTOR_HIDDEN_SIZE
        or num_code_groups != P16_NUM_CODE_GROUPS
        or vocab_size != P16_VOCAB_SIZE
    ):
        return "unsupported-model"
    else:
        return None


def is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    else:
        pass
    if isinstance(value, int):
        return value != 0
    else:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        pass
    return False


def prepare_npu_kernels(
    allocator: BaseTokenToKVPoolAllocator, max_running_requests: int, vocab_size: int
) -> None:
    """Warm allocation and seeded sampling with the resolved pool and vocabulary."""
    if not isinstance(allocator, NPUPagedTokenToKVPoolAllocator):
        return
    else:
        pass
    assert max_running_requests > 0
    assert vocab_size > 0
    started = time.perf_counter()
    kernel = allocator_kernels.alloc_extend_kernel
    extend_parameter = kernel.arg_names.index("max_num_extend_tokens")
    if extend_parameter in kernel.constexprs:
        original_function = kernel.fn
        function = FunctionType(
            original_function.__code__,
            original_function.__globals__,
            original_function.__name__,
            original_function.__defaults__,
            original_function.__closure__,
        )
        # Clone annotations so the dependency function is not mutated.
        function.__annotations__ = dict(original_function.__annotations__)
        function.__annotations__.pop("max_num_extend_tokens")
        kernel = triton.jit(
            function,
            do_not_specialize=["free_page_ptr", "max_num_extend_tokens"],
        )
        allocator_kernels.alloc_extend_kernel = kernel
    else:
        pass

    device = allocator.free_pages.device
    page_size = allocator.page_size
    batch_buckets = [
        1 << exponent for exponent in range((max_running_requests - 1).bit_length() + 1)
    ]
    for batch_size in batch_buckets:
        prefix_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        sequence_lengths = torch.full_like(prefix_lengths, page_size + 1)
        last_positions = torch.full_like(prefix_lengths, -1)
        free_pages = torch.arange(
            1, 2 * batch_size + 1, dtype=allocator.free_pages.dtype, device=device
        )
        output = torch.empty(
            batch_size * (page_size + 1), dtype=torch.int64, device=device
        )
        kernel[(batch_size,)](
            prefix_lengths,
            sequence_lengths,
            last_positions,
            free_pages,
            output,
            batch_size,
            page_size,
            triton.next_power_of_2(output.numel()),
        )
        multinomial_with_seed(
            torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device),
            torch.zeros(batch_size, dtype=torch.int64, device=device),
            torch.zeros(batch_size, dtype=torch.int64, device=device),
        )
    torch.get_device_module(device).synchronize(device)
    logger.info(
        f"Qwen3-TTS NPU kernels prepared: page_size={page_size}, vocab_size={vocab_size}, "
        f"batch_buckets={batch_buckets}, elapsed_s={time.perf_counter() - started:.3f}"
    )


# note (luojiaxuan): on 1.7B and 0.6B Base the share of generated leading silence these probes cover
# saturates at a -50 dBFS ceiling, and no speech frame falls in the set up to -40 dBFS.
SILENCE_PROBE_FLOOR_DBFS = -90
SILENCE_PROBE_CEILING_DBFS = -50
SILENCE_PROBE_STEP_DB = 5
SILENCE_PROBE_SECONDS = 8.0
# note (luojiaxuan): white, pink and brown noise; recorded room tone is not white.
SILENCE_PROBE_SPECTRAL_EXPONENTS = (0.0, 0.5, 1.0)


def colored_noise(
    spectral_exponent: float, num_samples: int, generator: np.random.Generator
) -> np.ndarray:
    """Unit-RMS noise whose amplitude spectrum falls as frequency ** -spectral_exponent."""
    spectrum = np.fft.rfft(generator.standard_normal(num_samples))
    frequencies = np.fft.rfftfreq(num_samples)
    frequencies[0] = frequencies[1]
    noise = np.fft.irfft(spectrum / frequencies**spectral_exponent, n=num_samples)
    return (noise / np.sqrt(np.mean(noise**2))).astype(np.float32)


class SpeechTokenizerCodes(Protocol):
    @property
    def audio_codes(self) -> list[torch.Tensor]: ...


class SilenceProbeSpeechTokenizer(Protocol):
    def get_input_sample_rate(self) -> int: ...

    def encode(
        self, audios: list[npt.NDArray[np.float32]], /, *, sr: int
    ) -> SpeechTokenizerCodes: ...


def derive_silence_codec_ids(
    speech_tokenizer: SilenceProbeSpeechTokenizer, device: str
) -> torch.Tensor:
    """Codebook-0 ids the checkpoint's own codec assigns to stationary noise up to the ceiling."""
    sample_rate = speech_tokenizer.get_input_sample_rate()
    generator = np.random.default_rng(0)
    shapes = [
        colored_noise(exponent, int(SILENCE_PROBE_SECONDS * sample_rate), generator)
        for exponent in SILENCE_PROBE_SPECTRAL_EXPONENTS
    ]
    levels_dbfs = (
        float("-inf"),
        *range(
            SILENCE_PROBE_FLOOR_DBFS,
            SILENCE_PROBE_CEILING_DBFS + 1,
            SILENCE_PROBE_STEP_DB,
        ),
    )
    waveforms = [
        np.float32(10 ** (level / 20)) * shape
        for level in levels_dbfs
        for shape in shapes
    ]
    codes = speech_tokenizer.encode(waveforms, sr=sample_rate).audio_codes
    return torch.unique(torch.cat([code[:, 0] for code in codes])).to(device)


class Qwen3TtsEngineBuilder(TtsEngineBuilder[Qwen3TTSSGLangRequestData]):
    model_name = "Qwen3-TTS"
    context_length = 8192
    model_arch_override = "Qwen3TTSTalker"
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )
    supports_full_prefill_cuda_graph = CAPABILITIES.supports_full_prefill_cuda_graph

    def __init__(
        self,
        *,
        attn_implementation: str | None = None,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
        code_predictor_spec_verify_size: int = 0,
        code_predictor_spec_backend: Literal[
            "cuda_graph", "eager", "full_round_graph"
        ] = "cuda_graph",
        code_predictor_spec_probability_backend: Literal[
            "split_top_k", "radix_compact", "radix_compact_4bit"
        ] = "split_top_k",
        code_predictor_spec_lm_head_backend: Literal[
            "grouped_triton", "warp_gemv", "warp_accumulate"
        ] = "grouped_triton",
        code_predictor_spec_rng_backend: Literal[
            "per_round", "frame_pool"
        ] = "per_round",
        code_predictor_spec_acceptance_backend: Literal[
            "compact", "dense_indexed"
        ] = "compact",
        code_predictor_spec_opening_pair_graph: bool = False,
        reference_encoder_cuda_graph_bucket_frames: Sequence[int] = (
            DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
        ),
        leading_silence_mask_frames: int = (
            qwen3_stages.DEFAULT_LEADING_SILENCE_MASK_FRAMES
        ),
    ) -> None:
        speculative_resources_requested(code_predictor_spec_verify_size)
        self.code_predictor_spec_verify_size = code_predictor_spec_verify_size
        self.code_predictor_spec_backend = code_predictor_spec_backend
        self.code_predictor_spec_probability_backend = (
            code_predictor_spec_probability_backend
        )
        self.code_predictor_spec_lm_head_backend = code_predictor_spec_lm_head_backend
        self.code_predictor_spec_rng_backend = code_predictor_spec_rng_backend
        self.code_predictor_spec_acceptance_backend = (
            code_predictor_spec_acceptance_backend
        )
        self.code_predictor_spec_opening_pair_graph = (
            code_predictor_spec_opening_pair_graph
        )
        self.attn_implementation = attn_implementation
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.reference_encoder_cuda_graph_bucket_frames = tuple(
            reference_encoder_cuda_graph_bucket_frames
        )
        self.leading_silence_mask_frames = leading_silence_mask_frames
        self.silence_codec_ids: torch.Tensor | None = None
        self.wrapper: Qwen3TTSModel | None = None
        self.talker_model: Qwen3TTSTalker | None = None
        self.stream_output_builder: (
            StreamOutputBuilder[Qwen3TTSSGLangRequestData] | None
        ) = None
        # note (ratish): the factory assigns this before generation_defaults runs; a builder
        # that never builds reads the defaults of an unknown checkpoint.
        self.checkpoint_dir: str = ""

    def resolve_checkpoint(self, model_path: str) -> str:
        qwen3_stages.apply_qwen_tts_transformers_compatibility_patches()
        qwen_tts = importlib.import_module("qwen_tts")
        if not hasattr(qwen_tts, "Qwen3TTSModel"):
            raise ImportError("qwen_tts does not expose Qwen3TTSModel")
        else:
            pass

        return super().resolve_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        qwen3_stages.apply_qwen_tts_transformers_compatibility_patches()
        qwen3_stages.register_qwen3_tts_hf_config()

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> GenerationDefaults:
        # note(ratish): the decode graph ladder follows the running bound, so it
        # is not set here.
        defaults: GenerationDefaults = {
            "max_running_requests": QWEN3_TTS_MAX_RUNNING_REQUESTS,
            "max_queued_requests": QWEN3_TTS_MAX_QUEUED_REQUESTS,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.85,
            "max_prefill_tokens": 8192,
            "sampling_backend": (
                "flashinfer" if current_platform.is_cuda() else "pytorch"
            ),
            "trust_remote_code": True,
            # note (luojiaxuan): under load prefills coalesce into one extend
            # batch, so the ladder must reach well past a single prompt.
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
        }
        if self.checkpoint_dir:
            configured_model_type = load_qwen3_tts_checkpoint_config(
                self.checkpoint_dir
            ).get("tts_model_type")
            model_type = (
                normalize_qwen3_tts_model_type(configured_model_type)
                if isinstance(configured_model_type, str) and configured_model_type
                else None
            )
        else:
            model_type = None
        if model_type in {"base", "custom_voice"}:
            # note (Zhuofeng-Li): full prefill avoids host launches between layers.
            defaults["cuda_graph_backend_prefill"] = CudaGraphBackend.FULL
        else:
            pass
        return defaults

    def customize_server_args(self, server_args: ServerArgs) -> None:
        cfg = resolved_view(server_args)
        prefill = cfg.cuda_graph_config.prefill
        if (
            prefill.backend not in (CudaGraphBackend.BREAKABLE, CudaGraphBackend.FULL)
            or not prefill.bs
        ):
            return
        else:
            pass
        prefill_fields: dict[str, list[int] | int] = {}
        # note (luojiaxuan): ladders deferred to SGLang can still omit bucket 1 and force cached single-token prefills to run eagerly.
        if (
            prefill.bs[0] != 1
            and self.server_args_overrides.get("cuda_graph_bs_prefill") is None
            and "bs" not in nested_prefill_overrides(self.server_args_overrides)
        ):
            prefill_fields["bs"] = [1, *prefill.bs]
        else:
            pass
        # note (ratish): fa3 runs a split KV combine over every request slot, so only
        # fa4 holds every running request; fa3 keeps SGLang's slot count.
        if (
            prefill.backend == CudaGraphBackend.FULL
            and prefill.full_prefill_max_req is None
            and attention_backends_of(cfg)[0] == "fa4"
        ):
            prefill_fields["full_prefill_max_req"] = cfg.max_running_requests
        else:
            pass
        if prefill_fields:
            override_server_args(
                server_args,
                "qwen3_tts.prefill_cuda_graph_shape",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, **prefill_fields
                ),
            )
        else:
            pass

    def before_memory_pool(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        del gpu_id
        from qwen_tts import Qwen3TTSModel
        from transformers import AutoProcessor

        # note(ratish): the tokenizer and the predictor graphs live for the whole
        # process, so they are attached before sglang reads free memory for the pool.
        model = model_worker.model_runner.model
        self.talker_model = model
        p16_enable_value = os.environ.get(P16_ENABLE_ENV)
        if p16_enable_value not in (None, "0", "1"):
            raise ValueError(f"{P16_ENABLE_ENV} must be 0 or 1")
        else:
            pass
        p16_enabled = p16_enable_value == "1"
        resolved_checkpoint_path = Path(checkpoint_dir).resolve()
        telemetry_path = os.environ.get("SGLANG_OMNI_QTTS_P16_TELEMETRY_PATH")
        p16_static_fallback_reason: str | None = None
        if p16_enabled or telemetry_path is not None:
            p16_static_fallback_reason = p16_model_fallback_reason(
                model_type=model.tts_model_type,
                model_size=model.tts_model_size,
                checkpoint_dir=resolved_checkpoint_path,
                talker_hidden_size=int(model.config.hidden_size),
                predictor_hidden_size=int(
                    model.config.code_predictor_config.hidden_size
                ),
                num_code_groups=int(model.config.num_code_groups),
                vocab_size=int(model.config.code_predictor_config.vocab_size),
            )
        else:
            pass
        if p16_static_fallback_reason is None and (
            p16_enabled or telemetry_path is not None
        ):
            if (
                not current_platform.is_cuda()
                or model.predictor_device.type != "cuda"
                or "H200"
                not in torch.cuda.get_device_properties(model.predictor_device).name
                or torch.cuda.get_device_capability(model.predictor_device) != (9, 0)
            ):
                p16_static_fallback_reason = "unsupported-device"
            elif int(get_parallel().tp_size) != 1:
                p16_static_fallback_reason = "unsupported-tensor-parallel-size"
            elif (
                bool(get_exec().graph.disable_cuda_graph)
                or current_platform.get_device_graph_backend(model.predictor_device)
                is None
            ):
                p16_static_fallback_reason = "cuda-graph-unavailable"
            elif model.predictor_cutedsl_layers is not None:
                p16_static_fallback_reason = "unsupported-predictor-backend"
            else:
                pass
        else:
            pass
        model.code_predictor_p16_static_fallback_reason = p16_static_fallback_reason
        draft_loaded = False
        loaded_draft_sha256: str | None = None
        if p16_enabled:
            draft_path = os.environ.get(P16_DRAFT_PATH_ENV)
            draft_sha256 = os.environ.get(P16_DRAFT_SHA256_ENV)
            if not draft_path or not draft_sha256:
                raise ValueError(
                    f"{P16_DRAFT_PATH_ENV} and {P16_DRAFT_SHA256_ENV} are required when P16 is enabled"
                )
            else:
                pass
            draft_checkpoint = load_p16_draft_checkpoint(draft_path, draft_sha256)
            draft_loaded = True
            loaded_draft_sha256 = draft_checkpoint.sha256
            model.code_predictor_p16_draft_sha256 = draft_checkpoint.sha256
            p16_model_config = load_qwen3_tts_checkpoint_config(checkpoint_dir)
            p16_fallback_reason = p16_static_fallback_reason
            if p16_fallback_reason is None and (
                p16_model_config.get("tts_model_type") != "custom_voice"
                or p16_model_config.get("tts_model_size") != "1b7"
            ):
                p16_fallback_reason = "unsupported-model"
            else:
                pass
            if p16_fallback_reason is None:
                model.validate_p16_native_target_projection()
                projected_tables = model.predictor_projected_embeddings
                if projected_tables is None:
                    raise RuntimeError("P16 requires projected codec embedding tables")
                else:
                    pass
                model.code_predictor_p16_draft = P16DraftRuntime.create(
                    draft_checkpoint,
                    projected_tables,
                    model.predictor_device,
                )
                model.code_predictor_p16_enabled = True
                model.configure_code_predictor_speculation(
                    4,
                    backend="full_round_graph",
                    probability_backend="split_top_k",
                    lm_head_backend="warp_accumulate",
                    rng_backend="frame_pool",
                    acceptance_backend="compact",
                    opening_pair_graph=True,
                )
                model.code_predictor_p16_enabled = (
                    model.code_predictor_spec_effective_width == 4
                    and model.code_predictor_spec_fallback_reason is None
                )
                if not model.code_predictor_p16_enabled:
                    model.code_predictor_p16_static_fallback_reason = (
                        model.code_predictor_spec_fallback_reason
                        or "unsupported-predictor-capability"
                    )
                else:
                    pass
            else:
                model.code_predictor_p16_draft = None
                model.code_predictor_p16_enabled = False
                model.configure_code_predictor_speculation(0)
                model.code_predictor_spec_fallback_reason = p16_fallback_reason
                logger.warning(
                    f"P16 enabled but this Qwen3-TTS worker will use native Predictor decoding: {p16_fallback_reason}"
                )
        else:
            model.code_predictor_p16_draft = None
            model.code_predictor_p16_enabled = False
            model.code_predictor_p16_draft_sha256 = None
            model.configure_code_predictor_speculation(0)
        if telemetry_path is not None:
            model.code_predictor_p16_telemetry = P16WorkerTelemetry(
                telemetry_path,
                enabled=p16_enabled,
                draft_sha256=loaded_draft_sha256,
                draft_loaded=draft_loaded,
            )
        else:
            model.code_predictor_p16_telemetry = None
        speech_tokenizer = qwen3_stages.load_qwen3_tts_tokenizer(
            checkpoint_dir,
            device=device,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            float32_encoder=model.tts_model_type == "base",
        )
        model.load_speech_tokenizer(speech_tokenizer)
        # note (luojiaxuan): only Base checkpoints serve x-vector clones, so the
        # other variants skip the probe and the per-step mask.
        if self.leading_silence_mask_frames > 0 and model.tts_model_type == "base":
            # note (ratish): the probe runs this thread's first GEMM on the device, and a cuBLAS workspace made inside its float32 activations keeps them reserved when the KV pool is sized.
            if current_platform.is_cuda():
                torch.cuda.current_blas_handle()
            else:
                pass
            self.silence_codec_ids = derive_silence_codec_ids(speech_tokenizer, device)
            logger.info(
                f"Qwen3-TTS masks {self.silence_codec_ids.numel()} silence codec ids "
                f"for the first {self.leading_silence_mask_frames} frames of "
                f"x-vector-only clones: {self.silence_codec_ids.tolist()}"
            )
        else:
            self.leading_silence_mask_frames = 0
            self.silence_codec_ids = torch.empty(0, dtype=torch.long, device=device)
        processor = AutoProcessor.from_pretrained(
            checkpoint_dir,
            fix_mistral_regex=True,
        )
        self.wrapper = Qwen3TTSModel(
            model=model,
            processor=processor,
            generate_defaults=qwen3_stages.load_qwen3_tts_generate_defaults(
                checkpoint_dir
            ),
        )
        disable_cuda_graph = bool(resolved_view(server_args).disable_cuda_graph)
        request_builders.set_qwen3_tts_preprocessing_context(
            model=model,
            wrapper=self.wrapper,
            device=torch.device(device),
            reference_encoder_graph_bucket_frames=(
                ()
                if disable_cuda_graph
                else self.reference_encoder_cuda_graph_bucket_frames
            ),
        )
        if disable_cuda_graph:
            return
        else:
            pass
        if model.speaker_encoder_graph_runner is not None:
            model.speaker_encoder_graph_runner.capture(
                self.reference_encoder_cuda_graph_bucket_frames,
                speech_tokenizer.model.encode_downsample_rate,
            )
        else:
            pass
        # note(ratish): the bucket warmups also build cuDNN's attention plans,
        # which otherwise land inside the first serving step of each batch size.
        subtalker = request_builders.resolve_subtalker_sampling(
            self.wrapper._merge_generate_kwargs()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        model.capture_predictor_graphs(
            do_sample=subtalker.do_sample,
            top_k=subtalker.top_k,
            top_p=subtalker.top_p,
        )

    def setup_model(
        self,
        *,
        model_worker: object,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: object,
    ) -> None:
        # note(ratish): everything Qwen3-TTS attaches stays resident, so it all
        # runs in before_memory_pool and nothing is left for after the pool.
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if is_truthy(overrides.get("enable_torch_compile", False)):
            raise ValueError("Qwen3-TTS torch.compile is not supported")
        else:
            pass
        if (
            current_platform.is_xpu()
            and self.device.split(":")[0] == "xpu"
            and overrides.get("attention_backend") is None
            and overrides.get("prefill_attention_backend") is None
            and overrides.get("decode_attention_backend") is None
        ):
            overrides["attention_backend"] = "intel_xpu"
        else:
            pass
        # note (ratish): fa4 avoids fa3's per-slot split KV combine but runs FP8 KV only
        # on SM100, so a quantized KV cache keeps the default backend.
        prefill_backend = nested_prefill_overrides(overrides).get(
            "backend", overrides.get("cuda_graph_backend_prefill")
        )
        if (
            prefill_backend == CudaGraphBackend.FULL
            and overrides.get("attention_backend") is None
            and overrides.get("prefill_attention_backend") is None
            and overrides.get("kv_cache_dtype") in (None, "auto")
            and get_platform().is_sm90
        ):
            overrides["prefill_attention_backend"] = "fa4"
        else:
            pass

    def post_scheduler_setup(
        self, scheduler: OmniScheduler[Qwen3TTSSGLangRequestData], model_runner: object
    ) -> None:
        del model_runner
        schedule = get_schedule()
        running = int(schedule.max_running_requests)
        context = int(get_model().context_length)
        if current_platform.is_npu():
            prepare_npu_kernels(
                scheduler.token_to_kv_pool_allocator,
                running,
                scheduler.tp_worker.model_runner.model.vocab_size,
            )
        else:
            pass
        pool = scheduler.tp_worker.model_runner.token_to_kv_pool
        k_bytes, v_bytes = pool.get_kv_size_bytes()
        logger.info(
            "Qwen3-TTS KV pool holds %d tokens, %.2f GiB, against a configured "
            "maximum demand of %d (%d running x %d context), mem_fraction_static %.3f",
            int(scheduler.max_total_num_tokens),
            (int(k_bytes) + int(v_bytes)) / float(1 << 30),
            running * context,
            running,
            context,
            float(schedule.mem_fraction_static),
        )

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> Qwen3TTSModelRunner:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.qwen3_tts.model_runner"
        )

        return model_runner_mod.Qwen3TTSModelRunner(
            model_worker,
            output_proc,
            leading_silence_mask_frames=self.leading_silence_mask_frames,
            silence_codec_ids=self.silence_codec_ids,
        )

    def make_adapters(
        self, model: Qwen3TTSTalker | Qwen3TTSPromptFrontend | None
    ) -> tuple[
        Callable[[StagePayload], Qwen3TTSSGLangRequestData],
        Callable[[Qwen3TTSSGLangRequestData], StagePayload],
    ]:
        request_builder, result_adapter, self.stream_output_builder = (
            request_builders.make_qwen3_tts_scheduler_adapters(
                model=model,
                wrapper=self.wrapper,
            )
        )
        return request_builder, result_adapter

    def extra_scheduler_kwargs(self) -> SchedulerExtras[Qwen3TTSSGLangRequestData]:
        # note (ratish): requests build on the scheduler thread: preprocessing already did the
        # model work, and a pooled build that misses the pass waits a whole engine step for admission.
        # note (ratish): on the default stream the talker's steps run in order with the vocoder's whole-utterance decodes, so the talker takes a stream of its own, above the vocoder's.
        return {
            "stream_output_builder": self.stream_output_builder,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "should_run_top_priority_stream": current_platform.is_cuda(),
        }

    def make_abort_callback(self) -> Callable[[str], None]:
        def abort_request(request_id: str) -> None:
            request_builders.cleanup_prepared_qwen3_tts_request(request_id)
            talker_model = self.talker_model
            if talker_model is not None:
                telemetry = talker_model.code_predictor_p16_telemetry
                if telemetry is not None:
                    telemetry.discard_request(request_id)
                else:
                    pass
            else:
                pass

        return abort_request

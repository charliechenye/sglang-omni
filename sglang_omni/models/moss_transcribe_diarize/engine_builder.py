# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize SGLang engine builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache

from sglang_omni.models.moss_transcribe_diarize import CAPABILITIES, request_builders
from sglang_omni.models.moss_transcribe_diarize.encoder_service import (
    BatchedAudioEncoderService,
)
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    CudaGraphConfig,
    build_default_prefill_cuda_graph_bs,
)


def _force_calibration_server_args_overrides(
    server_args_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Make every decoder CUDA-graph/compile input eager for calibration.

    SGLang gives nested ``cuda_graph_config`` precedence over convenience
    flags, while explicit convenience flags can in turn override the legacy
    ``disable_cuda_graph`` switch.  Calibration owns all of those knobs so an
    operator's normal-serving settings cannot re-enable a graph around the
    collector.
    """

    normalized = dict(server_args_overrides or {})
    normalized["disable_cuda_graph"] = True
    normalized["enable_torch_compile"] = False
    normalized["cuda_graph_backend_decode"] = CudaGraphBackend.DISABLED
    normalized["cuda_graph_backend_prefill"] = CudaGraphBackend.DISABLED

    config = normalized.get("cuda_graph_config")
    if isinstance(config, CudaGraphConfig):
        config = config.to_dict()
    elif config is None:
        config = {}
    elif not isinstance(config, Mapping):
        raise ValueError(
            "MOSS-TD KV calibration requires cuda_graph_config to be a mapping"
        )
    else:
        config = dict(config)

    for phase in ("decode", "prefill"):
        phase_config = config.get(phase)
        if phase_config is None:
            phase_config = {}
        elif not isinstance(phase_config, Mapping):
            raise ValueError(
                "MOSS-TD KV calibration requires each cuda_graph_config phase "
                "to be a mapping"
            )
        else:
            phase_config = dict(phase_config)
        phase_config["backend"] = CudaGraphBackend.DISABLED
        config[phase] = phase_config
    normalized["cuda_graph_config"] = config
    return normalized


class MossTranscribeDiarizeEngineBuilder(AsrEngineBuilder):
    model_name = "MOSS-Transcribe-Diarize"
    model_arch_override = "MossTranscribeDiarizeForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int | None,
        context_length: int | None,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        encoder_cache_size_bytes: int,
        enable_torch_compile: bool,
        torch_compile_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
        prefill_coalesce_when_idle: bool,
        prefill_coalesce_requires_pending_builds: bool,
        prefill_coalesce_after_builds_during_decode: bool,
        encoder_chunk_buckets: list[int],
        encoder_torch_compile: bool,
        encoder_max_batch_size: int,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        stream_emit_interval_s: float,
        kv_calibration_output_path: str | None = None,
        kv_calibration_checkpoint_interval_s: float = 30.0,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.requested_max_new_tokens = max_new_tokens
        self.requested_context_length = context_length
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.encoder_cache_size_bytes = encoder_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.torch_compile_max_bs = torch_compile_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.prefill_coalesce_when_idle = prefill_coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = (
            prefill_coalesce_requires_pending_builds
        )
        self.prefill_coalesce_after_builds_during_decode = (
            prefill_coalesce_after_builds_during_decode
        )
        self.encoder_chunk_buckets = encoder_chunk_buckets
        self.encoder_torch_compile = encoder_torch_compile
        self.encoder_max_batch_size = encoder_max_batch_size
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.stream_emit_interval_s = stream_emit_interval_s
        self.kv_calibration_output_path = kv_calibration_output_path
        self.kv_calibration_checkpoint_interval_s = (
            kv_calibration_checkpoint_interval_s
        )
        self.processor: Any = None
        self.tokenizer: Any = None
        self.audio_encoder_service: BatchedAudioEncoderService | None = None
        self.kv_calibration_collector: Any = None
        self.max_new_tokens = 0
        self.context_length = 0

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoProcessor

        from sglang_omni.models.moss_transcribe_diarize import stages

        with stages._missing_additional_chat_templates_compat():
            self.processor = AutoProcessor.from_pretrained(
                checkpoint_dir, trust_remote_code=True
            )
        self.tokenizer = self.processor.tokenizer
        self.max_new_tokens = (
            int(self.requested_max_new_tokens)
            if self.requested_max_new_tokens is not None
            else stages._default_max_new_tokens(checkpoint_dir)
        )
        self.context_length = (
            int(self.requested_context_length)
            if self.requested_context_length is not None
            else stages._default_context_length(checkpoint_dir)
        )

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        # note (Xinyu): cached-prefix extends commonly contain one or two new
        # tokens, so keep exact graph buckets below the shared ladder's 4-token
        # floor instead of failing the prefill padding-factor replay guard.
        prefill_cuda_graph_bs = [
            1,
            2,
            *build_default_prefill_cuda_graph_bs(4096),
        ]
        defaults = {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": self.kv_calibration_output_path is not None,
            "disable_overlap_schedule": True,
            "enable_torch_compile": (
                False
                if self.kv_calibration_output_path is not None
                else self.enable_torch_compile
            ),
            "torch_compile_max_bs": self.torch_compile_max_bs,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            "cuda_graph_backend_prefill": (
                CudaGraphBackend.DISABLED
                if self.kv_calibration_output_path is not None
                else CudaGraphBackend.BREAKABLE
            ),
            "cuda_graph_bs_prefill": prefill_cuda_graph_bs,
            "dtype": dtype,
        }
        if self.kv_calibration_output_path is not None:
            defaults.update(_force_calibration_server_args_overrides(None))
        return defaults

    def prepare_server_args_overrides(
        self,
        server_args_overrides: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if self.kv_calibration_output_path is None:
            return server_args_overrides
        return _force_calibration_server_args_overrides(server_args_overrides)

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # note (Dayuxiaoshui): context_length is an explicit server-args
        # parameter, so consume the operator override before the shared builder
        # expands overrides.
        if self.kv_calibration_output_path is not None:
            # Keep this guard at the final builder merge point as well as in the
            # pre-merge hook: direct builder users must not accidentally capture
            # a graph or compile away the hooks.
            overrides.update(_force_calibration_server_args_overrides(overrides))
        if "context_length" in overrides:
            self.context_length = int(overrides.pop("context_length"))

    def customize_server_args(self, server_args: Any) -> None:
        # note (Dayuxiaoshui): adapters must use the context length finalized by
        # ServerArgs, matching the pre-refactor factory behavior.
        self.context_length = int(server_args.context_length)

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        if self.kv_calibration_output_path is not None:
            if generation_cuda_graph_enabled or bool(
                getattr(server_args, "enable_torch_compile", False)
            ):
                raise RuntimeError(
                    "MOSS-TD KV calibration requires CUDA graphs and decoder "
                    "torch.compile to be disabled"
                )
            from sglang_omni.models.moss_transcribe_diarize.kv_calibration import (
                attach_moss_td_kv_calibration,
            )

            self.kv_calibration_collector = attach_moss_td_kv_calibration(
                model,
                model_path=self.checkpoint_dir,
                output_path=self.kv_calibration_output_path,
                checkpoint_interval_s=self.kv_calibration_checkpoint_interval_s,
            )
        input_feature_len = int(self.processor.feature_extractor.nb_max_frames)
        if self.encoder_torch_compile:
            model.compile_encoder(self.encoder_chunk_buckets, input_feature_len)
        elif generation_cuda_graph_enabled:
            model.init_encoder_graphs(self.encoder_chunk_buckets, input_feature_len)
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        model.init_encoder_cache(self.encoder_cache_size_bytes)

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        del server_args
        self.audio_encoder_service = BatchedAudioEncoderService(
            model,
            max_batch_size=self.encoder_max_batch_size,
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_moss_transcribe_diarize_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
            duration_scaled_default=self.requested_max_new_tokens is None,
            audio_encoder_service=self.audio_encoder_service,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "stream_output_builder": (
                request_builders.make_moss_transcribe_diarize_stream_output_builder(
                    tokenizer=self.tokenizer,
                    min_emit_interval_s=self.stream_emit_interval_s,
                )
            ),
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "prefill_coalesce_when_idle": self.prefill_coalesce_when_idle,
            "prefill_coalesce_requires_pending_builds": (
                self.prefill_coalesce_requires_pending_builds
            ),
            "prefill_coalesce_after_builds_during_decode": (
                self.prefill_coalesce_after_builds_during_decode
            ),
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
        }

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        if self.kv_calibration_collector is None:
            return {}
        # Finalization belongs after Stage joins the scheduler thread.  The
        # regular shutdown callback runs from OmniScheduler.stop(), before
        # model execution has necessarily quiesced.
        return {
            "post_quiescence_callback": self.kv_calibration_collector.finalize
        }

    def cleanup_build_failure(self) -> None:
        if self.kv_calibration_collector is not None:
            self.kv_calibration_collector.abort()
            self.kv_calibration_collector = None

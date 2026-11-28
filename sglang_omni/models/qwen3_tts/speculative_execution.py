# SPDX-License-Identifier: Apache-2.0
"""Fixed P16 full-round execution for the Qwen3-TTS CodePredictor."""

# P15A measured radix compact, while this production path intentionally keeps
# split Top-K for its simpler supported implementation. P15A latency does not
# transfer to P16; this path needs its own H200 performance measurement.

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Literal, TypeAlias, TypedDict

import torch
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils.common import is_pin_memory_available

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    TOP_K_CHUNKS,
    build_speculative_probability_candidates,
    fused_raw_logit_block_k,
    uses_split_top_k,
)
from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    SpeculativeAcceptanceBatch,
    accept_speculative_candidates_compact,
    sample_compact_probability_rows_fused,
)
from sglang_omni.models.qwen3_tts.speculative_commit import (
    commit_speculative_codes,
    commit_speculative_state,
)
from sglang_omni.models.qwen3_tts.speculative_prepare import (
    SpeculativeVerifierInputs,
    prepare_speculative_round_metadata,
    prepare_speculative_verification_inputs,
)
from sglang_omni.models.qwen3_tts.speculative_rng import speculative_uniforms_frame_pool
from sglang_omni.models.qwen3_tts.speculative_runtime import (
    is_fatal_cuda_capture_error,
    speculative_opening_code_output_view,
    speculative_sampling_fallback_reason,
)
from sglang_omni.models.qwen3_tts.speculative_workspace import (
    PredictorGraphSignature,
    PredictorSpeculativeWorkspace,
    compact_candidate_view,
    compact_scratch_view,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphPool, ReplayableGraph
from sglang_omni.utils.predictor_grouped_heads import (
    grouped_lm_head_warp_accumulate,
    grouped_lm_head_warp_accumulate_available,
)
from sglang_omni.utils.predictor_layers import resolve_fused_predictor_width

logger = logging.getLogger("sglang_omni.models.qwen3_tts.sglang_model")

_P16_VERIFY_WIDTH = 4
_P16_TOP_K = 50
_PREDICTOR_SPECULATIVE_GRAPH_MAX_KEYS = 14
_SPECULATIVE_WORKSPACE_MAX_TOP_K = _P16_TOP_K
_PREDICTOR_GRAPH_WARMUP_PASSES = 2

PredictorSpeculativeOpeningGraphKey: TypeAlias = tuple[int, PredictorGraphSignature]
SpeculativeGraphCaptureOutputs: TypeAlias = tuple[str | None, str, str] | None
PredictorSpeculativeAuditObserver: TypeAlias = Callable[
    [Literal["opening", "round"], dict[str, torch.Tensor | int]], None
]
SpeculativeGraphCaptureOperation: TypeAlias = Callable[
    [], SpeculativeGraphCaptureOutputs
]


class PredictorSpeculativeStats(TypedDict):
    requested_width: int
    effective_width: int
    graph_mode: str
    probability_backend: str
    probability_backend_execution_counts: dict[str, int]
    acceptance_backend: str
    acceptance_backend_execution_counts: dict[str, int]
    lm_head_backend: str
    rng_backend: str
    fused_predictor_scratch_rows: int
    effective_width_histogram: dict[str, int]
    fallback_reason: str | None
    frames: int
    output_code_frames: int
    rounds: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    committed_codes: int
    round_committed_codes: int
    bonus_tokens: int
    rejections: int
    average_codes_per_round: float
    packed_transformer_forwards: int
    packed_query_slots: int
    padded_query_slots: int
    draft_head_evaluations: int
    target_head_evaluations: int
    fused_predictor_forwards: int
    generic_predictor_forwards: int
    standalone_correction_forward_count: int
    full_round_graph_replays: int
    full_round_graph_replay_coverage: float
    full_round_graph_capture_time_ms: float
    opening_pair_graph_replays: int
    opening_pair_graph_replay_coverage: float
    opening_pair_graph_capture_time_ms: float
    graph_capture_failures: int
    graph_captures: int
    full_round_graph_captures: int
    full_round_graph_capture_failures: int
    opening_pair_graph_captures: int
    opening_pair_graph_capture_failures: int
    graph_capture_memory_delta_bytes: int
    graph_capture_time_ms: float
    workspace_bytes: int
    workspace_allocations: int
    configuration_time_ms: float
    nvtx_enabled: bool
    native_fallback_count: int
    attempted_frames: int
    native_fallback_fraction: float


class PredictorSpeculativeOpeningGraph:
    """Persistent graph for the opening pair and frame-state initialization."""

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device,
        hidden_size: int,
        hidden_dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.device_module = torch.get_device_module(device)
        self.layer0_codes = torch.empty(
            (batch_size, 1), device=device, dtype=torch.long
        )
        self.talker_hidden = torch.empty(
            (batch_size, 1, hidden_size), device=device, dtype=hidden_dtype
        )
        self.semantic_positions = torch.empty(
            (batch_size,), device=device, dtype=torch.long
        )
        self.graph: ReplayableGraph | None = None

    @torch.no_grad()
    def replay(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor,
    ) -> None:
        if (
            layer0_codes.shape != (self.batch_size, 1)
            or talker_hidden.shape[0] != self.batch_size
            or talker_hidden.shape[1] != 1
            or semantic_positions.shape != (self.batch_size,)
        ):
            raise ValueError("opening-pair graph inputs do not match its key")
        else:
            pass
        if (
            layer0_codes.device != self.device
            or talker_hidden.device != self.device
            or semantic_positions.device != self.device
        ):
            raise ValueError("opening-pair graph inputs must share its device")
        else:
            pass
        with self.device_module.device(self.device):
            self.layer0_codes.copy_(layer0_codes)
            self.talker_hidden.copy_(talker_hidden)
            self.semantic_positions.copy_(semantic_positions)
            if self.graph is None:
                raise RuntimeError("opening-pair graph has not been captured")
            else:
                pass
            self.graph.replay()


class PredictorSpeculativeRoundGraph:
    """Persistent graph for one complete speculative verification round."""

    def __init__(
        self,
        graph: ReplayableGraph,
        *,
        proposal_probability_backend: str | None,
        target_probability_backend: str,
    ) -> None:
        self.graph = graph
        self.proposal_probability_backend = proposal_probability_backend
        self.target_probability_backend = target_probability_backend

    def replay(self) -> None:
        self.graph.replay()


class PredictorSpeculativeMixin:
    def code_predictor_p16_runtime_fallback_reason(self) -> str | None:
        """Check model/device capabilities without requiring P16 to be enabled."""
        if not current_platform.is_cuda() or self.predictor_device.type != "cuda":
            return "cuda-unavailable"
        elif self.predictor_k_cache.dtype is not torch.bfloat16:
            return "unsupported-predictor-dtype"
        elif int(get_parallel().tp_size) != 1:
            return "unsupported-tensor-parallel-size"
        elif (
            bool(get_exec().graph.disable_cuda_graph)
            or current_platform.get_device_graph_backend(self.predictor_device) is None
        ):
            return "cuda-graph-unavailable"
        elif self.predictor_cutedsl_layers is not None:
            return "unsupported-predictor-backend"
        elif self.predictor_fused_layers is None:
            return "fused-predictor-unavailable"
        elif not grouped_lm_head_warp_accumulate_available():
            return "draft-head-kernel-unavailable"
        else:
            pass

        vocab_size = int(self.config.code_predictor_config.vocab_size)
        if not all(
            isinstance(lm_head.quant_method, UnquantizedLinearMethod)
            and lm_head.bias is None
            and lm_head.weight.device == self.predictor_device
            and lm_head.weight.dtype is torch.bfloat16
            and lm_head.weight.is_contiguous()
            and tuple(lm_head.weight.shape)
            == (
                vocab_size,
                int(self.code_predictor.model.norm.weight.shape[0]),
            )
            for lm_head in self.code_predictor.lm_head
        ):
            return "unsupported-draft-head-layout"
        else:
            pass
        attention = self.code_predictor.model.layers[0].self_attn
        if (
            resolve_fused_predictor_width(
                4,
                1,
                self.predictor_fused_layers.max_rows,
                int(attention.num_heads),
                int(attention.num_kv_heads),
            )
            != 4
        ):
            return "insufficient-fused-predictor-capacity"
        else:
            return None

    def code_predictor_p16_graph_pool(self) -> DeviceGraphPool:
        """Share one bounded graph memory pool across the P16 graph set."""
        graph_pool = self.code_predictor_spec_graph_pool
        if graph_pool is None:
            graph_pool = self.predictor_device_module.graph_pool_handle()
            self.code_predictor_spec_graph_pool = graph_pool
        else:
            pass
        return graph_pool

    def prepare_code_predictor_speculative_workspace(self) -> None:
        """Reserve the supported P16 workspace after request sampling is staged."""
        if (
            self.code_predictor_spec_requested_width <= 0
            or self.code_predictor_spec_effective_width <= 0
        ):
            return
        else:
            pass
        sampling_fallback = speculative_sampling_fallback_reason(
            self.sub_request_ids,
            self.sub_sample_rows,
            has_unbounded_top_k=self.sub_sampled_has_unbounded_top_k,
        )
        if (
            sampling_fallback is None
            and fused_raw_logit_block_k(self.sub_sampled_max_top_k) is not None
            and self.sub_sampled_max_top_k <= _SPECULATIVE_WORKSPACE_MAX_TOP_K
        ):
            self.ensure_code_predictor_speculative_workspace()
        else:
            pass

    def code_predictor_p16_frame_ineligibility_reason(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None,
    ) -> str | None:
        """Resolve P16 eligibility independently of the feature flag."""
        batch_size = int(layer0_codes.shape[0])
        static_reason = self.code_predictor_p16_static_fallback_reason
        if static_reason is not None:
            return static_reason
        elif batch_size != 1:
            return "unsupported_batch"
        elif layer0_codes.shape[1] != 1 or talker_hidden.shape[:2] != (1, 1):
            return "unsupported_sequence_shape"
        elif self.sub_sampling_parameters != ((0.9, 1.0, 50, True),):
            return "unsupported_sampling"
        elif self.sub_has_argmax_rows or not self.sub_has_sampled_rows:
            return "unsupported_sampling"
        else:
            pass
        runtime_reason = self.code_predictor_p16_runtime_fallback_reason()
        if runtime_reason is not None:
            return runtime_reason
        elif self.code_predictor_p16_enabled and self.code_predictor_p16_draft is None:
            return "p16_draft_unavailable"
        elif (
            self.code_predictor_p16_enabled
            and self.code_predictor_spec_lm_head_weights is None
        ):
            return "draft_head_unavailable"
        else:
            pass
        sampling_reason = speculative_sampling_fallback_reason(
            self.sub_request_ids,
            self.sub_sample_rows,
            has_unbounded_top_k=self.sub_sampled_has_unbounded_top_k,
        )
        if sampling_reason is not None:
            return "unsupported_sampling"
        else:
            pass
        if (
            self.sub_sampled_max_top_k != _P16_TOP_K
            or fused_raw_logit_block_k(self.sub_sampled_max_top_k) is None
        ):
            return "unsupported_sampling"
        elif self.predictor_graph_signature(batch_size, semantic_positions) is None:
            return "unsupported_graph_signature"
        else:
            pass
        return None

    def code_predictor_p16_target_logits(
        self, hidden_states: torch.Tensor, head_index: int
    ) -> torch.Tensor:
        """Apply the native target head to the native incremental input shape."""
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("P16 target projection expects one native query row")
        else:
            pass
        logits, _ = self.code_predictor.lm_head[head_index](hidden_states)
        return logits

    def set_code_predictor_speculative_audit_observer(
        self, observer: PredictorSpeculativeAuditObserver | None
    ) -> None:
        if observer is self.code_predictor_spec_audit_observer:
            return
        elif observer is None:
            self.code_predictor_spec_audit_observer = None
            if (
                not self.code_predictor_spec_round_graphs
                and not self.code_predictor_spec_opening_graphs
            ):
                self.code_predictor_spec_audit_hidden_states = None
                self.code_predictor_spec_audit_opening_logits = None
            else:
                pass
        elif self.code_predictor_spec_effective_width < 1:
            raise RuntimeError(
                "speculative graph audit requires an enabled full-round graph"
            )
        elif (
            self.code_predictor_spec_round_graphs
            or self.code_predictor_spec_opening_graphs
        ):
            raise RuntimeError(
                "attach the speculative audit observer before graph capture"
            )
        else:
            self.code_predictor_spec_audit_observer = observer
            self.code_predictor_spec_audit_hidden_states = torch.empty(
                (
                    self.predictor_max_batch_size,
                    self.code_predictor_spec_requested_width,
                    int(self.predictor_fused_layers.shape.hidden_size),
                ),
                device=self.predictor_device,
                dtype=self.predictor_k_cache.dtype,
            )
            self.code_predictor_spec_audit_opening_logits = torch.empty(
                (
                    self.predictor_max_batch_size,
                    int(self.config.code_predictor_config.vocab_size),
                ),
                device=self.predictor_device,
                dtype=self.predictor_k_cache.dtype,
            )

    def p16_telemetry_counters(self) -> dict[str, int]:
        """Read host-maintained execution counters without touching CUDA state."""
        return {
            "frames": self.code_predictor_spec_frame_count,
            "rounds": self.code_predictor_spec_round_count,
            "proposed": self.code_predictor_spec_proposed_count,
            "accepted": self.code_predictor_spec_accepted_count,
            "committed": self.code_predictor_spec_committed_count,
            "spec_graph_replays": self.code_predictor_spec_full_round_graph_replay_count,
            "spec_openings": self.code_predictor_spec_opening_graph_replay_count,
            "graph_capture_failures": (
                self.code_predictor_spec_full_round_graph_capture_failure_count
                + self.code_predictor_spec_opening_graph_capture_failure_count
            ),
            "eager_spec_rounds": 0,
            "invalid_probability_rows": self.code_predictor_spec_invalid_probability_rows,
            "cuda_errors": self.code_predictor_spec_cuda_errors,
        }

    @contextmanager
    def code_predictor_spec_nvtx_range(self, label: str) -> Generator[None, None, None]:
        """Mark a speculative stage when optional CUDA NVTX ranges are enabled."""
        if (
            not self.code_predictor_spec_nvtx_enabled
            or self.predictor_device.type != "cuda"
        ):
            yield
        else:
            with torch.cuda.nvtx.range(f"Qwen3-TTS CodePredictor: {label}"):
                yield

    def reserve_code_predictor_speculative_workspace(self) -> None:
        """Reserve one bounded workspace before SGLang sizes its KV pool."""
        if self.code_predictor_spec_requested_width <= 0:
            return
        else:
            pass
        if self.code_predictor_spec_workspaces:
            return
        else:
            pass
        if (
            self.sub_speculative_seed_tensor is None
            or self.code_predictor_spec_metadata_device is None
            or self.code_predictor_spec_metadata_host is None
            or self.predictor_fused_layers is None
        ):
            raise RuntimeError("speculative workspace buffers were not configured")
        else:
            pass

        candidate_width = fused_raw_logit_block_k(_SPECULATIVE_WORKSPACE_MAX_TOP_K)
        if candidate_width is None:
            raise RuntimeError("bounded speculative workspace Top-K is unavailable")
        else:
            pass
        workspace = PredictorSpeculativeWorkspace(
            max_batch_size=1,
            max_verify_width=_P16_VERIFY_WIDTH,
            vocab_size=int(self.config.code_predictor_config.vocab_size),
            hidden_size=int(self.predictor_fused_layers.shape.hidden_size),
            hidden_dtype=self.predictor_k_cache.dtype,
            device=self.predictor_device,
            max_top_k=_SPECULATIVE_WORKSPACE_MAX_TOP_K,
            candidate_width=candidate_width,
            split_top_k_chunks=TOP_K_CHUNKS,
            seed_rows=self.sub_speculative_seed_tensor,
            uniforms=self.sub_speculative_uniforms_tensor,
            rng_pool=self.sub_speculative_rng_pool_tensor,
            metadata_device=self.code_predictor_spec_metadata_device,
            metadata_host=self.code_predictor_spec_metadata_host,
            verifier_input_buffers=self.code_predictor_spec_input_buffers,
        )
        self.code_predictor_spec_workspaces[_SPECULATIVE_WORKSPACE_MAX_TOP_K] = (
            workspace
        )
        self.code_predictor_spec_workspace_allocation_count += 1

    def ensure_code_predictor_speculative_workspace(
        self,
    ) -> PredictorSpeculativeWorkspace:
        """Return the startup-reserved workspace for supported request signatures."""
        if self.code_predictor_spec_requested_width <= 0:
            raise RuntimeError("speculative workspace is disabled at width zero")
        else:
            pass
        candidate_width = fused_raw_logit_block_k(self.sub_sampled_max_top_k)
        if (
            candidate_width is None
            or self.sub_sampled_max_top_k > _SPECULATIVE_WORKSPACE_MAX_TOP_K
        ):
            raise RuntimeError(
                "request Top-K exceeds the reserved speculative workspace"
            )
        else:
            pass
        workspace = self.code_predictor_spec_workspaces.get(
            _SPECULATIVE_WORKSPACE_MAX_TOP_K
        )
        if workspace is None:
            raise RuntimeError(
                "speculative workspace was not reserved before KV sizing"
            )
        elif (
            workspace.max_batch_size < 1
            or workspace.max_verify_width < self.code_predictor_spec_requested_width
            or workspace.candidate_width < candidate_width
        ):
            raise RuntimeError(
                "reserved speculative workspace has insufficient capacity"
            )
        else:
            pass
        self.code_predictor_spec_active_workspace = workspace
        return workspace

    @contextmanager
    def preserve_speculative_capture_state(
        self,
        workspace: PredictorSpeculativeWorkspace,
        batch_size: int,
    ) -> Generator[None, None, None]:
        """Restore live frame and KV state after a graph warmup pass."""
        state_tensors = (
            workspace.committed_depths[:batch_size],
            workspace.logical_kv_lengths[:batch_size],
            workspace.pending_tokens[:batch_size],
            workspace.semantic_positions[:batch_size],
            workspace.preceding_hidden[:batch_size],
            workspace.active_rows[:batch_size],
            self.output_codes[:batch_size],
            self.output_embeds[:batch_size],
        )
        saved_state = tuple(tensor.clone() for tensor in state_tensors)
        saved_key_cache = self.predictor_k_cache.clone()
        saved_value_cache = self.predictor_v_cache.clone()
        saved_correction_watch = self.code_predictor_spec_correction_forward_watch
        saved_correction_count = (
            self.code_predictor_spec_standalone_correction_forward_count
        )
        try:
            yield
        finally:
            self.predictor_device_module.current_stream(
                self.predictor_device
            ).synchronize()
            for tensor, saved in zip(state_tensors, saved_state, strict=True):
                tensor.copy_(saved)
            self.predictor_k_cache.copy_(saved_key_cache)
            self.predictor_v_cache.copy_(saved_value_cache)
            self.code_predictor_spec_correction_forward_watch = saved_correction_watch
            self.code_predictor_spec_standalone_correction_forward_count = (
                saved_correction_count
            )

    def capture_code_predictor_spec_graph(
        self,
        run_once: SpeculativeGraphCaptureOperation,
        *,
        graph_pool: DeviceGraphPool | None = None,
    ) -> tuple[ReplayableGraph, SpeculativeGraphCaptureOutputs, float, int]:
        """Capture already-warmed operations on a persistent graph pool."""
        device = self.predictor_device
        module = self.predictor_device_module
        backend = current_platform.get_device_graph_backend(device)
        if backend is None:
            raise RuntimeError("the selected device has no CUDA graph backend")
        else:
            pass
        if self.predictor_capture_stream is None:
            self.predictor_capture_stream = module.Stream(device=device)
        else:
            pass
        capture_stream = self.predictor_capture_stream
        current_stream = module.current_stream(device)
        capture_stream.wait_stream(current_stream)
        memory_before = torch.cuda.memory_allocated(device)
        capture_started = time.perf_counter()
        captured: ReplayableGraph | None = None
        capture_outputs: SpeculativeGraphCaptureOutputs = None
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                module.device(device),
                current_platform.graph_capture_attention(),
                module.stream(capture_stream),
            ):
                with backend.capture(
                    pool=(
                        self.predictor_graph_memory_pool()
                        if graph_pool is None
                        else graph_pool
                    ),
                    stream=capture_stream,
                    thread_local_errors=True,
                ) as captured:
                    capture_outputs = run_once()
        except Exception:
            if captured is not None:
                try:
                    captured.reset()
                except Exception:
                    pass
            else:
                pass
            raise
        finally:
            current_stream.wait_stream(capture_stream)
            if gc_was_enabled:
                gc.enable()
            else:
                pass
        if captured is None:
            raise RuntimeError("device graph capture returned no replayable graph")
        else:
            pass
        elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
        memory_delta = max(torch.cuda.memory_allocated(device) - memory_before, 0)
        return captured, capture_outputs, elapsed_ms, memory_delta

    def try_replay_full_round_graph(
        self,
        batch_size: int,
        round_depth: int,
        verify_width: int,
        workspace: PredictorSpeculativeWorkspace,
        run_once: SpeculativeGraphCaptureOperation,
    ) -> tuple[str | None, str, str] | None:
        """Lazily capture and replay one complete, compact-acceptance round."""
        signature = self.predictor_graph_signature(
            batch_size, workspace.semantic_positions[:batch_size]
        )
        if signature is None:
            raise RuntimeError("P16 full-round graph signature changed after opening")
        else:
            pass
        key = (
            batch_size,
            round_depth,
            verify_width,
            signature,
        )
        graph = self.code_predictor_spec_round_graphs.get(key)
        if graph is not None:
            graph.replay()
            self.code_predictor_spec_full_round_graph_replay_count += 1
            return (
                graph.proposal_probability_backend,
                graph.target_probability_backend,
                "compact",
            )
        else:
            pass
        if key in self.code_predictor_spec_round_graph_disabled:
            raise RuntimeError(f"P16 full-round graph is disabled for key={key}")
        else:
            pass
        if graph is None:
            if (
                len(self.code_predictor_spec_round_graphs)
                >= _PREDICTOR_SPECULATIVE_GRAPH_MAX_KEYS
                or current_platform.get_device_graph_backend(self.predictor_device)
                is None
                or bool(get_exec().graph.disable_cuda_graph)
            ):
                self.code_predictor_spec_round_graph_disabled.add(key)
                raise RuntimeError("P16 full-round graph cache capacity is exhausted")
            else:
                pass
            try:
                with self.preserve_speculative_capture_state(workspace, batch_size):
                    for _ in range(_PREDICTOR_GRAPH_WARMUP_PASSES):
                        run_once()
                    else:
                        pass
                with self.preserve_speculative_capture_state(workspace, batch_size):
                    graph_pool = self.code_predictor_p16_graph_pool()
                    captured_graph, captured_outputs, elapsed_ms, memory_delta = (
                        self.capture_code_predictor_spec_graph(
                            run_once, graph_pool=graph_pool
                        )
                    )
            except Exception as error:
                self.code_predictor_spec_round_graph_disabled.add(key)
                self.code_predictor_spec_full_round_graph_capture_failure_count += 1
                if is_fatal_cuda_capture_error(error):
                    raise
                else:
                    pass
                logger.warning(
                    f"Disabling CodePredictor full-round graph for key={key}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"P16 full-round graph capture failed for key={key}"
                ) from error
            else:
                pass
            if not isinstance(captured_outputs, tuple) or len(captured_outputs) != 3:
                captured_graph.reset()
                self.code_predictor_spec_round_graph_disabled.add(key)
                self.code_predictor_spec_full_round_graph_capture_failure_count += 1
                raise RuntimeError("P16 full-round graph capture metadata is invalid")
            else:
                pass
            proposal_backend, target_backend, acceptance_backend = captured_outputs
            if acceptance_backend != "compact":
                captured_graph.reset()
                self.code_predictor_spec_round_graph_disabled.add(key)
                self.code_predictor_spec_full_round_graph_capture_failure_count += 1
                raise RuntimeError(
                    "full-round graph capture did not use compact acceptance"
                )
            else:
                pass
            graph = PredictorSpeculativeRoundGraph(
                captured_graph,
                proposal_probability_backend=proposal_backend,
                target_probability_backend=target_backend,
            )
            self.code_predictor_spec_round_graphs[key] = graph
            self.code_predictor_spec_full_round_graph_capture_count += 1
            self.code_predictor_spec_full_round_graph_capture_time_ms += elapsed_ms
            self.code_predictor_spec_graph_capture_time_ms += elapsed_ms
            self.code_predictor_spec_graph_memory_delta_bytes += memory_delta
        else:
            pass
        graph.replay()
        self.code_predictor_spec_full_round_graph_replay_count += 1
        return (
            graph.proposal_probability_backend,
            graph.target_probability_backend,
            "compact",
        )

    def execute_speculative_opening_pair(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor,
        workspace: PredictorSpeculativeWorkspace,
    ) -> None:
        """Run opening pair, codebook-1 sampling, and initial round staging."""
        batch_size = int(layer0_codes.shape[0])
        predictor_dtype = self.predictor_k_cache.dtype
        code_outputs = speculative_opening_code_output_view(
            self.output_codes, batch_size
        )
        summed_embeddings = self.output_embeds[:batch_size].unsqueeze(1)
        summed_embeddings.zero_()
        layer0_embed = self.get_input_embeddings()(layer0_codes).to(
            dtype=predictor_dtype
        )
        code_outputs[:, 0].copy_(layer0_codes[:, 0])
        summed_embeddings[:, 0, :].add_(layer0_embed[:, 0, :])
        pair_embeds = self.code_predictor.project_input(
            torch.cat(
                (talker_hidden.to(dtype=predictor_dtype), layer0_embed),
                dim=1,
            )
        )
        pair_hidden = self.predictor_forward_tokens(
            token_embeds=pair_embeds,
            batch_size=batch_size,
            cache_len=0,
        )
        code1_logits, _ = self.code_predictor.lm_head[0](pair_hidden[:, 1:, :])
        if self.code_predictor_spec_audit_observer is not None:
            audit_opening_logits = self.code_predictor_spec_audit_opening_logits
            if audit_opening_logits is None:
                raise RuntimeError("speculative opening audit logits are missing")
            else:
                pass
            audit_opening_logits[:batch_size].copy_(code1_logits[:, -1, :])
        else:
            pass
        sub_positions = self.sub_seed_positions(semantic_positions)[0]
        codec_tables = self.code_predictor.model.codec_embedding.weight
        projected_tables = self.predictor_projected_embeddings
        projected_table = None if projected_tables is None else projected_tables[0]
        self.sample_subtalker_embedding(
            code1_logits[:, -1, :],
            codec_tables[0],
            summed_embeddings[:, 0, :],
            code_outputs[:, 1],
            projected_table,
            self.predictor_embedding_buffer[:batch_size],
            (
                None
                if self.predictor_projected_buffer is None
                else self.predictor_projected_buffer[:batch_size]
            ),
            sub_positions,
        )
        workspace.committed_depths[:batch_size].fill_(1)
        workspace.logical_kv_lengths[:batch_size].fill_(2)
        workspace.pending_tokens[:batch_size].copy_(code_outputs[:, 1])
        workspace.preceding_hidden[:batch_size].copy_(pair_hidden[:, 1, :])
        workspace.active_rows[:batch_size].fill_(True)

    def resolve_speculative_opening_graph_key(
        self,
        batch_size: int,
        semantic_positions: torch.Tensor,
    ) -> PredictorSpeculativeOpeningGraphKey | None:
        signature = self.predictor_graph_signature(batch_size, semantic_positions)
        if signature is None:
            return None
        else:
            pass
        return (
            batch_size,
            signature,
        )

    def try_replay_speculative_opening_graph(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor,
        workspace: PredictorSpeculativeWorkspace,
    ) -> bool:
        """Capture lazily, then replay the independent opening-pair graph."""
        batch_size = int(layer0_codes.shape[0])
        key = self.resolve_speculative_opening_graph_key(batch_size, semantic_positions)
        if key is None:
            return False
        elif key in self.code_predictor_spec_opening_graph_disabled:
            return False
        else:
            pass
        graph = self.code_predictor_spec_opening_graphs.get(key)
        if graph is None:
            if (
                len(self.code_predictor_spec_opening_graphs) >= 1
                or current_platform.get_device_graph_backend(self.predictor_device)
                is None
            ):
                self.code_predictor_spec_opening_graph_disabled.add(key)
                return False
            else:
                pass
            graph = PredictorSpeculativeOpeningGraph(
                batch_size,
                device=self.predictor_device,
                hidden_size=int(talker_hidden.shape[-1]),
                hidden_dtype=self.predictor_k_cache.dtype,
            )

            def run_once() -> None:
                self.execute_speculative_opening_pair(
                    graph.layer0_codes,
                    graph.talker_hidden,
                    graph.semantic_positions,
                    workspace,
                )

            try:
                with self.preserve_speculative_capture_state(workspace, batch_size):
                    graph.layer0_codes.copy_(layer0_codes)
                    graph.talker_hidden.copy_(talker_hidden)
                    graph.semantic_positions.copy_(semantic_positions)
                    for _ in range(_PREDICTOR_GRAPH_WARMUP_PASSES):
                        run_once()
                    else:
                        pass
                    graph_pool = self.code_predictor_p16_graph_pool()
                    captured_graph, _, elapsed_ms, memory_delta = (
                        self.capture_code_predictor_spec_graph(
                            run_once, graph_pool=graph_pool
                        )
                    )
            except Exception as error:
                self.code_predictor_spec_opening_graph_disabled.add(key)
                self.code_predictor_spec_opening_graph_capture_failure_count += 1
                if is_fatal_cuda_capture_error(error):
                    raise
                else:
                    pass
                logger.warning(
                    f"Disabling CodePredictor opening-pair graph for key={key}",
                    exc_info=True,
                )
                return False
            else:
                pass
            graph.graph = captured_graph
            self.code_predictor_spec_opening_graphs[key] = graph
            self.code_predictor_spec_opening_graph_capture_count += 1
            self.code_predictor_spec_opening_graph_capture_time_ms += elapsed_ms
            self.code_predictor_spec_graph_capture_time_ms += elapsed_ms
            self.code_predictor_spec_graph_memory_delta_bytes += memory_delta
        else:
            pass
        graph.replay(layer0_codes, talker_hidden, semantic_positions)
        self.code_predictor_spec_opening_graph_replay_count += 1
        return True

    @torch.no_grad()
    def code_predictor_forward_speculative(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Execute one complete Predictor frame with packed speculative rounds."""
        batch_size = int(layer0_codes.shape[0])
        if (
            layer0_codes.shape[1] != 1
            or talker_hidden.shape[0] != batch_size
            or talker_hidden.shape[1] != 1
        ):
            return None
        else:
            pass
        ineligibility_reason = self.code_predictor_p16_frame_ineligibility_reason(
            layer0_codes, talker_hidden, semantic_positions
        )
        if ineligibility_reason is not None:
            return None
        else:
            pass
        maximum_frame_width = _P16_VERIFY_WIDTH
        workspace = self.ensure_code_predictor_speculative_workspace()
        packed_head_weights = self.code_predictor_spec_lm_head_weights
        if packed_head_weights is None:
            raise RuntimeError("grouped Predictor head weights are unavailable")
        else:
            pass
        num_groups = int(self.config.num_code_groups)
        vocab_size = int(self.config.code_predictor_config.vocab_size)
        codec_tables = self.code_predictor.model.codec_embedding.weight
        projected_tables = self.predictor_projected_embeddings
        predictor_dtype = self.predictor_k_cache.dtype
        semantic_positions = self.normalize_semantic_positions(
            semantic_positions,
            batch_size=batch_size,
            seq_len=1,
            device=layer0_codes.device,
        )[:, 0]
        workspace.semantic_positions[:batch_size].copy_(semantic_positions)
        semantic_positions = workspace.semantic_positions[:batch_size]

        result_codes = self.output_codes[:batch_size].unsqueeze(-1)
        code_outputs = result_codes[:, :, 0]
        summed_embeddings = self.output_embeds[:batch_size].unsqueeze(1)
        committed_depths = workspace.committed_depths[:batch_size]
        logical_kv_lengths = workspace.logical_kv_lengths[:batch_size]
        pending_tokens = workspace.pending_tokens[:batch_size]
        preceding_hidden = workspace.preceding_hidden[:batch_size]
        active_rows = workspace.active_rows[:batch_size]
        opening_graph_replayed = False
        with self.code_predictor_spec_nvtx_range("opening pair graph"):
            opening_graph_replayed = self.try_replay_speculative_opening_graph(
                layer0_codes,
                talker_hidden,
                semantic_positions,
                workspace,
            )
        if not opening_graph_replayed:
            raise RuntimeError("P16 opening-pair graph did not replay")
        else:
            pass
        audit_observer = self.code_predictor_spec_audit_observer
        if audit_observer is not None:
            audit_observer(
                "opening",
                {
                    "layer0_codes": layer0_codes,
                    "talker_hidden": talker_hidden,
                    "semantic_positions": semantic_positions,
                    "opening_logits": (
                        self.code_predictor_spec_audit_opening_logits[:batch_size]
                    ),
                    "opening_sampling_seeds": self.sub_sampling_seed_tensor[
                        :batch_size
                    ],
                    "opening_sampling_positions": self.sub_seed_positions(
                        semantic_positions
                    )[0],
                    "output_codes": code_outputs,
                    "pending_tokens": pending_tokens,
                    "preceding_hidden": preceding_hidden,
                    "key_cache": self.predictor_k_cache[:, :batch_size, :2],
                    "value_cache": self.predictor_v_cache[:, :batch_size, :2],
                },
            )
        else:
            pass
        self.code_predictor_spec_fused_predictor_forward_count += 1
        metadata_device = workspace.metadata_device
        metadata_host = workspace.metadata_host
        metadata_device_rows = metadata_device[:batch_size]
        metadata_host_rows = metadata_host[:batch_size]
        metadata_host_is_pinned = metadata_host.is_pinned()
        current_stream = self.predictor_device_module.current_stream(
            self.predictor_device
        )
        temperatures = self.sub_temperature_tensor[:batch_size]
        top_ks = self.sub_top_k_tensor[:batch_size]
        top_ps = self.sub_top_p_tensor[:batch_size]
        speculative_seeds = workspace.seed_rows[:batch_size]
        if workspace.rng_pool is None or workspace.uniforms is None:
            raise RuntimeError("P16 persistent RNG buffers are unavailable")
        else:
            pass
        frame_rng_pool = speculative_uniforms_frame_pool(
            speculative_seeds,
            semantic_positions,
            num_groups,
            num_groups - 2,
            int(workspace.rng_pool.shape[3]),
            workspace.rng_pool[:batch_size],
        )
        max_top_k = int(self.sub_sampled_max_top_k)
        candidate_count = fused_raw_logit_block_k(max_top_k)
        if candidate_count is None:
            raise RuntimeError("speculative Top-K candidate width is unavailable")
        else:
            pass
        has_top_p = bool(self.sub_sampled_has_top_p)
        round_index = 0
        local_round_count = 0
        local_proposed_count = 0
        local_accepted_count = 0
        local_committed_count = 0
        local_bonus_count = 0
        local_rejection_count = 0
        local_packed_forward_count = 0
        local_fused_forward_count = 0
        local_draft_evaluation_count = 0
        local_target_evaluation_count = 0
        local_packed_query_slots = 0
        local_padded_query_slots = 0
        local_width_histogram: dict[int, int] = {}
        local_probability_backend_counts = {"split_top_k": 0}
        local_acceptance_backend_counts = {"compact": 0}

        round_uniforms = workspace.uniforms[:batch_size]

        def prepare_speculative_round_operation(
            verify_width: int, round_depth: int
        ) -> SpeculativeGraphCaptureOperation:
            proposal_width = verify_width - 1
            round_workspace = workspace.rounds[verify_width]
            proposal_rows = batch_size * proposal_width
            target_rows = batch_size * verify_width
            proposal_counts = round_workspace.proposal_counts[:batch_size]
            valid_query_counts = round_workspace.valid_query_counts[:batch_size]
            proposal_head_indices = round_workspace.proposal_head_indices[
                :proposal_rows
            ]
            proposal_probability_buffer = compact_candidate_view(
                round_workspace.proposal_probabilities,
                proposal_rows,
                candidate_count,
            )
            proposal_token_id_buffer = compact_candidate_view(
                round_workspace.proposal_token_ids,
                proposal_rows,
                candidate_count,
            )
            if round_workspace.split_candidate_keys is None:
                proposal_split_keys = None
            else:
                proposal_split_keys = compact_scratch_view(
                    round_workspace.split_candidate_keys,
                    proposal_rows,
                    TOP_K_CHUNKS * candidate_count,
                )
            if round_workspace.split_candidate_keys is None:
                target_split_keys = None
            else:
                split_key_rows = compact_scratch_view(
                    round_workspace.split_candidate_keys,
                    proposal_rows + target_rows,
                    TOP_K_CHUNKS * candidate_count,
                )
                target_split_keys = split_key_rows[proposal_rows:]
            active_proposal_rows = round_workspace.proposal_active_rows[:proposal_rows]
            proposal_temperatures = round_workspace.proposal_temperatures[
                :proposal_rows
            ]
            proposal_top_ks = round_workspace.proposal_top_ks[:proposal_rows]
            proposal_top_ps = round_workspace.proposal_top_ps[:proposal_rows]
            target_temperatures = round_workspace.target_temperatures[:target_rows]
            target_top_ks = round_workspace.target_top_ks[:target_rows]
            target_top_ps = round_workspace.target_top_ps[:target_rows]

            def run_round_device_ops() -> tuple[str | None, str, str]:
                with self.code_predictor_spec_nvtx_range("proposal preparation"):
                    round_metadata_prepared = prepare_speculative_round_metadata(
                        committed_depths,
                        active_rows,
                        temperatures,
                        top_ks,
                        top_ps,
                        round_workspace.proposal_head_offsets[:proposal_rows],
                        proposal_head_indices,
                        active_proposal_rows,
                        proposal_counts,
                        valid_query_counts,
                        proposal_temperatures,
                        proposal_top_ks,
                        proposal_top_ps,
                        target_temperatures,
                        target_top_ks,
                        target_top_ps,
                        verify_width,
                        num_groups,
                    )
                if not round_metadata_prepared:
                    raise RuntimeError(
                        "speculative round metadata kernel is unavailable"
                    )
                else:
                    pass
                proposal_uniforms = round_uniforms[:, 0, :proposal_width]
                acceptance_uniforms = round_uniforms[:, 1, :proposal_width]
                correction_uniforms = round_uniforms[:, 2, :verify_width]
                bonus_uniforms = round_uniforms[:, 3, :verify_width]
                proposal_probability_backend: str | None = None
                if proposal_width > 0:
                    draft_runtime = self.code_predictor_p16_draft
                    if draft_runtime is None:
                        raise RuntimeError("P16 draft runtime was not loaded")
                    else:
                        pass
                    round_workspace.proposal_tokens[:batch_size].zero_()
                    draft_hidden = preceding_hidden
                    for proposal_index in range(proposal_width):
                        proposal_row = proposal_index * batch_size
                        proposal_row_slice = slice(
                            proposal_row, proposal_row + batch_size
                        )
                        proposal_depth = round_depth + proposal_index
                        if proposal_depth > 14:
                            raise RuntimeError(
                                "P16 proposal depth exceeded the final codebook"
                            )
                        else:
                            pass
                        if proposal_index == 0:
                            proposal_input_tokens = pending_tokens
                        else:
                            proposal_input_tokens = round_workspace.proposal_tokens[
                                :batch_size, proposal_index - 1
                            ]
                        draft_hidden_output = (
                            draft_runtime.output
                            if proposal_index % 2 == 0
                            else draft_runtime.output_alternate
                        )
                        with self.code_predictor_spec_nvtx_range("draft MLP"):
                            draft_hidden = draft_runtime.step(
                                draft_hidden,
                                proposal_input_tokens,
                                proposal_depth,
                                hidden_output=draft_hidden_output,
                            )
                        with self.code_predictor_spec_nvtx_range("draft GEMV"):
                            proposal_logits = grouped_lm_head_warp_accumulate(
                                draft_hidden,
                                packed_head_weights,
                                None,
                                proposal_head_indices[proposal_row_slice],
                                active_proposal_rows[proposal_row_slice],
                                output=round_workspace.proposal_logits[
                                    proposal_row_slice
                                ],
                            )
                        with self.code_predictor_spec_nvtx_range("draft probabilities"):
                            proposal_candidates = (
                                build_speculative_probability_candidates(
                                    proposal_logits,
                                    proposal_temperatures[proposal_row_slice],
                                    proposal_top_ks[proposal_row_slice],
                                    proposal_top_ps[proposal_row_slice],
                                    active_proposal_rows[proposal_row_slice],
                                    max_top_k=max_top_k,
                                    has_top_p=has_top_p,
                                    has_unbounded_top_k=False,
                                    probability_output=proposal_probability_buffer[
                                        proposal_row_slice
                                    ],
                                    token_id_output=proposal_token_id_buffer[
                                        proposal_row_slice
                                    ],
                                    split_candidate_keys=(
                                        None
                                        if proposal_split_keys is None
                                        else proposal_split_keys[proposal_row_slice]
                                    ),
                                )
                            )
                        if proposal_candidates is None:
                            raise RuntimeError(
                                "P16 proposal probability kernel rejected a supported shape"
                            )
                        else:
                            pass
                        _, _ = proposal_candidates
                        if not uses_split_top_k(
                            proposal_logits, max_top_k, candidate_count
                        ):
                            raise RuntimeError(
                                "P16 probability builder did not execute split Top-K"
                            )
                        else:
                            pass
                        proposal_probability_backend = "split_top_k"
                        with self.code_predictor_spec_nvtx_range("proposal sampling"):
                            sample_compact_probability_rows_fused(
                                proposal_probability_buffer[proposal_row_slice],
                                proposal_token_id_buffer[proposal_row_slice],
                                proposal_uniforms[
                                    :, proposal_index : proposal_index + 1
                                ],
                                active_proposal_rows[proposal_row_slice].view(
                                    batch_size, 1
                                ),
                                output=round_workspace.proposal_tokens[
                                    :batch_size,
                                    proposal_index : proposal_index + 1,
                                ],
                            )
                    else:
                        pass
                    proposal_probabilities_compact = proposal_probability_buffer
                    proposal_ids_compact = proposal_token_id_buffer
                else:
                    round_workspace.proposal_tokens[:batch_size].zero_()
                    proposal_probabilities_compact = proposal_probability_buffer
                    proposal_ids_compact = proposal_token_id_buffer
                draft_tokens = round_workspace.proposal_tokens[:batch_size]
                input_buffer = round_workspace.verifier_inputs
                source_tables = (
                    codec_tables if projected_tables is None else projected_tables
                )
                with self.code_predictor_spec_nvtx_range(
                    "verification input preparation"
                ):
                    inputs_prepared = prepare_speculative_verification_inputs(
                        pending_tokens,
                        draft_tokens,
                        committed_depths,
                        logical_kv_lengths,
                        active_rows,
                        proposal_counts,
                        source_tables,
                        input_buffer,
                    )
                if not inputs_prepared:
                    raise RuntimeError(
                        "speculative verifier preparation kernel is unavailable"
                    )
                else:
                    pass
                query_embeddings = input_buffer.token_embeddings[:batch_size]
                packed_positions = input_buffer.positions[: batch_size * verify_width]
                query_depths = input_buffer.target_head_indices[:batch_size]
                valid_queries = input_buffer.valid_queries[:batch_size]
                with self.code_predictor_spec_nvtx_range("packed verifier"):
                    verified_hiddens = self.predictor_forward_packed_tokens(
                        token_embeds=query_embeddings,
                        positions=packed_positions,
                        batch_size=batch_size,
                    )
                target_logits = round_workspace.target_logits[:target_rows]
                with self.code_predictor_spec_nvtx_range(
                    "native per-query target heads"
                ):
                    for query_index in range(verify_width):
                        query_depth = round_depth + query_index
                        if query_depth > 14:
                            raise RuntimeError(
                                "P16 target head exceeded the final codebook"
                            )
                        else:
                            pass
                        query_hidden = verified_hiddens[
                            :batch_size, query_index : query_index + 1, :
                        ].contiguous()
                        query_logits = self.code_predictor_p16_target_logits(
                            query_hidden, query_depth
                        )
                        query_row_start = query_index * batch_size
                        target_logits[
                            query_row_start : query_row_start + batch_size
                        ].copy_(query_logits[:, 0, :])
                audit_hidden_states = self.code_predictor_spec_audit_hidden_states
                if audit_observer is not None:
                    if audit_hidden_states is None:
                        raise RuntimeError(
                            "speculative audit hidden-state storage is missing"
                        )
                    else:
                        pass
                    audit_hidden_states[:batch_size, :verify_width].copy_(
                        verified_hiddens
                    )
                else:
                    pass
                with self.code_predictor_spec_nvtx_range("target probabilities"):
                    target_probability_buffer = compact_candidate_view(
                        round_workspace.target_probabilities,
                        target_rows,
                        candidate_count,
                    )
                    target_token_id_buffer = compact_candidate_view(
                        round_workspace.target_token_ids,
                        target_rows,
                        candidate_count,
                    )
                    target_candidates = build_speculative_probability_candidates(
                        target_logits,
                        target_temperatures,
                        target_top_ks,
                        target_top_ps,
                        valid_queries.reshape(-1).contiguous(),
                        max_top_k=max_top_k,
                        has_top_p=has_top_p,
                        has_unbounded_top_k=False,
                        probability_output=target_probability_buffer,
                        token_id_output=target_token_id_buffer,
                        split_candidate_keys=target_split_keys,
                    )
                if target_candidates is None:
                    raise RuntimeError(
                        "speculative target probability kernel rejected a preflighted shape"
                    )
                else:
                    pass
                target_probabilities_compact, target_ids_compact = target_candidates
                target_probabilities_compact = target_probability_buffer
                target_ids_compact = target_token_id_buffer
                if not uses_split_top_k(target_logits, max_top_k, candidate_count):
                    raise RuntimeError(
                        "P16 probability builder did not execute split Top-K"
                    )
                else:
                    pass
                target_probability_backend = "split_top_k"
                acceptance_outputs = round_workspace.acceptance_outputs
                live_acceptance_outputs = SpeculativeAcceptanceBatch(
                    committed_tokens=acceptance_outputs.committed_tokens[:batch_size],
                    accepted_draft_counts=acceptance_outputs.accepted_draft_counts[
                        :batch_size
                    ],
                    committed_token_counts=acceptance_outputs.committed_token_counts[
                        :batch_size
                    ],
                    rejected_rows=acceptance_outputs.rejected_rows[:batch_size],
                    bonus_rows=acceptance_outputs.bonus_rows[:batch_size],
                    invalid_rows=acceptance_outputs.invalid_rows[:batch_size],
                )
                with self.code_predictor_spec_nvtx_range("acceptance"):
                    decision = accept_speculative_candidates_compact(
                        draft_tokens,
                        proposal_probabilities_compact,
                        proposal_ids_compact,
                        target_probabilities_compact,
                        target_ids_compact,
                        proposal_counts,
                        active_rows,
                        acceptance_uniforms,
                        correction_uniforms,
                        bonus_uniforms,
                        outputs=live_acceptance_outputs,
                    )
                    actual_acceptance_backend = "compact"

                with self.code_predictor_spec_nvtx_range("codec commit"):
                    commit_speculative_codes(
                        decision.committed_tokens,
                        decision.committed_token_counts,
                        committed_depths,
                        codec_tables,
                        code_outputs,
                        summed_embeddings[:, 0, :],
                    )
                with self.code_predictor_spec_nvtx_range("state transition"):
                    commit_speculative_state(
                        decision.committed_tokens,
                        decision.accepted_draft_counts,
                        decision.committed_token_counts,
                        proposal_counts,
                        decision.bonus_rows,
                        decision.rejected_rows,
                        decision.invalid_rows,
                        valid_query_counts,
                        verified_hiddens,
                        pending_tokens,
                        preceding_hidden,
                        committed_depths,
                        logical_kv_lengths,
                        active_rows,
                        metadata_device_rows,
                        num_groups,
                    )
                return (
                    proposal_probability_backend,
                    target_probability_backend,
                    actual_acceptance_backend,
                )

            return run_round_device_ops

        current_round_width = 0
        current_round_depth = 1
        current_round_operation: SpeculativeGraphCaptureOperation | None = None

        def run_round_device_ops() -> tuple[str | None, str, str]:
            nonlocal current_round_operation
            if current_round_operation is None:
                current_round_operation = prepare_speculative_round_operation(
                    current_round_width, current_round_depth
                )
            else:
                pass
            return current_round_operation()

        staged_round_width = num_groups - 2
        round_depth = 1
        try:
            while staged_round_width > 0:
                requested_round_width = min(maximum_frame_width, staged_round_width)
                verify_width = requested_round_width
                proposal_width = verify_width - 1
                round_uniforms[:, 0, :proposal_width].copy_(
                    frame_rng_pool[:, 0, round_index, :proposal_width]
                )
                round_uniforms[:, 1, :proposal_width].copy_(
                    frame_rng_pool[:, 1, round_index, :proposal_width]
                )
                round_uniforms[:, 2, :verify_width].copy_(
                    frame_rng_pool[:, 2, round_index, :verify_width]
                )
                round_uniforms[:, 3, :verify_width].copy_(
                    frame_rng_pool[:, 3, round_index, :verify_width]
                )

                current_round_width = verify_width
                current_round_depth = round_depth
                current_round_operation = None
                if audit_observer is None:
                    audit_pending_tokens = None
                else:
                    audit_pending_tokens = pending_tokens.clone()

                actual_backends = self.try_replay_full_round_graph(
                    batch_size,
                    round_depth,
                    verify_width,
                    workspace,
                    run_round_device_ops,
                )
                if actual_backends is None:
                    raise RuntimeError("P16 full-round graph replay was unavailable")
                else:
                    pass
                (
                    actual_proposal_backend,
                    actual_target_backend,
                    actual_acceptance_backend,
                ) = actual_backends
                if actual_proposal_backend is not None:
                    local_probability_backend_counts[actual_proposal_backend] += 1
                else:
                    pass
                local_probability_backend_counts[actual_target_backend] += 1
                local_acceptance_backend_counts[actual_acceptance_backend] += 1
                with self.code_predictor_spec_nvtx_range("metadata D2H"):
                    metadata_host_rows.copy_(
                        metadata_device_rows,
                        non_blocking=metadata_host_is_pinned,
                    )
                with self.code_predictor_spec_nvtx_range("host synchronization"):
                    current_stream.synchronize()
                with self.code_predictor_spec_nvtx_range("CPU orchestration"):
                    round_metadata_rows = metadata_host_rows.tolist()
                    invalid_probability_rows = sum(
                        int(row[8] != 0) for row in round_metadata_rows
                    )
                    self.code_predictor_spec_invalid_probability_rows += (
                        invalid_probability_rows
                    )
                    if invalid_probability_rows:
                        raise RuntimeError(
                            "compact speculative acceptance produced invalid probability mass"
                        )
                    else:
                        pass
                active_count = 0
                proposed_count = 0
                accepted_count = 0
                committed_count = 0
                bonus_count = 0
                rejection_count = 0
                valid_target_count = 0
                next_depths: list[int] = []
                next_active_rows: list[bool] = []
                for row in round_metadata_rows:
                    active_count += row[1]
                    proposed_count += row[2]
                    accepted_count += row[0]
                    committed_count += row[3]
                    bonus_count += row[4]
                    rejection_count += row[5]
                    valid_target_count += row[6]
                    next_depths.append(row[7])
                    next_active_rows.append(bool(row[1]))
                else:
                    pass
                round_depth = next_depths[0]
                self.code_predictor_spec_correction_forward_watch = any(
                    row[5] != 0 for row in round_metadata_rows
                )
                local_round_count += 1
                local_proposed_count += proposed_count
                local_accepted_count += accepted_count
                local_committed_count += committed_count
                local_bonus_count += bonus_count
                local_rejection_count += rejection_count
                local_packed_forward_count += 1
                local_fused_forward_count += 1
                local_draft_evaluation_count += proposed_count
                local_target_evaluation_count += valid_target_count
                local_width_histogram[verify_width] = (
                    local_width_histogram.get(verify_width, 0) + 1
                )
                graph_batch_size = batch_size
                packed_query_slots = graph_batch_size * verify_width
                local_packed_query_slots += packed_query_slots
                local_padded_query_slots += packed_query_slots - valid_target_count
                round_index += 1
                if audit_observer is not None:
                    proposal_rows = batch_size * proposal_width
                    target_rows = batch_size * verify_width
                    proposal_round_workspace = workspace.rounds[verify_width]
                    proposal_probabilities = compact_candidate_view(
                        proposal_round_workspace.proposal_probabilities,
                        proposal_rows,
                        candidate_count,
                    )
                    proposal_token_ids = compact_candidate_view(
                        proposal_round_workspace.proposal_token_ids,
                        proposal_rows,
                        candidate_count,
                    )
                    target_probabilities = compact_candidate_view(
                        proposal_round_workspace.target_probabilities,
                        target_rows,
                        candidate_count,
                    )
                    target_token_ids = compact_candidate_view(
                        proposal_round_workspace.target_token_ids,
                        target_rows,
                        candidate_count,
                    )
                    audit_hidden_states = self.code_predictor_spec_audit_hidden_states
                    if audit_pending_tokens is None or audit_hidden_states is None:
                        raise RuntimeError(
                            "speculative audit frame state is incomplete"
                        )
                    else:
                        pass
                    audit_observer(
                        "round",
                        {
                            "round_depth": current_round_depth,
                            "verify_width": verify_width,
                            "logical_kv_length": current_round_depth + 1,
                            "proposal_count": proposed_count,
                            "pending_tokens": audit_pending_tokens,
                            "proposal_tokens": proposal_round_workspace.proposal_tokens[
                                :batch_size, :proposal_width
                            ],
                            "proposal_logits": proposal_round_workspace.proposal_logits[
                                :proposal_rows
                            ].view(proposal_width, batch_size, vocab_size)[:, 0],
                            "proposal_probabilities": proposal_probabilities.view(
                                proposal_width, batch_size, candidate_count
                            )[:, 0],
                            "proposal_token_ids": proposal_token_ids.view(
                                proposal_width, batch_size, candidate_count
                            )[:, 0],
                            "target_logits": proposal_round_workspace.target_logits[
                                :target_rows
                            ].view(batch_size, verify_width, vocab_size)[0],
                            "target_hidden_states": audit_hidden_states[
                                :batch_size, :verify_width
                            ],
                            "target_probabilities": target_probabilities.view(
                                batch_size, verify_width, candidate_count
                            )[0],
                            "target_token_ids": target_token_ids.view(
                                batch_size, verify_width, candidate_count
                            )[0],
                            "uniforms": round_uniforms[:batch_size],
                            "accepted_draft_counts": (
                                proposal_round_workspace.acceptance_outputs.accepted_draft_counts[
                                    :batch_size
                                ]
                            ),
                            "committed_token_counts": (
                                proposal_round_workspace.acceptance_outputs.committed_token_counts[
                                    :batch_size
                                ]
                            ),
                            "committed_tokens": (
                                proposal_round_workspace.acceptance_outputs.committed_tokens[
                                    :batch_size
                                ]
                            ),
                            "rejected_rows": (
                                proposal_round_workspace.acceptance_outputs.rejected_rows[
                                    :batch_size
                                ]
                            ),
                            "bonus_rows": (
                                proposal_round_workspace.acceptance_outputs.bonus_rows[
                                    :batch_size
                                ]
                            ),
                            "invalid_rows": (
                                proposal_round_workspace.acceptance_outputs.invalid_rows[
                                    :batch_size
                                ]
                            ),
                            "output_codes": code_outputs,
                            "pending_after": pending_tokens,
                            "preceding_hidden": preceding_hidden,
                            "valid_kv_length": current_round_depth
                            + 1
                            + committed_count,
                            "key_cache": self.predictor_k_cache[
                                :,
                                :batch_size,
                                : current_round_depth + 1 + committed_count,
                            ],
                            "value_cache": self.predictor_v_cache[
                                :,
                                :batch_size,
                                : current_round_depth + 1 + committed_count,
                            ],
                        },
                    )
                else:
                    pass
                if active_count == 0:
                    break
                else:
                    pass
                staged_round_width = max(0, num_groups - 1 - round_depth)
        finally:
            self.code_predictor_spec_round_count += local_round_count
            self.code_predictor_spec_proposed_count += local_proposed_count
            self.code_predictor_spec_accepted_count += local_accepted_count
            self.code_predictor_spec_committed_count += local_committed_count
            self.code_predictor_spec_bonus_count += local_bonus_count
            self.code_predictor_spec_rejection_count += local_rejection_count
            self.code_predictor_spec_packed_forward_count += local_packed_forward_count
            self.code_predictor_spec_fused_predictor_forward_count += (
                local_fused_forward_count
            )
            self.code_predictor_spec_draft_head_evaluation_count += (
                local_draft_evaluation_count
            )
            self.code_predictor_spec_target_head_evaluation_count += (
                local_target_evaluation_count
            )
            self.code_predictor_spec_packed_query_slots += local_packed_query_slots
            self.code_predictor_spec_padded_query_slots += local_padded_query_slots
            for width, count in local_width_histogram.items():
                self.code_predictor_spec_effective_width_histogram[width] = (
                    self.code_predictor_spec_effective_width_histogram.get(width, 0)
                    + count
                )
            else:
                pass
            for backend, count in local_probability_backend_counts.items():
                self.code_predictor_spec_probability_backend_execution_counts[
                    backend
                ] += count
            else:
                pass
            for backend, count in local_acceptance_backend_counts.items():
                self.code_predictor_spec_acceptance_backend_execution_counts[
                    backend
                ] += count
            else:
                pass

        self.code_predictor_spec_correction_forward_watch = False
        self.code_predictor_spec_frame_count += batch_size
        return result_codes, summed_embeddings

    def configure_code_predictor_p16(self, enabled: bool) -> None:
        """Configure the one supported P16 execution path, or release it."""
        if type(enabled) is not bool:
            raise TypeError("P16 enable flag must be boolean")
        else:
            pass
        configuration_started = time.perf_counter()
        self.code_predictor_spec_requested_width = 4 if enabled else 0
        self.code_predictor_spec_effective_width = 0
        self.code_predictor_spec_fallback_reason = None
        self.code_predictor_spec_opening_graphs.clear()
        self.code_predictor_spec_opening_graph_disabled.clear()
        self.code_predictor_spec_round_graphs.clear()
        self.code_predictor_spec_round_graph_disabled.clear()
        self.code_predictor_spec_graph_pool = None
        self.code_predictor_spec_input_buffers.clear()
        self.code_predictor_spec_workspaces.clear()
        self.code_predictor_spec_active_workspace = None
        self.code_predictor_spec_lm_head_weights = None
        self.code_predictor_spec_metadata_device = None
        self.code_predictor_spec_metadata_host = None
        self.sub_speculative_seed_tensor = None
        self.sub_speculative_uniforms_tensor = None
        self.sub_speculative_rng_pool_tensor = None
        if not enabled:
            self.code_predictor_spec_configuration_time_ms = (
                time.perf_counter() - configuration_started
            ) * 1000.0
            return
        else:
            pass

        fallback_reason = self.code_predictor_p16_runtime_fallback_reason()
        if fallback_reason is not None:
            self.code_predictor_spec_fallback_reason = fallback_reason
            self.code_predictor_spec_configuration_time_ms = (
                time.perf_counter() - configuration_started
            ) * 1000.0
            return
        else:
            pass

        self.sub_speculative_seed_tensor = torch.zeros(
            1, device=self.predictor_device, dtype=torch.long
        )
        self.sub_speculative_uniforms_tensor = torch.empty(
            (1, 4, _P16_VERIFY_WIDTH),
            device=self.predictor_device,
            dtype=torch.float32,
        )
        self.sub_speculative_rng_pool_tensor = torch.empty(
            (1, 4, int(self.config.num_code_groups) - 2, 4),
            device=self.predictor_device,
            dtype=torch.float32,
        )
        head_count = int(self.config.num_code_groups) - 1
        vocab_size = int(self.config.code_predictor_config.vocab_size)
        head_weight = self.code_predictor.lm_head[0].weight
        self.code_predictor_spec_lm_head_weights = torch.empty(
            (head_count, vocab_size, int(head_weight.shape[1])),
            device=self.predictor_device,
            dtype=torch.bfloat16,
        )
        self.copy_code_predictor_spec_lm_head_weights()
        self.code_predictor_spec_metadata_device = torch.empty(
            (1, 9), device=self.predictor_device, dtype=torch.long
        )
        self.code_predictor_spec_metadata_host = torch.empty(
            (1, 9),
            dtype=torch.long,
            pin_memory=is_pin_memory_available(self.predictor_device),
        )
        hidden_size = int(self.predictor_fused_layers.shape.hidden_size)
        for verify_width in range(1, _P16_VERIFY_WIDTH + 1):
            self.code_predictor_spec_input_buffers[verify_width] = (
                SpeculativeVerifierInputs(
                    token_embeddings=torch.empty(
                        (1, verify_width, hidden_size),
                        device=self.predictor_device,
                        dtype=self.predictor_k_cache.dtype,
                    ),
                    positions=torch.empty(
                        verify_width, device=self.predictor_device, dtype=torch.long
                    ),
                    target_head_indices=torch.empty(
                        (1, verify_width),
                        device=self.predictor_device,
                        dtype=torch.long,
                    ),
                    valid_queries=torch.empty(
                        (1, verify_width),
                        device=self.predictor_device,
                        dtype=torch.bool,
                    ),
                )
            )
        self.code_predictor_spec_effective_width = 4
        self.reserve_code_predictor_speculative_workspace()
        self.code_predictor_spec_configuration_time_ms = (
            time.perf_counter() - configuration_started
        ) * 1000.0

    def copy_code_predictor_spec_lm_head_weights(self) -> None:
        """Refresh the persistent grouped-head layout after checkpoint loading."""
        packed_weights = self.code_predictor_spec_lm_head_weights
        if packed_weights is None:
            return
        else:
            pass
        for head_index, lm_head in enumerate(self.code_predictor.lm_head):
            packed_weights[head_index].copy_(lm_head.weight)

    def predictor_speculative_stats(self) -> PredictorSpeculativeStats:
        average_codes_per_round = (
            self.code_predictor_spec_committed_count
            / self.code_predictor_spec_round_count
            if self.code_predictor_spec_round_count > 0
            else 0.0
        )
        workspace_tensors: dict[int, torch.Tensor] = {}
        for workspace in self.code_predictor_spec_workspaces.values():
            for tensor in workspace.tensor_storage:
                if tensor.numel() > 0:
                    workspace_tensors[tensor.data_ptr()] = tensor
                else:
                    pass
        for verifier_inputs in self.code_predictor_spec_input_buffers.values():
            for tensor in (
                verifier_inputs.token_embeddings,
                verifier_inputs.positions,
                verifier_inputs.target_head_indices,
                verifier_inputs.valid_queries,
            ):
                if tensor.numel() > 0:
                    workspace_tensors[tensor.data_ptr()] = tensor
                else:
                    pass
        for opening_graph in self.code_predictor_spec_opening_graphs.values():
            for tensor in (
                opening_graph.layer0_codes,
                opening_graph.talker_hidden,
                opening_graph.semantic_positions,
            ):
                if tensor.numel() > 0:
                    workspace_tensors[tensor.data_ptr()] = tensor
                else:
                    pass
        for tensor in (
            self.sub_speculative_seed_tensor,
            self.sub_speculative_uniforms_tensor,
            self.sub_speculative_rng_pool_tensor,
            self.code_predictor_spec_metadata_device,
            self.code_predictor_spec_metadata_host,
        ):
            if tensor is not None and tensor.numel() > 0:
                workspace_tensors[tensor.data_ptr()] = tensor
            else:
                pass
        workspace_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in workspace_tensors.values()
        )
        return {
            "requested_width": self.code_predictor_spec_requested_width,
            "effective_width": self.code_predictor_spec_effective_width,
            "graph_mode": (
                "full_round"
                if self.code_predictor_spec_requested_width > 0
                else "disabled"
            ),
            "probability_backend": "split_top_k",
            "probability_backend_execution_counts": dict(
                self.code_predictor_spec_probability_backend_execution_counts
            ),
            "acceptance_backend": "compact",
            "acceptance_backend_execution_counts": dict(
                self.code_predictor_spec_acceptance_backend_execution_counts
            ),
            "lm_head_backend": "warp_accumulate",
            "rng_backend": "frame_pool",
            "fused_predictor_scratch_rows": (
                0
                if self.predictor_fused_layers is None
                else self.predictor_fused_layers.max_rows
            ),
            "effective_width_histogram": {
                str(width): count
                for width, count in self.code_predictor_spec_effective_width_histogram.items()
            },
            "fallback_reason": self.code_predictor_spec_fallback_reason,
            "frames": self.code_predictor_spec_frame_count,
            "output_code_frames": self.code_predictor_output_frame_count,
            "rounds": self.code_predictor_spec_round_count,
            "proposed_draft_tokens": self.code_predictor_spec_proposed_count,
            "accepted_draft_tokens": self.code_predictor_spec_accepted_count,
            "committed_codes": (
                self.code_predictor_spec_committed_count
                + self.code_predictor_spec_frame_count
            ),
            "round_committed_codes": self.code_predictor_spec_committed_count,
            "bonus_tokens": self.code_predictor_spec_bonus_count,
            "rejections": self.code_predictor_spec_rejection_count,
            "average_codes_per_round": average_codes_per_round,
            "packed_transformer_forwards": (
                self.code_predictor_spec_packed_forward_count
            ),
            "packed_query_slots": self.code_predictor_spec_packed_query_slots,
            "padded_query_slots": self.code_predictor_spec_padded_query_slots,
            "draft_head_evaluations": self.code_predictor_spec_draft_head_evaluation_count,
            "target_head_evaluations": self.code_predictor_spec_target_head_evaluation_count,
            "fused_predictor_forwards": self.code_predictor_spec_fused_predictor_forward_count,
            "generic_predictor_forwards": self.code_predictor_generic_forward_count,
            "standalone_correction_forward_count": (
                self.code_predictor_spec_standalone_correction_forward_count
            ),
            "full_round_graph_replays": (
                self.code_predictor_spec_full_round_graph_replay_count
            ),
            "full_round_graph_replay_coverage": (
                self.code_predictor_spec_full_round_graph_replay_count
                / self.code_predictor_spec_round_count
                if self.code_predictor_spec_round_count > 0
                else 0.0
            ),
            "full_round_graph_capture_time_ms": (
                self.code_predictor_spec_full_round_graph_capture_time_ms
            ),
            "opening_pair_graph_replays": (
                self.code_predictor_spec_opening_graph_replay_count
            ),
            "opening_pair_graph_replay_coverage": (
                self.code_predictor_spec_opening_graph_replay_count
                / self.code_predictor_spec_frame_count
                if self.code_predictor_spec_frame_count > 0
                else 0.0
            ),
            "opening_pair_graph_capture_time_ms": (
                self.code_predictor_spec_opening_graph_capture_time_ms
            ),
            "graph_capture_failures": (
                self.code_predictor_spec_full_round_graph_capture_failure_count
                + self.code_predictor_spec_opening_graph_capture_failure_count
            ),
            "graph_captures": (
                self.code_predictor_spec_full_round_graph_capture_count
                + self.code_predictor_spec_opening_graph_capture_count
            ),
            "full_round_graph_captures": (
                self.code_predictor_spec_full_round_graph_capture_count
            ),
            "full_round_graph_capture_failures": (
                self.code_predictor_spec_full_round_graph_capture_failure_count
            ),
            "opening_pair_graph_captures": (
                self.code_predictor_spec_opening_graph_capture_count
            ),
            "opening_pair_graph_capture_failures": (
                self.code_predictor_spec_opening_graph_capture_failure_count
            ),
            "graph_capture_memory_delta_bytes": (
                self.code_predictor_spec_graph_memory_delta_bytes
            ),
            "graph_capture_time_ms": self.code_predictor_spec_graph_capture_time_ms,
            "workspace_bytes": workspace_bytes,
            "workspace_allocations": (
                self.code_predictor_spec_workspace_allocation_count
            ),
            "configuration_time_ms": self.code_predictor_spec_configuration_time_ms,
            "nvtx_enabled": self.code_predictor_spec_nvtx_enabled,
            "native_fallback_count": self.code_predictor_spec_native_fallback_count,
            "attempted_frames": (
                self.code_predictor_spec_frame_count
                + self.code_predictor_spec_native_fallback_count
            ),
            "native_fallback_fraction": (
                self.code_predictor_spec_native_fallback_count
                / (
                    self.code_predictor_spec_frame_count
                    + self.code_predictor_spec_native_fallback_count
                )
                if (
                    self.code_predictor_spec_frame_count
                    + self.code_predictor_spec_native_fallback_count
                )
                > 0
                else 0.0
            ),
        }

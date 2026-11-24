# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Qwen3-TTS talker wrapper."""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    TypedDict,
)

import torch
from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod,
    get_bf16_gemm_backend,
)
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_context, get_exec, get_parallel, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import add_prefix
from sglang.srt.utils.common import is_pin_memory_available
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import (  # noqa: E501
    Qwen3OmniMoeTalkerDenseMLP,
    ResizeMLP,
    bind_default_weight_loaders,
)
from sglang_omni.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextAttention,
)
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.cutedsl.runtime import (
    HEAD_DIM,
    KV_HEADS,
    MAX_DECODE_ROWS,
    QTTS_BACKEND,
)
from sglang_omni.models.qwen3_tts.predictor_kernels import (
    gather_codec_embedding_and_add,
    short_cache_gqa_npu,
)
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    build_speculative_probability_candidates_selected,
    fused_raw_logit_block_k,
    sample_codes_and_gather_codec_embedding,
    sample_from_logits_with_seed_top_k_top_p,
    sample_from_logprobs_with_seed_npu,
    sample_from_sorted_logprobs_with_seed_small_k,
    sample_top_k,
)
from sglang_omni.models.qwen3_tts.speaker_encoder_cuda_graph import (
    Qwen3TTSSpeakerEncoderCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    sample_compact_probability_rows_fused,
)
from sglang_omni.models.qwen3_tts.speculative_commit import (
    commit_speculative_codes,
    commit_speculative_state,
)
from sglang_omni.models.qwen3_tts.speculative_prepare import (
    SpeculativeVerifierInputs,
    prepare_speculative_verification_inputs,
)
from sglang_omni.models.qwen3_tts.speculative_rng import (
    derive_speculative_request_seed,
    speculative_uniforms_all_purposes,
    speculative_uniforms_frame_pool,
)
from sglang_omni.models.qwen3_tts.speculative_runtime import (
    accept_speculative_candidates_compact,
    speculative_sampling_fallback_reason,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph
from sglang_omni.sampling.sampler_compile import (
    SamplerCompileForms,
    compile_sampler_functions,
)
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.utils.predictor_grouped_heads import (
    grouped_lm_head_cuda_available,
    grouped_lm_head_linear,
    grouped_lm_head_warp_gemv_available,
)
from sglang_omni.utils.predictor_layers import (
    ATTENTION_ROWS,
    decode_attention,
    fused_predictor_scratch_rows,
    resolve_fused_predictor_layers,
    resolve_fused_predictor_width,
)
from sglang_omni.utils.predictor_speculation import (
    is_fatal_cuda_capture_error,
    resolve_speculative_capability,
    resolve_speculative_round_width,
    speculative_resources_requested,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import get_group_rank_size
from sglang_omni.vendor.sglang.layers import ReplicatedLinear, RMSNorm
from sglang_omni.vendor.sglang.models import (
    FusedSetKVBufferArg,
    apply_qk_norm,
    enable_fused_set_kv_buffer,
)

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSTokenizer
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSConfig,
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )

    from sglang_omni.models.qwen3_tts.request_builders import VoicePrompt
else:
    pass

logger = logging.getLogger(__name__)

if QTTS_BACKEND == "cutedsl":
    from sglang_omni.models.qwen3_tts.cutedsl.argmax_embedding import argmax_embedding
    from sglang_omni.models.qwen3_tts.cutedsl.embedding_update import embedding_update
    from sglang_omni.models.qwen3_tts.cutedsl.runtime import supports_auxiliary
    from sglang_omni.models.qwen3_tts.cutedsl.sampling import (
        sample_logits,
        sample_sorted,
    )
    from sglang_omni.models.qwen3_tts.cutedsl_backend import build_cutedsl_layers
else:
    pass

PredictorGraphSignature: TypeAlias = tuple[
    Literal["argmax", "sampled"], int, bool, bool, bool
]
PredictorGraphKey: TypeAlias = tuple[
    int, Literal["argmax", "sampled"], int, bool, bool, bool
]

QTTS_PREDICTOR_GRAPH_ENV = "SGLANG_OMNI_QTTS_PREDICTOR_GRAPH"
_PREDICTOR_GRAPH_MAX_LAZY_KEYS = 32
_PREDICTOR_GRAPH_MAX_FAILURES = 8
_PREDICTOR_GRAPH_WARMUP_PASSES = 2
# Note: (Jiaxin Deng) 50 is on the ladder because it is the family checkpoint
# default, keeping the dominant signature's kernel width exactly as before.
_PREDICTOR_TOP_K_LADDER = (4, 8, 16, 32, 50, 64, 128, 256, 512, 1024)


def predictor_graph_env_override() -> bool | None:
    """The operator's explicit choice, or None to leave it to the platform."""
    value = os.environ.get(QTTS_PREDICTOR_GRAPH_ENV)
    if value is None:
        return None
    else:
        pass
    return value.strip().lower() not in ("0", "false", "off", "no")


def predictor_gqa_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    is_causal: bool,
) -> torch.Tensor:
    """Run Predictor GQA with shape-specific kernels for non-causal NPU queries."""
    if q.device.type == "npu" and not is_causal:
        output = short_cache_gqa_npu(q, key, value)
        if output is not None:
            return output
        else:
            pass
        fused_attention = getattr(
            getattr(torch.ops, "npu", None),
            "npu_fused_infer_attention_score",
            None,
        )
        if fused_attention is not None:
            output, _ = fused_attention(
                q.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                input_layout="BSND",
                scale=1.0 / math.sqrt(q.shape[-1]),
            )
            return output.transpose(1, 2)
        else:
            pass
    else:
        pass

    return torch.nn.functional.scaled_dot_product_attention(
        q,
        key,
        value,
        is_causal=is_causal,
        enable_gqa=True,
    )


def quantize_predictor_top_k(max_top_k: int, vocab_size: int) -> int | None:
    """Smallest ladder width covering max_top_k, or None to use the full sort."""
    for step in _PREDICTOR_TOP_K_LADDER:
        if step >= max_top_k:
            return step if step < vocab_size else None
        else:
            pass
    return None


def predictor_signature_terms(
    sampled_top_ks: list[int],
    sampled_top_ps: list[float],
    vocab_size: int,
) -> tuple[int, bool, bool]:
    """The one signature rule, shared by the batch path and the startup capture."""
    bounded_top_ks = [
        int(top_k) for top_k in sampled_top_ks if 0 < int(top_k) < vocab_size
    ]
    has_top_p = any(0.0 < float(top_p) < 1.0 for top_p in sampled_top_ps)
    has_unbounded_top_k = len(bounded_top_ks) != len(sampled_top_ks)
    max_top_k = 0
    max_bounded_top_k = max(bounded_top_ks, default=0)
    if max_bounded_top_k > 0 and not has_unbounded_top_k:
        # Note: (Jiaxin Deng) ladder-quantized so predictor-graph keys are
        # shared across request top_k values; per-row masks keep true k.
        quantized = quantize_predictor_top_k(max_bounded_top_k, vocab_size)
        if quantized is None:
            has_unbounded_top_k = True
        else:
            max_top_k = quantized
    else:
        pass
    return max_top_k, has_top_p, has_unbounded_top_k


def sample_seeded_categorical(
    logprobs: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if logprobs.device.type == "npu":
        return sample_from_logprobs_with_seed_npu(logprobs, seeds, positions)
    else:
        pass
    return multinomial_with_seed(logprobs, seeds, positions).view(-1)


class PredictorGraphStats(TypedDict):
    enabled: bool | None
    startup_graph_count: int
    lazy_graph_count: int
    replays: int
    eager_runs: int
    capacity_fallbacks: int
    capture_failures: int


class PredictorSpeculativeStats(TypedDict):
    requested_width: int
    effective_width: int
    graph_mode: str
    probability_backend: str
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
    graph_replays: int
    verifier_graph_replays: int
    graph_capture_failures: int
    graph_captures: int
    graph_capture_memory_delta_bytes: int
    graph_capture_time_ms: float
    configuration_time_ms: float
    native_fallback_count: int
    attempted_frames: int
    native_fallback_fraction: float


class PredictorDecodeGraph:
    """Device graph over the full per-token predictor chain for one batch bucket.

    One graph per (bucket, sampling signature): the signature pins the host
    branches of the sampling path (argmax vs sampled, top-k bound, top-p
    presence), so replay reproduces the bits of the eager pass.
    Per-step inputs reach the captured region through persistent device
    buffers written with device-side copies before replay. Holds no reference
    to the talker: a cycle would put the graph's finalizer behind the
    cyclic collector.
    """

    def __init__(
        self,
        batch_size: int,
        signature: PredictorGraphSignature,
        *,
        device: torch.device,
        hidden_size: int,
        hidden_dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        self.signature = signature
        self.device = device
        self.device_module = torch.get_device_module(device)
        self.layer0_codes = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.talker_hidden = torch.zeros(
            batch_size, 1, hidden_size, dtype=hidden_dtype, device=device
        )
        self.semantic_positions = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self.graph: ReplayableGraph | None = None
        self.result_codes: torch.Tensor | None = None
        self.summed_embeddings: torch.Tensor | None = None

    @torch.no_grad()
    def replay(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        live = layer0_codes.shape[0]
        if live > self.batch_size:
            raise ValueError(
                "Qwen3-TTS predictor graph bucket is too small: "
                f"bucket={self.batch_size}, live={live}"
            )
        else:
            pass
        with self.device_module.device(self.device):
            self.layer0_codes[:live].copy_(layer0_codes)
            self.talker_hidden[:live].copy_(talker_hidden)
            if semantic_positions is None:
                self.semantic_positions.zero_()
            else:
                self.semantic_positions[:live].copy_(semantic_positions.reshape(live))
            if live < self.batch_size:
                self.layer0_codes[live:].zero_()
                self.talker_hidden[live:].zero_()
                if semantic_positions is not None:
                    self.semantic_positions[live:].zero_()
                else:
                    pass
            else:
                pass
            self.graph.replay()
        assert self.result_codes is not None
        assert self.summed_embeddings is not None
        return self.result_codes[:live], self.summed_embeddings[:live]


class PredictorSpeculativeVerifyGraph:
    """Persistent graph for packed verification and grouped target heads."""

    def __init__(
        self,
        batch_size: int,
        verify_width: int,
        *,
        device: torch.device,
        hidden_size: int,
        hidden_dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        self.verify_width = verify_width
        self.device = device
        self.device_module = torch.get_device_module(device)
        self.token_embeds = torch.zeros(
            batch_size,
            verify_width,
            hidden_size,
            device=device,
            dtype=hidden_dtype,
        )
        self.positions = torch.full(
            (batch_size * verify_width,),
            -1,
            device=device,
            dtype=torch.long,
        )
        self.target_head_indices = (
            torch.arange(1, verify_width + 1, device=device, dtype=torch.long)
            .expand(batch_size, verify_width)
            .contiguous()
        )
        self.target_valid_queries = torch.ones(
            (batch_size, verify_width), device=device, dtype=torch.bool
        )
        self.target_hidden_rows = torch.arange(
            batch_size * verify_width, device=device, dtype=torch.long
        )
        self.graph: ReplayableGraph | None = None
        self.hidden_states: torch.Tensor | None = None
        self.target_logits: torch.Tensor | None = None

    @torch.no_grad()
    def replay(
        self,
        token_embeds: torch.Tensor,
        positions: torch.Tensor,
        target_head_indices: torch.Tensor,
        target_valid_queries: torch.Tensor,
    ) -> torch.Tensor:
        live = int(token_embeds.shape[0])
        if live > self.batch_size or token_embeds.shape[1] != self.verify_width:
            raise ValueError("packed Predictor graph shape does not match its bucket")
        else:
            pass
        if positions.shape != (live * self.verify_width,):
            raise ValueError("packed Predictor graph positions have an invalid shape")
        elif target_head_indices.shape != (live, self.verify_width):
            raise ValueError(
                "packed Predictor graph target heads have an invalid shape"
            )
        elif target_valid_queries.shape != (live, self.verify_width):
            raise ValueError("packed Predictor graph target mask has an invalid shape")
        else:
            pass
        with self.device_module.device(self.device):
            self.token_embeds[:live].copy_(token_embeds)
            self.positions[: live * self.verify_width].copy_(positions)
            self.target_head_indices[:live].copy_(target_head_indices)
            self.target_valid_queries[:live].copy_(target_valid_queries)
            if live < self.batch_size:
                self.token_embeds[live:].zero_()
                self.positions[live * self.verify_width :].fill_(-1)
                self.target_head_indices[live:].zero_()
                self.target_valid_queries[live:].zero_()
            else:
                pass
            assert self.graph is not None
            self.graph.replay()
        assert self.hidden_states is not None
        assert self.target_logits is not None
        return self.hidden_states[:live]


class Qwen3TTSTalkerDecoderLayer(nn.Module):
    def __init__(
        self,
        config: "Qwen3TTSTalkerConfig | Qwen3TTSTalkerCodePredictorConfig",
        layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            config=config,
            prefix=add_prefix("self_attn", prefix),
            dual_chunk_attention_config=None,
            alt_stream=None,
        )
        self.mlp = Qwen3OmniMoeTalkerDenseMLP(
            config.hidden_size,
            config.intermediate_size,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3TTSTalkerTextModel(nn.Module):
    def __init__(self, config: "Qwen3TTSTalkerConfig", prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3TTSTalkerDecoderLayer(
                    config,
                    idx,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cutedsl_layers: nn.ModuleList | None = None

        max_batch_size = get_schedule().max_running_requests
        self.decode_feedback_embedding = nn.Embedding(
            max_batch_size,
            config.hidden_size,
            device=self.codec_embedding.weight.device,
            dtype=self.codec_embedding.weight.dtype,
        )
        self.decode_feedback_embedding.weight.requires_grad_(False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        return self.text_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_decode():
            hidden_states = self.decode_feedback_embedding(input_ids)
        else:
            hidden_states = input_embeds

        if (
            self.cutedsl_layers is not None
            and forward_batch.forward_mode.is_decode()
            and 1 <= hidden_states.shape[0] <= MAX_DECODE_ROWS
            and enable_fused_set_kv_buffer(forward_batch)
        ):
            cache_pool = get_token_to_kv_pool()
            first_cache = cache_pool.get_key_buffer(self.start_layer)
            if (
                first_cache.dtype == torch.bfloat16
                and first_cache.is_contiguous()
                and first_cache.shape[1:] == (KV_HEADS, HEAD_DIM)
            ):
                token_positions = positions[0] if positions.ndim == 2 else positions
                deferred = self.cutedsl_layers[0].pre_attention.deferred_residual
                residual = None
                combined = torch.empty_like(hidden_states) if deferred else None
                next_residual = torch.empty_like(hidden_states) if deferred else None
                for idx in range(self.start_layer, self.end_layer):
                    attention = self.layers[idx].self_attn
                    fused_layer = self.cutedsl_layers[idx]
                    q = fused_layer.pre_attention(
                        hidden_states,
                        token_positions,
                        attention.rotary_emb.cos_sin_cache,
                        forward_batch.out_cache_loc,
                        cache_pool.get_key_buffer(attention.layer_id),
                        cache_pool.get_value_buffer(attention.layer_id),
                        residual=residual,
                        residual_output=combined,
                    )
                    attention_output = attention.attn(
                        q, None, None, forward_batch, save_kv_cache=False
                    )
                    if deferred:
                        hidden_states = fused_layer.post_attention(
                            attention_output, combined, residual_output=next_residual
                        )
                        residual = next_residual
                    else:
                        hidden_states = fused_layer.post_attention(
                            attention_output, hidden_states
                        )
                if deferred:
                    return self.norm(hidden_states, residual)[0]
                else:
                    return self.norm(hidden_states)
            else:
                pass
        else:
            pass

        residual = None
        layers = self.layers
        for idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = layers[idx](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )
        if residual is None:
            return self.norm(hidden_states)
        else:
            pass
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def load_codec_table(
    param: nn.Parameter, loaded_weight: torch.Tensor, codebook_index: int
) -> None:
    param.data[codebook_index].copy_(loaded_weight)


class Qwen3TTSCodecTables(nn.Module):
    """The residual codebooks' embedding tables as one (codebooks, vocab, hidden) weight;
    each checkpoint table loads into its codebook's slice."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(num_codebooks, vocab_size, hidden_size), requires_grad=False
        )
        self.weight.weight_loader = load_codec_table
        self.register_buffer(
            "row_offsets", torch.arange(num_codebooks) * vocab_size, persistent=False
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed (..., codebooks) codes as (..., codebooks, hidden) rows in one lookup."""
        return torch.nn.functional.embedding(
            codes + self.row_offsets, self.weight.flatten(0, 1)
        )

    @staticmethod
    def make_params_mapping(num_codebooks: int) -> dict[str, tuple[str, int]]:
        """Each codebook's checkpoint table name -> the stacked weight's name and the codebook."""
        module_name = "code_predictor.model.codec_embedding"
        return {
            f"{module_name}.{codebook_index}.weight": (
                f"{module_name}.weight",
                codebook_index,
            )
            for codebook_index in range(num_codebooks)
        }


class Qwen3TTSCodePredictor(nn.Module):
    def __init__(self, config: "Qwen3TTSTalkerConfig", prefix: str = "") -> None:
        super().__init__()
        self.config = config
        cp_config = config.code_predictor_config
        self.model = nn.Module()
        self.model.codec_embedding = Qwen3TTSCodecTables(
            config.num_code_groups - 1, cp_config.vocab_size, config.hidden_size
        )
        self.model.layers = nn.ModuleList(
            [
                Qwen3TTSTalkerDecoderLayer(
                    cp_config,
                    idx,
                    prefix=add_prefix(f"model.layers.{idx}", prefix),
                )
                for idx in range(cp_config.num_hidden_layers)
            ]
        )
        self.model.norm = RMSNorm(cp_config.hidden_size, eps=cp_config.rms_norm_eps)
        self.lm_head = nn.ModuleList(
            [
                ReplicatedLinear(
                    cp_config.hidden_size,
                    cp_config.vocab_size,
                    bias=False,
                    prefix=add_prefix(f"lm_head.{idx}", prefix),
                )
                for idx in range(config.num_code_groups - 1)
            ]
        )
        if cp_config.hidden_size != config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                config.hidden_size, cp_config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = None

    def project_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.small_to_mtp_projection is None:
            return hidden_states
        else:
            pass
        return self.small_to_mtp_projection(hidden_states)


@dataclass(frozen=True, kw_only=True)
class Qwen3TTSPromptConstants:
    """Prompt rows that depend on the checkpoint alone, each embedded once the way a
    request embeds it, as (1, rows, hidden); feedback_pad_embed is the one (hidden,)
    row decode reads after the text ends, in the codec embedding's dtype."""

    tts_bos_embed: torch.Tensor
    tts_eos_embed: torch.Tensor
    tts_pad_embed: torch.Tensor
    codec_pad_bos_embed: torch.Tensor
    codec_prefill_embeds: Mapping[int | None, torch.Tensor]
    custom_voice_speaker_embeds: Mapping[str, torch.Tensor]
    feedback_pad_embed: torch.Tensor


class Qwen3TTSPromptBuilderMixin:
    """Prompt construction shared by the talker and the standalone prompt frontend.

    Expects model (embedding tables), text_projection,
    code_predictor.model.codec_embedding, speaker_encoder, speaker_encoder_graph_runner,
    speech_tokenizer, config, root_config and speaker_encoder_sample_rate on the instance.
    """

    @property
    def device(self) -> torch.device:
        return self.model.codec_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.codec_embedding.weight.dtype

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_text_embeddings(self) -> nn.Embedding:
        return self.model.get_text_embeddings()

    def load_speech_tokenizer(
        self, speech_tokenizer: "Qwen3TTSTokenizer | None"
    ) -> None:
        self.speech_tokenizer = speech_tokenizer

    def get_supported_languages(self):
        return ["auto", *list(self.config.codec_language_id.keys())]

    def get_supported_speakers(self):
        return (self.config.spk_id or {}).keys()

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr):
        if sr != self.speaker_encoder_sample_rate:
            raise ValueError(
                f"Expected {self.speaker_encoder_sample_rate}Hz reference audio"
            )
        else:
            pass
        if self.speaker_encoder_graph_runner is None:
            raise RuntimeError("Qwen3-TTS speaker encoder is not loaded")
        else:
            pass
        return self.speaker_encoder_graph_runner.embed(audio)

    @torch.inference_mode()
    def generate_speaker_prompt(
        self, voice_clone_prompt: VoicePrompt
    ) -> list[torch.Tensor]:
        return [
            emb.to(self.device).to(self.dtype)
            for emb in voice_clone_prompt["ref_spk_embedding"]
        ]

    def build_instruct_embed(
        self,
        instruct_id: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if instruct_id is None:
            return None
        else:
            pass
        return self.text_projection(self.get_text_embeddings()(instruct_id))

    def build_prompt_constants(self) -> Qwen3TTSPromptConstants:
        """Embed the checkpoint-only prompt rows, each at the shape a request embeds it."""
        special_ids = torch.tensor(
            [
                [
                    self.root_config.tts_bos_token_id,
                    self.root_config.tts_eos_token_id,
                    self.root_config.tts_pad_token_id,
                ]
            ],
            device=self.device,
            dtype=torch.long,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.text_projection(
            self.get_text_embeddings()(special_ids)
        ).chunk(3, dim=1)
        codec_embedding = self.get_input_embeddings()
        codec_prefill_embeds: dict[int | None, torch.Tensor] = {}
        for language_id in (None, *set(self.config.codec_language_id.values())):
            if language_id is None:
                codec_prefill = [
                    self.config.codec_nothink_id,
                    self.config.codec_think_bos_id,
                    self.config.codec_think_eos_id,
                ]
            else:
                codec_prefill = [
                    self.config.codec_think_id,
                    self.config.codec_think_bos_id,
                    language_id,
                    self.config.codec_think_eos_id,
                ]
            codec_prefill_embeds[language_id] = codec_embedding(
                torch.tensor([codec_prefill], device=self.device, dtype=torch.long)
            )
        speaker_ids = self.config.spk_id or {}
        return Qwen3TTSPromptConstants(
            tts_bos_embed=tts_bos_embed,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            codec_pad_bos_embed=codec_embedding(
                torch.tensor(
                    [[self.config.codec_pad_id, self.config.codec_bos_id]],
                    device=self.device,
                    dtype=torch.long,
                )
            ),
            codec_prefill_embeds=codec_prefill_embeds,
            custom_voice_speaker_embeds={
                str(name)
                .lower(): codec_embedding(
                    torch.tensor([speaker_id], device=self.device, dtype=torch.long)
                )
                .view(1, 1, -1)
                for name, speaker_id in speaker_ids.items()
            },
            feedback_pad_embed=self.text_projection(
                self.get_text_embeddings()(
                    torch.tensor(
                        [[self.root_config.tts_pad_token_id]],
                        device=self.device,
                        dtype=torch.long,
                    )
                )
            )
            .squeeze(0)
            .squeeze(0)
            .to(
                device=codec_embedding.weight.device, dtype=codec_embedding.weight.dtype
            ),
        )

    def resolve_language_id(
        self,
        *,
        language: str,
        voice: str | None = None,
    ) -> int | None:
        # Note(yzxiao): QwenLM/Qwen3-TTS (qwen-tts 0.1.1) also applies speaker
        # dialects to Chinese. Keep explicit languages unchanged here so existing
        # Chinese requests do not switch to a dialect when using Eric or Dylan.
        if language.lower() != "auto":
            return self.config.codec_language_id[language.lower()]
        else:
            pass
        if voice is None:
            return None
        else:
            pass
        spk_is_dialect = self.config.spk_is_dialect or {}
        dialect = spk_is_dialect.get(voice.lower())
        if isinstance(dialect, str) and dialect:
            return self.config.codec_language_id.get(dialect)
        else:
            pass
        return None

    def finish_text_prompt(
        self,
        *,
        talker_input_embed: torch.Tensor,
        input_id: torch.Tensor,
        codec_last_embed: torch.Tensor,
        prompt_constants: Qwen3TTSPromptConstants,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codec_pad_bos_embed = prompt_constants.codec_pad_bos_embed
        if non_streaming_mode:
            text_all = self.text_projection(
                self.get_text_embeddings()(input_id[:, 3:-5])
            )
            text_all = torch.cat([text_all, prompt_constants.tts_eos_embed], dim=1)
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    text_all
                    + codec_pad_bos_embed[:, :1].expand(-1, int(text_all.shape[1]), -1),
                    prompt_constants.tts_pad_embed + codec_pad_bos_embed[:, 1:],
                ],
                dim=1,
            )
            return talker_input_embed, prompt_constants.tts_pad_embed
        else:
            pass

        first_text = (
            self.text_projection(self.get_text_embeddings()(input_id[:, 3:4]))
            + codec_last_embed
        )
        talker_input_embed = torch.cat([talker_input_embed, first_text], dim=1)
        trailing_text_hidden = torch.cat(
            [
                self.text_projection(self.get_text_embeddings()(input_id[:, 4:-5])),
                prompt_constants.tts_eos_embed,
            ],
            dim=1,
        )
        return talker_input_embed, trailing_text_hidden

    def apply_instruct_prefix(
        self,
        talker_input_embed: torch.Tensor,
        instruct_id: torch.Tensor | None,
    ) -> torch.Tensor:
        instruct_embed = self.build_instruct_embed(instruct_id)
        if instruct_embed is None:
            return talker_input_embed
        else:
            pass
        return torch.cat([instruct_embed, talker_input_embed], dim=1)

    def build_conditioned_prompt_prefix(
        self,
        *,
        input_id: torch.Tensor,
        codec_input: torch.Tensor,
        prompt_constants: Qwen3TTSPromptConstants,
    ) -> torch.Tensor:
        role_embed = self.text_projection(self.get_text_embeddings()(input_id[:, :3]))
        prompt_embed = (
            torch.cat(
                [
                    prompt_constants.tts_pad_embed.expand(
                        -1, codec_input.shape[1] - 2, -1
                    ),
                    prompt_constants.tts_bos_embed,
                ],
                dim=1,
            )
            + codec_input[:, :-1]
        )
        return torch.cat([role_embed, prompt_embed], dim=1)

    def build_voice_clone_inputs(
        self,
        *,
        input_id: torch.Tensor,
        ref_id: torch.Tensor | None,
        voice_clone_prompt: VoicePrompt,
        language: str,
        non_streaming_mode: bool,
        prompt_constants: Qwen3TTSPromptConstants,
        instruct_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
        speaker_embed = voice_clone_spk_embeds[0]

        codec_input = torch.cat(
            [
                prompt_constants.codec_prefill_embeds[
                    self.resolve_language_id(language=language)
                ],
                speaker_embed.view(1, 1, -1),
                prompt_constants.codec_pad_bos_embed,
            ],
            dim=1,
        )
        talker_input_embed = self.build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            prompt_constants=prompt_constants,
        )

        ref_code = None
        ref_codes = voice_clone_prompt.get("ref_code")
        if ref_codes is not None:
            ref_code = ref_codes[0]
        else:
            pass

        if ref_code is not None and voice_clone_prompt["icl_mode"][0]:
            if ref_id is None:
                raise ValueError("Qwen3-TTS ICL mode requires ref_text tokens")
            else:
                pass
            icl_embed, trailing_text_hidden = self.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id[:, 3:-2],
                ref_code=ref_code.to(self.device),
                prompt_constants=prompt_constants,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_embed], dim=1)
        else:
            talker_input_embed, trailing_text_hidden = self.finish_text_prompt(
                talker_input_embed=talker_input_embed,
                input_id=input_id,
                codec_last_embed=codec_input[:, -1:],
                prompt_constants=prompt_constants,
                non_streaming_mode=non_streaming_mode,
            )

        talker_input_embed = self.apply_instruct_prefix(
            talker_input_embed,
            instruct_id,
        )
        attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]), device=self.device, dtype=torch.long
        )
        return talker_input_embed, attention_mask, trailing_text_hidden, ref_code

    def build_custom_voice_inputs(
        self,
        *,
        input_id: torch.Tensor,
        voice: str,
        language: str,
        non_streaming_mode: bool,
        prompt_constants: Qwen3TTSPromptConstants,
        instruct_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        speaker_embeds = prompt_constants.custom_voice_speaker_embeds
        if not speaker_embeds:
            raise ValueError(
                "Qwen3-TTS CustomVoice requires a checkpoint with configured spk_id"
            )
        else:
            pass
        speaker_key = voice.lower()
        if speaker_key not in speaker_embeds:
            supported = ", ".join(sorted(str(key) for key in self.config.spk_id))
            raise ValueError(
                f"Unsupported Qwen3-TTS CustomVoice speaker {voice!r}. "
                f"Supported speakers: {supported}"
            )
        else:
            pass

        codec_input = torch.cat(
            [
                prompt_constants.codec_prefill_embeds[
                    self.resolve_language_id(language=language, voice=speaker_key)
                ],
                speaker_embeds[speaker_key],
                prompt_constants.codec_pad_bos_embed,
            ],
            dim=1,
        )
        talker_input_embed = self.build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            prompt_constants=prompt_constants,
        )
        talker_input_embed, trailing_text_hidden = self.finish_text_prompt(
            talker_input_embed=talker_input_embed,
            input_id=input_id,
            codec_last_embed=codec_input[:, -1:],
            prompt_constants=prompt_constants,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = self.apply_instruct_prefix(
            talker_input_embed,
            instruct_id,
        )
        attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]), device=self.device, dtype=torch.long
        )
        return talker_input_embed, attention_mask, trailing_text_hidden, None

    def build_voice_design_inputs(
        self,
        *,
        input_id: torch.Tensor,
        language: str,
        non_streaming_mode: bool,
        prompt_constants: Qwen3TTSPromptConstants,
        instruct_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        if instruct_id is None:
            raise ValueError("Qwen3-TTS VoiceDesign requires instructions")
        else:
            pass

        codec_input = torch.cat(
            [
                prompt_constants.codec_prefill_embeds[
                    self.resolve_language_id(language=language)
                ],
                prompt_constants.codec_pad_bos_embed,
            ],
            dim=1,
        )
        talker_input_embed = self.build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            prompt_constants=prompt_constants,
        )
        talker_input_embed, trailing_text_hidden = self.finish_text_prompt(
            talker_input_embed=talker_input_embed,
            input_id=input_id,
            codec_last_embed=codec_input[:, -1:],
            prompt_constants=prompt_constants,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = self.apply_instruct_prefix(
            talker_input_embed,
            instruct_id,
        )
        attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]), device=self.device, dtype=torch.long
        )
        return talker_input_embed, attention_mask, trailing_text_hidden, None

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        prompt_constants: Qwen3TTSPromptConstants,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tts_pad_embed = prompt_constants.tts_pad_embed
        codec_pad_bos_embed = prompt_constants.codec_pad_bos_embed
        text_embed = self.text_projection(
            self.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, prompt_constants.tts_eos_embed], dim=1)
        codec_embed = torch.cat(
            [
                self.get_input_embeddings()(ref_code[:, :1]),
                self.code_predictor.model.codec_embedding(
                    ref_code[:, 1 : self.config.num_code_groups]
                ),
            ],
            dim=1,
        )
        codec_embed = codec_embed.sum(1).unsqueeze(0)
        codec_embed = torch.cat([codec_pad_bos_embed[:, 1:], codec_embed], dim=1)
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + codec_pad_bos_embed[:, :1].expand(
                -1, text_lens, -1
            )
            icl_input_embed = torch.cat(
                [icl_input_embed, codec_embed + tts_pad_embed], dim=1
            )
            return icl_input_embed, tts_pad_embed
        else:
            pass
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
        else:
            pass
        text_embed = torch.cat(
            [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1
        )
        return text_embed + codec_embed, tts_pad_embed


class Qwen3TTSTalker(Qwen3TTSPromptBuilderMixin, nn.Module):
    """Qwen3-TTS Base talker with SGLang-managed KV cache for the main AR loop."""

    def __init__(
        self,
        config: "Qwen3TTSConfig",
        quant_config: object = None,
        prefix: str = "",
    ) -> None:
        del quant_config
        super().__init__()
        root_config = config
        config = root_config.talker_config
        self.root_config = root_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.tts_model_type = root_config.tts_model_type
        self.tokenizer_type = root_config.tokenizer_type
        self.tts_model_size = root_config.tts_model_size
        self.speaker_encoder_sample_rate = (
            root_config.speaker_encoder_config.sample_rate
        )

        self.text_projection = ResizeMLP(
            config.text_hidden_size,
            config.text_hidden_size,
            config.hidden_size,
            prefix=add_prefix("text_projection", prefix),
        )
        self.model = Qwen3TTSTalkerTextModel(config, prefix=add_prefix("model", prefix))
        self.codec_head = ReplicatedLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            prefix=add_prefix("codec_head", prefix),
        )
        self.code_predictor = Qwen3TTSCodePredictor(
            config,
            prefix=add_prefix("code_predictor", prefix),
        )
        self.codec_table_params_mapping = Qwen3TTSCodecTables.make_params_mapping(
            config.num_code_groups - 1
        )

        if self.tts_model_type == "base":
            apply_qwen_tts_transformers_compatibility_patches()
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

            self.speaker_encoder = Qwen3TTSSpeakerEncoder(
                root_config.speaker_encoder_config
            )
            self.speaker_encoder_graph_runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(
                self.speaker_encoder, sample_rate=self.speaker_encoder_sample_rate
            )
        else:
            self.speaker_encoder = None
            self.speaker_encoder_graph_runner = None
        self.speech_tokenizer = None

        server_args = get_context().server_args
        max_batch_size = get_schedule().max_running_requests
        self.predictor_max_batch_size = max_batch_size
        hidden_size = config.hidden_size
        predictor_len = config.num_code_groups + 1
        device = self.model.codec_embedding.weight.device
        dtype = self.model.codec_embedding.weight.dtype
        self.decode_feedback_embedding = self.model.decode_feedback_embedding
        cp_layers = self.code_predictor.model.layers
        cp_attn = cp_layers[0].self_attn
        self.predictor_positions = torch.arange(
            predictor_len, device=device, dtype=torch.long
        )
        self.predictor_position_rows = (
            self.predictor_positions[:, None]
            .expand(predictor_len, max_batch_size)
            .contiguous()
        )
        self.predictor_device = device
        self.predictor_device_module = torch.get_device_module(device)
        # note(ratish): slot major, so the rope kernel stores k and v as one
        # row per (batch row, slot) and the attention reads a transposed view.
        self.predictor_k_cache = torch.zeros(
            len(cp_layers),
            max_batch_size,
            predictor_len,
            cp_attn.num_kv_heads,
            cp_attn.head_dim,
            device=device,
            dtype=dtype,
        )
        self.predictor_v_cache = torch.zeros_like(self.predictor_k_cache)
        self.predictor_k_rows = [
            layer_cache.view(max_batch_size * predictor_len, -1)
            for layer_cache in self.predictor_k_cache
        ]
        self.predictor_v_rows = [
            layer_cache.view(max_batch_size * predictor_len, -1)
            for layer_cache in self.predictor_v_cache
        ]
        self.predictor_cache_slots = (
            torch.arange(max_batch_size, device=device, dtype=torch.long)[None, :]
            * predictor_len
            + self.predictor_positions[:, None]
        ).contiguous()
        # note(ratish): the first pass carries two tokens per request, request by
        # request, so a batch's positions and cache slots are a prefix of these.
        self.predictor_pair_positions = self.predictor_positions[:2].repeat(
            max_batch_size
        )
        self.predictor_pair_cache_slots = self.predictor_cache_slots[:2].t().reshape(-1)
        self.predictor_rope_stores_kv = self.resolve_predictor_rope_store(
            cp_attn, device=device
        )
        self.predictor_fused_layers = resolve_fused_predictor_layers(
            self.code_predictor, predictor_len, max_batch_size, device, dtype
        )
        self.predictor_cutedsl_layers: nn.ModuleList | None = None
        self.sampled_token_ids = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )
        self.output_codes = torch.zeros(
            max_batch_size,
            config.num_code_groups,
            dtype=torch.long,
            device=device,
        )
        self.output_embeds = torch.zeros(
            max_batch_size, hidden_size, device=device, dtype=dtype
        )
        self.predictor_embedding_buffer = torch.empty(
            max_batch_size, hidden_size, device=device, dtype=dtype
        )
        projection = self.code_predictor.small_to_mtp_projection
        if projection is None:
            self.predictor_projected_embeddings = None
            self.predictor_projected_buffer = None
        else:
            # note (ratish): the projection of an embedding row is that row of the
            # projected table; the last codebook feeds no predictor step, so has none.
            self.predictor_projected_embeddings = torch.empty(
                config.num_code_groups - 2,
                config.code_predictor_config.vocab_size,
                projection.out_features,
                device=device,
                dtype=dtype,
            )
            self.predictor_projected_buffer = torch.empty(
                max_batch_size, projection.out_features, device=device, dtype=dtype
            )
        self.sub_batch_size = 0
        self.sub_temperature_tensor = torch.full(
            (max_batch_size,), 0.9, device=device, dtype=torch.float32
        )
        self.sub_top_p_tensor = torch.ones(
            max_batch_size, device=device, dtype=torch.float32
        )
        self.sub_top_k_tensor = torch.full(
            (max_batch_size,), 50, device=device, dtype=torch.long
        )
        self.semantic_sampling_seed_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.long
        )
        self.sub_sampling_seed_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.long
        )
        self.sub_speculative_seed_tensor: torch.Tensor | None = None
        self.sub_speculative_round_index_tensor: torch.Tensor | None = None
        self.sub_speculative_uniforms_tensor: torch.Tensor | None = None
        self.sub_speculative_rng_pool_tensor: torch.Tensor | None = None
        self.sub_do_sample_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.bool
        )
        self.sub_seed_offsets = torch.arange(
            1, config.num_code_groups, device=device, dtype=torch.long
        )
        self.sub_has_sampled_rows = False
        self.sub_has_argmax_rows = False
        self.sub_user_seed_rows: tuple[bool, ...] = ()
        self.sub_request_ids: tuple[str, ...] = ()
        self.sub_sample_rows: tuple[bool, ...] = ()
        self.sub_sampled_has_top_p = False
        self.sub_sampled_max_top_k = 0
        self.sub_sampled_has_unbounded_top_k = False
        self.decode_prep_rids: list[tuple[str, int]] | None = None
        self.decode_prep_epoch = 0
        self.predictor_graph_batch_sizes = self.normalize_predictor_graph_batch_sizes(
            server_args,
            max_batch_size=max_batch_size,
        )
        self.predictor_graphs: dict[PredictorGraphKey, PredictorDecodeGraph] = {}
        self.predictor_graph_disabled: set[PredictorGraphKey] = set()
        # note(ratish): None until the startup capture, which runs before the
        # KV pool is sized, or the first decode resolves it.
        self.predictor_graph_enabled: bool | None = None
        self.predictor_graph_failure_count = 0
        self.predictor_graph_capacity_fallback_count = 0
        self.predictor_graph_capacity_warned = False
        self.predictor_graph_capture_count = 0
        self.predictor_graph_startup_count = 0
        self.predictor_graph_replay_count = 0
        self.predictor_eager_count = 0
        self.predictor_graph_pool = None
        self.predictor_capture_stream: torch.Stream | None = None
        self.code_predictor_spec_backend = "cuda_graph"
        self.code_predictor_spec_probability_backend: Literal[
            "split_top_k", "radix_compact"
        ] = "split_top_k"
        self.code_predictor_spec_lm_head_backend: Literal[
            "grouped_triton", "warp_gemv"
        ] = "grouped_triton"
        self.code_predictor_spec_rng_backend: Literal["per_round", "frame_pool"] = (
            "per_round"
        )
        self.code_predictor_spec_graphs: dict[
            tuple[int, int], PredictorSpeculativeVerifyGraph
        ] = {}
        self.code_predictor_spec_lm_head_weights: torch.Tensor | None = None
        self.code_predictor_spec_input_buffers: dict[int, SpeculativeVerifierInputs] = (
            {}
        )
        self.code_predictor_spec_metadata_device: torch.Tensor | None = None
        self.code_predictor_spec_metadata_host: torch.Tensor | None = None
        self.code_predictor_spec_effective_width_histogram: dict[int, int] = {}
        self.code_predictor_spec_graph_memory_delta_bytes = 0
        self.code_predictor_spec_graph_capture_time_ms = 0.0
        self.code_predictor_spec_configuration_time_ms = 0.0
        self.code_predictor_spec_fused_predictor_forward_count = 0
        self.code_predictor_generic_forward_count = 0
        self.code_predictor_spec_standalone_correction_forward_count = 0
        self.code_predictor_spec_correction_forward_watch = False
        self.code_predictor_spec_draft_head_evaluation_count = 0
        self.code_predictor_spec_target_head_evaluation_count = 0
        self.code_predictor_spec_requested_width = 0
        self.code_predictor_spec_effective_width = 0
        self.code_predictor_spec_fallback_reason: str | None = None
        self.code_predictor_spec_native_fallback_count = 0
        self.code_predictor_output_frame_count = 0
        self.code_predictor_spec_frame_count = 0
        self.code_predictor_spec_round_count = 0
        self.code_predictor_spec_proposed_count = 0
        self.code_predictor_spec_accepted_count = 0
        self.code_predictor_spec_committed_count = 0
        self.code_predictor_spec_bonus_count = 0
        self.code_predictor_spec_rejection_count = 0
        self.code_predictor_spec_packed_forward_count = 0
        self.code_predictor_spec_packed_query_slots = 0
        self.code_predictor_spec_padded_query_slots = 0
        self.code_predictor_spec_graph_replay_count = 0
        self.code_predictor_spec_graph_capture_failure_count = 0
        bind_default_weight_loaders(self)
        self.cached_params_dict = dict(self.named_parameters())
        self.sampler = None

    def prepare_decode_buffers(self, requests: list[SchedulerRequest]) -> None:
        batch_size = len(requests)
        if batch_size > self.sub_temperature_tensor.shape[0]:
            raise RuntimeError("Qwen3-TTS sampling buffers are too small")
        else:
            pass

        # Note: (Jiaxin Deng) every staged value here is static per request, so
        # an unchanged batch composition can reuse the previous staging wholesale.
        # Request ids alone are reusable across request lifetimes, so identity is
        # (request_id, per-data epoch); test doubles without request_id restage.
        rids: list[tuple[str, int]] | None = []
        for sched_req in requests:
            rid = sched_req.request_id
            if rid is None:
                rids = None
                break
            else:
                pass
            data = sched_req.data
            if data.decode_prep_epoch is None:
                self.decode_prep_epoch += 1
                data.decode_prep_epoch = self.decode_prep_epoch
            else:
                pass
            rids.append((rid, data.decode_prep_epoch))
        if rids is not None and rids == self.decode_prep_rids:
            return
        else:
            pass
        self.decode_prep_rids = None

        speculative_resources_configured = self.sub_speculative_seed_tensor is not None
        semantic_seeds: list[int] = []
        speculative_seeds: list[int] = []
        request_ids: list[str] = []
        sub_temperatures: list[float] = []
        sub_top_ps: list[float] = []
        sub_top_ks: list[int] = []
        sub_seeds: list[int] = []
        user_seed_rows: list[bool] = []
        sub_do_samples: list[bool] = []
        sample_rows: list[int] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            try:
                semantic_seed = int(data.semantic_sampling_seed)
                do_sample = bool(data.subtalker_dosample)
                subtalker_temperature = float(data.subtalker_temperature)
                subtalker_top_p = float(data.subtalker_top_p)
                subtalker_top_k = int(data.subtalker_top_k)
                subtalker_seed = int(data.subtalker_sampling_seed)
                if speculative_resources_configured:
                    user_seed_provided = data.user_seed_provided
                else:
                    user_seed_provided = False
            except AttributeError as exc:
                raise TypeError(
                    "Qwen3-TTS decode buffers require request data with semantic "
                    "and subtalker sampling fields"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Qwen3-TTS decode buffers require numeric semantic and "
                    "subtalker sampling fields"
                ) from exc
            if speculative_resources_configured and not isinstance(
                user_seed_provided, bool
            ):
                raise TypeError("user_seed_provided must be a boolean")
            else:
                pass
            semantic_seeds.append(semantic_seed)
            if speculative_resources_configured:
                request_id = sched_req.request_id
                if request_id:
                    request_ids.append(request_id)
                    speculative_seeds.append(
                        derive_speculative_request_seed(subtalker_seed, request_id)
                    )
                else:
                    request_ids.append("")
                    speculative_seeds.append(0)
                user_seed_rows.append(user_seed_provided)
            else:
                pass
            sub_do_samples.append(do_sample)
            # note(ratish): the sampler divides by the temperature, staged clamped
            # so the sub-steps read it without a kernel.
            sub_temperatures.append(
                max(subtalker_temperature, 1e-5) if do_sample else 1.0
            )
            sub_top_ps.append(subtalker_top_p if do_sample else 1.0)
            # Note (Shulei He): a greedy row's original top_k can be 0 or -1,
            # which would otherwise hit the full-sort branch.
            sub_top_ks.append(subtalker_top_k if do_sample else 1)
            sub_seeds.append(subtalker_seed)
            if do_sample:
                sample_rows.append(row_idx)
            else:
                pass

        predictor_vocab_size = int(self.config.code_predictor_config.vocab_size)
        max_top_k, has_top_p, has_unbounded_top_k = predictor_signature_terms(
            [sub_top_ks[row_idx] for row_idx in sample_rows],
            [sub_top_ps[row_idx] for row_idx in sample_rows],
            predictor_vocab_size,
        )
        self.sub_batch_size = batch_size
        self.sub_has_sampled_rows = bool(sample_rows)
        self.sub_has_argmax_rows = len(sample_rows) < batch_size
        self.sub_user_seed_rows = tuple(user_seed_rows)
        self.sub_request_ids = tuple(request_ids)
        self.sub_sample_rows = tuple(sub_do_samples)
        self.sub_sampled_has_top_p = has_top_p
        self.sub_sampled_max_top_k = max_top_k
        self.sub_sampled_has_unbounded_top_k = has_unbounded_top_k

        if batch_size == 0:
            self.decode_prep_rids = rids
            return
        else:
            pass

        # note(ratish): a pageable source waits for the last step's predictor.
        pin_memory = is_pin_memory_available(self.sub_temperature_tensor.device)
        for buffer, values in (
            (self.semantic_sampling_seed_tensor, semantic_seeds),
            (self.sub_temperature_tensor, sub_temperatures),
            (self.sub_top_p_tensor, sub_top_ps),
            (self.sub_top_k_tensor, sub_top_ks),
            (self.sub_sampling_seed_tensor, sub_seeds),
            (self.sub_do_sample_tensor, sub_do_samples),
        ):
            buffer[:batch_size].copy_(
                torch.tensor(values, dtype=buffer.dtype, pin_memory=pin_memory),
                non_blocking=True,
            )
        if self.sub_speculative_seed_tensor is not None:
            self.sub_speculative_seed_tensor[:batch_size].copy_(
                torch.tensor(
                    speculative_seeds, dtype=torch.long, pin_memory=pin_memory
                ),
                non_blocking=True,
            )
        else:
            pass
        self.decode_prep_rids = rids

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        input_embeds_are_projected: bool = False,
        omni_prefill_rids: list[str] | None = None,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected, omni_prefill_rids
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend():
            last_index = self.extend_last_index(forward_batch, hidden_states.device)
            hidden_states = hidden_states[last_index]
        else:
            pass
        logits, _ = self.codec_head(hidden_states)
        # note (ratish): the sampler divides and softmaxes in the logits' own dtype, and
        # FlashInfer's draw needs probabilities that sum to 1; a bf16 row that falls short
        # draws the last valid id, EOS.
        logits_output = LogitsProcessorOutput(
            next_token_logits=logits.float(),
            hidden_states=hidden_states,
        )
        return logits_output

    def extend_last_index(
        self,
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> torch.Tensor:
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            return torch.tensor([forward_batch.input_ids.shape[0] - 1], device=device)
        else:
            pass
        return torch.cumsum(extend_seq_lens.to(device=device), dim=0) - 1

    @torch.no_grad()
    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        else:
            pass
        self.code_predictor_output_frame_count += int(
            layer0_codes.shape[0] * layer0_codes.shape[1]
        )
        if self.code_predictor_spec_requested_width > 0:
            speculative_result = self.code_predictor_forward_speculative(
                layer0_codes,
                talker_hidden,
                semantic_positions,
            )
            if speculative_result is not None:
                return speculative_result
            else:
                self.code_predictor_spec_native_fallback_count += int(
                    layer0_codes.shape[0]
                )
                self.code_predictor_spec_fallback_reason = (
                    self.code_predictor_speculative_fallback_reason(
                        layer0_codes, talker_hidden
                    )
                )
        else:
            pass
        graph_result = self.predictor_forward_graphed(
            layer0_codes,
            talker_hidden,
            semantic_positions,
        )
        if graph_result is not None:
            self.predictor_graph_replay_count += 1
            return graph_result
        else:
            self.predictor_eager_count += 1
        result_codes, summed_embeddings = self.code_predictor_forward_incremental(
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
            semantic_positions=semantic_positions,
        )
        return result_codes, summed_embeddings

    def code_predictor_speculative_fallback_reason(
        self, layer0_codes: torch.Tensor, talker_hidden: torch.Tensor
    ) -> str:
        """Return the explicit reason the current request batch used native decode."""
        if self.code_predictor_spec_effective_width <= 0:
            return (
                self.code_predictor_spec_fallback_reason
                or "speculative-capability-unavailable"
            )
        else:
            pass
        sampling_reason = speculative_sampling_fallback_reason(
            self.sub_request_ids,
            self.sub_user_seed_rows,
            self.sub_sample_rows,
            has_unbounded_top_k=self.sub_sampled_has_unbounded_top_k,
            deterministic_mode=bool(
                get_exec().deterministic.enable_deterministic_inference
            ),
        )
        if sampling_reason is not None:
            return sampling_reason
        else:
            pass
        if fused_raw_logit_block_k(self.sub_sampled_max_top_k) is None:
            return "unsupported-sampling-configuration"
        else:
            pass
        if self.predictor_cutedsl_layers is not None:
            return "unsupported-predictor-backend"
        else:
            pass
        if (
            layer0_codes.shape[1] != 1
            or talker_hidden.shape[1] != 1
            or layer0_codes.device.type != "cuda"
            or talker_hidden.device != self.predictor_device
        ):
            return "unsupported-frame-input-shape"
        else:
            pass
        if self.code_predictor_spec_backend == "cuda_graph":
            return "speculative-verifier-graph-unavailable-for-batch"
        else:
            pass
        return "fused-predictor-capacity-unavailable"

    def resolve_code_predictor_speculative_frame_path(
        self, batch_size: int
    ) -> tuple[int, PredictorSpeculativeVerifyGraph | None] | None:
        """Resolve an eager or captured verifier before touching frame state."""
        sampling_fallback = speculative_sampling_fallback_reason(
            self.sub_request_ids,
            self.sub_user_seed_rows,
            self.sub_sample_rows,
            has_unbounded_top_k=self.sub_sampled_has_unbounded_top_k,
            deterministic_mode=bool(
                get_exec().deterministic.enable_deterministic_inference
            ),
        )
        if (
            self.code_predictor_spec_effective_width <= 0
            or batch_size <= 0
            or batch_size != self.sub_batch_size
            or self.predictor_device.type != "cuda"
            or self.predictor_k_cache.dtype is not torch.bfloat16
            or self.predictor_fused_layers is None
            or self.predictor_cutedsl_layers is not None
            or self.code_predictor_spec_lm_head_weights is None
            or sampling_fallback is not None
            or self.sub_has_argmax_rows
            or not self.sub_has_sampled_rows
            or fused_raw_logit_block_k(self.sub_sampled_max_top_k) is None
        ):
            return None
        else:
            pass

        attention = self.code_predictor.model.layers[0].self_attn
        if self.code_predictor_spec_backend == "eager":
            width = resolve_fused_predictor_width(
                self.code_predictor_spec_requested_width,
                batch_size,
                self.predictor_fused_layers.max_rows,
                int(attention.num_heads),
                int(attention.num_kv_heads),
            )
            if width < 1:
                return None
            else:
                pass
            return width, None
        else:
            pass

        for bucket_size in self.predictor_graph_batch_sizes:
            if bucket_size < batch_size:
                continue
            else:
                pass
            maximum_width = resolve_fused_predictor_width(
                self.code_predictor_spec_requested_width,
                bucket_size,
                self.predictor_fused_layers.max_rows,
                int(attention.num_heads),
                int(attention.num_kv_heads),
            )
            for width in range(maximum_width, 0, -1):
                graph = self.code_predictor_spec_graphs.get((bucket_size, width))
                if graph is not None:
                    return width, graph
                else:
                    pass
        return None

    def resolve_code_predictor_speculative_round_path(
        self, batch_size: int, requested_width: int
    ) -> tuple[int, PredictorSpeculativeVerifyGraph | None]:
        """Resolve a captured graph for this width or execute the same width eagerly."""
        if self.code_predictor_spec_backend == "eager":
            return requested_width, None
        else:
            pass
        for bucket_size in self.predictor_graph_batch_sizes:
            if bucket_size < batch_size:
                continue
            else:
                pass
            graph = self.code_predictor_spec_graphs.get((bucket_size, requested_width))
            if graph is not None:
                return requested_width, graph
            else:
                pass
        return requested_width, None

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
        frame_path = self.resolve_code_predictor_speculative_frame_path(batch_size)
        if frame_path is None:
            return None
        else:
            pass
        maximum_frame_width, _ = frame_path
        packed_head_weights = self.code_predictor_spec_lm_head_weights
        if packed_head_weights is None:
            raise RuntimeError("grouped Predictor head weights are unavailable")
        else:
            pass
        num_groups = int(self.config.num_code_groups)
        vocab_size = int(self.config.code_predictor_config.vocab_size)
        predictor_dtype = self.predictor_k_cache.dtype
        semantic_positions = self.normalize_semantic_positions(
            semantic_positions,
            batch_size=batch_size,
            seq_len=1,
            device=layer0_codes.device,
        )[:, 0]

        result_codes = self.output_codes[:batch_size].unsqueeze(-1)
        code_outputs = result_codes[:, :, 0]
        summed_embeddings = self.output_embeds[:batch_size].unsqueeze(1)
        summed_embeddings.zero_()
        layer0_embed = self.get_input_embeddings()(layer0_codes).to(
            dtype=predictor_dtype
        )
        code_outputs[:, 0].copy_(layer0_codes[:, 0])
        summed_embeddings[:, 0, :].add_(layer0_embed[:, 0, :])
        pair_embeds = self.code_predictor.project_input(
            torch.cat(
                (
                    talker_hidden.to(dtype=predictor_dtype),
                    layer0_embed,
                ),
                dim=1,
            )
        )
        pair_hidden = self.predictor_forward_tokens(
            token_embeds=pair_embeds,
            batch_size=batch_size,
            cache_len=0,
        )
        self.code_predictor_spec_fused_predictor_forward_count += 1
        code1_logits, _ = self.code_predictor.lm_head[0](pair_hidden[:, 1:, :])
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

        committed_depths = torch.ones(
            batch_size, device=self.predictor_device, dtype=torch.long
        )
        logical_kv_lengths = torch.full_like(committed_depths, 2)
        pending_tokens = code_outputs[:, 1].clone()
        preceding_hidden = pair_hidden[:, 1, :].contiguous()
        active_rows = torch.ones(
            batch_size, device=self.predictor_device, dtype=torch.bool
        )
        metadata_device = self.code_predictor_spec_metadata_device
        metadata_host = self.code_predictor_spec_metadata_host
        if metadata_device is None or metadata_host is None:
            raise RuntimeError("speculative metadata buffers were not configured")
        else:
            pass

        batch_rows = torch.arange(batch_size, device=self.predictor_device)
        temperatures = self.sub_temperature_tensor[:batch_size]
        top_ks = self.sub_top_k_tensor[:batch_size]
        top_ps = self.sub_top_p_tensor[:batch_size]
        speculative_seed_buffer = self.sub_speculative_seed_tensor
        round_index_buffer = self.sub_speculative_round_index_tensor
        uniform_buffer = self.sub_speculative_uniforms_tensor
        rng_pool_buffer = self.sub_speculative_rng_pool_tensor
        if (
            speculative_seed_buffer is None
            or round_index_buffer is None
            or (
                self.code_predictor_spec_rng_backend == "per_round"
                and uniform_buffer is None
            )
            or (
                self.code_predictor_spec_rng_backend == "frame_pool"
                and rng_pool_buffer is None
            )
        ):
            raise RuntimeError("speculative request seeds were not configured")
        else:
            pass
        speculative_seeds = speculative_seed_buffer[:batch_size]
        frame_rng_pool = None
        if self.code_predictor_spec_rng_backend == "frame_pool":
            frame_rng_pool = speculative_uniforms_frame_pool(
                speculative_seeds,
                semantic_positions,
                num_groups,
                num_groups - 2,
                int(rng_pool_buffer.shape[3]),
                rng_pool_buffer[:batch_size],
            )
        else:
            pass
        max_top_k = int(self.sub_sampled_max_top_k)
        has_top_p = bool(self.sub_sampled_has_top_p)
        round_index = 0

        staged_round_width = num_groups - 2
        while staged_round_width > 0:
            requested_round_width = min(maximum_frame_width, staged_round_width)
            verify_width, verify_graph = (
                self.resolve_code_predictor_speculative_round_path(
                    batch_size, requested_round_width
                )
            )
            proposal_width = verify_width - 1
            proposal_offsets = torch.arange(
                proposal_width, device=self.predictor_device, dtype=torch.long
            )
            proposal_counts = torch.minimum(
                (num_groups - 2 - committed_depths).clamp_min(0),
                torch.full_like(committed_depths, proposal_width),
            )
            proposal_counts = torch.where(
                active_rows, proposal_counts, torch.zeros_like(proposal_counts)
            )
            active_proposals = proposal_offsets.unsqueeze(
                0
            ) < proposal_counts.unsqueeze(1)
            if proposal_width > 0:
                proposal_head_rows = batch_rows.repeat_interleave(proposal_width)
                proposal_head_indices = (
                    committed_depths.unsqueeze(1) + proposal_offsets.unsqueeze(0)
                ).reshape(-1)
                proposal_logits = grouped_lm_head_linear(
                    preceding_hidden,
                    self.code_predictor_spec_lm_head_weights,
                    proposal_head_rows,
                    proposal_head_indices,
                    active_proposals.reshape(-1).contiguous(),
                    backend=self.code_predictor_spec_lm_head_backend,
                )
                flat_proposal_temperatures = temperatures.repeat_interleave(
                    proposal_width
                )
                flat_proposal_top_ks = top_ks.repeat_interleave(proposal_width)
                flat_proposal_top_ps = top_ps.repeat_interleave(proposal_width)
                proposal_candidates = build_speculative_probability_candidates_selected(
                    proposal_logits,
                    flat_proposal_temperatures.contiguous(),
                    flat_proposal_top_ks.contiguous(),
                    flat_proposal_top_ps.contiguous(),
                    active_proposals.reshape(-1).contiguous(),
                    max_top_k=max_top_k,
                    has_top_p=has_top_p,
                    has_unbounded_top_k=False,
                    backend=self.code_predictor_spec_probability_backend,
                )
                if proposal_candidates is None:
                    raise RuntimeError(
                        "speculative proposal probability kernel rejected a preflighted shape"
                    )
                else:
                    pass
                proposal_probabilities_compact, proposal_ids_compact = (
                    proposal_candidates
                )
            else:
                proposal_probabilities_compact = torch.empty(
                    (0, max_top_k), device=self.predictor_device, dtype=torch.float32
                )
                proposal_ids_compact = torch.empty(
                    (0, max_top_k), device=self.predictor_device, dtype=torch.int32
                )
            target_hidden_rows = torch.arange(
                batch_size * verify_width,
                device=self.predictor_device,
                dtype=torch.long,
            )
            if frame_rng_pool is None:
                round_index_buffer.fill_(round_index)
                (
                    proposal_uniforms,
                    acceptance_uniforms,
                    correction_uniforms,
                    bonus_uniforms,
                ) = speculative_uniforms_all_purposes(
                    speculative_seeds,
                    semantic_positions,
                    round_index_buffer,
                    num_groups,
                    proposal_width,
                    verify_width,
                    uniform_buffer[:batch_size],
                )
            else:
                proposal_uniforms = frame_rng_pool[:, 0, round_index, :proposal_width]
                acceptance_uniforms = frame_rng_pool[:, 1, round_index, :proposal_width]
                correction_uniforms = frame_rng_pool[:, 2, round_index, :verify_width]
                bonus_uniforms = frame_rng_pool[:, 3, round_index, :verify_width]
            if proposal_width > 0:
                draft_tokens = sample_compact_probability_rows_fused(
                    proposal_probabilities_compact,
                    proposal_ids_compact,
                    proposal_uniforms,
                    active_proposals,
                )
            else:
                draft_tokens = torch.empty(
                    (batch_size, 0), device=self.predictor_device, dtype=torch.long
                )
            valid_query_counts = torch.where(
                active_rows,
                proposal_counts + 1,
                torch.zeros_like(proposal_counts),
            )
            input_buffer = self.code_predictor_spec_input_buffers.get(verify_width)
            can_use_projected_table = (
                projected_tables is not None
                or self.code_predictor.small_to_mtp_projection is None
            )
            if input_buffer is not None and can_use_projected_table:
                source_tables = (
                    codec_tables if projected_tables is None else projected_tables
                )
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
            else:
                inputs_prepared = False
            if inputs_prepared:
                query_embeddings = input_buffer.token_embeddings[:batch_size]
                packed_positions = input_buffer.positions[: batch_size * verify_width]
                query_depths = input_buffer.target_head_indices[:batch_size]
                valid_queries = input_buffer.valid_queries[:batch_size]
            else:
                query_offsets = torch.arange(
                    verify_width, device=self.predictor_device, dtype=torch.long
                )
                valid_queries = query_offsets.unsqueeze(
                    0
                ) < valid_query_counts.unsqueeze(1)
                query_depths = committed_depths.unsqueeze(1) + query_offsets.unsqueeze(
                    0
                )
                if proposal_width > 0:
                    query_tokens = torch.cat(
                        (pending_tokens[:, None], draft_tokens), dim=1
                    )
                else:
                    query_tokens = pending_tokens[:, None]
                safe_query_depths = (query_depths - 1).clamp(
                    min=0, max=codec_tables.shape[0] - 1
                )
                safe_query_tokens = query_tokens.clamp(min=0, max=vocab_size - 1)
                if projected_tables is None:
                    raw_query_embeddings = codec_tables[
                        safe_query_depths, safe_query_tokens
                    ]
                    query_embeddings = self.code_predictor.project_input(
                        raw_query_embeddings
                    )
                else:
                    safe_projected_depths = (query_depths - 1).clamp(
                        min=0, max=projected_tables.shape[0] - 1
                    )
                    query_embeddings = projected_tables[
                        safe_projected_depths, safe_query_tokens
                    ]
                query_embeddings = torch.where(
                    valid_queries.unsqueeze(-1),
                    query_embeddings,
                    torch.zeros_like(query_embeddings),
                ).to(dtype=predictor_dtype)
                packed_positions = logical_kv_lengths.unsqueeze(
                    1
                ) + query_offsets.unsqueeze(0)
                packed_positions = (
                    torch.where(
                        valid_queries,
                        packed_positions,
                        torch.full_like(packed_positions, -1),
                    )
                    .reshape(-1)
                    .contiguous()
                )
            target_hidden_rows = torch.arange(
                batch_size * verify_width,
                device=self.predictor_device,
                dtype=torch.long,
            )
            if verify_graph is None:
                verified_hiddens = self.predictor_forward_packed_tokens(
                    token_embeds=query_embeddings,
                    positions=packed_positions,
                    batch_size=batch_size,
                )
                target_logits = grouped_lm_head_linear(
                    verified_hiddens.reshape(-1, verified_hiddens.shape[-1]),
                    packed_head_weights,
                    target_hidden_rows,
                    query_depths.reshape(-1),
                    valid_queries.reshape(-1).contiguous(),
                    backend=self.code_predictor_spec_lm_head_backend,
                )
            else:
                verified_hiddens = verify_graph.replay(
                    query_embeddings,
                    packed_positions,
                    query_depths,
                    valid_queries,
                )
                if verify_graph.target_logits is None:
                    raise RuntimeError("speculative graph has no target logits")
                else:
                    pass
                target_logits = verify_graph.target_logits[: batch_size * verify_width]
                self.code_predictor_spec_graph_replay_count += 1
            target_logits = target_logits.view(batch_size, verify_width, vocab_size)
            flat_target_temperatures = temperatures.repeat_interleave(verify_width)
            flat_target_top_ks = top_ks.repeat_interleave(verify_width)
            flat_target_top_ps = top_ps.repeat_interleave(verify_width)
            target_candidates = build_speculative_probability_candidates_selected(
                target_logits.reshape(-1, vocab_size).contiguous(),
                flat_target_temperatures.contiguous(),
                flat_target_top_ks.contiguous(),
                flat_target_top_ps.contiguous(),
                valid_queries.reshape(-1).contiguous(),
                max_top_k=max_top_k,
                has_top_p=has_top_p,
                has_unbounded_top_k=False,
                backend=self.code_predictor_spec_probability_backend,
            )
            if target_candidates is None:
                raise RuntimeError(
                    "speculative target probability kernel rejected a preflighted shape"
                )
            else:
                pass
            target_probabilities_compact, target_ids_compact = target_candidates
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
            )

            commit_speculative_codes(
                decision.committed_tokens,
                decision.committed_token_counts,
                committed_depths,
                codec_tables,
                code_outputs,
                summed_embeddings[:, 0, :],
            )
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
                metadata_device[:batch_size],
                num_groups,
            )
            metadata_host[:batch_size].copy_(
                metadata_device[:batch_size],
                non_blocking=metadata_host.is_pinned(),
            )
            current_stream = self.predictor_device_module.current_stream(
                self.predictor_device
            )
            current_stream.synchronize()
            round_metadata = metadata_host[:batch_size]
            if bool(round_metadata[:, 8].any()):
                raise RuntimeError(
                    "compact speculative acceptance produced invalid probability mass"
                )
            else:
                pass
            active_count = int(round_metadata[:, 1].sum())
            self.code_predictor_spec_correction_forward_watch = bool(
                round_metadata[:, 5].sum()
            )
            self.code_predictor_spec_round_count += 1
            self.code_predictor_spec_proposed_count += int(round_metadata[:, 2].sum())
            self.code_predictor_spec_accepted_count += int(round_metadata[:, 0].sum())
            self.code_predictor_spec_committed_count += int(round_metadata[:, 3].sum())
            self.code_predictor_spec_bonus_count += int(round_metadata[:, 4].sum())
            self.code_predictor_spec_rejection_count += int(round_metadata[:, 5].sum())
            self.code_predictor_spec_packed_forward_count += 1
            self.code_predictor_spec_effective_width_histogram[verify_width] = (
                self.code_predictor_spec_effective_width_histogram.get(verify_width, 0)
                + 1
            )
            self.code_predictor_spec_fused_predictor_forward_count += 1
            draft_evaluation_count = int(round_metadata[:, 2].sum())
            valid_target_count = int(round_metadata[:, 6].sum())
            self.code_predictor_spec_draft_head_evaluation_count += (
                draft_evaluation_count
            )
            self.code_predictor_spec_target_head_evaluation_count += valid_target_count
            graph_batch_size = (
                batch_size if verify_graph is None else verify_graph.batch_size
            )
            packed_query_slots = graph_batch_size * verify_width
            self.code_predictor_spec_packed_query_slots += packed_query_slots
            self.code_predictor_spec_padded_query_slots += (
                packed_query_slots - valid_target_count
            )
            round_index += 1
            if active_count == 0:
                break
            else:
                pass
            staged_round_width = resolve_speculative_round_width(
                maximum_frame_width,
                maximum_frame_width,
                num_groups,
                round_metadata[:, 7],
                round_metadata[:, 1].to(torch.bool),
            )

        self.code_predictor_spec_correction_forward_watch = False
        self.code_predictor_spec_frame_count += batch_size
        return result_codes, summed_embeddings

    def predictor_graph_stats(self) -> PredictorGraphStats:
        return {
            "enabled": self.predictor_graph_enabled,
            "startup_graph_count": self.predictor_graph_startup_count,
            "lazy_graph_count": len(self.predictor_graphs)
            - self.predictor_graph_startup_count,
            "replays": self.predictor_graph_replay_count,
            "eager_runs": self.predictor_eager_count,
            "capacity_fallbacks": self.predictor_graph_capacity_fallback_count,
            "capture_failures": self.predictor_graph_failure_count,
        }

    def configure_code_predictor_speculation(
        self,
        requested_width: int,
        *,
        backend: Literal["cuda_graph", "eager"] = "cuda_graph",
        probability_backend: Literal["split_top_k", "radix_compact"] = "split_top_k",
        lm_head_backend: Literal["grouped_triton", "warp_gemv"] = "grouped_triton",
        rng_backend: Literal["per_round", "frame_pool"] = "per_round",
    ) -> None:
        """Allocate opt-in scratch and resolve the selected speculative backend."""
        configuration_started = time.perf_counter()
        speculative_resources_requested(requested_width)
        if backend not in ("cuda_graph", "eager"):
            raise ValueError(
                "CodePredictor speculative backend must be cuda_graph or eager"
            )
        else:
            pass
        if probability_backend not in ("split_top_k", "radix_compact"):
            raise ValueError("unsupported CodePredictor probability backend")
        elif lm_head_backend not in ("grouped_triton", "warp_gemv"):
            raise ValueError("unsupported CodePredictor grouped-head backend")
        elif rng_backend not in ("per_round", "frame_pool"):
            raise ValueError("unsupported CodePredictor RNG backend")
        else:
            pass
        self.code_predictor_spec_requested_width = requested_width
        self.code_predictor_spec_backend = backend
        self.code_predictor_spec_probability_backend = probability_backend
        self.code_predictor_spec_lm_head_backend = lm_head_backend
        self.code_predictor_spec_rng_backend = rng_backend
        self.code_predictor_spec_effective_width = 0
        self.code_predictor_spec_fallback_reason = None
        self.code_predictor_spec_graphs.clear()
        self.code_predictor_spec_input_buffers.clear()
        if requested_width == 0:
            self.sub_speculative_seed_tensor = None
            self.sub_speculative_round_index_tensor = None
            self.sub_speculative_uniforms_tensor = None
            self.sub_speculative_rng_pool_tensor = None
            self.code_predictor_spec_lm_head_weights = None
            self.code_predictor_spec_metadata_device = None
            self.code_predictor_spec_metadata_host = None
            self.code_predictor_spec_configuration_time_ms = (
                time.perf_counter() - configuration_started
            ) * 1000.0
            return
        else:
            pass

        if self.sub_speculative_seed_tensor is None:
            self.sub_speculative_seed_tensor = torch.zeros(
                self.predictor_max_batch_size,
                device=self.predictor_device,
                dtype=torch.long,
            )
        else:
            pass
        if self.sub_speculative_round_index_tensor is None:
            self.sub_speculative_round_index_tensor = torch.zeros(
                (), device=self.predictor_device, dtype=torch.long
            )
        else:
            pass
        if rng_backend == "per_round":
            self.sub_speculative_rng_pool_tensor = None
            if (
                self.sub_speculative_uniforms_tensor is None
                or self.sub_speculative_uniforms_tensor.shape[2] < requested_width
            ):
                self.sub_speculative_uniforms_tensor = torch.empty(
                    (
                        self.predictor_max_batch_size,
                        4,
                        requested_width,
                    ),
                    device=self.predictor_device,
                    dtype=torch.float32,
                )
            else:
                pass
        else:
            self.sub_speculative_uniforms_tensor = None
            pool_round_count = int(self.config.num_code_groups) - 2
            if (
                self.sub_speculative_rng_pool_tensor is None
                or self.sub_speculative_rng_pool_tensor.shape[2] < pool_round_count
                or self.sub_speculative_rng_pool_tensor.shape[3] < requested_width
            ):
                self.sub_speculative_rng_pool_tensor = torch.empty(
                    (
                        self.predictor_max_batch_size,
                        4,
                        pool_round_count,
                        requested_width,
                    ),
                    device=self.predictor_device,
                    dtype=torch.float32,
                )
            else:
                pass

        attention = self.code_predictor.model.layers[0].self_attn
        if self.predictor_fused_layers is not None:
            requested_scratch_rows = fused_predictor_scratch_rows(
                self.predictor_max_batch_size, requested_width
            )
            if requested_scratch_rows > self.predictor_fused_layers.max_rows:
                self.predictor_fused_layers = resolve_fused_predictor_layers(
                    self.code_predictor,
                    self.config.num_code_groups + 1,
                    self.predictor_max_batch_size,
                    self.predictor_device,
                    self.predictor_k_cache.dtype,
                    max_tokens_per_row=requested_width,
                )
            else:
                pass
        else:
            pass
        head_count = int(self.config.num_code_groups) - 1
        vocab_size = int(self.config.code_predictor_config.vocab_size)
        lm_head_weights_supported = all(
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
        )
        grouped_head_kernel_available = (
            grouped_lm_head_cuda_available()
            if lm_head_backend == "grouped_triton"
            else grouped_lm_head_warp_gemv_available()
        )
        if (
            current_platform.is_cuda()
            and grouped_head_kernel_available
            and self.predictor_fused_layers is not None
            and self.predictor_k_cache.dtype is torch.bfloat16
            and int(get_parallel().tp_size) == 1
            and lm_head_weights_supported
        ):
            head_weight = self.code_predictor.lm_head[0].weight
            hidden_size = int(head_weight.shape[1])
            self.code_predictor_spec_lm_head_weights = torch.empty(
                (head_count, vocab_size, hidden_size),
                device=self.predictor_device,
                dtype=self.predictor_k_cache.dtype,
            )
            self.copy_code_predictor_spec_lm_head_weights()
            self.code_predictor_spec_metadata_device = torch.empty(
                (self.predictor_max_batch_size, 9),
                device=self.predictor_device,
                dtype=torch.long,
            )
            pin_memory = is_pin_memory_available(self.predictor_device)
            self.code_predictor_spec_metadata_host = torch.empty(
                (self.predictor_max_batch_size, 9),
                dtype=torch.long,
                pin_memory=pin_memory,
            )
            hidden_size = int(self.predictor_fused_layers.shape.hidden_size)
            for verify_width in range(1, requested_width + 1):
                self.code_predictor_spec_input_buffers[verify_width] = (
                    SpeculativeVerifierInputs(
                        token_embeddings=torch.empty(
                            (
                                self.predictor_max_batch_size,
                                verify_width,
                                hidden_size,
                            ),
                            device=self.predictor_device,
                            dtype=self.predictor_k_cache.dtype,
                        ),
                        positions=torch.empty(
                            self.predictor_max_batch_size * verify_width,
                            device=self.predictor_device,
                            dtype=torch.long,
                        ),
                        target_head_indices=torch.empty(
                            (self.predictor_max_batch_size, verify_width),
                            device=self.predictor_device,
                            dtype=torch.long,
                        ),
                        valid_queries=torch.empty(
                            (self.predictor_max_batch_size, verify_width),
                            device=self.predictor_device,
                            dtype=torch.bool,
                        ),
                    )
                )
        else:
            self.code_predictor_spec_lm_head_weights = None
            self.code_predictor_spec_metadata_device = None
            self.code_predictor_spec_metadata_host = None

        if (
            backend == "cuda_graph"
            and self.code_predictor_spec_lm_head_weights is not None
        ):
            self.capture_code_predictor_speculative_graphs(requested_width)
        else:
            pass

        graph_available = backend == "eager" or bool(self.code_predictor_spec_graphs)
        capacity_rows = (
            0
            if self.predictor_fused_layers is None
            else self.predictor_fused_layers.max_rows
        )
        largest_eligible_width = 0
        if self.predictor_fused_layers is not None:
            for batch_size in self.predictor_graph_batch_sizes:
                resolved_width = resolve_fused_predictor_width(
                    requested_width,
                    batch_size,
                    capacity_rows,
                    int(attention.num_heads),
                    int(attention.num_kv_heads),
                )
                largest_eligible_width = max(largest_eligible_width, resolved_width)
        else:
            pass
        if backend == "cuda_graph":
            largest_eligible_width = max(
                (width for _, width in self.code_predictor_spec_graphs if width >= 2),
                default=0,
            )
            graph_available = largest_eligible_width >= 2
        else:
            pass
        width_to_resolve = largest_eligible_width or requested_width
        resolution = resolve_speculative_capability(
            width_to_resolve,
            is_cuda=current_platform.is_cuda(),
            is_bfloat16=(
                self.predictor_k_cache.dtype is torch.bfloat16
                and lm_head_weights_supported
            ),
            tensor_parallel_size=int(get_parallel().tp_size),
            fused_predictor_available=self.predictor_fused_layers is not None,
            batch_size=1,
            packed_row_capacity=capacity_rows,
            query_heads=int(attention.num_heads),
            kv_heads=int(attention.num_kv_heads),
            attention_tile_capacity=ATTENTION_ROWS,
            vocab_size=int(self.config.code_predictor_config.vocab_size),
            grouped_lm_head_supported=(
                grouped_head_kernel_available and lm_head_weights_supported
            ),
            graph_available=graph_available,
            sampling_supported=(
                self.predictor_device.type == "cuda" and lm_head_weights_supported
            ),
            has_explicit_seed=False,
            has_greedy_rows=False,
            deterministic_mode=bool(
                get_exec().deterministic.enable_deterministic_inference
            ),
        )
        self.code_predictor_spec_effective_width = resolution.effective_width
        self.code_predictor_spec_fallback_reason = resolution.fallback_reason
        self.code_predictor_spec_configuration_time_ms = (
            time.perf_counter() - configuration_started
        ) * 1000.0
        if resolution.fallback_reason is not None:
            logger.warning(
                "Qwen3-TTS CodePredictor speculative decoding requested at width %d "
                "but will use native decoding: %s",
                requested_width,
                resolution.fallback_reason,
            )
        else:
            pass

    def copy_code_predictor_spec_lm_head_weights(self) -> None:
        """Refresh the persistent grouped-head layout after checkpoint loading."""
        packed_weights = self.code_predictor_spec_lm_head_weights
        if packed_weights is None:
            return
        else:
            pass
        for head_index, lm_head in enumerate(self.code_predictor.lm_head):
            packed_weights[head_index].copy_(lm_head.weight)

    def capture_code_predictor_speculative_graphs(self, requested_width: int) -> int:
        """Capture verifier graphs for supported batch buckets before decoding."""
        if self.predictor_fused_layers is None:
            return 0
        else:
            pass
        if bool(get_exec().graph.disable_cuda_graph):
            return 0
        else:
            pass
        backend = current_platform.get_device_graph_backend(self.predictor_device)
        if backend is None:
            return 0
        else:
            pass
        attention = self.code_predictor.model.layers[0].self_attn
        capture_count = 0
        capture_started = time.perf_counter()
        for bucket_size in reversed(self.predictor_graph_batch_sizes):
            maximum_verify_width = resolve_fused_predictor_width(
                requested_width,
                bucket_size,
                self.predictor_fused_layers.max_rows,
                int(attention.num_heads),
                int(attention.num_kv_heads),
            )
            if maximum_verify_width < 1:
                continue
            else:
                pass
            for verify_width in range(maximum_verify_width, 0, -1):
                try:
                    graph = self.capture_code_predictor_speculative_verify_graph(
                        bucket_size, verify_width, backend
                    )
                except Exception as error:
                    if is_fatal_cuda_capture_error(error):
                        raise
                    else:
                        pass
                    self.code_predictor_spec_graph_capture_failure_count += 1
                    logger.warning(
                        "Failed to capture CodePredictor verifier graph for batch=%d width=%d",
                        bucket_size,
                        verify_width,
                        exc_info=True,
                    )
                else:
                    self.code_predictor_spec_graphs[(bucket_size, verify_width)] = graph
                    capture_count += 1
        self.code_predictor_spec_graph_capture_time_ms += (
            time.perf_counter() - capture_started
        ) * 1000.0
        return capture_count

    def capture_code_predictor_speculative_verify_graph(
        self, batch_size: int, verify_width: int, backend: DeviceGraphBackend
    ) -> PredictorSpeculativeVerifyGraph:
        """Capture packed verification and indexed target heads on stable buffers."""
        packed_head_weights = self.code_predictor_spec_lm_head_weights
        if packed_head_weights is None:
            raise RuntimeError("grouped Predictor head weights are unavailable")
        else:
            pass
        graph = PredictorSpeculativeVerifyGraph(
            batch_size,
            verify_width,
            device=self.predictor_device,
            hidden_size=int(self.predictor_fused_layers.shape.hidden_size),
            hidden_dtype=self.output_embeds.dtype,
        )
        graph.positions.view(batch_size, verify_width).copy_(
            torch.arange(verify_width, device=self.predictor_device).expand(
                batch_size, verify_width
            )
        )
        module = self.predictor_device_module
        if self.predictor_capture_stream is None:
            self.predictor_capture_stream = module.Stream(device=self.predictor_device)
        else:
            pass
        capture_stream = self.predictor_capture_stream
        current_stream = module.current_stream(self.predictor_device)
        capture_stream.wait_stream(current_stream)
        memory_before = torch.cuda.memory_allocated(self.predictor_device)
        captured = None
        fatal_capture_error = False

        def run_once() -> tuple[torch.Tensor, torch.Tensor]:
            hidden_states = self.predictor_forward_packed_tokens(
                token_embeds=graph.token_embeds,
                positions=graph.positions,
                batch_size=batch_size,
            )
            target_logits = grouped_lm_head_linear(
                hidden_states.reshape(batch_size * verify_width, -1),
                packed_head_weights,
                graph.target_hidden_rows,
                graph.target_head_indices.reshape(-1),
                graph.target_valid_queries.reshape(-1),
                backend=self.code_predictor_spec_lm_head_backend,
            )
            return hidden_states, target_logits

        try:
            with (
                module.device(self.predictor_device),
                current_platform.graph_capture_attention(),
                module.stream(capture_stream),
            ):
                run_once()
                with backend.capture(
                    pool=self.predictor_graph_memory_pool(),
                    stream=capture_stream,
                    thread_local_errors=True,
                ) as captured:
                    graph.hidden_states, graph.target_logits = run_once()
        except Exception as error:
            if is_fatal_cuda_capture_error(error):
                fatal_capture_error = True
                raise
            else:
                pass
            if captured is not None:
                try:
                    captured.reset()
                except Exception:
                    pass
            else:
                pass
            raise
        finally:
            if fatal_capture_error:
                pass
            else:
                current_stream.wait_stream(capture_stream)
        memory_after = torch.cuda.memory_allocated(self.predictor_device)
        self.code_predictor_spec_graph_memory_delta_bytes += max(
            memory_after - memory_before, 0
        )
        graph.graph = captured
        if graph.hidden_states is None or graph.target_logits is None:
            raise RuntimeError("CodePredictor verifier graph captured no outputs")
        else:
            pass
        return graph

    def predictor_speculative_stats(self) -> PredictorSpeculativeStats:
        average_codes_per_round = (
            self.code_predictor_spec_committed_count
            / self.code_predictor_spec_round_count
            if self.code_predictor_spec_round_count > 0
            else 0.0
        )
        return {
            "requested_width": self.code_predictor_spec_requested_width,
            "effective_width": self.code_predictor_spec_effective_width,
            "graph_mode": (
                "disabled"
                if self.code_predictor_spec_requested_width == 0
                else (
                    "verifier_only"
                    if self.code_predictor_spec_backend == "cuda_graph"
                    else "eager"
                )
            ),
            "probability_backend": self.code_predictor_spec_probability_backend,
            "lm_head_backend": self.code_predictor_spec_lm_head_backend,
            "rng_backend": self.code_predictor_spec_rng_backend,
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
            "graph_replays": self.code_predictor_spec_graph_replay_count,
            "verifier_graph_replays": self.code_predictor_spec_graph_replay_count,
            "graph_capture_failures": (
                self.code_predictor_spec_graph_capture_failure_count
            ),
            "graph_captures": len(self.code_predictor_spec_graphs),
            "graph_capture_memory_delta_bytes": (
                self.code_predictor_spec_graph_memory_delta_bytes
            ),
            "graph_capture_time_ms": self.code_predictor_spec_graph_capture_time_ms,
            "configuration_time_ms": self.code_predictor_spec_configuration_time_ms,
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

    @staticmethod
    def normalize_predictor_graph_batch_sizes(
        server_args: ServerArgs,
        *,
        max_batch_size: int,
    ) -> tuple[int, ...]:
        from sglang_omni.scheduling.generation_batch_policy import (
            get_decode_cuda_graph_bs,
        )

        raw_batch_sizes = get_decode_cuda_graph_bs(server_args)
        if raw_batch_sizes is None:
            # Note: (Jiaxin Deng) mirrors the backbone's default capture list
            # ([1, 2, 4, 8, 12, 16] at max_running_requests=16).
            raw_batch_sizes = (1, 2, 4, *range(8, int(max_batch_size) + 1, 4))
        else:
            pass
        normalized = sorted(
            {
                int(batch_size)
                for batch_size in raw_batch_sizes
                if 1 <= int(batch_size) <= int(max_batch_size)
            }
        )
        if not normalized or normalized[-1] < int(max_batch_size):
            normalized.append(int(max_batch_size))
        else:
            pass
        return tuple(normalized)

    def predictor_graph_bucket_size(self, batch_size: int) -> int | None:
        for bucket_size in self.predictor_graph_batch_sizes:
            if bucket_size >= batch_size:
                return bucket_size
            else:
                pass
        return None

    def predictor_graph_signature(
        self,
        batch_size: int,
        semantic_positions: torch.Tensor | None,
    ) -> PredictorGraphSignature | None:
        if semantic_positions is not None:
            if semantic_positions.device != self.predictor_device:
                return None
            else:
                pass
            if (
                semantic_positions.ndim not in (1, 2)
                or semantic_positions.shape[0] != batch_size
            ):
                return None
            else:
                pass
            if semantic_positions.ndim == 2 and semantic_positions.shape[1] != 1:
                return None
            else:
                pass
        else:
            pass
        if not self.sub_has_sampled_rows:
            return ("argmax", 0, False, False, False)
        else:
            pass
        if semantic_positions is None:
            return None
        else:
            pass
        return (
            "sampled",
            int(self.sub_sampled_max_top_k),
            bool(self.sub_sampled_has_top_p),
            bool(self.sub_sampled_has_unbounded_top_k),
            bool(self.sub_has_argmax_rows),
        )

    @contextmanager
    def predictor_graph_capture_state(
        self, bucket_size: int, signature: PredictorGraphSignature
    ) -> Generator[None, None, None]:
        saved = (
            self.sub_batch_size,
            self.sub_has_sampled_rows,
            self.sub_sampled_max_top_k,
            self.sub_sampled_has_top_p,
            self.sub_sampled_has_unbounded_top_k,
            self.sub_has_argmax_rows,
        )
        try:
            self.sub_batch_size = bucket_size
            self.sub_has_sampled_rows = signature[0] == "sampled"
            (
                _,
                self.sub_sampled_max_top_k,
                self.sub_sampled_has_top_p,
                self.sub_sampled_has_unbounded_top_k,
                self.sub_has_argmax_rows,
            ) = signature
            yield
        finally:
            (
                self.sub_batch_size,
                self.sub_has_sampled_rows,
                self.sub_sampled_max_top_k,
                self.sub_sampled_has_top_p,
                self.sub_sampled_has_unbounded_top_k,
                self.sub_has_argmax_rows,
            ) = saved

    def predictor_graph_memory_pool(self):
        # Note: (Jiaxin Deng) one shared pool across keys; private per-graph
        # pools would retain intermediates per key and scale with diversity.
        if self.predictor_graph_pool is None:
            self.predictor_graph_pool = self.predictor_device_module.graph_pool_handle()
        else:
            pass
        return self.predictor_graph_pool

    def resolve_predictor_graph_enabled(self) -> bool:
        # Device first: a device that cannot record answers without published config.
        if current_platform.get_device_graph_backend(self.predictor_device) is None:
            return False
        else:
            pass
        # Note: (Jiaxin Deng) capture under TP would record collectives; the
        # graphed chain is only validated single-rank, so TP stays eager.
        if int(get_parallel().tp_size) != 1:
            return False
        else:
            pass
        override = predictor_graph_env_override()
        if override is None:
            should_capture = current_platform.enable_tts_predictor_graph()
        else:
            should_capture = override
        if not should_capture:
            return False
        else:
            pass
        return not bool(get_exec().graph.disable_cuda_graph)

    def capture_predictor_graphs(
        self,
        *,
        do_sample: bool,
        top_k: int,
        top_p: float,
    ) -> int:
        """Capture the bucket ladder of the signatures a batch gets when its rows
        sample with these values, with and without argmax rows mixed in. Buckets
        go in descending order so the smaller ones reuse the pool of the larger
        ones. The set is captured whole and the lazy capture budget counts only
        keys beyond it. The mixed signature skips bucket 1: a mixed batch holds
        a sampled row and an argmax row."""
        if self.predictor_graph_enabled is None:
            self.predictor_graph_enabled = self.resolve_predictor_graph_enabled()
        else:
            pass
        if not self.predictor_graph_enabled:
            return 0
        else:
            pass
        signatures: list[PredictorGraphSignature]
        if do_sample:
            max_top_k, has_top_p, has_unbounded_top_k = predictor_signature_terms(
                [int(top_k)],
                [float(top_p)],
                int(self.config.code_predictor_config.vocab_size),
            )
            signatures = [
                ("sampled", max_top_k, has_top_p, has_unbounded_top_k, has_argmax_rows)
                for has_argmax_rows in (False, True)
            ]
        else:
            signatures = [("argmax", 0, False, False, False)]
        started = time.perf_counter()
        captured_before = len(self.predictor_graphs)
        for signature in signatures:
            for bucket_size in reversed(self.predictor_graph_batch_sizes):
                if signature[4] and bucket_size < 2:
                    continue
                else:
                    pass
                key = (bucket_size, *signature)
                if key in self.predictor_graphs:
                    continue
                else:
                    pass
                self.predictor_graphs[key] = self.capture_predictor_graph(
                    bucket_size, signature
                )
                self.predictor_graph_capture_count += 1
                self.predictor_graph_startup_count += 1
        captured = len(self.predictor_graphs) - captured_before
        elapsed_s = time.perf_counter() - started
        logger.info(
            f"Captured {captured} Qwen3-TTS predictor graphs for "
            f"signatures={signatures} in {elapsed_s:.1f} s"
        )
        return captured

    @torch.no_grad()
    def capture_predictor_graph(
        self,
        bucket_size: int,
        signature: PredictorGraphSignature,
    ) -> PredictorDecodeGraph:
        """One stream per talker for warmups and captures: the allocator only
        reuses a pool block on the stream that freed it. Automatic collection
        is off for the capture because a graph finalizer reached by the
        cyclic collector while a stream is capturing destroys its pool inside
        the capture."""
        device = self.predictor_device
        module = self.predictor_device_module
        backend = current_platform.get_device_graph_backend(device)
        if self.predictor_capture_stream is None:
            self.predictor_capture_stream = module.Stream(device=device)
        else:
            pass
        capture_stream = self.predictor_capture_stream
        current_stream = module.current_stream(device)
        graph = PredictorDecodeGraph(
            bucket_size,
            signature,
            device=device,
            hidden_size=int(self.output_embeds.shape[-1]),
            hidden_dtype=self.output_embeds.dtype,
        )
        # note(ratish): the buffers are zero filled on the current stream and
        # layer0_codes is an embedding index, so the capture stream waits here.
        capture_stream.wait_stream(current_stream)

        def run_once() -> Tuple[torch.Tensor, torch.Tensor]:
            return self.code_predictor_forward_incremental(
                graph.layer0_codes,
                graph.talker_hidden,
                semantic_positions=graph.semantic_positions,
            )

        gc_was_enabled = gc.isenabled()
        gc.disable()
        captured = None
        try:
            # note(ratish): the outer stream context restores the current stream
            # when a failed capture raises from capture_end before the graph
            # context restores it.
            with (
                module.device(device),
                current_platform.graph_capture_attention(),
                self.predictor_graph_capture_state(bucket_size, signature),
                module.stream(capture_stream),
            ):
                for _ in range(_PREDICTOR_GRAPH_WARMUP_PASSES):
                    run_once()
                with backend.capture(
                    pool=self.predictor_graph_memory_pool(),
                    stream=capture_stream,
                    thread_local_errors=True,
                ) as captured:
                    graph.result_codes, graph.summed_embeddings = run_once()
        except Exception:
            # Note: (Jiaxin Deng) release the graph's private memory pool
            # eagerly; the raising object may linger on traceback frames.
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
        graph.graph = captured
        if graph.result_codes is None or graph.summed_embeddings is None:
            raise RuntimeError("Qwen3-TTS predictor graph captured no outputs")
        else:
            pass
        return graph

    def predictor_forward_graphed(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if self.predictor_graph_enabled is None:
            self.predictor_graph_enabled = self.resolve_predictor_graph_enabled()
        else:
            pass
        if not self.predictor_graph_enabled:
            return None
        else:
            pass
        batch_size, seq_len = layer0_codes.shape
        if seq_len != 1 or batch_size == 0:
            return None
        else:
            pass
        if layer0_codes.dtype not in (torch.int, torch.long):
            return None
        else:
            pass
        graph_device = self.predictor_device
        if layer0_codes.device != graph_device or talker_hidden.device != graph_device:
            return None
        else:
            pass
        if batch_size != self.sub_batch_size:
            return None
        else:
            pass
        if self.predictor_device_module.is_current_stream_capturing():
            return None
        else:
            pass
        signature = self.predictor_graph_signature(batch_size, semantic_positions)
        if signature is None:
            return None
        else:
            pass
        bucket_size = self.predictor_graph_bucket_size(batch_size)
        if bucket_size is None:
            return None
        else:
            pass
        key = (bucket_size, *signature)
        if key in self.predictor_graph_disabled:
            return None
        else:
            pass
        graph = self.predictor_graphs.get(key)
        if graph is None:
            lazy_keys = len(self.predictor_graphs) - self.predictor_graph_startup_count
            if lazy_keys >= _PREDICTOR_GRAPH_MAX_LAZY_KEYS:
                self.predictor_graph_capacity_fallback_count += 1
                if not self.predictor_graph_capacity_warned:
                    self.predictor_graph_capacity_warned = True
                    logger.warning(
                        "Qwen3-TTS predictor graph cache holds %d keys beyond "
                        "the startup set; falling back to eager execution for "
                        "uncached key=%s",
                        lazy_keys,
                        key,
                    )
                else:
                    pass
                return None
            else:
                pass
            try:
                graph = self.capture_predictor_graph(bucket_size, signature)
            except Exception:
                self.predictor_graph_disabled.add(key)
                self.predictor_graph_failure_count += 1
                logger.warning(
                    "Disabling Qwen3-TTS predictor graph for key=%s",
                    key,
                    exc_info=True,
                )
                if self.predictor_graph_failure_count >= _PREDICTOR_GRAPH_MAX_FAILURES:
                    self.predictor_graph_enabled = False
                    logger.warning(
                        "Disabling Qwen3-TTS predictor graphs entirely "
                        "after %d capture failures",
                        self.predictor_graph_failure_count,
                    )
                else:
                    pass
                return None
            self.predictor_graphs[key] = graph
            self.predictor_graph_capture_count += 1
            logger.info("Captured Qwen3-TTS predictor graph for key=%s", key)
        else:
            pass
        result = graph.replay(layer0_codes, talker_hidden, semantic_positions)
        return result

    def code_predictor_forward_incremental(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        else:
            pass

        batch_size, seq_len = layer0_codes.shape
        semantic_positions = self.normalize_semantic_positions(
            semantic_positions,
            batch_size=batch_size,
            seq_len=seq_len,
            device=layer0_codes.device,
        )
        predictor_dtype = self.predictor_k_cache.dtype
        num_groups = self.config.num_code_groups
        result_codes = self.output_codes[:batch_size].unsqueeze(-1)
        summed_embeddings = self.output_embeds[:batch_size].unsqueeze(1)
        summed_embeddings.zero_()
        embedding_buffer = getattr(self, "predictor_embedding_buffer", None)
        if embedding_buffer is not None:
            embedding_buffer = embedding_buffer[:batch_size]
        else:
            pass
        projected_tables = self.predictor_projected_embeddings
        if projected_tables is None:
            projected_buffer = None
        else:
            projected_buffer = self.predictor_projected_buffer[:batch_size]

        for pos in range(seq_len):
            layer0_code = layer0_codes[:, pos : pos + 1]
            layer0_embed = self.get_input_embeddings()(layer0_code).to(
                dtype=predictor_dtype
            )
            pos_codes = result_codes[:, :, pos]
            pos_summed = summed_embeddings[:, pos, :]
            pos_codes[:, 0].copy_(layer0_code[:, 0])
            pos_summed.add_(layer0_embed[:, 0, :])

            # note(ratish): the talker hidden and the layer 0 embedding are the
            # first two tokens; one causal pass reads the predictor weights once.
            pair_embeds = self.code_predictor.project_input(
                torch.cat(
                    (
                        talker_hidden[:, pos : pos + 1, :].to(dtype=predictor_dtype),
                        layer0_embed,
                    ),
                    dim=1,
                )
            )
            # note(ratish): dense rows, the gemv and cutedsl lm_head GEMMs view
            # their input as 2D and read it with a fixed row stride.
            last_hidden = self.predictor_forward_tokens(
                token_embeds=pair_embeds,
                batch_size=batch_size,
                cache_len=0,
            )[:, 1:, :].contiguous()
            cache_len = pair_embeds.shape[1]

            sub_positions = (
                self.sub_seed_positions(semantic_positions[:, pos])
                if self.sub_has_sampled_rows
                else None
            )
            for layer_idx in range(num_groups - 1):
                logits, _ = self.code_predictor.lm_head[layer_idx](last_hidden)
                layer_sub_positions = (
                    None if sub_positions is None else sub_positions[layer_idx]
                )
                codec_table = self.code_predictor.model.codec_embedding.weight[
                    layer_idx
                ]
                has_next_step = layer_idx < num_groups - 2
                should_gather_projection = (
                    projected_tables is not None and has_next_step
                )
                new_embed, projected_embed = self.sample_subtalker_embedding(
                    logits[:, -1, :],
                    codec_table,
                    pos_summed,
                    pos_codes[:, layer_idx + 1],
                    projected_tables[layer_idx] if should_gather_projection else None,
                    embedding_buffer,
                    projected_buffer if should_gather_projection else None,
                    layer_sub_positions,
                )
                if has_next_step:
                    if projected_tables is None:
                        new_predictor_embed = self.code_predictor.project_input(
                            new_embed.unsqueeze(1)
                        )
                    else:
                        new_predictor_embed = projected_embed.unsqueeze(1)
                    last_hidden = self.predictor_forward_tokens(
                        token_embeds=new_predictor_embed,
                        batch_size=batch_size,
                        cache_len=cache_len,
                    )
                    cache_len += 1
                else:
                    pass
        return result_codes, summed_embeddings

    def sample_subtalker_embedding(
        self,
        logits: torch.Tensor,
        codec_table: torch.Tensor,
        accumulated: torch.Tensor,
        output_codes: torch.Tensor,
        projected_table: torch.Tensor | None,
        embedding_buffer: torch.Tensor | None,
        projected_buffer: torch.Tensor | None,
        sub_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select a codec token, write its code, and gather and accumulate its embeddings."""
        use_cutedsl = (
            QTTS_BACKEND == "cutedsl"
            and supports_auxiliary(logits)
            and logits.shape[1] == 2048
            and logits.is_contiguous()
            and codec_table.dtype == torch.bfloat16
            and codec_table.shape[1] in (1024, 2048)
            and codec_table.is_contiguous()
            and accumulated.dtype == torch.bfloat16
            and accumulated.is_contiguous()
        )
        if use_cutedsl and not self.sub_has_sampled_rows:
            _, embedding, projected = argmax_embedding(
                logits,
                codec_table,
                projected_table,
                accumulated=accumulated,
                out_codes=output_codes,
            )
            return embedding, projected
        elif (
            use_cutedsl
            and not self.sub_has_argmax_rows
            and not self.sub_sampled_has_unbounded_top_k
            and self.sub_sampled_max_top_k in (50, 64, 128)
        ):
            # note (Hao Wang): smaller top-k buckets retain PyTorch's distinct tie ordering.
            batch_size = logits.shape[0]
            _, embedding, projected = sample_logits(
                logits,
                self.sub_temperature_tensor[:batch_size],
                self.sub_top_k_tensor[:batch_size],
                self.sub_top_p_tensor[:batch_size],
                self.sub_sampling_seed_tensor[:batch_size],
                sub_positions,
                max_top_k=self.sub_sampled_max_top_k,
                table=codec_table,
                accumulated=accumulated,
                projected_table=projected_table,
                out_codes=output_codes,
            )
            return embedding, projected
        elif (
            not use_cutedsl
            and embedding_buffer is not None
            and embedding_buffer.dtype == accumulated.dtype
            and self.sample_subtalker_token_and_gather(
                logits,
                sub_positions=sub_positions,
                codes=output_codes,
                codec_table=codec_table,
                gathered=embedding_buffer,
                summed=accumulated,
                projected_weight=projected_table,
                projected=projected_buffer,
            )
        ):
            return embedding_buffer, projected_buffer
        else:
            next_code = self.sample_subtalker_token(logits, sub_positions=sub_positions)
            output_codes.copy_(next_code)
        if use_cutedsl:
            return embedding_update(
                next_code, codec_table, accumulated, projected_table
            )
        elif (
            embedding_buffer is not None
            and embedding_buffer.dtype == accumulated.dtype
            and gather_codec_embedding_and_add(
                next_code,
                codec_table,
                embedding_buffer,
                accumulated,
                projected_table,
                projected_buffer,
            )
        ):
            return embedding_buffer, projected_buffer
        else:
            embedding = torch.nn.functional.embedding(next_code, codec_table).to(
                accumulated.dtype
            )
            accumulated.add_(embedding)
            projected = (
                torch.nn.functional.embedding(next_code, projected_table)
                if projected_table is not None
                else None
            )
            return embedding, projected

    def normalize_semantic_positions(
        self,
        semantic_positions: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if semantic_positions is None:
            base = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            base = semantic_positions.to(device=device, dtype=torch.long)
            if base.ndim == 2:
                if base.shape != (batch_size, seq_len):
                    raise ValueError("Qwen3-TTS subtalker positions shape mismatch")
                else:
                    pass
                return base
            else:
                pass
            if base.ndim != 1 or base.shape[0] != batch_size:
                raise ValueError("Qwen3-TTS subtalker positions shape mismatch")
            else:
                pass
        offsets = torch.arange(seq_len, device=device, dtype=torch.long)
        return base.unsqueeze(1) + offsets.unsqueeze(0)

    def sub_seed_positions(self, semantic_positions: torch.Tensor) -> torch.Tensor:
        """Seed positions of every sub-step of one decode position, sub-step
        first, so each sub-step reads its row without a kernel."""
        group_stride = max(int(self.config.num_code_groups) - 1, 1)
        return torch.add(
            self.sub_seed_offsets.unsqueeze(1),
            semantic_positions.unsqueeze(0),
            alpha=group_stride,
        )

    def sample_subtalker_token(
        self,
        logits: torch.Tensor,
        *,
        sub_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        if logits.shape[0] == 0:
            return torch.empty((0,), device=logits.device, dtype=torch.long)
        else:
            pass
        batch_size = int(logits.shape[0])
        if batch_size > self.sub_batch_size:
            raise RuntimeError("Qwen3-TTS subtalker sampling buffers are too small")
        else:
            pass

        if not self.sub_has_sampled_rows:
            return torch.argmax(logits, dim=-1).to(dtype=torch.long)
        else:
            pass

        sampled_tokens = self.sample_subtalker_token_seeded(
            logits,
            sub_positions=sub_positions,
        )
        if not self.sub_has_argmax_rows:
            return sampled_tokens
        else:
            pass
        argmax_tokens = torch.argmax(logits, dim=-1).to(dtype=torch.long)
        return torch.where(
            self.sub_do_sample_tensor[:batch_size],
            sampled_tokens,
            argmax_tokens,
        )

    def sample_subtalker_token_and_gather(
        self,
        logits: torch.Tensor,
        *,
        sub_positions: torch.Tensor | None,
        codes: torch.Tensor,
        codec_table: torch.Tensor,
        gathered: torch.Tensor,
        summed: torch.Tensor,
        projected_weight: torch.Tensor | None,
        projected: torch.Tensor | None,
    ) -> bool:
        """Sample a codebook's codes into codes and gather their codec rows in the
        sampler's last launch when every row samples; False leaves both to the caller.
        """
        batch_size = int(logits.shape[0])
        if (
            not self.sub_has_sampled_rows
            or self.sub_has_argmax_rows
            or batch_size > self.sub_batch_size
        ):
            return False
        else:
            pass
        return sample_codes_and_gather_codec_embedding(
            logits,
            self.sub_temperature_tensor[:batch_size],
            self.sub_top_k_tensor[:batch_size],
            self.sub_top_p_tensor[:batch_size],
            self.sub_sampling_seed_tensor[:batch_size],
            sub_positions,
            max_top_k=int(self.sub_sampled_max_top_k),
            has_top_p=bool(self.sub_sampled_has_top_p),
            codes=codes,
            embedding_weight=codec_table,
            gathered=gathered,
            accumulated=summed,
            projected_weight=projected_weight,
            projected=projected,
        )

    def sample_subtalker_token_seeded(
        self,
        logits: torch.Tensor,
        *,
        sub_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(logits.shape[0])
        vocab_size = int(logits.shape[-1])
        temperatures = self.sub_temperature_tensor[:batch_size]
        top_ks = self.sub_top_k_tensor[:batch_size]
        top_ps = self.sub_top_p_tensor[:batch_size]
        seeds = self.sub_sampling_seed_tensor[:batch_size]
        max_top_k = int(self.sub_sampled_max_top_k)
        has_unbounded_top_k = bool(self.sub_sampled_has_unbounded_top_k)

        if logits.is_cuda:
            fused_sampled = sample_from_logits_with_seed_top_k_top_p(
                logits,
                temperatures,
                top_ks,
                top_ps,
                seeds,
                sub_positions,
                max_top_k=max_top_k,
                has_top_p=bool(self.sub_sampled_has_top_p),
            )
            if fused_sampled is not None:
                return fused_sampled.to(torch.long)
            else:
                pass
        else:
            pass

        scores = logits.float() / temperatures.unsqueeze(1)
        if max_top_k > 0 and max_top_k < vocab_size and not has_unbounded_top_k:
            sorted_scores, sorted_idx = torch.topk(scores, max_top_k, dim=-1)
            if not self.sub_sampled_has_top_p:
                sampled = sample_top_k(
                    sorted_scores, sorted_idx, top_ks, seeds, sub_positions
                )
                if sampled is not None:
                    return sampled
                else:
                    pass
            else:
                pass
            rank = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
            keep_top_k = rank < top_ks.unsqueeze(1)
            sorted_scores = sorted_scores.masked_fill(~keep_top_k, -float("inf"))
        else:
            sorted_scores, sorted_idx = torch.sort(scores, dim=-1, descending=True)
            rank = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
            keep_all = (top_ks <= 0) | (top_ks >= vocab_size)
            keep_top_k = keep_all.unsqueeze(1) | (rank < top_ks.unsqueeze(1))
            sorted_scores = sorted_scores.masked_fill(~keep_top_k, -float("inf"))

        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        if self.sub_sampled_has_top_p:
            active_top_p = (top_ps > 0.0) & (top_ps < 1.0)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            remove = (
                cdf - sorted_probs >= top_ps.unsqueeze(1)
            ) & active_top_p.unsqueeze(1)
            remove[:, 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, -float("inf"))
        else:
            pass
        sorted_probs = sorted_probs.masked_fill(~keep_top_k, -float("inf"))
        sorted_logprobs = torch.where(
            sorted_probs > 0,
            torch.log(sorted_probs),
            torch.full_like(sorted_probs, -float("inf")),
        )

        if (
            QTTS_BACKEND == "cutedsl"
            and supports_auxiliary(logits)
            and sorted_logprobs.shape[1] <= 2048
        ):
            return sample_sorted(sorted_logprobs, sorted_idx, seeds, sub_positions)
        else:
            pass
        sampled = sample_from_sorted_logprobs_with_seed_small_k(
            sorted_logprobs,
            sorted_idx,
            seeds,
            sub_positions,
        )
        if sampled is not None:
            return sampled.to(torch.long)
        else:
            pass

        sampled_rank = sample_seeded_categorical(
            sorted_logprobs,
            seeds,
            sub_positions,
        ).to(device=logits.device, dtype=torch.long)
        return sorted_idx.gather(1, sampled_rank.unsqueeze(1)).view(-1).to(torch.long)

    def predictor_forward_tokens(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        """Run the predictor stack on token_embeds [batch, tokens, hidden] at slots
        cache_len onward: one token at any slot, or the pair at slots 0 and 1."""
        if token_embeds.ndim != 3 or token_embeds.shape[0] != batch_size:
            raise ValueError(
                "Predictor embeddings must have shape [batch, tokens, hidden]"
            )
        else:
            pass
        if self.code_predictor_spec_correction_forward_watch:
            self.code_predictor_spec_standalone_correction_forward_count += 1
            self.code_predictor_spec_correction_forward_watch = False
        else:
            pass
        num_tokens, hidden_size = token_embeds.shape[1:]
        expected_hidden_size = int(
            self.code_predictor.model.layers[0].input_layernorm.weight.shape[0]
        )
        if hidden_size != expected_hidden_size:
            raise ValueError(
                "Predictor embedding hidden size does not match model layer weights"
            )
        else:
            pass
        if num_tokens == 1:
            positions = self.predictor_position_rows[cache_len, :batch_size]
            cache_slots = self.predictor_cache_slots[cache_len, :batch_size]
        else:
            assert (num_tokens, cache_len) == (2, 0), "pair tables hold slots 0 and 1"
            positions = self.predictor_pair_positions[: 2 * batch_size]
            cache_slots = self.predictor_pair_cache_slots[: 2 * batch_size]
        num_rows = batch_size * num_tokens
        if (
            self.predictor_cutedsl_layers is not None
            and 1 <= num_rows <= MAX_DECODE_ROWS
        ):
            hidden_states = token_embeds.reshape(num_rows, hidden_size)
            end = cache_len + num_tokens
            for layer_idx, fused_layer in enumerate(self.predictor_cutedsl_layers):
                attention = self.code_predictor.model.layers[layer_idx].self_attn
                q = fused_layer.pre_attention(
                    hidden_states,
                    positions,
                    attention.rotary_emb.cos_sin_cache,
                    cache_slots,
                    self.predictor_k_rows[layer_idx].view(-1, KV_HEADS, HEAD_DIM),
                    self.predictor_v_rows[layer_idx].view(-1, KV_HEADS, HEAD_DIM),
                )
                attention_output = torch.empty_like(q)
                decode_attention(
                    q=q,
                    tokens_per_row=num_tokens,
                    k_cache=self.predictor_k_cache[layer_idx, :batch_size],
                    v_cache=self.predictor_v_cache[layer_idx, :batch_size],
                    end=end,
                    attention_output=attention_output,
                )
                hidden_states = fused_layer.post_attention(
                    attention_output,
                    hidden_states,
                )
            return self.code_predictor.model.norm(hidden_states).view(
                batch_size, num_tokens, hidden_size
            )
        else:
            pass
        fused = self.predictor_fused_layers
        if fused is not None and fused.covers(num_rows):
            return fused.forward(
                layers=self.code_predictor.model.layers,
                final_norm=self.code_predictor.model.norm,
                token_embeds=token_embeds,
                batch_size=batch_size,
                cache_len=cache_len,
                positions=positions,
                k_cache=self.predictor_k_cache,
                v_cache=self.predictor_v_cache,
            )
        else:
            pass
        self.code_predictor_generic_forward_count += 1
        # note(ratish): 2D rows for the fused add and norm, which every
        # backend's kernel expects and which writes both operands in place.
        residual = token_embeds.reshape(num_rows, 1, hidden_size)
        mlp_out: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.code_predictor.model.layers):
            if mlp_out is None:
                normed = layer.input_layernorm(residual.reshape(-1, hidden_size))
            else:
                normed, residual = layer.input_layernorm(
                    mlp_out, residual.reshape(-1, hidden_size)
                )
                residual = residual.reshape(num_rows, 1, hidden_size)
            attn_input = self.predictor_cached_self_attention(
                layer_idx=layer_idx,
                attn=layer.self_attn,
                hidden_states=normed.reshape(batch_size, num_tokens, hidden_size),
                positions=positions,
                cache_slots=cache_slots,
                cache_len=cache_len,
            )
            residual = self.predictor_o_proj_add_residual(
                layer.self_attn.o_proj,
                attn_input,
                residual,
            )
            normed = layer.post_attention_layernorm(residual.reshape(-1, hidden_size))
            mlp_out = layer.mlp(normed)
        normed, _ = self.code_predictor.model.norm(
            mlp_out, residual.reshape(-1, hidden_size)
        )
        return normed.reshape(batch_size, num_tokens, hidden_size)

    def predictor_forward_packed_tokens(
        self,
        *,
        token_embeds: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Run packed Predictor queries only through the production fused stack."""
        if token_embeds.ndim != 3 or token_embeds.shape[0] != batch_size:
            raise ValueError(
                "packed Predictor embeddings must have shape [batch, width, hidden]"
            )
        else:
            pass
        if positions.shape != (batch_size * token_embeds.shape[1],):
            raise ValueError("packed Predictor positions must align with query rows")
        elif positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("packed Predictor positions must use an integer dtype")
        else:
            pass
        if (
            token_embeds.device != self.predictor_device
            or positions.device != self.predictor_device
        ):
            raise ValueError("packed Predictor tensors must use the Predictor device")
        else:
            pass
        fused = self.predictor_fused_layers
        packed_rows = batch_size * token_embeds.shape[1]
        if fused is None or not fused.covers(packed_rows):
            raise RuntimeError("packed Predictor fused scratch capacity is unavailable")
        else:
            pass
        if token_embeds.shape[2] != fused.shape.hidden_size:
            raise ValueError(
                "packed Predictor hidden size does not match fused layer weights"
            )
        else:
            pass
        attention = self.code_predictor.model.layers[0].self_attn
        if (
            token_embeds.shape[1]
            * (int(attention.num_heads) // int(attention.num_kv_heads))
            > ATTENTION_ROWS
        ):
            raise RuntimeError("packed Predictor exceeds fused attention tile capacity")
        else:
            pass
        return fused.forward_packed(
            layers=self.code_predictor.model.layers,
            final_norm=self.code_predictor.model.norm,
            token_embeds=token_embeds,
            batch_size=batch_size,
            positions=positions,
            k_cache=self.predictor_k_cache,
            v_cache=self.predictor_v_cache,
        )

    @staticmethod
    def predictor_o_proj_add_residual(
        o_proj: RowParallelLinear,
        attn_input: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Run the Predictor attention output projection and residual add."""

        weight = o_proj.weight
        # note(ratish): the fusion stands in for sglang's linear only where that
        # linear is torch's GEMM: batch invariant mode overrides aten::addmm but
        # not the out variant, and the cutedsl backend replaces F.linear on sm100.
        use_fused_addmm = (
            attn_input.is_cuda
            and not is_batch_invariant_mode_enabled()
            and not get_bf16_gemm_backend().is_cutedsl()
            and not torch.is_grad_enabled()
            and get_group_rank_size(o_proj.tp_group)[1] == 1
            and o_proj.bias is None
            and weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.is_contiguous()
            and attn_input.dtype == torch.bfloat16
            and attn_input.ndim == 2
            and attn_input.is_contiguous()
            and attn_input.device == weight.device
            and attn_input.shape[1] == weight.shape[1]
            and residual.is_cuda
            and residual.dtype == torch.bfloat16
            and residual.ndim == 3
            and residual.is_contiguous()
            and residual.device == weight.device
            and residual.shape == (attn_input.shape[0], 1, weight.shape[0])
        )
        if use_fused_addmm:
            residual_2d = residual.reshape(attn_input.shape[0], weight.shape[0])
            # Note (Jun Liu): The caller replaces ``residual`` immediately after
            # this call. Reuse its storage so addmm's beta term needs no copy.
            torch.addmm(
                residual_2d,
                attn_input,
                weight.t(),
                out=residual_2d,
            )
            return residual
        else:
            pass

        attn_output, _ = o_proj(attn_input)
        return residual + attn_output.reshape(
            attn_input.shape[0], 1, residual.shape[-1]
        )

    @staticmethod
    def resolve_predictor_rope_store(
        attn: Qwen3OmniMoeThinkerTextAttention, *, device: torch.device
    ) -> bool:
        """Resolve store support before capture: a supported CUDA head size
        does not imply CUDA dispatch when the backend override selects Torch."""
        return (
            device.type == "cuda"
            and attn.compatible_with_fused_kv_buffer
            and not attn.rotary_emb.use_fallback_kernel
            and get_fused_op_backend()
            not in (KernelBackend.TORCH, KernelBackend.TORCH_COMPILE)
        )

    def predictor_cached_self_attention(
        self,
        *,
        layer_idx: int,
        attn: Qwen3OmniMoeThinkerTextAttention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        batch_size, num_tokens, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_size)
        qkv, _ = attn.qkv_proj(flat_hidden)
        q_linear, k_linear, v = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
        )
        q, k = apply_qk_norm(
            q=q_linear,
            k=k_linear,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            head_dim=attn.head_dim,
            alt_stream=attn.alt_stream,
        )
        if self.predictor_rope_stores_kv:
            store = FusedSetKVBufferArg(
                value=v,
                k_buffer=self.predictor_k_rows[layer_idx],
                v_buffer=self.predictor_v_rows[layer_idx],
                cache_loc=cache_slots,
            )
        else:
            store = None
        q, k = attn.rotary_emb(
            positions.to(device=flat_hidden.device, dtype=torch.long),
            q,
            k,
            fused_set_kv_buffer_arg=store,
        )
        end = cache_len + num_tokens
        if store is None:
            self.predictor_k_cache[layer_idx, :batch_size, cache_len:end].copy_(
                k.view(batch_size, num_tokens, attn.num_kv_heads, attn.head_dim)
            )
            self.predictor_v_cache[layer_idx, :batch_size, cache_len:end].copy_(
                v.view(batch_size, num_tokens, attn.num_kv_heads, attn.head_dim)
            )
        else:
            pass
        q = q.reshape(batch_size, num_tokens, attn.num_heads, attn.head_dim).transpose(
            1, 2
        )
        cached_k = self.predictor_k_cache[layer_idx, :batch_size, :end].transpose(1, 2)
        cached_v = self.predictor_v_cache[layer_idx, :batch_size, :end].transpose(1, 2)
        attn_output = predictor_gqa_attention(
            q,
            cached_k,
            cached_v,
            num_heads=attn.num_heads,
            num_key_value_heads=attn.num_kv_heads,
            is_causal=num_tokens > 1,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size * num_tokens, attn.num_heads * attn.head_dim
        )
        return attn_output

    def precompile_kernels_after_loading(self) -> None:
        """Compile the repetition penalty every request reaches and the seeded draw a
        request seed reaches, on the codec head's fp32 logits, and load FlashInfer's
        sampling module when it draws the unseeded requests."""
        compile_sampler_functions(
            SamplerCompileForms(
                device=self.codec_head.weight.device,
                vocab_size=self.config.vocab_size,
                seeded_log_probability_dtype=torch.float64,
                penalized_logits_dtype=torch.float32,
                should_load_flashinfer_sampling=(
                    get_exec().kernel.sampling_backend == "flashinfer"
                ),
            )
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = self.cached_params_dict
        stacked_params = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        for name, loaded_weight in weights:
            if name.startswith("talker."):
                target = name[len("talker.") :]
            elif name.startswith("speaker_encoder."):
                target = name
            else:
                continue

            codec_table_slot = self.codec_table_params_mapping.get(target)
            if codec_table_slot is not None:
                param_name, codebook_index = codec_table_slot
                param = params_dict[param_name]
                param.weight_loader(param, loaded_weight, codebook_index)
                continue
            else:
                pass

            handled = False
            for param_name, weight_name, shard_id in stacked_params:
                if weight_name in target:
                    param = params_dict.get(target.replace(weight_name, param_name))
                    if param is not None:
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                        break
                    else:
                        pass
                else:
                    pass
            if handled:
                continue
            else:
                pass
            param = params_dict.get(target)
            if param is not None:
                param.weight_loader(param, loaded_weight)
            else:
                pass
        self.post_load_weights()

    @torch.no_grad()
    def post_load_weights(self) -> None:
        """Rebuild the projected codec tables; the loaders call this after every
        weight load."""
        self.copy_code_predictor_spec_lm_head_weights()
        if self.predictor_projected_embeddings is None:
            pass
        else:
            for table, codec_table in zip(
                self.predictor_projected_embeddings,
                self.code_predictor.model.codec_embedding.weight,
            ):
                table.copy_(self.code_predictor.project_input(codec_table))

        if QTTS_BACKEND == "cutedsl":
            if get_parallel().tp_size != 1:
                logger.warning("CuTeDSL decode disabled for tensor parallelism")
            elif (
                get_exec().deterministic.enable_deterministic_inference
                or is_batch_invariant_mode_enabled()
            ):
                logger.warning("CuTeDSL decode disabled for deterministic inference")
            else:
                self.model.cutedsl_layers = build_cutedsl_layers(self.model.layers)
                self.predictor_cutedsl_layers = build_cutedsl_layers(
                    self.code_predictor.model.layers, is_predictor=True
                )
                logger.info(
                    f"Qwen3-TTS CuTeDSL decode: talker={self.model.cutedsl_layers is not None}, "
                    f"predictor={self.predictor_cutedsl_layers is not None}; token rows <=32"
                )
        else:
            pass


EntryClass = Qwen3TTSTalker

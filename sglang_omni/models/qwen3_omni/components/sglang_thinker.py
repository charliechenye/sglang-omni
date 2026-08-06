# SPDX-License-Identifier: Apache-2.0
"""SGLang text-only thinker wrapper for Qwen3-Omni.

The upstream SGLang Qwen3-Omni class builds ``thinker.audio_tower`` and
``thinker.visual`` inside the thinker process. Our pipeline already owns those
encoders as standalone stages and injects their embeddings before thinker
prefill, so this wrapper keeps only the text model and LM head.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel, load_fused_expert_weights
from sglang.srt.utils import add_prefix, logger

from sglang_omni.quantization import get_weight_preprocessor


def _config_has_mrope(config: Any) -> bool:
    """Return whether a Qwen config declares multi-dimensional RoPE."""
    if config is None:
        return False
    for field_name in ("rope_parameters", "rope_scaling"):
        value = getattr(config, field_name, None)
        if isinstance(value, Mapping) and "mrope_section" in value:
            return True
    return False


def _qwen_mrope_enabled(*configs: Any) -> bool:
    """Expose the outer model's M-RoPE contract to upstream graph capture."""
    return any(_config_has_mrope(config) for config in configs)


def _cpu_int_list(values: Any, *, name: str) -> list[int]:
    """Convert scheduler-owned host metadata without a device synchronization."""
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise RuntimeError(f"{name} must be a CPU tensor")
        values = values.tolist()
    return [int(value) for value in values]


def _qwen_mm_inputs(forward_batch: Any) -> tuple[list[Any], list[Any]]:
    """Return per-request multimodal shells and their flattened items."""
    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if mm_inputs is None:
        return [], []
    if not isinstance(mm_inputs, (list, tuple)):
        mm_inputs = [mm_inputs]

    flattened: list[Any] = []
    for mm_input in mm_inputs:
        if mm_input is None:
            continue
        items = getattr(mm_input, "mm_items", None) or ()
        flattened.extend(item for item in items if item is not None)
    return list(mm_inputs), flattened


def _is_standard_qwen_audio_item(item: Any) -> bool:
    modality = getattr(item, "modality", None)
    modality_name = getattr(modality, "name", modality)
    is_precomputed = getattr(item, "is_precomputed_embedding", None)
    if callable(is_precomputed):
        is_precomputed = is_precomputed()
    else:
        is_precomputed = getattr(item, "precomputed_embeddings", None) is not None
    return modality_name in ("AUDIO", "audio") and bool(is_precomputed)


def _standard_qwen_audio_items(forward_batch: Any) -> tuple[list[Any], bool]:
    """Classify only the Qwen-native audio contract.

    The classification is deliberately about representation: a precomputed
    audio item can be consumed by this model's ordinary ``forward``. It does
    not ask whether an upstream CUDA graph will replay the batch.
    """
    mm_inputs, items = _qwen_mm_inputs(forward_batch)
    has_mm_shell = any(mm_input is not None for mm_input in mm_inputs)
    if not items:
        return [], has_mm_shell
    if not all(_is_standard_qwen_audio_item(item) for item in items):
        raise RuntimeError(
            "Qwen3-Omni multimodal payload is not representable by the "
            "standard thinker forward contract"
        )
    return items, has_mm_shell


def _qwen_audio_positions(item: Any) -> torch.Tensor:
    model_specific_data = getattr(item, "model_specific_data", None) or {}
    positions = model_specific_data.get("positions_cpu")
    if not isinstance(positions, torch.Tensor):
        raise RuntimeError(
            "Qwen3-Omni native audio items must carry positions_cpu metadata"
        )
    if positions.device.type != "cpu" or positions.dim() != 1:
        raise RuntimeError("Qwen3-Omni audio positions_cpu must be a 1-D CPU tensor")
    if positions.dtype != torch.long:
        raise RuntimeError("Qwen3-Omni audio positions_cpu must use int64")
    if positions.numel() > 1 and not torch.all(positions[1:] > positions[:-1]):
        raise RuntimeError("Qwen3-Omni audio positions_cpu must be sorted")
    return positions


def _forward_mode_flag(forward_mode: Any, name: str) -> bool:
    value = getattr(forward_mode, name, False)
    return bool(value() if callable(value) else value)


class Qwen3OmniThinkerForCausalLM(nn.Module):
    """Qwen3-Omni thinker text model without duplicated audio/vision towers."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.root_config = config
        self.thinker_config = getattr(config, "thinker_config", config)
        self.config = getattr(self.thinker_config, "text_config", self.thinker_config)
        self.is_mrope_enabled = _qwen_mrope_enabled(
            self.root_config,
            self.thinker_config,
            self.config,
        )

        self.model = Qwen3MoeLLMModel(
            config=self.config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(self.config)

    def _compose_standard_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None,
        audio_items: list[Any],
    ) -> torch.Tensor:
        """Compose text and native audio rows into the supplied model buffer.

        ``input_embeds`` is the stable model buffer selected by the outer
        forward from the upstream model argument or replay ForwardBatch. The
        live serving ForwardBatch keeps its eligibility input empty. With a
        supplied buffer, text is copied into that stable storage first and
        audio rows are then replaced in place.

        Pros: the encoder output remains owned by the multimodal item, text is
        embedded exactly once, and eager and graph paths share the same model
        contract. Cons: the graph-compatible path still needs one text-to-stable
        copy and one audio placement operation; the encoder output may require a
        device/dtype conversion when its producer differs from the thinker.
        """
        embed_tokens = self.model.get_input_embeddings()
        embed_input_ids = input_ids.clamp(
            min=0, max=embed_tokens.num_embeddings - 1
        )
        text_embeds = embed_tokens(embed_input_ids)
        target = input_embeds
        if target is None:
            target = text_embeds
        else:
            target.copy_(text_embeds)

        mm_inputs, _ = _qwen_mm_inputs(forward_batch)
        extend_lens = _cpu_int_list(
            getattr(forward_batch, "extend_seq_lens_cpu", None),
            name="extend_seq_lens_cpu",
        )
        prefix_lens = _cpu_int_list(
            getattr(forward_batch, "extend_prefix_lens_cpu", None),
            name="extend_prefix_lens_cpu",
        )
        if not prefix_lens:
            prefix_lens = [0] * len(extend_lens)
        if len(mm_inputs) != len(extend_lens) or len(prefix_lens) != len(
            extend_lens
        ):
            raise RuntimeError(
                "Qwen3-Omni multimodal and extend metadata do not have the "
                "same batch length"
            )

        expected_items = [
            item
            for mm_input in mm_inputs
            if mm_input is not None
            for item in (getattr(mm_input, "mm_items", None) or ())
            if item is not None
        ]
        if len(expected_items) != len(audio_items) or any(
            expected is not actual
            for expected, actual in zip(expected_items, audio_items)
        ):
            raise RuntimeError("Qwen3-Omni audio item metadata changed during forward")

        source_parts: list[torch.Tensor] = []
        destination_parts: list[torch.Tensor] = []
        flat_start = 0
        for request_index, mm_input in enumerate(mm_inputs):
            extend_len = extend_lens[request_index]
            prefix_len = prefix_lens[request_index]
            for item in (getattr(mm_input, "mm_items", None) or ()):
                if item is None:
                    continue
                positions_cpu = _qwen_audio_positions(item)
                source = getattr(item, "precomputed_embeddings", None)
                if not isinstance(source, torch.Tensor) or source.dim() != 2:
                    raise RuntimeError(
                        "Qwen3-Omni native audio items must carry a 2-D "
                        "precomputed_embeddings tensor"
                    )
                if source.shape[0] != positions_cpu.numel():
                    raise ValueError(
                        "Qwen3-Omni audio rows changed after request construction"
                    )
                if source.shape[1] != target.shape[1]:
                    raise ValueError(
                        "Qwen3-Omni audio embedding width does not match the "
                        "thinker hidden size"
                    )

                in_chunk = (positions_cpu >= prefix_len) & (
                    positions_cpu < prefix_len + extend_len
                )
                source_indices = in_chunk.nonzero(as_tuple=True)[0]
                if source_indices.numel() == 0:
                    continue
                source_start = int(source_indices[0])
                source_count = int(source_indices.numel())
                source_end = source_start + source_count
                selected_positions = positions_cpu[source_start:source_end]
                destination_cpu = selected_positions - prefix_len + flat_start
                if int(destination_cpu[-1]) >= target.shape[0]:
                    raise RuntimeError(
                        "Qwen3-Omni audio placement exceeds the forward buffer"
                    )

                source_parts.append(source[source_start:source_end])
                destination_parts.append(destination_cpu)
            flat_start += extend_len

        if destination_parts:
            destination_cpu = (
                destination_parts[0]
                if len(destination_parts) == 1
                else torch.cat(destination_parts, dim=0)
            )
            destination = destination_cpu.to(device=target.device, non_blocking=True)

            first_source = source_parts[0]
            same_source_layout = all(
                source_part.device == first_source.device
                and source_part.dtype == first_source.dtype
                for source_part in source_parts[1:]
            )
            if same_source_layout:
                source_rows = (
                    first_source
                    if len(source_parts) == 1
                    else torch.cat(source_parts, dim=0)
                )
            else:
                source_rows = torch.cat(
                    [
                        source_part.to(
                            device=target.device,
                            dtype=target.dtype,
                            non_blocking=True,
                        )
                        for source_part in source_parts
                    ],
                    dim=0,
                )
            if source_rows.device != target.device or source_rows.dtype != target.dtype:
                source_rows = source_rows.to(
                    device=target.device,
                    dtype=target.dtype,
                    non_blocking=True,
                )
            target.index_copy_(0, destination, source_rows)

        return target

    @property
    def thinker(self) -> "Qwen3OmniThinkerForCausalLM":
        # Existing Qwen thinker runner/hook code expects model.thinker.model.
        return self

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: Any,
        get_embedding: bool = False,
        pp_proxy_tensors: Any | None = None,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ):
        del get_embedding
        if getattr(forward_batch, "mrope_positions", None) is not None:
            positions = forward_batch.mrope_positions

        audio_items, has_mm_shell = _standard_qwen_audio_items(forward_batch)
        model_input_ids = input_ids
        # The serving batch keeps this field empty so upstream owns graph
        # eligibility. During graph replay, upstream replaces it with the
        # graph-owned stable buffer on its static ForwardBatch.
        stable_input_embeds = input_embeds
        if stable_input_embeds is None:
            stable_input_embeds = getattr(forward_batch, "input_embeds", None)
        model_input_embeds = stable_input_embeds
        phase = getattr(forward_batch, "forward_mode", None)
        is_decode = _forward_mode_flag(phase, "is_decode")
        is_target_verify = _forward_mode_flag(phase, "is_target_verify")
        pp_group = getattr(self.model, "pp_group", None)
        is_first_pp_rank = getattr(pp_group, "is_first_rank", True)

        if not is_decode and not is_target_verify and is_first_pp_rank:
            effective_input_ids = input_ids
            if effective_input_ids is None:
                effective_input_ids = getattr(forward_batch, "input_ids", None)
            if effective_input_ids is None:
                raise RuntimeError("Qwen3-Omni forward requires input_ids for prefill")

            if audio_items:
                model_input_embeds = self._compose_standard_audio_embeddings(
                    effective_input_ids,
                    forward_batch,
                    stable_input_embeds,
                    audio_items,
                )
                model_input_ids = None
            elif (
                stable_input_embeds is not None
                and (input_embeds is None or not has_mm_shell)
            ):
                # Upstream graph replay supplies the graph-owned stable buffer as
                # this ForwardBatch field (or model argument). Text-only eager
                # must keep input_ids so the inner model performs its ordinary
                # embedding path. An empty M-RoPE shell is still standard; an
                # explicit embedding argument on a legacy shell is preserved.
                embed_tokens = self.model.get_input_embeddings()
                text_embeds = embed_tokens(
                    effective_input_ids.clamp(
                        min=0, max=embed_tokens.num_embeddings - 1
                    )
                )
                stable_input_embeds.copy_(text_embeds)
                model_input_embeds = stable_input_embeds
                model_input_ids = None

        hidden_states = self.model(
            input_ids=model_input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=model_input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            input_deepstack_embeds=input_deepstack_embeds,
        )
        logits_input_ids = input_ids
        if logits_input_ids is None:
            logits_input_ids = getattr(forward_batch, "input_ids", None)
        return self.logits_processor(
            logits_input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load only thinker text/LM-head weights from the Omni checkpoint."""
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        base_expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        num_experts = self.config.num_experts

        preprocess_weight = get_weight_preprocessor(
            self.root_config, fp8_scale_inverted=True
        )

        for name, loaded_weight in weights:
            name = name.replace("model.language_model.", "model.")
            if name.startswith("thinker."):
                name = name[len("thinker.") :]
            elif name.startswith(("talker.", "code2wav.")):
                continue

            if name.startswith(("audio_tower.", "visual.")):
                continue

            is_fused_expert = False
            expert_params_mapping = base_expert_params_mapping

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue

                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(ignore_suffixes) and mapped not in params_dict:
                    continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                loaded_weight = preprocess_weight(mapped, loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        loaded = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            gate_weight, up_weight = loaded.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                mapped, params_dict, gate_weight, "w1", num_experts
                            )
                            load_fused_expert_weights(
                                mapped, params_dict, up_weight, "w3", num_experts
                            )
                        else:
                            load_fused_expert_weights(
                                mapped,
                                params_dict,
                                loaded,
                                shard_id,
                                num_experts,
                            )
                    else:
                        if (
                            mapped.endswith(ignore_suffixes)
                            and mapped not in params_dict
                        ):
                            continue
                        param = params_dict.get(mapped)
                        if param is None:
                            continue
                        loaded_weight = preprocess_weight(mapped, loaded_weight)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    param = params_dict.get(name)
                    if param is not None:
                        loaded_weight = preprocess_weight(name, loaded_weight)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    elif name.startswith(("model.", "lm_head.")):
                        logger.warning(
                            "Loaded thinker weight %s not found in text-only params",
                            name,
                        )


EntryClass = Qwen3OmniThinkerForCausalLM

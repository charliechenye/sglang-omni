# SPDX-License-Identifier: Apache-2.0
"""SGLang text-only thinker wrapper for Qwen3-Omni.

The upstream SGLang Qwen3-Omni class builds ``thinker.audio_tower`` and
``thinker.visual`` inside the thinker process. Our pipeline owns those encoders
as standalone stages and carries their qualified precomputed audio output into
the text model's normal forward, so this wrapper keeps only the text model and
LM head.
"""

from __future__ import annotations

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


def _cpu_int_list(values: Any, *, field_name: str) -> list[int]:
    """Read SGLang's CPU chunk metadata without creating a device sync."""
    if values is None:
        raise RuntimeError(
            f"Qwen3-Omni audio placement requires {field_name} CPU metadata"
        )
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise RuntimeError(
                f"Qwen3-Omni audio placement expected CPU {field_name}, "
                f"got {values.device}"
            )
        values = values.tolist()
    return [int(value) for value in values]


def _qwen_mm_items(forward_batch: Any) -> list[Any]:
    """Flatten standard multimodal items while preserving their tensor refs."""
    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if not mm_inputs:
        return []
    items: list[Any] = []
    for mm_input in mm_inputs:
        if mm_input is None:
            continue
        items.extend(getattr(mm_input, "mm_items", ()) or ())
    return items


def _compose_qwen_prefill_embeddings(
    *,
    input_ids: torch.Tensor,
    forward_batch: Any,
    input_embedding: Any,
    base_input_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose text and standard precomputed audio embeddings for model.forward.

    The normal eager path returns the text embedding allocation directly and
    scatters audio once across the flattened batch. If upstream supplies its
    graph-owned stable input buffer, this copies the composed text rows into
    that buffer before the same audio scatter. A live base is accepted only for
    a mixed batch where the legacy hook already composed another modality. This
    helper only implements the Qwen model contract; upstream remains
    responsible for choosing replay or eager execution.
    """
    if base_input_embeds is not None:
        target = base_input_embeds
    else:
        vocab_size = int(input_embedding.num_embeddings)
        text_embeds = input_embedding(input_ids.clamp(min=0, max=vocab_size - 1))

        target = getattr(forward_batch, "input_embeds", None)
        if target is not None:
            if target.shape != text_embeds.shape:
                raise RuntimeError(
                    "Qwen3-Omni stable input_embeds shape does not match prefill ids: "
                    f"buffer={tuple(target.shape)}, text={tuple(text_embeds.shape)}"
                )
            # Ownership: this is the one full copy into SGLang's stable graph
            # buffer. The live eager ForwardBatch remains input_embeds=None.
            target.copy_(text_embeds)
        else:
            target = text_embeds

    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if not mm_inputs or not _qwen_mm_items(forward_batch):
        return target

    prefix_lens = _cpu_int_list(
        getattr(forward_batch, "extend_prefix_lens_cpu", None),
        field_name="extend_prefix_lens",
    )
    extend_lens = _cpu_int_list(
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        field_name="extend_seq_lens",
    )
    if len(mm_inputs) != len(prefix_lens) or len(mm_inputs) != len(extend_lens):
        raise RuntimeError(
            "Qwen3-Omni multimodal metadata is not aligned with the prefill batch: "
            f"mm_inputs={len(mm_inputs)}, prefix_lens={len(prefix_lens)}, "
            f"extend_lens={len(extend_lens)}"
        )

    source_ranges: list[tuple[torch.Tensor, Any, int, int]] = []
    destination_rows: list[int] = []
    flat_request_start = 0

    for mm_input, prefix_len, extend_len in zip(
        mm_inputs, prefix_lens, extend_lens
    ):
        if mm_input is None:
            flat_request_start += extend_len
            continue

        request_items = getattr(mm_input, "mm_items", ()) or ()
        for item in request_items:
            is_audio = getattr(item, "is_audio", lambda: False)()
            is_precomputed = getattr(
                item, "is_precomputed_embedding", lambda: False
            )()
            if not (is_audio and is_precomputed):
                raise RuntimeError(
                    "Qwen3-Omni standard model.forward only supports "
                    "precomputed audio items; visual/deepstack items use the "
                    "legacy semantic path"
                )

            source = getattr(item, "precomputed_embeddings", None)
            if not isinstance(source, torch.Tensor) or source.ndim != 2:
                raise ValueError(
                    "Qwen3-Omni precomputed audio item must contain a "
                    "[tokens, hidden] tensor"
                )
            if source.shape[1] != target.shape[1]:
                raise ValueError(
                    "Qwen3-Omni audio embedding width does not match text model: "
                    f"audio={source.shape[1]}, text={target.shape[1]}"
                )

            offsets = getattr(item, "offsets", None)
            if not offsets:
                raise ValueError(
                    "Qwen3-Omni precomputed audio item is missing prompt offsets"
                )

            source_cursor = 0
            for span in offsets:
                if len(span) != 2:
                    raise ValueError(
                        f"Qwen3-Omni audio offset must be (start, end), got {span!r}"
                    )
                span_start, span_end = (int(span[0]), int(span[1]))
                if span_start < 0 or span_end < span_start:
                    raise ValueError(
                        f"Qwen3-Omni audio offset is invalid: {span!r}"
                    )

                span_length = span_end - span_start + 1
                source_end = source_cursor + span_length
                if source_end > source.shape[0]:
                    raise ValueError(
                        "Qwen3-Omni audio offsets consume more rows than the "
                        f"precomputed tensor: consumed={source_end}, "
                        f"available={source.shape[0]}"
                    )

                overlap_start = max(span_start, prefix_len)
                overlap_end = min(span_end, prefix_len + extend_len - 1)
                if overlap_start <= overlap_end:
                    local_start = source_cursor + (overlap_start - span_start)
                    local_end = source_cursor + (overlap_end - span_start) + 1
                    source_ranges.append((source, item, local_start, local_end))
                    destination_rows.extend(
                        range(
                            flat_request_start + overlap_start - prefix_len,
                            flat_request_start + overlap_end - prefix_len + 1,
                        )
                    )
                source_cursor = source_end

            if source_cursor != source.shape[0]:
                raise ValueError(
                    "Qwen3-Omni audio offsets do not consume the complete "
                    f"precomputed tensor: consumed={source_cursor}, "
                    f"available={source.shape[0]}"
                )

        flat_request_start += extend_len

    if not source_ranges:
        return target

    first_source, first_item, first_start, _ = source_ranges[0]
    contiguous_single_source = all(
        source is first_source
        and item is first_item
        and start == previous_end
        for (source, item, start, _end), (_, _, _previous_start, previous_end) in zip(
            source_ranges[1:], source_ranges
        )
    )
    if contiguous_single_source:
        source_rows = first_source[first_start : source_ranges[-1][3]]
    else:
        source_rows = torch.cat(
            [source[start:end] for source, _item, start, end in source_ranges],
            dim=0,
        )
    if source_rows.device != target.device or source_rows.dtype != target.dtype:
        source_rows = source_rows.to(
            device=target.device,
            dtype=target.dtype,
            non_blocking=True,
        )
    destination = torch.tensor(
        destination_rows,
        dtype=torch.long,
        device=target.device,
    )
    # Exactly one batch-wide placement keeps eager and graph-owned paths on the
    # same source/chunk contract and avoids per-request or per-modality scatters.
    target.index_copy_(0, destination, source_rows)
    return target


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

    @property
    def thinker(self) -> "Qwen3OmniThinkerForCausalLM":
        # Existing Qwen thinker runner/hook code expects model.thinker.model.
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        get_embedding: bool = False,
        pp_proxy_tensors: Any | None = None,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ):
        del get_embedding
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        model_input_ids = input_ids
        model_input_embeds = input_embeds
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = bool(
            forward_mode is not None
            and getattr(forward_mode, "is_decode", lambda: False)()
        )
        is_target_verify = bool(
            forward_mode is not None
            and getattr(forward_mode, "is_target_verify", lambda: False)()
        )

        if not is_decode and not is_target_verify:
            mm_items = _qwen_mm_items(forward_batch)
            pp_group = getattr(self.model, "pp_group", None)
            is_first_pipeline_rank = pp_group is None or pp_group.is_first_rank

            # A live input_embeds value is the existing legacy Omni hook
            # contract. If it shares a batch with standard audio items, use it
            # as the already-composed text/visual base and place audio into it;
            # pure supported audio arrives with no live value.
            if is_first_pipeline_rank and (mm_items or model_input_embeds is None):
                if input_ids is None:
                    raise RuntimeError(
                        "Qwen3-Omni standard prefill requires input_ids when "
                        "input_embeds is not supplied by the legacy hook"
                    )
                model_input_embeds = _compose_qwen_prefill_embeddings(
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_embedding=self.model.get_input_embeddings(),
                    base_input_embeds=model_input_embeds,
                )
                model_input_ids = None

        hidden_states = self.model(
            input_ids=model_input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=model_input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            input_deepstack_embeds=input_deepstack_embeds,
        )
        logits_input_ids = (
            input_ids
            if input_ids is not None
            else getattr(forward_batch, "input_ids", input_ids)
        )
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

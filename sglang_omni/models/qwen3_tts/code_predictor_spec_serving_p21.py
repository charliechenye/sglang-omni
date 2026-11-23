# SPDX-License-Identifier: Apache-2.0
"""P21: opt-in REAL serving CodePredictor K2 prototype (not production-ready).

Unlike P17/P18/P19 counterfactuals, this function actually writes the codec
codes/embeddings returned to the live speech response. It executes ONE fused
two-row target verifier per K2 round and logically commits the accepted
proposal plus optional bonus or the positive-residual correction.

This is a C1-only integration and latency-feasibility experiment:
- Main Talker remains the genuine serving CUDA Graph path.
- The CodePredictor's native full-chain Graph is replaced with host-driven
  speculative K2 only when an explicit post-health ARM file exists.
- Each accepted length is read on the CPU; no full-chain speculative CUDA
  Graph, native-seeded RNG coupling, C8/C16 claim, or speedup is implied.
- We REQUIRE a complete HTTP natural-stop/quality check before treating
  experimental latency as decision evidence. Never silently fall back.
"""
from __future__ import annotations

import hashlib
import logging
import os

import torch

from .code_predictor_spec_p17_wide import (
    check_wide_backend, make_wide_round, prepare_head_bank,
)

_LOG = logging.getLogger(__name__)
ENV = "SGLANG_OMNI_QTTS_CP_P21_SERVING_K2"
ARM_ENV = "SGLANG_OMNI_QTTS_CP_P21_ARM_FILE"
SOURCE_ENV = "SGLANG_OMNI_QTTS_CP_P21_SOURCE_SHA"


def armed() -> bool:
    requested = os.environ.get(ENV)
    if requested not in (None, "0", "1"):
        raise RuntimeError("P21_SERVING_K2_FLAG_MUST_BE_ZERO_OR_ONE")
    if requested != "1":
        return False
    path = os.environ.get(ARM_ENV)
    if not path or not os.path.isabs(path):
        raise RuntimeError("P21_SERVING_K2_REQUIRES_ABSOLUTE_ARM_PATH")
    if not os.path.isfile(path):
        return False
    source = os.environ.get(SOURCE_ENV)
    if not source or len(source) != 40:
        raise RuntimeError("P21_SERVING_K2_REQUIRES_SOURCE_SHA")
    return True


def _request_seed(native_seed: int, semantic_position: int) -> int:
    # Independent dedicated seed stream, not native SGLang bitwise coupling.
    # Stable across A/B/B/A for the same requested seed and semantic position.
    key = f"p21-private-draft|{native_seed}|{semantic_position}".encode()
    raw = hashlib.sha256(key).digest()
    return int.from_bytes(raw[:8], "little") & ((1 << 63) - 1)


def next_depth(depth: int, committed: int) -> int:
    """Host-side logical K2 offset; the physical second KV row is speculative."""
    if (type(depth) is not int or type(committed) is not int
            or depth not in range(1, 14)
            or committed not in (1, 2)
            or depth + committed > 15):
        raise RuntimeError("P21_INVALID_COMMITTED_LENGTH_OR_DEPTH")
    return depth + committed


@torch.no_grad()
def serving_k2(
    model, layer0_codes: torch.Tensor, talker_hidden: torch.Tensor,
    semantic_positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write an actual 16-codebook C1 frame through seven-ish K2 rounds."""
    model._p21_last_frame_stats = None
    if (
        layer0_codes.shape != (1, 1)
        or layer0_codes.dtype not in (torch.int, torch.long)
        or not layer0_codes.is_cuda
        or talker_hidden.shape[0:2] != (1, 1)
        or talker_hidden.ndim != 3
        or not model.sub_has_sampled_rows or model.sub_has_argmax_rows
        or model.sub_batch_size != 1
        or model.config.num_code_groups != 16
        or torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError("P21_REQUIRES_SAMPLED_CUDA_C1_16GROUP_NOT_CAPTURING")
    if any(os.environ.get(env) for env in (
        "SGLANG_OMNI_QTTS_CP_P18_DIR",
        "SGLANG_OMNI_QTTS_CP_P19_COST_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    )):
        raise RuntimeError("P21_REQUIRES_NO_P18_P19_OR_LEGACY_K2_PROBES")
    if os.environ.get("SGLANG_OMNI_QTTS_CP_SPEC_K", "0") != "0":
        raise RuntimeError("P21_CONFLICT_WITH_LEGACY_SPEC_K")
    dev = layer0_codes.device
    if "H200" not in torch.cuda.get_device_name(dev):
        raise RuntimeError("P21_H200_REQUIRED")
    if len(model.code_predictor.model.layers) != 5:
        raise RuntimeError("P21_REQUIRES_NATIVE_FIVE_LAYER_PREDICTOR")

    position = model.normalize_semantic_positions(
        semantic_positions, batch_size=1, seq_len=1, device=dev
    )
    seed_positions = model.sub_seed_positions(position[:, 0])
    native_seed = int(model.sub_sampling_seed_tensor[0].item())
    semantic_pos = int(position[0, 0].item())
    generator = torch.Generator(device=dev)
    generator.manual_seed(_request_seed(native_seed, semantic_pos))
    temp = float(model.sub_temperature_tensor[0].item())
    k = int(model.sub_top_k_tensor[0].item())
    p = float(model.sub_top_p_tensor[0].item())
    if (abs(temp - 0.9) > 1e-5 or k != 50 or abs(p - 1.0) > 1e-6):
        raise RuntimeError("P21_SAMPLING_SIGNATURE_ONLY_T09_TOPK50_TOPP1")

    banks = getattr(model, "_p21_prepacked_head_banks", None)
    if banks is None:
        banks = {}
        for depth in range(1, 14):
            check_wide_backend(model, width=2, cache_len=depth + 1)
            banks[depth] = prepare_head_bank(
                model, depth=depth, width=2
            )
        model._p21_prepacked_head_banks = banks

    model_dtype = model.predictor_k_cache.dtype
    codec_tables = model.code_predictor.model.codec_embedding.weight
    projected_tables = model.predictor_projected_embeddings
    embeddings = model.output_embeds[:1]
    codes = model.output_codes[:1]
    embeddings.zero_()
    codes[:, 0].copy_(layer0_codes[:, 0])
    code0_embedding = model.get_input_embeddings()(layer0_codes).to(
        dtype=model_dtype
    )
    embeddings.add_(code0_embedding[:, 0, :])

    pair = model.code_predictor.project_input(torch.cat(
        (talker_hidden.to(dtype=model_dtype), code0_embedding), dim=1,
    ))
    old_hidden = model.predictor_forward_tokens(
        token_embeds=pair, batch_size=1, cache_len=0,
    )[:, 1:, :].contiguous().clone()
    first_logits, _ = model.code_predictor.lm_head[0](old_hidden)
    first_embed, first_projected = model.sample_subtalker_embedding(
        first_logits[:, -1, :], codec_tables[0],
        embeddings, codes[:, 1],
        projected_tables[0] if projected_tables is not None else None,
        model.predictor_embedding_buffer[:1],
        model.predictor_projected_buffer[:1]
        if projected_tables is not None else None,
        seed_positions[0],
    )
    if projected_tables is not None:
        last_embedding = first_projected.unsqueeze(1).clone()
    else:
        last_embedding = model.code_predictor.project_input(
            first_embed.unsqueeze(1)
        ).clone()

    offsets = torch.arange(2, device=dev, dtype=torch.long)
    depth, cache_len = 1, 2
    rounds = accepted = rejected = 0

    while depth < 15:
        if depth == 14:
            # No two-row bonus head at the terminal codebook. Original
            # native final AR step is the ONLY valid tail.
            hidden = model.predictor_forward_tokens(
                token_embeds=last_embedding,
                batch_size=1, cache_len=cache_len,
            )
            logits, _ = model.code_predictor.lm_head[14](hidden)
            model.sample_subtalker_embedding(
                logits[:, -1, :], codec_tables[14],
                embeddings, codes[:, 15],
                None, model.predictor_embedding_buffer[:1],
                None, seed_positions[14],
            )
            depth += 1
            continue

        check_wide_backend(model, width=2, cache_len=cache_len)
        # P17 K2 consumes 4 uniforms for one q/p accept-residual
        # decision PLUS one separate bonus draw at index 4.
        # Using four silently leaves the bonus selector with an empty
        # [4:5] slice. Exact shape is an ABI invariant.
        uniforms = torch.rand(
            (5,), device=dev, dtype=torch.float32, generator=generator
        ).clamp(1e-4, 1.0 - 1e-4)
        step = make_wide_round(
            model, depth=depth, width=2,
            cache_len_gpu=torch.tensor(
                cache_len, device=dev, dtype=torch.long,
            ),
            old_hidden=old_hidden,
            last_embedding=last_embedding,
            accumulated=embeddings,
            uniforms=uniforms,
            head_bank=banks[depth],
            offset_ids=offsets,
            temperature=temp, top_p=p,
        )
        committed = int(step["committed_count"].item())
        advanced_depth = next_depth(depth, committed)
        candidate_codes = step["committed_codes"][:committed]
        if candidate_codes.shape != (committed,):
            raise RuntimeError("P21_INVALID_COMMITTED_CODES_SHAPE")
        # Device-side assert for invalid -1 code; do not allow unknown
        #/unaccepted speculative suffix into real returned output.
        if bool((candidate_codes < 0).any().item()):
            raise RuntimeError("P21_UNCOMMITTED_SPECULATIVE_CODE")
        codes[:, depth + 1: depth + 1 + committed].copy_(
            candidate_codes.view(1, committed)
        )
        embeddings.copy_(step["next_sum"])
        old_hidden = step["next_hidden"].clone()
        last_embedding = step["next_embedding"].clone()
        rounds += 1
        accepted += committed == 2
        rejected += committed == 1
        cache_len += committed
        depth = advanced_depth
        if cache_len != depth + 1:
            raise RuntimeError("P21_CACHE_LEN_DEPTH_INVARIANT_FAILED")

    if depth != 15 or not torch.equal(codes[:, 0], layer0_codes[:, 0]):
        raise RuntimeError("P21_TERMINAL_FRAME_CODE_OR_DEPTH_INVALID")
    model._p21_last_frame_stats = {
        "rounds": rounds,
        "accepted_drafts": accepted,
        "rejected_drafts": rejected,
        "committed_residual": 15,
        "semantic_position": semantic_pos,
    }
    return codes, embeddings

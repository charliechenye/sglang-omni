# SPDX-License-Identifier: Apache-2.0
"""Research-only lossless-formula CodePredictor speculation (K=2, single row).

Based on NVIDIA TensorRT-Edge-LLM's head-reuse proposal, causal verification,
and rejection/residual sampling. This PyTorch implementation is intentionally
slow and synchronization-heavy: correctness before performance.

Gate: SGLANG_OMNI_QTTS_CP_SPEC_K=2 and a post-/health ARM file.
NOT an optimized kernel, CUDA-Graph path, or upstream speedup claim.
"""
from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Callable

from . import code_predictor_spec_diagnostics as diag

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

logger = logging.getLogger(__name__)


def offset_causal_mask(cache_len: int, tokens: int, device: torch.device) -> torch.Tensor:
    """Causal query/key visibility with an existing committed KV prefix.

    For query i within a new chunk, allow all old keys and new keys <= i.
    Default SDPA is_causal=True is NOT appropriate when Q_len < KV_len:
    its upper-left alignment would incorrectly hide the committed prefix.
    """
    if cache_len < 0 or tokens < 1:
        raise ValueError("invalid cached prefix or chunk width")
    q_positions = cache_len + torch.arange(tokens, device=device)
    key_positions = torch.arange(cache_len + tokens, device=device)
    return key_positions[None, :] <= q_positions[:, None]


def sampling_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Reference of target's top-k-before-top-p categorical distribution.

    BF16 logits are upcast to FP32. Ties and fused-CUDA rounding have not
    been proved bit-identical to the live SGLang sampler.
    """
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError("expected [batch, vocab] logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be integer")
    if not torch.isfinite(logits).all().item():
        raise ValueError("invalid nonfinite logits")
    vocab = logits.shape[-1]
    k = vocab if top_k <= 0 or top_k >= vocab else top_k
    scaled = logits.float() / float(temperature)
    top_values, top_indices = torch.topk(scaled, k, dim=-1, sorted=True)
    kept = torch.softmax(top_values, dim=-1)
    if top_p < 1.0:
        # SGLang removes tokens with cumulative mass *before* the token >= p.
        remove = torch.cumsum(kept, dim=-1) - kept >= top_p
        remove[:, 0] = False
        kept = kept.masked_fill(remove, 0.0)
        kept = kept / kept.sum(dim=-1, keepdim=True)
    full = torch.zeros_like(scaled)
    full.scatter_(dim=-1, index=top_indices, src=kept)
    return full


def overlap_probability(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact theoretical mean acceptance of one standard speculative draft."""
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("p and q must have identical [batch, vocab] shapes")
    if (p < 0).any().item() or (q < 0).any().item():
        raise ValueError("negative probability")
    if not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5):
        raise ValueError("p is not normalized")
    if not torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-5):
        raise ValueError("q is not normalized")
    return torch.minimum(p, q).sum(-1)


def accept_or_residual(
    proposal: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, bool]:
    """Return a committed token from p, and whether q's proposal survived.

    q-generated proposal accepted with min(1,p(x)/q(x)). On rejection,
    sample normalized (p-q)+. No approximation/fallback to plain p allowed.
    """
    if p.shape != q.shape or p.shape[0] != 1:
        raise ValueError("P0 supports one row of normalized distributions")
    token = int(proposal.item())
    p_x = float(p[0, token].item())
    q_x = float(q[0, token].item())
    if q_x <= 0 or not math.isfinite(p_x) or not math.isfinite(q_x):
        raise ValueError("proposal not in q support")
    accept_prob = min(1.0, p_x / q_x)
    draw = float(torch.rand((), device=p.device, generator=generator).item())
    if draw < accept_prob:
        return proposal.reshape(1), True
    residual = torch.clamp(p - q, min=0.0)
    residual_mass = float(residual.sum().item())
    if residual_mass <= 1e-9:
        raise RuntimeError("rejected proposal but no residual probability mass")
    correction = torch.multinomial(residual / residual_mass, 1, generator=generator)
    return correction.reshape(1), False


def embed_residual_code(
    *,
    depth: int,
    code: torch.Tensor,
    codec_tables: torch.Tensor,
    projected_tables: torch.Tensor | None,
    project_input: Callable[[torch.Tensor], torch.Tensor],
    accumulation_dtype: torch.dtype,
    predictor_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read residual code embedding and (if needed) its Predictor projection.

    For N residual codebook depths, Omni deliberately materializes only N-1
    *projected* tables: the last residual code contributes to codec summation,
    but does not feed another Predictor step in the original AR decoder.

    The experimental K=2 verifier can put that final code into a draft suffix
    even though it will never consume its hidden state. Produce its projection
    directly from the ordinary embedding instead of indexing a nonexistent
    projected_tables[N-1] entry.
    """
    if codec_tables.ndim != 3:
        raise ValueError("expected [residual_depths, vocab, hidden] codec tables")
    total_depths = codec_tables.shape[0]
    if not 1 <= depth <= total_depths:
        raise ValueError(
            f"invalid residual depth {depth}, expected 1..{total_depths}"
        )
    if code.ndim != 1 or code.shape[0] != 1 or code.dtype != torch.long:
        raise ValueError("K2 P0 expects a single int64 code ID")
    if projected_tables is not None and (
        projected_tables.ndim != 3
        or projected_tables.shape[0] != total_depths - 1
    ):
        raise ValueError(
            "Omni projected tables must have exactly one fewer depth than "
            f"codec tables: {total_depths - 1} projected vs {total_depths} residual"
        )
    raw = F.embedding(code, codec_tables[depth - 1]).to(dtype=accumulation_dtype)
    if projected_tables is not None and depth <= projected_tables.shape[0]:
        # Existing production fast path; no projection GEMM.
        projected = F.embedding(
            code, projected_tables[depth - 1]
        ).unsqueeze(1)
    else:
        # Last depth (or models without preprojected embeddings):
        # never index projected_tables[total_depths-1].
        projected = project_input(raw.unsqueeze(1).to(dtype=predictor_dtype))
    return raw, projected.contiguous()


def speculative_k2(
    model: "Qwen3TTSTalker",
    layer0_codes: torch.Tensor,
    talker_hidden: torch.Tensor,
    semantic_positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real K=2 proposal->causal verify->accept/residual->bonus loop.

    Restricted to a single sampled CUDA row and 16 codec groups. Uses existing
    on-policy target weights, KV cache, projection and first-token sampler.
    This debug version synchronizes every round for acceptance decisions and
    logs counts; it deliberately does not promise latency gains.
    """
    if (
        layer0_codes.shape != (1, 1)
        or talker_hidden.ndim != 3
        or talker_hidden.shape[:2] != (1, 1)
        or layer0_codes.device.type != "cuda"
        or not model.sub_has_sampled_rows
        or model.sub_has_argmax_rows
        or model.sub_batch_size != 1
        or model.config.num_code_groups != 16
    ):
        raise RuntimeError(
            "CP speculative K2 P0 requires C1 sampled CUDA, one frame per call, "
            "16 RVQ groups and no mixed argmax rows"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CP speculative P0 must not be CUDA-graph captured")

    dtype = model.predictor_k_cache.dtype
    pos = model.normalize_semantic_positions(
        semantic_positions, batch_size=1, seq_len=1, device=layer0_codes.device
    )
    seed_positions = model.sub_seed_positions(pos[:, 0])
    generator = torch.Generator(device=layer0_codes.device)
    source_seed = int(model.sub_sampling_seed_tensor[0].item())
    position_seed = int(pos[0, 0].item())
    generator.manual_seed(
        (source_seed ^ (position_seed * 0x9E3779B185EBCA87) ^ 0x4659505A65) & ((1 << 63) - 1)
    )
    t = float(model.sub_temperature_tensor[0].item())
    k = int(model.sub_top_k_tensor[0].item())
    p_cut = float(model.sub_top_p_tensor[0].item())

    # Source-locked opt-in diagnostics. A successful HTTP 200 never silently
    # becomes a parity or sampled-distribution proof.
    frame_index = int(getattr(model, "_qt_cp_spec_frames", 0)) + 1
    diag_root, do_parity = diag.frame_enabled(frame_index)
    depth_events: list[dict] | None = [] if diag_root is not None else None
    parity_checks: list[dict] = []

    codec_tables = model.code_predictor.model.codec_embedding.weight
    projected_tables = model.predictor_projected_embeddings
    pos_codes = model.output_codes[:1]
    pos_summed = model.output_embeds[:1]
    pos_summed.zero_()
    pos_codes[:, 0].copy_(layer0_codes[:, 0])
    layer0_embedding = model.get_input_embeddings()(layer0_codes).to(dtype=dtype)
    pos_summed.add_(layer0_embedding[:, 0, :])

    pair = model.code_predictor.project_input(
        torch.cat((talker_hidden.to(dtype=dtype), layer0_embedding), dim=1)
    )
    h0 = model.predictor_forward_tokens(
        token_embeds=pair, batch_size=1, cache_len=0
    )[:, 1:, :].contiguous()
    logits1, _ = model.code_predictor.lm_head[0](h0)
    next_code_embedding, projected_code_embedding = model.sample_subtalker_embedding(
        logits1[:, -1, :],
        codec_tables[0],
        pos_summed,
        pos_codes[:, 1],
        projected_tables[0] if projected_tables is not None else None,
        getattr(model, "predictor_embedding_buffer", None)[:1]
        if getattr(model, "predictor_embedding_buffer", None) is not None else None,
        model.predictor_projected_buffer[:1] if projected_tables is not None else None,
        seed_positions[0],
    )
    if projected_tables is not None:
        last_embedding = projected_code_embedding.unsqueeze(1).clone()
    else:
        last_embedding = model.code_predictor.project_input(
            next_code_embedding.unsqueeze(1)
        ).clone()

    def embedding_for_code(depth: int, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return embed_residual_code(
            depth=depth,
            code=code,
            codec_tables=codec_tables,
            projected_tables=projected_tables,
            project_input=model.code_predictor.project_input,
            accumulation_dtype=pos_summed.dtype,
            predictor_dtype=dtype,
        )

    def commit(depth: int, code: torch.Tensor) -> torch.Tensor:
        raw, projected = embedding_for_code(depth, code)
        pos_codes[:, depth].copy_(code)
        pos_summed.add_(raw)
        return projected

    depth = 1  # last committed depth; code_1 already written above
    cache_len = 2
    rounds = 0
    proposed = 0
    accepted = 0
    bonus = 0
    rejected = 0
    while depth < 15:
        rounds += 1
        # q for code_(depth+1) is head[depth] on the *old* hidden,
        # before consuming the most recent committed code.
        next_head = model.code_predictor.lm_head[depth]
        logits_draft, _ = next_head(h0)
        q = sampling_probs(logits_draft[:, -1, :], temperature=t, top_k=k, top_p=p_cut)
        proposal = torch.multinomial(q, 1, generator=generator).reshape(1)
        proposed += 1
        _, proposed_embedding = embedding_for_code(depth + 1, proposal)

        # Snapshot the committed KV prefix BEFORE packed verification when
        # P1 audit is armed. No output outside the append may be modified.
        if do_parity:
            prefix_k = model.predictor_k_cache[:, :1, :cache_len].clone()
            prefix_v = model.predictor_v_cache[:, :1, :cache_len].clone()
        # Run the true target once for both the previously committed code and
        # the proposed future code. Position-aware causal masking is essential.
        verified = model.predictor_forward_tokens(
            token_embeds=torch.cat((last_embedding, proposed_embedding), dim=1),
            batch_size=1,
            cache_len=cache_len,
        )
        h_after_committed = verified[:, 0:1, :].clone()
        h_after_proposal = verified[:, 1:2, :].clone()
        logits_target, _ = next_head(h_after_committed)
        p = sampling_probs(logits_target[:, -1, :], temperature=t, top_k=k, top_p=p_cut)

        if do_parity:
            # Pure oracle replay, no new sampling. It compares the live
            # checkpoint, same prefix and both KV slots, then restores packed
            # cache in a finally block before continuing target generation.
            parity = diag.check_packed_verifier(
                model,
                cache_len=cache_len,
                depth=depth,
                last_embedding=last_embedding,
                proposed_embedding=proposed_embedding,
                packed_hidden=verified,
                prefix_k_before=prefix_k,
                prefix_v_before=prefix_v,
            )
            parity_checks.append(parity)
            if not parity["pass"]:
                assert diag_root is not None
                diag.write_trace(
                    diag_root, frame_index,
                    {
                        "frame": frame_index,
                        "status": "P1_REAL_PACKED_TARGET_PARITY_FAIL",
                        "failed_parity": parity,
                        "completed_depth_events": depth_events,
                    }, failed=True,
                )
                diag.assert_diagnostic_parity(parity)

        corrected_or_accepted, survived = accept_or_residual(
            proposal, p, q, generator
        )
        if depth_events is not None:
            proposal_depth = depth + 1
            p_x = p.gather(1, proposal.view(1, 1))
            q_x = q.gather(1, proposal.view(1, 1))
            depth_events.append({
                "proposal_depth": proposal_depth,
                "expected_overlap": float(overlap_probability(p, q)[0].item()),
                "accepted": bool(survived),
                "proposal_code": int(proposal[0].item()),
                "committed_code": int(corrected_or_accepted[0].item()),
                "p_proposal": float(p_x[0, 0].item()),
                "q_proposal": float(q_x[0, 0].item()),
                "target_mass_on_q_support": float(p.masked_select(q > 0).sum().item()),
                "proposal_mass_on_p_support": float(q.masked_select(p > 0).sum().item()),
                "p_support": int((p > 0).sum().item()),
                "q_support": int((q > 0).sum().item()),
                "bonus_depth": proposal_depth + 1
                if survived and proposal_depth < 15 else None,
            })
        depth += 1
        commit_embed = commit(depth, corrected_or_accepted)

        if survived:
            accepted += 1
            cache_len += 2
            h0 = h_after_proposal
            if depth == 15:
                break
            # Bonus from the verified trailing hidden, no additional target call.
            bonus_head = model.code_predictor.lm_head[depth]
            bonus_logits, _ = bonus_head(h0)
            bonus_prob = sampling_probs(
                bonus_logits[:, -1, :], temperature=t, top_k=k, top_p=p_cut
            )
            bonus_code = torch.multinomial(bonus_prob, 1, generator=generator).reshape(1)
            depth += 1
            bonus += 1
            last_embedding = commit(depth, bonus_code)
        else:
            rejected += 1
            cache_len += 1
            h0 = h_after_committed
            last_embedding = commit_embed
            # The verifier wrote a KV suffix for the rejected draft. Its slot
            # lies beyond cache_len and is overwritten on the next verification.

    count = frame_index
    model._qt_cp_spec_frames = count
    if depth_events is not None:
        assert diag_root is not None
        diag.validate_depth_events(depth_events)
        embedding_check = diag.check_embedding_sum(
            model, layer0_embedding, pos_codes, pos_summed,
        )
        payload = {
            "frame": count,
            "source_sha": os.environ.get(
                "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
            ),
            "semantic_position": position_seed,
            "sampling": {"temperature": t, "top_k": k, "top_p": p_cut},
            "total_residual_codes": 15,
            "proposed": proposed,
            "accepted": accepted,
            "rejected": rejected,
            "bonus": bonus,
            "rounds": rounds,
            "output_codes": pos_codes[0].detach().cpu().tolist(),
            "embedding_sum": embedding_check,
            "per_depth": depth_events,
            "target_parity": {
                "executed": bool(parity_checks),
                "checks": parity_checks,
                "pass": all(x["pass"] for x in parity_checks),
            },
        }
        if not embedding_check["close"]:
            payload["status"] = "P1_EMBEDDING_SUM_PARITY_FAIL"
            diag.write_trace(diag_root, count, payload, failed=True)
            raise RuntimeError(
                f"CP_SPEC_K2_EMBEDDING_SUM_PARITY_FAIL frame={count}"
            )
        payload["status"] = "P1_P2_K2_TRACE_CAPTURED_NOT_SAMPLER_OR_AUDIO_PARITY"
        diag.write_trace(diag_root, count, payload)
    logger.info(
        "QTTS_CP_SPEC_K2 frame=%d rounds=%d proposed=%d accepted=%d "
        "bonus=%d rejected=%d committed_residual=15",
        count, rounds, proposed, accepted, bonus, rejected,
    )
    return pos_codes.unsqueeze(-1), pos_summed.unsqueeze(1)

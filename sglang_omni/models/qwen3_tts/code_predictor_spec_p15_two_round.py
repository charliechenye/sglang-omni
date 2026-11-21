# SPDX-License-Identifier: Apache-2.0
"""P15 experimental two-round, acceptance-dependent, single CUDA Graph.

P14 proved one causal q -> verifier -> p -> accept/residual round.
P15 tests whether Round 2 consumes Round 1 GPU-selected head, embedding,
hidden, code sum and KV offset. The real fused five-layer Predictor is used.

The first round starts at the real on-policy depth=9, cache_len=10
(proposal depth=10). Round 2 chooses depth=10 or 11 using a GPU predicate.
P10A's device-offset fused verifier runs both packed two-token forwards.
Bonus heads are placed INSIDE GPU conditional branches, not unconditionally
computed and masked. Source-level conditionality is NOT GPU-kernel trace
proof; capture and parity alone cannot establish actual kernel skip.

STRICT LIMITS: no complete 14-depth decoder, no native seeded RNG parity,
no HTTP throughput/quality claim, no production selection.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import statistics
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch._higher_order_ops.cond import cond_op
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    CUDAGraphCaptureControlFlowOpDispatchMode,
)

from .code_predictor_spec import sampling_probs
from .code_predictor_spec_p10a_dynamic_kv import dynamic_fused_predictor
from .code_predictor_spec_p12a_selector import (
    fast_selector, _full_dist,
)
from .code_predictor_spec_p14_causal_round import (
    K, TRIPLES, single_support, gpu_accept,
)
from .code_predictor_spec_p9b import cpu_reference
from .code_predictor_spec_p7_dispatch import compare

ENV = "SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR"
SCHEMA = "qwen3tts-cpspec-p15-two-round-device-branch-real-fused-v1"
FIRST_DEPTH = 9
FIRST_CACHE_LEN = 10
MAX_CACHE_SLOTS = 16
ROUND_FIELDS = (
    "draft", "committed", "accepted", "bonus",
    "q_logits", "p_logits", "q_ids", "q_probs", "p_ids", "p_probs",
    "next_hidden", "next_embedding", "next_sum",
    "next_cache_len", "next_depth", "bonus_executed",
)
BRANCHES = ("AA", "AR", "RA", "RR")
CASES = tuple(itertools.product(TRIPLES, repeat=2))
SUCCESS = (
    "P15_REAL_TWO_ROUND_GPU_CAUSAL_GRAPH_PARITY_ALL4_"
    "NOT_SEEDED_RNG_OR_14DEPTH_OR_E2E"
)
INCOMPLETE = "P15_REAL_TWO_ROUND_PARITY_PASS_BRANCH_COVERAGE_INCOMPLETE"


def _project(model, index: int, code: torch.Tensor,
             accum_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather codec sum AND Predictor-projected embedding for codebook index."""
    codec = model.code_predictor.model.codec_embedding.weight
    raw = F.embedding(code, codec[index]).to(dtype=accum_dtype)
    projected = model.predictor_projected_embeddings
    if projected is None or index >= projected.shape[0]:
        embedding = model.code_predictor.project_input(
            raw.unsqueeze(1).to(dtype=model.predictor_k_cache.dtype)
        ).contiguous()
    else:
        embedding = F.embedding(code, projected[index]).unsqueeze(1).contiguous()
    return raw, embedding


def gpu_round(
    model, *, depth: int, cache_len_gpu: torch.Tensor,
    hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated: torch.Tensor, uniforms: torch.Tensor,
    indices_gpu: torch.Tensor, temperature: float, top_p: float,
) -> tuple[torch.Tensor, ...]:
    """One true causal round; no host-read decisions; conditional bonus head."""
    head = model.code_predictor.lm_head[depth]
    bonus_head = model.code_predictor.lm_head[depth + 1]
    q_logits = head(hidden)[0][:, -1, :]
    q_ids, _, q_probs, draft = fast_selector(
        q_logits, uniforms[0:1], temperature=temperature, top_p=top_p,
    )
    _, draft_embedding = _project(model, depth, draft, accumulated.dtype)
    packed = torch.cat((last_embedding, draft_embedding), dim=1)
    verified = dynamic_fused_predictor(
        model, pair=packed, cache_len_gpu=cache_len_gpu,
        indices_gpu=indices_gpu,
    )
    before = verified[:, :1, :].clone()
    after = verified[:, 1:2, :].clone()
    p_logits = head(before)[0][:, -1, :]
    p_ids, p_probs = single_support(
        p_logits, temperature=temperature, top_p=top_p, fast=True,
    )
    decision = gpu_accept(
        p_ids, p_probs, q_ids, q_probs, uniforms, draft,
    )
    committed, accepted = decision["committed"], decision["accepted"]
    committed_raw, committed_proj = _project(
        model, depth, committed, accumulated.dtype,
    )

    def bonus_yes(after_hidden, committed_code, raw, projected):
        bonus_logits = bonus_head(after_hidden)[0][:, -1, :]
        _, _, _, bonus_code = fast_selector(
            bonus_logits, uniforms[3:4],
            temperature=temperature, top_p=top_p,
        )
        bonus_raw, bonus_proj = _project(
            model, depth + 1, bonus_code, accumulated.dtype,
        )
        return (
            bonus_code, bonus_raw, bonus_proj,
            torch.ones_like(committed_code, dtype=torch.int32),
        )

    def bonus_no(after_hidden, committed_code, raw, projected):
        return (
            torch.zeros_like(committed_code),
            torch.zeros_like(raw),
            torch.zeros_like(projected),
            torch.zeros_like(committed_code, dtype=torch.int32),
        )

    bonus, bonus_raw, bonus_projected, bonus_executed = cond_op(
        accepted.reshape(()), bonus_yes, bonus_no,
        (after, committed, committed_raw, committed_proj),
    )
    next_hidden = torch.where(accepted.reshape(1, 1, 1), after, before)
    next_embedding = torch.where(
        accepted.reshape(1, 1, 1), bonus_projected, committed_proj,
    )
    next_sum = accumulated + committed_raw + bonus_raw
    step = 1 + accepted.to(dtype=torch.long).reshape(())
    next_cache_len = cache_len_gpu + step
    next_depth = torch.full_like(cache_len_gpu, depth) + step
    return (
        draft, committed, accepted, bonus,
        q_logits, p_logits, q_ids, q_probs, p_ids, p_probs,
        next_hidden, next_embedding, next_sum,
        next_cache_len, next_depth, bonus_executed,
    )


def two_gpu_rounds(
    model, *, first_depth: int, cache_len_gpu: torch.Tensor,
    hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated: torch.Tensor, uniforms: torch.Tensor,
    indices_gpu: torch.Tensor, temperature: float, top_p: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Round 2 head is GPU-selected by actual Round 1 acceptance."""
    first = gpu_round(
        model, depth=first_depth, cache_len_gpu=cache_len_gpu,
        hidden=hidden, last_embedding=last_embedding,
        accumulated=accumulated, uniforms=uniforms[0],
        indices_gpu=indices_gpu, temperature=temperature, top_p=top_p,
    )

    def accepted_branch(h, emb, summed, offset, u):
        return gpu_round(
            model, depth=first_depth + 2, cache_len_gpu=offset,
            hidden=h, last_embedding=emb, accumulated=summed,
            uniforms=u, indices_gpu=indices_gpu,
            temperature=temperature, top_p=top_p,
        )

    def rejected_branch(h, emb, summed, offset, u):
        return gpu_round(
            model, depth=first_depth + 1, cache_len_gpu=offset,
            hidden=h, last_embedding=emb, accumulated=summed,
            uniforms=u, indices_gpu=indices_gpu,
            temperature=temperature, top_p=top_p,
        )

    second = cond_op(
        first[ROUND_FIELDS.index("accepted")].reshape(()),
        accepted_branch, rejected_branch,
        (
            first[ROUND_FIELDS.index("next_hidden")],
            first[ROUND_FIELDS.index("next_embedding")],
            first[ROUND_FIELDS.index("next_sum")],
            first[ROUND_FIELDS.index("next_cache_len")],
            uniforms[1],
        ),
    )
    return first, second


@torch.no_grad()
def eager_oracle_round(
    model, *, depth: int, cache_len: int,
    hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated: torch.Tensor, uniforms: tuple[float, ...],
    temperature: float, top_p: float,
) -> dict:
    """Independent host-controlled static-offset reference OUTSIDE graph."""
    head = model.code_predictor.lm_head[depth]
    q_logits = head(hidden)[0][:, -1, :].clone()
    q_dense = sampling_probs(
        q_logits, temperature=temperature, top_k=K, top_p=top_p,
    )
    draft_id = int(torch.searchsorted(
        q_dense[0].float().cpu().cumsum(-1).contiguous(),
        torch.tensor(uniforms[0], dtype=torch.float32),
    ).clamp(max=q_dense.shape[-1] - 1).item())
    draft = torch.tensor([draft_id], device=hidden.device, dtype=torch.long)
    _, draft_emb = _project(model, depth, draft, accumulated.dtype)
    verified = model.predictor_forward_tokens(
        token_embeds=torch.cat((last_embedding, draft_emb), dim=1),
        batch_size=1, cache_len=cache_len,
    )
    before, after = verified[:, :1, :].clone(), verified[:, 1:2, :].clone()
    p_logits = head(before)[0][:, -1, :].clone()
    p_dense = sampling_probs(
        p_logits, temperature=temperature, top_k=K, top_p=top_p,
    )
    decision = cpu_reference(p_dense, q_dense, uniforms[:3])
    if draft_id != decision["draft"]:
        raise RuntimeError("P15_CPU_DRAFT_REFERENCE_INCONSISTENT")
    committed = torch.tensor(
        [decision["committed"]], device=hidden.device, dtype=torch.long,
    )
    accepted = decision["accepted"]
    raw, projected = _project(model, depth, committed, accumulated.dtype)
    if accepted:
        bonus_logits = model.code_predictor.lm_head[depth + 1](
            after
        )[0][:, -1, :]
        bonus_dense = sampling_probs(
            bonus_logits, temperature=temperature, top_k=K, top_p=top_p,
        )
        bonus_id = int(torch.searchsorted(
            bonus_dense[0].float().cpu().cumsum(-1).contiguous(),
            torch.tensor(uniforms[3], dtype=torch.float32),
        ).clamp(max=bonus_dense.shape[-1] - 1).item())
        bonus = torch.tensor([bonus_id], dtype=torch.long, device=hidden.device)
        bonus_raw, bonus_projected = _project(
            model, depth + 1, bonus, accumulated.dtype,
        )
        new_embedding = bonus_projected
        new_sum = accumulated + raw + bonus_raw
    else:
        bonus_id = 0
        new_embedding = projected
        new_sum = accumulated + raw
    next_cache_len = cache_len + 1 + int(accepted)
    next_depth = depth + 1 + int(accepted)
    return {
        "draft": draft_id,
        "committed": int(committed[0].item()),
        "accepted": bool(accepted),
        "bonus": bonus_id,
        "bonus_executed": int(accepted),
        "q_dense": q_dense.clone(),
        "p_dense": p_dense.clone(),
        "q_logits": q_logits,
        "p_logits": p_logits,
        "next_hidden": (after if accepted else before).clone(),
        "next_embedding": new_embedding.clone(),
        "next_sum": new_sum.clone(),
        "next_cache_len": next_cache_len,
        "next_depth": next_depth,
    }


def _check_case(
    graph_rounds: tuple[dict, dict], oracle_rounds: tuple[dict, dict],
    *, graph_k: torch.Tensor, graph_v: torch.Tensor,
    oracle_k: torch.Tensor, oracle_v: torch.Tensor,
    prefix_k: torch.Tensor, prefix_v: torch.Tensor,
    temperature: float, top_p: float,
) -> str:
    for number, (actual, expected) in enumerate(
        zip(graph_rounds, oracle_rounds), start=1
    ):
        for name in (
            "draft", "committed", "accepted", "bonus",
            "bonus_executed", "next_depth", "next_cache_len",
        ):
            value = actual[name].reshape(-1)[0].item()
            if value != expected[name]:
                raise RuntimeError(
                    f"P15_ROUND{number}_{name.upper()}_PARITY_FAIL "
                    f"gpu={value} oracle={expected[name]}"
                )
        for name in ("next_hidden", "next_embedding", "next_sum"):
            if not compare(actual[name], expected[name])["close"]:
                raise RuntimeError(f"P15_ROUND{number}_{name.upper()}_PARITY_FAIL")
        for letter in ("q", "p"):
            if not compare(actual[f"{letter}_logits"],
                           expected[f"{letter}_logits"])["close"]:
                raise RuntimeError(
                    f"P15_ROUND{number}_{letter.upper()}_LOGITS_PARITY_FAIL"
                )
            distribution = _full_dist(
                actual[f"{letter}_ids"], actual[f"{letter}_probs"],
            )
            if not torch.allclose(
                distribution, expected[f"{letter}_dense"],
                atol=1e-6, rtol=1e-6,
            ):
                deviation = (
                    distribution - expected[f"{letter}_dense"]
                ).abs().max().item()
                raise RuntimeError(
                    f"P15_ROUND{number}_{letter.upper()}_DENSE_PARITY_FAIL "
                    f"max_abs={deviation}"
                )
    for kind, got, expected, prefix in (
        ("K", graph_k, oracle_k, prefix_k),
        ("V", graph_v, oracle_v, prefix_v),
    ):
        if not compare(got, expected)["close"]:
            raise RuntimeError(f"P15_FULL_PHYSICAL_{kind}_KV_PARITY_FAIL")
        if not torch.equal(got[:, :, :FIRST_CACHE_LEN],
                           prefix[:, :, :FIRST_CACHE_LEN]):
            raise RuntimeError(f"P15_COMMITTED_PREFIX_{kind}_MUTATED")
    if graph_rounds[1]["next_depth"].item() != (
        oracle_rounds[0]["next_depth"] + 1 +
        int(oracle_rounds[1]["accepted"])
    ):
        raise RuntimeError("P15_ROUND2_DID_NOT_CONSUME_ROUND1_DEPTH")
    return (
        ("A" if oracle_rounds[0]["accepted"] else "R") +
        ("A" if oracle_rounds[1]["accepted"] else "R")
    )


@torch.no_grad()
def capture_two_rounds(
    model, *, depth: int, cache_len: int, draft_hidden: torch.Tensor,
    last_embedding: torch.Tensor, accumulated_before: torch.Tensor,
    temperature: float, top_k: int, top_p: float, repeats: int = 16,
) -> dict:
    """Capture real causal two-round graph and compare full independent oracle."""
    dev = last_embedding.device
    fused = model.predictor_fused_layers
    if (
        depth != FIRST_DEPTH or cache_len != FIRST_CACHE_LEN
        or top_k != K or not 0 < top_p <= 1 or temperature <= 0
        or not 8 <= repeats <= 128
        or model.predictor_cutedsl_layers is not None
        or fused is None or not fused.covers(2)
        or len(model.code_predictor.model.layers) != 5
        or model.predictor_k_cache.shape[2] < MAX_CACHE_SLOTS
        or model.predictor_v_cache.shape[2] < MAX_CACHE_SLOTS
        or draft_hidden.shape != last_embedding.shape
        or draft_hidden.shape[:2] != (1, 1)
        or accumulated_before.ndim != 2
        or accumulated_before.shape[0] != 1
        or not all(x.is_cuda and x.device == dev for x in
                   (draft_hidden, last_embedding, accumulated_before))
        or torch.cuda.is_current_stream_capturing()
        or "H200" not in torch.cuda.get_device_name(dev)
    ):
        raise RuntimeError(
            "P15_UNSUPPORTED_REAL_H200_FUSED_ON_POLICY_DEPTH10_CHECKPOINT"
        )
    original_k = model.predictor_k_cache[:, :1, :MAX_CACHE_SLOTS].clone()
    original_v = model.predictor_v_cache[:, :1, :MAX_CACHE_SLOTS].clone()
    hidden = draft_hidden.detach().clone().contiguous()
    embedding = last_embedding.detach().clone().contiguous()
    summed = accumulated_before.detach().clone().contiguous()
    frozen_inputs = (hidden.clone(), embedding.clone(), summed.clone())
    cache_start = torch.tensor(cache_len, dtype=torch.long, device=dev)
    indices = torch.arange(2, dtype=torch.long, device=dev)
    u = torch.tensor(CASES[0], dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    graph = torch.cuda.CUDAGraph()
    captured = False
    report = {
        "schema": SCHEMA,
        "status": "P15_NOT_YET_VALIDATED",
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED",
        ),
        "pid": os.getpid(),
        "gpu": torch.cuda.get_device_name(dev),
        "predictor_layers": 5,
        "backend": "matched_fused_predictor_device_offset",
        "initial_depth": depth,
        "initial_cache_len": cache_len,
        "captured_graph_count": 1,
        "capture_contains_gpu_second_round_cond": True,
        "capture_contains_conditional_bonus": True,
        "bonus_kernel_skip_trace_proven": False,
        "native_seeded_sampler_parity_proven": False,
        "cases": [],
        "branch_counts": {branch: 0 for branch in BRANCHES},
        "sampling": {"temperature": temperature, "top_k": top_k, "top_p": top_p},
    }

    def restore():
        model.predictor_k_cache[:, :1, :MAX_CACHE_SLOTS].copy_(original_k)
        model.predictor_v_cache[:, :1, :MAX_CACHE_SLOTS].copy_(original_v)

    def snapshot():
        return (
            model.predictor_k_cache[:, :1, :MAX_CACHE_SLOTS].clone(),
            model.predictor_v_cache[:, :1, :MAX_CACHE_SLOTS].clone(),
        )

    try:
        # CUDA kernels must be JIT-warmed OUTSIDE graph capture. Do not call
        # the nested GPU control-flow graph here (only its constituent kernels).
        with torch.inference_mode():
            for offset in (10, 11, 12):
                restore()
                dynamic_fused_predictor(
                    model, pair=torch.cat((embedding, embedding), dim=1),
                    cache_len_gpu=torch.tensor(
                        offset, device=dev, dtype=torch.long,
                    ), indices_gpu=indices,
                )
            for idx in (depth, depth + 1, depth + 2, depth + 3):
                logits = model.code_predictor.lm_head[idx](hidden)[0][:, -1, :]
                ids, _, probs, draft = fast_selector(
                    logits, u[0, 0:1],
                    temperature=temperature, top_p=top_p,
                )
                single_support(
                    logits, temperature=temperature, top_p=top_p, fast=True,
                )
                gpu_accept(ids, probs, ids, probs, u[0], draft)
        torch.cuda.synchronize(dev)
        restore()

        with (
            torch.cuda.graph(
                graph, stream=stream, capture_error_mode="thread_local",
            ),
            CUDAGraphCaptureControlFlowOpDispatchMode(),
            torch.inference_mode(),
        ):
            gpu_first, gpu_second = two_gpu_rounds(
                model, first_depth=depth, cache_len_gpu=cache_start,
                hidden=hidden, last_embedding=embedding,
                accumulated=summed, uniforms=u, indices_gpu=indices,
                temperature=temperature, top_p=top_p,
            )
        captured = True
        torch.cuda.synchronize(dev)
        graph_outputs = (gpu_first, gpu_second)
        restore()

        for index, uniforms in enumerate(CASES):
            u.copy_(torch.tensor(uniforms, dtype=torch.float32, device=dev))
            restore()
            graph.replay()
            torch.cuda.synchronize(dev)
            actual = tuple({
                key: tensor.detach().clone()
                for key, tensor in zip(ROUND_FIELDS, output)
            } for output in graph_outputs)
            graph_k, graph_v = snapshot()

            restore()
            expected_first = eager_oracle_round(
                model, depth=depth, cache_len=cache_len,
                hidden=hidden, last_embedding=embedding,
                accumulated=summed, uniforms=uniforms[0],
                temperature=temperature, top_p=top_p,
            )
            expected_second = eager_oracle_round(
                model, depth=expected_first["next_depth"],
                cache_len=expected_first["next_cache_len"],
                hidden=expected_first["next_hidden"],
                last_embedding=expected_first["next_embedding"],
                accumulated=expected_first["next_sum"],
                uniforms=uniforms[1], temperature=temperature, top_p=top_p,
            )
            oracle_k, oracle_v = snapshot()
            branch = _check_case(
                actual, (expected_first, expected_second),
                graph_k=graph_k, graph_v=graph_v,
                oracle_k=oracle_k, oracle_v=oracle_v,
                prefix_k=original_k, prefix_v=original_v,
                temperature=temperature, top_p=top_p,
            )
            report["branch_counts"][branch] += 1
            report["cases"].append({
                "case_index": index,
                "uniforms": [list(uniforms[0]), list(uniforms[1])],
                "round1": {
                    key: expected_first[key] for key in (
                        "draft", "committed", "accepted", "bonus",
                        "next_depth", "next_cache_len", "bonus_executed",
                    )
                },
                "round2": {
                    key: expected_second[key] for key in (
                        "draft", "committed", "accepted", "bonus",
                        "next_depth", "next_cache_len", "bonus_executed",
                    )
                },
                "transition": branch,
                "full_physical_kv_parity": True,
                "independent_dense_sampler_parity": True,
                "device_state_consumed_by_round2": True,
            })

        if any(report["branch_counts"][branch] == 0 for branch in BRANCHES):
            report["status"] = INCOMPLETE
        else:
            report["status"] = SUCCESS

        # Optional CUDA graph envelope; always restore the original prefix.
        event_blocks = []
        for _ in range(4):
            restore()
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize(dev)
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            event_ms = begin.elapsed_time(end) / repeats
            if not math.isfinite(event_ms) or event_ms <= 0:
                raise RuntimeError("P15_INVALID_CUDA_EVENT_ENVELOPE")
            event_blocks.append(event_ms)
        report["cuda_graph_event_ms_blocks"] = event_blocks
        report["median_cuda_graph_event_ms"] = statistics.median(event_blocks)
        report["limitations"] = [
            "Two real conditional rounds starting at on-policy depth10 only",
            "GPU branch controls head and device-offset verifier in second round",
            "Bonus head inside conditional branch, but no Nsight kernel-skip proof",
            "Uniforms are externally supplied; not native request seeded RNG",
            "No full 14-depth GPU loop, no HTTP throughput or audio quality claim",
            "CUDA graph event envelope is not matched native serving speedup",
        ]
        for value, saved in zip(
            (hidden, embedding, summed), frozen_inputs,
        ):
            if not torch.equal(value, saved):
                raise RuntimeError("P15_ON_POLICY_PROBE_INPUT_MUTATION")
        return report
    finally:
        # Failed conditional capture may poison CUDA. Do NOT retry in-worker.
        if captured:
            torch.cuda.synchronize(dev)
            restore()
            torch.cuda.synchronize(dev)
            if not torch.equal(
                model.predictor_k_cache[:, :1, :MAX_CACHE_SLOTS], original_k
            ) or not torch.equal(
                model.predictor_v_cache[:, :1, :MAX_CACHE_SLOTS], original_v
            ):
                raise RuntimeError("P15_FULL_PHYSICAL_KV_NOT_RESTORED")


def maybe_probe(
    model, *, frame: int, proposal_depth: int, cache_len: int,
    draft_hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated_before: torch.Tensor, temperature: float,
    top_k: int, top_p: float,
) -> None:
    out = os.environ.get(ENV)
    if not out:
        return
    conflicts = (
        "SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR",
        "SGLANG_OMNI_QTTS_CP_P8_DIR",
        "SGLANG_OMNI_QTTS_CP_P7_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR",
        "SGLANG_OMNI_QTTS_CP_P9B_DIR",
        "SGLANG_OMNI_QTTS_CP_P10A_DIR",
        "SGLANG_OMNI_QTTS_CP_P10B_DIR",
        "SGLANG_OMNI_QTTS_CP_P11A_DIR",
        "SGLANG_OMNI_QTTS_CP_P12A_DIR",
        "SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE",
    )
    if any(os.environ.get(name) for name in conflicts):
        raise RuntimeError("P15_CONFLICTING_HISTORICAL_PROBE_OR_HYBRID")
    if proposal_depth != FIRST_DEPTH + 1:
        return
    if getattr(model, "_qt_p15_done", False):
        return
    output = Path(out).expanduser().resolve()
    try:
        report = capture_two_rounds(
            model, depth=proposal_depth - 1, cache_len=cache_len,
            draft_hidden=draft_hidden, last_embedding=last_embedding,
            accumulated_before=accumulated_before,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
    except BaseException:
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        (output / (
            f"failure_pid{os.getpid()}_depth{proposal_depth:02d}.txt"
        )).write_text(traceback.format_exc())
        raise
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    report["frame"] = frame
    name = (
        f"two_round_pid{os.getpid()}_frame{frame:04d}"
        f"_depth{proposal_depth:02d}.json"
    )
    with (output / name).open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    model._qt_p15_done = True

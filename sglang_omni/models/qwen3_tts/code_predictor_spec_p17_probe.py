# SPDX-License-Identifier: Apache-2.0
"""P17 H200 real on-policy K=2/3/4 full-round and stage graph cost probe.

Two independent observations are required:
(1) per-case correctness against NATIVE independent head calls + an
    eager STATIC-END fused verifier and dense CPU accept/residual oracle;
(2) measured source-locked unmarked CUDA Graph envelopes for FULL round,
    multi-head proposal and matched dynamic verifier-only components.

Each checkpoint is genuine from the existing debug K2 path. Samples
over a frozen checkpoint probe conditional acceptance, NOT the global
on-policy distribution. No GPU loop, seeded RNG or production E2E claim.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import traceback
from pathlib import Path

import torch

from .code_predictor_spec import sampling_probs
from .code_predictor_spec_p15_two_round import _project
from .code_predictor_spec_p9b import cpu_reference
from .code_predictor_spec_p17_wide import (
    WIDTHS, UNIFORM_COUNT, SCHEMA, TOP_K, VOCAB,
    check_wide_backend, make_q_proposals, make_wide_round,
    prefix_accept_count, prepare_head_bank, static_fused_verifier,
    wide_device_fused_predictor,
)

ENV = "SGLANG_OMNI_QTTS_CP_P17_MULTIWIDTH_DIR"
DEPTHS = (1, 9)  # first proposal depth 2 / 10 in existing live debug K2
DRAW_CASES = 16
REPEAT = 16
BLOCKS = 4
SUCCESS = "P17_REAL_MULTIWIDTH_SAMPLE_AND_COST_PARITY_PASS_NOT_E2E_OR_SPEEDUP"


def _cpu_sample(probs: torch.Tensor, u: float) -> int:
    dist = probs.detach().float().cpu()[0]
    return int(torch.searchsorted(
        dist.cumsum(-1).contiguous(), torch.tensor(u, dtype=torch.float32),
    ).clamp(max=dist.numel()-1).item())


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor, *,
             atol: float = 0.0, rtol: float = 0.0) -> None:
    if actual.shape != expected.shape or not torch.allclose(
        actual, expected, atol=atol, rtol=rtol, equal_nan=False,
    ):
        if actual.shape == expected.shape:
            error = (actual.float()-expected.float()).abs().max().item()
        else:
            error = float("inf")
        raise RuntimeError(
            f"P17_{name}_PARITY_FAIL max_abs={error:.8g} "
            f"atol={atol} rtol={rtol}"
        )


@torch.no_grad()
def independent_reference(
    model, *, depth: int, width: int, cache_len: int,
    old_hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated: torch.Tensor, uniform: tuple[float, ...],
    graph_row: dict[str, torch.Tensor],
    graph_k: torch.Tensor, graph_v: torch.Tensor,
    prefix_k: torch.Tensor, prefix_v: torch.Tensor,
    temperature: float, top_p: float,
) -> dict:
    """Same real heads via module.forward, CPU dense p/q and static fused KV."""
    n = width - 1
    q_dense, drafts = [], []
    for j in range(n):
        native_q = model.code_predictor.lm_head[depth+j](old_hidden)[0][:, -1, :]
        dense_q = sampling_probs(
            native_q, temperature=temperature, top_k=TOP_K, top_p=top_p,
        )
        _compare(
            f"K{width}_DRAFT_HEAD{j}_DISTRIBUTION",
            sampling_probs(
                graph_row["q_logits"][j:j+1],
                temperature=temperature, top_k=TOP_K, top_p=top_p,
            ), dense_q, atol=1e-6, rtol=1e-6,
        )
        q_dense.append(dense_q)
        draft = _cpu_sample(dense_q, uniform[4*j])
        drafts.append(draft)
    _compare(
        f"K{width}_Q_DRAFT_CODES",
        graph_row["drafts"],
        torch.tensor(drafts, device=old_hidden.device, dtype=torch.long),
    )
    draft_embeddings = []
    for j, code in enumerate(drafts):
        _, proj = _project(
            model, depth+j, torch.tensor(
                [code], device=old_hidden.device, dtype=torch.long,
            ), accumulated.dtype,
        )
        draft_embeddings.append(proj)
    packed = torch.cat((last_embedding, *draft_embeddings), dim=1)
    reference_hidden = static_fused_verifier(
        model, packed=packed, cache_len=cache_len,
    )
    _compare(
        f"K{width}_FULL_VERIFIER_HIDDEN",
        graph_row["verified_hidden"], reference_hidden,
    )
    _compare(
        f"K{width}_PHYSICAL_K", graph_k,
        model.predictor_k_cache[:, :1, :16],
    )
    _compare(
        f"K{width}_PHYSICAL_V", graph_v,
        model.predictor_v_cache[:, :1, :16],
    )
    _compare(
        f"K{width}_COMMITTED_PREFIX_K",
        graph_k[:, :, :cache_len], prefix_k[:, :, :cache_len],
    )
    _compare(
        f"K{width}_COMMITTED_PREFIX_V",
        graph_v[:, :, :cache_len], prefix_v[:, :, :cache_len],
    )

    p_dense, decisions = [], []
    for j in range(width):
        native_p = model.code_predictor.lm_head[depth+j](
            reference_hidden[:, j:j+1, :],
        )[0][:, -1, :]
        target = sampling_probs(
            native_p, temperature=temperature, top_k=TOP_K, top_p=top_p,
        )
        _compare(
            f"K{width}_TARGET_HEAD{j}_DISTRIBUTION",
            sampling_probs(
                graph_row["p_logits"][j:j+1],
                temperature=temperature, top_k=TOP_K, top_p=top_p,
            ), target, atol=1e-6, rtol=1e-6,
        )
        p_dense.append(target)
        if j < n:
            decisions.append(cpu_reference(
                target, q_dense[j], (
                    uniform[4*j], uniform[4*j+1], uniform[4*j+2],
                ),
            ))
    accepted_flags = tuple(row["accepted"] for row in decisions)
    prefix = prefix_accept_count(accepted_flags)
    if prefix == n:
        bonus = _cpu_sample(p_dense[n], uniform[4*n])
    else:
        bonus = None
    committed = [
        decisions[j]["committed"] if j <= prefix else -1
        for j in range(n)
    ]
    committed.append(bonus if bonus is not None else -1)
    count = prefix + 1
    expected_tensor = torch.tensor(
        committed, device=old_hidden.device, dtype=torch.long,
    )
    _compare(
        f"K{width}_COMMITTED_CODES",
        graph_row["committed_codes"], expected_tensor,
    )
    _compare(
        f"K{width}_ACCEPT_FLAGS",
        graph_row["accepted_flags"],
        torch.tensor(
            accepted_flags, dtype=torch.bool, device=old_hidden.device,
        ),
    )
    if int(graph_row["accepted_prefix"].item()) != prefix:
        raise RuntimeError(f"P17_K{width}_ACCEPTED_PREFIX_FAIL")
    if int(graph_row["committed_count"].item()) != count:
        raise RuntimeError(f"P17_K{width}_COMMITTED_COUNT_FAIL")
    if int(graph_row["next_depth"].item()) != depth+count:
        raise RuntimeError(f"P17_K{width}_NEXT_DEPTH_FAIL")
    if int(graph_row["next_cache_len"].item()) != cache_len+count:
        raise RuntimeError(f"P17_K{width}_NEXT_CACHE_LEN_FAIL")

    ref_sum = accumulated.clone()
    last_projected = None
    for j in range(count):
        code = committed[j]
        if code < 0:
            raise RuntimeError("P17_ORACLE_NEGATIVE_COMMITTED_CODE")
        raw, projected = _project(
            model, depth+j,
            torch.tensor([code], device=old_hidden.device, dtype=torch.long),
            accumulated.dtype,
        )
        ref_sum = ref_sum + raw
        last_projected = projected
    _compare(f"K{width}_CODE_SUM", graph_row["next_sum"], ref_sum)
    _compare(
        f"K{width}_LAST_PROJECTED", graph_row["next_embedding"],
        last_projected,
    )
    _compare(
        f"K{width}_NEXT_DRAFT_HIDDEN", graph_row["next_hidden"],
        reference_hidden[:, count-1:count, :],
    )
    return {
        "draft_codes": drafts,
        "accepted_flags": list(accepted_flags),
        "accepted_prefix": prefix,
        "committed_codes": committed,
        "committed_count": count,
        "first_rejection": prefix if prefix < n else None,
        "bonus_committed": bonus is not None,
    }


def _clone_graph_row(row: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in row.items()}


def _time_graph(
    g: torch.cuda.CUDAGraph, dev: torch.device, *,
    uniforms_gpu: torch.Tensor, cases: list[list[float]],
    restore, vary_uniforms: bool,
) -> list[dict]:
    # Fix P15's last-uniform-only timing ambiguity: four distinct inputs
    # and repeated blocks. The verifier-only cell is explicitly a fixed
    # packed-input micrograph, so it does NOT pretend to vary its input.
    case_indices = (0, 5, 10, 15) if vary_uniforms else (0,)
    blocks = []
    for case_idx in case_indices:
        uniforms_gpu.copy_(torch.tensor(
            cases[case_idx], dtype=torch.float32, device=dev,
        ))
        for repeat_index in range(BLOCKS):
            restore()
            for _ in range(4):
                g.replay()
            torch.cuda.synchronize(dev)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPEAT):
                g.replay()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / REPEAT
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError("P17_INVALID_CUDA_EVENT_ENVELOPE")
            blocks.append({
                "case_index": case_idx, "repeat_index": repeat_index,
                "cuda_event_ms": ms,
            })
    return blocks


@torch.no_grad()
def capture_real_multiwidth(
    model, *, depth: int, cache_len: int, frame: int,
    draft_hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated_before: torch.Tensor,
    temperature: float, top_k: int, top_p: float,
) -> dict:
    """Independent width 2/3/4 captures using SAME actual checkpoint."""
    dev = last_embedding.device
    if (
        depth not in DEPTHS or cache_len != depth+1
        or top_k != TOP_K or not 0 < top_p <= 1 or temperature <= 0
        or draft_hidden.shape != last_embedding.shape
        or draft_hidden.shape[:2] != (1, 1)
        or accumulated_before.ndim != 2 or accumulated_before.shape[0] != 1
        or not all(x.is_cuda and x.device == dev for x in
                   (draft_hidden, last_embedding, accumulated_before))
        or "H200" not in torch.cuda.get_device_name(dev)
        or torch.cuda.is_current_stream_capturing()
        or model.predictor_k_cache.shape[2] < 16
    ):
        raise RuntimeError("P17_UNSUPPORTED_REAL_C1_H200_ON_POLICY_CHECKPOINT")
    for width in WIDTHS:
        check_wide_backend(model, width, cache_len)
    old_k = model.predictor_k_cache[:, :1, :16].clone()
    old_v = model.predictor_v_cache[:, :1, :16].clone()
    hidden = draft_hidden.detach().clone().contiguous()
    last = last_embedding.detach().clone().contiguous()
    summed = accumulated_before.detach().clone().contiguous()
    frozen = (hidden.clone(), last.clone(), summed.clone())
    root_seed = 20261122 + 97*frame + depth
    gen = torch.Generator().manual_seed(root_seed)
    cases = torch.rand(
        (DRAW_CASES, UNIFORM_COUNT), generator=gen, dtype=torch.float32,
    ).clamp(1e-4, 1-1e-4).tolist()
    uniforms_gpu = torch.tensor(cases[0], dtype=torch.float32, device=dev)
    start_gpu = torch.tensor(cache_len, dtype=torch.long, device=dev)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    result = {
        "schema": SCHEMA,
        "status": "P17_UNVALIDATED",
        "pid": os.getpid(),
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED",
        ),
        "frame": frame, "last_committed_depth": depth,
        "cache_len": cache_len, "gpu": torch.cuda.get_device_name(dev),
        "predictor_layers": len(model.code_predictor.model.layers),
        "backend": "P17_real_fused_bf16_5layer_wide",
        "sampling": {
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
        },
        "uniform_seed": root_seed,
        "uniform_cases_per_width": DRAW_CASES,
        "widths": {},
        "measurement_limits": [
            "One selected real K2 on-policy checkpoint; repeated uniforms",
            "Multiple heads use batched torch.bmm; backend parity required",
            "One wide verification, NOT two K2 conditional graph rounds",
            "Each candidate performs K-1 accept/residual GPU kernels",
            "No native seeded sampler, full K2/3/4 decoder or serving QPS",
            "Stage graphs are frozen-input component envelopes, NOT additive",
        ],
    }

    def restore():
        model.predictor_k_cache[:, :1, :16].copy_(old_k)
        model.predictor_v_cache[:, :1, :16].copy_(old_v)

    captured = False
    failed = False
    try:
        with torch.inference_mode():
            for width in WIDTHS:
                bank = prepare_head_bank(model, depth=depth, width=width)
                offsets = torch.arange(width, device=dev, dtype=torch.long)
                def operation():
                    return make_wide_round(
                        model, depth=depth, width=width,
                        cache_len_gpu=start_gpu, old_hidden=hidden,
                        last_embedding=last, accumulated=summed,
                        uniforms=uniforms_gpu, head_bank=bank,
                        offset_ids=offsets,
                        temperature=temperature, top_p=top_p,
                    )
                def proposal():
                    return make_q_proposals(
                        model, depth=depth, width=width,
                        old_hidden=hidden, head_bank=bank,
                        uniforms=uniforms_gpu,
                        temperature=temperature, top_p=top_p,
                        accumulation_dtype=summed.dtype,
                    )
                # Prepare a fixed real draft packed input for verifier-only
                # cost isolation, not a standalone speculative algorithm.
                restore()
                q_outputs = proposal()
                frozen_packed = torch.cat((last, q_outputs[-1]), dim=1).clone()
                def verifier_only():
                    return wide_device_fused_predictor(
                        model, packed=frozen_packed,
                        cache_len_gpu=start_gpu, offsets_gpu=offsets,
                    )
                # Compile every real width's kernels before capture.
                restore()
                operation()
                restore()
                proposal()
                restore()
                verifier_only()
                torch.cuda.synchronize(dev)

                graphs = {}
                outputs = None
                for name, invoke in (
                    ("full_round", operation),
                    ("proposal_only", proposal),
                    ("verifier_only", verifier_only),
                ):
                    restore()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        g, stream=stream, capture_error_mode="thread_local",
                    ), torch.inference_mode():
                        output = invoke()
                    graphs[name] = g
                    if name == "full_round":
                        outputs = output
                torch.cuda.synchronize(dev)
                captured = True
                if outputs is None:
                    raise RuntimeError("P17_MISSING_CAPTURED_FULL_ROUND_OUTPUT")
                observations = []
                for index, uniforms in enumerate(cases):
                    uniforms_gpu.copy_(torch.tensor(
                        uniforms, dtype=torch.float32, device=dev,
                    ))
                    restore()
                    graphs["full_round"].replay()
                    torch.cuda.synchronize(dev)
                    graph_row = _clone_graph_row(outputs)
                    graph_k = model.predictor_k_cache[:, :1, :16].clone()
                    graph_v = model.predictor_v_cache[:, :1, :16].clone()
                    restore()
                    evidence = independent_reference(
                        model, depth=depth, width=width,
                        cache_len=cache_len, old_hidden=hidden,
                        last_embedding=last, accumulated=summed,
                        uniform=tuple(uniforms),
                        graph_row=graph_row, graph_k=graph_k, graph_v=graph_v,
                        prefix_k=old_k, prefix_v=old_v,
                        temperature=temperature, top_p=top_p,
                    )
                    observations.append({
                        "index": index,
                        "accepted_prefix": evidence["accepted_prefix"],
                        "committed_count": evidence["committed_count"],
                        "first_rejection": evidence["first_rejection"],
                        "bonus_committed": evidence["bonus_committed"],
                    })

                # Fix first uniform across widths for a source-matched
                # runtime envelope; per-width decision counts are separate.
                uniforms_gpu.copy_(torch.tensor(
                    cases[0], dtype=torch.float32, device=dev,
                ))
                event_blocks = {}
                for name in ("full_round", "proposal_only", "verifier_only"):
                    restore()
                    event_blocks[name] = _time_graph(
                        graphs[name], dev, uniforms_gpu=uniforms_gpu,
                        cases=cases, restore=restore,
                        vary_uniforms=(name != "verifier_only"),
                    )
                medians = {
                    name: statistics.median(
                        row["cuda_event_ms"] for row in vals
                    ) for name, vals in event_blocks.items()
                }
                hist = {
                    str(committed): sum(
                        row["committed_count"] == committed
                        for row in observations
                    )
                    for committed in range(1, width+1)
                }
                mean_committed = statistics.mean(
                    row["committed_count"] for row in observations
                )
                result["widths"][str(width)] = {
                    "on_policy_real_checkpoint": True,
                    "batched_multihead_and_static_fused_parity": True,
                    "full_physical_kv_parity": True,
                    "independent_cpu_accept_residual_parity": True,
                    "case_count": len(observations),
                    "accepted_prefix_histogram": {
                        str(p): sum(
                            row["accepted_prefix"] == p
                            for row in observations
                        ) for p in range(width)
                    },
                    "committed_count_histogram": hist,
                    "mean_committed_per_round_fixed_checkpoint": mean_committed,
                    "uniform_case_seed": root_seed,
                    "timed_full_and_proposal_case_indices": [0, 5, 10, 15],
                    "verifier_only_frozen_case_index": 0,
                    "cuda_graph_event_ms_blocks": event_blocks,
                    "cuda_graph_event_ms_medians": medians,
                    "full_round_ms_per_committed_code_SCREEN_ONLY": (
                        medians["full_round"] / mean_committed
                    ),
                    "cases": observations,
                }
        if set(result["widths"]) != {"2", "3", "4"}:
            raise RuntimeError("P17_MISSING_K_WIDTH_CASES")
        for value, expected in zip((hidden, last, summed), frozen):
            if not torch.equal(value, expected):
                raise RuntimeError("P17_LIVE_INPUT_MUTATED")
        result["status"] = SUCCESS
        return result
    except BaseException:
        failed = True
        raise
    finally:
        # Capture errors may poison the CUDA context; best effort only.
        # Preserve first exception and require fresh worker on failure.
        try:
            torch.cuda.synchronize(dev)
            restore()
            torch.cuda.synchronize(dev)
            if not torch.equal(
                model.predictor_k_cache[:, :1, :16], old_k,
            ) or not torch.equal(
                model.predictor_v_cache[:, :1, :16], old_v,
            ):
                raise RuntimeError("P17_WIDE_CACHE_NOT_FULLY_RESTORED")
        except BaseException:
            if not failed:
                raise


def maybe_probe(
    model, *, frame: int, proposal_depth: int, cache_len: int,
    draft_hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated_before: torch.Tensor,
    temperature: float, top_k: int, top_p: float,
) -> None:
    raw_dir = os.environ.get(ENV)
    if not raw_dir:
        return
    # Activation-only test: prove real K2 dispatch without starting the
    # expensive K2/K3/K4 capture. Not a correctness or speedup gate.
    if os.environ.get("SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_ONLY") == "1":
        return
    conflicts = (
        "SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR",
        "SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR",
        "SGLANG_OMNI_QTTS_CP_P16A_DIR",
        "SGLANG_OMNI_QTTS_CP_P8_DIR",
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
        raise RuntimeError("P17_CONFLICTING_OLDER_PROBE_OR_HYBRID")
    depth = proposal_depth-1
    if depth not in DEPTHS:
        return
    max_count = int(os.environ.get(
        "SGLANG_OMNI_QTTS_CP_P17_MAX_FRAMES_PER_DEPTH", "1",
    ))
    if not 1 <= max_count <= 3:
        raise ValueError("P17_MAX_FRAMES_PER_DEPTH must be 1..3")
    counts = getattr(model, "_qt_p17_counts", None)
    if counts is None:
        counts = {}
        model._qt_p17_counts = counts
    if counts.get(depth, 0) >= max_count:
        return
    out = Path(raw_dir).expanduser().resolve()
    try:
        report = capture_real_multiwidth(
            model, depth=depth, cache_len=cache_len, frame=frame,
            draft_hidden=draft_hidden,
            last_embedding=last_embedding,
            accumulated_before=accumulated_before,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
    except BaseException:
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        (out / (
            f"failure_pid{os.getpid()}_frame{frame:04d}_depth{depth:02d}.txt"
        )).write_text(traceback.format_exc())
        raise
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = out / (
        f"wide_pid{os.getpid()}_frame{frame:04d}_depth{depth:02d}.json"
    )
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    counts[depth] = counts.get(depth, 0) + 1

# SPDX-License-Identifier: Apache-2.0
"""P19A: matched C1 H200 CUDA Graph *component* costs at native-AR checkpoints.

Each arm starts from the SAME real P18 native-eager AR state, with the same
physical K/V prefix. AR executes the real model's 1-token predictor forward,
head, native seeded sampler, embedding update. K2/3/4 execute P17's full wide
round. Captured replays must match each arm's eager output and physical KV
exactly; original native state and RNG are restored.

These are independently captured, fixed-checkpoint graphs: NOT a whole
Predictor Graph-ON, not causal speculative serving and NOT E2E speedup.
"""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import traceback
from pathlib import Path
from typing import Callable

from .code_predictor_spec_p19_diag import diagnostic_config, event as diag_event
from .code_predictor_spec_p19_profile import (
    CAPTURES, DEFAULT_OPS, EAGER_REFERENCE, EAGER_WARMUPS,
    FIRST_GRAPH_REPLAYS, MAX_RETAINED_GRAPHS, MIN_REPLAY_OPS,
    current_profile,
)

import torch

from .code_predictor_spec_p17_wide import (
    check_wide_backend, make_wide_round, prepare_head_bank,
)
from .code_predictor_spec_p18_census import feasible_widths

SCHEMA = "qwen3tts-p19a-matched-graph-component-cost-v1"
ENV = "SGLANG_OMNI_QTTS_CP_P19_COST_DIR"
MAX_FRAMES = "SGLANG_OMNI_QTTS_CP_P19_FRAMES_PER_REQUEST"
REPEATS = 12
BLOCKS = 4
WARM_REPLAYS = 3
COST_ARMS = ("AR", "2", "3", "4")


def _clone(result: dict) -> dict[str, torch.Tensor]:
    if not isinstance(result, dict) or not result:
        raise RuntimeError("P19_EMPTY_ARM_RESULT")
    if not all(isinstance(v, torch.Tensor) for v in result.values()):
        raise RuntimeError("P19_ARM_RETURN_MUST_BE_DEVICE_TENSORS")
    return {k: v.detach().clone() for k, v in result.items()}


def _assert_exact(label: str, actual: torch.Tensor, expected: torch.Tensor):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not torch.equal(actual, expected)):
        if actual.shape == expected.shape:
            diff = (actual.float() - expected.float()).abs().max().item()
        else:
            diff = float("inf")
        raise RuntimeError(
            f"P19_{label}_GRAPH_EAGER_PARITY_FAIL max_abs={diff:g} "
            f"actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )


def _one_arm(
    *, operation: Callable[[], dict[str, torch.Tensor]],
    restore: Callable[[], None], model, arm: str,
    device: torch.device, old_k: torch.Tensor, old_v: torch.Tensor,
    prefix_len: int, trace: Callable[[str], None] | None = None,
    profile: str = "DEFAULT", keepalive: list | None = None,
) -> dict:
    """Compile -> capture -> exact eager-vs-replay -> timed replay blocks."""
    def mark(stage: str) -> None:
        if trace is not None:
            trace(stage)

    mark("P19_ARM_EAGER_BEGIN")
    restore()
    eager = _clone(operation())
    mark("P19_ARM_EAGER_OUTPUT_CAPTURED")
    eager_k = model.predictor_k_cache[:, :1, :16].detach().clone()
    eager_v = model.predictor_v_cache[:, :1, :16].detach().clone()
    restore()
    for _ in range(2):
        mark("P19_ARM_WARMUP_BEGIN")
        operation()
        restore()
    torch.cuda.synchronize(device)
    mark("P19_ARM_EAGER_AND_WARMUP_SYNC_PASS")

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    restore()
    mark("P19_ARM_GRAPH_CAPTURE_BEGIN")
    gc_guard = profile == "GC_GUARD"
    old_gc = gc.isenabled()
    if gc_guard:
        mark("P19_ARM_GC_CAPTURE_GUARD_ENTER")
        gc.disable()
    try:
        with torch.cuda.graph(
            graph, stream=capture_stream, capture_error_mode="thread_local"
        ), torch.inference_mode():
            outputs = operation()
    finally:
        if gc_guard:
            if old_gc:
                gc.enable()
            mark("P19_ARM_GC_CAPTURE_GUARD_EXIT")
    mark("P19_ARM_GRAPH_CAPTURE_RETURNED")
    torch.cuda.synchronize(device)
    mark("P19_ARM_GRAPH_CAPTURE_SYNC_PASS")
    restore()
    mark("P19_ARM_FIRST_REPLAY_BEGIN")
    graph.replay()
    torch.cuda.synchronize(device)
    mark("P19_ARM_FIRST_REPLAY_SYNC_PASS")
    replay = _clone(outputs)
    if set(replay) != set(eager):
        raise RuntimeError(f"P19_{arm}_OUTPUT_SCHEMA_DRIFT")
    for name in replay:
        _assert_exact(f"{arm}_{name}", replay[name], eager[name])
    _assert_exact(
        f"{arm}_PHYSICAL_KV_K",
        model.predictor_k_cache[:, :1, :16], eager_k,
    )
    _assert_exact(
        f"{arm}_PHYSICAL_KV_V",
        model.predictor_v_cache[:, :1, :16], eager_v,
    )
    # Eager arm legitimately writes its verification/AR KV suffix.
    # Only the truly committed prefix must remain immutable.
    _assert_exact(
        f"{arm}_PREFIX_K", eager_k[:, :, :prefix_len],
        old_k[:, :, :prefix_len],
    )
    _assert_exact(
        f"{arm}_PREFIX_V", eager_v[:, :, :prefix_len],
        old_v[:, :, :prefix_len],
    )

    mark("P19_ARM_REPLAY_EAGER_PARITY_PASS")
    records = []
    if profile == "MIN_REPLAY":
        # One replay with exact parity, but NO timed/warmup replays.
        # Still capture and retire 50 graphs at the 14 native depths.
        mark("P19_ARM_MIN_REPLAY_NO_TIMING_PASS")
    else:
        for block in range(BLOCKS):
            mark(f"P19_ARM_TIMING_BLOCK_{block}_BEGIN")
            restore()
            for _ in range(WARM_REPLAYS):
                graph.replay()
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPEATS):
                graph.replay()
            stop.record()
            stop.synchronize()
            mark(f"P19_ARM_TIMING_BLOCK_{block}_SYNC_PASS")
            duration = start.elapsed_time(stop) / REPEATS
            if not math.isfinite(duration) or duration <= 0:
                raise RuntimeError(f"P19_{arm}_INVALID_GRAPH_EVENT")
            records.append({
                "block": block, "replays": REPEATS, "cuda_event_ms": duration,
            })
    if profile == "RETAIN_GRAPH":
        if keepalive is None or len(keepalive) >= MAX_RETAINED_GRAPHS:
            raise RuntimeError("P19_RETAIN_GRAPH_OWNERSHIP_CAP_INVALID")
        # Keep graph AND capture-backed output tensors/stream alive until
        # this disposable one-request H200 WORKER exits. No later replay.
        # Intentionally diagnostic-only; never retain graphs in serving.
        keepalive.append((graph, outputs, capture_stream))
        torch.cuda.synchronize(device)
        mark("P19_ARM_GRAPH_RETAINED_AND_SYNC_PASS")
    else:
        del graph
        torch.cuda.synchronize(device)
        mark("P19_ARM_GRAPH_RELEASE_AND_SYNC_PASS")
    return {
        "parity": "CAPTURED_REPLAY_VS_EAGER_OUTPUT_AND_FULL_KV_PASS",
        "cuda_graph_event_ms_blocks": records,
        "median_cuda_event_ms": (
            statistics.median(r["cuda_event_ms"] for r in records)
            if records else None
        ),
        "replays_per_block": REPEATS if records else 0,
    }


def _matched_eager_arm(
    *, operation: Callable[[], dict[str, torch.Tensor]],
    restore: Callable[[], None], device: torch.device,
    mark: Callable[[str], None],
) -> None:
    """65 real eager invocations matching original 3+1+61 graph-arm calls.

    Preserve the original restore cadence: reference; two warmups;
    capture-surrogate; first-replay surrogate; four blocks each with
    a single restore, three warm calls and twelve more calls.
    This is WORKLOAD count matching, NOT an equivalent GPU allocator,
    kernel launch overhead, CUDA graph memory pool or performance result.
    """
    count = 0
    restore()
    reference = operation()
    if not isinstance(reference, dict) or not reference:
        raise RuntimeError("P19_MATCHED_EAGER_REFERENCE_EMPTY")
    count += EAGER_REFERENCE
    torch.cuda.synchronize(device)
    mark("P19_ARM_MATCHED_EAGER_REFERENCE_SYNC_PASS")
    restore()
    for _ in range(EAGER_WARMUPS):
        operation()
        count += 1
        restore()
    torch.cuda.synchronize(device)
    mark("P19_ARM_MATCHED_EAGER_WARMUPS_SYNC_PASS")
    restore()
    operation()
    count += CAPTURES
    torch.cuda.synchronize(device)
    mark("P19_ARM_MATCHED_EAGER_CAPTURE_SURROGATE_SYNC_PASS")
    restore()
    operation()
    count += FIRST_GRAPH_REPLAYS
    torch.cuda.synchronize(device)
    mark("P19_ARM_MATCHED_EAGER_FIRST_SURROGATE_SYNC_PASS")
    for block in range(BLOCKS):
        restore()
        for _ in range(WARM_REPLAYS):
            operation()
            count += 1
        torch.cuda.synchronize(device)
        for _ in range(REPEATS):
            operation()
            count += 1
        torch.cuda.synchronize(device)
        mark(f"P19_ARM_MATCHED_EAGER_BLOCK_{block}_SYNC_PASS")
    if count != DEFAULT_OPS:
        raise RuntimeError(
            f"P19_MATCHED_EAGER_WRONG_OPERATION_COUNT {count}"
        )
    mark("P19_ARM_MATCHED_EAGER_TOTAL_65_SYNC_PASS")


def select_cost_frame(
    chosen: set[int], *, frame: int, depth: int,
    max_frames: int, diagnostic_depth: int | None = None,
) -> bool:
    """Choose real frames at depth 1 normally, at selected depth in diagnosis.

    A targeted D9 diagnostic is not visited at D1 by maybe_measure,
    so it MUST be able to select its first frame when D9 is reached.
    In the formal all-depth cohort, selection remains anchored at D1
    and every later depth of that chosen frame is measured.
    """
    if not 1 <= depth <= 14 or not 1 <= max_frames <= 2:
        raise ValueError("P19_INVALID_COST_FRAME_SELECTION_CONTRACT")
    if diagnostic_depth is not None:
        if not 1 <= diagnostic_depth <= 14:
            raise ValueError("P19_INVALID_DIAGNOSTIC_TARGET_DEPTH")
        if depth != diagnostic_depth:
            return False
    if frame in chosen:
        return True
    entry_depth = diagnostic_depth if diagnostic_depth is not None else 1
    if depth != entry_depth or len(chosen) >= max_frames:
        return False
    chosen.add(frame)
    return True


@torch.no_grad()
def maybe_measure(
    model, *, case: dict, req_hash: str, frame: int, depth: int,
    cache_len: int, semantic_position: torch.Tensor,
    old_hidden: torch.Tensor, last_embedding: torch.Tensor,
    accumulated: torch.Tensor, uniforms_gpu: torch.Tensor,
) -> None:
    root_raw = os.environ.get(ENV)
    if not root_raw:
        return
    from .code_predictor_spec_p18_census import armed

    if not armed():
        return
    diagnostic = diagnostic_config()
    profile = current_profile(diagnostic)
    cumulative = (diagnostic is not None and diagnostic[0] in
                  ("ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER"))
    # Targeted diagnostics run ONE depth; cumulative diagnostics run
    # each native depth 1..terminal on ONE already selected frame.
    if diagnostic is not None and (
        (cumulative and depth > diagnostic[1])
        or (not cumulative and depth != diagnostic[1])
    ):
        return
    if diagnostic is not None and diagnostic[0] == "OFF":
        diag_event(
            "P19_COST_PROBE_SKIPPED_OFF", depth=depth,
            frame=frame, case_id=case["case_id"], arm="OFF",
        )
        return
    root = Path(root_raw)
    if not root.is_absolute() or root_raw != str(root):
        raise RuntimeError("P19_COST_DIR_MUST_BE_ABSOLUTE")
    if (os.environ.get("SGLANG_OMNI_QTTS_CP_SPEC_K") != "0"
            or torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("P19_REQUIRES_NATIVE_EAGER_K2_OFF")
    max_frames = int(os.environ.get(MAX_FRAMES, "1"))
    if not 1 <= max_frames <= 2:
        raise RuntimeError("P19_COST_FRAMES_PER_REQUEST_MUST_BE_1_OR_2")
    req_select = getattr(model, "_p19_selected_frames", None)
    if req_select is None:
        req_select = {}
        model._p19_selected_frames = req_select
    chosen = req_select.setdefault(req_hash, set())
    if not select_cost_frame(
        chosen, frame=frame, depth=depth, max_frames=max_frames,
        diagnostic_depth=(
            diagnostic[1] if diagnostic is not None and not cumulative
            else None
        ),
    ):
        return
    if diagnostic is not None:
        diag_event(
            "P19_COST_TARGET_FRAME_SELECTED",
            depth=depth, frame=frame, arm=diagnostic[0],
            case_id=case["case_id"],
        )

    source_sha = os.environ.get("SGLANG_OMNI_QTTS_CP_P18_SOURCE_SHA", "")
    if len(source_sha) != 40 or source_sha != os.environ.get(
        "SGLANG_OMNI_QTTS_CP_P19_SOURCE_SHA"
    ):
        raise RuntimeError("P19_P18_SAME_WORKER_SOURCE_SHA_REQUIRED")
    if not (1 <= depth <= 14 and cache_len == depth + 1):
        raise RuntimeError("P19_NATIVE_DEPTH_INVALID")
    if not all(x.is_cuda and x.device == old_hidden.device for x in (
        old_hidden, last_embedding, accumulated, uniforms_gpu,
        semantic_position,
    )):
        raise RuntimeError("P19_NATIVE_C1_CUDA_INPUT_REQUIRED")
    if "H200" not in torch.cuda.get_device_name(old_hidden.device):
        raise RuntimeError("P19_REQUIRES_H200")
    if old_hidden.shape != last_embedding.shape or old_hidden.shape[0:2] != (1, 1):
        raise RuntimeError("P19_C1_PREDICTOR_SHAPE_FAIL")
    path = root / "rows" / (
        f"pid{os.getpid()}_req{req_hash}_f{frame:07d}_d{depth:02d}.json"
    )
    if path.exists():
        raise RuntimeError("P19_DUPLICATE_SOURCE_LOCKED_COST")
    old_hidden = old_hidden.detach().clone().contiguous()
    last_embedding = last_embedding.detach().clone().contiguous()
    accumulated = accumulated.detach().clone().contiguous()
    uniform_input = uniforms_gpu.detach().clone().contiguous()
    immutable_inputs = (
        old_hidden.clone(), last_embedding.clone(),
        accumulated.clone(), uniform_input.clone(),
    )
    old_k = model.predictor_k_cache[:, :1, :16].clone()
    old_v = model.predictor_v_cache[:, :1, :16].clone()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(old_hidden.device).clone()
    temp = float(model.sub_temperature_tensor[0].item())
    top_k = int(model.sub_top_k_tensor[0].item())
    top_p = float(model.sub_top_p_tensor[0].item())
    if abs(temp - 0.9) > 1e-5 or top_k != 50 or abs(top_p - 1) > 1e-5:
        raise RuntimeError("P19_SAMPLING_PARAMETERS_NOT_FROZEN")
    device = old_hidden.device

    embedding_buf = getattr(model, "predictor_embedding_buffer", None)
    if embedding_buf is not None:
        embedding_buf = embedding_buf[:1].detach().clone().contiguous()
    projected_buf = getattr(model, "predictor_projected_buffer", None)
    if projected_buf is not None:
        projected_buf = projected_buf[:1].detach().clone().contiguous()
    original_embed_buf = embedding_buf.detach().clone() if embedding_buf is not None else None
    original_projected_buf = (
        projected_buf.detach().clone() if projected_buf is not None else None
    )

    tmp_code = torch.empty((1,), dtype=torch.long, device=device)
    tmp_sum = accumulated.clone()
    seed_positions = model.sub_seed_positions(semantic_position)[depth]
    codec_table = model.code_predictor.model.codec_embedding.weight[depth]
    projected_weights = model.predictor_projected_embeddings
    proj_table = (
        projected_weights[depth]
        if projected_weights is not None and depth < 14 else None
    )

    def restore():
        model.predictor_k_cache[:, :1, :16].copy_(old_k)
        model.predictor_v_cache[:, :1, :16].copy_(old_v)
        tmp_code.zero_()
        tmp_sum.copy_(accumulated)
        if embedding_buf is not None:
            embedding_buf.copy_(original_embed_buf)
        if projected_buf is not None:
            projected_buf.copy_(original_projected_buf)

    def native_step() -> dict[str, torch.Tensor]:
        # Exactly the native single-token predictor/head/sampler/embedding path
        # at this source-locked real depth. Private output buffers have the
        # same shapes, dtypes and native backend eligibility as serving.
        tmp_sum.copy_(accumulated)
        hidden = model.predictor_forward_tokens(
            token_embeds=last_embedding, batch_size=1, cache_len=cache_len,
        )
        logits, _ = model.code_predictor.lm_head[depth](hidden)
        embed, projected = model.sample_subtalker_embedding(
            logits[:, -1, :], codec_table,
            tmp_sum, tmp_code,
            proj_table,
            embedding_buf,
            projected_buf if proj_table is not None else None,
            seed_positions,
        )
        if depth < 14:
            if projected_weights is None:
                projected_next = model.code_predictor.project_input(
                    embed.unsqueeze(1)
                ).contiguous()
            else:
                projected_next = projected.unsqueeze(1).contiguous()
        else:
            # No next step after code15, so no projected token needed.
            projected_next = last_embedding
        return {
            "native_next_code": tmp_code.clone(),
            "native_next_sum": tmp_sum.clone(),
            "native_next_embedding": projected_next.clone(),
            "native_hidden": hidden.clone(),
        }

    records = {}
    failed = False
    retained = None
    if profile == "RETAIN_GRAPH":
        held_hash = getattr(model, "_p19_diag_retained_req_hash", None)
        if held_hash is None:
            model._p19_diag_retained_req_hash = req_hash
            model._p19_diag_retained_graphs = []
        elif held_hash != req_hash:
            raise RuntimeError("P19_RETAINED_GRAPHS_CROSS_REQUEST")
        retained = model._p19_diag_retained_graphs
        if len(retained) >= MAX_RETAINED_GRAPHS:
            raise RuntimeError("P19_RETAINED_GRAPHS_EXCEED_MAXIMUM")
    try:
        candidate_arms = ("AR", *(str(w) for w in feasible_widths(depth)))
        if diagnostic is not None and diagnostic[0] not in (
            "ALL_GRAPH", "ALL_DEPTHS_GRAPH", "ALL_DEPTHS_EAGER"
        ):
            requested = diagnostic[0].split("_", 1)[0]
            wanted = "AR" if requested == "AR" else requested[1:]
            if wanted not in candidate_arms:
                raise RuntimeError("P19_DIAG_ARM_ILLEGAL_AT_THIS_DEPTH")
            candidate_arms = (wanted,)
        for arm in candidate_arms:
            def phase(stage: str) -> None:
                if diagnostic is not None:
                    diag_event(
                        stage, depth=depth, arm=arm, frame=frame,
                        case_id=case["case_id"],
                    )

            phase("P19_ARM_SELECTED")
            if arm == "AR":
                operation = native_step
            else:
                w = int(arm)
                check_wide_backend(model, w, cache_len)
                bank = prepare_head_bank(model, depth=depth, width=w)
                offsets = torch.arange(w, dtype=torch.long, device=device)
                gpu_len = torch.tensor(cache_len, dtype=torch.long, device=device)

                def operation(
                    w=w, bank=bank, offsets=offsets, gpu_len=gpu_len,
                ):
                    return make_wide_round(
                        model, depth=depth, width=w,
                        cache_len_gpu=gpu_len,
                        old_hidden=old_hidden,
                        last_embedding=last_embedding,
                        accumulated=accumulated,
                        uniforms=uniform_input,
                        head_bank=bank, offset_ids=offsets,
                        temperature=temp, top_p=top_p,
                    )
            if diagnostic is not None and diagnostic[0].endswith("_EAGER"):
                # Diagnostic ONLY: never create full P19 matched cost rows.
                phase("P19_EAGER_ONLY_OPERATION_BEGIN")
                if profile == "MATCHED_EAGER":
                    _matched_eager_arm(
                        operation=operation, restore=restore,
                        device=device, mark=phase,
                    )
                else:
                    restore()
                    result = operation()
                    torch.cuda.synchronize(device)
                    if not isinstance(result, dict) or not result:
                        raise RuntimeError("P19_EAGER_ONLY_EMPTY_ARM_RESULT")
                phase("P19_EAGER_ONLY_DEVICE_SYNC_PASS")
                restore()
            else:
                records[arm] = _one_arm(
                    operation=operation, restore=restore, model=model,
                    arm=arm, device=device, old_k=old_k, old_v=old_v,
                    prefix_len=cache_len, trace=phase,
                    profile=profile, keepalive=retained,
                )
            if not all(torch.equal(x, y) for x, y in zip(
                (old_hidden, last_embedding, accumulated, uniform_input),
                immutable_inputs,
            )):
                raise RuntimeError(
                    f"P19_{arm}_FROZEN_NATIVE_CHECKPOINT_INPUT_MUTATED"
                )
        restore()
        if diagnostic is not None:
            if (profile == "RETAIN_GRAPH"
                    and depth == diagnostic[1]):
                if retained is None or len(retained) != MAX_RETAINED_GRAPHS:
                    raise RuntimeError(
                        "P19_RETAIN_GRAPH_EXPECTED_50_LIVE_GRAPHS"
                    )
                diag_event(
                    "P19_RETAINED_50_GRAPHS_LIVE_UNTIL_WORKER_EXIT",
                    depth=depth, arm=diagnostic[0], frame=frame,
                    case_id=case["case_id"],
                )
            diag_event(
                "P19_DIAGNOSTIC_ARM_RETURNED_NO_COST_SELECTION",
                depth=depth, arm=diagnostic[0], frame=frame,
                case_id=case["case_id"],
            )
        if (not torch.equal(torch.get_rng_state(), cpu_rng)
                or not torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
                or not torch.equal(model.predictor_k_cache[:, :1, :16], old_k)
                or not torch.equal(model.predictor_v_cache[:, :1, :16], old_v)):
            raise RuntimeError("P19_COST_KV_OR_RNG_RESTORATION_FAILED")
        if diagnostic is not None:
            # Never pass an isolated one-arm result into the full all-arm
            # analyzer. The phase trace is the diagnostic deliverable.
            return
        report = {
            "schema": SCHEMA, "source_sha": source_sha,
            "pid": os.getpid(), "gpu": torch.cuda.get_device_name(device),
            "case_id": case["case_id"], "prompt_id": case["prompt_id"],
            "seed": case["seed"], "request_hash": req_hash,
            "request_nonce": case["request_nonce"],
            "frame": frame, "last_committed_depth": depth,
            "cache_len": cache_len,
            "trajectory": "NATIVE_AR_EAGER_CHECKPOINT_NOT_SPEC_ROLLOUT",
            "cost_metric": "CUDA_GRAPH_EVENT_ENVELOPE_FIXED_CHECKPOINT",
            "matched_native_sampler": "QWEN3_TTS_REAL_SEEDED_SUBTALKER",
            "uniforms": "P18_SHARED_QP_FIXED",
            "frames_per_request_measured": max_frames,
            "arms": records,
        }
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("x", encoding="utf-8") as out:
            json.dump(report, out, sort_keys=True, indent=2)
            out.write("\n")
    except BaseException:
        failed = True
        if diagnostic is not None:
            diag_event(
                "P19_DIAGNOSTIC_EXCEPTION_ENTERED",
                depth=depth, arm=diagnostic[0], frame=frame,
                case_id=case["case_id"],
            )
        failure = root / "failures" / (
            f"pid{os.getpid()}_req{req_hash}_f{frame:07d}_d{depth:02d}.txt"
        )
        failure.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            torch.cuda.synchronize(device)
            restore()
            # A failed graph capture must not leave a changed CPU/CUDA
            # RNG stream on the continuing native request. The prior
            # parity check still fails the experiment on any mutation.
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device=device)
            torch.cuda.synchronize(device)
            if (not torch.equal(model.predictor_k_cache[:, :1, :16], old_k)
                    or not torch.equal(model.predictor_v_cache[:, :1, :16], old_v)):
                raise RuntimeError("P19_FINAL_PHYSICAL_KV_RESTORE_FAILED")
            if diagnostic is not None:
                diag_event(
                    "P19_DIAGNOSTIC_FINAL_RESTORE_SYNC_PASS",
                    depth=depth, arm=diagnostic[0], frame=frame,
                    case_id=case["case_id"],
                )
                if (not failed
                        and cumulative and depth == diagnostic[1]):
                    request_id = getattr(
                        model, "_p18_current_request_id", None
                    )
                    if not isinstance(request_id, str) or not request_id:
                        raise RuntimeError(
                            "P19_NATIVE_NEXT_FRAME_REQUEST_ID_MISSING"
                        )
                    if getattr(
                        model, "_p19_native_boundary_next_req", None
                    ) is not None:
                        raise RuntimeError(
                            "P19_NATIVE_NEXT_FRAME_PREVIOUS_PROBE_PENDING"
                        )
                    # CPU-only flag. The next genuine native Predictor
                    # invocation traces its initial GPU operations before
                    # P18 .item() can surface an asynchronous CUDA fault.
                    model._p19_native_boundary_next_req = request_id
                    model._p19_native_boundary_from_frame = frame
                    model._p19_native_boundary_case_id = case["case_id"]
                    diag_event(
                        "P19_SELECTED_FRAME_WAITING_NEXT_NATIVE",
                        depth=depth, frame=frame, arm=diagnostic[0],
                        case_id=case["case_id"],
                    )
        except BaseException:
            if not failed:
                raise

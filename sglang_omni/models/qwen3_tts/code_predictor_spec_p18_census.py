# SPDX-License-Identifier: Apache-2.0
"""P18A: source-locked native-AR-EAGER on-policy depth census (H200/C1).

Opt-in only, after a post-health ARM. Never silently use K2 trajectories.
The real native sampler commits its code before this observer runs; we
temporarily evaluate K2/K3/K4 counterfactuals at the frozen pre-next-token
KV state, verify them against P17's independent oracle, then restore EVERY
physical KV slot. One shared predeclared uniform tuple per checkpoint.
NO native CUDA Graph / HTTP latency or production speedup conclusions.
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path

import torch

from .code_predictor_spec_p17_probe import (
    _clone_graph_row, independent_reference,
)
from .code_predictor_spec_p17_wide import (
    UNIFORM_COUNT, check_wide_backend, make_wide_round,
    prepare_head_bank,
)

SCHEMA = "qwen3tts-p18a-native-ar-eager-all-depth-v1"
DIR_ENV = "SGLANG_OMNI_QTTS_CP_P18_DIR"
ARM_ENV = "SGLANG_OMNI_QTTS_CP_P18_ARM_FILE"
CASE_ENV = "SGLANG_OMNI_QTTS_CP_P18_CASE_FILE"
FRAMES_ENV = "SGLANG_OMNI_QTTS_CP_P18_FRAMES_PER_REQUEST"
SOURCE_ENV = "SGLANG_OMNI_QTTS_CP_P18_SOURCE_SHA"
SAMPLING = {"temperature": 0.9, "top_k": 50, "top_p": 1.0}
MAX_DEPTH = 14  # last committed codec code (code15 is still native AR)
FULL_WIDTHS = (2, 3, 4)


def armed() -> bool:
    root = os.environ.get(DIR_ENV, "")
    marker = os.environ.get(ARM_ENV, "")
    return bool(root and marker and os.path.isfile(marker))


def feasible_widths(depth: int) -> tuple[int, ...]:
    """lm_head indices are 0..14; last verifier row needs its own head."""
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError("P18_DEPTH_MUST_BE_1_TO_14")
    return tuple(w for w in FULL_WIDTHS if depth + w - 1 <= 14)


def uniform_seed(source_sha: str, case_id: str, frame: int, depth: int) -> int:
    """Stable paired counterfactual uniforms, not native codebook RNG."""
    payload = f"p18a-v1|{source_sha}|{case_id}|{frame}|{depth}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _case() -> dict:
    name = os.environ.get(CASE_ENV)
    if not name:
        raise RuntimeError("P18_CASE_PATH_MISSING")
    path = Path(name)
    if not path.is_absolute():
        raise RuntimeError("P18_CASE_PATH_NOT_ABSOLUTE")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or set(record) != {
        "case_id", "prompt_id", "seed", "request_nonce",
    }:
        raise RuntimeError("P18_CASE_SCHEMA_DRIFT")
    if (not isinstance(record["case_id"], str)
            or not record["case_id"]
            or not isinstance(record["prompt_id"], str)
            or not isinstance(record["seed"], int)
            or not isinstance(record["request_nonce"], str)):
        raise RuntimeError("P18_CASE_INVALID_IDENTITY")
    return record


def _write_once(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as out:
        json.dump(record, out, sort_keys=True, indent=2)
        out.write("\n")


@torch.no_grad()
def maybe_census(
    model, *, depth: int, cache_len: int, old_hidden: torch.Tensor,
    last_embedding: torch.Tensor, accumulated: torch.Tensor,
    native_code: torch.Tensor, semantic_position: torch.Tensor,
) -> None:
    """Call AFTER native code[d] was sampled, BEFORE its KV append.

    This is the original native-AR checkpoint: old_hidden comes from the
    preceding code, last_embedding is the actual newly sampled code[d],
    and cache_len == depth+1. P17 q-draft uses this exact state.
    """
    if not armed():
        return
    if (os.environ.get("SGLANG_OMNI_QTTS_CP_SPEC_K", "0") != "0"
            or torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("P18_REQUIRES_NATIVE_EAGER_AND_K2_OFF")
    if cache_len != depth + 1 or depth not in range(1, MAX_DEPTH + 1):
        raise RuntimeError("P18_BAD_NATIVE_DEPTH_OR_KV_POSITION")
    if (not model.sub_has_sampled_rows or model.sub_has_argmax_rows
            or model.sub_batch_size != 1
            or model.config.num_code_groups != 16
            or old_hidden.shape != last_embedding.shape
            or old_hidden.shape[0:2] != (1, 1)
            or accumulated.ndim != 2 or accumulated.shape[0] != 1
            or semantic_position.numel() != 1
            or native_code.numel() != 1
            or not all(t.is_cuda for t in (
                old_hidden, last_embedding, accumulated, native_code,
            ))
            or "H200" not in torch.cuda.get_device_name(old_hidden.device)):
        raise RuntimeError("P18_REQUIRES_NATIVE_SAMPLED_H200_C1")
    # P19 CUDA700 attribution: a CPU-only durable breadcrumb BEFORE
    # reading a CUDA scalar. .item() can merely surface a stale
    # asynchronous error from previous native/P19 kernels.
    # No extra synchronization is introduced unless explicitly enabled
    # in a DIAGNOSTIC-ONLY run (never a timed cost cohort).
    diag = os.environ.get("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM") is not None
    if diag:
        from .code_predictor_spec_p19_diag import event

        event("P18_CENSUS_ENTER_BEFORE_FRAME_ITEM", depth=depth)
        if os.environ.get("SGLANG_OMNI_QTTS_CP_P19_DIAG_FENCE") == "1":
            torch.cuda.synchronize(old_hidden.device)
            event("P18_CENSUS_EXPLICIT_DEVICE_FENCE_PASS", depth=depth)
    frame = int(semantic_position.item())
    if diag:
        event("P18_CENSUS_FRAME_ITEM_PASS", depth=depth, frame=frame)
    req_id = getattr(model, "_p18_current_request_id", None)
    if not isinstance(req_id, str) or not req_id:
        raise RuntimeError("P18_MISSING_ENGINE_REQUEST_ID")

    case = _case()
    by_request = getattr(model, "_p18_frames_by_request", None)
    if by_request is None:
        by_request = {}
        model._p18_frames_by_request = by_request
    request_state = by_request.setdefault(req_id, {
        "case_id": case["case_id"], "frames": set(),
    })
    if request_state["case_id"] != case["case_id"]:
        raise RuntimeError("P18_REQUEST_ID_REUSED_ACROSS_CASES")
    captured = request_state["frames"]
    max_frames = int(os.environ.get(FRAMES_ENV, "4"))
    if not 1 <= max_frames <= 16:
        raise RuntimeError("P18_MAX_FRAMES_OUT_OF_RANGE")
    if frame not in captured:
        if depth != 1 or len(captured) >= max_frames:
            return
        captured.add(frame)
    source_sha = os.environ.get(SOURCE_ENV, "")
    if len(source_sha) != 40:
        raise RuntimeError("P18_MISSING_SOURCE_ATTESTATION")

    root = Path(os.environ[DIR_ENV])
    if not root.is_absolute():
        raise RuntimeError("P18_DIR_NOT_ABSOLUTE")
    req_hash = hashlib.sha256(req_id.encode()).hexdigest()
    file = root / "rows" / (
        f"pid{os.getpid()}_req{req_hash}_f{frame:07d}_d{depth:02d}.json"
    )
    if file.exists():
        raise RuntimeError("P18_DUPLICATE_CHECKPOINT")
    # Wide fused kernels may reuse scratch buffers that a live native AR
    # tensor aliases. Snapshot the ORIGINAL views as well as K/V so this
    # read-only observer can detect and restore non-KV side effects.
    live_hidden, live_last = old_hidden, last_embedding
    live_sum, live_code = accumulated, native_code
    old_hidden = old_hidden.detach().clone().contiguous()
    last_embedding = last_embedding.detach().clone().contiguous()
    accumulated = accumulated.detach().clone().contiguous()
    old_code = native_code.detach().clone()
    old_k = model.predictor_k_cache[:, :1, :16].clone()
    old_v = model.predictor_v_cache[:, :1, :16].clone()
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state(old_hidden.device).clone()

    def restore():
        model.predictor_k_cache[:, :1, :16].copy_(old_k)
        model.predictor_v_cache[:, :1, :16].copy_(old_v)

    failed = False
    try:
        native_temp = float(model.sub_temperature_tensor[0].item())
        native_top_k = int(model.sub_top_k_tensor[0].item())
        native_top_p = float(model.sub_top_p_tensor[0].item())
        if (abs(native_temp - SAMPLING["temperature"]) > 1e-5
                or native_top_k != SAMPLING["top_k"]
                or abs(native_top_p - SAMPLING["top_p"]) > 1e-5):
            raise RuntimeError("P18_SAMPLING_PARAMETERS_MISMATCH")
        u_seed = uniform_seed(source_sha, case["case_id"], frame, depth)
        generator = torch.Generator(device="cpu").manual_seed(u_seed)
        uniforms = torch.rand(
            (UNIFORM_COUNT,), generator=generator,
            dtype=torch.float32,
        ).clamp(1e-4, 1 - 1e-4).tolist()
        uniforms_gpu = torch.tensor(
            uniforms, device=old_hidden.device, dtype=torch.float32,
        )
        widths = {}
        for width in feasible_widths(depth):
            check_wide_backend(model, width, cache_len)
            bank = prepare_head_bank(model, depth=depth, width=width)
            restore()
            result = make_wide_round(
                model, depth=depth, width=width,
                cache_len_gpu=torch.tensor(
                    cache_len, device=old_hidden.device, dtype=torch.long,
                ),
                old_hidden=old_hidden, last_embedding=last_embedding,
                accumulated=accumulated, uniforms=uniforms_gpu,
                head_bank=bank, offset_ids=torch.arange(
                    width, device=old_hidden.device, dtype=torch.long,
                ),
                temperature=native_temp, top_p=native_top_p,
            )
            graph_row = _clone_graph_row(result)
            graph_k = model.predictor_k_cache[:, :1, :16].clone()
            graph_v = model.predictor_v_cache[:, :1, :16].clone()
            restore()
            # P17 oracle checks independent actual native heads, q/p
            # probabilities, positive residual decisions, projected
            # embeddings and ALL physical K/V against a static verifier.
            evidence = independent_reference(
                model, depth=depth, width=width, cache_len=cache_len,
                old_hidden=old_hidden, last_embedding=last_embedding,
                accumulated=accumulated, uniform=tuple(uniforms),
                graph_row=graph_row, graph_k=graph_k, graph_v=graph_v,
                prefix_k=old_k, prefix_v=old_v,
                temperature=native_temp, top_p=native_top_p,
            )
            widths[str(width)] = {
                "status": "INDEPENDENT_P17_DISTRIBUTION_AND_KV_PARITY_PASS",
                "accepted_prefix": evidence["accepted_prefix"],
                "accepted_flags": evidence["accepted_flags"],
                "committed_count": evidence["committed_count"],
                "first_rejection": evidence["first_rejection"],
                "bonus_committed": evidence["bonus_committed"],
            }
        restore()
        # P19A opt-in: same real native-AR checkpoint, case, depth and
        # P18 paired q/p uniforms, now with measured native AR vs wide
        # Graph component envelopes. Independent P17 oracle ABOVE
        # must pass first. Never substitute P17 cross-SHA timing.
        if os.environ.get("SGLANG_OMNI_QTTS_CP_P19_COST_DIR"):
            from .code_predictor_spec_p19_cost import maybe_measure

            maybe_measure(
                model, case=case, req_hash=req_hash, frame=frame,
                depth=depth, cache_len=cache_len,
                semantic_position=semantic_position,
                old_hidden=old_hidden, last_embedding=last_embedding,
                accumulated=accumulated, uniforms_gpu=uniforms_gpu,
            )
            restore()
        if (not torch.equal(live_hidden, old_hidden)
                or not torch.equal(live_last, last_embedding)
                or not torch.equal(live_sum, accumulated)
                or not torch.equal(live_code, old_code)):
            raise RuntimeError("P18_LIVE_NATIVE_INPUT_OR_OUTPUT_MUTATED")
        if (not torch.equal(model.predictor_k_cache[:, :1, :16], old_k)
                or not torch.equal(model.predictor_v_cache[:, :1, :16], old_v)
                or not torch.equal(torch.get_rng_state(), rng_cpu)
                or not torch.equal(
                    torch.cuda.get_rng_state(old_hidden.device), rng_cuda
                )):
            raise RuntimeError("P18_PHYSICAL_KV_OR_RNG_STATE_MUTATED")
        record = {
            "schema": SCHEMA,
            "source_sha": source_sha,
            "pid": os.getpid(),
            "gpu": torch.cuda.get_device_name(old_hidden.device),
            "trajectory": "NATIVE_AR_EAGER_CODEBOOK_SAMPLING",
            "probe": "P17_INDEPENDENT_ORACLE_COUNTERFACTUAL",
            "case_id": case["case_id"],
            "prompt_id": case["prompt_id"],
            "request_nonce": case["request_nonce"],
            "request_hash": req_hash,
            "requested_seed": case["seed"],
            "native_sub_sampling_seed": int(
                model.sub_sampling_seed_tensor[0].item()
            ),
            "frame": frame,
            "last_committed_depth": depth,
            "cache_len": cache_len,
            "native_committed_code": int(native_code.item()),
            "uniform_seed": u_seed,
            "sampling": {
                "temperature": native_temp,
                "top_k": native_top_k,
                "top_p": native_top_p,
            },
            "widths": widths,
        }
        _write_once(file, record)
    except BaseException:
        failed = True
        failure = root / "failures" / (
            f"pid{os.getpid()}_req{req_hash}_f{frame:07d}_d{depth:02d}.txt"
        )
        failure.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        # Failed CUDA kernels can poison the context; preserve primary error.
        try:
            torch.cuda.synchronize(old_hidden.device)
            restore()
            # Restore ALL original views, even on a detected scratch alias.
            # An incorrect mutation still raises above and invalidates run.
            live_hidden.copy_(old_hidden)
            live_last.copy_(last_embedding)
            live_sum.copy_(accumulated)
            live_code.copy_(old_code)
            torch.cuda.synchronize(old_hidden.device)
            if (not torch.equal(model.predictor_k_cache[:, :1, :16], old_k)
                    or not torch.equal(model.predictor_v_cache[:, :1, :16], old_v)):
                raise RuntimeError("P18_FINAL_FULL_KV_RESTORE_FAIL")
        except BaseException:
            if not failed:
                raise

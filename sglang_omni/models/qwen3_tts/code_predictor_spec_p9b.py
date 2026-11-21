# SPDX-License-Identifier: Apache-2.0
"""P9B: full GPU tensor proposal/accept/residual draw from REAL K2 p/q logits.

Research-only C1 post-health opt-in trace. All proposal, p/q top-k/top-p,
accept min(1,p/q), residual max(p-q,0), and replacement sampling execute
entirely on CUDA. No per-round .item() or Python decisions inside graphs.
An independent CUDA Generator drives one graph replay (tested explicitly).

The current CPU-debug SpecDec output is NOT replaced by this probe:
P9B does not yet incorporate its result into Predictor KV/bonus or the
14-depth WHILE. The numerical reference is the existing *PyTorch*
sampling_probs, not proven bit-identical to Omni's native fused sampler.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .code_predictor_spec import sampling_probs

if TYPE_CHECKING:
    from .sglang_model import Qwen3TTSTalker

ROOT_ENV = "SGLANG_OMNI_QTTS_CP_P9B_DIR"
SCHEMA = "qwen3tts-cpspec-p9b-real-logits-gpu-sampling-v1"
TEST_UNIFORMS = (
    (0.125, 0.250, 0.375),
    (0.625, 0.750, 0.875),
    (0.995, 0.015, 0.515),
    (0.333, 0.333, 0.333),
)


def gpu_probs(
    logits: torch.Tensor, *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Capture-safe top-k THEN top-p categorical distribution.

    No Python .item() / Tensor.__bool__ / sync in the GPU hot path.
    Validate scalar configs and input shapes *before* graph capture.
    """
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("P9B requires a [1,vocab] logit row")
    vocab = logits.shape[-1]
    if (
        not math.isfinite(temperature) or temperature <= 0
        or not math.isfinite(top_p) or not 0 < top_p <= 1
        or not isinstance(top_k, int) or isinstance(top_k, bool)
        or not 1 <= top_k <= vocab
    ):
        raise ValueError("P9B requires finite T>0, 0<top_p<=1, 1<=top_k<=vocab")
    score = logits.float() / temperature
    vals, indices = torch.topk(score, top_k, dim=-1, sorted=True)
    prob = torch.softmax(vals, dim=-1)
    if top_p < 1:
        remove = torch.cumsum(prob, dim=-1) - prob >= top_p
        # The first token is always kept because top_p > 0.
        prob = torch.where(remove, torch.zeros_like(prob), prob)
        prob = prob / prob.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(score)
    return out.scatter(dim=-1, index=indices, src=prob)


def gpu_round(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-GPU K2 single-proposal categorical draw with positive residual.

    uniforms: [1,3] with columns (draft_q, accept, residual_draw).
    Explicit inputs make exact deterministic CPU oracle parity possible.
    Production RNG test separately replaces uniforms by torch.rand using a
    private graph-local CUDA generator (not the live request RNG).
    """
    if logits_p.shape != logits_q.shape or uniforms.shape != (1, 3):
        raise ValueError("P9B expected same [1,V] p/q and [1,3] uniforms")
    p = gpu_probs(logits_p, temperature=temperature, top_k=top_k, top_p=top_p)
    q = gpu_probs(logits_q, temperature=temperature, top_k=top_k, top_p=top_p)
    u_q, u_a, u_r = uniforms[:, :1], uniforms[:, 1:2], uniforms[:, 2:3]

    # CDF draw is lossless (modulo FP32 probability rounding) and needs
    # no host scalar. Clamp protects the top cumulative float rounding.
    draft = torch.searchsorted(
        q.cumsum(-1).contiguous(), u_q.contiguous()
    ).clamp(max=q.shape[-1] - 1).to(torch.long)
    q_x = q.gather(1, draft)
    p_x = p.gather(1, draft)
    threshold = torch.minimum(
        torch.ones_like(p_x), p_x / q_x.clamp_min(1e-20)
    )
    survived = u_a < threshold
    positive = (p - q).clamp_min(0)
    mass = positive.sum(dim=-1, keepdim=True)
    # If p==q, a rejection cannot mathematically occur. Still compute a
    # finite alternate CDF for the non-selected branch, without branching.
    residual = torch.where(
        mass > 1e-12,
        positive / mass.clamp_min(1e-12),
        p,
    )
    corrected = torch.searchsorted(
        residual.cumsum(-1).contiguous(), u_r.contiguous()
    ).clamp(max=p.shape[-1] - 1).to(torch.long)
    committed = torch.where(survived, draft, corrected)
    return committed, survived, draft, p, q


def cpu_reference(
    p: torch.Tensor, q: torch.Tensor, uniforms: tuple[float, float, float]
) -> dict:
    """CPU per-case oracle, using already computed FP32 target distributions."""
    if p.shape != q.shape or p.ndim != 2 or p.shape[0] != 1:
        raise ValueError("P9B CPU reference shapes invalid")
    p0, q0 = p[0].detach().float().cpu(), q[0].detach().float().cpu()

    def sample(dist: torch.Tensor, u: float) -> int:
        return int(
            torch.searchsorted(
                dist.cumsum(-1).contiguous(),
                torch.tensor(u, dtype=dist.dtype),
            ).clamp(max=dist.numel()-1).item()
        )

    proposal = sample(q0, uniforms[0])
    threshold = min(1., float(p0[proposal]) / max(1e-20, float(q0[proposal])))
    accepted = uniforms[1] < threshold
    residual = (p0 - q0).clamp_min(0)
    if float(residual.sum()) > 1e-12:
        residual /= residual.sum()
    else:
        residual = p0
    corrected = sample(residual, uniforms[2])
    return {
        "draft": proposal,
        "accepted": bool(accepted),
        "committed": proposal if accepted else corrected,
    }


def profile_real_logits(
    *,
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repeats: int = 64,
) -> dict:
    """Graph-capture real checkpoint logits, deterministic exact oracle + RNG."""
    if (
        not logits_p.is_cuda or not logits_q.is_cuda
        or logits_p.shape != logits_q.shape or logits_p.ndim != 2
        or logits_p.shape[0] != 1
        or not 16 <= repeats <= 256
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("P9B requires real single CUDA logit row and 16..256 repeats")
    device = logits_p.device
    saved_p = logits_p.detach().clone().contiguous()
    saved_q = logits_q.detach().clone().contiguous()
    u = torch.tensor(TEST_UNIFORMS[0], device=device, dtype=torch.float32).reshape(1, 3)
    original_p = saved_p.clone()
    original_q = saved_q.clone()

    # Existing PyTorch q/p distribution reference is intentionally
    # independent of the capture-safe implementation. It syncs only in
    # the diagnostic/control path, never in the captured graph.
    p_ref = sampling_probs(
        saved_p, temperature=temperature, top_k=top_k, top_p=top_p
    )
    q_ref = sampling_probs(
        saved_q, temperature=temperature, top_k=top_k, top_p=top_p
    )
    report = {
        "status": "P9B_GPU_SAMPLER_NOT_YET_VALIDATED",
        "gpu": torch.cuda.get_device_name(device),
        "vocab": saved_p.shape[-1],
        "sampling": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
        "overlap": float(torch.minimum(p_ref, q_ref).sum().item()),
        "checks": [],
    }
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.inference_mode():
        for _ in range(3):
            gpu_round(
                saved_p, saved_q, u,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        graph, stream=side, capture_error_mode="thread_local"
    ), torch.inference_mode():
        token, survived, proposal, got_p, got_q = gpu_round(
            saved_p, saved_q, u,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
    # CUDA graph capture RECORDS kernels but does not execute them.
    # Synchronize alone cannot initialize graph output tensors: the
    # previous P9B H200 test compared unexecuted got_p/got_q to the
    # actual reference and falsely reported ~5-6% distribution error.
    graph.replay()
    torch.cuda.synchronize(device)
    report["deterministic_graph_replayed_before_distribution_check"] = True
    report["distribution_p_max_abs"] = float((got_p - p_ref).abs().max().item())
    report["distribution_q_max_abs"] = float((got_q - q_ref).abs().max().item())
    if not torch.allclose(got_p, p_ref, rtol=1e-6, atol=1e-6) or not torch.allclose(
        got_q, q_ref, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError(
            f"P9B_GPU_DISTRIBUTION_PARITY_FAIL p={report['distribution_p_max_abs']} "
            f"q={report['distribution_q_max_abs']}"
        )
    for triple in TEST_UNIFORMS:
        u.copy_(torch.tensor(triple, device=device, dtype=torch.float32).view(1, 3))
        graph.replay()
        torch.cuda.synchronize(device)
        exp = cpu_reference(p_ref, q_ref, triple)
        actual = {
            "draft": int(proposal[0, 0].item()),
            "committed": int(token[0, 0].item()),
            "accepted": bool(survived[0, 0].item()),
        }
        passed = exp == actual
        report["checks"].append({
            "uniforms": list(triple),
            "actual": actual,
            "expected": exp,
            "pass": passed,
        })
        if not passed:
            raise RuntimeError(
                f"P9B_GPU_DRAW_PARITY_FAIL uniforms={triple} "
                f"actual={actual} expected={exp}"
            )

    # A separate PRIVATE generator is mandatory. Never advance Omni's
    # production fused sampler generator or the current K2 debug generator.
    rng = torch.Generator(device=device)
    rng.manual_seed(20261121)
    rng_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        rng_graph, stream=side, capture_error_mode="thread_local"
    ), torch.inference_mode():
        graph_uniforms = torch.rand(
            (1, 3), device=device, generator=rng, dtype=torch.float32
        )
        random_token, random_accepted, _, _, _ = gpu_round(
            saved_p, saved_q, graph_uniforms,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
    torch.cuda.synchronize(device)
    observations = []
    for _ in range(4):
        rng_graph.replay()
        torch.cuda.synchronize(device)
        observations.append(graph_uniforms.detach().clone().cpu().tolist()[0])
    if len({tuple(row) for row in observations}) != 4:
        raise RuntimeError("P9B graph RNG did not change across replays")
    report["independent_graph_rng_replay_unique"] = True

    for _ in range(6):
        rng_graph.replay()
    torch.cuda.synchronize(device)
    event_a = torch.cuda.Event(enable_timing=True)
    event_b = torch.cuda.Event(enable_timing=True)
    event_a.record()
    for _ in range(repeats):
        rng_graph.replay()
    event_b.record()
    event_b.synchronize()
    ms = event_a.elapsed_time(event_b) / repeats
    if not math.isfinite(ms) or ms <= 0:
        raise RuntimeError("P9B invalid GPU sampling Graph event cost")
    report["gpu_sampler_with_independent_rng_graph_ms"] = ms
    report["output_input_parity"] = (
        torch.equal(saved_p, original_p) and torch.equal(saved_q, original_q)
    )
    if not report["output_input_parity"]:
        raise RuntimeError("P9B GPU sampling unexpectedly mutated model logits")
    report["status"] = (
        "P9B_REAL_LOGITS_GPU_PROPOSAL_ACCEPT_RESIDUAL_AND_RNG_PASS_"
        "NOT_FULL_K2"
    )
    report["limitations"] = [
        "Uses actual K2 target/draft logits from live checkpoint",
        "GPU samples/accept/reject run in a standalone CUDA Graph, not live Predictor loop",
        "Probability parity is against PyTorch sampling_probs, NOT native fused SGLang RNG",
        "Independent private CUDA graph RNG; not linked to model request seed",
        "GPU sampler timing includes top-k/top-p and RNG graph launches",
        "No 14-depth KV/bonus state updates, EOS equivalence or production speedup",
    ]
    return report


def maybe_probe(
    model: "Qwen3TTSTalker",
    *,
    frame_index: int,
    proposal_depth: int,
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    root = os.environ.get(ROOT_ENV)
    if not root:
        return
    if any(os.environ.get(name) for name in (
        "SGLANG_OMNI_QTTS_CP_P4_DIR",
        "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_P7_DIR",
        "SGLANG_OMNI_QTTS_CP_P8_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    )):
        raise RuntimeError("P9B cannot coexist with other predictor probes")
    seen = getattr(model, "_qt_cp_p9b_seen", None)
    if seen is None:
        seen = set()
        model._qt_cp_p9b_seen = seen
    if proposal_depth not in (2, 10) or proposal_depth in seen:
        return
    folder = Path(root).expanduser().resolve()
    try:
        report = profile_real_logits(
            logits_p=logits_p,
            logits_q=logits_q,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeats=64,
        )
    except BaseException:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (folder / f"failure_pid{os.getpid()}_depth{proposal_depth:02d}.txt").open(
            "x", encoding="utf-8"
        ) as handle:
            handle.write(traceback.format_exc())
        raise
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    report.update({
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "pid": os.getpid(),
        "frame": frame_index,
        "proposal_depth": proposal_depth,
    })
    dest = folder / (
        f"sample_pid{os.getpid()}_frame{frame_index:04d}_depth{proposal_depth:02d}.json"
    )
    with dest.open("x", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, **report}, handle, indent=2)
        handle.write("\n")
    seen.add(proposal_depth)

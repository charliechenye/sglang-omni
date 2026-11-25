# SPDX-License-Identifier: Apache-2.0
"""P12B real 5-layer one-token target AR vs packed-two K2 verifier cost.

Captures two FIXED CUDA graphs from the exact SAME live on-policy K2
prefix, weights, backend, token embedding and GPU. Times replay using
ABBA blocks. Requires hidden output and prefix/first-written K/V parity
before accepting timings; restores all 16 KV slots on any logical
failure after capture. No live sampled code or RNG is changed.

Measured one-token forward and two-token forward are not interchangeable
operations: one-token AR also needs token sampling and code commit,
while packed verifier includes two token rows plus draft cost. Reports
costs separately, NEVER treats forward call count as GPU work or claims
production hybrid speedup.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import traceback
from pathlib import Path

import torch

from .code_predictor_spec_p7_dispatch import compare

ENV = "SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR"
SCHEMA = "qwen3tts-cpspec-p12b-real-target-ar-vs-packed-verifier-cost-v1"
DEPTHS = (4, 10)


def capture_ar_vs_packed(
    model, *,
    last_embedding: torch.Tensor,
    proposed_embedding: torch.Tensor,
    cache_len: int,
    repeats: int = 32,
) -> dict:
    dev = last_embedding.device
    if (
        last_embedding.shape[:2] != (1, 1)
        or proposed_embedding.shape != last_embedding.shape
        or not last_embedding.is_cuda or not proposed_embedding.is_cuda
        or cache_len < 2 or cache_len + 2 > 16
        or model.predictor_k_cache.shape[2] < 16
        or not 8 <= repeats <= 256
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("P12B requires real on-policy C1 one/pair CUDA embeddings and 16 KV slots")
    pair = torch.cat((last_embedding, proposed_embedding), dim=1).detach().clone().contiguous()
    one = last_embedding.detach().clone().contiguous()
    original_one = one.clone()
    original_pair = pair.clone()
    cache_k = model.predictor_k_cache
    cache_v = model.predictor_v_cache
    old_k = cache_k[:, :1, :16].clone()
    old_v = cache_v[:, :1, :16].clone()

    def restore():
        cache_k[:, :1, :16].copy_(old_k)
        cache_v[:, :1, :16].copy_(old_v)

    def snapshot():
        return cache_k[:, :1, :16].clone(), cache_v[:, :1, :16].clone()

    def execute_one():
        return model.predictor_forward_tokens(
            token_embeds=one.clone(), batch_size=1, cache_len=cache_len
        ).clone()

    def execute_pair():
        return model.predictor_forward_tokens(
            token_embeds=pair.clone(), batch_size=1, cache_len=cache_len
        ).clone()

    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    captured = False
    report = {
        "schema": SCHEMA,
        "status": "P12B_NOT_PARITY_VERIFIED",
        "gpu": torch.cuda.get_device_name(dev),
        "predictor_layers": len(model.code_predictor.model.layers),
        "source_sha": os.environ.get(
            "SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA", "UNATTESTED"
        ),
        "cache_len": cache_len,
        "repeats": repeats,
    }
    try:
        with torch.inference_mode():
            for fn in (execute_one, execute_pair):
                restore()
                for _ in range(2):
                    fn()
        torch.cuda.synchronize(dev)
        captured_graphs = {}
        for name, fn in (("ar_one", execute_one), ("packed_two", execute_pair)):
            restore()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                g, stream=side, capture_error_mode="thread_local"
            ), torch.inference_mode():
                out = fn()
            captured_graphs[name] = (g, out)
        torch.cuda.synchronize(dev)
        captured = True

        restore()
        with torch.inference_mode():
            eager_one = execute_one()
        one_k, one_v = snapshot()

        restore()
        with torch.inference_mode():
            eager_pair = execute_pair()
        pair_k, pair_v = snapshot()
        # Same first token, same pre-existing K/V context. The packed
        # second token is a causal suffix and cannot change its first row.
        checks = {
            "first_hidden": compare(eager_one[:, :1, :], eager_pair[:, :1, :]),
            "first_written_k": compare(
                one_k[:, :, cache_len:cache_len+1],
                pair_k[:, :, cache_len:cache_len+1],
            ),
            "first_written_v": compare(
                one_v[:, :, cache_len:cache_len+1],
                pair_v[:, :, cache_len:cache_len+1],
            ),
            "original_prefix_k": compare(
                pair_k[:, :, :cache_len], old_k[:, :, :cache_len],
            ),
            "original_prefix_v": compare(
                pair_v[:, :, :cache_len], old_v[:, :, :cache_len],
            ),
            "persistent_one_input": compare(one, original_one),
            "persistent_pair_input": compare(pair, original_pair),
        }
        for name, (g, output) in captured_graphs.items():
            restore()
            g.replay()
            torch.cuda.synchronize(dev)
            exp = eager_one if name == "ar_one" else eager_pair
            observed_k, observed_v = snapshot()
            reference_k, reference_v = (
                (one_k, one_v) if name == "ar_one" else (pair_k, pair_v)
            )
            checks[f"{name}_hidden"] = compare(output, exp)
            checks[f"{name}_entire_k"] = compare(observed_k, reference_k)
            checks[f"{name}_entire_v"] = compare(observed_v, reference_v)
        report["checks"] = checks
        report["parity_pass"] = all(c["close"] for c in checks.values())
        if not report["parity_pass"]:
            bad = {n: c["max_abs"] for n, c in checks.items() if not c["close"]}
            raise RuntimeError(f"P12B_ACTUAL_AR_VS_PACKED_KV_PARITY_FAIL: {bad}")

        def event_ms(g):
            for _ in range(4):
                g.replay()
            torch.cuda.synchronize(dev)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(repeats):
                g.replay()
            b.record()
            b.synchronize()
            ms = a.elapsed_time(b) / repeats
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError("P12B CUDA event time invalid")
            return ms

        orders = (
            ("ar_one", "packed_two"), ("packed_two", "ar_one"),
            ("packed_two", "ar_one"), ("ar_one", "packed_two"),
        )
        timings = {n: [] for n in ("ar_one", "packed_two")}
        for order in orders:
            for name in order:
                restore()
                timings[name].append(event_ms(captured_graphs[name][0]))
        report["event_ms_by_block"] = timings
        report["median_cuda_event_ms"] = {
            name: statistics.median(rows) for name, rows in timings.items()
        }
        report["one_over_pair_ratio"] = (
            report["median_cuda_event_ms"]["ar_one"] /
            report["median_cuda_event_ms"]["packed_two"]
        )
        report["status"] = (
            "P12B_REAL_TARGET_AR_ONE_VS_PACKED_TWO_KV_PARITY_PASS_"
            "COMPONENT_TIMING_NOT_E2E"
        )
        report["limitations"] = [
            "Real model static cache_len and on-policy committed prefix only",
            "One-token AR and two-token verifier are not equivalent compute units",
            "No sampling/embedding/bonus, model KV state restored after probe",
            "CUDA event graph envelopes not pure GPU busy time",
            "Does not prove hybrid sample quality, native RNG equivalence or speedup",
        ]
    finally:
        if captured:
            torch.cuda.synchronize(dev)
            restore()
            torch.cuda.synchronize(dev)
    return report


def maybe_profile(
    model, *, frame: int, depth: int, cache_len: int,
    last_embedding: torch.Tensor, proposed_embedding: torch.Tensor,
):
    root = os.environ.get(ENV)
    if not root:
        return
    if depth not in DEPTHS or depth in getattr(model, "_qt_p12b_seen", set()):
        return
    if any(os.environ.get(e) for e in (
        "SGLANG_OMNI_QTTS_CP_P12A_DIR",
        "SGLANG_OMNI_QTTS_CP_P11A_DIR",
        "SGLANG_OMNI_QTTS_CP_P10A_DIR",
        "SGLANG_OMNI_QTTS_CP_P9A_DIR",
        "SGLANG_OMNI_QTTS_CP_P5_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    )):
        raise RuntimeError("P12B target-cost probe must run alone")
    dest = Path(root).expanduser().resolve()
    try:
        result = capture_ar_vs_packed(
            model, last_embedding=last_embedding,
            proposed_embedding=proposed_embedding,
            cache_len=cache_len, repeats=32,
        )
    except BaseException:
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        (dest / f"failure_pid{os.getpid()}_depth{depth:02d}.txt").write_text(
            traceback.format_exc()
        )
        raise
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    result.update({"frame": frame, "depth": depth, "pid": os.getpid()})
    file = dest / f"ar_vs_pair_pid{os.getpid()}_frame{frame:04d}_depth{depth:02d}.json"
    with file.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    seen = getattr(model, "_qt_p12b_seen", set())
    seen.add(depth)
    model._qt_p12b_seen = seen

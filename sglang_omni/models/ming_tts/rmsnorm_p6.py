"""Experimental, opt-in HF Qwen2 RMSNorm substitution for Ming AudioVAE.

Not a production feature. Call after checkpoint loading, before graph capture.
"""
from __future__ import annotations

import logging
import os

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

logger = logging.getLogger(__name__)


def maybe_replace_audio_vae_qwen_norms(audio_vae: torch.nn.Module) -> int:
    mode = os.environ.get("MING_AUDIOVAE_RMSNORM_P6", "hf")
    if mode == "hf":
        return 0
    if mode != "sglang":
        raise ValueError(f"unsupported MING_AUDIOVAE_RMSNORM_P6={mode!r}")
    from sglang_omni.vendor.sglang.layers import RMSNorm

    model = audio_vae.decoder.decoder
    targets = [(parent, name, child)
               for parent in model.modules()
               for name, child in parent.named_children()
               if isinstance(child, Qwen2RMSNorm)]
    if not targets:
        raise RuntimeError("P6 enabled but no Qwen2RMSNorm found in AudioVAE decoder")
    expected = 2 * len(model.layers) + 1
    if len(targets) != expected:
        raise RuntimeError(f"P6 norm count mismatch: {len(targets)} != {expected}")
    # Validate all before any mutation; fail closed rather than silently fallback.
    for _, _, old in targets:
        w = old.weight
        if w.device.type != "cuda" or w.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"P6 requires CUDA BF16/FP16 weights, got {w.device} {w.dtype}")
    for parent, name, old in targets:
        w = old.weight
        fused = RMSNorm(
            w.numel(), eps=float(old.variance_epsilon),
            cast_x_before_out_mul=True, weight_dtype=w.dtype,
        ).to(device=w.device)
        # Reuse the EXACT loaded parameter rather than constructing a new value.
        fused.weight = w
        setattr(parent, name, fused)
    logger.warning("P6 experimental Ming AudioVAE HF-semantics fused RMSNorm enabled: %d modules",len(targets))
    return len(targets)

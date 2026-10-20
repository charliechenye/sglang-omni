# SPDX-License-Identifier: Apache-2.0
"""Opt-in Fun-CosyVoice3 HiFT SineGen2 harmonic-multiplier caching.

The pinned CosyVoice SineGen2.forward constructs a nine-element float32
CPU tensor and transfers it to the device on EVERY HiFT call. A pageable
36-byte HtoD copy was correlated to a ~20ms host stream synchronization
in two controlled H200 traces. This helper leaves upstream CosyVoice
unmodified and patches only the live, inference-mode SineGen2 INSTANCE.

This patch is intentionally experimental and disabled by default.
"""
from __future__ import annotations

import inspect
from types import MethodType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cosyvoice.hifigan.generator import CausalHiFTGenerator


HARMONIC_BUFFER = "_sglang_cached_harmonic_multiplier"


def patch_hift_sinegen_cache(
    hift: "CausalHiFTGenerator", *, device: torch.device,
) -> None:
    """Cache fixed float32 harmonic weights and preserve SineGen2 eval math.

    Mirrors the pinned CosyVoice (074ca6d) SineGen2.forward except for its
    per-call FloatTensor(...).to(f0.device). Do not patch class/global imports.
    Fail closed on unknown HiFT layout, noncausal/training mode or re-patching.
    """
    from cosyvoice.hifigan.generator import SineGen2

    sine_gen = hift.m_source.l_sin_gen
    pinned_source = inspect.getsource(SineGen2.forward)
    if (
        "torch.FloatTensor([[range(1, self.harmonic_num + 2)]])"
        not in pinned_source
        or "self.sine_waves[:, :sine_waves.shape[1]]" not in pinned_source
    ):
        raise RuntimeError(
            "P3 CosyVoice SineGen2.forward source contract changed; "
            "review copied inference semantics before enabling the cache"
        )
    if type(sine_gen) is not SineGen2:
        raise RuntimeError(
            "P3 harmonic cache requires the exact pinned CosyVoice SineGen2 class"
        )
    if (
        sine_gen.training
        or sine_gen.causal is not True
        or sine_gen.flag_for_pulse is not False
        or sine_gen.harmonic_num != 8
        or sine_gen.dim != 9
    ):
        raise RuntimeError(
            "P3 harmonic cache requires eval-mode causal nine-harmonic SineGen2"
        )
    if hasattr(sine_gen, HARMONIC_BUFFER):
        raise RuntimeError("P3 SineGen2 harmonic cache already installed")
    device = torch.device(device)
    if (
        sine_gen.rand_ini.device != device
        or sine_gen.sine_waves.device != device
        or sine_gen.rand_ini.shape != (1, 9)
        or sine_gen.sine_waves.ndim != 3
        or sine_gen.sine_waves.shape[0] != 1
        or sine_gen.sine_waves.shape[2] != 9
    ):
        raise RuntimeError(
            "P3 requires HiFT SineGen2 persistent noise/phase arrays on the target device"
        )

    # Original FloatTensor is ALWAYS float32 even under BF16 autocast.
    # The nonpersistent registered buffer moves with the module if needed
    # without adding any key to existing HiFT checkpoint state_dicts.
    sine_gen.register_buffer(
        HARMONIC_BUFFER,
        torch.arange(1, 10, device=device, dtype=torch.float32).view(1, 1, 9),
        persistent=False,
    )

    def forward(self: SineGen2, f0: torch.Tensor):
        # Mathematically identical to the source's
        # torch.FloatTensor([[range(1, 10)]]).to(f0.device).
        fn = torch.multiply(f0, getattr(self, HARMONIC_BUFFER))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if self.training is False and self.causal is True:
            noise = (
                noise_amp
                * self.sine_waves[:, : sine_waves.shape[1]].to(sine_waves.device)
            )
        else:
            # Kept as a correctness guard for source-equivalent semantics,
            # although the experimental patch requires eval + causal.
            noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

    sine_gen.forward = MethodType(forward, sine_gen)

# SPDX-License-Identifier: Apache-2.0
"""P3 SineGen harmonic cache: numerical parity and fail-closed boundaries.

The pinned upstream CosyVoice SineGen2 normally allocates a 7.2M x 9
noise tensor in its constructor. These tests initialize a deliberately small
REAL SineGen2 instance (using its actual forward/_f02sine/_f02uv methods)
without allocating the giant production noise array.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.fun_cosyvoice3.hift_sinegen_cache import (
    HARMONIC_BUFFER,
    patch_hift_sinegen_cache,
)


def _real_sinegen(device: torch.device):
    generator = pytest.importorskip("cosyvoice.hifigan.generator")
    cls = generator.SineGen2
    # Call Module.__init__ but avoid expensive fixed 300-second
    # SineGen2.__init__ noise tensor; retain the real class/forward.
    sine = cls.__new__(cls)
    torch.nn.Module.__init__(sine)
    sine.sine_amp = 0.1
    sine.noise_std = 0.003
    sine.harmonic_num = 8
    sine.dim = 9
    sine.sampling_rate = 24000
    sine.voiced_threshold = 10
    sine.flag_for_pulse = False
    sine.upsample_scale = 10
    sine.causal = True
    sine.rand_ini = torch.linspace(0, 0.5, 9, device=device).reshape(1, 9)
    sine.rand_ini[:, 0] = 0
    sine.sine_waves = torch.linspace(
        0, 0.9, 1 * 80 * 9, device=device
    ).reshape(1, 80, 9)
    sine.eval()
    return sine


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_real_cosyvoice_sinegen_output_parity(device_kind: str) -> None:
    if device_kind == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA optional for CPU-only conformance runs")
    device = torch.device(device_kind)
    sine = _real_sinegen(device)
    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=sine))
    f0 = torch.linspace(0, 280, 80, device=device).view(1, 80, 1)

    with torch.inference_mode():
        original = sine(f0)
        state_before = sorted(sine.state_dict())
        patch_hift_sinegen_cache(hift, device=device)
        optimized = sine(f0)

    assert len(original) == len(optimized) == 3
    for before, after in zip(original, optimized, strict=True):
        # No new math or RNG is introduced by this cache.
        torch.testing.assert_close(after, before, atol=0, rtol=0)
    assert sorted(sine.state_dict()) == state_before
    assert HARMONIC_BUFFER not in sine.state_dict()
    assert HARMONIC_BUFFER in sine._buffers
    assert sine._buffers[HARMONIC_BUFFER].device == device
    assert sine._buffers[HARMONIC_BUFFER].dtype == torch.float32
    torch.testing.assert_close(
        sine._buffers[HARMONIC_BUFFER],
        torch.arange(1, 10, dtype=torch.float32, device=device).view(1, 1, 9),
        atol=0, rtol=0,
    )


def test_p3_flag_is_explicit_and_typed() -> None:
    assert FactoryArgs().enable_hift_sinegen_cache is None
    assert FactoryArgs(enable_hift_sinegen_cache=True).enable_hift_sinegen_cache
    assert FactoryArgs(enable_hift_sinegen_cache="false").enable_hift_sinegen_cache is False


def test_unknown_sinegen_and_duplicate_patches_fail_closed() -> None:
    torch_device = torch.device("cpu")
    bad = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=torch.nn.Identity()))
    with pytest.raises(RuntimeError, match="exact pinned"):
        patch_hift_sinegen_cache(bad, device=torch_device)

    sine = _real_sinegen(torch_device)
    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=sine))
    patch_hift_sinegen_cache(hift, device=torch_device)
    with pytest.raises(RuntimeError, match="already installed"):
        patch_hift_sinegen_cache(hift, device=torch_device)


def test_non_causal_or_wrong_device_fails_closed() -> None:
    dev = torch.device("cpu")
    sine = _real_sinegen(dev)
    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=sine))
    sine.causal = False
    with pytest.raises(RuntimeError, match="eval-mode causal"):
        patch_hift_sinegen_cache(hift, device=dev)
    sine.causal = True
    sine.train()
    with pytest.raises(RuntimeError, match="eval-mode causal"):
        patch_hift_sinegen_cache(hift, device=dev)
    sine.eval()
    sine.harmonic_num = 9
    with pytest.raises(RuntimeError, match="eval-mode causal"):
        patch_hift_sinegen_cache(hift, device=dev)


def test_sinegen_pinned_forward_keeps_expected_h2d_source() -> None:
    import inspect
    generator = pytest.importorskip("cosyvoice.hifigan.generator")
    source = inspect.getsource(generator.SineGen2.forward)
    # If upstream changes, review the duplicated inference-only forward.
    assert (
        "torch.FloatTensor([[range(1, self.harmonic_num + 2)]])"
        in source
    )
    assert "self.sine_waves[:, :sine_waves.shape[1]]" in source

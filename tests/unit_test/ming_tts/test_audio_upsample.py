from __future__ import annotations

from typing import Literal

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.ming_tts.audio_decode import (
    AudioVAEFixedStreamingStateBank,
    AudioVAEFixedStreamingTransition,
)
from sglang_omni.models.ming_tts.audio_upsample import (
    HAS_TRITON,
    upsample_linear_bth,
)


def make_upsample_transition(
    upsample_backend: Literal["native", "triton_bth"], cuda_device: torch.device
) -> AudioVAEFixedStreamingTransition:
    capacity = 1
    max_step_latents = 16
    hidden_size = 896
    transition = AudioVAEFixedStreamingTransition.__new__(
        AudioVAEFixedStreamingTransition
    )
    transition._capacity = capacity
    transition._max_step_latents = max_step_latents
    transition._device = cuda_device
    transition._input_dtype = torch.bfloat16
    transition.state = AudioVAEFixedStreamingStateBank(
        upsample_pending=torch.zeros(
            (capacity, max_step_latents, hidden_size),
            device=cuda_device,
            dtype=torch.bfloat16,
        ),
        upsample_pending_lengths=torch.zeros(
            capacity, device=cuda_device, dtype=torch.long
        ),
        upsample_left_anchor=torch.zeros(
            (capacity, 1, hidden_size), device=cuda_device, dtype=torch.bfloat16
        ),
        qwen_keys=torch.empty((0,), device=cuda_device),
        qwen_values=torch.empty((0,), device=cuda_device),
        qwen_positions=torch.empty((0,), device=cuda_device, dtype=torch.long),
        istft_audio_overlap=torch.empty((0,), device=cuda_device),
    )
    transition.upsampler = torch.nn.Upsample(
        scale_factor=4,
        mode="linear",
        align_corners=False,
        recompute_scale_factor=None,
    )
    transition.upsample_backend = upsample_backend
    transition.scale_factor = 4
    transition.max_frames = 2 * max_step_latents * transition.scale_factor
    transition.hidden_size = hidden_size
    return transition


@pytest.mark.parametrize(
    ("batch_size", "input_time_steps", "hidden_size"),
    [
        (1, 18, 896),
        (1, 34, 896),
        (2, 22, 896),
        (4, 34, 896),
        (8, 18, 896),
        (8, 34, 896),
        (1, 1, 896),
        (2, 2, 896),
        (4, 3, 896),
        (8, 3, 896),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and Triton",
)
def test_upsample_linear_bth_exactly_matches_pytorch(
    batch_size: int, input_time_steps: int, hidden_size: int
) -> None:
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    timeline = torch.randn(
        (batch_size, input_time_steps, hidden_size),
        device=cuda_device,
        dtype=torch.bfloat16,
    )
    expected_bth = F.interpolate(
        timeline.transpose(1, 2),
        scale_factor=4,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)

    upsampled_bth = upsample_linear_bth(timeline, scale_factor=4)

    assert upsampled_bth.shape == expected_bth.shape
    assert upsampled_bth.is_contiguous()
    assert upsampled_bth.stride() == (
        input_time_steps * 4 * hidden_size,
        hidden_size,
        1,
    )
    assert torch.equal(upsampled_bth, expected_bth)
    assert torch.equal(upsampled_bth[:, 0], timeline[:, 0])
    assert torch.equal(upsampled_bth[:, -1], timeline[:, -1])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and Triton",
)
def test_upsample_linear_bth_supports_strided_timeline_views() -> None:
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    timeline_storage = torch.randn(
        (2, 44, 896), device=cuda_device, dtype=torch.bfloat16
    )
    timeline = timeline_storage[:, ::2, :]
    assert not timeline.is_contiguous()

    expected_bth = F.interpolate(
        timeline.transpose(1, 2),
        scale_factor=4,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    upsampled_bth = upsample_linear_bth(timeline, scale_factor=4)

    assert upsampled_bth.is_contiguous()
    assert torch.equal(upsampled_bth, expected_bth)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and Triton",
)
def test_upsample_linear_bth_cuda_graph_replay_tracks_input_updates() -> None:
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    timeline = torch.randn((8, 34, 896), device=cuda_device, dtype=torch.bfloat16)
    upsample_linear_bth(timeline, scale_factor=4)
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        graph_output_bth = upsample_linear_bth(timeline, scale_factor=4)

    cuda_graph.replay()
    torch.cuda.synchronize(cuda_device)
    expected_bth = F.interpolate(
        timeline.transpose(1, 2),
        scale_factor=4,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    assert graph_output_bth.is_contiguous()
    assert torch.equal(graph_output_bth, expected_bth)

    timeline.add_(0.25)
    cuda_graph.replay()
    torch.cuda.synchronize(cuda_device)
    expected_bth = F.interpolate(
        timeline.transpose(1, 2),
        scale_factor=4,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    assert torch.equal(graph_output_bth, expected_bth)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and Triton",
)
def test_triton_upsampling_matches_native_across_streaming_chunks() -> None:
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    native_transition = make_upsample_transition("native", cuda_device)
    triton_transition = make_upsample_transition("triton_bth", cuda_device)
    expected_frame_lengths = (0, 32, 96)

    for current_length, terminal, expected_frame_length in zip(
        (8, 16, 8), (False, False, True), expected_frame_lengths, strict=True
    ):
        current = torch.zeros((1, 16, 896), device=cuda_device, dtype=torch.bfloat16)
        current[:, :current_length] = torch.randn(
            (1, current_length, 896), device=cuda_device, dtype=torch.bfloat16
        )
        current_lengths = torch.tensor([current_length], device=cuda_device)
        execution_mask = torch.ones((1,), device=cuda_device, dtype=torch.bool)
        terminal_mask = torch.tensor([terminal], device=cuda_device)
        native_frames, native_lengths, native_next_state = native_transition.upsample(
            current, current_lengths, execution_mask, terminal_mask
        )
        triton_frames, triton_lengths, triton_next_state = triton_transition.upsample(
            current, current_lengths, execution_mask, terminal_mask
        )

        assert native_lengths.tolist() == [expected_frame_length]
        assert torch.equal(triton_lengths, native_lengths)
        assert torch.equal(triton_frames, native_frames)
        for native_tensor, triton_tensor in zip(
            native_next_state, triton_next_state, strict=True
        ):
            assert torch.equal(triton_tensor, native_tensor)
        for transition, next_state in (
            (native_transition, native_next_state),
            (triton_transition, triton_next_state),
        ):
            (
                transition.state.upsample_pending,
                transition.state.upsample_pending_lengths,
                transition.state.upsample_left_anchor,
            ) = next_state


def test_upsample_linear_bth_rejects_unsupported_inputs() -> None:
    with pytest.raises(ValueError, match="BTH tensor"):
        upsample_linear_bth(torch.empty((1, 2)), scale_factor=4)
    with pytest.raises(ValueError, match="positive integer"):
        upsample_linear_bth(torch.empty((1, 2, 896)), scale_factor=0)
    with pytest.raises(ValueError, match="BF16"):
        upsample_linear_bth(torch.empty((1, 2, 896)), scale_factor=4)
    with pytest.raises(ValueError, match="CUDA tensor"):
        upsample_linear_bth(
            torch.empty((1, 2, 896), dtype=torch.bfloat16), scale_factor=4
        )

# SPDX-License-Identifier: Apache-2.0
"""CUDA parity tests for fused Drama 3 decode-state operations."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("triton")

from sglang_omni.models.fishaudio_drama3 import decode_state_kernels
from sglang_omni.models.fishaudio_drama3.decode_state import Drama3DecodeStatePool
from sglang_omni.models.fishaudio_drama3.model_inputs import Drama3DecodeBuffers
from sglang_omni.models.fishaudio_drama3.model_runner import Drama3ModelRunner
from sglang_omni.models.fishaudio_drama3.sampling import RAS_WINDOW_FRAMES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("row_count", [1, 8, 16, 32])
@pytest.mark.parametrize("arrangement", ["ordered", "permuted", "sparse"])
def test_fused_decode_state_matches_baseline(
    row_count: int, arrangement: str
) -> None:
    device = torch.device("cuda")
    frame_width = 5
    request_slot_count = 64
    generator = torch.Generator(device=device).manual_seed(20261029 + row_count)
    baseline_state = Drama3DecodeStatePool.allocate(
        request_slot_count=request_slot_count,
        frame_width=frame_width,
        device=device,
    )
    baseline_buffers = Drama3DecodeBuffers.allocate(
        row_count=row_count + 7,
        frame_width=frame_width,
        device=device,
    )
    baseline_state.frames.copy_(
        torch.randint(
            -100,
            100,
            baseline_state.frames.shape,
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
    )
    baseline_state.ras_windows.copy_(
        torch.randint(
            -5,
            100,
            baseline_state.ras_windows.shape,
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
    )
    baseline_state.temperature.copy_(
        torch.rand(
            baseline_state.temperature.shape, device=device, generator=generator
        )
    )
    baseline_state.top_k.copy_(
        torch.randint(
            1,
            128,
            baseline_state.top_k.shape,
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
    )
    baseline_state.top_p.copy_(
        torch.rand(baseline_state.top_p.shape, device=device, generator=generator)
    )
    for buffer_name, expected_dtype in (
        ("frames", torch.int64),
        ("ras_windows", torch.int64),
        ("temperature", torch.float32),
        ("top_k", torch.int64),
        ("top_p", torch.float32),
        ("sampled_frames", torch.int64),
    ):
        buffer_tensor = getattr(baseline_buffers, buffer_name)
        if expected_dtype == torch.int64:
            buffer_tensor.copy_(
                torch.randint(
                    -100,
                    100,
                    buffer_tensor.shape,
                    dtype=expected_dtype,
                    device=device,
                    generator=generator,
                )
            )
        else:
            buffer_tensor.copy_(
                torch.rand(buffer_tensor.shape, device=device, generator=generator)
            )
    candidate_state = Drama3DecodeStatePool(
        frames=baseline_state.frames.clone(),
        ras_windows=baseline_state.ras_windows.clone(),
        temperature=baseline_state.temperature.clone(),
        top_k=baseline_state.top_k.clone(),
        top_p=baseline_state.top_p.clone(),
        row_indices=baseline_state.row_indices.clone(),
    )
    candidate_buffers = Drama3DecodeBuffers(
        frames=baseline_buffers.frames.clone(),
        ras_windows=baseline_buffers.ras_windows.clone(),
        temperature=baseline_buffers.temperature.clone(),
        top_k=baseline_buffers.top_k.clone(),
        top_p=baseline_buffers.top_p.clone(),
        sampled_frames=baseline_buffers.sampled_frames.clone(),
    )
    if arrangement == "ordered":
        initial_rows = list(range(1, row_count + 1))
    elif arrangement == "permuted":
        initial_rows = (
            torch.randperm(request_slot_count, generator=generator, device=device)[
                :row_count
            ]
            .add_(1)
            .tolist()
        )
    else:
        initial_rows = list(range(1, row_count * 2, 2))

    def runner_for(
        state_pool: Drama3DecodeStatePool,
        buffers: Drama3DecodeBuffers,
        fused_decode_state: bool,
    ) -> Drama3ModelRunner:
        model_runner = object.__new__(Drama3ModelRunner)
        model_runner.state_pool = state_pool
        model_runner.decode_buffers = buffers
        model_runner.row_launch_key = None
        model_runner.row_upload_buffers = []
        model_runner.row_upload_slot = 0
        model_runner.fused_decode_state = fused_decode_state
        model_runner.decode_state_kernels = (
            decode_state_kernels if fused_decode_state else None
        )
        return model_runner

    baseline_runner = runner_for(baseline_state, baseline_buffers, False)
    candidate_runner = runner_for(candidate_state, candidate_buffers, True)
    row_sets = [initial_rows, list(reversed(initial_rows))]
    state_names = ("frames", "ras_windows", "temperature", "top_k", "top_p")
    buffer_names = (
        "frames",
        "ras_windows",
        "temperature",
        "top_k",
        "top_p",
        "sampled_frames",
    )
    padded_buffers_before = tuple(
        getattr(candidate_buffers, buffer_name)[row_count:].clone()
        for buffer_name in buffer_names
    )
    for step_index, row_slots in enumerate(row_sets):
        if step_index:
            reused_slot = initial_rows[-1]
            replacement_frames = torch.arange(
                frame_width, dtype=torch.int64, device=device
            ) + 7000
            replacement_ras = torch.arange(
                RAS_WINDOW_FRAMES, dtype=torch.int64, device=device
            ) - 900
            for state_pool in (baseline_state, candidate_state):
                state_pool.frames[reused_slot].copy_(replacement_frames)
                state_pool.ras_windows[reused_slot].copy_(replacement_ras)
                state_pool.temperature[reused_slot] = 0.75
                state_pool.top_k[reused_slot] = 9
                state_pool.top_p[reused_slot] = 0.85
        else:
            pass
        with patch.object(baseline_runner, "state_rows", return_value=row_slots):
            baseline_runner.gather_decode_rows([])
        with patch.object(candidate_runner, "state_rows", return_value=row_slots):
            candidate_runner.gather_decode_rows([])
        for buffer_name, expected_dtype in (
            ("frames", torch.int64),
            ("ras_windows", torch.int64),
            ("temperature", torch.float32),
            ("top_k", torch.int64),
            ("top_p", torch.float32),
        ):
            baseline_buffer = getattr(baseline_buffers, buffer_name)
            candidate_buffer = getattr(candidate_buffers, buffer_name)
            assert baseline_buffer.dtype == expected_dtype
            assert candidate_buffer.dtype == expected_dtype
            assert torch.equal(baseline_buffer, candidate_buffer)
        for buffer_index, buffer_name in enumerate(buffer_names):
            assert torch.equal(
                getattr(candidate_buffers, buffer_name)[row_count:],
                padded_buffers_before[buffer_index],
            )

        active_state_rows = torch.tensor(row_slots, dtype=torch.int64, device=device)
        untouched_state_mask = torch.ones(
            baseline_state.frames.shape[0], dtype=torch.bool, device=device
        )
        untouched_state_mask[active_state_rows] = False
        untouched_state_before = tuple(
            getattr(candidate_state, state_name)[untouched_state_mask].clone()
            for state_name in state_names
        )

        sampled_frames = (
            torch.arange(
                row_count * frame_width, dtype=torch.int64, device=device
            ).reshape(row_count, frame_width)
            + 1000 * (step_index + 1)
        )
        baseline_buffers.sampled_frames[:row_count].copy_(sampled_frames)
        candidate_buffers.sampled_frames[:row_count].copy_(sampled_frames)
        old_ras_windows = baseline_state.ras_windows[
            torch.tensor(row_slots, dtype=torch.int64, device=device)
        ].clone()
        baseline_snapshot = baseline_runner.publish_decode_frames(row_count)
        candidate_snapshot = candidate_runner.publish_decode_frames(row_count)
        assert baseline_snapshot.shape == (row_count, frame_width)
        assert candidate_snapshot.shape == (row_count, frame_width)
        assert baseline_snapshot.dtype == torch.int64
        assert candidate_snapshot.dtype == torch.int64
        assert torch.equal(baseline_snapshot, candidate_snapshot)
        assert (
            baseline_snapshot.untyped_storage().data_ptr()
            != baseline_buffers.sampled_frames.untyped_storage().data_ptr()
        )
        assert (
            candidate_snapshot.untyped_storage().data_ptr()
            != candidate_buffers.sampled_frames.untyped_storage().data_ptr()
        )
        assert (
            candidate_snapshot.untyped_storage().data_ptr()
            != candidate_state.frames.untyped_storage().data_ptr()
        )
        expected_ras_windows = torch.cat(
            (old_ras_windows[:, 1:], sampled_frames[:, :1]), dim=1
        )
        assert torch.equal(
            candidate_state.ras_windows[
                torch.tensor(row_slots, dtype=torch.int64, device=device)
            ],
            expected_ras_windows,
        )

        baseline_snapshot_copy = baseline_snapshot.clone()
        candidate_snapshot_copy = candidate_snapshot.clone()
        baseline_buffers.sampled_frames[:row_count].fill_(-1)
        candidate_buffers.sampled_frames[:row_count].fill_(-1)
        assert torch.equal(baseline_snapshot, baseline_snapshot_copy)
        assert torch.equal(candidate_snapshot, candidate_snapshot_copy)
        for state_index, state_name in enumerate(state_names):
            assert torch.equal(
                getattr(baseline_state, state_name),
                getattr(candidate_state, state_name),
            )
            assert torch.equal(
                getattr(candidate_state, state_name)[untouched_state_mask],
                untouched_state_before[state_index],
            )
        for buffer_name in buffer_names:
            assert torch.equal(
                getattr(baseline_buffers, buffer_name),
                getattr(candidate_buffers, buffer_name),
            )
        assert torch.equal(
            baseline_buffers.sampled_frames, candidate_buffers.sampled_frames
        )
        assert torch.equal(baseline_state.row_indices, candidate_state.row_indices)

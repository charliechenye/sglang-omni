# SPDX-License-Identifier: Apache-2.0
"""Fused Triton operations for Drama 3 decode state."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang_omni.models.fishaudio_drama3.decode_state import Drama3DecodeStatePool
from sglang_omni.models.fishaudio_drama3.model_inputs import Drama3DecodeBuffers
from sglang_omni.models.fishaudio_drama3.sampling import RAS_WINDOW_FRAMES


@triton.jit
def _gather(
    pool_frames,
    pool_ras,
    pool_temperature,
    pool_top_k,
    pool_top_p,
    row_indices,
    output_frames,
    output_ras,
    output_temperature,
    output_top_k,
    output_top_p,
    FRAME_WIDTH: tl.constexpr,
    FRAME_BLOCK: tl.constexpr,
    RAS_WINDOW_FRAMES: tl.constexpr,
):
    batch_row = tl.program_id(0)
    slot = tl.load(row_indices + batch_row)
    frame_column = tl.arange(0, FRAME_BLOCK)
    ras_column = tl.arange(0, RAS_WINDOW_FRAMES)
    frame = tl.load(
        pool_frames + slot * FRAME_WIDTH + frame_column,
        mask=frame_column < FRAME_WIDTH,
        other=0,
    )
    ras = tl.load(pool_ras + slot * RAS_WINDOW_FRAMES + ras_column)
    temperature = tl.load(pool_temperature + slot)
    top_k = tl.load(pool_top_k + slot)
    top_p = tl.load(pool_top_p + slot)
    tl.store(
        output_frames + batch_row * FRAME_WIDTH + frame_column,
        frame,
        mask=frame_column < FRAME_WIDTH,
    )
    tl.store(output_ras + batch_row * RAS_WINDOW_FRAMES + ras_column, ras)
    tl.store(output_temperature + batch_row, temperature)
    tl.store(output_top_k + batch_row, top_k)
    tl.store(output_top_p + batch_row, top_p)


@triton.jit
def _publish(
    pool_frames,
    pool_ras,
    row_indices,
    sampled_frames,
    snapshot,
    FRAME_WIDTH: tl.constexpr,
    FRAME_BLOCK: tl.constexpr,
    RAS_WINDOW_FRAMES: tl.constexpr,
):
    batch_row = tl.program_id(0)
    slot = tl.load(row_indices + batch_row)
    frame_column = tl.arange(0, FRAME_BLOCK)
    ras_column = tl.arange(0, RAS_WINDOW_FRAMES)
    sampled = tl.load(
        sampled_frames + batch_row * FRAME_WIDTH + frame_column,
        mask=frame_column < FRAME_WIDTH,
        other=0,
    )
    shifted_ras = tl.load(
        pool_ras + slot * RAS_WINDOW_FRAMES + ras_column + 1,
        mask=ras_column < RAS_WINDOW_FRAMES - 1,
        other=0,
    )
    first_frame_token = tl.load(sampled_frames + batch_row * FRAME_WIDTH)
    tl.store(
        snapshot + batch_row * FRAME_WIDTH + frame_column,
        sampled,
        mask=frame_column < FRAME_WIDTH,
    )
    tl.store(
        pool_frames + slot * FRAME_WIDTH + frame_column,
        sampled,
        mask=frame_column < FRAME_WIDTH,
    )
    tl.store(
        pool_ras + slot * RAS_WINDOW_FRAMES + ras_column,
        tl.where(ras_column == RAS_WINDOW_FRAMES - 1, first_frame_token, shifted_ras),
    )


def gather(
    state_pool: Drama3DecodeStatePool,
    decode_buffers: Drama3DecodeBuffers,
    row_count: int,
) -> None:
    frame_width = state_pool.frames.shape[1]
    _gather[(row_count,)](
        state_pool.frames,
        state_pool.ras_windows,
        state_pool.temperature,
        state_pool.top_k,
        state_pool.top_p,
        state_pool.row_indices,
        decode_buffers.frames,
        decode_buffers.ras_windows,
        decode_buffers.temperature,
        decode_buffers.top_k,
        decode_buffers.top_p,
        frame_width,
        triton.next_power_of_2(frame_width),
        RAS_WINDOW_FRAMES,
        num_warps=4,
    )


def publish(
    state_pool: Drama3DecodeStatePool,
    decode_buffers: Drama3DecodeBuffers,
    row_count: int,
) -> torch.Tensor:
    snapshot = torch.empty_like(decode_buffers.sampled_frames[:row_count])
    frame_width = state_pool.frames.shape[1]
    _publish[(row_count,)](
        state_pool.frames,
        state_pool.ras_windows,
        state_pool.row_indices,
        decode_buffers.sampled_frames,
        snapshot,
        frame_width,
        triton.next_power_of_2(frame_width),
        RAS_WINDOW_FRAMES,
        num_warps=4,
    )
    return snapshot

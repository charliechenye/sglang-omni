# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 DiT on a packed sequence: the rows of a Flow batch concatenated
along the sequence for every per token module, attention within each row."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

logger = logging.getLogger(__name__)

# note (ratish, chenyang): a row's chunks share a key prefix, so FA3 pages are one frame.
FA3_PAGE_SIZE = 1
FA3_DTYPES = (torch.float16, torch.bfloat16)


@torch.library.custom_op(
    "sglang_omni::cosyvoice3_native_mish",
    mutates_args=(),
    device_types="cuda",
)
def cosyvoice3_native_mish(x: torch.Tensor) -> torch.Tensor:
    return F.mish(x)


@cosyvoice3_native_mish.register_fake
def _fake_cosyvoice3_native_mish(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "sglang_omni::cosyvoice3_native_layer_norm",
    mutates_args=(),
    device_types="cuda",
)
def cosyvoice3_native_layer_norm(
    x: torch.Tensor,
    normalized_size: int,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x.float(), (normalized_size,), None, None, eps)


@cosyvoice3_native_layer_norm.register_fake
def _fake_cosyvoice3_native_layer_norm(
    x: torch.Tensor,
    normalized_size: int,
    eps: float,
) -> torch.Tensor:
    del normalized_size, eps
    return torch.empty_like(x, dtype=torch.float32)


@torch.library.custom_op(
    "sglang_omni::cosyvoice3_ragged_attention",
    mutates_args=(),
    device_types="cuda",
)
def cosyvoice3_ragged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        causal=False,
    )


@cosyvoice3_ragged_attention.register_fake
def _fake_cosyvoice3_ragged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    del k_cache, v_cache, cache_seqlens, page_table, cu_seqlens_q, max_seqlen_q
    return torch.empty_like(q)


@dataclass(frozen=True)
class PackedRows:
    lengths: tuple[int, ...]
    starts_host: torch.Tensor
    row_ids: torch.Tensor
    positions: torch.Tensor

    @property
    def total(self) -> int:
        return sum(self.lengths)

    @property
    def width(self) -> int:
        return max(self.lengths)


def pack_rows(lengths: Sequence[int], device: torch.device) -> PackedRows:
    lengths = tuple(int(length) for length in lengths)
    starts_host = F.pad(torch.tensor(lengths, dtype=torch.int64).cumsum(0), (1, 0))
    starts = starts_host.to(device)
    total = int(starts_host[-1])
    row_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(lengths, dtype=torch.int64, device=device),
        output_size=total,
    )
    positions = torch.arange(total, device=device) - starts[row_ids]
    return PackedRows(
        lengths=lengths,
        starts_host=starts_host.to(torch.int32),
        row_ids=row_ids,
        positions=positions,
    )


def gather_rows(padded: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    """(rows, width, channels) -> (1, total, channels), each row's first
    length frames in row order."""
    width = padded.shape[1]
    flat = padded.reshape(padded.shape[0] * width, padded.shape[2])
    return flat[rows.row_ids * width + rows.positions].unsqueeze(0)


def scatter_rows(packed: torch.Tensor, rows: PackedRows, width: int) -> torch.Tensor:
    """(1, total, channels) -> (rows, width, channels), zero past each row's
    length."""
    channels = packed.shape[2]
    row_count = rows.starts_host.shape[0] - 1
    flat = packed.new_zeros(row_count * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(row_count, width, channels)


def chunk_causal_mask(
    length: int, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """(length, length) bool: a frame attends every frame of its chunk and of
    the chunks before it, CosyVoice's subsequent_chunk_mask."""
    position = torch.arange(length, device=device)
    chunk_end = (position // chunk_size + 1) * chunk_size
    return position.unsqueeze(0) < chunk_end.unsqueeze(1)


def chunk_segments(
    lengths: Sequence[int], chunk_size: int | None
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """One query segment per (row, chunk), reading that row's frames
    [0, chunk end). Without a chunk size a row is one segment."""
    segment_rows: list[int] = []
    segment_ends: list[int] = []
    offsets: list[int] = [0]
    for row, length in enumerate(lengths):
        span = length if chunk_size is None else chunk_size
        frame = 0
        while frame < length:
            end = min((frame // span + 1) * span, length)
            segment_rows.append(row)
            segment_ends.append(end)
            offsets.append(offsets[-1] + end - frame)
            frame = end
    return tuple(segment_rows), tuple(segment_ends), tuple(offsets)


class RowAttention:
    """Attention within each row of a packed sequence, computed as the padded
    DiT computes it: one SDPA call over the rows scattered to the padded layout,
    under the row's key mask and, for hops, the chunk causal mask, built once
    per Flow call.
    """

    def __init__(self, rows: PackedRows, *, chunk_size: int | None, heads: int) -> None:
        self.rows = rows
        self.heads = heads
        width = rows.width
        device = rows.row_ids.device
        lengths = torch.tensor(rows.lengths, device=device)
        keys = torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        if chunk_size is None:
            mask = keys.unsqueeze(1).expand(-1, width, -1)
        else:
            mask = keys.unsqueeze(1) & chunk_causal_mask(width, chunk_size, device)
        self.mask = mask.unsqueeze(1)

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, total, heads * head_dim). Returns the same
        shape."""
        row_count, width = len(self.rows.lengths), self.rows.width
        padded = scatter_rows(torch.cat((query, key, value), dim=-1), self.rows, width)
        query, key, value = (
            part.view(row_count, width, self.heads, -1).transpose(1, 2)
            for part in padded.chunk(3, dim=-1)
        )
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask)
        return gather_rows(out.transpose(1, 2).reshape(row_count, width, -1), self.rows)


class RaggedRowAttention:
    """Row attention on the packed sequence via FA3 paged KV, no pad-to-widest."""

    def __init__(
        self,
        rows: PackedRows,
        *,
        chunk_size: int | None,
        heads: int,
        head_dim: int,
    ) -> None:
        self.heads = heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        device = rows.row_ids.device
        segment_rows, segment_ends, offsets = chunk_segments(rows.lengths, chunk_size)
        self.cache_seqlens = torch.tensor(
            segment_ends, dtype=torch.int32, device=device
        )
        self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.max_seqlen_q = max(end - start for start, end in pairwise(offsets))
        starts = rows.starts_host[list(segment_rows)].to(device)
        # note (ratish): FA3 page ids must land inside the packed keys; pad with page 0.
        page = torch.arange(max(segment_ends), dtype=torch.int32, device=device)
        self.page_table = torch.where(
            page.unsqueeze(0) < self.cache_seqlens.unsqueeze(1),
            starts.unsqueeze(1) + page.unsqueeze(0),
            0,
        )

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, total, heads * head_dim). Returns the same
        shape."""
        page_shape = (-1, FA3_PAGE_SIZE, self.heads, self.head_dim)
        out = flash_attn_with_kvcache(
            q=query[0].reshape(-1, self.heads, self.head_dim),
            k_cache=key[0].reshape(page_shape),
            v_cache=value[0].reshape(page_shape),
            cache_seqlens=self.cache_seqlens,
            page_table=self.page_table,
            cu_seqlens_q=self.cu_seqlens_q,
            max_seqlen_q=self.max_seqlen_q,
            causal=False,
        )
        return out.reshape(1, -1, self.heads * self.head_dim)


def mark_compile_dynamic(rows: PackedRows, attention: RaggedRowAttention) -> None:
    torch._dynamo.mark_dynamic(attention.page_table, (0, 1))
    torch._dynamo.mark_dynamic(attention.cu_seqlens_q, 0)
    torch._dynamo.mark_dynamic(attention.cache_seqlens, 0)
    torch._dynamo.mark_dynamic(rows.starts_host, 0)
    torch._dynamo.mark_dynamic(rows.row_ids, 0)
    torch._dynamo.mark_dynamic(rows.positions, 0)


def layer_norm_eager_numerics(
    layer_norm: torch.nn.LayerNorm,
    x: torch.Tensor,
) -> torch.Tensor:
    return cosyvoice3_native_layer_norm(
        x,
        int(layer_norm.normalized_shape[0]),
        float(layer_norm.eps),
    )


def attn_norm_eager_numerics(
    block: torch.nn.Module,
    h: torch.Tensor,
    time_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    modulation = block.attn_norm.linear(block.attn_norm.silu(time_embedding))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
        modulation, 6, dim=1
    )
    normalized = layer_norm_eager_numerics(block.attn_norm.norm, h)
    normalized = normalized * (1 + scale_msa[:, None]) + shift_msa[:, None]
    return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


def final_norm_eager_numerics(
    norm_out: torch.nn.Module,
    h: torch.Tensor,
    time_embedding: torch.Tensor,
) -> torch.Tensor:
    modulation = norm_out.linear(norm_out.silu(time_embedding))
    scale, shift = torch.chunk(modulation, 2, dim=1)
    normalized = layer_norm_eager_numerics(norm_out.norm, h)
    return normalized * (1 + scale)[:, None, :] + shift[:, None, :]


PackedRowAttention = RowAttention | RaggedRowAttention


class PackedDiT:
    """DiT.forward over a packed sequence with the same modules in the same
    order; conv pos-emb stays padded, attention is ragged on FA3 half-precision
    CUDA and padded elsewhere.
    """

    def __init__(self, dit: torch.nn.Module, *, device: str | torch.device) -> None:
        self.dit = dit
        device = torch.device(device)
        self.is_ragged = device.type == "cuda" and _is_fa3_supported()
        self.compiled_causal_forward = None
        logger.info(
            "Fun-CosyVoice3 Flow row attention on %s: %s",
            device,
            "ragged FA3" if self.is_ragged else "padded SDPA",
        )

    @property
    def chunk_size(self) -> int:
        return int(self.dit.static_chunk_size)

    def row_attention(
        self, rows: PackedRows, *, streaming: bool, dtype: torch.dtype
    ) -> PackedRowAttention:
        attention = self.dit.transformer_blocks[0].attn
        chunk_size = self.chunk_size if streaming else None
        if self.is_ragged and dtype in FA3_DTYPES:
            ragged = RaggedRowAttention(
                rows,
                chunk_size=chunk_size,
                heads=attention.heads,
                head_dim=attention.inner_dim // attention.heads,
            )
            if streaming and self.compiled_causal_forward is not None:
                mark_compile_dynamic(rows, ragged)
            return ragged
        else:
            pass
        return RowAttention(rows, chunk_size=chunk_size, heads=attention.heads)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: PackedRowAttention,
    ) -> torch.Tensor:
        """x, mu, cond, spks: (1, total, channels); t: (1,). Returns
        (1, total, out_channels)."""
        if (
            self.compiled_causal_forward is not None
            and isinstance(attention, RaggedRowAttention)
            and attention.chunk_size is not None
        ):
            return self.compiled_causal_forward(
                x, mu, spks, cond, t, rows, attention
            )
        return self.forward_eager(x, mu, spks, cond, t, rows, attention)

    def forward_eager(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: PackedRowAttention,
    ) -> torch.Tensor:
        dit = self.dit
        t = dit.time_embed(t)
        h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
        h = self.conv_pos_embed(h, rows) + h
        rope = self.rope(rows)
        residual = h
        for block in dit.transformer_blocks:
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(h, emb=t)
            h = h + gate_msa.unsqueeze(1) * self.attend(
                block.attn, norm, rope, attention
            )
            ff_norm = block.ff_norm(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
        if dit.long_skip_connection is not None:
            h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
        else:
            pass
        h = dit.norm_out(h, t)
        return dit.proj_out(h)

    def enable_causal_torch_compile(self) -> None:
        if not self.is_ragged:
            raise RuntimeError("PackedDiT causal torch.compile requires ragged FA3")
        self.compiled_causal_forward = torch.compile(
            self.forward_ragged_causal,
            backend="inductor",
            dynamic=True,
            fullgraph=True,
            options={"emulate_precision_casts": True},
        )

    def disable_causal_torch_compile(self) -> None:
        self.compiled_causal_forward = None

    def forward_ragged_causal(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: RaggedRowAttention,
    ) -> torch.Tensor:
        dit = self.dit
        time_embedding = dit.time_embed(t)
        h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
        width = attention.page_table.shape[1]
        h = self.conv_pos_embed_ragged_compiled(h, rows, width) + h
        rope = self.rope_ragged_compiled(rows, width)
        residual = h
        for block in dit.transformer_blocks:
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                attn_norm_eager_numerics(block, h, time_embedding)
            )
            h = h + gate_msa.unsqueeze(1) * self.attend_ragged_compiled(
                block.attn, norm, rope, attention
            )
            ff_norm = layer_norm_eager_numerics(block.ff_norm, h)
            ff_norm = ff_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
        if dit.long_skip_connection is not None:
            h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
        else:
            pass
        h = final_norm_eager_numerics(dit.norm_out, h, time_embedding)
        return dit.proj_out(h)

    def conv_pos_embed_ragged_compiled(
        self,
        h: torch.Tensor,
        rows: PackedRows,
        width: int,
    ) -> torch.Tensor:
        row_count = rows.starts_host.shape[0] - 1
        padded = scatter_rows(h, rows, width)
        module = self.dit.input_embed.conv_pos_embed
        embedded = padded.permute(0, 2, 1)
        embedded = F.pad(embedded, (module.kernel_size - 1, 0, 0, 0))
        embedded = cosyvoice3_native_mish(module.conv1[0](embedded))
        embedded = F.pad(embedded, (module.kernel_size - 1, 0, 0, 0))
        embedded = cosyvoice3_native_mish(module.conv2[0](embedded))
        embedded = embedded.permute(0, 2, 1)
        assert embedded.shape[0] == row_count
        return gather_rows(embedded, rows)

    def rope_ragged_compiled(
        self,
        rows: PackedRows,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs, scale = self.dit.rotary_embed.forward_from_seq_len(width)
        assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
        freqs = freqs[:, rows.positions]
        return freqs.cos(), freqs.sin()

    @staticmethod
    def attend_ragged_compiled(
        attn: torch.nn.Module,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        attention: RaggedRowAttention,
    ) -> torch.Tensor:
        x = x.to(attn.to_q.weight.dtype)
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)
        rotate_in_place(query, *rope)
        rotate_in_place(key, *rope)
        page_shape = (-1, FA3_PAGE_SIZE, attention.heads, attention.head_dim)
        out = cosyvoice3_ragged_attention(
            query[0].reshape(-1, attention.heads, attention.head_dim),
            key[0].reshape(page_shape),
            value[0].reshape(page_shape),
            attention.cache_seqlens,
            attention.page_table,
            attention.cu_seqlens_q,
            attention.max_seqlen_q,
        ).reshape(1, -1, attention.heads * attention.head_dim)
        return attn.to_out[1](attn.to_out[0](out.to(query.dtype)))

    def conv_pos_embed(self, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
        padded = scatter_rows(h, rows, rows.width)
        return gather_rows(self.dit.input_embed.conv_pos_embed(padded), rows)

    def rope(self, rows: PackedRows) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin, (1, total, rotary dims) each, in float32."""
        freqs, scale = self.dit.rotary_embed.forward_from_seq_len(rows.width)
        assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
        freqs = freqs[:, rows.positions]
        return freqs.cos(), freqs.sin()

    @staticmethod
    def attend(
        attn: torch.nn.Module,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        attention: PackedRowAttention,
    ) -> torch.Tensor:
        # note (ratish): under autocast to_q, to_k and to_v would each cast the
        # float32 norm output again.
        x = x.to(attn.to_q.weight.dtype)
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)
        rotate_in_place(query, *rope)
        rotate_in_place(key, *rope)
        out = attention(query, key, value).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](out))


def rotate_in_place(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """x: (1, total, heads * head_dim). Interleaved RoPE in float32 on the
    rotary dims, rounded back into x."""
    # note (ratish): the DiT rotates only the first rotary dims of the
    # flattened heads, so the rest of x is never copied.
    rotary = x[..., : cos.shape[-1]]
    half = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), dim=-1).flatten(-2)
    rotary.copy_(rotary * cos + half * sin)


def solve_flow_euler_packed(
    estimator: Any,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    rows: PackedRows,
    *,
    cfg_rate: float,
    streaming: bool,
) -> torch.Tensor:
    """Euler steps over a packed sequence with classifier free guidance: the
    conditional rows and their unconditional twins share one DiT call."""
    total = noise.shape[1]
    twin_rows = pack_rows(rows.lengths * 2, noise.device)
    attention = estimator.row_attention(
        twin_rows, streaming=streaming, dtype=spks.dtype
    )
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    spks_cfg = spks_cfg[twin_rows.row_ids].unsqueeze(0)
    flow_time = torch.zeros(1, device=noise.device, dtype=spks.dtype)
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(1, len(time_span)):
        flow_time[:] = t
        vector_field = estimator.forward(
            torch.cat((x, x), dim=1),
            mu_cfg,
            spks_cfg,
            cond_cfg,
            flow_time,
            twin_rows,
            attention,
        )
        conditional = vector_field[:, :total]
        unconditional = vector_field[:, total:]
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if step < len(time_span) - 1:
            dt = time_span[step + 1] - t
        else:
            pass
    return x.float()

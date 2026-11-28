# SPDX-License-Identifier: Apache-2.0
"""Pinned P16 Chained K3 draft weights and BF16 inference kernels."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None

P16_DRAFT_SHA256 = "ea97e6eec7c2d51d32fc6dc4c80e36dfe0aa2da3ed1eb526b4e5bc99002d2d9f"
P16_MANIFEST_SHA256 = "00c3029c70677459095ee402b5af1612004cca1629e1bcb52b644b2faf7f5ca0"
P16_DRAFT_WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "hidden_scale": (),
    "embed_scale": (),
    "in_proj.weight": (1024, 2063),
    "in_proj.bias": (1024,),
    "out_proj.weight": (1024, 1024),
    "out_proj.bias": (1024,),
}


@dataclass(frozen=True)
class P16DraftCheckpoint:
    """Validated CPU tensors from the immutable P13 D-w1024-600k artifact."""

    weights: dict[str, torch.Tensor]
    sha256: str


@dataclass
class P16DraftRuntime:
    """P14 BF16 LUT and scratch used by the fixed chained draft path."""

    hidden_weight: torch.Tensor
    output_weight: torch.Tensor
    output_bias: torch.Tensor
    embedding_depth_lut: torch.Tensor
    activation: torch.Tensor
    output: torch.Tensor
    output_alternate: torch.Tensor

    @classmethod
    @torch.no_grad()
    def create(
        cls,
        checkpoint: P16DraftCheckpoint,
        projected_codec_tables: torch.Tensor,
        device: torch.device,
    ) -> P16DraftRuntime:
        if projected_codec_tables.shape != (14, 2048, 1024):
            raise ValueError(
                "P16 requires fourteen projected 2048 by 1024 codec tables"
            )
        elif projected_codec_tables.dtype is not torch.bfloat16:
            raise ValueError("P16 projected codec tables must use BF16")
        elif projected_codec_tables.device != device:
            raise ValueError("P16 draft tensors and codec tables must share a device")
        else:
            pass

        # P14/P15A first materialize the checkpoint in BF16, then fold the
        # rounded parameters in FP32. Folding the original FP32 values changes
        # the draft distribution and therefore the acceptance trajectory.
        weights = {
            name: tensor.to(device=device, dtype=torch.bfloat16).float()
            for name, tensor in checkpoint.weights.items()
        }
        hidden_scale = weights["hidden_scale"]
        embedding_scale = weights["embed_scale"]
        if float(hidden_scale.item()) <= 0.0 or float(embedding_scale.item()) <= 0.0:
            raise ValueError("P16 normalization scales must be positive")
        else:
            pass
        input_weight = weights["in_proj.weight"]
        input_bias = weights["in_proj.bias"]
        input_hidden_weight = (
            (input_weight[:, :1024] / hidden_scale).to(torch.bfloat16).contiguous()
        )
        input_embedding_weight = input_weight[:, 1024:2048].contiguous()
        input_depth_weight = input_weight[:, 2048:].contiguous()
        embedding_depth_lut = (
            torch.stack(
                [
                    (
                        F.linear(
                            (projected_codec_tables[depth] / embedding_scale).float(),
                            input_embedding_weight,
                            input_bias,
                        )
                        + input_depth_weight[:, depth + 1]
                    ).to(torch.bfloat16)
                    for depth in range(14)
                ]
            )
            .reshape(14 * 2048, 1024)
            .contiguous()
        )
        output_weight = (
            (weights["out_proj.weight"] * hidden_scale).to(torch.bfloat16).contiguous()
        )
        output_bias = (
            (weights["out_proj.bias"] * hidden_scale).to(torch.bfloat16).contiguous()
        )
        draft_tensors = (
            input_hidden_weight,
            output_weight,
            output_bias,
            embedding_depth_lut,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in draft_tensors):
            raise ValueError("P16 transformed draft tensors contain nonfinite values")
        else:
            pass
        return cls(
            hidden_weight=input_hidden_weight,
            output_weight=output_weight,
            output_bias=output_bias,
            embedding_depth_lut=embedding_depth_lut,
            activation=torch.empty((1, 1024), device=device, dtype=torch.bfloat16),
            output=torch.empty((1, 1024), device=device, dtype=torch.bfloat16),
            output_alternate=torch.empty(
                (1, 1024), device=device, dtype=torch.bfloat16
            ),
        )

    @torch.no_grad()
    def step(
        self,
        hidden_state: torch.Tensor,
        token_id: torch.Tensor,
        depth: int,
        *,
        activation_output: torch.Tensor | None = None,
        hidden_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth < 1 or depth > 14:
            raise ValueError("P16 draft depth must be in [1, 14]")
        elif hidden_state.shape != (1, 1024) or token_id.shape != (1,):
            raise ValueError("P16 draft inputs must be one hidden row and one token")
        elif (
            hidden_state.dtype is not torch.bfloat16 or token_id.dtype is not torch.long
        ):
            raise ValueError(
                "P16 draft inputs must be BF16 hidden state and int64 token"
            )
        elif (
            hidden_state.device != self.hidden_weight.device
            or token_id.device != hidden_state.device
        ):
            raise ValueError("P16 draft inputs must share the draft device")
        elif triton is None or tl is None:
            raise RuntimeError("P16 draft execution requires Triton")
        else:
            pass
        activation = self.activation if activation_output is None else activation_output
        output = self.output if hidden_output is None else hidden_output
        if activation.shape != (1, 1024) or output.shape != (1, 1024):
            raise ValueError("P16 draft output buffers have an invalid shape")
        elif (
            activation.dtype is not torch.bfloat16 or output.dtype is not torch.bfloat16
        ):
            raise ValueError("P16 draft output buffers must use BF16")
        elif (
            activation.device != hidden_state.device
            or output.device != hidden_state.device
        ):
            raise ValueError("P16 draft output buffers must share the input device")
        else:
            pass
        input_projection[(1024,)](
            hidden_state,
            token_id,
            self.embedding_depth_lut,
            self.hidden_weight,
            activation,
            DEPTH=depth,
            WIDTH=1024,
            BLOCK_ROWS=1,
            num_warps=4,
        )
        output_projection[(256,)](
            hidden_state,
            activation,
            self.output_weight,
            self.output_bias,
            output,
            WIDTH=1024,
            OUTPUTS_PER_CTA=4,
            num_warps=4,
        )
        return output


def load_p16_draft_checkpoint(
    checkpoint_path: str | Path,
    expected_sha256: str,
) -> P16DraftCheckpoint:
    """Load the pinned P13 draft with the restricted torch weights-only loader."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"P16 draft checkpoint does not exist: {path}")
    elif not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("P16 draft SHA256 must be 64 lowercase hexadecimal characters")
    elif expected_sha256 != P16_DRAFT_SHA256:
        raise ValueError("P16 only accepts the pinned D-w1024-600k draft SHA256")
    else:
        pass
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while block := checkpoint_file.read(1024 * 1024):
            digest.update(block)
        else:
            pass
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("P16 draft checkpoint SHA256 mismatch")
    else:
        pass

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("P16 draft checkpoint must contain a dictionary")
    else:
        pass
    metadata_fields = {
        "width",
        "arm",
        "state_dict",
        "manifest_sha256",
        "epochs",
        "stage",
    }
    if set(checkpoint) != metadata_fields:
        raise ValueError("P16 draft checkpoint metadata fields do not match P13 schema")
    elif checkpoint["width"] != 1024 or checkpoint["arm"] != "D":
        raise ValueError("P16 draft checkpoint must be P13 arm D with width 1024")
    elif checkpoint["stage"] != "600k" or checkpoint["epochs"] != 10:
        raise ValueError("P16 draft checkpoint has an unsupported P13 training recipe")
    elif checkpoint["manifest_sha256"] != P16_MANIFEST_SHA256:
        raise ValueError("P16 draft checkpoint training manifest does not match")
    else:
        pass
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or set(state_dict) != set(
        P16_DRAFT_WEIGHT_SHAPES
    ):
        raise ValueError(
            "P16 draft checkpoint tensor keys do not match the expected model"
        )
    else:
        pass
    draft_weights: dict[str, torch.Tensor] = {}
    for name, expected_shape in P16_DRAFT_WEIGHT_SHAPES.items():
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("P16 draft state dictionary must contain named tensors")
        elif tuple(tensor.shape) != expected_shape or tensor.dtype is not torch.float32:
            raise ValueError(f"P16 draft tensor {name} has an invalid shape or dtype")
        elif not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"P16 draft tensor {name} contains nonfinite values")
        else:
            pass
        draft_weights[name] = tensor.contiguous()
    if float(draft_weights["hidden_scale"].item()) <= 0.0:
        raise ValueError("P16 hidden normalization scale must be positive")
    elif float(draft_weights["embed_scale"].item()) <= 0.0:
        raise ValueError("P16 embedding normalization scale must be positive")
    else:
        pass
    return P16DraftCheckpoint(weights=draft_weights, sha256=actual_sha256)


if triton is not None:

    @triton.jit
    def input_projection(
        hidden_states,
        token_ids,
        embedding_depth_lut,
        hidden_weight,
        activation,
        DEPTH: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hidden_offsets = tl.arange(0, 1024)
        token_id = tl.load(token_ids).to(tl.int32)
        table_row = (DEPTH - 1) * 2048 + tl.minimum(tl.maximum(token_id, 0), 2047)
        hidden = tl.load(hidden_states + hidden_offsets).to(tl.float32)
        embedding_depth = tl.load(embedding_depth_lut + table_row * WIDTH + rows).to(
            tl.float32
        )
        weight_rows = tl.load(
            hidden_weight + rows[:, None] * 1024 + hidden_offsets[None, :],
            mask=rows[:, None] < WIDTH,
            other=0.0,
        ).to(tl.float32)
        projected = tl.sum(weight_rows * hidden[None, :], axis=1) + embedding_depth
        projected_bf16 = projected.to(tl.bfloat16).to(tl.float32)
        silu = (projected_bf16 / (1.0 + tl.exp(-projected_bf16))).to(tl.bfloat16)
        tl.store(activation + rows, silu, mask=rows < WIDTH)

    @triton.jit
    def output_projection(
        hidden_states,
        activation,
        output_weight,
        output_bias,
        output,
        WIDTH: tl.constexpr,
        OUTPUTS_PER_CTA: tl.constexpr,
    ):
        output_offsets = tl.program_id(0) * OUTPUTS_PER_CTA + tl.arange(
            0, OUTPUTS_PER_CTA
        )
        width_offsets = tl.arange(0, WIDTH)
        activation_values = tl.load(activation + width_offsets).to(tl.float32)
        weight_values = tl.load(
            output_weight + output_offsets[:, None] * WIDTH + width_offsets[None, :]
        ).to(tl.float32)
        hidden_values = tl.load(hidden_states + output_offsets).to(tl.float32)
        bias_values = tl.load(output_bias + output_offsets).to(tl.float32)
        residual = (hidden_values + bias_values).to(tl.bfloat16).to(tl.float32)
        result = tl.sum(weight_values * activation_values[None, :], axis=1) + residual
        tl.store(output + output_offsets, result.to(tl.bfloat16))

else:
    input_projection = None
    output_projection = None

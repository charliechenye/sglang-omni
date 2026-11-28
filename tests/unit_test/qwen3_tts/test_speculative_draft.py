# SPDX-License-Identifier: Apache-2.0
"""CPU validation tests for the pinned P16 draft checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sglang_omni.models.qwen3_tts import speculative_draft


class FixedDigest:
    def update(self, value: bytes) -> None:
        del value

    def hexdigest(self) -> str:
        return speculative_draft.P16_DRAFT_SHA256


def valid_checkpoint() -> dict[str, object]:
    return {
        "width": 1024,
        "arm": "D",
        "state_dict": {
            "hidden_scale": torch.tensor(1.0),
            "embed_scale": torch.tensor(1.0),
            "in_proj.weight": torch.ones((2, 3), dtype=torch.float32),
            "in_proj.bias": torch.ones((2,), dtype=torch.float32),
            "out_proj.weight": torch.ones((2, 2), dtype=torch.float32),
            "out_proj.bias": torch.ones((2,), dtype=torch.float32),
        },
        "manifest_sha256": speculative_draft.P16_MANIFEST_SHA256,
        "epochs": 10,
        "stage": "600k",
    }


def configure_fake_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
) -> None:
    checkpoint_path.write_bytes(b"pinned test artifact")
    monkeypatch.setattr(
        speculative_draft,
        "P16_DRAFT_WEIGHT_SHAPES",
        {
            "hidden_scale": (),
            "embed_scale": (),
            "in_proj.weight": (2, 3),
            "in_proj.bias": (2,),
            "out_proj.weight": (2, 2),
            "out_proj.bias": (2,),
        },
    )
    monkeypatch.setattr(
        speculative_draft.hashlib,
        "sha256",
        lambda: FixedDigest(),
    )

    def fake_torch_load(
        path: Path, *, map_location: str, weights_only: bool
    ) -> dict[str, object]:
        assert path == checkpoint_path
        assert map_location == "cpu"
        assert weights_only is True
        return checkpoint

    monkeypatch.setattr(speculative_draft.torch, "load", fake_torch_load)


def test_p16_draft_loader_rejects_missing_and_unpinned_artifacts(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        speculative_draft.load_p16_draft_checkpoint(
            missing, speculative_draft.P16_DRAFT_SHA256
        )

    checkpoint_path = tmp_path / "wrong.pt"
    checkpoint_path.write_bytes(b"wrong artifact")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        speculative_draft.load_p16_draft_checkpoint(
            checkpoint_path, speculative_draft.P16_DRAFT_SHA256
        )

    with pytest.raises(ValueError, match="only accepts the pinned"):
        speculative_draft.load_p16_draft_checkpoint(checkpoint_path, "a" * 64)


def test_p16_draft_loader_accepts_only_validated_weights_only_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "draft.pt"
    checkpoint = valid_checkpoint()
    configure_fake_checkpoint_loader(monkeypatch, checkpoint_path, checkpoint)

    loaded = speculative_draft.load_p16_draft_checkpoint(
        checkpoint_path, speculative_draft.P16_DRAFT_SHA256
    )

    assert loaded.sha256 == speculative_draft.P16_DRAFT_SHA256
    assert set(loaded.weights) == set(speculative_draft.P16_DRAFT_WEIGHT_SHAPES)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest", "training manifest"),
        ("width", "arm D with width 1024"),
        ("missing_weight", "tensor keys"),
        ("shape", "invalid shape or dtype"),
        ("dtype", "invalid shape or dtype"),
        ("nonfinite", "nonfinite values"),
        ("normalization", "normalization scale must be positive"),
    ],
)
def test_p16_draft_loader_rejects_malformed_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    checkpoint_path = tmp_path / f"{mutation}.pt"
    checkpoint = valid_checkpoint()
    state_dict = checkpoint["state_dict"]
    assert isinstance(state_dict, dict)
    if mutation == "manifest":
        checkpoint["manifest_sha256"] = "0" * 64
    elif mutation == "width":
        checkpoint["width"] = 512
    elif mutation == "missing_weight":
        del state_dict["out_proj.bias"]
    elif mutation == "shape":
        state_dict["in_proj.bias"] = torch.ones((3,), dtype=torch.float32)
    elif mutation == "dtype":
        state_dict["out_proj.bias"] = torch.ones((2,), dtype=torch.bfloat16)
    elif mutation == "nonfinite":
        state_dict["out_proj.bias"] = torch.tensor([float("nan"), 1.0])
    elif mutation == "normalization":
        state_dict["hidden_scale"] = torch.tensor(0.0)
    else:
        raise AssertionError(f"unknown checkpoint mutation: {mutation}")

    configure_fake_checkpoint_loader(monkeypatch, checkpoint_path, checkpoint)
    with pytest.raises(ValueError, match=message):
        speculative_draft.load_p16_draft_checkpoint(
            checkpoint_path, speculative_draft.P16_DRAFT_SHA256
        )

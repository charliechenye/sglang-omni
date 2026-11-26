# SPDX-License-Identifier: Apache-2.0
"""Load and validate frozen Qwen3-TTS Predictor frame artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

SAMPLING_TENSOR_NAMES = (
    "sub_temperature_tensor",
    "sub_top_p_tensor",
    "sub_top_k_tensor",
    "sub_sampling_seed_tensor",
    "sub_do_sample_tensor",
    "semantic_sampling_seed_tensor",
)
PREDICTOR_HEAD_COUNT = 15
PREDICTOR_CODE_COUNT = 16
FROZEN_FRAME_SCHEMA = "r2e0-depth-transitions-v1"
NATIVE_FRAME_SCHEMA = "r2e0-native-frame-v1"


@dataclass(frozen=True, kw_only=True)
class FrozenPredictorFrame:
    frame_path: Path
    request_id: str
    ordinal: int
    layer0_code: torch.Tensor
    talker_hidden: torch.Tensor
    native_codes: torch.Tensor
    native_embeddings: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    projected_embed: torch.Tensor
    codes: torch.Tensor
    semantic_positions: torch.Tensor
    sampling: dict[str, torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class FrozenPredictorArtifacts:
    frames: tuple[FrozenPredictorFrame, ...]
    lm_head_weights: torch.Tensor
    model_snapshot: str
    dataset_sha256: str
    capture_sha256: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact JSON must contain an object: {path}")
    else:
        pass
    return value


def require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"artifact field must be an object: {label}")
    else:
        pass
    return value


def require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"artifact field must be an array: {label}")
    else:
        pass
    return value


def require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact field must be a nonempty string: {label}")
    else:
        pass
    return value


def require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"artifact field must be an integer: {label}")
    else:
        pass
    return value


def require_tensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"artifact field must be a tensor: {label}")
    else:
        pass
    return value


def require_tensor_mapping(value: object, label: str) -> dict[str, torch.Tensor]:
    mapping = require_mapping(value, label)
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in mapping.items():
        if not isinstance(name, str):
            raise ValueError(f"artifact tensor name must be a string: {label}")
        else:
            pass
        tensors[name] = require_tensor(tensor, f"{label}.{name}")
    return tensors


def mapped_capture_frames(capture_root: Path) -> dict[str, Path]:
    capture_manifest = load_json_object(capture_root / "capture_manifest.json")
    if capture_manifest.get("status") != "R2E0_CAPTURE_COMPLETE":
        raise ValueError("capture manifest is not complete")
    else:
        pass
    capture_records = require_list(capture_manifest.get("records"), "records")
    native_frames: dict[str, Path] = {}
    for capture_record in capture_records:
        record = require_mapping(capture_record, "capture record")
        request_id = require_string(record.get("uid"), "capture record uid")
        done_path = capture_root / "raw" / request_id / "done.json"
        expected_done_sha = require_string(
            record.get("sha256"), f"{request_id}.done sha256"
        )
        if sha256_file(done_path) != expected_done_sha:
            raise ValueError(
                f"capture completion manifest checksum mismatch: {done_path}"
            )
        else:
            pass
        done_manifest = load_json_object(done_path)
        files = require_list(done_manifest.get("files"), f"{request_id}.files")
        for file_value in files:
            file_record = require_mapping(file_value, f"{request_id} frame file")
            name = require_string(file_record.get("name"), "frame file name")
            expected_sha = require_string(file_record.get("sha256"), "frame sha256")
            frame_path = done_path.parent / name
            if sha256_file(frame_path) != expected_sha:
                raise ValueError(f"captured frame checksum mismatch: {frame_path}")
            elif expected_sha in native_frames:
                raise ValueError(f"duplicate captured frame digest: {expected_sha}")
            else:
                native_frames[expected_sha] = frame_path
    return native_frames


def load_frozen_predictor_artifacts(
    capture_root: Path,
    dataset_root: Path,
    *,
    frame_limit: int,
    split: str,
    expected_model_snapshot: str | None,
) -> FrozenPredictorArtifacts:
    """Load a checksum-verified capture joined to its extracted native targets."""
    if frame_limit <= 0:
        raise ValueError("frame limit must be positive")
    elif split not in ("all", "train", "validation", "test"):
        raise ValueError("split must be all, train, validation, or test")
    else:
        pass
    summary_path = dataset_root / "summary.json"
    summary = load_json_object(summary_path)
    if summary.get("status") != "R2E0_EXTRACTION_COMPLETE":
        raise ValueError("dataset extraction is incomplete")
    else:
        pass
    capture_launch = load_json_object(capture_root / "launch_manifest.json")
    dataset_launch = load_json_object(dataset_root / "launch_manifest.json")
    capture_snapshot = require_string(
        capture_launch.get("model_snapshot"), "capture model_snapshot"
    )
    dataset_snapshot = require_string(
        dataset_launch.get("model_snapshot"), "dataset model_snapshot"
    )
    if capture_snapshot != dataset_snapshot:
        raise ValueError("capture and dataset checkpoints do not match")
    elif (
        expected_model_snapshot is not None
        and capture_snapshot != expected_model_snapshot
    ):
        raise ValueError("artifact checkpoint does not match the requested snapshot")
    else:
        pass
    if capture_launch.get("omni_sha") != dataset_launch.get("omni_sha"):
        raise ValueError(
            "capture and dataset were created from different Omni revisions"
        )
    else:
        pass

    head_weights_path = dataset_root / "lm_heads.pt"
    expected_head_weights_sha = require_string(
        summary.get("lm_heads_sha256"), "lm_heads_sha256"
    )
    if sha256_file(head_weights_path) != expected_head_weights_sha:
        raise ValueError("LM-head checkpoint checksum mismatch")
    else:
        pass
    head_weights_payload = require_mapping(
        torch.load(head_weights_path, map_location="cpu", weights_only=True),
        "lm_heads.pt",
    )
    head_weights = require_tensor(head_weights_payload.get("weight"), "lm_heads.weight")
    hidden_size = require_integer(summary.get("hidden_size"), "hidden_size")
    vocab_size = require_integer(summary.get("vocab_size"), "vocab_size")
    if (
        tuple(head_weights.shape) != (PREDICTOR_HEAD_COUNT, vocab_size, hidden_size)
        or head_weights.dtype is not torch.bfloat16
    ):
        raise ValueError("LM-head weights do not match the extracted checkpoint shape")
    else:
        pass

    captured_frames = mapped_capture_frames(capture_root)
    shard_records = require_list(summary.get("shards"), "shards")
    frames: list[FrozenPredictorFrame] = []
    for shard_value in shard_records:
        shard = require_mapping(shard_value, "shard manifest record")
        shard_split = require_string(shard.get("split"), "shard split")
        if split != "all" and shard_split != split:
            continue
        else:
            pass
        shard_name = require_string(shard.get("name"), "shard name")
        shard_path = dataset_root / shard_name
        expected_shard_sha = require_string(shard.get("sha256"), "shard sha256")
        if sha256_file(shard_path) != expected_shard_sha:
            raise ValueError(f"dataset shard checksum mismatch: {shard_path}")
        else:
            pass
        shard_payload = require_mapping(
            torch.load(shard_path, map_location="cpu", weights_only=True), shard_name
        )
        if shard_payload.get("schema") != FROZEN_FRAME_SCHEMA:
            raise ValueError(f"unsupported frozen frame schema in {shard_path}")
        else:
            pass
        hidden = require_tensor(shard_payload.get("hidden"), f"{shard_name}.hidden")
        logits = require_tensor(shard_payload.get("logits"), f"{shard_name}.logits")
        projected_embed = require_tensor(
            shard_payload.get("projected_embed"), f"{shard_name}.projected_embed"
        )
        codes = require_tensor(shard_payload.get("codes"), f"{shard_name}.codes")
        semantic_positions = require_tensor(
            shard_payload.get("semantic_positions"),
            f"{shard_name}.semantic_positions",
        )
        sampling = require_tensor_mapping(
            shard_payload.get("sampling"), f"{shard_name}.sampling"
        )
        records = require_list(shard_payload.get("records"), f"{shard_name}.records")
        frame_count = int(hidden.shape[0])
        if tuple(hidden.shape) != (frame_count, PREDICTOR_HEAD_COUNT, hidden_size):
            raise ValueError(
                f"native hidden states have an invalid shape: {shard_path}"
            )
        elif tuple(logits.shape) != (frame_count, PREDICTOR_HEAD_COUNT, vocab_size):
            raise ValueError(f"native logits have an invalid shape: {shard_path}")
        elif tuple(projected_embed.shape) != (
            frame_count,
            PREDICTOR_CODE_COUNT - 2,
            hidden_size,
        ):
            raise ValueError(
                f"projected embeddings have an invalid shape: {shard_path}"
            )
        elif tuple(codes.shape) != (frame_count, PREDICTOR_CODE_COUNT):
            raise ValueError(f"native code frame has an invalid shape: {shard_path}")
        elif tuple(semantic_positions.shape) != (frame_count, 1):
            raise ValueError(f"semantic positions have an invalid shape: {shard_path}")
        elif set(sampling) != set(SAMPLING_TENSOR_NAMES):
            raise ValueError("sampling tensors do not match the frozen frame schema")
        elif len(records) != frame_count:
            raise ValueError(f"record count does not match tensors in {shard_path}")
        elif any(
            tuple(tensor.shape) not in ((frame_count,), (frame_count, 1))
            for tensor in sampling.values()
        ):
            raise ValueError(
                f"sampling tensors have invalid row counts in {shard_path}"
            )
        else:
            pass

        for row_index, record_value in enumerate(records):
            record = require_mapping(record_value, f"{shard_name} record")
            raw_sha = require_string(record.get("raw_sha256"), "record raw_sha256")
            frame_path = captured_frames.get(raw_sha)
            if frame_path is None:
                raise ValueError(f"dataset record has no captured frame: {raw_sha}")
            else:
                pass
            raw_frame = require_mapping(
                torch.load(frame_path, map_location="cpu", weights_only=True),
                str(frame_path),
            )
            if raw_frame.get("schema") != NATIVE_FRAME_SCHEMA:
                raise ValueError(f"unsupported native frame schema: {frame_path}")
            else:
                pass
            request_id = require_string(record.get("request_id"), "record request_id")
            ordinal = require_integer(record.get("ordinal"), "record ordinal")
            if (
                raw_frame.get("request_id") != request_id
                or raw_frame.get("ordinal") != ordinal
            ):
                raise ValueError(
                    f"native frame identity does not match its dataset row"
                )
            else:
                pass
            raw_sampling = require_tensor_mapping(
                raw_frame.get("sampling"), f"{frame_path}.sampling"
            )
            row_sampling = {
                name: tensor[row_index].reshape(()).clone()
                for name, tensor in sampling.items()
            }
            if set(raw_sampling) != set(SAMPLING_TENSOR_NAMES):
                raise ValueError(f"captured sampling fields do not match: {frame_path}")
            elif any(
                raw_sampling[name].numel() != 1
                or not torch.equal(raw_sampling[name].reshape(()), row_sampling[name])
                for name in SAMPLING_TENSOR_NAMES
            ):
                raise ValueError("captured sampling values differ from dataset shard")
            else:
                pass
            raw_codes = require_tensor(raw_frame.get("native_codes"), "native_codes")
            if not torch.equal(raw_codes.reshape(-1), codes[row_index]):
                raise ValueError(
                    f"native codes differ from dataset targets: {frame_path}"
                )
            else:
                pass
            layer0_code = require_tensor(
                raw_frame.get("layer0_codes"), "layer0_codes"
            ).reshape(-1)
            talker_hidden = require_tensor(
                raw_frame.get("talker_hidden"), "talker_hidden"
            ).reshape(-1)
            native_embeddings = require_tensor(
                raw_frame.get("native_embeddings"), "native_embeddings"
            ).reshape(-1)
            raw_positions = require_tensor(
                raw_frame.get("semantic_positions"), "semantic_positions"
            ).reshape(-1)
            if layer0_code.numel() != 1:
                raise ValueError(f"captured layer-0 code must be scalar: {frame_path}")
            elif int(layer0_code.item()) != int(codes[row_index, 0].item()):
                raise ValueError(
                    f"captured layer-0 code differs from targets: {frame_path}"
                )
            elif talker_hidden.numel() != hidden_size:
                raise ValueError(
                    f"captured talker hidden has invalid shape: {frame_path}"
                )
            elif native_embeddings.numel() != hidden_size:
                raise ValueError(
                    f"captured native embedding has invalid shape: {frame_path}"
                )
            elif not torch.equal(
                raw_positions, semantic_positions[row_index].reshape(-1)
            ):
                raise ValueError(
                    f"captured semantic position differs from dataset: {frame_path}"
                )
            else:
                pass
            frames.append(
                FrozenPredictorFrame(
                    frame_path=frame_path,
                    request_id=request_id,
                    ordinal=ordinal,
                    layer0_code=layer0_code.contiguous(),
                    talker_hidden=talker_hidden.contiguous(),
                    native_codes=raw_codes.reshape(-1).contiguous(),
                    native_embeddings=native_embeddings.contiguous(),
                    hidden=hidden[row_index].contiguous(),
                    logits=logits[row_index].contiguous(),
                    projected_embed=projected_embed[row_index].contiguous(),
                    codes=codes[row_index].contiguous(),
                    semantic_positions=semantic_positions[row_index].contiguous(),
                    sampling=row_sampling,
                )
            )

    if not frames:
        raise ValueError(f"no frozen frames matched split {split!r}")
    else:
        pass
    if len(frames) > frame_limit:
        selected_indices = (
            torch.linspace(0, len(frames) - 1, steps=frame_limit, dtype=torch.float64)
            .round()
            .to(dtype=torch.long)
        )
        frames = [frames[int(index)] for index in selected_indices]
    else:
        pass
    return FrozenPredictorArtifacts(
        frames=tuple(frames),
        lm_head_weights=head_weights.contiguous(),
        model_snapshot=capture_snapshot,
        dataset_sha256=sha256_file(summary_path),
        capture_sha256=sha256_file(capture_root / "capture_manifest.json"),
    )

# SPDX-License-Identifier: Apache-2.0
"""Test-only FP8 KV-cache calibration for MOSS-Transcribe-Diarize.

The collector attaches pre-forward hooks to the Qwen3 ``RadixAttention``
modules.  At that boundary Qwen3 has already applied K/RoPE processing and the
same K/V tensors are about to be handed to SGLang's attention/cache path.

The hot path only performs device-side reductions and updates.  Host copies
and atomic JSON writes happen at the configured checkpoint interval and during
explicit finalization on stage shutdown.
"""

from __future__ import annotations

import argparse
import datetime as datetime_module
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import torch

EXPECTED_NUM_LAYERS = 28
FP8_E4M3_MAX = 448.0
RAW_ARTIFACT_NAME = "moss_td_kv_calibration"
RAW_SCHEMA_VERSION = 1
COLLECTOR_VERSION = "1.0.0"
VLLM_KV_CACHE_DTYPE = "float8_e4m3fn"
QWEN3_MODEL_TYPE = "qwen3"


class CalibrationValidationError(ValueError):
    """Raised when a raw calibration artifact is not safe to consume."""


def _utc_timestamp() -> str:
    return (
        datetime_module.datetime.now(datetime_module.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _git_metadata() -> tuple[str, bool]:
    """Return the source checkout's exact HEAD and dirty state.

    Calibration is intentionally refused when the source checkout cannot be
    identified.  A raw artifact without provenance is not useful for a later
    FP8 comparison.
    """

    source_root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        if not head:
            raise RuntimeError("git rev-parse returned an empty HEAD")
        return head, dirty
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise RuntimeError(
            "MOSS-TD KV calibration requires the sglang-omni source checkout "
            f"at {source_root} to have a readable git HEAD: {exc}"
        ) from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace *path* with a durable JSON document."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(
                payload,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _require_json_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationValidationError(f"{field} must be an integer")
    return int(value)


def _validate_amax(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CalibrationValidationError(f"{field} must be a finite number")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise CalibrationValidationError(f"{field} must be finite")
    if value_float <= 0.0:
        raise CalibrationValidationError(f"{field} must be > 0")


def validate_raw_calibration(payload: Mapping[str, Any]) -> None:
    """Validate a completed MOSS-TD raw calibration payload.

    The status check is deliberate: an atomically written checkpoint is still
    not a usable calibration until the collector has been finalized and all
    observations have passed these checks.
    """

    if not isinstance(payload, Mapping):
        raise CalibrationValidationError("raw calibration must be a JSON object")
    if payload.get("artifact") != RAW_ARTIFACT_NAME:
        raise CalibrationValidationError("unexpected raw calibration artifact name")
    if _require_json_integer(payload.get("schema_version"), field="schema_version") != (
        RAW_SCHEMA_VERSION
    ):
        raise CalibrationValidationError("unsupported raw calibration schema version")
    if payload.get("collector_version") != COLLECTOR_VERSION:
        raise CalibrationValidationError("unsupported calibration collector version")
    if payload.get("status") != "complete":
        raise CalibrationValidationError(
            f"raw calibration status is {payload.get('status')!r}, expected 'complete'"
        )

    model_path = payload.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise CalibrationValidationError("model_path must be a non-empty string")
    git_head = payload.get("git_head")
    if not isinstance(git_head, str) or not git_head.strip():
        raise CalibrationValidationError("git_head must be a non-empty string")
    if not isinstance(payload.get("git_dirty"), bool):
        raise CalibrationValidationError("git_dirty must be a boolean")
    if payload.get("model_type") != QWEN3_MODEL_TYPE:
        raise CalibrationValidationError(
            f"model_type must be {QWEN3_MODEL_TYPE!r}"
        )

    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str):
        raise CalibrationValidationError("timestamp must be an ISO-8601 string")
    try:
        datetime_module.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CalibrationValidationError(
            "timestamp must be a valid ISO-8601 string"
        ) from exc

    num_layers = _require_json_integer(payload.get("num_layers"), field="num_layers")
    if num_layers != EXPECTED_NUM_LAYERS:
        raise CalibrationValidationError(
            f"expected exactly {EXPECTED_NUM_LAYERS} layers, got {num_layers}"
        )

    observed_layers = payload.get("observed_layers")
    if not isinstance(observed_layers, list):
        raise CalibrationValidationError("observed_layers must be a list")
    observed = [
        _require_json_integer(layer, field="observed_layers entry")
        for layer in observed_layers
    ]
    expected_layers = list(range(EXPECTED_NUM_LAYERS))
    if len(observed) != EXPECTED_NUM_LAYERS or sorted(observed) != expected_layers:
        raise CalibrationValidationError(
            "calibration did not observe exactly all 28 layers: "
            f"{sorted(set(observed))}"
        )
    observed_layer_count = _require_json_integer(
        payload.get("observed_layer_count"), field="observed_layer_count"
    )
    if observed_layer_count != EXPECTED_NUM_LAYERS:
        raise CalibrationValidationError(
            "observed_layer_count must be exactly "
            f"{EXPECTED_NUM_LAYERS}, got {observed_layer_count}"
        )

    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise CalibrationValidationError("layers must be a list")
    if len(layers) != EXPECTED_NUM_LAYERS:
        raise CalibrationValidationError(
            f"expected exactly {EXPECTED_NUM_LAYERS} layer records, got {len(layers)}"
        )

    layer_indices: list[int] = []
    for position, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise CalibrationValidationError(
                f"layers[{position}] must be a JSON object"
            )
        layer_index = _require_json_integer(
            layer.get("layer"), field=f"layers[{position}].layer"
        )
        layer_indices.append(layer_index)
        _validate_amax(
            layer.get("k_amax"),
            field=f"layer {layer_index} k_amax",
        )
        _validate_amax(
            layer.get("v_amax"),
            field=f"layer {layer_index} v_amax",
        )

    if sorted(layer_indices) != expected_layers:
        raise CalibrationValidationError(
            "layer records do not contain exactly one record for every layer 0-27: "
            f"{sorted(layer_indices)}"
        )


def read_raw_calibration(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a completed raw calibration artifact."""

    artifact_path = Path(path).expanduser()
    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CalibrationValidationError(
            f"could not read valid JSON from {artifact_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CalibrationValidationError("raw calibration must be a JSON object")
    validate_raw_calibration(payload)
    return payload


def _json_safe_float(value: float) -> float | str:
    """Keep invalid device values visible without emitting non-standard JSON."""

    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _device_amax(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor on-device while preserving any non-finite observation."""

    detached = tensor.detach()
    candidate = detached.abs().amax().to(dtype=torch.float32)
    has_nonfinite = (~torch.isfinite(detached)).any()
    return torch.where(
        has_nonfinite,
        torch.full_like(candidate, float("nan")),
        candidate,
    )


class MossTDKVCalibrationCollector:
    """Collect device-side per-layer K/V maxima from RadixAttention inputs."""

    def __init__(
        self,
        attention_modules: Sequence[Any],
        *,
        model_path: str,
        output_path: str | os.PathLike[str],
        device: torch.device | str,
        checkpoint_interval_s: float = 30.0,
        git_metadata: tuple[str, bool] | None = None,
    ) -> None:
        self._attention_modules = tuple(attention_modules)
        if len(self._attention_modules) != EXPECTED_NUM_LAYERS:
            raise CalibrationValidationError(
                f"MOSS-TD calibration requires exactly {EXPECTED_NUM_LAYERS} "
                f"attention layers, got {len(self._attention_modules)}"
            )
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        if isinstance(checkpoint_interval_s, bool):
            raise ValueError("checkpoint_interval_s must be a positive number")
        try:
            checkpoint_interval_s = float(checkpoint_interval_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "checkpoint_interval_s must be a positive number"
            ) from exc
        if not math.isfinite(checkpoint_interval_s) or checkpoint_interval_s <= 0.0:
            raise ValueError(
                "checkpoint_interval_s must be a finite positive number"
            )

        self._model_path = model_path
        self._output_path = Path(output_path).expanduser()
        if not self._output_path.name:
            raise ValueError("output_path must name an artifact file")
        if self._output_path.exists() and self._output_path.is_dir():
            raise ValueError("output_path must not be a directory")
        self._checkpoint_interval_s = checkpoint_interval_s
        self._device = torch.device(device)
        self._running_amax = torch.zeros(
            (EXPECTED_NUM_LAYERS, 2), dtype=torch.float32, device=self._device
        )
        self._seen_layers: set[int] = set()
        self._hook_calls = 0
        self._handles: list[Any] = []
        self._closed = False
        self._timestamp = _utc_timestamp()
        self._next_checkpoint_at = time.monotonic() + checkpoint_interval_s
        if git_metadata is None:
            git_metadata = _git_metadata()
        git_head, git_dirty = git_metadata
        self._metadata = {
            "artifact": RAW_ARTIFACT_NAME,
            "collector_version": COLLECTOR_VERSION,
            "schema_version": RAW_SCHEMA_VERSION,
            "model_type": QWEN3_MODEL_TYPE,
            "model_path": model_path,
            "git_head": git_head,
            "git_dirty": bool(git_dirty),
            "timestamp": self._timestamp,
            "num_layers": EXPECTED_NUM_LAYERS,
        }

        for index, module in enumerate(self._attention_modules):
            module_layer_id = getattr(module, "layer_id", index)
            try:
                module_layer_id = int(module_layer_id)
            except (TypeError, ValueError) as exc:
                raise CalibrationValidationError(
                    f"RadixAttention layer {index} has invalid layer_id "
                    f"{module_layer_id!r}"
                ) from exc
            if module_layer_id != index:
                raise CalibrationValidationError(
                    f"RadixAttention layer order mismatch: module {index} has "
                    f"layer_id {module_layer_id}"
                )
            register_hook = getattr(module, "register_forward_pre_hook", None)
            if not callable(register_hook):
                raise TypeError(
                    f"layer {index} is not a hookable RadixAttention module"
                )

        try:
            self._write_snapshot(status="in_progress")
            for index, module in enumerate(self._attention_modules):
                self._handles.append(
                    module.register_forward_pre_hook(self._make_hook(index))
                )
        except BaseException:
            self._remove_hooks()
            raise

    def _make_hook(self, layer_index: int):
        def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            self._capture(layer_index, inputs)

        return capture

    def _capture(self, layer_index: int, inputs: tuple[Any, ...]) -> None:
        if self._closed:
            return
        self._hook_calls += 1
        if len(inputs) < 3:
            raise RuntimeError(
                f"layer {layer_index} RadixAttention hook received fewer than "
                "three positional inputs; cannot identify K/V"
            )
        key, value = inputs[1], inputs[2]
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"layer {layer_index} RadixAttention hook did not receive tensor K/V"
            )
        if key.numel() == 0 or value.numel() == 0:
            raise RuntimeError(f"layer {layer_index} received an empty K/V tensor")
        if key.device != self._device or value.device != self._device:
            raise RuntimeError(
                f"layer {layer_index} K/V device mismatch: expected {self._device}, "
                f"got {key.device} and {value.device}"
            )

        # These reductions and maxima remain on the model device. In
        # particular, there is no host scalar read, CPU copy, or synchronization
        # here.
        with torch.no_grad():
            key_amax = _device_amax(key)
            value_amax = _device_amax(value)
            for slot, candidate in (
                (self._running_amax[layer_index, 0], key_amax),
                (self._running_amax[layer_index, 1], value_amax),
            ):
                # Preserve every non-finite observation so final artifact
                # validation cannot accidentally accept it, independent of
                # backend reduction semantics.
                finite_max = torch.maximum(slot, candidate)
                slot.copy_(
                    torch.where(torch.isfinite(candidate), finite_max, candidate)
                )
        self._seen_layers.add(layer_index)
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        now = time.monotonic()
        if now < self._next_checkpoint_at:
            return
        self._write_snapshot(status="in_progress")
        self._next_checkpoint_at = now + self._checkpoint_interval_s

    def _snapshot_values(self) -> list[list[float | str]]:
        # .cpu() is intentionally only used at periodic checkpoints/finalize;
        # it synchronizes once for the whole 28x2 tensor rather than once per
        # layer or decode step.
        values = self._running_amax.detach().cpu().tolist()
        return [
            [_json_safe_float(float(key)), _json_safe_float(float(value))]
            for key, value in values
        ]

    def _payload(
        self,
        *,
        status: str,
        validation_error: str | None = None,
    ) -> dict[str, Any]:
        values = self._snapshot_values()
        payload = {
            **self._metadata,
            "status": status,
            "observed_layer_count": len(self._seen_layers),
            "observed_layers": sorted(self._seen_layers),
            "layers": [
                {
                    "layer": index,
                    "k_amax": values[index][0],
                    "v_amax": values[index][1],
                }
                for index in range(EXPECTED_NUM_LAYERS)
            ],
            "last_checkpoint_timestamp": _utc_timestamp(),
            "hook_calls": self._hook_calls,
        }
        if validation_error is not None:
            payload["validation_error"] = validation_error
        return payload

    def _write_snapshot(self, *, status: str) -> None:
        _atomic_write_json(self._output_path, self._payload(status=status))

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finalize(self) -> None:
        """Remove hooks and atomically publish a valid completed artifact."""

        if self._closed:
            return
        self._remove_hooks()
        try:
            payload = self._payload(status="complete")
            try:
                validate_raw_calibration(payload)
            except CalibrationValidationError as exc:
                try:
                    self._write_payload_with_error(payload, str(exc))
                finally:
                    self._closed = True
                raise
            _atomic_write_json(self._output_path, payload)
        finally:
            self._closed = True

    def _write_payload_with_error(
        self,
        payload: Mapping[str, Any],
        validation_error: str,
    ) -> None:
        invalid_payload = dict(payload)
        invalid_payload["status"] = "invalid"
        invalid_payload["validation_error"] = validation_error
        _atomic_write_json(self._output_path, invalid_payload)

    def abort(self) -> None:
        """Detach without finalizing, leaving an unusable in-progress artifact."""

        if self._closed:
            return
        self._remove_hooks()
        self._closed = True


def find_moss_td_radix_attention_modules(model: Any) -> list[Any]:
    """Find MOSS-TD's 28 per-layer RadixAttention modules without type coupling."""

    try:
        layers = model.language_model.model.layers
    except AttributeError as exc:
        raise CalibrationValidationError(
            "MOSS-TD model does not expose language_model.model.layers"
        ) from exc
    if len(layers) != EXPECTED_NUM_LAYERS:
        raise CalibrationValidationError(
            f"expected exactly {EXPECTED_NUM_LAYERS} decoder layers, got {len(layers)}"
        )

    attention_modules: list[Any] = []
    for index, layer in enumerate(layers):
        try:
            attention = layer.self_attn.attn
        except AttributeError as exc:
            raise CalibrationValidationError(
                f"decoder layer {index} does not expose self_attn.attn"
            ) from exc
        attention_modules.append(attention)
    return attention_modules


def attach_moss_td_kv_calibration(
    model: Any,
    *,
    model_path: str,
    output_path: str | os.PathLike[str],
    checkpoint_interval_s: float = 30.0,
    git_metadata: tuple[str, bool] | None = None,
) -> MossTDKVCalibrationCollector:
    """Attach the opt-in MOSS-TD calibration collector to a loaded model."""

    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise CalibrationValidationError(
            "cannot determine the MOSS-TD model device for calibration"
        ) from exc
    return MossTDKVCalibrationCollector(
        find_moss_td_radix_attention_modules(model),
        model_path=model_path,
        output_path=output_path,
        device=parameter.device,
        checkpoint_interval_s=checkpoint_interval_s,
        git_metadata=git_metadata,
    )


def _validate_margin(margin: Any) -> float:
    if isinstance(margin, bool) or not isinstance(margin, Real):
        raise ValueError("margin is required and must be a finite positive number")
    margin_float = float(margin)
    if not math.isfinite(margin_float) or margin_float <= 0.0:
        raise ValueError("margin is required and must be a finite positive number")
    return margin_float


def convert_raw_calibration_to_vllm_legacy(
    raw_artifact_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    margin: float,
) -> dict[str, Any]:
    """Convert a completed raw artifact to PR #1128's legacy scale schema.

    The legacy schema has one scale per layer, so conversion uses the larger of
    that layer's independent K and V maxima.  This preserves both tensors with
    the shared scale that the upstream loader consumes.
    """

    margin_float = _validate_margin(margin)
    raw = read_raw_calibration(raw_artifact_path)
    layers = sorted(raw["layers"], key=lambda layer: int(layer["layer"]))
    scaling_factor: dict[str, float] = {}
    for layer in layers:
        layer_index = int(layer["layer"])
        shared_amax = max(float(layer["k_amax"]), float(layer["v_amax"]))
        scale = shared_amax / FP8_E4M3_MAX * margin_float
        if not math.isfinite(scale) or scale <= 0.0:
            raise CalibrationValidationError(
                f"converted scale for layer {layer_index} is not finite and > 0"
            )
        scaling_factor[str(layer_index)] = scale

    payload = {
        "model_type": raw["model_type"],
        "kv_cache": {
            "dtype": VLLM_KV_CACHE_DTYPE,
            # MOSS-TD's engine is TP=1; the legacy format still requires the
            # outer TP-rank map consumed by SGLang's loader.
            "scaling_factor": {"0": scaling_factor},
        },
    }
    _atomic_write_json(Path(output_path).expanduser(), payload)
    return payload


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or convert MOSS-TD FP8 KV calibration artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--raw-artifact", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--raw-artifact", required=True)
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument(
        "--margin",
        required=True,
        type=float,
        help="Required explicit upstream calibration margin; no default is used.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.command == "validate":
        read_raw_calibration(args.raw_artifact)
        print(f"valid raw calibration: {args.raw_artifact}")
        return 0
    if args.command == "convert":
        convert_raw_calibration_to_vllm_legacy(
            args.raw_artifact,
            args.output,
            margin=args.margin,
        )
        print(f"wrote vLLM-legacy KV scales: {args.output}")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTOR_VERSION",
    "EXPECTED_NUM_LAYERS",
    "FP8_E4M3_MAX",
    "MossTDKVCalibrationCollector",
    "RAW_ARTIFACT_NAME",
    "RAW_SCHEMA_VERSION",
    "CalibrationValidationError",
    "attach_moss_td_kv_calibration",
    "convert_raw_calibration_to_vllm_legacy",
    "find_moss_td_radix_attention_modules",
    "main",
    "read_raw_calibration",
    "validate_raw_calibration",
]

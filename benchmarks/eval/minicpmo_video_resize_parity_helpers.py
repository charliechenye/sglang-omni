# SPDX-License-Identifier: Apache-2.0
"""Pure comparison and frozen-cohort helpers for the MiniCPM-o ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.dataset.videomme import VideoMMESample


@dataclass(frozen=True, kw_only=True)
class PixelValueMetrics:
    """Exact and numerical comparison metrics for aligned pixel tensors."""

    exact_equal: bool
    max_abs: float | None
    mean_abs: float | None
    rel_l2: float | None


@dataclass(frozen=True, kw_only=True)
class ProcessorParityReport:
    """Processor-output comparison with a conservative exact-parity verdict."""

    input_ids_equal: bool
    baseline_input_ids_shape: list[int]
    candidate_input_ids_shape: list[int]
    image_bound_equal: bool
    baseline_tgt_sizes_count: int
    candidate_tgt_sizes_count: int
    baseline_tgt_sizes_shapes: list[list[int]]
    candidate_tgt_sizes_shapes: list[list[int]]
    tgt_sizes_equal: bool
    baseline_pixel_tensor_count: int
    candidate_pixel_tensor_count: int
    pixel_tensor_count_equal: bool
    baseline_pixel_shapes: list[list[int]]
    candidate_pixel_shapes: list[list[int]]
    pixel_shapes_equal: bool
    baseline_pixel_dtypes: list[str]
    candidate_pixel_dtypes: list[str]
    pixel_dtypes_equal: bool
    pixel_exact_equal: bool
    pixel_max_abs: float | None
    pixel_mean_abs: float | None
    pixel_rel_l2: float | None
    parity_pass: bool


def flatten_tensors(value: object) -> list[torch.Tensor]:
    """Collect tensor leaves in their existing nested order."""
    if isinstance(value, torch.Tensor):
        return [value]
    else:
        pass
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for child in value:
            tensors.extend(flatten_tensors(child))
        return tensors
    else:
        pass
    if isinstance(value, dict):
        tensors = []
        for child in value.values():
            tensors.extend(flatten_tensors(child))
        return tensors
    else:
        pass
    return []


def exact_value_equal(left: object, right: object) -> bool:
    """Compare nested processor values without ambiguous tensor equality."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    else:
        pass
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return False
    else:
        pass
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            exact_value_equal(left_child, right_child)
            for left_child, right_child in zip(left, right)
        )
    else:
        pass
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            exact_value_equal(left[key], right[key]) for key in left
        )
    else:
        pass
    return type(left) is type(right) and left == right


def tensor_shape(value: object) -> list[int]:
    """Return the first tensor shape in a processor value."""
    tensors = flatten_tensors(value)
    if tensors:
        return list(tensors[0].shape)
    else:
        pass
    return []


def compare_pixel_values(
    baseline_tensors: list[torch.Tensor],
    candidate_tensors: list[torch.Tensor],
) -> PixelValueMetrics:
    """Compare aligned pixel tensors, including exact equality and error norms."""
    if len(baseline_tensors) != len(candidate_tensors):
        return PixelValueMetrics(
            exact_equal=False,
            max_abs=None,
            mean_abs=None,
            rel_l2=None,
        )
    else:
        pass
    if any(
        baseline.shape != candidate.shape
        for baseline, candidate in zip(baseline_tensors, candidate_tensors)
    ):
        return PixelValueMetrics(
            exact_equal=False,
            max_abs=None,
            mean_abs=None,
            rel_l2=None,
        )
    else:
        pass

    exact_equal = True
    total_abs = 0.0
    total_squared_error = 0.0
    total_squared_baseline = 0.0
    total_elements = 0
    max_abs = 0.0
    for baseline, candidate in zip(baseline_tensors, candidate_tensors):
        if baseline.dtype != candidate.dtype or not torch.equal(baseline, candidate):
            exact_equal = False
        else:
            pass
        baseline_float = baseline.detach().cpu().to(dtype=torch.float64)
        candidate_float = candidate.detach().cpu().to(dtype=torch.float64)
        difference = baseline_float - candidate_float
        if difference.numel():
            total_abs += float(difference.abs().sum().item())
            total_squared_error += float(torch.square(difference).sum().item())
            total_squared_baseline += float(torch.square(baseline_float).sum().item())
            total_elements += difference.numel()
            max_abs = max(max_abs, float(difference.abs().max().item()))
        else:
            pass
    mean_abs = total_abs / total_elements if total_elements else 0.0
    rel_l2 = (
        total_squared_error**0.5 / total_squared_baseline**0.5
        if total_squared_baseline
        else (0.0 if total_squared_error == 0.0 else float("inf"))
    )
    return PixelValueMetrics(
        exact_equal=exact_equal,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rel_l2=rel_l2,
    )


def compare_processor_outputs(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> ProcessorParityReport:
    """Compare the processor outputs required for exact preprocessing parity."""
    baseline_pixel_tensors = flatten_tensors(baseline["pixel_values"])
    candidate_pixel_tensors = flatten_tensors(candidate["pixel_values"])
    pixel_shapes = [list(tensor.shape) for tensor in baseline_pixel_tensors]
    candidate_pixel_shapes = [list(tensor.shape) for tensor in candidate_pixel_tensors]
    pixel_dtypes = [str(tensor.dtype) for tensor in baseline_pixel_tensors]
    candidate_pixel_dtypes = [str(tensor.dtype) for tensor in candidate_pixel_tensors]
    pixel_shapes_equal = pixel_shapes == candidate_pixel_shapes
    pixel_dtypes_equal = pixel_dtypes == candidate_pixel_dtypes
    pixels_structurally_aligned = (
        len(baseline_pixel_tensors) == len(candidate_pixel_tensors)
        and pixel_shapes_equal
    )
    pixel_metrics = compare_pixel_values(
        baseline_pixel_tensors if pixels_structurally_aligned else [],
        candidate_pixel_tensors if pixels_structurally_aligned else [],
    )
    pixel_exact_equal = pixels_structurally_aligned and pixel_metrics.exact_equal
    pixel_max_abs = pixel_metrics.max_abs if pixels_structurally_aligned else None
    pixel_mean_abs = pixel_metrics.mean_abs if pixels_structurally_aligned else None
    pixel_rel_l2 = pixel_metrics.rel_l2 if pixels_structurally_aligned else None
    input_ids_equal = exact_value_equal(baseline["input_ids"], candidate["input_ids"])
    image_bound_equal = exact_value_equal(
        baseline["image_bound"], candidate["image_bound"]
    )
    baseline_tgt_sizes = flatten_tensors(baseline["tgt_sizes"])
    candidate_tgt_sizes = flatten_tensors(candidate["tgt_sizes"])
    tgt_sizes_equal = exact_value_equal(
        baseline["tgt_sizes"], candidate["tgt_sizes"]
    )
    structural_equal = all(
        (
            input_ids_equal,
            image_bound_equal,
            tgt_sizes_equal,
            len(baseline_pixel_tensors) == len(candidate_pixel_tensors),
            pixel_shapes_equal,
            pixel_dtypes_equal,
        )
    )
    return ProcessorParityReport(
        input_ids_equal=input_ids_equal,
        baseline_input_ids_shape=tensor_shape(baseline["input_ids"]),
        candidate_input_ids_shape=tensor_shape(candidate["input_ids"]),
        image_bound_equal=image_bound_equal,
        baseline_tgt_sizes_count=len(baseline_tgt_sizes),
        candidate_tgt_sizes_count=len(candidate_tgt_sizes),
        baseline_tgt_sizes_shapes=[list(tensor.shape) for tensor in baseline_tgt_sizes],
        candidate_tgt_sizes_shapes=[
            list(tensor.shape) for tensor in candidate_tgt_sizes
        ],
        tgt_sizes_equal=tgt_sizes_equal,
        baseline_pixel_tensor_count=len(baseline_pixel_tensors),
        candidate_pixel_tensor_count=len(candidate_pixel_tensors),
        pixel_tensor_count_equal=len(baseline_pixel_tensors)
        == len(candidate_pixel_tensors),
        baseline_pixel_shapes=pixel_shapes,
        candidate_pixel_shapes=candidate_pixel_shapes,
        pixel_shapes_equal=pixel_shapes_equal,
        baseline_pixel_dtypes=pixel_dtypes,
        candidate_pixel_dtypes=candidate_pixel_dtypes,
        pixel_dtypes_equal=pixel_dtypes_equal,
        pixel_exact_equal=pixel_exact_equal,
        pixel_max_abs=pixel_max_abs,
        pixel_mean_abs=pixel_mean_abs,
        pixel_rel_l2=pixel_rel_l2,
        parity_pass=structural_equal and pixel_exact_equal,
    )


def read_cohort_sample_ids(path: Path) -> list[str]:
    """Read ordered sample IDs from a frozen cohort JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("sample_ids")
        if entries is None:
            entries = payload.get("samples")
        else:
            pass
        if entries is None:
            entries = payload.get("per_sample")
        else:
            pass
    else:
        raise ValueError(f"Unsupported cohort JSON root in {path}")
    if not isinstance(entries, list):
        raise ValueError(f"Cohort JSON must contain an ordered list in {path}")
    else:
        pass

    sample_ids: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            sample_ids.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("sample_id"), str):
            sample_ids.append(entry["sample_id"])
        else:
            raise ValueError(f"Cohort entry has no string sample_id: {entry!r}")
    if not sample_ids:
        raise ValueError(f"Cohort JSON contains no sample IDs: {path}")
    else:
        pass
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Cohort JSON contains duplicate sample IDs: {path}")
    else:
        pass
    return sample_ids


def select_cohort_samples(
    samples: list[VideoMMESample],
    sample_ids: list[str],
) -> list[VideoMMESample]:
    """Filter samples in the exact order requested by a frozen cohort."""
    samples_by_id = {sample.sample_id: sample for sample in samples}
    missing_sample_ids = [
        sample_id for sample_id in sample_ids if sample_id not in samples_by_id
    ]
    if missing_sample_ids:
        raise ValueError(
            f"Cohort samples are missing from the dataset: {missing_sample_ids}"
        )
    else:
        pass
    return [samples_by_id[sample_id] for sample_id in sample_ids]

# SPDX-License-Identifier: Apache-2.0
"""Offline MiniCPM-o video resize ablation and processor parity harness."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import ProcessorMixin

from benchmarks.dataset.videomme import VideoMMESample, load_videomme_samples
from benchmarks.eval.minicpmo_video_resize_parity_helpers import (
    compare_processor_outputs,
    read_cohort_sample_ids,
    select_cohort_samples,
)
from sglang_omni.models.minicpm_o.components.preprocessor import (
    MiniCPMOPreprocessor,
    video_to_images,
)
from sglang_omni.preprocessing.video import (
    compute_video_resize_geometry,
    decode_video_path,
    resize_video_tensor,
)


@dataclass(frozen=True, kw_only=True)
class ArmResult:
    """One offline preprocessing arm's tensor and timing results."""

    video: torch.Tensor
    images: list[Image.Image]
    processed: dict[str, object]
    sglang_resize_ms: float
    video_to_images_ms: float
    processor_ms: float
    total_ms: float


def build_prompt_text(
    preprocessor: MiniCPMOPreprocessor,
    sample: VideoMMESample,
    image_count: int,
) -> str:
    """Reproduce serving placeholder insertion and chat-template rendering."""
    messages = preprocessor.messages_with_media_placeholders(
        [{"role": "user", "content": sample.prompt}],
        num_images=image_count,
        num_audios=0,
    )
    return preprocessor.render_chat_template(messages)


def run_arm(
    decoded_video: torch.Tensor,
    prompt_text: str,
    processor: ProcessorMixin,
    *,
    apply_sglang_resize: bool,
    min_pixels: int | None,
    max_pixels: int | None,
    total_pixels: int | None,
) -> ArmResult:
    """Run one offline arm from the shared decoded tensor to processor output."""
    total_started_ns = time.perf_counter_ns()
    resize_started_ns = time.perf_counter_ns()
    if apply_sglang_resize:
        geometry = compute_video_resize_geometry(
            decoded_video,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
        )
        video = resize_video_tensor(decoded_video, geometry).float()
    else:
        video = decoded_video
    resize_duration_ms = (time.perf_counter_ns() - resize_started_ns) / 1_000_000.0

    video_to_images_started_ns = time.perf_counter_ns()
    images = video_to_images(video)
    video_to_images_duration_ms = (
        time.perf_counter_ns() - video_to_images_started_ns
    ) / 1_000_000.0

    processor_started_ns = time.perf_counter_ns()
    processed = processor(
        prompt_text,
        images=[images],
        audios=None,
        return_tensors="pt",
        max_slice_nums=1,
        use_image_id=False,
    )
    processor_duration_ms = (
        time.perf_counter_ns() - processor_started_ns
    ) / 1_000_000.0
    total_duration_ms = (time.perf_counter_ns() - total_started_ns) / 1_000_000.0
    return ArmResult(
        video=video,
        images=images,
        processed=dict(processed),
        sglang_resize_ms=resize_duration_ms if apply_sglang_resize else 0.0,
        video_to_images_ms=video_to_images_duration_ms,
        processor_ms=processor_duration_ms,
        total_ms=total_duration_ms,
    )


def run_sample(
    sample: VideoMMESample,
    preprocessor: MiniCPMOPreprocessor,
    processor: ProcessorMixin,
    *,
    video_fps: float | None,
    video_max_frames: int | None,
    video_min_pixels: int | None,
    video_max_pixels: int | None,
    video_total_pixels: int | None,
) -> dict[str, object]:
    """Decode one sample once and compare the two processor arms."""
    decode_started_ns = time.perf_counter_ns()
    decoded_video, sample_fps = decode_video_path(
        sample.video_path,
        fps=video_fps,
        max_frames=video_max_frames,
        min_pixels=video_min_pixels,
        max_pixels=video_max_pixels,
        total_pixels=video_total_pixels,
    )
    decode_duration_ms = (time.perf_counter_ns() - decode_started_ns) / 1_000_000.0
    decoded_values = {
        "shape": list(decoded_video.shape),
        "dtype": str(decoded_video.dtype),
        "min": float(decoded_video.min().item()),
        "max": float(decoded_video.max().item()),
        "is_contiguous": decoded_video.is_contiguous(),
    }

    prompt_text = build_prompt_text(preprocessor, sample, int(decoded_video.shape[0]))
    baseline = run_arm(
        decoded_video,
        prompt_text,
        processor,
        apply_sglang_resize=True,
        min_pixels=video_min_pixels,
        max_pixels=video_max_pixels,
        total_pixels=video_total_pixels,
    )
    candidate = run_arm(
        decoded_video,
        prompt_text,
        processor,
        apply_sglang_resize=False,
        min_pixels=video_min_pixels,
        max_pixels=video_max_pixels,
        total_pixels=video_total_pixels,
    )
    parity = compare_processor_outputs(baseline.processed, candidate.processed)
    return {
        "sample_id": sample.sample_id,
        "frame_count": int(decoded_video.shape[0]),
        "sample_fps": float(sample_fps),
        "decoded_shape": decoded_values["shape"],
        "decoded_dtype": decoded_values["dtype"],
        "decoded_min": decoded_values["min"],
        "decoded_max": decoded_values["max"],
        "decoded_is_contiguous": decoded_values["is_contiguous"],
        "baseline_resized_shape": list(baseline.video.shape),
        "candidate_shape": list(candidate.video.shape),
        "input_ids_equal": parity.input_ids_equal,
        "input_ids_shape": {
            "baseline": parity.baseline_input_ids_shape,
            "candidate": parity.candidate_input_ids_shape,
        },
        "image_bound_equal": parity.image_bound_equal,
        "tgt_sizes_count": {
            "baseline": parity.baseline_tgt_sizes_count,
            "candidate": parity.candidate_tgt_sizes_count,
        },
        "tgt_sizes_shapes": {
            "baseline": parity.baseline_tgt_sizes_shapes,
            "candidate": parity.candidate_tgt_sizes_shapes,
        },
        "tgt_sizes_equal": parity.tgt_sizes_equal,
        "pixel_tensor_count_equal": parity.pixel_tensor_count_equal,
        "pixel_tensor_count": {
            "baseline": parity.baseline_pixel_tensor_count,
            "candidate": parity.candidate_pixel_tensor_count,
        },
        "pixel_shapes_equal": parity.pixel_shapes_equal,
        "pixel_shapes": {
            "baseline": parity.baseline_pixel_shapes,
            "candidate": parity.candidate_pixel_shapes,
        },
        "pixel_dtypes_equal": parity.pixel_dtypes_equal,
        "pixel_dtypes": {
            "baseline": parity.baseline_pixel_dtypes,
            "candidate": parity.candidate_pixel_dtypes,
        },
        "pixel_exact_equal": parity.pixel_exact_equal,
        "pixel_max_abs": parity.pixel_max_abs,
        "pixel_mean_abs": parity.pixel_mean_abs,
        "pixel_rel_l2": parity.pixel_rel_l2,
        "sglang_resize_ms": baseline.sglang_resize_ms,
        "video_to_images_baseline_ms": baseline.video_to_images_ms,
        "video_to_images_candidate_ms": candidate.video_to_images_ms,
        "baseline_processor_ms": baseline.processor_ms,
        "candidate_processor_ms": candidate.processor_ms,
        "baseline_total_ms": baseline.total_ms,
        "candidate_total_ms": candidate.total_ms,
        "decode_ms": decode_duration_ms,
        "parity_pass": parity.parity_pass,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the offline parity harness argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare MiniCPM-o processor outputs with and without SGLang resize"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--cohort-file", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames", type=int, default=128)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=401408)
    parser.add_argument("--video-total-pixels", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the offline parity harness and write one JSON report."""
    args = build_argument_parser().parse_args(argv)
    sample_ids = read_cohort_sample_ids(args.cohort_file)
    samples = select_cohort_samples(
        load_videomme_samples(repo_id=args.repo_id, split=args.split),
        sample_ids,
    )
    preprocessor = MiniCPMOPreprocessor(args.model_path)
    processor = preprocessor.processor
    sample_reports = [
        run_sample(
            sample,
            preprocessor,
            processor,
            video_fps=args.video_fps,
            video_max_frames=args.video_max_frames,
            video_min_pixels=args.video_min_pixels,
            video_max_pixels=args.video_max_pixels,
            video_total_pixels=args.video_total_pixels,
        )
        for sample in samples
    ]
    report = {
        "config": {
            "model_path": args.model_path,
            "repo_id": args.repo_id,
            "cohort_file": str(args.cohort_file),
            "split": args.split,
            "video_fps": args.video_fps,
            "video_max_frames": args.video_max_frames,
            "video_min_pixels": args.video_min_pixels,
            "video_max_pixels": args.video_max_pixels,
            "video_total_pixels": args.video_total_pixels,
            "max_slice_nums": 1,
            "use_image_id": False,
            "sample_ids": sample_ids,
        },
        "samples": sample_reports,
        "summary": {
            "sample_count": len(sample_reports),
            "parity_pass_count": sum(
                1 for sample_report in sample_reports if sample_report["parity_pass"]
            ),
            "all_parity_pass": all(
                sample_report["parity_pass"] for sample_report in sample_reports
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
else:
    pass

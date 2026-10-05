# SPDX-License-Identifier: Apache-2.0
"""Experimental MiniCPM-o vision measurements on the frozen Video-MME cohort."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# note (Codex): qwen-vl-utils reads the forced reader during import.
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"

import torch

from benchmarks.eval.minicpmo_vision_experiments import (
    BASELINE_BATCH_SIZE,
    ResamplerExperiment,
    VisionBatchExperiment,
    print_trial,
    resampler_experiment,
    vision_batch_sweep,
)
from benchmarks.eval.minicpmo_vision_inputs import preprocess_requests, resolve_requests
from benchmarks.eval.minicpmo_vision_measurement import (
    CudaPhases,
    EncoderForward,
    Trial,
    benchmark_forward,
    phase_hooks,
)
from benchmarks.eval.minicpmo_vision_metrics import ParityLimits, compare_outputs
from benchmarks.eval.minicpmo_vision_profile import (
    ModelDetails,
    ProfileArtifacts,
    inspect_model,
    profile_encoder,
)
from benchmarks.eval.minicpmo_vision_workload import (
    FROZEN_SAMPLE_IDS,
    VIDEO_FPS,
    VIDEO_MAX_FRAMES,
    VIDEO_MAX_PIXELS,
    CensusEntry,
    CensusSummary,
    Selection,
    load_census,
    print_census,
    select_request,
    summarize_census,
)
from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor

Mode = Literal["census", "decompose", "vision-batch", "resampler", "profile"]
EXPECTED_BASE_SHA = "a266a0894d8964ca592a86cb198b446acef8908a"


def resolve_measurement_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Measurement modes require CUDA; census supports CPU preprocessing"
        )
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        pass
    return device


@dataclass(kw_only=True)
class RunReport:
    mode: Mode
    model_id: str
    videomme_repo: str
    sample_ids: list[str]
    git_head: str
    expected_base_sha: str
    model: ModelDetails
    parity_limits: ParityLimits
    vision_batch_size: int
    video_fps: float = VIDEO_FPS
    video_max_frames: int = VIDEO_MAX_FRAMES
    video_max_pixels: int = VIDEO_MAX_PIXELS
    video_reader: str = "torchcodec"
    census: list[CensusEntry] = field(default_factory=list)
    census_summary: CensusSummary | None = None
    selection: Selection | None = None
    decomposition: Trial | None = None
    vision_batch: VisionBatchExperiment | None = None
    resampler: ResamplerExperiment | None = None
    profile: ProfileArtifacts | None = None
    substantial_drift: bool = False
    failed: bool = False


def artifact_path(path: Path) -> Path:
    resolved_path = path.expanduser().resolve()
    if any(
        (directory / ".git").exists()
        for directory in [resolved_path, *resolved_path.parents]
    ):
        raise ValueError(
            f"Write benchmark artifacts outside Git worktrees: {resolved_path}"
        )
    else:
        return resolved_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["census", "decompose", "vision-batch", "resampler", "profile"],
        required=True,
    )
    parser.add_argument(
        "--model-id",
        "--model-path",
        default=os.environ.get("SGLANG_MINICPMO_MODEL_ID", "openbmb/MiniCPM-o-4_5"),
    )
    parser.add_argument(
        "--videomme-repo",
        default=os.environ.get(
            "SGLANG_VIDEOMME_CI_REPO_ID", "zhaochenyang20/Video_MME_ci"
        ),
    )
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument(
        "--cohort-file",
        type=Path,
        help="JSON IDs, or samples/per_sample records with sample_id and optional video_path/prompt",
    )
    cohort.add_argument("--sample-ids", nargs="+", default=list(FROZEN_SAMPLE_IDS))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id", choices=list(FROZEN_SAMPLE_IDS))
    selection.add_argument(
        "--sample-selection", choices=["median", "high"], default="median"
    )
    parser.add_argument(
        "--census-file",
        type=Path,
        help="Reuse census.json for selection; only preprocess the selected request",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--vision-batch-sizes",
        type=int,
        nargs="+",
        choices=[8, 16, 32, 64],
        default=[16, 32, 64],
    )
    parser.add_argument(
        "--vision-batch-size",
        type=int,
        choices=[8, 16, 32, 64],
        default=BASELINE_BATCH_SIZE,
        help="Batch size for a single decompose measurement",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--resampler-warmup", type=int, default=10)
    parser.add_argument("--resampler-iters", type=int, default=50)
    parser.add_argument("--profile-iters", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "minicpmo-vision-profile",
    )
    parser.add_argument("--no-chrome-trace", action="store_true")
    parser.add_argument("--output-json", type=Path)
    limits = ParityLimits()
    parser.add_argument(
        "--parity-max-abs", type=float, default=limits.max_absolute_error
    )
    parser.add_argument(
        "--parity-max-rel-l2", type=float, default=limits.max_relative_l2
    )
    parser.add_argument(
        "--parity-min-cosine", type=float, default=limits.min_cosine_similarity
    )
    arguments = parser.parse_args()
    if (
        arguments.warmup < 5
        or arguments.iters < 20
        or arguments.resampler_warmup < 10
        or arguments.resampler_iters < 50
    ):
        parser.error(
            "Require full-encoder warmup>=5/iters>=20 and resampler warmup>=10/iters>=50"
        )
    else:
        pass
    if (
        arguments.parity_max_abs <= 0
        or arguments.parity_max_rel_l2 <= 0
        or not 0 <= arguments.parity_min_cosine <= 1
    ):
        parser.error(
            "Parity error limits must be positive; cosine limit must be in [0, 1]"
        )
    else:
        pass
    if arguments.mode == "census" and arguments.census_file is not None:
        parser.error(
            "census must generate fresh shapes; --census-file is for measurement modes"
        )
    else:
        return arguments


def main() -> None:
    arguments = parse_arguments()
    repository = Path(__file__).resolve().parents[2]
    if arguments.output_json is not None:
        arguments.output_json = artifact_path(arguments.output_json)
    else:
        pass
    if arguments.mode == "profile":
        arguments.profile_dir = artifact_path(arguments.profile_dir)
        arguments.profile_dir.mkdir(parents=True, exist_ok=True)
    else:
        pass
    if arguments.mode != "census":
        device = resolve_measurement_device(arguments.device)
        arguments.device = str(device)
        torch.cuda.set_device(device)
        print(f"measurement_device={arguments.device}", flush=True)
    else:
        pass
    requests = resolve_requests(
        arguments.cohort_file, arguments.sample_ids, arguments.videomme_repo
    )
    preprocessor = MiniCPMOPreprocessor(arguments.model_id, speech_enabled=False)
    model = inspect_model(
        preprocessor.model_dir, None, arguments.dtype, arguments.device
    )
    selection: Selection | None = None
    if arguments.census_file is not None:
        census = load_census(
            arguments.census_file,
            requests,
            arguments.model_id,
            arguments.videomme_repo,
            model.checkpoint_config_sha256,
        )
        selection = select_request(
            census, arguments.sample_id, arguments.sample_selection
        )
        selected_requests = [
            request for request in requests if request.sample_id == selection.sample_id
        ]
        cached = asyncio.run(preprocess_requests(preprocessor, selected_requests))
        previous = next(
            entry for entry in census if entry.sample_id == selection.sample_id
        )
        if (
            cached[0].census.tgt_sizes != previous.tgt_sizes
            or cached[0].census.patch_counts != previous.patch_counts
        ):
            raise ValueError("Production preprocessing shapes changed; rerun census")
        else:
            pass
    else:
        cached = asyncio.run(preprocess_requests(preprocessor, requests))
        census = [request.census for request in cached]
    report = RunReport(
        mode=arguments.mode,
        model_id=arguments.model_id,
        videomme_repo=arguments.videomme_repo,
        sample_ids=[request.sample_id for request in requests],
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        expected_base_sha=EXPECTED_BASE_SHA,
        model=model,
        parity_limits=ParityLimits(
            max_absolute_error=arguments.parity_max_abs,
            max_relative_l2=arguments.parity_max_rel_l2,
            min_cosine_similarity=arguments.parity_min_cosine,
        ),
        vision_batch_size=(
            arguments.vision_batch_size
            if arguments.mode == "decompose"
            else BASELINE_BATCH_SIZE
        ),
        census=census,
        census_summary=summarize_census(census),
    )
    if arguments.mode == "census":
        print("=== MODEL ===")
        print(json.dumps(asdict(model), indent=2))
        print("attention backend/pass_strided_qkv: not inspected in CPU-only census")
        assert report.census_summary is not None
        print_census(census, report.census_summary)
    else:
        if selection is None:
            selection = select_request(
                census, arguments.sample_id, arguments.sample_selection
            )
        else:
            pass
        selected = next(
            request for request in cached if request.sample_id == selection.sample_id
        )
        report.selection = selection
        print(
            f"=== SELECTION sample={selection.sample_id} ===\n"
            f"{selection.reason}; metric={selection.metric} "
            f"target={selection.target_attention_work_proxy:.1f} "
            f"actual={selection.actual_attention_work_proxy} "
            f"total_patches={selection.actual_total_patches}",
            flush=True,
        )
        encoder = MiniCPMOImageEncoder(
            preprocessor.model_dir, device=arguments.device, dtype=arguments.dtype
        )
        encoder.vision_batch_size = (
            arguments.vision_batch_size
            if arguments.mode == "decompose"
            else BASELINE_BATCH_SIZE
        )
        report.vision_batch_size = encoder.vision_batch_size
        report.model = inspect_model(
            preprocessor.model_dir, encoder, arguments.dtype, arguments.device
        )
        print("=== MODEL ===")
        print(json.dumps(asdict(report.model), indent=2), flush=True)
        operation = EncoderForward(encoder=encoder, inputs=selected.inputs)
        if arguments.mode == "decompose":
            print(
                f"=== DECOMPOSITION sample={selection.sample_id} "
                f"vision_batch_size={encoder.vision_batch_size} ===",
                flush=True,
            )
            phases = CudaPhases()
            with phase_hooks(encoder, phases):
                trial, output = benchmark_forward(
                    operation,
                    arm="decomposition",
                    batch_size=encoder.vision_batch_size,
                    device=encoder.device,
                    warmup=arguments.warmup,
                    iters=arguments.iters,
                    phases=phases,
                )
            if output is not None:
                with torch.inference_mode():
                    baseline_output = operation().detach().cpu()
                trial.parity = compare_outputs(
                    baseline_output, output, report.parity_limits
                )
            else:
                pass
            report.decomposition = trial
            print_trial(trial)
            for phase, milliseconds in trial.phases_ms.items():
                print(
                    f"{phase:18s} mean_ms={milliseconds.mean:.3f} median_ms={milliseconds.median:.3f} p95_ms={milliseconds.p95:.3f}"
                )
            if trial.gpu_ms is not None and trial.cpu_wall_ms is not None:
                print(
                    f"total_gpu          mean_ms={trial.gpu_ms.mean:.3f} median_ms={trial.gpu_ms.median:.3f}"
                )
                print(
                    f"cpu_wall_total     mean_ms={trial.cpu_wall_ms.mean:.3f} median_ms={trial.cpu_wall_ms.median:.3f}"
                )
            else:
                pass
            print("Phase timings include event/hook overhead", flush=True)
        elif arguments.mode == "vision-batch":
            print(
                f"=== VISION BATCH SWEEP sample={selection.sample_id} ===", flush=True
            )
            report.vision_batch = vision_batch_sweep(
                operation,
                arguments.vision_batch_sizes,
                arguments.warmup,
                arguments.iters,
                report.parity_limits,
            )
        elif arguments.mode == "resampler":
            print(
                f"=== RESAMPLER sample={selection.sample_id} vision_batch_size=16 ===",
                flush=True,
            )
            report.resampler = resampler_experiment(
                operation,
                arguments.warmup,
                arguments.iters,
                arguments.resampler_warmup,
                arguments.resampler_iters,
                report.parity_limits,
            )
            print(
                f"full_encoder_status={report.resampler.full_encoder_status}",
                flush=True,
            )
        else:
            print(
                f"=== PROFILE sample={selection.sample_id} vision_batch_size=16 ===",
                flush=True,
            )
            report.profile = profile_encoder(
                operation,
                arguments.profile_dir,
                arguments.warmup,
                arguments.profile_iters,
                not arguments.no_chrome_trace,
            )
            print(json.dumps(asdict(report.profile), indent=2), flush=True)
    trials = []
    if report.vision_batch is not None:
        trials.extend(report.vision_batch.visits)
    else:
        pass
    if report.decomposition is not None:
        trials.append(report.decomposition)
    else:
        pass
    if report.resampler is not None:
        trials.extend(
            [
                *report.resampler.isolated_visits,
                *report.resampler.full_encoder_visits,
            ]
        )
    else:
        pass
    report.substantial_drift = any(
        trial.parity is not None and trial.parity.substantial_drift for trial in trials
    )
    baseline_oom_arms = {
        "decomposition",
        "bs16-A",
        "bs16-B",
        "R0-A",
        "R0-B",
        "V0-A",
        "V0-B",
    }
    report.failed = report.substantial_drift or any(
        trial.status == "OOM" and trial.arm in baseline_oom_arms for trial in trials
    )
    if arguments.output_json is not None:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(
            json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
        )
        print(f"summary saved: {arguments.output_json}", flush=True)
    else:
        pass
    if report.failed:
        raise SystemExit(2)
    else:
        pass


if __name__ == "__main__":
    main()
else:
    pass

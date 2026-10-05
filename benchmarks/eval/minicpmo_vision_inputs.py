# SPDX-License-Identifier: Apache-2.0
"""Resolve frozen media and invoke the production MiniCPM-o preprocessor."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import TypeAdapter

from benchmarks.dataset.videomme import load_videomme_samples
from benchmarks.eval.minicpmo_vision_workload import (
    VIDEO_FPS,
    VIDEO_MAX_FRAMES,
    VIDEO_MAX_PIXELS,
    VISION_BATCH_SIZES,
    CachedRequest,
    CensusEntry,
    EncoderInputs,
    FrozenRequest,
    validate_frozen_ids,
)
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.proto.request import OmniRequest, StagePayload


def resolve_requests(
    cohort_file: Path | None, sample_ids: list[str], videomme_repo: str
) -> list[FrozenRequest]:
    if cohort_file is None:
        requests = [FrozenRequest(sample_id=sample_id) for sample_id in sample_ids]
    else:
        document: object = json.loads(cohort_file.read_text())
        if isinstance(document, dict):
            document = document.get(
                "samples", document.get("per_sample", document.get("sample_ids"))
            )
        else:
            pass
        decoded = TypeAdapter(list[str] | list[FrozenRequest]).validate_python(document)
        requests = [
            FrozenRequest(sample_id=record) if isinstance(record, str) else record
            for record in decoded
        ]
    validate_frozen_ids([request.sample_id for request in requests])
    if any(
        request.video_path is None or request.prompt is None for request in requests
    ):
        samples = load_videomme_samples(repo_id=videomme_repo)
        indexed_samples = {sample.sample_id: sample for sample in samples}
        missing = [
            request.sample_id
            for request in requests
            if request.sample_id not in indexed_samples
        ]
        if missing:
            raise ValueError(
                f"Frozen samples missing from Video-MME repo {videomme_repo}: {missing}"
            )
        else:
            for request in requests:
                sample = indexed_samples[request.sample_id]
                request.video_path = request.video_path or sample.video_path
                request.prompt = request.prompt or sample.prompt
    else:
        pass
    for request in requests:
        assert request.video_path is not None and request.prompt is not None
        video_path = Path(request.video_path).expanduser()
        if not video_path.is_absolute() and cohort_file is not None:
            video_path = cohort_file.parent / video_path
        else:
            pass
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Frozen sample {request.sample_id}: missing video {video_path}"
            )
        else:
            request.video_path = str(video_path.resolve())
    return requests


async def preprocess_requests(
    preprocessor: MiniCPMOPreprocessor, requests: list[FrozenRequest]
) -> list[CachedRequest]:
    cached_requests = []
    for request in requests:
        assert request.video_path is not None and request.prompt is not None
        payload = StagePayload(
            request_id=request.sample_id,
            request=OmniRequest(
                inputs={
                    "messages": [{"role": "user", "content": request.prompt}],
                    "videos": [request.video_path],
                    "video_fps": VIDEO_FPS,
                    "video_max_frames": VIDEO_MAX_FRAMES,
                    "video_max_pixels": VIDEO_MAX_PIXELS,
                    "use_audio_in_video": False,
                }
            ),
            data=None,
        )
        processed = await preprocessor(payload)
        if not isinstance(processed.data, dict):
            raise TypeError(f"Invalid production payload for {request.sample_id}")
        else:
            encoder_inputs = processed.data["encoder_inputs"]
        if not isinstance(encoder_inputs, dict):
            raise TypeError(f"Invalid encoder inputs for {request.sample_id}")
        else:
            inputs = EncoderInputs.model_validate(encoder_inputs["image_encoder"])
        target_sizes = inputs.tgt_sizes.detach().cpu()
        if (
            target_sizes.ndim != 2
            or target_sizes.shape != (len(inputs.pixel_values), 2)
            or not inputs.pixel_values
        ):
            raise ValueError(
                f"Invalid production vision shapes for {request.sample_id}: {target_sizes.shape}"
            )
        else:
            pass
        if not bool((target_sizes > 0).all()) or any(
            tensor.device.type != "cpu" for tensor in inputs.pixel_values
        ):
            raise ValueError(
                f"Expected positive target sizes and CPU pixel slices for {request.sample_id}"
            )
        else:
            pass
        patch_counts = [
            int(patch_count)
            for patch_count in (target_sizes[:, 0] * target_sizes[:, 1]).tolist()
        ]
        video_stat = Path(request.video_path).stat()
        census = CensusEntry(
            sample_id=request.sample_id,
            video_path=request.video_path,
            video_size_bytes=video_stat.st_size,
            video_mtime_ns=video_stat.st_mtime_ns,
            num_slices=len(inputs.pixel_values),
            tgt_sizes=target_sizes.tolist(),
            patch_counts=patch_counts,
            total_patches=sum(patch_counts),
            attention_work_proxy=sum(patch_count**2 for patch_count in patch_counts),
            max_patches=max(patch_counts),
            vpm_chunks={
                batch_size: math.ceil(len(patch_counts) / batch_size)
                for batch_size in VISION_BATCH_SIZES
            },
        )
        cached_requests.append(
            CachedRequest(sample_id=request.sample_id, inputs=inputs, census=census)
        )
        print(
            f"preprocessed sample={request.sample_id} slices={census.num_slices} patches={census.total_patches}",
            flush=True,
        )
    return cached_requests

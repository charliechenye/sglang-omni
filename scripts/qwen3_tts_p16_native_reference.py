#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run upstream Qwen3-TTS native Predictor frames from a sealed input artifact."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

P16_REFERENCE_ROOT_ENV = "SGLANG_OMNI_QTTS_P16_REFERENCE_ROOT"
P16_UPSTREAM_BASE_SHA = "a6e2fc6d98ab6abbed1c0ea40d0a7d6d038f3d74"
NATIVE_TALKER_SOURCE_RELATIVE_PATH = "sglang_omni/models/qwen3_tts/sglang_model.py"
NATIVE_REFERENCE_IMPORTS: dict[str, str] = {
    "talker": "sglang_omni/models/qwen3_tts/sglang_model.py",
    "engine_builder": "sglang_omni/models/qwen3_tts/engine_builder.py",
    "request_builders": "sglang_omni/models/qwen3_tts/request_builders.py",
    "scheduler": "sglang_omni/scheduling/omni_scheduler.py",
}
reference_root = os.environ.get(P16_REFERENCE_ROOT_ENV)
REPOSITORY_ROOT = (
    Path(reference_root).resolve()
    if reference_root is not None
    else Path(__file__).resolve().parents[1]
)
if str(REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import SchedulerRequest


class NativeReferenceSample(TypedDict):
    request_id: str
    uid: str
    ordinal: int
    seed: int
    code0: int
    hidden: torch.Tensor


class NativeReferenceRecord(TypedDict):
    request_id: str
    uid: str
    ordinal: int
    seed: int
    codes_sha256: str
    feedback_sha256: str


def native_reference_talker(
    scheduler: OmniScheduler[Qwen3TTSSGLangRequestData],
) -> Qwen3TTSTalker:
    model_runner = scheduler.model_runner
    if model_runner is None:
        raise RuntimeError("native reference scheduler has no model runner")
    talker = model_runner.model
    if not isinstance(talker, Qwen3TTSTalker):
        raise RuntimeError("native reference scheduler model is not a Qwen3-TTS Talker")
    return talker


def native_reference_import_source_paths(talker: Qwen3TTSTalker) -> dict[str, str]:
    imported = {
        "talker": type(talker),
        "engine_builder": Qwen3TtsEngineBuilder,
        "request_builders": Qwen3TTSSGLangRequestData,
        "scheduler": OmniScheduler,
    }
    paths = {}
    for name, imported_type in imported.items():
        source_path = Path(inspect.getfile(imported_type)).resolve(strict=True)
        expected_path = REPOSITORY_ROOT / NATIVE_REFERENCE_IMPORTS[name]
        if source_path != expected_path:
            raise RuntimeError(
                f"native reference imported {name} from {source_path}, expected {expected_path}"
            )
        paths[name] = str(source_path)
    return paths


class NativeReferenceInputIdentity(TypedDict):
    upstream_commit: str
    model_revision: str
    model_identity: dict[str, object]
    cohort_sha256: str
    sampling: dict[str, bool | float | int]
    graph_configuration: dict[str, int]


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def native_reference_output_rows(
    sample: NativeReferenceSample,
    codes: torch.Tensor,
    feedback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, NativeReferenceRecord]:
    if codes.ndim == 0 or codes.shape[0] != 1:
        raise ValueError("native reference code output must contain one batch row")
    elif feedback.ndim == 0 or feedback.shape[0] != 1:
        raise ValueError("native reference feedback output must contain one batch row")
    codes_row = codes.detach().cpu()[0].contiguous().clone()
    feedback_row = feedback.detach().cpu()[0].contiguous().clone()
    record: NativeReferenceRecord = {
        "request_id": sample["request_id"],
        "uid": sample["uid"],
        "ordinal": sample["ordinal"],
        "seed": sample["seed"],
        "codes_sha256": tensor_sha256(codes_row),
        "feedback_sha256": tensor_sha256(feedback_row),
    }
    return codes_row, feedback_row, record


def validate_model_identity(model_path: Path, identity: Mapping[str, object]) -> None:
    model_name = identity.get("model")
    snapshot_name = identity.get("snapshot")
    if not isinstance(model_name, str) or not isinstance(snapshot_name, str):
        raise RuntimeError("native reference model identity is incomplete")
    if model_name != "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice":
        raise RuntimeError("native reference model name is not the pinned model")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("native reference model config is missing")
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("native reference model weight shards are missing")
    actual_model_identity = {
        "model": model_name,
        "snapshot": snapshot_name,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "shards": {
            shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()
            for shard in shards
        },
    }
    if model_path.name != snapshot_name or actual_model_identity != identity:
        raise RuntimeError("native reference model snapshot identity does not match")


def validate_reference_identity(identity: Mapping[str, object]) -> None:
    if identity.get("upstream_commit") != P16_UPSTREAM_BASE_SHA:
        raise RuntimeError("native reference input has the wrong upstream commit")
    elif identity.get("sampling") != {
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
    }:
        raise RuntimeError("native reference input has unsupported sampling settings")
    elif identity.get("graph_configuration") != {
        "max_running_requests": 1,
        "cuda_graph_max_bs": 1,
        "tp_size": 1,
    }:
        raise RuntimeError("native reference input has unsupported graph settings")


def parse_reference_sample(sample: object) -> NativeReferenceSample:
    if not isinstance(sample, dict):
        raise RuntimeError("native reference sample is not a mapping")
    request_id = sample.get("request_id")
    uid = sample.get("uid")
    ordinal = sample.get("ordinal")
    seed = sample.get("seed")
    code0 = sample.get("code0")
    hidden = sample.get("hidden")
    if (
        not isinstance(request_id, str)
        or not request_id
        or not isinstance(uid, str)
        or not uid
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(code0, bool)
        or not isinstance(code0, int)
        or not 0 <= code0 < 2048
        or not isinstance(hidden, torch.Tensor)
        or hidden.shape != (2048,)
        or hidden.dtype is not torch.bfloat16
        or not bool(torch.isfinite(hidden).all())
    ):
        raise RuntimeError("native reference sample fields are invalid")
    return {
        "request_id": request_id,
        "uid": uid,
        "ordinal": ordinal,
        "seed": seed,
        "code0": code0,
        "hidden": hidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    arguments = parser.parse_args()

    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    if actual_commit != arguments.expected_commit:
        raise RuntimeError(
            f"upstream reference checkout is {actual_commit}, expected {arguments.expected_commit}"
        )

    artifact = torch.load(arguments.inputs, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise RuntimeError("native reference input artifact is not a mapping")
    identity = artifact.get("identity")
    inputs = artifact.get("inputs")
    if not isinstance(identity, dict) or not isinstance(inputs, list):
        raise RuntimeError("native reference input artifact is malformed")
    reference_identity: NativeReferenceInputIdentity = identity
    validate_reference_identity(reference_identity)
    model_path = arguments.model.resolve(strict=True)
    if reference_identity["model_revision"] != arguments.model_revision:
        raise RuntimeError("native reference input model revision does not match")
    validate_model_identity(model_path, reference_identity["model_identity"])
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("independent upstream native reference requires CUDA")

    for name in (
        "SGLANG_OMNI_QTTS_P16_ENABLE",
        "SGLANG_OMNI_QTTS_P16_DRAFT_PATH",
        "SGLANG_OMNI_QTTS_P16_DRAFT_SHA256",
        "SGLANG_OMNI_QTTS_P16_TELEMETRY_PATH",
    ):
        os.environ.pop(name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    builder = Qwen3TtsEngineBuilder(leading_silence_mask_frames=0)
    scheduler = builder.build(
        str(model_path),
        device=str(device),
        gpu_id=0,
        dtype="bfloat16",
        server_args_overrides={
            "tp_size": 1,
            "max_running_requests": 1,
            "max_queued_requests": 4,
            "cuda_graph_max_bs": 1,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
        },
    )
    try:
        talker = native_reference_talker(scheduler)
        import_source_paths = native_reference_import_source_paths(talker)
        talker_source_path = Path(import_source_paths["talker"])

        code_outputs: list[torch.Tensor] = []
        feedback_outputs: list[torch.Tensor] = []
        records: list[NativeReferenceRecord] = []
        graph_stats_before = talker.predictor_graph_stats()
        for untrusted_sample in inputs:
            sample = parse_reference_sample(untrusted_sample)
            request_data = Qwen3TTSSGLangRequestData(
                semantic_sampling_seed=int(sample["seed"]),
                subtalker_dosample=True,
                subtalker_temperature=0.9,
                subtalker_top_p=1.0,
                subtalker_top_k=50,
                subtalker_sampling_seed=int(sample["seed"]),
            )
            request = SchedulerRequest(
                request_id=sample["request_id"], data=request_data
            )
            talker.prepare_decode_buffers([request])
            layer0_codes = torch.tensor(
                [[int(sample["code0"])]], device=device, dtype=torch.long
            )
            hidden = sample["hidden"].to(device=device, dtype=torch.bfloat16)
            talker_hidden = hidden.view(1, 1, -1).contiguous()
            semantic_positions = torch.tensor(
                [int(sample["ordinal"])], device=device, dtype=torch.long
            )
            codes, feedback = talker.code_predictor_forward(
                layer0_codes,
                talker_hidden,
                semantic_positions=semantic_positions,
            )
            torch.cuda.synchronize(device)
            code_row, feedback_row, record = native_reference_output_rows(
                sample, codes, feedback
            )
            code_outputs.append(code_row)
            feedback_outputs.append(feedback_row)
            records.append(record)
        graph_stats_after = talker.predictor_graph_stats()
        actual_replays = graph_stats_after["replays"] - graph_stats_before["replays"]
        if (
            graph_stats_after["enabled"] is not True
            or actual_replays != len(inputs)
            or graph_stats_after["eager_runs"] != graph_stats_before["eager_runs"]
        ):
            raise RuntimeError(
                "upstream native reference did not replay a CUDA Graph for every input"
            )
        output = {
            "identity": identity,
            "metadata": {
                "upstream_commit": actual_commit,
                "model_revision": arguments.model_revision,
                "model_source_path": str(talker_source_path),
                "model_source_path_relative": talker_source_path.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "reference_source_root": str(REPOSITORY_ROOT),
                "imported_source_paths": import_source_paths,
                "cohort_sha256": identity["cohort_sha256"],
                "sampling": identity["sampling"],
                "graph_configuration": {
                    "max_running_requests": 1,
                    "cuda_graph_max_bs": 1,
                    "tp_size": 1,
                },
                "native_graph": {
                    "enabled": graph_stats_after["enabled"],
                    "startup_graph_count": graph_stats_after["startup_graph_count"],
                    "replay_count": actual_replays,
                    "eager_run_delta": 0,
                },
                "records": records,
            },
            "codes": torch.stack(code_outputs),
            "feedback": torch.stack(feedback_outputs),
        }
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, arguments.out)
    finally:
        scheduler.stop()
        del scheduler

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

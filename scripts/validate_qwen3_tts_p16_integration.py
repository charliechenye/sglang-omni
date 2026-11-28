#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit the P16 production wrapper on the pinned Qwen3-TTS validation cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_tts.engine_builder import (
    P16_DRAFT_PATH_ENV,
    P16_DRAFT_SHA256_ENV,
    P16_ENABLE_ENV,
    P16_MODEL_REVISION,
    Qwen3TtsEngineBuilder,
)
from sglang_omni.models.qwen3_tts.request_builders import (
    Qwen3TTSSGLangRequestData,
    normalize_qwen3_tts_seed,
)
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    build_speculative_probability_candidates,
    uses_split_top_k,
)
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.models.qwen3_tts.speculative_draft import (
    P16_DRAFT_SHA256,
    load_p16_draft_checkpoint,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.utils.json import JsonValue
from sglang_omni.utils.predictor_grouped_heads import grouped_lm_head_warp_accumulate

P12_MANIFEST_SHA256 = "87ae5928d52fe1675b43c24ea15d7e054df3e0b4810e1c3ca1b42a4a4a040957"
P12_TEACHER_BASE = "caf13d4d4bd0eab7edb7186675fb7307a5a0ecdc"
P12_TEACHER_MODE = "native_fused_fixed_prefix"
P15_COHORT_SHA256 = "c823f32c837b4979c0e74e4b1f4c60b53f40089e9f2a7119be6b11bb8ce24725"
P16_SCHEMA = "qtts-p16-integration-v1"
P16_UPSTREAM_BASE_SHA = "a6e2fc6d98ab6abbed1c0ea40d0a7d6d038f3d74"
P16_NATIVE_REFERENCE_ROOT_ENV = "SGLANG_OMNI_QTTS_P16_REFERENCE_ROOT"
P16_NATIVE_REFERENCE_IMPORTS: dict[str, str] = {
    "talker": "sglang_omni/models/qwen3_tts/sglang_model.py",
    "engine_builder": "sglang_omni/models/qwen3_tts/engine_builder.py",
    "request_builders": "sglang_omni/models/qwen3_tts/request_builders.py",
    "scheduler": "sglang_omni/scheduling/omni_scheduler.py",
}
P16_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
P16_GROUPS = 16
P16_TALKER_HIDDEN_SIZE = 2048
P16_CODE_PREDICTOR_HIDDEN_SIZE = 1024
P16_VOCAB_SIZE = 2048
P16_VERIFY_WIDTH = 4
P16_MAX_TOP_K = 50
P16_CANDIDATE_WIDTH = 64

AuditValue: TypeAlias = torch.Tensor | int
AuditRecord: TypeAlias = dict[str, AuditValue]


@dataclass(frozen=True)
class ValidationArguments:
    model: Path
    draft: Path
    p12_root: Path
    p15_root: Path
    frames: int
    seeds: int
    out: Path


@dataclass(frozen=True)
class ValidationFrame:
    uid: str
    ordinal: int
    hidden: torch.Tensor
    code0: int


@dataclass
class CapturedFrame:
    opening: AuditRecord | None = None
    rounds: list[AuditRecord] = field(default_factory=list)


class P16AuditCapture:
    def __init__(self) -> None:
        self.current: CapturedFrame | None = None

    def begin(self) -> CapturedFrame:
        if self.current is not None:
            raise RuntimeError(
                "P16 audit frame was not released after sequential replay"
            )
        frame = CapturedFrame()
        self.current = frame
        return frame

    def release(self, frame: CapturedFrame) -> None:
        if self.current is not frame:
            raise RuntimeError("P16 audit release does not match the active frame")
        self.current = None

    def __call__(
        self,
        event: Literal["opening", "round"],
        tensors: dict[str, torch.Tensor | int],
    ) -> None:
        record: AuditRecord = {}
        for name, value in tensors.items():
            if isinstance(value, torch.Tensor):
                record[name] = value.detach().cpu().clone()
            else:
                record[name] = int(value)
        if event == "opening":
            if self.current is None:
                raise RuntimeError("P16 opening audit arrived without a frame")
            else:
                self.current.opening = record
        else:
            if self.current is None or self.current.opening is None:
                raise RuntimeError("P16 round audit arrived before its opening event")
            self.current.rounds.append(record)


@dataclass
class AuditTotals:
    native_target_rows: int = 0
    compact_probability_rows: int = 0
    proposal_rows: int = 0
    valid_kv_rounds: int = 0
    rounds: int = 0
    rejections: int = 0
    bonuses: int = 0
    width_one_rounds: int = 0


class ValidationFailure(RuntimeError):
    def __init__(self, check: str, detail: str, context: dict[str, JsonValue]) -> None:
        self.check = check
        self.detail = detail
        self.context = context
        super().__init__(f"{check}: {detail}")


def require(
    condition: bool,
    check: str,
    detail: str,
    context: dict[str, JsonValue] | None = None,
) -> None:
    if not condition:
        raise ValidationFailure(check, detail, context or {})


def parse_arguments() -> ValidationArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--p12-root", type=Path, required=True)
    parser.add_argument("--p15-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(4, 128), required=True)
    parser.add_argument("--seeds", type=int, choices=(1, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parsed = parser.parse_args()
    return ValidationArguments(
        model=parsed.model,
        draft=parsed.draft,
        p12_root=parsed.p12_root,
        p15_root=parsed.p15_root,
        frames=parsed.frames,
        seeds=parsed.seeds,
        out=parsed.out,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path: Path) -> JsonValue:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON artifact {path}: {exc}") from exc


def require_mapping(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_string(mapping: dict[str, JsonValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"JSON field {key} must be a string")
    return value


def require_integer(mapping: dict[str, JsonValue], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"JSON field {key} must be an integer")
    return value


def tensor_field(record: AuditRecord, name: str) -> torch.Tensor:
    value = record.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"P16 audit field {name} is not a tensor")
    return value


def integer_field(record: AuditRecord, name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"P16 audit field {name} is not an integer")
    return value


def model_identity(model_path: Path) -> dict[str, JsonValue]:
    resolved = model_path.resolve(strict=True)
    require(
        resolved.is_dir() and resolved.name == P16_MODEL_REVISION,
        "pinned_model",
        "--model must be an existing pinned snapshot directory",
    )
    config_path = resolved / "config.json"
    shards = sorted(resolved.glob("*.safetensors"))
    require(
        config_path.is_file() and bool(shards),
        "pinned_model",
        "model config or weight shards are missing",
    )
    return {
        "model": P16_MODEL,
        "snapshot": P16_MODEL_REVISION,
        "config_sha256": sha256_file(config_path),
        "shards": {shard.name: sha256_file(shard) for shard in shards},
    }


def validation_rows(
    p12_root: Path,
    model_identity_value: dict[str, JsonValue],
    frame_count: int,
    seed_count: int,
) -> tuple[list[ValidationFrame], list[ValidationFrame], str, dict[str, JsonValue]]:
    sealed_root = p12_root / "sealed"
    manifest_path = sealed_root / "manifest.json"
    manifest = require_mapping(read_json(manifest_path), "P12 sealed manifest")
    require(
        sha256_file(manifest_path) == P12_MANIFEST_SHA256,
        "p12_manifest",
        "P12 sealed manifest hash does not match the pinned dataset",
    )
    require(
        require_string(manifest, "status") == "P12_RAW_DATASET_COMPLETE",
        "p12_manifest",
        "P12 sealed manifest is not complete",
    )
    admission_path = sealed_root / "teacher-admission.json"
    admission = require_mapping(read_json(admission_path), "P12 teacher admission")
    require(
        sha256_file(admission_path)
        == require_string(manifest, "teacher_admission_sha256"),
        "p12_teacher_admission",
        "P12 teacher admission hash differs from the sealed manifest",
    )
    require(
        require_string(admission, "status") == "P12_NATIVE_TEACHER_ADMITTED"
        and require_string(admission, "teacher_mode") == P12_TEACHER_MODE
        and require_integer(admission, "batch_size") == 32
        and require_string(admission, "native_base") == P12_TEACHER_BASE,
        "p12_teacher_admission",
        "P12 admission does not identify the pinned native teacher",
    )
    admission_identity = require_mapping(
        admission.get("identity"), "P12 model identity"
    )
    require(
        admission_identity == model_identity_value,
        "p12_model_identity",
        "P12 validation data were admitted for a different model snapshot",
    )
    require(
        manifest.get("identity") == admission_identity,
        "p12_model_identity",
        "P12 sealed manifest model identity differs from teacher admission",
    )
    shard_values = manifest.get("shards")
    if not isinstance(shard_values, list):
        raise RuntimeError("P12 manifest shards must be a list")
    validation_shards: list[tuple[Path, str]] = []
    for shard_value in shard_values:
        shard = require_mapping(shard_value, "P12 shard record")
        if require_string(shard, "split") == "validation":
            shard_name = require_string(shard, "name")
            shard_path = (sealed_root / shard_name).resolve(strict=True)
            require(
                shard_path.is_relative_to(sealed_root.resolve()),
                "p12_validation_shard",
                "P12 validation shard path escapes the sealed directory",
            )
            shard_hash = require_string(shard, "sha256")
            require(
                sha256_file(shard_path) == shard_hash,
                "p12_validation_shard",
                f"P12 validation shard hash mismatch: {shard_name}",
            )
            validation_shards.append((shard_path, shard_hash))
    counts = require_mapping(manifest.get("counts"), "P12 split counts")
    validation_count = sum(
        value
        for name, value in counts.items()
        if name.startswith("validation/") and isinstance(value, int)
    )
    require(
        validation_count == 2082 and bool(validation_shards),
        "p12_validation_rows",
        "P12 must contain the pinned 2082 validation frames",
    )

    sealed_frames: list[ValidationFrame] = []
    seen_frames: set[tuple[str, int]] = set()
    for shard_path, _ in validation_shards:
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != "p12-sealed-v1":
            raise RuntimeError(f"invalid sealed validation shard {shard_path.name}")
        hidden = payload.get("hidden")
        codes = payload.get("codes")
        requests = payload.get("requests")
        if not isinstance(hidden, torch.Tensor) or not isinstance(codes, torch.Tensor):
            raise RuntimeError(f"validation tensors are missing in {shard_path.name}")
        elif (
            hidden.ndim != 2
            or hidden.shape[1] != P16_TALKER_HIDDEN_SIZE
            or hidden.dtype is not torch.bfloat16
        ):
            raise RuntimeError(
                f"validation hidden rows have the wrong shape or dtype in {shard_path.name}"
            )
        elif codes.ndim != 2 or codes.shape[0] != hidden.shape[0] or codes.shape[1] < 1:
            raise RuntimeError(
                f"validation code rows have the wrong shape in {shard_path.name}"
            )
        elif not torch.isfinite(hidden).all():
            raise RuntimeError(
                f"validation hidden rows contain nonfinite values in {shard_path.name}"
            )
        elif not isinstance(requests, list):
            raise RuntimeError(
                f"validation request index is missing in {shard_path.name}"
            )
        for request_value in requests:
            request = require_mapping(request_value, "P12 validation request")
            uid = require_string(request, "uid")
            offset = require_integer(request, "offset")
            frame_total = require_integer(request, "frames")
            require(
                offset >= 0
                and frame_total >= 0
                and offset + frame_total <= hidden.shape[0],
                "p12_validation_rows",
                f"P12 request frame range is invalid for {uid}",
            )
            for ordinal in range(frame_total):
                row_index = offset + ordinal
                code0 = int(codes[row_index, 0])
                require(
                    0 <= code0 < P16_VOCAB_SIZE,
                    "p12_validation_rows",
                    f"P12 codebook-0 token is out of bounds for {uid}:{ordinal}",
                )
                identity = (uid, ordinal)
                require(
                    identity not in seen_frames,
                    "p12_validation_rows",
                    "duplicate P12 frame identity",
                )
                seen_frames.add(identity)
                sealed_frames.append(
                    ValidationFrame(
                        uid=uid,
                        ordinal=ordinal,
                        hidden=hidden[row_index].clone(),
                        code0=code0,
                    )
                )
    require(
        len(sealed_frames) == 2082,
        "p12_validation_rows",
        "P12 validation frame count is not 2082",
    )
    cohort = [
        sealed_frames[index * len(sealed_frames) // frame_count]
        for index in range(frame_count)
    ]
    identity_value: dict[str, JsonValue] = {
        "p12_manifest": P12_MANIFEST_SHA256,
        "teacher_admission_sha256": sha256_file(admission_path),
        "validation_shards": [shard_hash for _, shard_hash in validation_shards],
        "cohort": [[frame.uid, frame.ordinal] for frame in cohort],
        "seeds": seed_count,
    }
    cohort_sha = json_digest(json.dumps(identity_value, sort_keys=True))
    if frame_count == 128 and seed_count == 4:
        require(
            cohort_sha == P15_COHORT_SHA256,
            "p14_cohort",
            f"full P14 cohort hash differs: {cohort_sha}",
        )
    return sealed_frames, cohort, cohort_sha, identity_value


def seed_for(uid: str, ordinal: int, repeat: int) -> int:
    return int(json_digest(f"p14-v1:{uid}:{ordinal}:{repeat}")[:15], 16)


def validate_p15_gate(p15_root: Path) -> tuple[dict[str, JsonValue], Path]:
    if p15_root.is_file():
        candidates = [p15_root]
    else:
        candidates = sorted(p15_root.rglob("decision.json"))
    passing: list[tuple[dict[str, JsonValue], Path]] = []
    for path in candidates:
        try:
            decision = require_mapping(read_json(path), "P15A decision")
        except RuntimeError:
            continue
        identity = require_mapping(decision.get("identity"), "P15A identity")
        if (
            decision.get("status") == "P15A_NATIVE_HEAD_PASS"
            and identity.get("cohort_sha256") == P15_COHORT_SHA256
        ):
            passing.append((decision, path))
    require(
        len(passing) == 1,
        "p15_gate",
        "exactly one full pinned P15A head pass is required",
    )
    decision, decision_path = passing[0]
    checks = require_mapping(decision.get("checks"), "P15A checks")
    require(
        checks.get("full_128x4_audit") is True
        and checks.get("B_exact_target") is True
        and checks.get("C_exact_target") is True
        and checks.get("B_C_exact_trace") is True,
        "p15_gate",
        "P15A did not pass the full target and trajectory parity gates",
    )
    identity = require_mapping(decision.get("identity"), "P15A identity")
    arms = identity.get("arms")
    require(
        isinstance(arms, list) and "B" in arms and "C" in arms,
        "p15_gate",
        "P15A pass does not include packed-native and sequential arms",
    )
    return decision, decision_path


def require_h200() -> torch.device:
    require(torch.cuda.is_available(), "h200_runtime", "CUDA is unavailable")
    require(
        torch.cuda.device_count() == 1,
        "h200_runtime",
        "exactly one visible H200 is required",
    )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    require(
        "H200" in properties.name, "h200_runtime", f"visible GPU is {properties.name}"
    )
    require(
        torch.cuda.get_device_capability(device) == (9, 0),
        "h200_runtime",
        "P16 requires an SM90 H200",
    )
    require(
        torch.cuda.is_bf16_supported(),
        "h200_runtime",
        "BF16 CUDA execution is unavailable",
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def current_git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    )
    require(
        not status, "candidate_tree", "validator requires a clean committed worktree"
    )
    return sha


def write_json_atomic(path: Path, value: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_talker(
    model_path: Path,
    device: torch.device,
    *,
    enabled: bool,
    draft_path: Path | None = None,
    max_running_requests: int = 1,
) -> tuple[OmniScheduler[Qwen3TTSSGLangRequestData], Qwen3TTSTalker]:
    variable_names = (
        P16_ENABLE_ENV,
        P16_DRAFT_PATH_ENV,
        P16_DRAFT_SHA256_ENV,
        "SGLANG_OMNI_QTTS_P16_TELEMETRY_PATH",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    prior_environment = {name: os.environ.get(name) for name in variable_names}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.pop("SGLANG_OMNI_QTTS_P16_TELEMETRY_PATH", None)
    if enabled:
        if draft_path is None:
            raise ValueError("enabled P16 construction requires a local draft file")
        os.environ[P16_ENABLE_ENV] = "1"
        os.environ[P16_DRAFT_PATH_ENV] = str(draft_path.resolve(strict=True))
        os.environ[P16_DRAFT_SHA256_ENV] = P16_DRAFT_SHA256
    else:
        os.environ.pop(P16_ENABLE_ENV, None)
        os.environ[P16_DRAFT_PATH_ENV] = "/p16-disabled-draft-must-not-be-read"
        os.environ[P16_DRAFT_SHA256_ENV] = "invalid-disabled-digest"
    try:
        builder = Qwen3TtsEngineBuilder(leading_silence_mask_frames=0)
        scheduler = builder.build(
            str(model_path),
            device=device.type,
            gpu_id=device.index or 0,
            dtype="bfloat16",
            server_args_overrides={
                "tp_size": 1,
                "max_running_requests": max_running_requests,
                "max_queued_requests": 4,
                "cuda_graph_max_bs": max_running_requests,
                "disable_cuda_graph": False,
                "disable_overlap_schedule": True,
                "enable_torch_compile": False,
            },
        )
        talker = builder.talker_model
        if talker is None:
            raise RuntimeError("production Qwen3-TTS builder did not retain its Talker")
        return scheduler, talker
    finally:
        for name, value in prior_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def close_runtime(
    scheduler: OmniScheduler[Qwen3TTSSGLangRequestData],
) -> None:
    scheduler.stop()
    del scheduler
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def create_request(
    request_id: str,
    seed: int,
    *,
    top_k: int = P16_MAX_TOP_K,
    temperature: float = 0.9,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> SchedulerRequest:
    request_data = Qwen3TTSSGLangRequestData(
        semantic_sampling_seed=seed,
        subtalker_sampling_seed=seed,
        subtalker_dosample=do_sample,
        subtalker_temperature=temperature,
        subtalker_top_p=top_p,
        subtalker_top_k=top_k,
    )
    return SchedulerRequest(request_id=request_id, data=request_data)


def input_tensors(
    frames: list[ValidationFrame], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer0_codes = torch.tensor(
        [[frame.code0] for frame in frames], device=device, dtype=torch.long
    )
    talker_hidden = torch.stack([frame.hidden for frame in frames]).to(
        device=device,
        dtype=torch.bfloat16,
    )
    talker_hidden = talker_hidden.unsqueeze(1).contiguous()
    semantic_positions = torch.tensor(
        [frame.ordinal for frame in frames], device=device, dtype=torch.long
    )
    return layer0_codes, talker_hidden, semantic_positions


def run_public_predictor(
    talker: Qwen3TTSTalker,
    frames: list[ValidationFrame],
    requests: list[SchedulerRequest],
) -> tuple[torch.Tensor, torch.Tensor]:
    talker.prepare_decode_buffers(requests)
    layer0_codes, talker_hidden, semantic_positions = input_tensors(
        frames, talker.predictor_device
    )
    codes, summed_embeddings = talker.code_predictor_forward(
        layer0_codes,
        talker_hidden,
        semantic_positions=semantic_positions,
    )
    torch.cuda.synchronize(talker.predictor_device)
    return codes.detach().cpu().clone(), summed_embeddings.detach().cpu().clone()


def run_native_incremental(
    talker: Qwen3TTSTalker,
    frames: list[ValidationFrame],
    requests: list[SchedulerRequest],
) -> tuple[torch.Tensor, torch.Tensor]:
    talker.prepare_decode_buffers(requests)
    layer0_codes, talker_hidden, semantic_positions = input_tensors(
        frames, talker.predictor_device
    )
    codes, summed_embeddings = talker.code_predictor_forward_incremental(
        layer0_codes,
        talker_hidden,
        semantic_positions=semantic_positions,
    )
    torch.cuda.synchronize(talker.predictor_device)
    return codes.detach().cpu().clone(), summed_embeddings.detach().cpu().clone()


def build_compact_probabilities(
    logits: torch.Tensor,
    *,
    active_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = int(logits.shape[0])
    device = logits.device
    temperatures = torch.full((row_count,), 0.9, device=device, dtype=torch.float32)
    top_ks = torch.full((row_count,), P16_MAX_TOP_K, device=device, dtype=torch.long)
    top_ps = torch.ones((row_count,), device=device, dtype=torch.float32)
    if active_rows is None:
        active_rows = torch.ones((row_count,), device=device, dtype=torch.bool)
    candidates = build_speculative_probability_candidates(
        logits.contiguous(),
        temperatures,
        top_ks,
        top_ps,
        active_rows,
        max_top_k=P16_MAX_TOP_K,
        has_top_p=False,
        has_unbounded_top_k=False,
    )
    if candidates is None:
        raise RuntimeError("split Top-K probability builder rejected an audited shape")
    probabilities, token_ids = candidates
    require(
        uses_split_top_k(logits, P16_MAX_TOP_K, P16_CANDIDATE_WIDTH),
        "compact_probability",
        "validator did not execute split Top-K",
    )
    require(
        probabilities.shape == (row_count, P16_CANDIDATE_WIDTH)
        and token_ids.shape == probabilities.shape,
        "compact_probability",
        "compact probability support has an unexpected shape",
    )
    return probabilities, token_ids


def require_compact_support(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    context: dict[str, JsonValue],
) -> None:
    require(
        probabilities.ndim == 2 and token_ids.shape == probabilities.shape,
        "probability_support",
        "probability and token support rows are misaligned",
        context,
    )
    require(
        bool(torch.isfinite(probabilities).all())
        and bool((probabilities >= 0.0).all()),
        "probability_support",
        "compact probabilities contain nonfinite or negative mass",
        context,
    )
    positive = probabilities > 0.0
    require(
        bool(
            ((token_ids[positive] >= 0) & (token_ids[positive] < P16_VOCAB_SIZE)).all()
        ),
        "probability_support",
        "positive probability mass has an out-of-vocabulary token ID",
        context,
    )
    masses = probabilities.sum(dim=1)
    require(
        bool(torch.all(torch.abs(masses - 1.0) <= 1.0e-4)),
        "probability_support",
        "active compact probability rows are not normalized",
        context,
    )


def sample_compact_row(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    uniform: float,
) -> int:
    require(
        0.0 <= uniform < 1.0, "rng_domain", "compact sampling uniform is outside [0, 1)"
    )
    cumulative = torch.cumsum(probabilities.to(torch.float32), dim=0)
    positive = probabilities > 0.0
    positive_ranks = torch.nonzero(positive, as_tuple=False).flatten()
    require(
        positive_ranks.numel() > 0,
        "probability_support",
        "compact row has empty positive support",
    )
    rank = int(torch.count_nonzero(cumulative <= uniform))
    rank = min(rank, int(positive_ranks[-1]))
    return int(token_ids[rank])


def probability_at(
    probabilities: torch.Tensor, token_ids: torch.Tensor, token: int
) -> torch.Tensor:
    return torch.where(
        token_ids == token, probabilities, torch.zeros_like(probabilities)
    ).sum()


def hash_tensor(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(
        str((str(contiguous.dtype), tuple(contiguous.shape))).encode()
    )
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def context_for(
    frame: ValidationFrame, seed: int, round_depth: int | None = None
) -> dict[str, JsonValue]:
    context: dict[str, JsonValue] = {
        "uid": frame.uid,
        "ordinal": frame.ordinal,
        "seed": seed,
    }
    if round_depth is not None:
        context["round_depth"] = round_depth
    return context


def assert_tensor_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    check: str,
    context: dict[str, JsonValue],
) -> None:
    require(
        torch.equal(actual, expected),
        check,
        f"tensor mismatch; actual={hash_tensor(actual)} expected={hash_tensor(expected)}",
        context,
    )


def capture_frame(
    talker: Qwen3TTSTalker,
    frame: ValidationFrame,
    request: SchedulerRequest,
    seed: int,
    audit_capture: P16AuditCapture,
) -> tuple[torch.Tensor, torch.Tensor, CapturedFrame, dict[str, int]]:
    request_data = request.data
    if not isinstance(request_data, Qwen3TTSSGLangRequestData):
        raise RuntimeError(
            "P16 request data did not use the Qwen3-TTS request contract"
        )
    if talker.code_predictor_spec_audit_observer is not audit_capture:
        talker.set_code_predictor_speculative_audit_observer(audit_capture)
    talker.prepare_decode_buffers([request])
    expected_seed = int(request_data.subtalker_sampling_seed)
    require(
        talker.sub_speculative_seed_tensor is not None
        and int(talker.sub_speculative_seed_tensor[0]) == expected_seed,
        "request_seed_staging",
        "request-owned speculative seed differs from the stable subtalker seed",
        context_for(frame, seed),
    )
    layer0_codes, talker_hidden, semantic_positions = input_tensors(
        [frame], talker.predictor_device
    )
    eligibility_reason = talker.code_predictor_p16_frame_ineligibility_reason(
        layer0_codes,
        talker_hidden,
        semantic_positions,
    )
    require(
        eligibility_reason is None,
        "p16_eligibility",
        f"cohort frame is unexpectedly ineligible: {eligibility_reason}",
        context_for(frame, seed),
    )
    audit_frame = audit_capture.begin()
    counters_before = talker.p16_telemetry_counters()
    fallback_before = talker.code_predictor_spec_native_fallback_count
    layer0_codes, talker_hidden, semantic_positions = input_tensors(
        [frame], talker.predictor_device
    )
    output_codes, summed_embeddings = talker.code_predictor_forward(
        layer0_codes,
        talker_hidden,
        semantic_positions=semantic_positions,
    )
    torch.cuda.synchronize(talker.predictor_device)
    counters_after = talker.p16_telemetry_counters()
    counter_delta = {
        name: counters_after[name] - counters_before[name] for name in counters_before
    }
    require(
        talker.code_predictor_spec_native_fallback_count == fallback_before,
        "graph_replays_no_fallback",
        "eligible P16 frame used the native fallback",
        context_for(frame, seed),
    )
    require(
        counter_delta["frames"] == 1
        and counter_delta["spec_openings"] == 1
        and counter_delta["rounds"] == counter_delta["spec_graph_replays"]
        and counter_delta["rounds"] > 0
        and counter_delta["committed"] == 14,
        "graph_replays_no_fallback",
        f"speculative frame counters are inconsistent: {counter_delta}",
        context_for(frame, seed),
    )
    require(
        counter_delta["graph_capture_failures"] == 0
        and counter_delta["eager_spec_rounds"] == 0
        and counter_delta["invalid_probability_rows"] == 0
        and counter_delta["cuda_errors"] == 0,
        "graph_replays_no_fallback",
        f"P16 serving path recorded a fallback or error: {counter_delta}",
        context_for(frame, seed),
    )
    require(
        audit_frame.opening is not None
        and len(audit_frame.rounds) == counter_delta["rounds"],
        "audit_trace",
        "production audit observer missed an opening or speculative round",
        context_for(frame, seed),
    )
    stats = talker.predictor_speculative_stats()
    require(
        stats["full_round_graph_replay_coverage"] == 1.0
        and stats["opening_pair_graph_replay_coverage"] == 1.0
        and stats["full_round_graph_captures"] > 0
        and stats["opening_pair_graph_captures"] > 0,
        "graph_replays_no_fallback",
        "actual full-round and opening-pair graph handles did not cover every replay",
        context_for(frame, seed),
    )
    probability_counts = stats["probability_backend_execution_counts"]
    require(
        probability_counts["split_top_k"] > 0
        and sum(probability_counts.values()) == probability_counts["split_top_k"],
        "probability_backend",
        "P16 did not execute its configured split Top-K probability backend",
        context_for(frame, seed),
    )
    return (
        output_codes.detach().cpu().clone(),
        summed_embeddings.detach().cpu().clone(),
        audit_frame,
        counter_delta,
    )


def audit_sequential_reference(
    talker: Qwen3TTSTalker,
    frame: ValidationFrame,
    request: SchedulerRequest,
    seed: int,
    candidate_codes: torch.Tensor,
    candidate_embedding: torch.Tensor,
    captured: CapturedFrame,
    totals: AuditTotals,
) -> dict[str, JsonValue]:
    context = context_for(frame, seed)
    opening = captured.opening
    rounds = captured.rounds
    if opening is None:
        raise RuntimeError("P16 production trace is incomplete")
    talker.prepare_decode_buffers([request])
    device = talker.predictor_device
    layer0_codes, talker_hidden, semantic_positions = input_tensors([frame], device)
    require(
        torch.equal(tensor_field(opening, "layer0_codes"), layer0_codes.cpu())
        and torch.equal(tensor_field(opening, "talker_hidden"), talker_hidden.cpu())
        and torch.equal(
            tensor_field(opening, "semantic_positions"), semantic_positions.cpu()
        ),
        "opening_input_exact",
        "production opening inputs differ from the sealed P12 frame",
        context,
    )
    layer0_embedding = talker.get_input_embeddings()(layer0_codes).to(torch.bfloat16)
    pair_embeddings = talker.code_predictor.project_input(
        torch.cat((talker_hidden, layer0_embedding), dim=1)
    )
    pair_hidden = talker.predictor_forward_tokens(
        token_embeds=pair_embeddings,
        batch_size=1,
        cache_len=0,
    )
    opening_hidden = pair_hidden[:, 1, :].contiguous()
    opening_logits, _ = talker.code_predictor.lm_head[0](opening_hidden)
    opening_logits = opening_logits.contiguous()
    assert_tensor_equal(
        tensor_field(opening, "preceding_hidden"),
        opening_hidden.cpu(),
        "opening_hidden_exact",
        context,
    )
    assert_tensor_equal(
        tensor_field(opening, "opening_logits"),
        opening_logits.cpu(),
        "native_target_logits_exact",
        context,
    )
    expected_positions = talker.sub_seed_positions(semantic_positions)[0]
    assert_tensor_equal(
        tensor_field(opening, "opening_sampling_positions"),
        expected_positions.cpu(),
        "opening_rng_exact",
        context,
    )
    assert_tensor_equal(
        tensor_field(opening, "opening_sampling_seeds"),
        talker.sub_sampling_seed_tensor[:1].cpu(),
        "opening_rng_exact",
        context,
    )
    sampled_opening_token = talker.sample_subtalker_token_seeded(
        opening_logits,
        sub_positions=expected_positions,
    )
    opening_codes = tensor_field(opening, "output_codes")
    require(
        int(sampled_opening_token[0]) == int(opening_codes[0, 1]),
        "opening_sample_exact",
        "native seeded opening sample differs from the production codebook-1 token",
        context,
    )
    require(
        int(opening_codes[0, 0]) == frame.code0
        and int(candidate_codes[0, 0, 0]) == frame.code0
        and int(candidate_codes[0, 1, 0]) == int(sampled_opening_token[0]),
        "opening_codes_exact",
        "opening pair code prefix does not match the sealed input and native sample",
        context,
    )
    opening_probabilities, opening_ids = build_compact_probabilities(opening_logits)
    require_compact_support(opening_probabilities, opening_ids, context)
    require(
        bool(
            (opening_ids[0] == int(sampled_opening_token[0]))
            .logical_and(opening_probabilities[0] > 0.0)
            .any()
        ),
        "opening_probability_support",
        "sampled opening token is absent from compact target support",
        context,
    )
    assert_tensor_equal(
        tensor_field(opening, "key_cache"),
        talker.predictor_k_cache[:, :1, :2].detach().cpu(),
        "committed_kv_exact",
        context,
    )
    assert_tensor_equal(
        tensor_field(opening, "value_cache"),
        talker.predictor_v_cache[:, :1, :2].detach().cpu(),
        "committed_kv_exact",
        context,
    )

    code_prefix = torch.zeros((1, P16_GROUPS), device=device, dtype=torch.long)
    code_prefix[:, 0] = frame.code0
    pending_token = int(sampled_opening_token[0])
    code_prefix[:, 1] = pending_token
    committed_depth = 1
    preceding_hidden = opening_hidden
    projected_tables = talker.predictor_projected_embeddings
    if projected_tables is None:
        raise RuntimeError("P16 sequential reference requires projected codec tables")
    draft_runtime = talker.code_predictor_p16_draft
    packed_head_weights = talker.code_predictor_spec_lm_head_weights
    if draft_runtime is None or packed_head_weights is None:
        raise RuntimeError(
            "P16 sequential reference requires the loaded draft and target heads"
        )
    proposal_index_total = 0
    target_query_total = 1
    rejection_total = 0
    bonus_total = 0
    round_evidence: list[JsonValue] = []
    for captured_round in rounds:
        round_depth = integer_field(captured_round, "round_depth")
        verify_width = integer_field(captured_round, "verify_width")
        proposal_count = integer_field(captured_round, "proposal_count")
        round_context = context_for(frame, seed, round_depth)
        require(
            1 <= verify_width <= P16_VERIFY_WIDTH
            and proposal_count == verify_width - 1
            and integer_field(captured_round, "logical_kv_length") == round_depth + 1,
            "round_shape",
            "production round width or logical KV prefix is inconsistent",
            round_context,
        )
        require(
            round_depth == committed_depth,
            "round_depth",
            "production depth differs from the independently committed prefix",
            round_context,
        )
        proposal_tokens = (
            tensor_field(captured_round, "proposal_tokens").to(device).reshape(-1)
        )
        proposal_logits_expected: list[torch.Tensor] = []
        proposal_probabilities_expected: list[torch.Tensor] = []
        proposal_ids_expected: list[torch.Tensor] = []
        draft_hidden = preceding_hidden
        draft_input_token = torch.tensor(
            [pending_token], device=device, dtype=torch.long
        )
        for proposal_index in range(proposal_count):
            proposal_depth = round_depth + proposal_index
            draft_output = (
                draft_runtime.output
                if proposal_index % 2 == 0
                else draft_runtime.output_alternate
            )
            draft_hidden = draft_runtime.step(
                draft_hidden,
                draft_input_token,
                proposal_depth,
                hidden_output=draft_output,
            )
            head_indices = torch.tensor(
                [proposal_depth], device=device, dtype=torch.long
            )
            active_rows = torch.ones((1,), device=device, dtype=torch.bool)
            proposal_logits = grouped_lm_head_warp_accumulate(
                draft_hidden.contiguous(),
                packed_head_weights,
                None,
                head_indices,
                active_rows,
            )
            proposal_probabilities, proposal_ids = build_compact_probabilities(
                proposal_logits
            )
            require_compact_support(proposal_probabilities, proposal_ids, round_context)
            proposal_logits_expected.append(proposal_logits[0].detach().cpu())
            proposal_probabilities_expected.append(
                proposal_probabilities[0].detach().cpu()
            )
            proposal_ids_expected.append(proposal_ids[0].detach().cpu())
            uniforms = tensor_field(captured_round, "uniforms")
            proposal_uniform = float(uniforms[0, 0, proposal_index])
            proposed_token = sample_compact_row(
                proposal_probabilities[0], proposal_ids[0], proposal_uniform
            )
            require(
                proposed_token == int(proposal_tokens[proposal_index]),
                "proposal_sample_exact",
                "actual proposal token differs from the independently sampled q row",
                round_context,
            )
            draft_input_token = torch.tensor(
                [proposed_token], device=device, dtype=torch.long
            )
            proposal_index_total += 1
        recorded_proposal_logits = tensor_field(captured_round, "proposal_logits")
        recorded_proposal_probabilities = tensor_field(
            captured_round, "proposal_probabilities"
        )
        recorded_proposal_ids = tensor_field(captured_round, "proposal_token_ids")
        if proposal_count:
            assert_tensor_equal(
                recorded_proposal_logits,
                torch.stack(proposal_logits_expected),
                "draft_logits_exact",
                round_context,
            )
            assert_tensor_equal(
                recorded_proposal_probabilities,
                torch.stack(proposal_probabilities_expected),
                "compact_probability_exact",
                round_context,
            )
            assert_tensor_equal(
                recorded_proposal_ids,
                torch.stack(proposal_ids_expected),
                "compact_probability_exact",
                round_context,
            )
        else:
            require(
                recorded_proposal_logits.shape == (0, P16_VOCAB_SIZE)
                and recorded_proposal_probabilities.shape == (0, P16_CANDIDATE_WIDTH)
                and recorded_proposal_ids.shape == (0, P16_CANDIDATE_WIDTH),
                "width_one_tail",
                "W1 tail contains nonempty proposal rows",
                round_context,
            )
            totals.width_one_rounds += 1

        query_hidden_rows: list[torch.Tensor] = []
        query_logits_rows: list[torch.Tensor] = []
        for query_index in range(verify_width):
            target_depth = round_depth + query_index
            token_id = (
                pending_token
                if query_index == 0
                else int(proposal_tokens[query_index - 1])
            )
            table_depth = target_depth - 1
            query_embedding = (
                projected_tables[table_depth, token_id].view(1, 1, -1).contiguous()
            )
            query_hidden = talker.predictor_forward_tokens(
                token_embeds=query_embedding,
                batch_size=1,
                cache_len=round_depth + 1 + query_index,
            )[:, 0:1, :].contiguous()
            native_logits, _ = talker.code_predictor.lm_head[target_depth](query_hidden)
            native_logits = native_logits.contiguous()
            p16_logits = talker.code_predictor_p16_target_logits(
                query_hidden, target_depth
            )
            assert_tensor_equal(
                p16_logits,
                native_logits,
                "native_target_logits_exact",
                round_context,
            )
            query_hidden_rows.append(query_hidden[0, 0].detach().cpu())
            query_logits_rows.append(native_logits[0, 0].detach().cpu())
        sequential_hidden = torch.stack(query_hidden_rows)
        sequential_logits = torch.stack(query_logits_rows)
        assert_tensor_equal(
            tensor_field(captured_round, "target_hidden_states")[0],
            sequential_hidden,
            "packed_hidden_exact",
            round_context,
        )
        assert_tensor_equal(
            tensor_field(captured_round, "target_logits"),
            sequential_logits,
            "native_target_logits_exact",
            round_context,
        )
        active_target_rows = (
            torch.arange(verify_width, device=device) < proposal_count + 1
        )
        target_probabilities, target_ids = build_compact_probabilities(
            sequential_logits.to(device),
            active_rows=active_target_rows,
        )
        require_compact_support(target_probabilities, target_ids, round_context)
        assert_tensor_equal(
            tensor_field(captured_round, "target_probabilities"),
            target_probabilities.cpu(),
            "compact_probability_exact",
            round_context,
        )
        assert_tensor_equal(
            tensor_field(captured_round, "target_token_ids"),
            target_ids.cpu(),
            "compact_probability_exact",
            round_context,
        )
        target_query_total += verify_width
        totals.native_target_rows += verify_width
        totals.compact_probability_rows += verify_width + proposal_count
        totals.proposal_rows += proposal_count

        uniforms = tensor_field(captured_round, "uniforms")
        accepted_count = 0
        rejected = False
        for proposal_index in range(proposal_count):
            target_token = int(proposal_tokens[proposal_index])
            target_probability = probability_at(
                target_probabilities[proposal_index],
                target_ids[proposal_index],
                target_token,
            )
            draft_probability = probability_at(
                proposal_probabilities_expected[proposal_index].to(device),
                proposal_ids_expected[proposal_index].to(device),
                target_token,
            )
            if float(draft_probability) > 0.0:
                ratio = float(
                    torch.minimum(
                        torch.ones((), device=device, dtype=torch.float32),
                        target_probability / draft_probability,
                    )
                )
            else:
                ratio = 0.0
            acceptance_uniform = float(uniforms[0, 1, proposal_index])
            require(
                0.0 <= acceptance_uniform < 1.0,
                "rng_domain",
                "acceptance uniform is outside [0, 1)",
                round_context,
            )
            if acceptance_uniform >= ratio:
                rejected = True
                break
            else:
                accepted_count += 1
        if rejected:
            correction_row = accepted_count
            residual = torch.zeros_like(target_probabilities[correction_row])
            for candidate_index, target_id in enumerate(target_ids[correction_row]):
                target_token = int(target_id)
                if target_token < 0:
                    continue
                draft_probability = probability_at(
                    proposal_probabilities_expected[correction_row].to(device),
                    proposal_ids_expected[correction_row].to(device),
                    target_token,
                )
                residual[candidate_index] = torch.clamp(
                    target_probabilities[correction_row, candidate_index]
                    - draft_probability,
                    min=0.0,
                )
            residual64 = residual.to(torch.float64)
            residual_scale = residual64.max()
            require(
                float(residual_scale) > 0.0,
                "residual_mass",
                "rejection residual has no positive mass",
                round_context,
            )
            scaled_residual = residual64 / residual_scale
            residual_mass = scaled_residual.sum()
            require(
                float(residual_mass) > 0.0 and bool(torch.isfinite(residual_mass)),
                "residual_mass",
                "rejection residual mass is invalid",
                round_context,
            )
            normalized_residual = (scaled_residual / residual_mass).to(torch.float32)
            correction_uniform = float(uniforms[0, 2, correction_row])
            correction_token = sample_compact_row(
                normalized_residual,
                target_ids[correction_row],
                correction_uniform,
            )
            committed_tokens_expected = [
                int(proposal_tokens[index]) for index in range(accepted_count)
            ] + [correction_token]
            rejection_total += 1
            totals.rejections += 1
        else:
            bonus_row = accepted_count
            bonus_uniform = float(uniforms[0, 3, bonus_row])
            bonus_token = sample_compact_row(
                target_probabilities[bonus_row],
                target_ids[bonus_row],
                bonus_uniform,
            )
            committed_tokens_expected = [
                int(proposal_tokens[index]) for index in range(accepted_count)
            ] + [bonus_token]
            bonus_total += 1
            totals.bonuses += 1
        committed_count = len(committed_tokens_expected)
        recorded_accepted = int(
            tensor_field(captured_round, "accepted_draft_counts")[0]
        )
        recorded_count = int(tensor_field(captured_round, "committed_token_counts")[0])
        recorded_rejected = bool(tensor_field(captured_round, "rejected_rows")[0])
        recorded_bonus = bool(tensor_field(captured_round, "bonus_rows")[0])
        recorded_invalid = bool(tensor_field(captured_round, "invalid_rows")[0])
        require(
            not recorded_invalid
            and recorded_accepted == accepted_count
            and recorded_count == committed_count
            and recorded_rejected == rejected
            and recorded_bonus == (not rejected),
            "acceptance_decision_exact",
            "production first-rejection decision or commit metadata differs from reference",
            round_context,
        )
        recorded_committed = tensor_field(captured_round, "committed_tokens")[
            0, :committed_count
        ]
        assert_tensor_equal(
            recorded_committed,
            torch.tensor(committed_tokens_expected, dtype=torch.long),
            "committed_prefix_exact",
            round_context,
        )
        valid_kv_length = round_depth + 1 + committed_count
        require(
            integer_field(captured_round, "valid_kv_length") == valid_kv_length,
            "valid_kv_length_exact",
            "production valid KV length differs from the committed prefix",
            round_context,
        )
        assert_tensor_equal(
            tensor_field(captured_round, "key_cache"),
            talker.predictor_k_cache[:, :1, :valid_kv_length].detach().cpu(),
            "committed_kv_exact",
            round_context,
        )
        assert_tensor_equal(
            tensor_field(captured_round, "value_cache"),
            talker.predictor_v_cache[:, :1, :valid_kv_length].detach().cpu(),
            "committed_kv_exact",
            round_context,
        )
        totals.valid_kv_rounds += 1
        totals.rounds += 1
        committed_tensor = torch.tensor(
            committed_tokens_expected, device=device, dtype=torch.long
        )
        code_prefix[0, round_depth + 1 : round_depth + 1 + committed_count] = (
            committed_tensor
        )
        committed_prefix_end = round_depth + 1 + committed_count
        assert_tensor_equal(
            tensor_field(captured_round, "output_codes")[0, :committed_prefix_end],
            code_prefix[0, :committed_prefix_end].cpu(),
            "committed_prefix_exact",
            round_context,
        )
        pending_token = committed_tokens_expected[-1]
        accepted_target_hidden = sequential_hidden[accepted_count].view(1, -1)
        assert_tensor_equal(
            tensor_field(captured_round, "pending_after"),
            torch.tensor([pending_token], dtype=torch.long),
            "carried_state_exact",
            round_context,
        )
        assert_tensor_equal(
            tensor_field(captured_round, "preceding_hidden"),
            accepted_target_hidden,
            "carried_state_exact",
            round_context,
        )
        used_uniforms = torch.cat(
            (
                uniforms[0, 0, :proposal_count],
                uniforms[0, 1, :proposal_count],
                uniforms[0, 2, :verify_width],
                uniforms[0, 3, :verify_width],
            )
        )
        round_row: dict[str, JsonValue] = {
            "depth": round_depth,
            "verify_width": verify_width,
            "proposal_count": proposal_count,
            "accepted": accepted_count,
            "committed": committed_count,
            "rejected": rejected,
            "bonus": not rejected,
            "target_logits_sha256": hash_tensor(sequential_logits),
            "target_hidden_sha256": hash_tensor(sequential_hidden),
            "target_probability_sha256": hash_tensor(target_probabilities.cpu()),
            "uniforms_sha256": hash_tensor(used_uniforms),
            "proposal_probability_sha256": hash_tensor(
                tensor_field(captured_round, "proposal_probabilities")
            ),
            "valid_kv_length": valid_kv_length,
        }
        round_evidence.append(round_row)
        preceding_hidden = accepted_target_hidden.to(device)
        committed_depth += committed_count
    require(
        int(totals.rounds) > 0 and int(code_prefix[0, 15]) in range(P16_VOCAB_SIZE),
        "final_codes_exact",
        "sequential reference did not produce the final codec group",
        context,
    )
    assert_tensor_equal(
        code_prefix.cpu().unsqueeze(-1),
        candidate_codes,
        "same_seed_B_C_trace_exact",
        context,
    )
    codec_tables = talker.code_predictor.model.codec_embedding.weight
    reference_embedding = talker.get_input_embeddings()(
        torch.tensor([[frame.code0]], device=device, dtype=torch.long)
    )[0, 0].to(torch.bfloat16)
    for code_depth in range(1, P16_GROUPS):
        codec_embedding = F.embedding(
            code_prefix[0, code_depth],
            codec_tables[code_depth - 1],
        )
        reference_embedding.add_(codec_embedding)
    assert_tensor_equal(
        candidate_embedding[0, 0],
        reference_embedding.cpu(),
        "codec_embedding_sum_exact",
        context,
    )
    return {
        "uid": frame.uid,
        "ordinal": frame.ordinal,
        "seed": seed,
        "code_sha256": hash_tensor(candidate_codes),
        "feedback_sha256": hash_tensor(candidate_embedding),
        "round_count": len(rounds),
        "target_rows": target_query_total,
        "proposal_rows": proposal_index_total,
        "rejections": rejection_total,
        "bonuses": bonus_total,
        "width_one_rounds": sum(
            integer_field(round_record, "verify_width") == 1 for round_record in rounds
        ),
        "rounds": round_evidence,
    }


def run_native_controls(
    talker: Qwen3TTSTalker,
    validation_frames: list[ValidationFrame],
    seed_count: int,
) -> tuple[dict[str, JsonValue], torch.Tensor, torch.Tensor]:
    codes: list[torch.Tensor] = []
    feedback: list[torch.Tensor] = []
    graph_stats_before = talker.predictor_graph_stats()
    repeatability_exact = True
    incremental_exact = True
    for frame_index, frame in enumerate(validation_frames):
        for repeat in range(seed_count):
            seed = seed_for(frame.uid, frame.ordinal, repeat)
            request = create_request(
                f"p16-control:{frame.uid}:{frame.ordinal}:{seed}", seed
            )
            public_codes, public_feedback = run_public_predictor(
                talker, [frame], [request]
            )
            native_codes, native_feedback = run_native_incremental(
                talker, [frame], [request]
            )
            incremental_exact = (
                torch.equal(public_codes, native_codes)
                and torch.equal(public_feedback, native_feedback)
                and incremental_exact
            )
            codes.append(public_codes[0].clone())
            feedback.append(public_feedback[0].clone())
            if frame_index == 0 and repeat == 0:
                repeated_codes, repeated_feedback = run_public_predictor(
                    talker, [frame], [request]
                )
                repeatability_exact = torch.equal(
                    public_codes, repeated_codes
                ) and torch.equal(public_feedback, repeated_feedback)
    graph_stats_after = talker.predictor_graph_stats()
    graph_replay_delta = graph_stats_after["replays"] - graph_stats_before["replays"]
    graph_eager_delta = (
        graph_stats_after["eager_runs"] - graph_stats_before["eager_runs"]
    )
    candidate_graph_exact = (
        graph_stats_after["enabled"] is True
        and graph_replay_delta == len(codes) + 1
        and graph_eager_delta == 0
    )
    off_no_draft_allocation = (
        talker.code_predictor_p16_draft is None
        and talker.code_predictor_p16_draft_sha256 is None
        and talker.code_predictor_spec_requested_width == 0
        and talker.code_predictor_spec_effective_width == 0
        and talker.sub_speculative_seed_tensor is None
        and talker.sub_speculative_uniforms_tensor is None
        and talker.sub_speculative_rng_pool_tensor is None
        and talker.code_predictor_spec_lm_head_weights is None
        and not talker.code_predictor_spec_workspaces
        and not talker.code_predictor_spec_input_buffers
        and not talker.code_predictor_spec_round_graphs
        and not talker.code_predictor_spec_opening_graphs
    )
    require(
        off_no_draft_allocation,
        "off_no_draft_allocation",
        "P16 resources were allocated while OFF",
    )
    return (
        {
            "candidate_native_graph_matches_incremental": incremental_exact,
            "candidate_repeatability_exact": repeatability_exact,
            "candidate_native_graph_replayed": candidate_graph_exact,
            "off_no_draft_allocation": off_no_draft_allocation,
        },
        torch.stack(codes),
        torch.stack(feedback),
    )


def native_reference_samples(
    frames: list[ValidationFrame], seed_count: int
) -> list[dict[str, JsonValue | torch.Tensor]]:
    samples: list[dict[str, JsonValue | torch.Tensor]] = []
    for frame in frames:
        for repeat in range(seed_count):
            seed = seed_for(frame.uid, frame.ordinal, repeat)
            samples.append(
                {
                    "request_id": f"p16-control:{frame.uid}:{frame.ordinal}:{seed}",
                    "uid": frame.uid,
                    "ordinal": frame.ordinal,
                    "seed": seed,
                    "code0": frame.code0,
                    "hidden": frame.hidden.detach().cpu().clone(),
                }
            )
    return samples


def write_torch_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        with temporary_path.open("rb") as output:
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def run_upstream_native_reference(
    model_path: Path,
    input_artifact: Path,
    output_artifact: Path,
) -> dict[str, object]:
    """Run pinned upstream native Predictor code in an isolated checkout."""
    checkout_root = Path(tempfile.mkdtemp(prefix="qtts-p16-upstream-")).resolve()
    checkout = checkout_root / "sglang-omni"
    helper_source = REPOSITORY_ROOT / "scripts/qwen3_tts_p16_native_reference.py"
    environment = os.environ.copy()
    # The parent owns its rendezvous port; the independent reference must claim its own.
    environment.pop("MASTER_PORT", None)
    for name in (
        "SGLANG_OMNI_QTTS_P16_ENABLE",
        "SGLANG_OMNI_QTTS_P16_DRAFT_PATH",
        "SGLANG_OMNI_QTTS_P16_DRAFT_SHA256",
        "SGLANG_OMNI_QTTS_P16_TELEMETRY_PATH",
        P16_NATIVE_REFERENCE_ROOT_ENV,
        "PYTHONPATH",
    ):
        environment.pop(name, None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                str(REPOSITORY_ROOT),
                str(checkout),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", P16_UPSTREAM_BASE_SHA],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        environment[P16_NATIVE_REFERENCE_ROOT_ENV] = str(checkout)
        result = subprocess.run(
            [
                sys.executable,
                str(helper_source),
                "--model",
                str(model_path),
                "--inputs",
                str(input_artifact),
                "--out",
                str(output_artifact),
                "--expected-commit",
                P16_UPSTREAM_BASE_SHA,
                "--model-revision",
                P16_MODEL_REVISION,
            ],
            cwd=checkout,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stdout + "\n" + result.stderr)[-8000:]
            raise RuntimeError(
                f"upstream native reference subprocess failed ({result.returncode}):\n{detail}"
            )
        return torch.load(output_artifact, map_location="cpu", weights_only=True)
    finally:
        shutil.rmtree(checkout_root, ignore_errors=True)


def validate_native_reference(
    reference: dict[str, object],
    *,
    input_identity: dict[str, JsonValue],
    candidate_codes: torch.Tensor,
    candidate_feedback: torch.Tensor,
) -> tuple[bool, dict[str, JsonValue]]:
    reference_identity = require_mapping(
        reference.get("identity"), "native reference identity"
    )
    reference_metadata = require_mapping(
        reference.get("metadata"), "native reference metadata"
    )
    require(
        reference_identity == input_identity,
        "default_off_reference_identity",
        "upstream reference inputs differ from candidate OFF inputs",
    )
    require(
        reference_metadata.get("upstream_commit") == P16_UPSTREAM_BASE_SHA
        and reference_metadata.get("model_revision") == P16_MODEL_REVISION
        and reference_metadata.get("cohort_sha256") == input_identity["cohort_sha256"],
        "default_off_reference_identity",
        "upstream reference metadata does not match the pinned validation identity",
    )
    require(
        reference_metadata.get("model_source_path_relative")
        == "sglang_omni/models/qwen3_tts/sglang_model.py"
        and isinstance(reference_metadata.get("model_source_path"), str)
        and Path(reference_metadata["model_source_path"]).is_absolute(),
        "default_off_reference_import",
        "upstream reference did not record the pinned Talker source path",
    )
    source_root_value = reference_metadata.get("reference_source_root")
    imported_source_paths = reference_metadata.get("imported_source_paths")
    require(
        isinstance(source_root_value, str)
        and Path(source_root_value).is_absolute()
        and Path(source_root_value) == Path(source_root_value).resolve()
        and isinstance(imported_source_paths, dict)
        and set(imported_source_paths) == set(P16_NATIVE_REFERENCE_IMPORTS),
        "default_off_reference_import",
        "upstream reference did not record its pinned import source root and modules",
    )
    source_root = Path(source_root_value)
    for module_name, relative_path in P16_NATIVE_REFERENCE_IMPORTS.items():
        imported_source_path = imported_source_paths[module_name]
        expected_source_path = source_root / relative_path
        require(
            isinstance(imported_source_path, str)
            and Path(imported_source_path).is_absolute()
            and imported_source_path == str(expected_source_path),
            "default_off_reference_import",
            f"upstream reference imported {module_name} outside its recorded source checkout",
        )
    require(
        reference_metadata["model_source_path"] == imported_source_paths["talker"],
        "default_off_reference_import",
        "upstream Talker source path differs from the imported Talker module",
    )
    graph_evidence = require_mapping(
        reference_metadata.get("native_graph"), "upstream native graph evidence"
    )
    require(
        graph_evidence.get("enabled") is True
        and graph_evidence.get("replay_count") == len(candidate_codes)
        and graph_evidence.get("eager_run_delta") == 0,
        "default_off_reference_graph",
        "upstream native reference did not replay one CUDA Graph per input",
    )
    reference_codes = reference.get("codes")
    reference_feedback = reference.get("feedback")
    require(
        isinstance(reference_codes, torch.Tensor)
        and isinstance(reference_feedback, torch.Tensor),
        "default_off_reference_outputs",
        "upstream reference did not save output code and feedback tensors",
    )
    require(
        reference_codes.shape == candidate_codes.shape
        and reference_feedback.shape == candidate_feedback.shape
        and reference_codes.dtype is candidate_codes.dtype
        and reference_feedback.dtype is candidate_feedback.dtype,
        "default_off_reference_outputs",
        "upstream reference output shapes or dtypes differ from candidate outputs",
    )
    records = reference_metadata.get("records")
    input_records = reference.get("inputs")
    require(
        isinstance(records, list)
        and len(records) == len(candidate_codes)
        and all(isinstance(record, dict) for record in records),
        "default_off_reference_outputs",
        "upstream reference output records do not cover every candidate input",
    )
    for index, record in enumerate(records):
        require(
            record.get("codes_sha256") == hash_tensor(reference_codes[index])
            and record.get("feedback_sha256") == hash_tensor(reference_feedback[index]),
            "default_off_reference_outputs",
            "upstream reference tensor hashes do not match its output artifact",
            {"index": index},
        )
    require(
        isinstance(input_records, list)
        and len(input_records) == len(records)
        and all(isinstance(sample, dict) for sample in input_records),
        "default_off_reference_inputs",
        "native reference artifact does not retain every input row",
    )
    for index, (record, sample) in enumerate(zip(records, input_records)):
        require(
            all(
                record.get(field) == sample.get(field)
                for field in ("request_id", "uid", "ordinal", "seed")
            ),
            "default_off_reference_identity",
            "native reference output row is associated with the wrong input or order",
            {"index": index},
        )
    exact = torch.equal(candidate_codes, reference_codes) and torch.equal(
        candidate_feedback, reference_feedback
    )
    return exact, {
        "upstream_commit": P16_UPSTREAM_BASE_SHA,
        "model_revision": P16_MODEL_REVISION,
        "model_source_path": reference_metadata["model_source_path"],
        "model_source_path_relative": reference_metadata["model_source_path_relative"],
        "reference_source_root": source_root_value,
        "imported_source_paths": imported_source_paths,
        "cohort_sha256": input_identity["cohort_sha256"],
        "sampling": input_identity["sampling"],
        "graph_configuration": input_identity["graph_configuration"],
        "native_graph": graph_evidence,
        "records": records,
        "codes_sha256": hash_tensor(reference_codes),
        "feedback_sha256": hash_tensor(reference_feedback),
    }


def audit_unsupported_native_fallback(
    talker: Qwen3TTSTalker,
    frames: list[ValidationFrame],
    requests: list[SchedulerRequest],
    expected_reason: str,
    check_name: str,
) -> bool:
    layer0_codes, talker_hidden, semantic_positions = input_tensors(
        frames, talker.predictor_device
    )
    talker.prepare_decode_buffers(requests)
    reason = talker.code_predictor_p16_frame_ineligibility_reason(
        layer0_codes,
        talker_hidden,
        semantic_positions,
    )
    require(
        reason == expected_reason,
        check_name,
        f"unexpected eligibility reason: {reason}",
    )
    fallback_before = talker.code_predictor_spec_native_fallback_count
    rounds_before = talker.code_predictor_spec_round_count
    replay_before = talker.code_predictor_spec_full_round_graph_replay_count
    candidate_codes, candidate_embedding = run_public_predictor(
        talker, frames, requests
    )
    require(
        talker.code_predictor_spec_native_fallback_count - fallback_before
        == len(frames)
        and talker.code_predictor_spec_round_count == rounds_before
        and talker.code_predictor_spec_full_round_graph_replay_count == replay_before,
        check_name,
        "unsupported request did not fall back before speculative state mutation",
    )
    native_codes, native_embedding = run_native_incremental(talker, frames, requests)
    require(
        torch.equal(candidate_codes, native_codes)
        and torch.equal(candidate_embedding, native_embedding),
        check_name,
        "unsupported request does not match independent native incremental replay",
    )
    return True


def audit_capacity_four_batch_fallback(
    model_path: Path,
    draft_path: Path,
    device: torch.device,
    cohort: list[ValidationFrame],
) -> bool:
    scheduler, talker = build_talker(
        model_path,
        device,
        enabled=True,
        draft_path=draft_path,
        max_running_requests=4,
    )
    try:
        require(
            talker.code_predictor_p16_enabled
            and talker.predictor_k_cache.shape[1] == 4
            and 4 in talker.predictor_graph_batch_sizes,
            "capacity_four_control",
            "B2/B4 fallback control was not constructed with capacity four",
        )
        for batch_size in (2, 4):
            batch_frames = cohort[:batch_size]
            batch_requests = [
                create_request(
                    f"p16-capacity-four-{batch_size}:{frame.uid}:{frame.ordinal}:{index}",
                    seed_for(frame.uid, frame.ordinal, index),
                )
                for index, frame in enumerate(batch_frames)
            ]
            audit_unsupported_native_fallback(
                talker,
                batch_frames,
                batch_requests,
                "unsupported_batch",
                "unsupported_batch_native_exact",
            )
        return True
    finally:
        close_runtime(scheduler)


def test_request_reset_and_reuse(
    talker: Qwen3TTSTalker,
    frame: ValidationFrame,
    first_seed: int,
    second_seed: int,
    audit_capture: P16AuditCapture,
    totals: AuditTotals,
) -> bool:
    request_id = f"p16-reuse:{frame.uid}:{frame.ordinal}"
    first_request = create_request(request_id, first_seed)
    first_codes, first_embedding, first_audit, _ = capture_frame(
        talker, frame, first_request, first_seed, audit_capture
    )
    if talker.sub_speculative_seed_tensor is None:
        raise RuntimeError("P16 request seed buffer disappeared during reuse audit")
    first_speculative_seed = int(talker.sub_speculative_seed_tensor[0])
    first_evidence = audit_sequential_reference(
        talker,
        frame,
        first_request,
        first_seed,
        first_codes,
        first_embedding,
        first_audit,
        totals,
    )
    audit_capture.release(first_audit)
    independent_request = create_request(f"{request_id}:different-id", first_seed)
    independent_codes, independent_embedding, independent_audit, _ = capture_frame(
        talker, frame, independent_request, first_seed, audit_capture
    )
    independent_speculative_seed = int(talker.sub_speculative_seed_tensor[0])
    independent_evidence = audit_sequential_reference(
        talker,
        frame,
        independent_request,
        first_seed,
        independent_codes,
        independent_embedding,
        independent_audit,
        totals,
    )
    audit_capture.release(independent_audit)
    if first_audit.opening is None or independent_audit.opening is None:
        raise RuntimeError("P16 request-ID audit did not capture its opening trace")
    opening_fields = (
        "opening_logits",
        "opening_sampling_seeds",
        "opening_sampling_positions",
        "output_codes",
        "pending_tokens",
        "preceding_hidden",
    )
    same_opening_trace = all(
        torch.equal(
            tensor_field(first_audit.opening, field_name),
            tensor_field(independent_audit.opening, field_name),
        )
        for field_name in opening_fields
    )
    second_request = create_request(request_id, second_seed)
    second_codes, second_embedding, second_audit, _ = capture_frame(
        talker, frame, second_request, second_seed, audit_capture
    )
    second_speculative_seed = int(talker.sub_speculative_seed_tensor[0])
    audit_sequential_reference(
        talker,
        frame,
        second_request,
        second_seed,
        second_codes,
        second_embedding,
        second_audit,
        totals,
    )
    audit_capture.release(second_audit)
    repeated_request = create_request(request_id, first_seed)
    repeated_codes, repeated_embedding, repeated_audit, _ = capture_frame(
        talker, frame, repeated_request, first_seed, audit_capture
    )
    repeated_speculative_seed = int(talker.sub_speculative_seed_tensor[0])
    repeated_evidence = audit_sequential_reference(
        talker,
        frame,
        repeated_request,
        first_seed,
        repeated_codes,
        repeated_embedding,
        repeated_audit,
        totals,
    )
    audit_capture.release(repeated_audit)
    require(
        torch.equal(first_codes, repeated_codes)
        and torch.equal(first_embedding, repeated_embedding)
        and torch.equal(first_codes, independent_codes)
        and torch.equal(first_embedding, independent_embedding)
        and same_opening_trace
        and len(first_audit.rounds or []) == len(repeated_audit.rounds or [])
        and first_evidence["rounds"] == repeated_evidence["rounds"]
        and first_evidence["rounds"] == independent_evidence["rounds"]
        and first_speculative_seed == repeated_speculative_seed
        and first_speculative_seed == independent_speculative_seed
        and first_speculative_seed != second_speculative_seed,
        "request_reset_and_reuse",
        "explicit seeds changed across request IDs or after request identity reuse",
        context_for(frame, first_seed),
    )
    return True


def check_invalid_request(
    talker: Qwen3TTSTalker, frame: ValidationFrame, seed: int
) -> bool:
    request = create_request(f"p16-invalid:{frame.uid}:{frame.ordinal}", seed)
    request_data = request.data
    if not isinstance(request_data, Qwen3TTSSGLangRequestData):
        raise RuntimeError("invalid-request fixture is not Qwen3-TTS request data")
    object.__setattr__(request_data, "semantic_sampling_seed", "invalid-seed")
    prior_seed = talker.sub_speculative_seed_tensor[0].detach().cpu().clone()
    try:
        talker.prepare_decode_buffers([request])
    except TypeError:
        require(
            torch.equal(
                talker.sub_speculative_seed_tensor[0].detach().cpu(), prior_seed
            ),
            "invalid_request_rejected",
            "invalid request mutated the persistent speculative seed buffer",
        )
        return True
    else:
        raise ValidationFailure(
            "invalid_request_rejected",
            "malformed sampling seed was accepted",
            context_for(frame, seed),
        )


def run_candidate_control_process(
    control: Literal["off", "capacity_four"],
    model_path: Path,
    draft_path: Path,
    inputs_path: Path,
    seed_count: int,
) -> dict[str, object]:
    # An SGLang engine owns its process for its entire lifetime.
    output_path = inputs_path.with_name(f"{control}.pt")
    environment = os.environ.copy()
    environment.pop("MASTER_PORT", None)
    print(f"P16_CONTROL_START {control}", flush=True)
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--control",
            control,
            "--model",
            str(model_path),
            "--draft",
            str(draft_path),
            "--inputs",
            str(inputs_path),
            "--seeds",
            str(seed_count),
            "--out",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    if process.returncode != 0:
        review_path = output_path.with_suffix(".review.json")
        if review_path.is_file():
            review = json.loads(review_path.read_text())
            raise ValidationFailure(
                review["check"], review["detail"], review["context"]
            )
        raise RuntimeError(
            f"P16 {control} control subprocess failed ({process.returncode}); "
            "see the preceding traceback"
        )
    print(f"P16_CONTROL_COMPLETE {control}", flush=True)
    return torch.load(output_path, map_location="cpu", weights_only=True)


def candidate_control_main() -> int:
    parser = argparse.ArgumentParser(description="Internal P16 candidate control")
    parser.add_argument("--control", choices=("off", "capacity_four"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--seeds", type=int, choices=(1, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        device = require_h200()
        artifact = torch.load(
            arguments.inputs, map_location="cpu", weights_only=True
        )
        # The coordinator serializes frames in frame-major, then seed order.
        # Both controls consume the same frozen rows as the upstream reference.
        cohort = [
            ValidationFrame(
                uid=sample["uid"],
                ordinal=sample["ordinal"],
                hidden=sample["hidden"],
                code0=sample["code0"],
            )
            for sample in artifact["inputs"][:: arguments.seeds]
        ]
        if arguments.control == "off":
            scheduler, talker = build_talker(
                arguments.model, device, enabled=False, max_running_requests=1
            )
            try:
                checks, codes, feedback = run_native_controls(
                    talker, cohort, arguments.seeds
                )
            finally:
                close_runtime(scheduler)
            result = {"checks": checks, "codes": codes, "feedback": feedback}
        else:
            result = {
                "unsupported_batch_native_exact": audit_capacity_four_batch_fallback(
                    arguments.model, arguments.draft, device, cohort
                )
            }
        write_torch_atomic(arguments.out, result)
    except ValidationFailure as error:
        write_json_atomic(
            arguments.out.with_suffix(".review.json"),
            {
                "check": error.check,
                "detail": error.detail,
                "context": error.context,
            },
        )
        raise
    return 0


def run_validation(arguments: ValidationArguments) -> dict[str, JsonValue]:
    candidate_sha = current_git_sha()
    require(
        not arguments.out.exists(),
        "receipt_path",
        "refusing to overwrite an existing P16 receipt",
    )
    require(
        arguments.out.is_absolute(), "receipt_path", "--out must be an absolute path"
    )
    native_reference_path = arguments.out.with_suffix(".native-reference.pt")
    native_reference_json_path = arguments.out.with_suffix(".native-reference.json")
    require(
        not native_reference_path.exists() and not native_reference_json_path.exists(),
        "native_reference_path",
        "refusing to overwrite an existing independent native reference artifact",
    )
    model_path = arguments.model.resolve(strict=True)
    draft_path = arguments.draft.resolve(strict=True)
    model_identity_value = model_identity(model_path)
    _, cohort, cohort_sha, cohort_identity = validation_rows(
        arguments.p12_root,
        model_identity_value,
        arguments.frames,
        arguments.seeds,
    )
    p15_decision, p15_decision_path = validate_p15_gate(arguments.p15_root)
    p15_identity = require_mapping(p15_decision.get("identity"), "P15A identity")
    require(
        p15_identity.get("cohort_sha256") == P15_COHORT_SHA256,
        "p15_gate",
        "P15A cohort identity differs from the P16 validation cohort",
    )
    require(
        sha256_file(draft_path) == P16_DRAFT_SHA256,
        "draft_checkpoint",
        "local draft file does not match the pinned P13 D-w1024-600k SHA256",
    )
    with tempfile.TemporaryDirectory(prefix="qtts-p16-bad-checkpoint-") as directory:
        bad_checkpoint_path = Path(directory) / "bad.pt"
        bad_checkpoint_path.write_bytes(b"invalid P16 checkpoint")
        try:
            load_p16_draft_checkpoint(bad_checkpoint_path, P16_DRAFT_SHA256)
        except ValueError as error:
            bad_checkpoint_rejected = "SHA256 mismatch" in str(error)
        else:
            bad_checkpoint_rejected = False
    require(
        bad_checkpoint_rejected,
        "bad_checkpoint_rejected",
        "missing draft checkpoint was accepted",
    )

    samples = native_reference_samples(cohort, arguments.seeds)
    reference_identity: dict[str, JsonValue] = {
        "upstream_commit": P16_UPSTREAM_BASE_SHA,
        "model_revision": P16_MODEL_REVISION,
        "model_identity": model_identity_value,
        "cohort_sha256": cohort_sha,
        "sampling": {
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 1.0,
            "top_k": P16_MAX_TOP_K,
        },
        "graph_configuration": {
            "max_running_requests": 1,
            "cuda_graph_max_bs": 1,
            "tp_size": 1,
        },
    }
    input_artifact = {
        "identity": reference_identity,
        "inputs": samples,
    }
    with tempfile.TemporaryDirectory(prefix="qtts-p16-native-inputs-") as directory:
        reference_inputs_path = Path(directory) / "native-inputs.pt"
        torch.save(input_artifact, reference_inputs_path)
        off_result = run_candidate_control_process(
            "off", model_path, draft_path, reference_inputs_path, arguments.seeds
        )
        off_checks = cast(dict[str, JsonValue], off_result["checks"])
        candidate_off_codes = cast(torch.Tensor, off_result["codes"])
        candidate_off_feedback = cast(torch.Tensor, off_result["feedback"])
        print("P16_CONTROL_START upstream_native_reference", flush=True)
        native_reference = run_upstream_native_reference(
            model_path,
            reference_inputs_path,
            native_reference_path,
        )
        native_reference_artifact = {
            **native_reference,
            "inputs": samples,
        }
        write_torch_atomic(native_reference_path, native_reference_artifact)
        default_off_exact, reference_metadata = validate_native_reference(
            native_reference_artifact,
            input_identity=reference_identity,
            candidate_codes=candidate_off_codes,
            candidate_feedback=candidate_off_feedback,
        )
        require(
            default_off_exact,
            "default_off_exact",
            "candidate OFF codec or feedback outputs differ from independent upstream native execution",
        )
        off_checks["default_off_exact"] = default_off_exact
        require(
            all(value is True for value in off_checks.values()),
            "default_off_candidate_graph",
            f"candidate OFF native controls failed: {off_checks}",
        )
        reference_metadata["artifact_sha256"] = sha256_file(native_reference_path)
        reference_metadata["input_records"] = [
            {
                "request_id": sample["request_id"],
                "uid": sample["uid"],
                "ordinal": sample["ordinal"],
                "seed": sample["seed"],
                "code0": sample["code0"],
                "hidden_sha256": hash_tensor(sample["hidden"]),
            }
            for sample in samples
        ]
        write_json_atomic(native_reference_json_path, reference_metadata)
        print("P16_CONTROL_COMPLETE default_off_exact", flush=True)

        batch_result = run_candidate_control_process(
            "capacity_four",
            model_path,
            draft_path,
            reference_inputs_path,
            arguments.seeds,
        )
        unsupported_batch_exact = batch_result["unsupported_batch_native_exact"] is True

    # The coordinator has not built an engine yet. Its only engine is C1 ON.
    os.environ.pop("MASTER_PORT", None)
    device = require_h200()
    print("P16_CONTROL_START candidate_on", flush=True)
    candidate_scheduler, talker = build_talker(
        model_path,
        device,
        enabled=True,
        draft_path=draft_path,
    )
    try:
        require(
            talker.code_predictor_p16_enabled
            and talker.predictor_max_batch_size == 1
            and talker.predictor_graph_batch_sizes == (1,)
            and talker.predictor_fused_layers.max_rows >= P16_VERIFY_WIDTH
            and int(talker.config.hidden_size) == P16_TALKER_HIDDEN_SIZE
            and int(talker.config.code_predictor_config.hidden_size)
            == P16_CODE_PREDICTOR_HIDDEN_SIZE
            and talker.code_predictor_p16_draft is not None
            and talker.code_predictor_p16_draft_sha256 == P16_DRAFT_SHA256
            and talker.code_predictor_spec_effective_width == P16_VERIFY_WIDTH
            and talker.code_predictor_spec_requested_width == P16_VERIFY_WIDTH,
            "checkpoint_weights_verified",
            "production builder did not load the pinned draft and graph configuration",
        )
        # Retained receipt field for compatibility; target verification now calls
        # the native LM-head module directly and does not select F.linear.
        native_target_flinear = False
        require(
            talker.code_predictor_p16_static_fallback_reason is None,
            "p16_eligibility",
            f"H200 worker is statically ineligible: {talker.code_predictor_p16_static_fallback_reason}",
        )
        try:
            normalize_qwen3_tts_seed(True)
        except ValueError:
            invalid_public_seed_rejected = True
        else:
            invalid_public_seed_rejected = False
        require(
            invalid_public_seed_rejected,
            "invalid_request_rejected",
            "public Qwen3-TTS request seed validator accepted a boolean seed",
        )

        totals = AuditTotals()
        audit_capture = P16AuditCapture()
        candidate_visits: list[dict[str, JsonValue]] = []
        same_seed_trace_exact = True
        for cohort_index, frame in enumerate(cohort):
            for repeat in range(arguments.seeds):
                seed = seed_for(frame.uid, frame.ordinal, repeat)
                request_id = f"p16:{frame.uid}:{frame.ordinal}:{seed}"
                request = create_request(request_id, seed)
                candidate_codes, candidate_embedding, captured, counters = (
                    capture_frame(
                        talker,
                        frame,
                        request,
                        seed,
                        audit_capture,
                    )
                )
                evidence = audit_sequential_reference(
                    talker,
                    frame,
                    request,
                    seed,
                    candidate_codes,
                    candidate_embedding,
                    captured,
                    totals,
                )
                audit_capture.release(captured)
                evidence["cohort_index"] = cohort_index
                counter_json: dict[str, JsonValue] = {}
                for name, value in counters.items():
                    counter_json[name] = value
                evidence["graph_counter_delta"] = counter_json
                candidate_visits.append(evidence)
        require(
            len(candidate_visits) == arguments.frames * arguments.seeds,
            "audit_coverage",
            "not every requested P14 frame and seed was audited",
        )

        first_frame = cohort[0]
        first_seed = seed_for(first_frame.uid, first_frame.ordinal, 0)
        reused_seed = seed_for(first_frame.uid, first_frame.ordinal, 1)
        reuse_ok = test_request_reset_and_reuse(
            talker,
            first_frame,
            first_seed,
            reused_seed,
            audit_capture,
            totals,
        )
        invalid_request_rejected = check_invalid_request(
            talker, first_frame, first_seed
        )
        unsupported_fallback_frames = 0

        unsupported_sampling_exact = True
        sampling_cases = (
            ("greedy", {"do_sample": False}),
            ("temperature", {"temperature": 0.8}),
            ("top-k", {"top_k": 49}),
            ("top-p", {"top_p": 0.95}),
        )
        for name, sampling_options in sampling_cases:
            sampling_request = create_request(
                f"p16-unsupported-{name}:{first_frame.uid}:{first_frame.ordinal}",
                first_seed,
                **sampling_options,
            )
            unsupported_sampling_exact = (
                audit_unsupported_native_fallback(
                    talker,
                    [first_frame],
                    [sampling_request],
                    "unsupported_sampling",
                    "unsupported_sampling_native_exact",
                )
                and unsupported_sampling_exact
            )
            unsupported_fallback_frames += 1

        actual_stats = talker.predictor_speculative_stats()
        graph_memory_within_budget = (
            not actual_stats["graph_capture_memory_budget_exceeded"]
            and actual_stats["graph_capture_memory_reserved_growth_bytes"]
            <= actual_stats["graph_capture_memory_budget_bytes"]
        )
        graph_capture_count_bounded = (
            actual_stats["full_round_graph_captures"] <= 14
            and actual_stats["opening_pair_graph_captures"] <= 1
        )
        no_cuda_errors = (
            talker.code_predictor_spec_cuda_errors == 0
            and talker.code_predictor_spec_invalid_probability_rows == 0
            and talker.code_predictor_spec_native_fallback_count
            == unsupported_fallback_frames
            and talker.code_predictor_spec_full_round_graph_replay_count
            == talker.code_predictor_spec_round_count
        )
        require(
            no_cuda_errors,
            "no_cuda_errors",
            "P16 recorded CUDA, invalid-row, eager, or graph coverage errors",
        )
        require(
            totals.rejections > 0
            and totals.bonuses > 0
            and totals.width_one_rounds > 0,
            "reject_and_bonus_audited",
            "cohort did not exercise rejection, bonus, and W1 tail paths",
        )
        require(
            totals.valid_kv_rounds == totals.rounds
            and totals.compact_probability_rows >= totals.native_target_rows
            and totals.native_target_rows > 0,
            "committed_kv_exact",
            "sequential target/probability/KV audit coverage is incomplete",
        )
        checks = {
            **off_checks,
            "bad_checkpoint_rejected": bad_checkpoint_rejected,
            "unsupported_batch_native_exact": unsupported_batch_exact,
            "unsupported_sampling_native_exact": unsupported_sampling_exact,
            "packed_hidden_exact": True,
            "native_target_logits_exact": True,
            "compact_probability_exact": True,
            "committed_kv_exact": True,
            "same_seed_B_C_trace_exact": same_seed_trace_exact,
            "reject_and_bonus_audited": True,
            "real_wrapper_used": type(talker).__name__ == "Qwen3TTSTalker",
            "graph_replays_no_fallback": (
                actual_stats["full_round_graph_replay_coverage"] == 1.0
                and actual_stats["opening_pair_graph_replay_coverage"] == 1.0
                and talker.code_predictor_spec_native_fallback_count
                == unsupported_fallback_frames
            ),
            "graph_capture_count_bounded": graph_capture_count_bounded,
            "graph_memory_within_budget": graph_memory_within_budget,
            "request_reset_and_reuse": reuse_ok,
            "invalid_request_rejected": invalid_request_rejected,
            "checkpoint_weights_verified": talker.code_predictor_p16_draft_sha256
            == P16_DRAFT_SHA256,
            "no_cuda_errors": no_cuda_errors,
        }
        require(
            all(checks.values()),
            "p16_checks",
            f"one or more required P16 checks failed: {[name for name, passed in checks.items() if not passed]}",
        )
        receipt: dict[str, JsonValue] = {
            "schema": P16_SCHEMA,
            "status": "P16_INTEGRATION_PASS",
            "candidate_sha": candidate_sha,
            "draft_sha256": P16_DRAFT_SHA256,
            "model_revision": P16_MODEL_REVISION,
            "production_entrypoint": "Qwen3TTSTalker.code_predictor_forward",
            "cuda_graph_on": True,
            "dtype": "bfloat16",
            "cohort_sha256": cohort_sha,
            "audited_visits": len(candidate_visits),
            "checks": checks,
            "g2_5_status": "NOT_CERTIFIED",
            "evidence": {
                "p14_cohort_identity": cohort_identity,
                "p15_decision_sha256": sha256_file(p15_decision_path),
                "model_identity": model_identity_value,
                "draft_sha256_verified": sha256_file(draft_path),
                "target_native_flinear_probe_exact": native_target_flinear,
                "audited_target_rows": totals.native_target_rows,
                "audited_compact_probability_rows": totals.compact_probability_rows,
                "audited_proposal_rows": totals.proposal_rows,
                "audited_rounds": totals.rounds,
                "committed_kv_exact_rounds": totals.valid_kv_rounds,
                "predictor_graph_diagnostics": actual_stats,
                "graph_memory_budget_semantics": (
                    "The 512 MiB limit is a post-capture reserved-memory growth guard; "
                    "it does not preallocate or reserve that memory before capture."
                ),
                "rejections": totals.rejections,
                "bonuses": totals.bonuses,
                "w1_tail_rounds": totals.width_one_rounds,
                "full_round_graph_replays": talker.code_predictor_spec_full_round_graph_replay_count,
                "opening_pair_graph_replays": talker.code_predictor_spec_opening_graph_replay_count,
                "full_round_graph_captures": talker.code_predictor_spec_full_round_graph_capture_count,
                "opening_pair_graph_captures": talker.code_predictor_spec_opening_graph_capture_count,
                "probability_backend_execution_counts": actual_stats[
                    "probability_backend_execution_counts"
                ],
                "visits": candidate_visits,
                "failure_rows": [],
            },
        }
        return receipt
    finally:
        close_runtime(candidate_scheduler)


def main() -> int:
    if sys.argv[1:2] == ["--control"]:
        return candidate_control_main()
    arguments = parse_arguments()
    review_path = arguments.out.with_name(f"{arguments.out.name}.review.json")
    try:
        receipt = run_validation(arguments)
        write_json_atomic(arguments.out, receipt)
    except Exception as error:
        failure: dict[str, JsonValue] = {
            "schema": P16_SCHEMA,
            "status": "P16_INTEGRATION_REVIEW",
            "error_type": type(error).__name__,
            "error": str(error),
            "failure_rows": [],
        }
        if isinstance(error, ValidationFailure):
            failure["failed_check"] = error.check
            failure["failure_rows"] = [error.context]
        if not arguments.out.exists() and not review_path.exists():
            write_json_atomic(review_path, failure)
        traceback.print_exc()
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    else:
        print(json.dumps(receipt, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Load pinned P14/P15A reference components for the P16 parity audit."""

from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import TypedDict

import torch
from torch import nn

P15A_SOURCE_COMMIT = "ab080c51e5d8977ff671219eacb2a82a3ff33966"
P15A_KERNEL_SOURCE_COMMIT = "577867c4b94c4da672b34a61917df0af813425dc"
P15A_RADIX_SOURCE_BLOB = "ed6ea57b802c44e2e7272356e0d7f4e9e8b9cff3"
P15A_RADIX_SOURCE_SHA256 = (
    "bc29666fe97900b1c0994ea1174feba75e7c86b4201ff2065f740003d930d667"
)
P15A_GROUPED_HEAD_SOURCE_BLOB = "c124a3e037a06592c299c0d8d1cc853d42cb2088"
P15A_GROUPED_HEAD_SOURCE_SHA256 = (
    "8866d91a0340d442978e577abe5d0da7dc883a68c3095af6dd65723c0e44313f"
)


class SourceDigest(TypedDict):
    commit: str
    sha256: str
    git_blob: str


class ReferenceDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2063, 1024, bias=True)
        self.out_proj = nn.Linear(1024, 1024, bias=True)
        self.register_buffer("hidden_scale", torch.ones(()))
        self.register_buffer("embed_scale", torch.ones(()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
        else:
            pass
    return digest.hexdigest()


def load_p15a_references(
    root: Path,
) -> tuple[ModuleType, ModuleType, dict[str, str]]:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != P15A_SOURCE_COMMIT:
        raise RuntimeError(
            f"P15A source checkout is {actual_commit}, expected {P15A_SOURCE_COMMIT}"
        )
    else:
        pass
    p14_directory = root / "experiments/qwen3_tts_spec_p14"
    p15a_directory = root / "experiments/qwen3_tts_spec_p15a"
    paths = {
        "fused_width.py": p14_directory / "fused_width.py",
        "runtime.py": p14_directory / "runtime.py",
        "head_runtime.py": p15a_directory / "head_runtime.py",
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("pinned P14/P15A source files are incomplete")
    else:
        pass
    sys.path.insert(0, str(p14_directory))
    importlib.invalidate_caches()
    fused_width = importlib.import_module("fused_width")
    head_spec = importlib.util.spec_from_file_location(
        "p15a_head_runtime", paths["head_runtime.py"]
    )
    if head_spec is None or head_spec.loader is None:
        raise RuntimeError("cannot load pinned P15A HeadRuntime")
    else:
        pass
    head_module = importlib.util.module_from_spec(head_spec)
    head_spec.loader.exec_module(head_module)
    if int(head_module.HeadRuntime.K) != 3:
        raise RuntimeError("pinned P15A HeadRuntime does not use chained K3")
    else:
        pass
    return (
        fused_width,
        head_module,
        {
            "commit": actual_commit,
            **{name: sha256_file(path) for name, path in paths.items()},
        },
    )


def load_pinned_kernel_module(
    root: Path,
    commit: str,
    source_path: str,
    module_name: str,
    expected_blob: str,
    expected_sha256: str,
) -> tuple[ModuleType, SourceDigest]:
    source = subprocess.check_output(
        ["git", "show", f"{commit}:{source_path}"], cwd=root
    )
    git_blob = hashlib.sha1(
        b"blob " + str(len(source)).encode() + b"\0" + source
    ).hexdigest()
    source_sha256 = hashlib.sha256(source).hexdigest()
    if git_blob != expected_blob or source_sha256 != expected_sha256:
        raise RuntimeError(f"pinned P15A source changed: {source_path}")
    else:
        pass
    source_directory = Path(tempfile.mkdtemp(prefix="qtts-p15a-kernel-source-"))
    atexit.register(shutil.rmtree, source_directory, ignore_errors=True)
    module_path = source_directory / Path(source_path).name
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned P15A source: {source_path}")
    else:
        pass
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, {
        "commit": commit,
        "sha256": source_sha256,
        "git_blob": git_blob,
    }


def load_p15a_radix_module(root: Path) -> tuple[ModuleType, SourceDigest]:
    return load_pinned_kernel_module(
        root,
        P15A_KERNEL_SOURCE_COMMIT,
        "sglang_omni/models/qwen3_tts/sampling_kernels.py",
        "p15a_radix_sampling_kernels",
        P15A_RADIX_SOURCE_BLOB,
        P15A_RADIX_SOURCE_SHA256,
    )


def load_p15a_grouped_head_module(root: Path) -> tuple[ModuleType, SourceDigest]:
    return load_pinned_kernel_module(
        root,
        P15A_KERNEL_SOURCE_COMMIT,
        "sglang_omni/utils/predictor_grouped_heads.py",
        "p15a_grouped_heads",
        P15A_GROUPED_HEAD_SOURCE_BLOB,
        P15A_GROUPED_HEAD_SOURCE_SHA256,
    )


def dense_probability(
    probabilities: torch.Tensor, token_ids: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    dense = torch.zeros(
        (probabilities.shape[0], vocab_size),
        device=probabilities.device,
        dtype=torch.float64,
    )
    dense.scatter_add_(1, token_ids.to(torch.long), probabilities.to(torch.float64))
    return dense


def distribution_delta(
    first: torch.Tensor,
    first_ids: torch.Tensor,
    second: torch.Tensor,
    second_ids: torch.Tensor,
    vocab_size: int,
) -> dict[str, bool | float | int]:
    first_dense = dense_probability(first, first_ids, vocab_size)
    second_dense = dense_probability(second, second_ids, vocab_size)
    difference = (first_dense - second_dense).abs()
    support_changes = int(
        torch.count_nonzero((first_dense > 0.0) != (second_dense > 0.0))
    )
    tv = 0.5 * difference.sum(dim=1)
    return {
        "exact": torch.equal(first_dense, second_dense),
        "support_changed_tokens": support_changes,
        "tv": float(tv[0]),
        "max_probability_abs": float(difference.max()),
    }


def summarize_deltas(
    values: list[dict[str, bool | float | int]],
) -> dict[str, bool | float | int]:
    return {
        "rows": len(values),
        "exact_rows": sum(bool(value["exact"]) for value in values),
        "support_changed_rows": sum(
            int(value["support_changed_tokens"] > 0) for value in values
        ),
        "support_changed_tokens": sum(
            int(value["support_changed_tokens"]) for value in values
        ),
        "max_tv": max((float(value["tv"]) for value in values), default=0.0),
        "mean_tv": (
            sum(float(value["tv"]) for value in values) / len(values) if values else 0.0
        ),
        "max_probability_abs": max(
            (float(value["max_probability_abs"]) for value in values),
            default=0.0,
        ),
    }

# SPDX-License-Identifier: Apache-2.0
"""P17 rootless H200 worktree environment readiness and provenance audit.

Never changes the machine or installs packages. Tests the *selected* venv,
CUDA driver/runtime, expected sglang-omni package versions and HF snapshot
pin; writes a local manifest without recording tokens or secrets.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_DISTS = {
    "torch": "2.14.1",
    "sglang": "0.5.23",
    "transformers": "5.19.0",
    "qwen-tts": "0.1.1",
}
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def _run(*args: str) -> dict[str, object]:
    try:
        p = subprocess.run(
            list(args), check=False, capture_output=True, text=True,
            timeout=45,
        )
        return {
            "exit_code": p.returncode,
            "stdout": p.stdout.strip()[:8000],
            "stderr": p.stderr.strip()[:4000],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def audit(root: Path, *, need_model: bool = True) -> dict:
    root = root.expanduser().resolve()
    source = root / "worktrees" / "p17-multiwidth"
    venv = root / "venvs" / "p17"
    report = {
        "schema": "qwen3tts-p17-rootless-worktree-env-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "P17_ENV_NOT_VERIFIED",
        "root": str(root),
        "worktree": str(source),
        "venv": str(venv),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "glibc": platform.libc_ver(),
        "checks": {},
    }
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
    if sys.prefix != str(venv) or not Path(sys.executable).resolve().is_relative_to(
        venv.resolve()
    ):
        # CPython venv/bin/python may resolve to shared python executable;
        # sys.prefix is the trustworthy venv boundary.
        if Path(sys.prefix).resolve() != venv.resolve():
            raise RuntimeError("P17_WRONG_VENV: activate root/env.sh first")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("P17_REQUIRES_PYTHON_3_12")
    if not (source / ".git").is_file():
        raise RuntimeError("P17_EXPECTED_LINKED_GIT_WORKTREE")
    git_status = _run("git", "-C", str(source), "status", "--porcelain")
    sha = _run("git", "-C", str(source), "rev-parse", "HEAD")
    branch = _run("git", "-C", str(source), "branch", "--show-current")
    checks["git"] = {
        "status": git_status, "sha": sha, "branch": branch,
    }
    if (
        sha.get("exit_code") != 0
        or len(str(sha.get("stdout", ""))) != 40
        or branch.get("stdout")
        != "experiment/qwen3-tts-cpspec-p17-multiwidth"
        or git_status.get("stdout") != ""
    ):
        raise RuntimeError("P17_WORKTREE_BRANCH_SHA_OR_DIRTY_STATUS_MISMATCH")

    installed: dict[str, str] = {}
    for name, expected in EXPECTED_DISTS.items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"P17_MISSING_PACKAGE_{name}") from exc
        installed[name] = version
        if version != expected:
            raise RuntimeError(
                f"P17_PIN_MISMATCH_{name} expected={expected}, actual={version}"
            )
    checks["packages"] = installed
    for name in ("qwen_tts", "sglang_omni", "triton", "flashinfer"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"P17_IMPORT_FAIL_{name}: {exc}") from exc

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("P17_TORCH_CUDA_UNAVAILABLE")
    count = torch.cuda.device_count()
    devices = []
    for i in range(count):
        prop = torch.cuda.get_device_properties(i)
        devices.append({
            "logical_device": i,
            "name": prop.name,
            "capability": f"{prop.major}.{prop.minor}",
            "memory_gib": round(prop.total_memory / 2**30, 3),
        })
    checks["cuda"] = {
        "torch_build_cuda": torch.version.cuda,
        "device_count": count,
        "devices": devices,
        "nvidia_smi": _run(
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
    }
    if not torch.version.cuda or not torch.version.cuda.startswith("13."):
        raise RuntimeError(
            f"P17_REQUIRES_CUDA_13_PYTORCH_WHEEL got={torch.version.cuda}"
        )
    if not devices or "H200" not in devices[0]["name"]:
        raise RuntimeError("P17_REQUIRES_VISIBLE_LOGICAL_DEVICE0_H200")
    if devices[0]["capability"] != "9.0":
        raise RuntimeError("P17_REQUIRES_SM90")
    # Actually launch a kernel: nvidia-smi alone does not prove PyTorch
    # and the installed driver/toolkit work together.
    x = torch.ones((32,), device="cuda")
    y = (x * 3).sum()
    torch.cuda.synchronize()
    if y.item() != 96.0:
        raise RuntimeError("P17_CUDA_KERNEL_SMOKE_FAILED")
    for binary in ("sox", "ffmpeg"):
        checks[binary] = shutil.which(binary)
        if not checks[binary]:
            raise RuntimeError(
                f"P17_MISSING_{binary.upper()}_EXECUTABLE; "
                "install in a user-controlled prefix, no sudo needed"
            )
    checks["gpu_usable"] = True

    model_manifest = root / "model" / "snapshot.json"
    if need_model:
        if not model_manifest.is_file():
            raise RuntimeError("P17_HF_MODEL_MANIFEST_MISSING")
        model_info = json.loads(model_manifest.read_text())
        snapshot_path = Path(model_info.get("snapshot_path", ""))
        revision = model_info.get("revision")
        if (
            model_info.get("repo_id") != MODEL
            or not isinstance(revision, str)
            or len(revision) != 40
            or snapshot_path.name != revision
            or not (snapshot_path / "config.json").is_file()
        ):
            raise RuntimeError("P17_HF_SNAPSHOT_INCOMPLETE_OR_UNPINNED")
        model_cache = (
            root / "cache" / "huggingface" / "hub"
            / "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        main_ref = model_cache / "refs" / "main"
        if not main_ref.is_file() or main_ref.read_text().strip() != revision:
            raise RuntimeError("P17_HF_CACHED_MAIN_REF_REVISION_DRIFT")
        checks["model"] = model_info
    report["status"] = (
        "P17_H200_ROOTLESS_WORKTREE_MODEL_ENV_READY"
        if need_model else "P17_H200_ROOTLESS_WORKTREE_VENV_READY_MODEL_PENDING"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "qwen3tts-p17-rootless-worktree-env-v1",
        "status": "P17_ENV_NOT_VERIFIED",
    }
    try:
        report = audit(args.root, need_model=not args.skip_model)
    except Exception as exc:
        report["status"] = "P17_ENV_PRECHECK_FAILED"
        report["first_error"] = f"{type(exc).__name__}: {exc}"
        out.write_text(json.dumps(report, indent=2) + "\n")
        raise
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"P17_ENV_STATUS={report['status']}")
    print(f"P17_ENV_REPORT={out}")


if __name__ == "__main__":
    main()

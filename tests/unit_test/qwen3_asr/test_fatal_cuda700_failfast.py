# SPDX-License-Identifier: Apache-2.0
"""CPU-only fatal CUDA700 stage fail-fast tests; no GPU required.

Never actually kill pytest itself: only a fresh subprocess executes
os._exit(70). In-process service hooks patch the process termination
function and assert it runs BEFORE traceback detachment or retry.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from sglang_omni.models.nemotron_transcribe.encoder_service import (
    NemotronTranscribePreLMEncoderService,
)
from sglang_omni.models.qwen3_asr import encoder_service as qwen_encoder
from sglang_omni.models.qwen3_asr.encoder_service import Qwen3ASRPreLMEncoderService
from sglang_omni.scheduling.fatal_gpu_context import (
    FATAL_CUDA_EXIT_CODE, is_fatal_cuda_context_error,
)


class TerminatedCudaStage(BaseException):
    """Test-only sentinel; must escape ordinary Exception recovery."""


@pytest.mark.parametrize("exc", [
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS"),
    RuntimeError("cudaErrorIllegalAddress"),
    RuntimeError("CUDA error: 700"),
])
def test_fatal_cuda700_strings_classified_without_cuda_calls(exc):
    assert is_fatal_cuda_context_error(exc)


@pytest.mark.parametrize("exc", [
    torch.OutOfMemoryError("CUDA out of memory"),
    RuntimeError("CUDA out of memory"),
    RuntimeError("STFT input shape mismatch"),
    ValueError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUDA driver not initialized"),
])
def test_nonfatal_or_invalid_input_not_kill_stage(exc):
    assert not is_fatal_cuda_context_error(exc)


def test_fatal_error_exits_before_detach_and_before_per_item_retry(monkeypatch):
    svc = object.__new__(Qwen3ASRPreLMEncoderService)
    seen = []

    def exit_process(exc, *, component):
        seen.append(("exit", type(exc).__name__, component))
        raise TerminatedCudaStage()

    def detach_should_not_run(_exc):
        raise AssertionError("Must not detach traceback before fatal exit")

    monkeypatch.setattr(qwen_encoder, "terminate_poisoned_cuda_stage", exit_process)
    monkeypatch.setattr(svc, "detach_failure", detach_should_not_run)

    exc = RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(TerminatedCudaStage):
        svc.handle_batch_failure([object(), object()], exc)
    assert seen == [("exit", "RuntimeError", "Qwen3-ASR")]


def test_fatal_error_from_nemotron_inherited_handler_also_exits(monkeypatch):
    svc = object.__new__(NemotronTranscribePreLMEncoderService)
    seen = []

    def exit_process(exc, *, component):
        seen.append(component)
        raise TerminatedCudaStage()

    monkeypatch.setattr(qwen_encoder, "terminate_poisoned_cuda_stage", exit_process)
    with pytest.raises(TerminatedCudaStage):
        svc.handle_batch_failure(
            [object(), object()],
            RuntimeError("CUDA error: an illegal memory access was encountered"),
        )
    assert seen == ["Nemotron Transcribe"]


def test_fatal_per_item_retry_also_fails_process_before_detach(monkeypatch):
    svc = object.__new__(Qwen3ASRPreLMEncoderService)
    seen = []

    def exit_process(exc, *, component):
        seen.append(component)
        raise TerminatedCudaStage()

    monkeypatch.setattr(qwen_encoder, "terminate_poisoned_cuda_stage", exit_process)
    with pytest.raises(TerminatedCudaStage):
        svc.handle_item_failure(
            object(),
            RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS"),
        )
    assert seen == ["Qwen3-ASR"]


def test_fatal_retry_defense_and_nonfatal_batch_retry_unchanged():
    svc = object.__new__(Qwen3ASRPreLMEncoderService)
    assert not svc.retry_batch(
        [object(), object()],
        RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS"),
    )
    assert svc.retry_batch(
        [object(), object()], RuntimeError("bad audio input")
    )
    assert not svc.retry_batch(
        [object()], RuntimeError("bad audio input")
    )


def test_ordinary_batch_failure_still_retries_without_killing(monkeypatch):
    svc = object.__new__(Qwen3ASRPreLMEncoderService)
    monkeypatch.setattr(
        qwen_encoder, "terminate_poisoned_cuda_stage",
        lambda *_a, **_kw: pytest.fail("Should not terminate on ordinary error"),
    )
    result = svc.handle_batch_failure(
        [object(), object()], RuntimeError("synthetic invalid audio")
    )
    assert isinstance(result, RuntimeError)
    assert "synthetic invalid audio" in str(result)
    assert svc.retry_batch([object(), object()], result)


def test_actual_fatal_stage_exit_code_in_isolated_subprocess():
    script = (
        "from sglang_omni.scheduling.fatal_gpu_context import "
        "terminate_poisoned_cuda_stage\n"
        "terminate_poisoned_cuda_stage("
        " RuntimeError('CUDA_ERROR_ILLEGAL_ADDRESS'), "
        " component='Nemotron Transcribe')\n"
        "print('SHOULD_NOT_EXECUTE')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert result.returncode == FATAL_CUDA_EXIT_CODE
    assert "FATAL_CUDA_CONTEXT_POISONED_FAILFAST" in result.stderr
    assert "component=Nemotron Transcribe" in result.stderr
    assert "SHOULD_NOT_EXECUTE" not in result.stdout

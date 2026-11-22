# SPDX-License-Identifier: Apache-2.0
"""Fatal CUDA-context error classification and process-level fail-fast.

CUDA illegal memory access (error 700) poisons the process CUDA context.
Continuing even a "best effort" encoder per-item retry may trigger further
invalid GPU accesses or allocator crashes. The service must not issue more
CUDA operations: log the failure and terminate its STAGE process so the
parent coordinator can detect stage death and fail pending requests.

Do not use for ordinary invalid inputs, benign RuntimeError or OOM.
This deliberately does not attempt to recover/reinitialize CUDA in-process.
"""
from __future__ import annotations

import os
import sys

import torch


FATAL_CUDA_EXIT_CODE = 70
_CUDA_ERROR_TOKENS = (
    "cuda_error_illegal_address",
    "cudaerrorillegaladdress",
    "cuda error: an illegal memory access",
    "cuda error: illegal memory access",
    "cuda error: 700",
    "cuda error 700",
    "cuda_error_illegal_address",
)


def is_fatal_cuda_context_error(exc: BaseException) -> bool:
    """Recognize known CUDA illegal-address errors without GPU API calls.

    The original torch.AcceleratorError may be converted to RuntimeError
    after a detached traceback; inspect text as well as exception class.
    No matching on arbitrary ValueError strings, and OOM is separate.
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return False
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    if "cuda" not in message and "illegal memory access" not in message:
        return False
    return any(token in message for token in _CUDA_ERROR_TOKENS)


def terminate_poisoned_cuda_stage(
    exc: BaseException, *, component: str
) -> None:
    """Stop the CURRENT process, not just the worker thread.

    Called before exception conversion, per-item retry and CUDA cleanup.
    Using os._exit is intentional: unwinding/destructing CUDA tensors after
    illegal-address can itself invoke the poisoned CUDA caching allocator.
    Coordinator will detect the stage process exit. No in-process recovery.
    """
    if not is_fatal_cuda_context_error(exc):
        raise ValueError("terminate_poisoned_cuda_stage called for nonfatal error")
    message = (
        f"FATAL_CUDA_CONTEXT_POISONED_FAILFAST component={component} "
        f"exit={FATAL_CUDA_EXIT_CODE} "
        f"exception_type={type(exc).__name__} "
        "cause=CUDA_ILLEGAL_ADDRESS_700; "
        "per-item retry and CUDA cleanup FORBIDDEN; "
        "stage process must be restarted\n"
    )
    # Low-level stderr output avoids logger locks after a fatal device error.
    os.write(2, message.encode("utf-8", errors="replace"))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(FATAL_CUDA_EXIT_CODE)

# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only retention audit for native Qwen3-Omni audio embeddings.

The serving integration can wire real request lifecycle callbacks into
``audit_request_lifecycle``. This module only observes ownership; it never
clears ``precomputed_embeddings`` and does not add a shared cleanup hook.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class LifetimeSnapshot:
    label: str
    supported: bool = True
    audio_embedding_tensor_bytes: int = 0
    cuda_memory_allocated: int | None = None
    cuda_memory_reserved: int | None = None
    req_retains_precomputed_embeddings: bool = False


@dataclass
class LifetimeAudit:
    request_id: str
    snapshots: list[LifetimeSnapshot] = field(default_factory=list)
    cleanup_implemented: bool = False
    cleanup_decision: str = (
        "No production cleanup decision: collect live serving evidence first."
    )

    def record(self, label: str, req: Any, *, supported: bool = True) -> None:
        tensors: list[torch.Tensor] = []
        seen: set[int] = set()
        mm_inputs = getattr(req, "multimodal_inputs", None)
        mm_inputs = mm_inputs if isinstance(mm_inputs, (list, tuple)) else [mm_inputs]
        for mm_input in mm_inputs:
            if mm_input is None:
                continue
            for item in getattr(mm_input, "mm_items", None) or ():
                tensor = getattr(item, "precomputed_embeddings", None)
                if isinstance(tensor, torch.Tensor) and id(tensor) not in seen:
                    seen.add(id(tensor))
                    tensors.append(tensor)

        allocated = reserved = None
        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
        self.snapshots.append(
            LifetimeSnapshot(
                label=label,
                supported=supported,
                audio_embedding_tensor_bytes=sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors
                ),
                cuda_memory_allocated=allocated,
                cuda_memory_reserved=reserved,
                req_retains_precomputed_embeddings=bool(tensors),
            )
        )

    def report(self) -> dict[str, Any]:
        return asdict(self)


def audit_request_lifecycle(
    req: Any,
    *,
    request_id: str,
    transitions: Mapping[str, Callable[[Any], None]] | None = None,
    session_transition: Callable[[Any], None] | None = None,
) -> LifetimeAudit:
    """Record native audio ownership at each supported lifecycle boundary.

    ``transitions`` may provide callbacks for ``after_final_prefill_chunk``,
    ``mid_decode``, ``after_completion``, ``after_retraction``, and
    ``after_abort``. A missing callback is still recorded as an observation of
    the current request object, which makes chunked/retraction coverage
    explicit in the resulting report.
    """
    audit = LifetimeAudit(request_id=request_id)
    transitions = transitions or {}
    audit.record("after_request_construction", req)
    for label in (
        "after_final_prefill_chunk",
        "mid_decode",
        "after_completion",
        "after_retraction",
        "after_abort",
    ):
        transition = transitions.get(label)
        if transition is not None:
            transition(req)
        audit.record(label, req)

    if session_transition is None:
        audit.record("session_request", req, supported=False)
    else:
        session_transition(req)
        audit.record("session_request", req)
    return audit


def write_audit(report: LifetimeAudit, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.report(), indent=2) + "\n")

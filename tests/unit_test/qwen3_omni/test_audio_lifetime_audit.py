# SPDX-License-Identifier: Apache-2.0
"""Retention accounting tests for the benchmark-only lifetime audit."""

from types import SimpleNamespace

import torch

from benchmarks.eval.qwen3_omni_audio_lifetime_audit import audit_request_lifecycle


def test_lifetime_audit_records_chunked_retraction_abort_and_session() -> None:
    audio = torch.empty((3, 4), dtype=torch.bfloat16)
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(
            mm_items=[SimpleNamespace(precomputed_embeddings=audio)]
        )
    )
    transitions = {
        "after_final_prefill_chunk": lambda current: setattr(
            current.multimodal_inputs.mm_items[0], "prefill_complete", True
        ),
        "after_retraction": lambda current: setattr(
            current.multimodal_inputs.mm_items[0], "retracted", True
        ),
    }

    audit = audit_request_lifecycle(
        req,
        request_id="audit-0",
        transitions=transitions,
        session_transition=lambda current: setattr(current, "session_seen", True),
    )

    assert [snapshot.label for snapshot in audit.snapshots] == [
        "after_request_construction",
        "after_final_prefill_chunk",
        "mid_decode",
        "after_completion",
        "after_retraction",
        "after_abort",
        "session_request",
    ]
    assert all(snapshot.req_retains_precomputed_embeddings for snapshot in audit.snapshots)
    assert all(
        snapshot.audio_embedding_tensor_bytes == audio.numel() * audio.element_size()
        for snapshot in audit.snapshots
    )
    assert audit.snapshots[-1].supported is True
    assert audit.cleanup_implemented is False


def test_lifetime_audit_marks_unsupported_session_without_inventing_cleanup() -> None:
    req = SimpleNamespace(multimodal_inputs=SimpleNamespace(mm_items=[]))

    audit = audit_request_lifecycle(req, request_id="audit-no-session")

    assert audit.snapshots[-1].label == "session_request"
    assert audit.snapshots[-1].supported is False
    assert audit.cleanup_implemented is False

# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the independent P16 native reference artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.audit_qwen3_tts_p15a_draft_parity import (
    audit_w1_tail_round,
    expected_mlp_conditioning_depths,
    expected_runtime_proposal_depths,
    proposal_depth_coverage,
)
from scripts.qwen3_tts_p16_native_reference import (
    NATIVE_TALKER_SOURCE_RELATIVE_PATH,
    P16_UPSTREAM_BASE_SHA,
    REPOSITORY_ROOT,
    NativeReferenceSample,
    native_reference_output_rows,
    native_reference_talker,
    native_talker_source_path,
)
from scripts.validate_qwen3_tts_p16_integration import (
    P16_MODEL_REVISION,
    ValidationFailure,
    validate_native_reference,
)
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker


def test_native_reference_uses_scheduler_model_runner_model() -> None:
    talker = object.__new__(Qwen3TTSTalker)
    scheduler = SimpleNamespace(model_runner=SimpleNamespace(model=talker))

    assert native_reference_talker(scheduler) is talker
    assert (
        native_talker_source_path(talker).relative_to(REPOSITORY_ROOT).as_posix()
        == NATIVE_TALKER_SOURCE_RELATIVE_PATH
    )


def test_native_reference_artifact_roundtrip_checks_row_hashes_and_shapes(
    tmp_path: Path,
) -> None:
    sample: NativeReferenceSample = {
        "request_id": "native-reference:sample:0",
        "uid": "sample",
        "ordinal": 0,
        "seed": 19,
        "code0": 23,
        "hidden": torch.zeros((2048,), dtype=torch.bfloat16),
    }
    codes = torch.arange(16, dtype=torch.long).reshape(1, 16, 1)
    feedback = (
        torch.arange(2048, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 2048)
    )
    code_row, feedback_row, record = native_reference_output_rows(
        sample, codes, feedback
    )
    input_identity = {
        "upstream_commit": P16_UPSTREAM_BASE_SHA,
        "model_revision": P16_MODEL_REVISION,
        "model_identity": {},
        "cohort_sha256": "a" * 64,
        "sampling": {
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 1.0,
            "top_k": 50,
        },
        "graph_configuration": {
            "max_running_requests": 1,
            "cuda_graph_max_bs": 1,
            "tp_size": 1,
        },
    }
    reference = {
        "identity": input_identity,
        "metadata": {
            "upstream_commit": P16_UPSTREAM_BASE_SHA,
            "model_revision": P16_MODEL_REVISION,
            "model_source_path": f"/tmp/upstream/{NATIVE_TALKER_SOURCE_RELATIVE_PATH}",
            "model_source_path_relative": NATIVE_TALKER_SOURCE_RELATIVE_PATH,
            "cohort_sha256": input_identity["cohort_sha256"],
            "sampling": input_identity["sampling"],
            "graph_configuration": input_identity["graph_configuration"],
            "native_graph": {
                "enabled": True,
                "replay_count": 1,
                "eager_run_delta": 0,
            },
            "records": [record],
        },
        "codes": torch.stack([code_row]),
        "feedback": torch.stack([feedback_row]),
    }
    artifact_path = tmp_path / "native-reference.pt"
    torch.save(reference, artifact_path)
    loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)

    exact, metadata = validate_native_reference(
        loaded,
        input_identity=input_identity,
        candidate_codes=torch.stack([code_row]),
        candidate_feedback=torch.stack([feedback_row]),
    )
    assert exact is True
    assert metadata["records"][0]["codes_sha256"] == record["codes_sha256"]
    assert metadata["records"][0]["feedback_sha256"] == record["feedback_sha256"]

    changed_reference = dict(loaded)
    changed_reference["codes"] = loaded["codes"].clone()
    changed_reference["codes"][0, 0, 0] += 1
    with pytest.raises(ValidationFailure, match="tensor hashes"):
        validate_native_reference(
            changed_reference,
            input_identity=input_identity,
            candidate_codes=torch.stack([code_row]),
            candidate_feedback=torch.stack([feedback_row]),
        )

    wrong_shape_reference = dict(loaded)
    wrong_shape_reference["codes"] = loaded["codes"].squeeze(-1)
    with pytest.raises(ValidationFailure, match="shapes or dtypes"):
        validate_native_reference(
            wrong_shape_reference,
            input_identity=input_identity,
            candidate_codes=torch.stack([code_row]),
            candidate_feedback=torch.stack([feedback_row]),
        )

    changed_candidate_codes = torch.stack([code_row]).clone()
    changed_candidate_codes[0, 0, 0] += 1
    exact_candidate, _ = validate_native_reference(
        loaded,
        input_identity=input_identity,
        candidate_codes=changed_candidate_codes,
        candidate_feedback=torch.stack([feedback_row]),
    )
    assert exact_candidate is False


def test_p15a_depth_coverage_separates_target_only_w1_depth() -> None:
    assert expected_mlp_conditioning_depths() == tuple(range(1, 15))
    assert expected_runtime_proposal_depths() == tuple(range(1, 14))

    coverage = proposal_depth_coverage({1, 4, 13})
    assert coverage["expected_proposal_depths"] == list(range(1, 14))
    assert coverage["observed_proposal_depths"] == [1, 4, 13]
    assert coverage["missing_proposal_depths"] == [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
    assert coverage["unexpected_proposal_depths"] == []


def test_p15a_w1_tail_requires_one_target_commit_and_no_proposals() -> None:
    assert audit_w1_tail_round(
        verify_width=1, proposal_count=0, committed_token_count=1
    )
    assert not audit_w1_tail_round(
        verify_width=4, proposal_count=3, committed_token_count=2
    )
    with pytest.raises(RuntimeError, match="zero draft proposals"):
        audit_w1_tail_round(verify_width=1, proposal_count=1, committed_token_count=1)
    with pytest.raises(RuntimeError, match="one committed target token"):
        audit_w1_tail_round(verify_width=1, proposal_count=0, committed_token_count=0)

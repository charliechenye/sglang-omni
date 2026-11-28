# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the independent P16 native reference artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.audit_qwen3_tts_p15a_draft_parity import (
    audit_outcome,
    audit_w1_tail_round,
    expected_mlp_conditioning_depths,
    expected_runtime_proposal_depths,
    numerical_parity,
    proposal_depth_coverage,
)
from scripts.qwen3_tts_p16_native_reference import (
    NATIVE_REFERENCE_IMPORTS,
    NATIVE_TALKER_SOURCE_RELATIVE_PATH,
    P16_UPSTREAM_BASE_SHA,
    REPOSITORY_ROOT,
    NativeReferenceSample,
    native_reference_output_rows,
    native_reference_import_source_paths,
    native_reference_talker,
)
from scripts.validate_qwen3_tts_p16_integration import (
    P16_MODEL_REVISION,
    P16_NATIVE_REFERENCE_IMPORTS,
    ValidationFailure,
    validate_native_reference,
)
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker


def test_native_reference_uses_scheduler_model_runner_model() -> None:
    assert P16_NATIVE_REFERENCE_IMPORTS == NATIVE_REFERENCE_IMPORTS
    talker = object.__new__(Qwen3TTSTalker)
    scheduler = SimpleNamespace(model_runner=SimpleNamespace(model=talker))

    assert native_reference_talker(scheduler) is talker
    imported_paths = native_reference_import_source_paths(talker)
    assert set(imported_paths) == set(NATIVE_REFERENCE_IMPORTS)
    assert Path(imported_paths["talker"]).relative_to(REPOSITORY_ROOT).as_posix() == (
        NATIVE_TALKER_SOURCE_RELATIVE_PATH
    )


def test_native_reference_artifact_roundtrip_checks_identity_order_hashes_and_shapes(
    tmp_path: Path,
) -> None:
    samples: list[NativeReferenceSample] = [
        {
            "request_id": "native-reference:sample:0",
            "uid": "sample",
            "ordinal": 0,
            "seed": 19,
            "code0": 23,
            "hidden": torch.zeros((2048,), dtype=torch.bfloat16),
        },
        {
            "request_id": "native-reference:sample:1",
            "uid": "sample",
            "ordinal": 1,
            "seed": 20,
            "code0": 24,
            "hidden": torch.ones((2048,), dtype=torch.bfloat16),
        },
    ]
    codes = torch.arange(32, dtype=torch.long).reshape(2, 16, 1)
    feedback = (
        torch.arange(4096, dtype=torch.float32).to(torch.bfloat16).reshape(2, 1, 2048)
    )
    output_rows = [
        native_reference_output_rows(
            sample, codes[index : index + 1], feedback[index : index + 1]
        )
        for index, sample in enumerate(samples)
    ]
    code_rows = torch.stack([row[0] for row in output_rows])
    feedback_rows = torch.stack([row[1] for row in output_rows])
    records = [row[2] for row in output_rows]
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
            "reference_source_root": "/tmp/upstream",
            "imported_source_paths": {
                name: f"/tmp/upstream/{relative_path}"
                for name, relative_path in NATIVE_REFERENCE_IMPORTS.items()
            },
            "cohort_sha256": input_identity["cohort_sha256"],
            "sampling": input_identity["sampling"],
            "graph_configuration": input_identity["graph_configuration"],
            "native_graph": {
                "enabled": True,
                "replay_count": 2,
                "eager_run_delta": 0,
            },
            "records": records,
        },
        "inputs": samples,
        "codes": code_rows,
        "feedback": feedback_rows,
    }
    artifact_path = tmp_path / "native-reference.pt"
    torch.save(reference, artifact_path)
    loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)

    exact, metadata = validate_native_reference(
        loaded,
        input_identity=input_identity,
        candidate_codes=code_rows,
        candidate_feedback=feedback_rows,
    )
    assert exact is True
    assert [record["request_id"] for record in metadata["records"]] == [
        sample["request_id"] for sample in samples
    ]
    assert [record["seed"] for record in metadata["records"]] == [
        sample["seed"] for sample in samples
    ]
    assert metadata["reference_source_root"] == "/tmp/upstream"
    assert (
        metadata["imported_source_paths"]
        == reference["metadata"]["imported_source_paths"]
    )
    for record, expected in zip(metadata["records"], records):
        assert record["codes_sha256"] == expected["codes_sha256"]
        assert record["feedback_sha256"] == expected["feedback_sha256"]

    changed_reference = dict(loaded)
    changed_reference["codes"] = loaded["codes"].clone()
    changed_reference["codes"][0, 0, 0] += 1
    with pytest.raises(ValidationFailure, match="tensor hashes"):
        validate_native_reference(
            changed_reference,
            input_identity=input_identity,
            candidate_codes=code_rows,
            candidate_feedback=feedback_rows,
        )

    wrong_shape_reference = dict(loaded)
    wrong_shape_reference["codes"] = loaded["codes"].squeeze(-1)
    with pytest.raises(ValidationFailure, match="shapes or dtypes"):
        validate_native_reference(
            wrong_shape_reference,
            input_identity=input_identity,
            candidate_codes=code_rows,
            candidate_feedback=feedback_rows,
        )

    reordered_reference = dict(loaded)
    reordered_reference["metadata"] = dict(loaded["metadata"])
    reordered_records = [dict(record) for record in loaded["metadata"]["records"]]
    for field in ("request_id", "uid", "ordinal", "seed"):
        reordered_records[0][field], reordered_records[1][field] = (
            reordered_records[1][field],
            reordered_records[0][field],
        )
    reordered_reference["metadata"]["records"] = reordered_records
    with pytest.raises(ValidationFailure, match="wrong input or order"):
        validate_native_reference(
            reordered_reference,
            input_identity=input_identity,
            candidate_codes=code_rows,
            candidate_feedback=feedback_rows,
        )

    candidate_import_reference = dict(loaded)
    candidate_import_reference["metadata"] = dict(loaded["metadata"])
    candidate_import_reference["metadata"]["imported_source_paths"] = dict(
        loaded["metadata"]["imported_source_paths"]
    )
    candidate_import_reference["metadata"]["imported_source_paths"][
        "engine_builder"
    ] = str(REPOSITORY_ROOT / NATIVE_REFERENCE_IMPORTS["engine_builder"])
    with pytest.raises(ValidationFailure, match="outside its recorded source checkout"):
        validate_native_reference(
            candidate_import_reference,
            input_identity=input_identity,
            candidate_codes=code_rows,
            candidate_feedback=feedback_rows,
        )

    changed_candidate_codes = code_rows.clone()
    changed_candidate_codes[0, 0, 0] += 1
    exact_candidate, _ = validate_native_reference(
        loaded,
        input_identity=input_identity,
        candidate_codes=changed_candidate_codes,
        candidate_feedback=feedback_rows,
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


def test_full_audit_coverage_and_numerical_parity_are_separate_gates() -> None:
    assert audit_outcome(
        full_audit=True,
        runtime_coverage_complete=False,
        numerical_passed=True,
    ) == ("P15A_DRAFT_PARITY_REVIEW", 1)
    assert audit_outcome(
        full_audit=False,
        runtime_coverage_complete=False,
        numerical_passed=True,
    ) == ("P15A_DRAFT_PARITY_SMOKE_PARTIAL_COVERAGE", 0)
    assert audit_outcome(
        full_audit=False,
        runtime_coverage_complete=False,
        numerical_passed=False,
    ) == ("P15A_DRAFT_PARITY_REVIEW", 1)

    totals = {
        "standalone_mlp_steps": 14,
        "standalone_mlp_exact": 14,
        "mlp_steps": 13,
        "mlp_exact": 13,
        "draft_logits": 130,
        "draft_logits_exact": 130,
        "captured_draft_logits_exact": 13,
        "candidate_probability_exact": 13,
    }
    transformed_weights = {"draft": {"exact": True}}
    assert numerical_parity(totals, transformed_weights)["status"] == "PASS"
    totals["standalone_mlp_exact"] = 13
    assert numerical_parity(totals, transformed_weights)["status"] == "FAIL"


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

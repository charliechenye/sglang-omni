#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the P16 draft path with the pinned P14/P15A runtime on P12 contexts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypedDict

import torch

from scripts import validate_qwen3_tts_p16_integration as p16
from scripts.qwen3_tts_p15a_parity_reference import (
    P15A_KERNEL_SOURCE_COMMIT,
    P15A_SOURCE_COMMIT,
    ReferenceDraft,
    distribution_delta,
    load_p15a_grouped_head_module,
    load_p15a_radix_module,
    load_p15a_references,
    summarize_deltas,
)
from sglang_omni.models.qwen3_tts.speculative_draft import load_p16_draft_checkpoint
from sglang_omni.utils.predictor_grouped_heads import grouped_lm_head_warp_accumulate


class NumericalParityReport(TypedDict):
    status: Literal["PASS", "FAIL"]
    checks: dict[str, bool]
    probability_backend_comparison: Literal["DIAGNOSTIC_ONLY_TV"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--p12-root", type=Path, required=True)
    parser.add_argument("--p15a-source-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(4, 128), required=True)
    parser.add_argument("--seeds", type=int, choices=(1, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def expected_mlp_conditioning_depths() -> tuple[int, ...]:
    return tuple(range(1, p16.P16_GROUPS - 1))


def expected_runtime_proposal_depths() -> tuple[int, ...]:
    return tuple(
        depth
        for depth in expected_mlp_conditioning_depths()
        if min(p16.P16_VERIFY_WIDTH, p16.P16_GROUPS - 1 - depth) > 1
    )


def proposal_depth_coverage(
    observed_depths: set[int],
) -> dict[str, list[int] | bool]:
    expected_depths = set(expected_runtime_proposal_depths())
    actual_depths = set(observed_depths)
    return {
        "expected_proposal_depths": sorted(expected_depths),
        "observed_proposal_depths": sorted(actual_depths),
        "missing_proposal_depths": sorted(expected_depths - actual_depths),
        "unexpected_proposal_depths": sorted(actual_depths - expected_depths),
    }


def numerical_parity(
    totals: Mapping[str, int],
    transformed_weights: Mapping[str, Mapping[str, bool | float | int]],
) -> NumericalParityReport:
    checks = {
        "transformed_weights_exact": all(
            bool(value["exact"]) for value in transformed_weights.values()
        ),
        "standalone_mlp_exact": totals["standalone_mlp_steps"] > 0
        and totals["standalone_mlp_exact"] == totals["standalone_mlp_steps"],
        "runtime_mlp_exact": totals["mlp_steps"] > 0
        and totals["mlp_exact"] == totals["mlp_steps"],
        "draft_logits_exact": totals["draft_logits"] > 0
        and totals["draft_logits_exact"] == totals["draft_logits"],
        "captured_draft_logits_exact": totals["captured_draft_logits_exact"]
        == totals["mlp_steps"],
        "production_probabilities_and_ids_exact": totals["candidate_probability_exact"]
        == totals["mlp_steps"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "probability_backend_comparison": "DIAGNOSTIC_ONLY_TV",
    }


def audit_outcome(
    *,
    full_audit: bool,
    runtime_coverage_complete: bool,
    numerical_passed: bool,
) -> tuple[str, int]:
    if not numerical_passed or (full_audit and not runtime_coverage_complete):
        return "P15A_DRAFT_PARITY_REVIEW", 1
    elif full_audit:
        return "P15A_DRAFT_PARITY_PASS", 0
    elif runtime_coverage_complete:
        return "P15A_DRAFT_PARITY_SMOKE_COVERAGE_COMPLETE", 0
    return "P15A_DRAFT_PARITY_SMOKE_PARTIAL_COVERAGE", 0


def audit_w1_tail_round(
    verify_width: int,
    proposal_count: int,
    committed_token_count: int,
) -> bool:
    if verify_width == 1:
        if proposal_count != 0 or committed_token_count != 1:
            raise RuntimeError(
                "W1 tail must have zero draft proposals and one committed target token"
            )
        return True
    else:
        return False


def main() -> int:
    arguments = parse_arguments()
    if arguments.out.exists() or not arguments.out.is_absolute():
        raise ValueError("--out must be a new absolute path")
    if not torch.cuda.is_available():
        raise RuntimeError("P15A draft numerical parity requires CUDA")
    model_path = arguments.model.resolve(strict=True)
    draft_path = arguments.draft.resolve(strict=True)
    device = p16.require_h200()
    model_identity = p16.model_identity(model_path)
    _, cohort, cohort_sha, cohort_identity = p16.validation_rows(
        arguments.p12_root,
        model_identity,
        arguments.frames,
        arguments.seeds,
    )
    fused_width_module, head_runtime_module, source_identity = load_p15a_references(
        arguments.p15a_source_root.resolve(strict=True)
    )
    radix_module, radix_identity = load_p15a_radix_module(p16.REPOSITORY_ROOT)
    grouped_head_module, grouped_head_identity = load_p15a_grouped_head_module(
        p16.REPOSITORY_ROOT
    )
    checkpoint = load_p16_draft_checkpoint(draft_path, p16.P16_DRAFT_SHA256)
    scheduler, talker = p16.build_talker(
        model_path,
        device,
        enabled=True,
        draft_path=draft_path,
        max_running_requests=1,
    )
    try:
        production_draft = talker.code_predictor_p16_draft
        projected_tables = talker.predictor_projected_embeddings
        if production_draft is None or projected_tables is None:
            raise RuntimeError(
                "P16 draft weights or projected codec tables are missing"
            )
        reference_net = ReferenceDraft().to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            for name, value in (
                ("hidden_scale", checkpoint.weights["hidden_scale"]),
                ("embed_scale", checkpoint.weights["embed_scale"]),
                ("in_proj.weight", checkpoint.weights["in_proj.weight"]),
                ("in_proj.bias", checkpoint.weights["in_proj.bias"]),
                ("out_proj.weight", checkpoint.weights["out_proj.weight"]),
                ("out_proj.bias", checkpoint.weights["out_proj.bias"]),
            ):
                if name == "hidden_scale":
                    reference_net.hidden_scale.copy_(value)
                elif name == "embed_scale":
                    reference_net.embed_scale.copy_(value)
                elif name == "in_proj.weight":
                    reference_net.in_proj.weight.copy_(value)
                elif name == "in_proj.bias":
                    reference_net.in_proj.bias.copy_(value)
                elif name == "out_proj.weight":
                    reference_net.out_proj.weight.copy_(value)
                else:
                    reference_net.out_proj.bias.copy_(value)
        reference_draft = fused_width_module.FusedWidth(reference_net, projected_tables)
        transformed = {
            "folded_hidden_weight": (
                production_draft.hidden_weight,
                reference_draft.wh,
            ),
            "folded_output_weight": (
                production_draft.output_weight,
                reference_draft.wo,
            ),
            "folded_output_bias": (
                production_draft.output_bias,
                reference_draft.bo,
            ),
            "embedding_depth_lut": (
                production_draft.embedding_depth_lut,
                reference_draft.table.reshape_as(production_draft.embedding_depth_lut),
            ),
        }
        transform_metrics: dict[str, dict[str, float | int | bool]] = {}
        for name, (actual, expected) in transformed.items():
            difference = (actual.float() - expected.float()).abs()
            transform_metrics[name] = {
                "exact": torch.equal(actual, expected),
                "equal_elements": int(torch.count_nonzero(actual == expected)),
                "elements": actual.numel(),
                "max_abs": float(difference.max()),
            }

        heads = talker.code_predictor_spec_lm_head_weights
        if heads is None:
            raise RuntimeError("packed Predictor LM-head weights are missing")
        reference_head = object.__new__(head_runtime_module.HeadRuntime)
        reference_head.radix = (
            radix_module.build_speculative_probability_candidates_radix
        )
        reference_head.ones = torch.ones(5, device=device, dtype=torch.bool)
        reference_head.temps = torch.full((5,), 0.9, device=device, dtype=torch.float32)
        reference_head.topk = torch.full((5,), 50, device=device, dtype=torch.long)
        reference_head.topp = torch.ones(5, device=device, dtype=torch.float32)

        totals = {
            "mlp_steps": 0,
            "mlp_exact": 0,
            "mlp_elements": 0,
            "mlp_equal_elements": 0,
            "standalone_mlp_steps": 0,
            "standalone_mlp_exact": 0,
            "standalone_mlp_elements": 0,
            "standalone_mlp_equal_elements": 0,
            "standalone_mlp_max_abs": 0.0,
            "standalone_depths": set(),
            "draft_logits": 0,
            "draft_logits_exact": 0,
            "captured_draft_logits_exact": 0,
            "candidate_probability_exact": 0,
            "candidate_probability_elements": 0,
            "candidate_probability_token_id_elements": 0,
            "proposal_chain_max_width": 0,
            "proposal_depths": set(),
            "w1_tail_rounds": 0,
            "w1_tail_proposals": 0,
            "w1_tail_committed_tokens": 0,
            "mlp_max_abs": 0.0,
            "draft_logits_max_abs": 0.0,
            "candidate_split_vs_p15a_radix_same_logits": [],
            "p15a_radix_candidate_vs_reference": [],
            "production_split_vs_p15a_reference": [],
        }
        audit_capture = p16.P16AuditCapture()
        for frame in cohort:
            for repeat in range(arguments.seeds):
                seed = p16.seed_for(frame.uid, frame.ordinal, repeat)
                request = p16.create_request(
                    f"p15a:{frame.uid}:{frame.ordinal}:{seed}", seed
                )
                _, _, captured, _ = p16.capture_frame(
                    talker, frame, request, seed, audit_capture
                )
                if captured.opening is None:
                    raise RuntimeError("P16 proposal trace has no opening context")
                opening_hidden = (
                    p16.tensor_field(captured.opening, "preceding_hidden")
                    .to(device=device, dtype=torch.bfloat16)
                    .contiguous()
                )
                opening_codes = p16.tensor_field(captured.opening, "output_codes")
                opening_pending = (
                    opening_codes[0, 1].to(device=device, dtype=torch.long).reshape(1)
                )
                for standalone_depth in expected_mlp_conditioning_depths():
                    depth_tensor = torch.tensor(
                        [standalone_depth], device=device, dtype=torch.long
                    )
                    candidate_hidden = production_draft.step(
                        opening_hidden,
                        opening_pending,
                        standalone_depth,
                    ).clone()
                    reference_hidden = reference_draft(
                        opening_hidden.contiguous(),
                        opening_pending.contiguous(),
                        depth_tensor,
                    ).clone()
                    hidden_difference = (
                        candidate_hidden.float() - reference_hidden.float()
                    ).abs()
                    totals["standalone_depths"].add(standalone_depth)
                    totals["standalone_mlp_steps"] += 1
                    totals["standalone_mlp_exact"] += int(
                        torch.equal(candidate_hidden, reference_hidden)
                    )
                    totals["standalone_mlp_elements"] += candidate_hidden.numel()
                    totals["standalone_mlp_equal_elements"] += int(
                        torch.count_nonzero(candidate_hidden == reference_hidden)
                    )
                    totals["standalone_mlp_max_abs"] = max(
                        totals["standalone_mlp_max_abs"],
                        float(hidden_difference.max()),
                    )
                round_start = opening_hidden
                for record in captured.rounds:
                    depth = p16.integer_field(record, "round_depth")
                    verify_width = p16.integer_field(record, "verify_width")
                    proposal_count = p16.integer_field(record, "proposal_count")
                    proposal_tokens = p16.tensor_field(record, "proposal_tokens").to(
                        device=device, dtype=torch.long
                    )
                    proposal_logits = p16.tensor_field(record, "proposal_logits").to(
                        device=device, dtype=torch.bfloat16
                    )
                    pending = p16.tensor_field(record, "pending_tokens").to(
                        device=device, dtype=torch.long
                    )
                    width = int(proposal_tokens.shape[1])
                    if width != proposal_count or proposal_count != verify_width - 1:
                        raise RuntimeError(
                            "actual proposal width differs from the fixed K3 round contract"
                        )
                    committed_token_count = int(
                        p16.tensor_field(record, "committed_token_counts")[0]
                    )
                    if audit_w1_tail_round(
                        verify_width, proposal_count, committed_token_count
                    ):
                        totals["w1_tail_rounds"] += 1
                        totals["w1_tail_proposals"] += proposal_count
                        totals["w1_tail_committed_tokens"] += committed_token_count
                    totals["proposal_chain_max_width"] = max(
                        totals["proposal_chain_max_width"], width
                    )
                    for proposal_index in range(width):
                        proposal_depth = depth + proposal_index
                        totals["proposal_depths"].add(proposal_depth)
                        token = (
                            pending
                            if proposal_index == 0
                            else proposal_tokens[:, proposal_index - 1]
                        )
                        candidate_hidden = production_draft.step(
                            round_start,
                            token,
                            proposal_depth,
                        ).clone()
                        token_id = token.contiguous()
                        depth_tensor = torch.tensor(
                            [proposal_depth], device=device, dtype=torch.long
                        )
                        reference_hidden = reference_draft(
                            round_start.contiguous(),
                            token_id,
                            depth_tensor,
                        ).clone()
                        hidden_difference = (
                            candidate_hidden.float() - reference_hidden.float()
                        ).abs()
                        totals["mlp_steps"] += 1
                        totals["mlp_exact"] += int(
                            torch.equal(candidate_hidden, reference_hidden)
                        )
                        totals["mlp_max_abs"] = max(
                            totals["mlp_max_abs"], float(hidden_difference.max())
                        )
                        totals["mlp_elements"] += candidate_hidden.numel()
                        totals["mlp_equal_elements"] += int(
                            torch.count_nonzero(candidate_hidden == reference_hidden)
                        )
                        head_index = torch.tensor(
                            [proposal_depth], device=device, dtype=torch.long
                        )
                        active = torch.ones(1, device=device, dtype=torch.bool)
                        candidate_logits = grouped_lm_head_warp_accumulate(
                            candidate_hidden,
                            heads,
                            None,
                            head_index,
                            active,
                        )
                        reference_logits = grouped_head_module.grouped_lm_head_linear(
                            reference_hidden,
                            heads,
                            None,
                            head_index,
                            active,
                            backend="warp_accumulate",
                        )
                        recorded_logits = proposal_logits[
                            proposal_index : proposal_index + 1
                        ].contiguous()
                        totals["captured_draft_logits_exact"] += int(
                            torch.equal(candidate_logits, recorded_logits)
                        )
                        logit_difference = (
                            candidate_logits.float() - reference_logits.float()
                        ).abs()
                        totals["draft_logits"] += candidate_logits.numel()
                        totals["draft_logits_exact"] += int(
                            torch.count_nonzero(candidate_logits == reference_logits)
                        )
                        totals["draft_logits_max_abs"] = max(
                            totals["draft_logits_max_abs"],
                            float(logit_difference.max()),
                        )
                        candidate_split, candidate_split_ids = (
                            p16.build_compact_probabilities(candidate_logits)
                        )
                        captured_probabilities = p16.tensor_field(
                            record, "proposal_probabilities"
                        )[proposal_index : proposal_index + 1].to(device=device)
                        captured_ids = p16.tensor_field(record, "proposal_token_ids")[
                            proposal_index : proposal_index + 1
                        ].to(device=device)
                        totals["candidate_probability_exact"] += int(
                            torch.equal(candidate_split, captured_probabilities)
                            and torch.equal(candidate_split_ids, captured_ids)
                        )
                        totals[
                            "candidate_probability_elements"
                        ] += candidate_split.numel()
                        totals[
                            "candidate_probability_token_id_elements"
                        ] += candidate_split_ids.numel()
                        candidate_radix, candidate_radix_ids = reference_head.compact(
                            candidate_logits.contiguous()
                        )
                        reference_radix, reference_radix_ids = reference_head.compact(
                            reference_logits.contiguous()
                        )
                        totals["candidate_split_vs_p15a_radix_same_logits"].append(
                            distribution_delta(
                                candidate_split,
                                candidate_split_ids,
                                candidate_radix,
                                candidate_radix_ids,
                                p16.P16_VOCAB_SIZE,
                            )
                        )
                        totals["p15a_radix_candidate_vs_reference"].append(
                            distribution_delta(
                                candidate_radix,
                                candidate_radix_ids,
                                reference_radix,
                                reference_radix_ids,
                                p16.P16_VOCAB_SIZE,
                            )
                        )
                        totals["production_split_vs_p15a_reference"].append(
                            distribution_delta(
                                candidate_split,
                                candidate_split_ids,
                                reference_radix,
                                reference_radix_ids,
                                p16.P16_VOCAB_SIZE,
                            )
                        )
                        round_start = candidate_hidden
                    round_start = (
                        p16.tensor_field(record, "preceding_hidden")
                        .to(device=device, dtype=torch.bfloat16)
                        .contiguous()
                    )
                audit_capture.release(captured)
        if set(totals["standalone_depths"]) != set(expected_mlp_conditioning_depths()):
            raise RuntimeError(
                "standalone MLP replay did not cover conditioning depths "
                f"1-14: {sorted(totals['standalone_depths'])}"
            )
        elif totals["proposal_chain_max_width"] < 2:
            raise RuntimeError("P12 cohort did not exercise chained draft proposals")
        elif totals["w1_tail_rounds"] == 0:
            raise RuntimeError("P12 cohort did not exercise the W1 target-only tail")
        elif (
            totals["w1_tail_proposals"] != 0
            or totals["w1_tail_committed_tokens"] != totals["w1_tail_rounds"]
        ):
            raise RuntimeError("W1 target-only rounds violated their commit contract")

        proposal_coverage = proposal_depth_coverage(totals["proposal_depths"])
        if proposal_coverage["unexpected_proposal_depths"]:
            raise RuntimeError(
                "actual K3 proposals included illegal depths: "
                f"{proposal_coverage['unexpected_proposal_depths']}"
            )

        runtime_coverage_complete = not proposal_coverage["missing_proposal_depths"]
        proposal_coverage["complete"] = runtime_coverage_complete
        proposal_coverage["required_for_full_audit"] = True
        numerical_counts: dict[str, int] = {
            "standalone_mlp_steps": totals["standalone_mlp_steps"],
            "standalone_mlp_exact": totals["standalone_mlp_exact"],
            "mlp_steps": totals["mlp_steps"],
            "mlp_exact": totals["mlp_exact"],
            "draft_logits": totals["draft_logits"],
            "draft_logits_exact": totals["draft_logits_exact"],
            "captured_draft_logits_exact": totals["captured_draft_logits_exact"],
            "candidate_probability_exact": totals["candidate_probability_exact"],
        }
        numerical = numerical_parity(numerical_counts, transform_metrics)
        status, exit_code = audit_outcome(
            full_audit=arguments.frames == 128 and arguments.seeds == 4,
            runtime_coverage_complete=runtime_coverage_complete,
            numerical_passed=numerical["status"] == "PASS",
        )

        output = {
            "schema": "qtts-p15a-draft-parity-v1",
            "status": status,
            "candidate_sha": p16.current_git_sha(),
            "reference": {
                "sglang_scripts_commit": P15A_SOURCE_COMMIT,
                "source_sha256": source_identity,
                "kernel_source_commit": P15A_KERNEL_SOURCE_COMMIT,
                "radix_kernel": radix_identity,
                "grouped_head_kernel": grouped_head_identity,
            },
            "model_revision": p16.P16_MODEL_REVISION,
            "draft_sha256": p16.P16_DRAFT_SHA256,
            "cohort_sha256": cohort_sha,
            "cohort_identity": cohort_identity,
            "standalone_mlp": {
                "expected_conditioning_depths": list(
                    expected_mlp_conditioning_depths()
                ),
                "observed_conditioning_depths": sorted(totals["standalone_depths"]),
                "steps": totals["standalone_mlp_steps"],
                "exact_steps": totals["standalone_mlp_exact"],
                "elements": totals["standalone_mlp_elements"],
                "exact_elements": totals["standalone_mlp_equal_elements"],
                "max_abs": totals["standalone_mlp_max_abs"],
            },
            "runtime_proposal_coverage": proposal_coverage,
            "numerical_parity": numerical,
            "proposal_chain_max_width": totals["proposal_chain_max_width"],
            "w1_tail": {
                "rounds": totals["w1_tail_rounds"],
                "draft_proposals": totals["w1_tail_proposals"],
                "committed_target_tokens": totals["w1_tail_committed_tokens"],
            },
            "transformed_weights": transform_metrics,
            "mlp": {
                "steps": totals["mlp_steps"],
                "exact_steps": totals["mlp_exact"],
                "elements": totals["mlp_elements"],
                "exact_elements": totals["mlp_equal_elements"],
                "max_abs": totals["mlp_max_abs"],
            },
            "draft_logits": {
                "elements": totals["draft_logits"],
                "exact_elements": totals["draft_logits_exact"],
                "max_abs": totals["draft_logits_max_abs"],
            },
            "probabilities": {
                "production_builder": "split_top_k",
                "p15a_builder": "radix_compact",
                "candidate_split_replayed_rows": totals["candidate_probability_exact"],
                "candidate_split_probability_elements": totals[
                    "candidate_probability_elements"
                ],
                "candidate_split_token_id_elements": totals[
                    "candidate_probability_token_id_elements"
                ],
                "split_vs_p15a_radix_same_logits": summarize_deltas(
                    totals["candidate_split_vs_p15a_radix_same_logits"]
                ),
                "p15a_radix_candidate_vs_reference_logits": summarize_deltas(
                    totals["p15a_radix_candidate_vs_reference"]
                ),
                "production_split_vs_p15a_reference": summarize_deltas(
                    totals["production_split_vs_p15a_reference"]
                ),
            },
        }
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        return exit_code
    finally:
        p16.close_runtime(scheduler)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Independent CUDA microbenchmarks for Qwen3-TTS speculative kernels."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    TOP_K_CHUNKS,
    build_speculative_probability_candidates_selected,
    fused_raw_logit_block_k,
)
from sglang_omni.models.qwen3_tts.speculative_acceptance import (
    SpeculativeAcceptanceBatch,
    accept_speculative_candidates_compact,
    accept_speculative_candidates_dense_indexed,
    sample_compact_probability_rows_fused,
    update_speculative_dense_probability_table,
)
from sglang_omni.utils.json import JsonValue
from sglang_omni.utils.predictor_grouped_heads import (
    grouped_lm_head_linear,
    grouped_lm_head_warp_accumulate_available,
    grouped_lm_head_warp_gemv_available,
)
from sglang_omni.utils.qwen3_tts_speculative_frozen import FrozenPredictorArtifacts
from sglang_omni.utils.qwen3_tts_speculative_kernel_checks import (
    allocate_acceptance_outputs,
    assert_grouped_lm_head_rounding_bound,
    assert_probability_parity,
    build_microbenchmark_tensors,
    cuda_median_ms,
)


def run_microbenchmarks(
    artifacts: FrozenPredictorArtifacts,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, JsonValue]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for speculative kernel microbenchmarks")
    else:
        pass
    device = torch.device("cuda")
    frame_count = len(artifacts.frames)
    hidden_states, logits, temperatures, top_ks, top_ps = build_microbenchmark_tensors(
        artifacts, device=device
    )
    _, head_count, hidden_size = hidden_states.shape
    vocab_size = logits.shape[-1]
    row_count = frame_count * head_count
    flat_hidden_states = hidden_states.reshape(row_count, hidden_size).contiguous()
    flat_reference_logits = logits.reshape(row_count, vocab_size).contiguous()
    head_indices = torch.arange(head_count, device=device, dtype=torch.long).repeat(
        frame_count
    )
    active_rows = torch.ones(row_count, device=device, dtype=torch.bool)
    packed_weights = artifacts.lm_head_weights.to(device=device).contiguous()
    target_logits = torch.empty_like(flat_reference_logits)
    linear_rows: dict[str, dict[str, float | int]] = {}
    lm_head_backends = ["grouped_triton"]
    if grouped_lm_head_warp_gemv_available():
        lm_head_backends.append("warp_gemv")
    else:
        pass
    if grouped_lm_head_warp_accumulate_available():
        lm_head_backends.append("warp_accumulate")
    else:
        pass
    for backend in lm_head_backends:

        def project_rows() -> None:
            grouped_lm_head_linear(
                flat_hidden_states,
                packed_weights,
                None,
                head_indices,
                active_rows,
                backend=backend,
                output=target_logits,
            )

        project_rows()
        assert_grouped_lm_head_rounding_bound(
            target_logits[:head_count],
            flat_hidden_states[:head_count],
            packed_weights,
            head_indices[:head_count],
        )
        linear_rows[backend] = {
            "median_cuda_ms": cuda_median_ms(
                project_rows, warmup=warmup, iterations=iterations
            ),
            "rows": row_count,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
        }

    max_top_k = int(top_ks.max().item())
    candidate_width = fused_raw_logit_block_k(max_top_k)
    if candidate_width is None:
        raise RuntimeError(f"frozen Top-K {max_top_k} is unsupported")
    else:
        pass
    row_temperatures = temperatures.repeat_interleave(head_count).contiguous()
    row_top_ks = top_ks.repeat_interleave(head_count).contiguous()
    row_top_ps = top_ps.repeat_interleave(head_count).contiguous()
    has_top_p = bool((top_ps < 1.0).any().item())
    row_active = torch.ones(row_count, device=device, dtype=torch.bool)
    split_result = build_speculative_probability_candidates_selected(
        flat_reference_logits,
        row_temperatures,
        row_top_ks,
        row_top_ps,
        row_active,
        max_top_k=max_top_k,
        has_top_p=has_top_p,
        has_unbounded_top_k=False,
        backend="split_top_k",
    )
    if split_result is None or split_result[2] != "split_top_k":
        raise RuntimeError("split Top-K microbenchmark did not execute split Top-K")
    else:
        pass
    split_probabilities, split_token_ids, _ = split_result
    probability_output = torch.empty(
        row_count, candidate_width, device=device, dtype=torch.float32
    )
    token_id_output = torch.empty(
        row_count, candidate_width, device=device, dtype=torch.int32
    )
    split_candidate_keys = torch.empty(
        row_count,
        TOP_K_CHUNKS * candidate_width,
        device=device,
        dtype=torch.int64,
    )
    radix_candidate_keys = torch.empty(
        row_count, candidate_width, device=device, dtype=torch.int64
    )
    probability_rows: dict[str, dict[str, float | int | str]] = {}
    probability_backends = ("split_top_k", "radix_compact", "radix_compact_4bit")
    for backend in probability_backends:

        def build_probabilities() -> None:
            result = build_speculative_probability_candidates_selected(
                flat_reference_logits,
                row_temperatures,
                row_top_ks,
                row_top_ps,
                row_active,
                max_top_k=max_top_k,
                has_top_p=has_top_p,
                has_unbounded_top_k=False,
                backend=backend,
                probability_output=probability_output,
                token_id_output=token_id_output,
                split_candidate_keys=split_candidate_keys,
                radix_candidate_keys=radix_candidate_keys,
            )
            if result is None or result[2] != backend:
                raise RuntimeError(
                    f"requested {backend} probability path did not execute"
                )
            else:
                pass

        build_probabilities()
        assert_probability_parity(
            probability_output,
            token_id_output,
            split_probabilities,
            split_token_ids,
        )
        probability_rows[backend] = {
            "median_cuda_ms": cuda_median_ms(
                build_probabilities, warmup=warmup, iterations=iterations
            ),
            "actual_backend": backend,
            "rows": row_count,
            "candidate_width": candidate_width,
        }

    sample_result = build_speculative_probability_candidates_selected(
        flat_reference_logits,
        row_temperatures,
        row_top_ks,
        row_top_ps,
        row_active,
        max_top_k=max_top_k,
        has_top_p=has_top_p,
        has_unbounded_top_k=False,
        backend="split_top_k",
    )
    assert sample_result is not None
    draft_probabilities = (
        sample_result[0]
        .view(frame_count, head_count, candidate_width)[:, 0]
        .contiguous()
    )
    draft_token_ids = (
        sample_result[1]
        .view(frame_count, head_count, candidate_width)[:, 0]
        .contiguous()
    )
    proposal_width = 1
    verify_width = proposal_width + 1
    proposal_uniforms = torch.full(
        (frame_count, proposal_width), 0.5, device=device, dtype=torch.float32
    )
    active_proposals = torch.ones(
        (frame_count, proposal_width), device=device, dtype=torch.bool
    )
    draft_tokens = sample_compact_probability_rows_fused(
        draft_probabilities,
        draft_token_ids,
        proposal_uniforms,
        active_proposals,
    )
    target_probabilities = (
        split_probabilities.view(frame_count, head_count, candidate_width)[
            :, :verify_width
        ]
        .reshape(frame_count * verify_width, candidate_width)
        .contiguous()
    )
    target_token_ids = (
        split_token_ids.view(frame_count, head_count, candidate_width)[:, :verify_width]
        .reshape(frame_count * verify_width, candidate_width)
        .contiguous()
    )
    proposal_counts = torch.ones(frame_count, device=device, dtype=torch.long)
    active_batch_rows = torch.ones(frame_count, device=device, dtype=torch.bool)
    acceptance_uniforms = torch.full(
        (frame_count, proposal_width), 0.5, device=device, dtype=torch.float32
    )
    correction_uniforms = torch.full(
        (frame_count, verify_width), 0.5, device=device, dtype=torch.float32
    )
    bonus_uniforms = torch.full_like(correction_uniforms, 0.5)
    compact_outputs = allocate_acceptance_outputs(frame_count, verify_width, device)
    dense_outputs = allocate_acceptance_outputs(frame_count, verify_width, device)
    dense_vocab_size = vocab_size
    dense_draft = torch.zeros(
        (frame_count, proposal_width, dense_vocab_size),
        device=device,
        dtype=torch.float32,
    )
    dense_target = torch.zeros(
        (frame_count, verify_width, dense_vocab_size),
        device=device,
        dtype=torch.float32,
    )
    draft_previous_ids = torch.zeros_like(draft_token_ids, dtype=torch.long)
    draft_current_ids = torch.empty_like(draft_previous_ids)
    target_previous_ids = torch.zeros_like(target_token_ids, dtype=torch.long)
    target_current_ids = torch.empty_like(target_previous_ids)
    candidate_offsets = torch.arange(candidate_width, device=device, dtype=torch.long)

    def update_dense_tables() -> None:
        update_speculative_dense_probability_table(
            draft_probabilities,
            draft_token_ids,
            dense_draft.view(frame_count, dense_vocab_size),
            draft_previous_ids,
            draft_current_ids,
        )
        update_speculative_dense_probability_table(
            target_probabilities,
            target_token_ids,
            dense_target.view(frame_count * verify_width, dense_vocab_size),
            target_previous_ids,
            target_current_ids,
        )

    update_dense_tables()

    def sample_proposals() -> None:
        sample_compact_probability_rows_fused(
            draft_probabilities,
            draft_token_ids,
            proposal_uniforms,
            active_proposals,
        )

    def compact_acceptance() -> SpeculativeAcceptanceBatch:
        return accept_speculative_candidates_compact(
            draft_tokens,
            draft_probabilities,
            draft_token_ids,
            target_probabilities,
            target_token_ids,
            proposal_counts,
            active_batch_rows,
            acceptance_uniforms,
            correction_uniforms,
            bonus_uniforms,
            outputs=compact_outputs,
        )

    def dense_acceptance() -> SpeculativeAcceptanceBatch:
        return accept_speculative_candidates_dense_indexed(
            draft_tokens,
            draft_probabilities,
            draft_token_ids,
            target_probabilities,
            target_token_ids,
            proposal_counts,
            active_batch_rows,
            acceptance_uniforms,
            correction_uniforms,
            bonus_uniforms,
            dense_draft,
            dense_target,
            candidate_offsets,
            outputs=dense_outputs,
        )

    def time_compact_acceptance() -> None:
        compact_acceptance()

    def time_dense_acceptance() -> None:
        dense_acceptance()

    compact_result = compact_acceptance()
    dense_result = dense_acceptance()
    for compact_output, dense_output in zip(
        (
            compact_result.committed_tokens,
            compact_result.accepted_draft_counts,
            compact_result.committed_token_counts,
            compact_result.rejected_rows,
            compact_result.bonus_rows,
            compact_result.invalid_rows,
        ),
        (
            dense_result.committed_tokens,
            dense_result.accepted_draft_counts,
            dense_result.committed_token_counts,
            dense_result.rejected_rows,
            dense_result.bonus_rows,
            dense_result.invalid_rows,
        ),
        strict=True,
    ):
        torch.testing.assert_close(compact_output, dense_output, atol=0, rtol=0)
    acceptance_rows = {
        "compact": {
            "median_cuda_ms": cuda_median_ms(
                time_compact_acceptance, warmup=warmup, iterations=iterations
            ),
            "rows": frame_count,
            "proposal_width": proposal_width,
        },
        "dense_indexed": {
            "table_update_median_cuda_ms": cuda_median_ms(
                update_dense_tables, warmup=warmup, iterations=iterations
            ),
            "median_cuda_ms": cuda_median_ms(
                time_dense_acceptance, warmup=warmup, iterations=iterations
            ),
            "rows": frame_count,
            "proposal_width": proposal_width,
        },
        "proposal_sampling": {
            "median_cuda_ms": cuda_median_ms(
                sample_proposals, warmup=warmup, iterations=iterations
            ),
            "rows": frame_count,
            "candidate_width": candidate_width,
        },
    }
    return {
        "schema": "qwen3-tts-speculative-kernel-microbench-v1",
        "model_snapshot": artifacts.model_snapshot,
        "dataset_sha256": artifacts.dataset_sha256,
        "capture_sha256": artifacts.capture_sha256,
        "device": torch.cuda.get_device_name(device),
        "frame_count": frame_count,
        "top_k": max_top_k,
        "grouped_lm_head": linear_rows,
        "probability_candidates": probability_rows,
        "sampling_and_acceptance": acceptance_rows,
        "warmup_iterations": warmup,
        "measurement_iterations": iterations,
    }

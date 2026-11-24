# SPDX-License-Identifier: Apache-2.0
"""CUDA full-frame and persistent-graph gates for Predictor speculation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.runtime_context import get_context
from torch import nn

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.models.qwen3_tts.sglang_model import (
    Qwen3TTSCodecTables,
    Qwen3TTSTalker,
)
from sglang_omni.vendor.sglang.layers import RMSNorm
from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
    DTYPE,
    FUSED_HIDDEN,
    TupleLinear,
    real_shape_layer,
    real_shape_talker,
)

pytestmark = pytest.mark.accelerator
CODEBOOK_VOCAB_SIZE = 2048


def identity_qk_norm(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    head_dim: int,
    alt_stream: torch.cuda.Stream | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(q_norm, nn.Module)
    assert isinstance(k_norm, nn.Module)
    assert head_dim > 0
    assert alt_stream is None or isinstance(alt_stream, torch.cuda.Stream)
    return q, k


def build_speculative_talker(
    device: torch.device, batch_size: int = 1
) -> Qwen3TTSTalker:
    layer_stack = [real_shape_layer(device) for _ in range(5)]
    final_norm = RMSNorm(FUSED_HIDDEN, eps=1e-6).to(device=device, dtype=DTYPE)
    talker = real_shape_talker(device, layer_stack, final_norm, fused=True)
    talker.model = SimpleNamespace(
        codec_embedding=SimpleNamespace(
            weight=torch.empty(1, device=device, dtype=DTYPE)
        )
    )
    talker.config = SimpleNamespace(
        num_code_groups=16,
        code_predictor_config=SimpleNamespace(vocab_size=CODEBOOK_VOCAB_SIZE),
    )
    talker.predictor_device = device
    talker.predictor_device_module = torch.get_device_module(device)
    talker.predictor_max_batch_size = batch_size
    talker.predictor_graph_batch_sizes = (batch_size,)
    talker.predictor_capture_stream = None
    talker.predictor_graph_pool = None
    talker.output_codes = torch.full(
        (batch_size, 16), -1, device=device, dtype=torch.long
    )
    talker.output_embeds = torch.zeros(
        (batch_size, FUSED_HIDDEN), device=device, dtype=DTYPE
    )
    talker.predictor_embedding_buffer = torch.empty(
        (batch_size, FUSED_HIDDEN), device=device, dtype=DTYPE
    )
    talker.predictor_projected_embeddings = None
    talker.predictor_projected_buffer = None
    talker.predictor_cutedsl_layers = None
    talker.sub_batch_size = batch_size
    talker.sub_temperature_tensor = torch.full((batch_size,), 0.9, device=device)
    talker.sub_top_p_tensor = torch.ones((batch_size,), device=device)
    talker.sub_top_k_tensor = torch.full(
        (batch_size,), 4, device=device, dtype=torch.long
    )
    talker.sub_sampling_seed_tensor = torch.full(
        (batch_size,), 1729, device=device, dtype=torch.long
    )
    talker.sub_speculative_seed_tensor = torch.full(
        (batch_size,), 8123, device=device, dtype=torch.long
    )
    talker.sub_seed_offsets = torch.arange(1, 16, device=device, dtype=torch.long)
    talker.sub_do_sample_tensor = torch.ones(
        (batch_size,), device=device, dtype=torch.bool
    )
    talker.sub_request_ids = tuple(
        f"unseeded-full-frame-test-{row}" for row in range(batch_size)
    )
    talker.sub_user_seed_rows = tuple(False for _ in range(batch_size))
    talker.sub_sample_rows = tuple(True for _ in range(batch_size))
    talker.sub_has_sampled_rows = True
    talker.sub_has_argmax_rows = False
    talker.sub_sampled_has_top_p = False
    talker.sub_sampled_max_top_k = 4
    talker.sub_sampled_has_unbounded_top_k = False
    talker.code_predictor_spec_requested_width = 0
    talker.code_predictor_spec_effective_width = 0
    talker.code_predictor_spec_backend = "cuda_graph"
    talker.code_predictor_spec_fallback_reason = None
    talker.code_predictor_spec_graphs = {}
    talker.code_predictor_spec_effective_width_histogram = {}
    talker.code_predictor_spec_graph_replay_count = 0
    talker.code_predictor_spec_graph_capture_failure_count = 0
    talker.code_predictor_spec_graph_memory_delta_bytes = 0
    talker.code_predictor_spec_graph_capture_time_ms = 0.0
    talker.code_predictor_spec_configuration_time_ms = 0.0
    talker.code_predictor_spec_frame_count = 0
    talker.code_predictor_spec_round_count = 0
    talker.code_predictor_spec_proposed_count = 0
    talker.code_predictor_spec_accepted_count = 0
    talker.code_predictor_spec_committed_count = 0
    talker.code_predictor_spec_bonus_count = 0
    talker.code_predictor_spec_rejection_count = 0
    talker.code_predictor_spec_packed_forward_count = 0
    talker.code_predictor_spec_packed_query_slots = 0
    talker.code_predictor_spec_padded_query_slots = 0
    talker.code_predictor_spec_fused_predictor_forward_count = 0
    talker.code_predictor_spec_draft_head_evaluation_count = 0
    talker.code_predictor_spec_target_head_evaluation_count = 0
    talker.code_predictor_spec_native_fallback_count = 0
    talker.code_predictor_spec_lm_head_weights = None
    talker.code_predictor_spec_metadata_device = None
    talker.code_predictor_spec_metadata_host = None

    codec_tables = Qwen3TTSCodecTables(15, CODEBOOK_VOCAB_SIZE, FUSED_HIDDEN).to(
        device=device, dtype=DTYPE
    )
    talker.code_predictor.model.codec_embedding = codec_tables
    talker.code_predictor.lm_head = nn.ModuleList(
        [
            TupleLinear(FUSED_HIDDEN, CODEBOOK_VOCAB_SIZE).to(
                device=device, dtype=DTYPE
            )
            for _ in range(15)
        ]
    )
    talker.code_predictor.project_input = nn.Linear(
        FUSED_HIDDEN, FUSED_HIDDEN, bias=False
    ).to(device=device, dtype=DTYPE)
    layer0_embedding = nn.Embedding(CODEBOOK_VOCAB_SIZE, FUSED_HIDDEN).to(
        device=device, dtype=DTYPE
    )

    def get_layer0_embeddings() -> nn.Embedding:
        return layer0_embedding

    talker.get_input_embeddings = get_layer0_embeddings
    talker.predictor_graphs = {}
    talker.predictor_graph_disabled = set()
    talker.predictor_graph_failure_count = 0
    talker.predictor_graph_capture_count = 0
    talker.predictor_graph_replay_count = 0
    talker.predictor_graph_startup_count = 0
    talker.predictor_graph_capacity_fallback_count = 0
    talker.predictor_graph_capacity_warned = False
    talker.predictor_eager_count = 0
    talker.predictor_graph_enabled = False
    talker.decode_prep_rids = None
    talker.decode_prep_epoch = 0
    return talker


def configure_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    execution_state = SimpleNamespace(
        graph=SimpleNamespace(disable_cuda_graph=False),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    monkeypatch.setattr(sglang_model_module, "get_exec", lambda: execution_state)
    monkeypatch.setattr(sglang_model_module, "QTTS_BACKEND", "default")
    monkeypatch.setattr(
        sglang_model_module,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )
    monkeypatch.setattr(
        sglang_model_module,
        "apply_qk_norm",
        identity_qk_norm,
    )


def configure_unseeded_request(talker: Qwen3TTSTalker, batch_size: int = 1) -> None:
    talker.sub_request_ids = tuple(
        f"unseeded-full-frame-test-{row}" for row in range(batch_size)
    )
    talker.sub_user_seed_rows = tuple(False for _ in range(batch_size))
    talker.sub_sample_rows = tuple(True for _ in range(batch_size))
    talker.sub_has_sampled_rows = True
    talker.sub_has_argmax_rows = False
    talker.sub_sampled_has_top_p = False
    talker.sub_sampled_max_top_k = 4
    talker.sub_sampled_has_unbounded_top_k = False


def verify_complete_frame(
    talker: Qwen3TTSTalker,
    codes: torch.Tensor,
    summed_embedding: torch.Tensor,
) -> None:
    assert codes.ndim == 3 and codes.shape[1:] == (16, 1)
    assert bool(((codes >= 0) & (codes < CODEBOOK_VOCAB_SIZE)).all())
    assert summed_embedding.shape == (codes.shape[0], 1, FUSED_HIDDEN)
    expected_embedding = talker.get_input_embeddings()(codes[:, :1, 0])[:, 0]
    codec_tables = talker.code_predictor.model.codec_embedding.weight
    for codebook_index in range(1, 16):
        expected_embedding.add_(
            codec_tables[codebook_index - 1, codes[:, codebook_index, 0]]
        )
    torch.testing.assert_close(
        summed_embedding[:, 0], expected_embedding, atol=0.0, rtol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_unseeded_eager_speculation_runs_a_complete_predictor_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(982)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(4, backend="eager")
        layer0_codes = torch.tensor([[9]], device=device, dtype=torch.long)
        talker_hidden = torch.randn((1, 1, FUSED_HIDDEN), device=device, dtype=DTYPE)
        semantic_positions = torch.tensor([7], device=device, dtype=torch.long)

        codes, summed_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, codes, summed_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == 1
    assert stats["native_fallback_count"] == 0
    assert stats["packed_transformer_forwards"] > 0
    assert stats["generic_predictor_forwards"] == 0
    assert stats["standalone_correction_forward_count"] == 0
    assert stats["committed_codes"] == 15
    assert stats["round_committed_codes"] == 14


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_speculative_verifier_graph_replays_the_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(618)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        talker.configure_code_predictor_speculation(4, backend="eager")
        layer0_codes = torch.tensor([[14]], device=device, dtype=torch.long)
        talker_hidden = torch.randn((1, 1, FUSED_HIDDEN), device=device, dtype=DTYPE)
        semantic_positions = torch.tensor([12], device=device, dtype=torch.long)
        eager_codes, eager_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        talker.configure_code_predictor_speculation(4, backend="cuda_graph")
        graph_codes, graph_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, graph_codes, graph_embedding)

    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embedding, eager_embedding)
    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == 2
    assert stats["native_fallback_count"] == 0
    assert stats["graph_captures"] > 0
    assert stats["graph_replays"] > 0
    assert stats["packed_transformer_forwards"] > 0
    assert stats["standalone_correction_forward_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_explicit_seed_fallback_does_not_block_later_unseeded_speculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(771)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        talker.configure_code_predictor_speculation(4, backend="eager")
        layer0_codes = torch.tensor([[3]], device=device, dtype=torch.long)
        talker_hidden = torch.randn((1, 1, FUSED_HIDDEN), device=device, dtype=DTYPE)
        semantic_positions = torch.tensor([5], device=device, dtype=torch.long)
        talker.sub_user_seed_rows = (True,)
        seeded_codes, seeded_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, seeded_codes, seeded_embedding)
        configure_unseeded_request(talker)
        unseeded_codes, unseeded_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, unseeded_codes, unseeded_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == 1
    assert stats["native_fallback_count"] == 1
    assert stats["native_fallback_fraction"] == 0.5
    assert stats["packed_transformer_forwards"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_mixed_explicit_seed_batch_falls_back_then_unseeded_batch_speculates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(913)
    configure_runtime_context(monkeypatch)
    batch_size = 2
    with get_context().override_server_args(max_running_requests=batch_size):
        talker = build_speculative_talker(device, batch_size)
        talker.configure_code_predictor_speculation(4, backend="eager")
        talker.sub_user_seed_rows = (False, True)
        layer0_codes = torch.tensor([[3], [14]], device=device, dtype=torch.long)
        talker_hidden = torch.randn(
            (batch_size, 1, FUSED_HIDDEN), device=device, dtype=DTYPE
        )
        semantic_positions = torch.tensor([5, 6], device=device, dtype=torch.long)

        mixed_codes, mixed_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, mixed_codes, mixed_embedding)
        fallback_stats = talker.predictor_speculative_stats()
        assert fallback_stats["frames"] == 0
        assert fallback_stats["native_fallback_count"] == batch_size
        assert fallback_stats["native_fallback_fraction"] == 1.0

        configure_unseeded_request(talker, batch_size)
        unseeded_codes, unseeded_embedding = talker.code_predictor_forward(
            layer0_codes, talker_hidden, semantic_positions
        )
        verify_complete_frame(talker, unseeded_codes, unseeded_embedding)

    stats = talker.predictor_speculative_stats()
    assert stats["frames"] == batch_size
    assert stats["native_fallback_count"] == batch_size
    assert stats["native_fallback_fraction"] == 0.5
    assert stats["packed_transformer_forwards"] > 0
    assert stats["standalone_correction_forward_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_speculative_frozen_frame_benchmark_requires_positive_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(2011)
    configure_runtime_context(monkeypatch)
    with get_context().override_server_args(max_running_requests=1):
        talker = build_speculative_talker(device)
        configure_unseeded_request(talker)
        layer0_codes = torch.tensor([[6]], device=device, dtype=torch.long)
        talker_hidden = torch.randn((1, 1, FUSED_HIDDEN), device=device, dtype=DTYPE)
        semantic_positions = torch.tensor([19], device=device, dtype=torch.long)

        def elapsed_milliseconds(frame_count: int) -> float:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(frame_count):
                talker.code_predictor_forward(
                    layer0_codes, talker_hidden, semantic_positions
                )
            end_event.record()
            torch.cuda.synchronize(device)
            return float(start_event.elapsed_time(end_event))

        talker.configure_code_predictor_speculation(4, backend="eager")
        elapsed_milliseconds(1)
        speculative_elapsed = elapsed_milliseconds(4)
        speculative_stats = talker.predictor_speculative_stats()
        if speculative_stats["frames"] <= 0:
            raise AssertionError("speculative benchmark executed zero frames")
        else:
            pass
        if speculative_stats["native_fallback_count"] != 0:
            raise AssertionError("speculative benchmark executed native fallback")
        else:
            pass

        talker.configure_code_predictor_speculation(0)
        elapsed_milliseconds(1)
        native_elapsed = elapsed_milliseconds(4)

    print(
        json.dumps(
            {
                "frames_per_backend": 4,
                "speculative_ms_per_frame": speculative_elapsed / 4,
                "native_ms_per_frame": native_elapsed / 4,
                "speculative": speculative_stats,
            },
            sort_keys=True,
        )
    )

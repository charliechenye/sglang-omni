# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks for the fixed P16 runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.speculative_execution import PredictorSpeculativeMixin
from sglang_omni.models.qwen3_tts.speculative_runtime import (
    is_fatal_cuda_capture_error,
    speculative_opening_code_output_view,
    speculative_sampling_fallback_reason,
)
from sglang_omni.utils.predictor_layers import (
    fused_predictor_scratch_rows,
    p16_fused_predictor_scratch_rows,
    resolve_fused_predictor_width,
)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_opening_pair_code_output_view_supports_batched_copies(
    batch_size: int,
) -> None:
    output_codes = torch.full((batch_size, 16), -1, dtype=torch.long)
    layer0_codes = torch.arange(10, 10 + batch_size, dtype=torch.long).unsqueeze(1)
    layer1_codes = torch.arange(30, 30 + batch_size, dtype=torch.long)
    pending_tokens = torch.full((batch_size,), -1, dtype=torch.long)

    code_outputs = speculative_opening_code_output_view(output_codes, batch_size)
    code_outputs[:, 0].copy_(layer0_codes[:, 0])
    code_outputs[:, 1].copy_(layer1_codes)
    pending_tokens.copy_(code_outputs[:, 1])

    assert code_outputs.shape == (batch_size, 16)
    assert code_outputs[:, 0].shape == (batch_size,)
    assert code_outputs[:, 1].shape == (batch_size,)
    assert pending_tokens.shape == (batch_size,)
    torch.testing.assert_close(code_outputs[:, 0], layer0_codes[:, 0])
    torch.testing.assert_close(code_outputs[:, 1], layer1_codes)
    torch.testing.assert_close(pending_tokens, layer1_codes)


def test_p16_sampling_eligibility_keeps_native_fallbacks() -> None:
    assert (
        speculative_sampling_fallback_reason(
            ("request",),
            (True,),
            has_unbounded_top_k=False,
        )
        is None
    )
    assert (
        speculative_sampling_fallback_reason(
            ("request-a", "request-b"),
            (True, False),
            has_unbounded_top_k=False,
        )
        == "greedy-or-mixed-sampling-requires-native-decoding"
    )
    assert (
        speculative_sampling_fallback_reason(
            ("request",),
            (True,),
            has_unbounded_top_k=True,
        )
        == "unsupported-sampling-configuration"
    )


@pytest.mark.parametrize("capacity", [1, 2, 4, 16, 40])
def test_p16_scratch_capacity_preserves_native_rows_and_supports_c1_w4(
    capacity: int,
) -> None:
    native_rows = fused_predictor_scratch_rows(capacity, 2)
    enabled_rows = p16_fused_predictor_scratch_rows(native_rows)

    assert enabled_rows == max(native_rows, 4)
    assert resolve_fused_predictor_width(4, 1, enabled_rows, 16, 8) == 4
    assert resolve_fused_predictor_width(2, capacity, enabled_rows, 16, 8) == min(
        2, enabled_rows // capacity
    )
    if capacity == 1:
        assert native_rows == 2
        assert enabled_rows == 4
    else:
        assert enabled_rows >= native_rows


def test_p16_scratch_resize_happens_before_predictor_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sglang_omni.models.qwen3_tts.speculative_execution as speculative_execution

    original = SimpleNamespace(shape=object(), max_rows=2)
    created: list[SimpleNamespace] = []

    def make_layers(shape, max_rows, device, dtype):
        del device, dtype
        value = SimpleNamespace(shape=shape, max_rows=max_rows)
        created.append(value)
        return value

    monkeypatch.setattr(speculative_execution, "FusedPredictorLayers", make_layers)
    talker = SimpleNamespace(
        predictor_fused_layers=original,
        predictor_graphs={},
        predictor_graph_capture_count=0,
        predictor_device=torch.device("cpu"),
        predictor_k_cache=torch.empty((), dtype=torch.bfloat16),
    )

    PredictorSpeculativeMixin.ensure_code_predictor_p16_fused_capacity(talker)

    assert created == [talker.predictor_fused_layers]
    assert talker.predictor_fused_layers.max_rows == 4
    assert talker.predictor_fused_layers.shape is original.shape

    talker.predictor_fused_layers = original
    talker.predictor_graphs = {"captured": object()}
    with pytest.raises(RuntimeError, match="after graph capture"):
        PredictorSpeculativeMixin.ensure_code_predictor_p16_fused_capacity(talker)


def p16_eligibility_talker(
    *, telemetry: object | None, enabled: bool = False
) -> object:
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=2048)
    )
    talker.model = SimpleNamespace(
        codec_embedding=SimpleNamespace(weight=torch.empty(1))
    )
    talker.sub_temperature_tensor = torch.empty(1, dtype=torch.float32)
    talker.sub_top_p_tensor = torch.empty(1, dtype=torch.float32)
    talker.sub_top_k_tensor = torch.empty(1, dtype=torch.long)
    talker.semantic_sampling_seed_tensor = torch.empty(1, dtype=torch.long)
    talker.sub_sampling_seed_tensor = torch.empty(1, dtype=torch.long)
    talker.sub_do_sample_tensor = torch.empty(1, dtype=torch.bool)
    talker.decode_prep_rids = None
    talker.decode_prep_epoch = 0
    talker.sub_seed_offsets = torch.arange(1, 16, dtype=torch.long)
    talker.code_predictor_spec_requested_width = 4 if enabled else 0
    talker.code_predictor_spec_effective_width = 0
    talker.code_predictor_p16_telemetry = telemetry
    talker.code_predictor_p16_enabled = False
    talker.code_predictor_p16_static_fallback_reason = None
    talker.code_predictor_p16_runtime_fallback_reason = lambda: None
    talker.predictor_graph_signature = lambda batch_size, positions: (  # noqa: ARG005
        "sampled",
        50,
        False,
        False,
        False,
    )
    return talker


def test_p16_off_telemetry_stages_metadata_for_feature_independent_eligibility() -> (
    None
):
    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    request = SimpleNamespace(
        request_id="off-telemetry",
        data=Qwen3TTSSGLangRequestData(
            semantic_sampling_seed=11,
            subtalker_dosample=True,
            subtalker_temperature=0.9,
            subtalker_top_p=1.0,
            subtalker_top_k=50,
            subtalker_sampling_seed=17,
        ),
    )
    off_with_telemetry = p16_eligibility_talker(telemetry=object())
    Qwen3TTSTalker.prepare_decode_buffers(off_with_telemetry, [request])
    layer0 = torch.zeros((1, 1), dtype=torch.long)
    hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
    assert (
        PredictorSpeculativeMixin.code_predictor_p16_frame_ineligibility_reason(
            off_with_telemetry, layer0, hidden, torch.zeros(1, dtype=torch.long)
        )
        is None
    )
    assert off_with_telemetry.sub_request_ids == ("off-telemetry",)
    assert off_with_telemetry.sub_sample_rows == (True,)
    assert off_with_telemetry.sub_sampling_parameters == ((0.9, 1.0, 50, True),)
    assert off_with_telemetry.code_predictor_spec_requested_width == 0
    assert off_with_telemetry.sub_speculative_seed_tensor is None

    off_without_telemetry = p16_eligibility_talker(telemetry=None)
    request_no_telemetry = SimpleNamespace(
        request_id="off-no-telemetry", data=request.data
    )
    Qwen3TTSTalker.prepare_decode_buffers(off_without_telemetry, [request_no_telemetry])
    assert off_without_telemetry.sub_request_ids == ()
    assert off_without_telemetry.sub_sample_rows == ()
    assert off_without_telemetry.sub_sampling_parameters == ()


def test_enabled_p16_eligibility_uses_the_same_staged_request_metadata() -> None:
    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = p16_eligibility_talker(telemetry=None, enabled=True)
    request = SimpleNamespace(
        request_id="on-eligible",
        data=Qwen3TTSSGLangRequestData(
            semantic_sampling_seed=11,
            subtalker_dosample=True,
            subtalker_temperature=0.9,
            subtalker_top_p=1.0,
            subtalker_top_k=50,
            subtalker_sampling_seed=17,
        ),
    )
    Qwen3TTSTalker.prepare_decode_buffers(talker, [request])
    candidate_width = 64
    workspace = SimpleNamespace(
        max_batch_size=1,
        max_verify_width=4,
        candidate_width=candidate_width,
    )
    talker.code_predictor_p16_enabled = True
    talker.code_predictor_p16_draft = object()
    talker.code_predictor_spec_lm_head_weights = torch.empty(1)
    talker.predictor_fused_layers = SimpleNamespace(max_rows=4)
    talker.sub_speculative_seed_tensor = torch.empty(1)
    talker.sub_speculative_uniforms_tensor = torch.empty(1)
    talker.sub_speculative_rng_pool_tensor = torch.empty(1)
    talker.code_predictor_spec_workspaces = {50: workspace}
    talker.code_predictor_spec_active_workspace = workspace
    talker.code_predictor_spec_effective_width = 4

    reason = PredictorSpeculativeMixin.code_predictor_p16_frame_ineligibility_reason(
        talker,
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1, 4), dtype=torch.float32),
        torch.zeros(1, dtype=torch.long),
    )

    assert reason is None
    assert talker.sub_request_ids == ("on-eligible",)
    assert talker.sub_sample_rows == (True,)
    assert talker.sub_sampling_parameters == ((0.9, 1.0, 50, True),)


def test_enabled_eligible_p16_frame_fails_if_resources_are_missing() -> None:
    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = p16_eligibility_talker(telemetry=None, enabled=True)
    request = SimpleNamespace(
        request_id="missing-resources",
        data=Qwen3TTSSGLangRequestData(
            semantic_sampling_seed=11,
            subtalker_dosample=True,
            subtalker_temperature=0.9,
            subtalker_top_p=1.0,
            subtalker_top_k=50,
            subtalker_sampling_seed=17,
        ),
    )
    Qwen3TTSTalker.prepare_decode_buffers(talker, [request])
    talker.code_predictor_p16_enabled = True
    layer0 = torch.zeros((1, 1), dtype=torch.long)
    hidden = torch.zeros((1, 1, 4), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="missing configured speculative resources"):
        PredictorSpeculativeMixin.code_predictor_p16_frame_ineligibility_reason(
            talker, layer0, hidden, torch.zeros(1, dtype=torch.long)
        )


@pytest.mark.parametrize(
    ("batch_size", "top_k", "expected_reason"),
    [(2, 50, "unsupported_batch"), (1, 49, "unsupported_sampling")],
)
def test_p16_eligibility_reports_staged_unsupported_request(
    batch_size: int, top_k: int, expected_reason: str
) -> None:
    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = p16_eligibility_talker(telemetry=object())
    talker.sub_temperature_tensor = torch.empty(batch_size, dtype=torch.float32)
    talker.sub_top_p_tensor = torch.empty(batch_size, dtype=torch.float32)
    talker.sub_top_k_tensor = torch.empty(batch_size, dtype=torch.long)
    talker.semantic_sampling_seed_tensor = torch.empty(batch_size, dtype=torch.long)
    talker.sub_sampling_seed_tensor = torch.empty(batch_size, dtype=torch.long)
    talker.sub_do_sample_tensor = torch.empty(batch_size, dtype=torch.bool)
    requests = [
        SimpleNamespace(
            request_id=f"unsupported-{batch_size}-{index}",
            data=Qwen3TTSSGLangRequestData(
                semantic_sampling_seed=11 + index,
                subtalker_dosample=True,
                subtalker_temperature=0.9,
                subtalker_top_p=1.0,
                subtalker_top_k=top_k,
                subtalker_sampling_seed=17 + index,
            ),
        )
        for index in range(batch_size)
    ]
    Qwen3TTSTalker.prepare_decode_buffers(talker, requests)

    reason = PredictorSpeculativeMixin.code_predictor_p16_frame_ineligibility_reason(
        talker,
        torch.zeros((batch_size, 1), dtype=torch.long),
        torch.zeros((batch_size, 1, 4), dtype=torch.float32),
        torch.zeros(batch_size, dtype=torch.long),
    )

    assert reason == expected_reason


@pytest.mark.parametrize(
    "message",
    ["CUDA error 719: unspecified launch failure", "device-side assert triggered"],
)
def test_fatal_cuda_capture_errors_are_not_retryable(message: str) -> None:
    assert is_fatal_cuda_capture_error(RuntimeError(message))


def test_ordinary_capture_error_is_not_fatal() -> None:
    assert not is_fatal_cuda_capture_error(RuntimeError("capture shape mismatch"))


def test_speculative_audit_observer_cannot_reuse_graphs_with_stale_buffers() -> None:
    talker = SimpleNamespace(
        code_predictor_spec_audit_observer=None,
        code_predictor_spec_effective_width=4,
        code_predictor_spec_round_graphs={"captured": object()},
        code_predictor_spec_opening_graphs={},
    )

    with pytest.raises(RuntimeError, match="before graph capture"):
        PredictorSpeculativeMixin.set_code_predictor_speculative_audit_observer(
            talker, lambda event, data: None
        )


def test_p16_terminal_cleanup_prevents_request_id_reuse_leaks() -> None:
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
    from sglang_omni.models.qwen3_tts.speculative_telemetry import P16GraphDiagnostics

    class Telemetry:
        def __init__(self) -> None:
            self.completed: list[str] = []
            self.discarded: list[str] = []
            self.graph_diagnostics: list[P16GraphDiagnostics | None] = []

        def complete_request(
            self,
            request_id: str,
            graph_diagnostics: P16GraphDiagnostics | None = None,
        ) -> None:
            self.completed.append(request_id)
            self.graph_diagnostics.append(graph_diagnostics)

        def discard_request(self, request_id: str) -> None:
            self.discarded.append(request_id)

    diagnostics: P16GraphDiagnostics = {
        "opening_pair_graph_captures": 0,
        "full_round_graph_captures": 0,
        "graph_captures": 0,
        "graph_capture_time_ms": 0.0,
        "graph_capture_memory_allocated_baseline_bytes": 0,
        "graph_capture_memory_allocated_current_bytes": 0,
        "graph_capture_memory_allocated_growth_bytes": 0,
        "graph_capture_memory_reserved_baseline_bytes": 0,
        "graph_capture_memory_reserved_current_bytes": 0,
        "graph_capture_memory_reserved_growth_bytes": 0,
        "graph_capture_memory_budget_bytes": 512 * 1024 * 1024,
        "graph_capture_memory_budget_exceeded": False,
    }
    telemetry = Telemetry()
    runner = SimpleNamespace(
        model=SimpleNamespace(
            code_predictor_p16_telemetry=telemetry,
            predictor_speculative_graph_diagnostics=lambda: diagnostics,
        ),
        pending_p16_frame_telemetry={"reused": "stale-frame"},
        pending_p16_finish_reasons={"reused": "abort"},
    )

    Qwen3TTSModelRunner.on_request_terminal(runner, "reused", succeeded=False)

    assert runner.pending_p16_frame_telemetry == {}
    assert runner.pending_p16_finish_reasons == {}
    assert telemetry.discarded == ["reused"]

    runner.pending_p16_frame_telemetry["reused"] = "new-frame"
    runner.pending_p16_finish_reasons["reused"] = None
    Qwen3TTSModelRunner.on_request_terminal(runner, "reused", succeeded=True)

    assert runner.pending_p16_frame_telemetry == {}
    assert runner.pending_p16_finish_reasons == {}
    assert telemetry.completed == ["reused"]
    assert telemetry.graph_diagnostics == [diagnostics]

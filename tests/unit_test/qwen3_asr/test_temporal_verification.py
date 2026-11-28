# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from array import array
from contextlib import nullcontext
from types import SimpleNamespace, TracebackType
from typing import Protocol

import numpy as np
import pytest
import sglang.srt.managers.schedule_batch as schedule_batch_module
import sglang.srt.managers.scheduler_components.batch_result_processor as batch_result_processor_module
import sglang.srt.model_executor.model_runner as native_model_runner_module
import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.cuda_graph_buffer_registry import CudaGraphBufferRegistry
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import (
    ModelRunnerOutput as SGLangModelRunnerOutput,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.sampling.sampling_params import SamplingParams

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.models.qwen3_asr.temporal_runner import Qwen3ASRTemporalModelRunner
from sglang_omni.models.qwen3_asr.temporal_scheduler import Qwen3ASRTemporalScheduler
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalBatchResult,
    Qwen3ASRTemporalForward,
    Qwen3ASRTemporalLogitsProcessor,
    Qwen3ASRTemporalLogitsProcessorOutput,
    attach_temporal_forward_inputs,
    clear_temporal_forward_inputs,
    verify_greedy_candidates,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.realtime import RealtimeTranscriptionContext
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class NoArgumentCallback(Protocol):
    def __call__(self) -> None: ...


def test_temporal_verification_commits_only_the_contiguous_target_prefix() -> None:
    """A mismatch must discard later guesses and return its corrective token."""
    cases = [
        ((21, 22), [21, 22, 2], (21, 22), 2, 14),
        ((21,), [21, 30], (21,), 30, 13),
        ((21, 22), [99, 22, 2], (), 99, 12),
        ((21, 22, 23, 24), [21, 22, 99, 24, 2], (21, 22), 99, 14),
        ((21, 22), [21, 2, 99], (21,), 2, 13),
        ((), [2], (), 2, 12),
    ]

    for (
        candidate_token_ids,
        target_token_ids,
        accepted_prefix,
        pending_token_id,
        boundary,
    ) in cases:
        verification = verify_greedy_candidates(
            request_id="final-3-1",
            row_index=0,
            prompt_boundary=12,
            candidate_token_ids=candidate_token_ids,
            target_token_ids=target_token_ids,
        )

        assert verification.accepted_prefix == accepted_prefix
        assert verification.pending_token_id == pending_token_id
        assert verification.owned_kv_boundary == boundary


@pytest.mark.parametrize(
    (
        "enabled",
        "is_final",
        "request_id",
        "max_new_tokens",
        "exclude_candidate_special",
        "expected_candidates",
    ),
    [
        (False, True, "final-3-1", 8, False, ()),
        (True, False, "final-3-1", 8, False, ()),
        (True, True, "stale-3-1", 8, False, ()),
        (True, True, "final-3-1", 1, False, ()),
        (True, True, "final-3-1", 2, False, (24,)),
        (True, True, "final-3-1", 8, False, (24, 25)),
        (True, True, "final-3-1", 8, True, ()),
    ],
)
def test_final_temporal_candidates_require_the_live_enabled_final_request(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    is_final: bool,
    request_id: str,
    max_new_tokens: int,
    exclude_candidate_special: bool,
    expected_candidates: tuple[int, ...],
) -> None:
    class TemporalTokenizer:
        eos_token_id = 2
        all_special_ids = [2, 42]

        def __len__(self) -> int:
            return 102

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|audio_pad|>"
            return 42

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            if text == "<asr_text>":
                return [100, 101]
            elif text == "abcdef":
                return [20, 21, 22, 23, 24, 25]
            else:
                raise AssertionError(f"unexpected tokenizer input {text!r}")

        def __call__(self, _text: str, *, add_special_tokens: bool = False):
            assert not add_special_tokens
            return SimpleNamespace(input_ids=[11, 42, 12, 13])

        def decode(self, token_ids: list[int], **_: object) -> str:
            return "".join(
                {20: "a", 21: "b", 22: "c", 23: "d", 24: "e", 25: "f"}[token_id]
                for token_id in token_ids
            )

    if exclude_candidate_special:
        TemporalTokenizer.all_special_ids = [2, 42, 24]
    else:
        pass

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=TemporalTokenizer(),
        max_new_tokens=max_new_tokens,
        feature_extractor=lambda *args, **kwargs: SimpleNamespace(
            input_features=torch.zeros((1, 128, 100)),
            attention_mask=torch.ones((1, 100), dtype=torch.long),
        ),
        enable_temporal_final_verification=enabled,
    )
    payload = StagePayload(
        request_id="final-3-1",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={
                "language": "English",
                "_asr_streaming": True,
                "_asr_streaming_prefix_text": "abcdef",
                "_asr_streaming_rollback_tokens": 2,
            },
            realtime_transcription_context=RealtimeTranscriptionContext(
                request_id=request_id,
                segment_id=3,
                attempt=1,
                is_final=is_final,
                previous_hypothesis="abcdef",
                language="English",
            ),
        ),
        data={},
    )

    request_data = request_builder(payload)

    if expected_candidates:
        assert request_data.temporal_request is not None
        assert request_data.temporal_request.candidate_token_ids == expected_candidates
        assert request_data.req.sampling_params.temperature == 1
        assert request_data.req.sampling_params.top_k == 1
    else:
        assert request_data.temporal_request is None
        if enabled and is_final and max_new_tokens == 1:
            assert request_data.temporal_fallback_reason == "output_budget_too_small"
        elif enabled and is_final and exclude_candidate_special:
            assert (
                request_data.temporal_fallback_reason
                == "invalid_temporal_candidate_ids"
            )
        elif enabled and is_final and request_id != "final-3-1":
            assert request_data.temporal_fallback_reason == "invalid_realtime_context"
        else:
            assert request_data.temporal_fallback_reason is None


def test_realtime_context_round_trips_and_rejects_malformed_wire_values() -> None:
    context = RealtimeTranscriptionContext(
        request_id="session-3-1",
        segment_id=3,
        attempt=1,
        is_final=True,
        previous_hypothesis="recognized words",
        language="English",
    )
    request = OmniRequest(
        inputs={"audio_bytes": b"wav"},
        realtime_transcription_context=context,
    )

    assert OmniRequest.from_dict(request.to_dict()) == request
    malformed = request.to_dict()
    malformed_context = malformed["realtime_transcription_context"]
    assert isinstance(malformed_context, dict)
    malformed_context["is_final"] = "true"
    assert OmniRequest.from_dict(malformed).realtime_transcription_context is None
    ordinary_request = OmniRequest(inputs={"audio_bytes": b"wav"})
    assert "realtime_transcription_context" not in ordinary_request.to_dict()


def test_temporal_logits_processor_uses_multimodal_forward_batch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = "final-3-1"
    candidate_token_ids = (20, 21)
    input_token_ids = torch.tensor([10, 11, 12, 20, 21], dtype=torch.long)
    positions = torch.arange(len(input_token_ids), dtype=torch.long)
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=input_token_ids,
        req_pool_indices=torch.tensor([0], dtype=torch.long),
        seq_lens=torch.tensor([5], dtype=torch.long),
        out_cache_loc=torch.arange(5, dtype=torch.long),
        seq_lens_sum=5,
    )
    forward_batch.rids = [request_id]
    forward_batch.positions = positions
    forward_batch.extend_seq_lens = torch.tensor([5], dtype=torch.int32)
    forward_batch.extend_seq_lens_cpu = [5]
    forward_batch.extend_prefix_lens_cpu = [0]
    forward_batch.capture_hidden_mode = CaptureHiddenMode.NULL
    forward_batch.input_embeds = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    hidden_states = torch.arange(5, dtype=torch.float32).reshape(5, 1)

    class RecordingLMHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.invocation_count = 0
            self.selected_rows: list[int] = []

        def forward(self, selected_hidden_states: torch.Tensor) -> torch.Tensor:
            self.invocation_count += 1
            self.selected_rows = [
                int(row[0]) for row in selected_hidden_states.tolist()
            ]
            logits = torch.full((len(self.selected_rows), 32), -100.0)
            prediction_by_input_row = {2: 20, 3: 21, 4: 2}
            for selected_row, input_row in enumerate(self.selected_rows):
                logits[selected_row, prediction_by_input_row[input_row]] = 100.0
            return logits

    native_processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(native_processor)
    native_processor.expected_aux_hidden_states = None
    native_processor.keep_last_hidden_states = False

    def run_lm_head(
        selected_hidden_states: torch.Tensor,
        lm_head: torch.nn.Module,
        logits_metadata: LogitsMetadata | ForwardBatch,
    ) -> torch.Tensor:
        assert isinstance(logits_metadata, (LogitsMetadata, ForwardBatch))
        return lm_head(selected_hidden_states)

    monkeypatch.setattr(
        native_processor,
        "_get_logits",
        run_lm_head,
    )
    temporal_processor = Qwen3ASRTemporalLogitsProcessor(native_processor)
    lm_head = RecordingLMHead()
    attach_temporal_forward_inputs(
        forward_batch,
        Qwen3ASRTemporalForward(
            request_id=request_id,
            prompt_boundary=3,
            candidate_token_ids=candidate_token_ids,
        ),
    )

    temporal_output = temporal_processor(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=forward_batch,
    )

    assert isinstance(temporal_output, Qwen3ASRTemporalLogitsProcessorOutput)
    assert temporal_output.temporal_verification is not None
    assert temporal_output.temporal_verification.accepted_prefix == candidate_token_ids
    assert temporal_output.temporal_verification.pending_token_id == 2
    assert temporal_output.pending_token_ids is not None
    assert temporal_output.pending_token_ids.tolist() == [2]
    assert temporal_output.next_token_logits is not None
    assert temporal_output.next_token_logits.shape == (1, 32)
    assert lm_head.invocation_count == 1
    assert lm_head.selected_rows == [2, 3, 4]

    clear_temporal_forward_inputs(forward_batch)
    lm_head.invocation_count = 0
    ordinary_output = temporal_processor(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=forward_batch,
    )

    assert not isinstance(ordinary_output, Qwen3ASRTemporalLogitsProcessorOutput)
    assert lm_head.invocation_count == 1
    assert lm_head.selected_rows == [4]


def test_temporal_prefill_graph_runner_preserves_ordinary_graph_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    from sglang_omni.models.qwen3_asr.temporal_prefill import (
        install_temporal_prefill_graph_gate,
    )

    runner = object.__new__(PrefillCudaGraphRunner)
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor([10], dtype=torch.long),
        req_pool_indices=torch.tensor([0], dtype=torch.long),
        seq_lens=torch.tensor([1], dtype=torch.long),
        out_cache_loc=torch.tensor([0], dtype=torch.long),
        seq_lens_sum=1,
    )
    native_graph_checks: list[ForwardBatch] = []

    def check_native_graph_eligibility(
        graph_runner: PrefillCudaGraphRunner,
        batch: ForwardBatch,
    ) -> bool:
        assert isinstance(graph_runner, PrefillCudaGraphRunner)
        native_graph_checks.append(batch)
        return True

    monkeypatch.setattr(
        PrefillCudaGraphRunner,
        "can_run_graph",
        check_native_graph_eligibility,
    )
    install_temporal_prefill_graph_gate(runner)

    assert runner.can_run_graph(forward_batch)
    assert native_graph_checks == [forward_batch]

    attach_temporal_forward_inputs(
        forward_batch,
        Qwen3ASRTemporalForward(
            request_id="final-3-1",
            prompt_boundary=1,
            candidate_token_ids=(20,),
        ),
    )
    assert not runner.can_run_graph(forward_batch)
    assert native_graph_checks == [forward_batch]
    clear_temporal_forward_inputs(forward_batch)
    assert runner.can_run_graph(forward_batch)
    assert native_graph_checks == [forward_batch, forward_batch]


def test_temporal_forward_survives_real_runner_dispatch_and_eager_batch_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eager ForwardBatch copy must not turn an admitted request into plain prefill."""
    import sglang_omni.models.qwen3_asr.engine_builder as engine_builder

    request_id = "final-3-1"
    candidate_token_ids = (20, 21)
    prompt_boundary = 3
    input_token_ids = torch.tensor([10, 11, 12, 20, 21], dtype=torch.long)
    native_graph_checks: list[ForwardBatch] = []
    graph_executions: list[ForwardBatch] = []
    eager_batches: list[ForwardBatch] = []
    published_next_token_ids: list[torch.Tensor | None] = []

    def make_forward_batch(token_ids: torch.Tensor) -> ForwardBatch:
        token_count = len(token_ids)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=token_ids,
            req_pool_indices=torch.tensor([0], dtype=torch.long),
            seq_lens=torch.tensor([token_count], dtype=torch.long),
            out_cache_loc=torch.arange(token_count, dtype=torch.long),
            seq_lens_sum=token_count,
        )
        forward_batch.rids = [request_id]
        forward_batch.positions = torch.arange(token_count, dtype=torch.long)
        forward_batch.extend_seq_lens = torch.tensor([token_count], dtype=torch.int32)
        forward_batch.extend_seq_lens_cpu = [token_count]
        forward_batch.extend_prefix_lens_cpu = [0]
        forward_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        forward_batch.input_embeds = torch.arange(
            token_count, dtype=torch.float32
        ).reshape(token_count, 1)
        return forward_batch

    native_processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(native_processor)
    native_processor.expected_aux_hidden_states = None
    native_processor.keep_last_hidden_states = False

    class RecordingLMHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.invocation_count = 0
            self.selected_rows: list[int] = []

        def forward(self, selected_hidden_states: torch.Tensor) -> torch.Tensor:
            self.invocation_count += 1
            self.selected_rows = [
                int(row[0]) for row in selected_hidden_states.tolist()
            ]
            logits = torch.full((len(self.selected_rows), 32), -100.0)
            prediction_by_input_row = {2: 20, 3: 21, 4: 2}
            for selected_row, input_row in enumerate(self.selected_rows):
                logits[selected_row, prediction_by_input_row.get(input_row, 2)] = 100.0
            return logits

    def run_lm_head(
        selected_hidden_states: torch.Tensor,
        lm_head: torch.nn.Module,
        logits_metadata: LogitsMetadata | ForwardBatch,
    ) -> torch.Tensor:
        assert isinstance(logits_metadata, (LogitsMetadata, ForwardBatch))
        return lm_head(selected_hidden_states)

    monkeypatch.setattr(native_processor, "_get_logits", run_lm_head)
    lm_head = RecordingLMHead()
    graph_runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)

    def native_can_run_graph(
        prefill_runner: PrefillCudaGraphRunner,
        forward_batch: ForwardBatch,
    ) -> bool:
        assert prefill_runner is graph_runner
        native_graph_checks.append(forward_batch)
        return True

    monkeypatch.setattr(PrefillCudaGraphRunner, "can_run_graph", native_can_run_graph)

    def execute_graph(forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        graph_executions.append(forward_batch)
        return LogitsProcessorOutput(next_token_logits=torch.zeros((1, 32)))

    graph_runner.execute = execute_graph
    sglang_runner = object.__new__(SGLModelRunner)
    sglang_runner.plain_decode_runner = None
    sglang_runner.model = SimpleNamespace(shared_experts_config=None)
    sglang_runner.device = "cpu"
    sglang_runner.decode_cuda_graph_runner = None
    sglang_runner.prefill_cuda_graph_runner = graph_runner
    sglang_runner.device_timer = None
    setattr(sglang_runner, "_prepare_eager_forward_batch", lambda _forward_batch: None)
    setattr(
        sglang_runner,
        "_maybe_execute_deferred_mamba_cow_and_clear",
        lambda _forward_batch: None,
    )
    setattr(sglang_runner, "_extend_forward_kwargs", lambda _forward_batch, _proxy: {})
    forward_raw = getattr(sglang_runner, "_forward_raw")  # noqa: leading-underscore

    monkeypatch.setattr(
        native_model_runner_module,
        "get_model",
        lambda: SimpleNamespace(override=lambda **_fields: nullcontext()),
    )
    monkeypatch.setattr(native_model_runner_module, "has_forward_context", lambda: True)
    monkeypatch.setattr(native_model_runner_module, "get_forward_context", lambda: None)
    monkeypatch.setattr(
        native_model_runner_module,
        "forward_context",
        lambda *_arguments, **_fields: nullcontext(),
    )
    monkeypatch.setattr(
        native_model_runner_module, "get_global_dwdp_manager", lambda: None
    )
    monkeypatch.setattr(
        native_model_runner_module,
        "_prefill_cuda_graph_allows_context_parallel",
        lambda *_arguments: True,
    )
    monkeypatch.setattr(
        native_model_runner_module,
        "device_timer_ctx",
        lambda *_arguments: nullcontext(),
    )

    eager_runner = object.__new__(EagerRunner)
    setattr(
        eager_runner,
        "_eager_registry",
        CudaGraphBufferRegistry(
            device=torch.device("cpu"),
            max_bs=1,
            max_num_tokens=len(input_token_ids),
        ),
    )
    temporal_processor = Qwen3ASRTemporalLogitsProcessor(native_processor)
    hidden_states = torch.arange(len(input_token_ids), dtype=torch.float32).reshape(
        len(input_token_ids), 1
    )

    class EagerExecution:
        def execute(
            self,
            forward_batch: ForwardBatch,
            pp_proxy_tensors: object | None = None,
        ) -> LogitsProcessorOutput:
            replay_batch = EagerRunner.load_batch(
                eager_runner,
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            eager_batches.append(replay_batch)
            return temporal_processor(
                input_ids=None,
                hidden_states=hidden_states,
                lm_head=lm_head,
                logits_metadata=replay_batch,
            )

    sglang_runner.eager_runner = EagerExecution()

    def native_forward(*, forward_batch: ForwardBatch) -> SGLangModelRunnerOutput:
        return forward_raw(forward_batch, pp_proxy_tensors=None)

    sglang_runner.forward = native_forward
    builder = object.__new__(engine_builder.Qwen3ASREngineBuilder)
    builder.enable_temporal_final_verification = True
    monkeypatch.setattr(builder, "uses_torch_mps", lambda: False)
    monkeypatch.setattr(engine_builder.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(engine_builder.current_platform, "is_xpu", lambda: False)
    model = SimpleNamespace(
        shared_experts_config=None,
        language_model=SimpleNamespace(logits_processor=native_processor),
    )
    sglang_runner.model = model
    builder.setup_model(
        model_worker=SimpleNamespace(model_runner=sglang_runner),
        checkpoint_dir="unused",
        device="cpu",
        gpu_id=0,
        server_args=SimpleNamespace(),
    )

    ordinary_before = forward_raw(
        make_forward_batch(torch.tensor([10], dtype=torch.long)),
        pp_proxy_tensors=None,
    )
    assert ordinary_before.can_run_graph
    assert len(graph_executions) == 1

    temporal_request = Qwen3ASRTemporalRequest(
        context=RealtimeTranscriptionContext(
            request_id=request_id,
            segment_id=3,
            attempt=1,
            is_final=True,
            previous_hypothesis="recognized words",
            language="English",
        ),
        candidate_token_ids=candidate_token_ids,
        prompt_boundary=prompt_boundary,
        admitted=True,
        attempted=True,
    )
    request_data = Qwen3ASRRequestData(temporal_request=temporal_request)
    scheduler_request = SchedulerRequest(request_id=request_id, data=request_data)
    schedule_batch = SimpleNamespace(
        reqs=[SimpleNamespace()],
        input_ids=input_token_ids,
        is_prefill_only=False,
    )
    forward_batch = make_forward_batch(input_token_ids)
    model_worker = object.__new__(ModelWorker)
    model_worker.dllm_algorithm = None
    model_worker.model_runner = sglang_runner
    model_worker.record_prefill_cuda_graph_usage = (
        lambda _forward_batch, *, can_run_graph: None
    )
    temporal_runner = object.__new__(Qwen3ASRTemporalModelRunner)
    temporal_runner.tp_worker = model_worker
    temporal_runner.retained_prefill_input_embeds = None
    temporal_runner.retain_prefill_input_embeds = False
    temporal_runner.execution_bridge = SimpleNamespace(
        publish_next_tokens=lambda _batch, token_ids: published_next_token_ids.append(
            token_ids
        )
    )
    temporal_runner.output_processor = SimpleNamespace(
        process=lambda _result, _scheduler_output: {
            request_id: RequestOutput(
                request_id=request_id,
                data="must wait for temporal precommit",
                extra={},
            )
        }
    )
    scheduler_output = SchedulerOutput(requests=[scheduler_request], batch_data=None)

    batch_result = temporal_runner.prepare_and_forward(
        forward_batch,
        schedule_batch,
        [scheduler_request],
        is_prefill=True,
    )

    assert not batch_result.can_run_cuda_graph
    assert isinstance(batch_result.logits_output, Qwen3ASRTemporalLogitsProcessorOutput)
    assert batch_result.logits_output.temporal_verification is not None
    assert batch_result.logits_output.temporal_verification.accepted_prefix == (
        candidate_token_ids
    )
    assert lm_head.invocation_count == 1
    assert lm_head.selected_rows == [2, 3, 4]
    assert len(eager_batches) == 1
    assert eager_batches[0] is not forward_batch
    assert eager_batches[0].input_ids is forward_batch.input_ids
    assert len(graph_executions) == 1
    assert get_temporal_forward_inputs(eager_batches[0]) is None

    temporal_runner.post_prefill(
        batch_result,
        forward_batch,
        schedule_batch,
        [scheduler_request],
    )
    assert batch_result.next_token_ids is not None
    assert batch_result.next_token_ids.tolist() == [2]
    temporal_runner.publish_next_tokens(
        batch_result,
        forward_batch,
        schedule_batch,
        [scheduler_request],
    )
    assert published_next_token_ids == [None]

    finalized_output = temporal_runner.finalize(
        batch_result,
        forward_batch,
        schedule_batch,
        scheduler_output,
    )
    assert finalized_output.outputs[request_id].data is None
    assert finalized_output.next_token_ids is not None
    assert finalized_output.next_token_ids.tolist() == [2]
    scheduler_result = Qwen3ASRTemporalScheduler.make_batch_result(finalized_output)
    assert isinstance(scheduler_result, Qwen3ASRTemporalBatchResult)
    assert scheduler_result.temporal_verification.request_id == request_id
    assert scheduler_result.native_result.next_token_ids is not None
    assert scheduler_result.native_result.next_token_ids.tolist() == [2]

    ordinary_after = forward_raw(
        make_forward_batch(torch.tensor([11], dtype=torch.long)),
        pp_proxy_tensors=None,
    )
    assert ordinary_after.can_run_graph
    assert len(graph_executions) == 2
    assert native_graph_checks[-1] is not forward_batch


def test_retracted_request_rebuilds_mrope_from_the_canonical_prompt() -> None:
    request_id = "final-3-1"
    prompt_token_ids = array("q", [10, 11, 12, 13])
    output_token_ids = array("q", [20, 21, 22, 23])
    temporal_request = Qwen3ASRTemporalRequest(
        context=RealtimeTranscriptionContext(
            request_id=request_id,
            segment_id=3,
            attempt=1,
            is_final=True,
            previous_hypothesis="recognized words",
            language="English",
        ),
        candidate_token_ids=(20, 21),
        prompt_boundary=len(prompt_token_ids),
        attempted=True,
    )
    request = Req(
        request_id,
        "",
        prompt_token_ids,
        SamplingParams(max_new_tokens=10, temperature=1, top_p=1, top_k=1),
        eos_token_ids={2},
        vocab_size=64,
    )
    request.output_ids = output_token_ids
    request.full_untruncated_fill_ids = prompt_token_ids + output_token_ids
    request.input_embeds = None
    request.multimodal_inputs = SimpleNamespace(
        mrope_positions=torch.arange(len(prompt_token_ids)).expand(3, -1).clone(),
        release_features=lambda: None,
    )
    request.kv.req_pool_idx = None
    request.kv.kv_allocated_len = 0
    request.kv.kv_committed_len = 0
    request.omni_data = Qwen3ASRRequestData(
        req=SimpleNamespace(), temporal_request=temporal_request
    )

    request.reset_for_retract()
    request.init_next_round_input(tree_cache=None)

    assert request.is_retracted
    assert not request.kv.holds_kv
    assert request.kv.kv_committed_len == 0
    assert request.origin_input_ids == prompt_token_ids
    assert request.output_ids == output_token_ids
    assert request.full_untruncated_fill_ids == prompt_token_ids + output_token_ids
    assert request.multimodal_inputs.mrope_positions.shape == (3, 8)
    assert torch.equal(
        request.multimodal_inputs.mrope_positions,
        torch.arange(8).expand(3, -1),
    )
    assert temporal_request.attempted
    assert not temporal_request.admitted


def test_abort_waits_until_temporal_batch_is_visible_to_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Req(
        "final-3-1",
        "",
        array("q", [10]),
        SamplingParams(max_new_tokens=3, temperature=1, top_p=1, top_k=1),
        eos_token_ids={2},
        vocab_size=64,
    )
    batch = ScheduleBatch(
        reqs=[request],
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor([1], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int64),
        orig_seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_sum=1,
    )
    running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    selection_started = threading.Event()
    finish_selection = threading.Event()
    abort_done = threading.Event()
    errors: list[BaseException] = []

    class RecordingAdmissionLock:
        def __init__(self) -> None:
            self.lock = threading.RLock()
            self.abort_attempted = threading.Event()
            self.abort_acquired = threading.Event()

        def __enter__(self) -> RecordingAdmissionLock:
            is_abort_thread = threading.current_thread().name == "qwen-abort"
            if is_abort_thread:
                self.abort_attempted.set()
                acquired = self.lock.acquire(blocking=False)
                if not acquired:
                    self.lock.acquire()
                else:
                    pass
                self.abort_acquired.set()
            else:
                self.lock.acquire()
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            self.lock.release()
            return False

    admission_lock = RecordingAdmissionLock()

    def build_native_prefill(
        _scheduler: OmniScheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        selection_started.set()
        if not finish_selection.wait(timeout=2):
            raise TimeoutError("test did not release native prefill build")
        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    monkeypatch.setattr(OmniScheduler, "build_new_batch_prefill", build_native_prefill)
    scheduler = object.__new__(Qwen3ASRTemporalScheduler)
    scheduler.request_admission_lock = admission_lock
    scheduler.cur_batch = None
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.enable_overlap = False
    scheduler.tp_size = 1
    scheduler.session_bridge = None
    scheduler.aborted_request_ids = set()
    scheduler.aborted_request_id_order = []
    scheduler.pending_request_builds = {}
    scheduler.early_request_builds = False
    scheduler.early_request_build_futures = {}
    scheduler.early_request_build_declined_ids = set()
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = []
    scheduler.waiting_queue = []
    scheduler.pending_stream_ingress = {}
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()

    def mark_running_request_aborted(request_id: str) -> bool:
        return scheduler.cur_batch is batch and batch.reqs[0].rid == request_id

    def mark_request_finished_immediately(
        _request_id: str, _batch_to_abort: ScheduleBatch | None = None
    ) -> list[Req]:
        return []

    def run_abort_callback(_request_id: str) -> None:
        return None

    def emit_model_path_end_once(_request_id: str, *, status: str) -> None:
        return None

    def drain_inbox_for_request(_request_id: str) -> None:
        return None

    scheduler.mark_running_request_aborted = mark_running_request_aborted
    scheduler.mark_request_finished_immediately = mark_request_finished_immediately
    scheduler.run_abort_callback = run_abort_callback
    scheduler.emit_model_path_end_once = emit_model_path_end_once
    scheduler.drain_inbox_for_request = drain_inbox_for_request

    def select_temporal_batch() -> None:
        try:
            scheduler.build_new_batch_prefill(running_batch)
        except BaseException as exc:
            errors.append(exc)

    def abort_request() -> None:
        try:
            scheduler.abort(request.rid)
        except BaseException as exc:
            errors.append(exc)
        finally:
            abort_done.set()

    selection_thread = threading.Thread(target=select_temporal_batch)
    selection_thread.start()
    assert selection_started.wait(timeout=2)
    abort_thread = threading.Thread(target=abort_request, name="qwen-abort")
    abort_thread.start()
    assert admission_lock.abort_attempted.wait(timeout=2)
    assert not admission_lock.abort_acquired.is_set()
    assert not abort_done.is_set()

    finish_selection.set()
    selection_thread.join(timeout=2)
    abort_thread.join(timeout=2)

    assert not selection_thread.is_alive()
    assert not abort_thread.is_alive()
    assert errors == []
    assert admission_lock.abort_acquired.is_set()
    assert scheduler.cur_batch is batch
    assert request.rid in scheduler.aborted_request_ids


@pytest.mark.parametrize("enabled", [False, True])
def test_inherited_build_runtime_uses_the_temporal_scheduler_contract(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    import sglang_omni.models.qwen3_asr.engine_builder as engine_builder
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    builder = object.__new__(engine_builder.Qwen3ASREngineBuilder)
    builder.enable_temporal_final_verification = enabled
    scheduler_kwargs_calls = 0
    scheduler_callbacks_calls = 0
    abort_callback = lambda: None
    request_finished_callback = lambda: None
    scheduler_callback = lambda: None
    scheduler_extras = {"extra_marker": "resolved-once"}
    scheduler_callbacks: dict[str, NoArgumentCallback] = {
        "callback_marker": scheduler_callback
    }
    captured_scheduler_arguments: list[dict[str, str | NoArgumentCallback | None]] = []

    def record_scheduler_init(
        scheduler: OmniScheduler,
        **arguments: str | NoArgumentCallback | None,
    ) -> None:
        captured_scheduler_arguments.append(arguments)

    monkeypatch.setattr(OmniScheduler, "__init__", record_scheduler_init)
    monkeypatch.setattr(
        Qwen3ASRTemporalScheduler,
        "__init__",
        record_scheduler_init,
    )
    monkeypatch.setattr(builder, "make_model_runner", lambda *_arguments: "runner")
    monkeypatch.setattr(builder, "make_adapters", lambda _model: ("builder", "adapter"))
    monkeypatch.setattr(builder, "make_abort_callback", lambda: abort_callback)
    monkeypatch.setattr(
        builder,
        "make_request_finished_callback",
        lambda: request_finished_callback,
    )

    def extra_scheduler_kwargs() -> dict[str, str]:
        nonlocal scheduler_kwargs_calls
        scheduler_kwargs_calls += 1
        return scheduler_extras

    def extra_scheduler_callbacks() -> dict[str, NoArgumentCallback]:
        nonlocal scheduler_callbacks_calls
        scheduler_callbacks_calls += 1
        return scheduler_callbacks

    monkeypatch.setattr(builder, "extra_scheduler_kwargs", extra_scheduler_kwargs)
    monkeypatch.setattr(builder, "extra_scheduler_callbacks", extra_scheduler_callbacks)

    scheduler, model_runner = builder.build_runtime(
        model_worker="worker",
        model="model",
        output_proc="output-processor",
        tree_cache="tree-cache",
        req_to_token_pool="request-pool",
        token_to_kv_pool_allocator="kv-allocator",
        server_args="server-args",
        model_config="model-config",
    )

    if enabled:
        assert isinstance(scheduler, Qwen3ASRTemporalScheduler)
        assert scheduler_callbacks_calls == 1
    else:
        assert type(scheduler) is OmniScheduler
        assert scheduler_callbacks_calls == 1
    assert model_runner == "runner"
    assert scheduler_kwargs_calls == 1
    assert len(captured_scheduler_arguments) == 1
    assert captured_scheduler_arguments[0] == {
        "tp_worker": "worker",
        "tree_cache": "tree-cache",
        "req_to_token_pool": "request-pool",
        "token_to_kv_pool_allocator": "kv-allocator",
        "server_args": "server-args",
        "model_config": "model-config",
        "model_runner": "runner",
        "request_builder": "builder",
        "result_adapter": "adapter",
        "abort_callback": abort_callback,
        "request_finished_callback": request_finished_callback,
        "callback_marker": scheduler_callback,
        "extra_marker": "resolved-once",
    }


def test_flag_off_setup_leaves_qwen_asr_components_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sglang_omni.models.qwen3_asr.engine_builder as engine_builder

    builder = object.__new__(engine_builder.Qwen3ASREngineBuilder)
    builder.enable_temporal_final_verification = False
    builder.uses_torch_mps = lambda: False
    logits_processor = SimpleNamespace(name="original-logits-processor")
    graph_runner_class = SimpleNamespace(name="original-graph-runner")
    model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            prefill_graph_runner_class=graph_runner_class,
            model=SimpleNamespace(
                language_model=SimpleNamespace(logits_processor=logits_processor)
            ),
        )
    )
    monkeypatch.setattr(engine_builder.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(engine_builder.current_platform, "is_xpu", lambda: False)

    builder.setup_model(
        model_worker=model_worker,
        checkpoint_dir="unused",
        device="cpu",
        gpu_id=0,
        server_args=SimpleNamespace(),
    )

    assert model_worker.model_runner.model.language_model.logits_processor is (
        logits_processor
    )
    assert model_worker.model_runner.prefill_graph_runner_class is graph_runner_class


@pytest.mark.parametrize(
    (
        "target_token_ids",
        "accepted_prefix",
        "pending_token_id",
        "cancel_before_commit",
    ),
    [
        ([20, 22, 23, 2], (20,), 22, False),
        ([30, 20, 21, 2], (), 30, False),
        ([20, 21, 22, 2], (20, 21, 22), 2, False),
        ([20, 21, 22, 2], (), 2, True),
    ],
)
def test_temporal_precommit_reclaims_only_the_rejected_suffix(
    monkeypatch: pytest.MonkeyPatch,
    target_token_ids: list[int],
    accepted_prefix: tuple[int, ...],
    pending_token_id: int,
    cancel_before_commit: bool,
) -> None:
    request_id = "final-3-1"
    prompt_token_ids = list(range(10))
    candidate_token_ids = (20, 21, 22)
    context = RealtimeTranscriptionContext(
        request_id=request_id,
        segment_id=3,
        attempt=1,
        is_final=True,
        previous_hypothesis="recognized words",
        language="English",
    )
    temporal_request = Qwen3ASRTemporalRequest(
        context=context,
        candidate_token_ids=candidate_token_ids,
        prompt_boundary=len(prompt_token_ids),
        admitted=True,
    )
    verification = verify_greedy_candidates(
        request_id=request_id,
        row_index=0,
        prompt_boundary=len(prompt_token_ids),
        candidate_token_ids=candidate_token_ids,
        target_token_ids=target_token_ids,
    )
    request = Req(
        request_id,
        "",
        array("q", prompt_token_ids),
        SamplingParams(
            max_new_tokens=8,
            temperature=1,
            top_p=1,
            top_k=1,
        ),
        eos_token_ids={2},
        vocab_size=64,
    )
    request.sampling_params.normalize(tokenizer=None)
    request.full_untruncated_fill_ids = array(
        "q", [*prompt_token_ids, *candidate_token_ids]
    )
    request.extend_end = len(prompt_token_ids) + len(candidate_token_ids)
    request.kv.req_pool_idx = 0
    request.kv.cache_protected_len = 0
    request.kv.cache_inserted_len = 0
    request.kv.kv_committed_len = len(request.full_untruncated_fill_ids)
    request.kv.kv_allocated_len = len(request.full_untruncated_fill_ids)
    request.multimodal_inputs = SimpleNamespace(
        mrope_positions=torch.arange(13).expand(3, -1).clone(),
        release_features=lambda: None,
    )
    if cancel_before_commit:
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        request.to_finish = FINISH_ABORT("cancelled before temporal commit")
    else:
        pass
    request_data = Qwen3ASRRequestData(
        req=request,
        temporal_request=temporal_request,
    )
    request.omni_data = request_data

    class RecordingAllocator:
        def __init__(self) -> None:
            self.freed: list[torch.Tensor] = []
            self.free_group_depth = 0

        def free(self, locations: torch.Tensor) -> None:
            self.freed.append(locations.clone())

        def free_group_begin(self) -> None:
            self.free_group_depth += 1

        def free_group_end(self) -> None:
            self.free_group_depth -= 1

    allocator = RecordingAllocator()
    token_row = torch.arange(1, 14, dtype=torch.long)
    scheduler = object.__new__(Qwen3ASRTemporalScheduler)
    scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=token_row.unsqueeze(0).clone()
    )
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.aborted_request_ids = set()
    schedule_batch = ScheduleBatch(
        reqs=[request],
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor([13], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([13], dtype=torch.int64),
        orig_seq_lens=torch.tensor([13], dtype=torch.int32),
        seq_lens_sum=13,
    )
    batch_result = Qwen3ASRTemporalBatchResult(
        native_result=GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.tensor([pending_token_id], dtype=torch.long),
        ),
        temporal_verification=verification,
    )
    committed_prefix = () if cancel_before_commit else accepted_prefix
    owned_boundary = len(prompt_token_ids) + len(committed_prefix)

    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    streamed_request_outputs: list[tuple[str, tuple[int, ...]]] = []

    def record_stream_output(
        scheduler: OmniScheduler,
        requests: list[Req],
        return_logprob: bool = False,
        skip_req: Req | None = None,
    ) -> None:
        assert isinstance(scheduler, OmniScheduler)
        assert not return_logprob
        streamed_request_outputs.extend(
            (request.rid, tuple(request.output_ids))
            for request in requests
            if request is not skip_req
        )

    monkeypatch.setattr(OmniScheduler, "stream_output", record_stream_output)
    monkeypatch.setattr(
        batch_result_processor_module,
        "get_memory",
        lambda: SimpleNamespace(enable_hisparse=False),
    )
    monkeypatch.setattr(
        batch_result_processor_module,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_decode_enable_offload_kvcache=False),
    )
    monkeypatch.setattr(
        batch_result_processor_module,
        "get_exec",
        lambda: SimpleNamespace(
            mamba=SimpleNamespace(
                enable_mamba_extra_buffer=False,
                enable_mamba_extra_buffer_lazy=False,
            )
        ),
    )
    monkeypatch.setattr(
        batch_result_processor_module,
        "get_observability",
        lambda: SimpleNamespace(enable_metrics=False),
    )
    monkeypatch.setattr(
        batch_result_processor_module,
        "checkpoint_kv_cache",
        lambda _request, _tree_cache: None,
    )
    released_requests: list[tuple[str, bool, bool]] = []

    def record_release(
        released_request: Req,
        tree_cache: SimpleNamespace,
        *,
        checkpoint: bool,
    ) -> None:
        assert tree_cache.page_size == 1
        assert released_request.skip_radix_cache_insert
        released_requests.append(
            (
                released_request.rid,
                checkpoint,
                released_request.skip_radix_cache_insert,
            )
        )

    monkeypatch.setattr(
        batch_result_processor_module, "release_kv_cache", record_release
    )
    monkeypatch.setattr(
        SchedulerBatchResultProcessor,
        "_get_prefill_hidden_capture_mode",
        staticmethod(lambda batch: CaptureHiddenMode.NULL),
    )
    tree_cache = SimpleNamespace(page_size=1, supports_mamba=lambda: False)

    def report_decode_stats(
        can_run_cuda_graph: bool,
        *,
        running_batch: ScheduleBatch,
        num_generated_tokens: int,
    ) -> None:
        assert not can_run_cuda_graph
        assert running_batch.forward_mode.is_decode()
        assert num_generated_tokens == 1
        return None

    def commit_decode(batch: ScheduleBatch) -> set[int]:
        assert len(batch.reqs) == 1
        return set()

    def abort_request(request: Req) -> None:
        assert request.rid == request_id

    metrics_reporter = SimpleNamespace(
        num_generated_tokens=0,
        forward_ct_decode=0,
        report_decode_stats=report_decode_stats,
    )
    result_processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(
            vocab_size=64,
            think_end_ids=None,
            self_opened_think_start_ids=None,
            is_encoder_decoder=False,
        ),
        token_to_kv_pool_allocator=allocator,
        tree_cache=tree_cache,
        hisparse_coordinator=None,
        req_to_token_pool=scheduler.req_to_token_pool,
        decode_offload_manager=None,
        metrics_collector=SimpleNamespace(),
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=SimpleNamespace(),
        logprob_result_processor=None,
        output_streamer=scheduler,
        beam_coordinator=SimpleNamespace(commit_decode=commit_decode),
        abort_request=abort_request,
    )
    schedule_batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    schedule_batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    schedule_batch.model_config = result_processor.model_config
    schedule_batch.device = "cpu"
    schedule_batch.req_pool_indices = torch.tensor([0], dtype=torch.int64)
    schedule_batch.req_pool_indices_cpu = torch.tensor([0], dtype=torch.int64)
    schedule_batch.req_to_token_pool = scheduler.req_to_token_pool
    schedule_batch.tree_cache = tree_cache
    schedule_batch.hisparse_coordinator = None
    schedule_batch.enable_overlap = False
    schedule_batch.decoding_reqs = []

    precommit_output_ids: list[tuple[int, ...]] = []
    precommit_fill_ids: list[tuple[int, ...]] = []
    precommit_kv_lengths: list[tuple[int, int]] = []
    precommit_sequence_lengths: list[tuple[int, int, int]] = []
    precommit_mrope_lengths: list[int] = []

    def process_native_result(
        native_scheduler: OmniScheduler,
        batch: ScheduleBatch,
        native_result: GenerationBatchResult,
    ) -> None:
        assert isinstance(native_scheduler, OmniScheduler)
        if batch.forward_mode.is_extend():
            precommit_output_ids.append(tuple(request.output_ids))
            precommit_fill_ids.append(tuple(request.full_untruncated_fill_ids))
            precommit_kv_lengths.append(
                (request.kv.kv_committed_len, request.kv.kv_allocated_len)
            )
            precommit_sequence_lengths.append(
                (
                    int(batch.seq_lens[0]),
                    int(batch.seq_lens_cpu[0]),
                    int(batch.orig_seq_lens[0]),
                )
            )
            precommit_mrope_lengths.append(
                request.multimodal_inputs.mrope_positions.shape[1]
            )
            result_processor.process_batch_result_prefill(batch, native_result)
        else:
            result_processor.process_batch_result_decode(batch, native_result)

    monkeypatch.setattr(OmniScheduler, "process_batch_result", process_native_result)
    scheduler.model_config = result_processor.model_config
    scheduler.request_admission_lock = threading.RLock()
    scheduler.model_runner = SimpleNamespace(
        execution_bridge=SimpleNamespace(
            publish_next_tokens=lambda batch, token_ids: published_next_tokens.append(
                token_ids.tolist()
            )
        )
    )
    published_next_tokens: list[list[int]] = []
    emitted_output_ids: list[tuple[int, ...]] = []
    failure_errors: list[Exception] = []

    def record_emitted_tokens(
        emitted_request: Req,
        emitted_request_data: Qwen3ASRRequestData,
    ) -> None:
        assert emitted_request_data.temporal_request is temporal_request
        emitted_output_ids.append(tuple(emitted_request.output_ids_through_stop))

    def record_batch_failure(batch: ScheduleBatch, error: Exception) -> None:
        assert batch is schedule_batch
        failure_errors.append(error)
        raise error

    monkeypatch.setattr(scheduler, "emit_speculative_tokens", record_emitted_tokens)
    monkeypatch.setattr(scheduler, "handle_batch_failure", record_batch_failure)
    scheduler.process_batch_result(schedule_batch, batch_result)
    assert failure_errors == []
    assert precommit_output_ids == [tuple(committed_prefix)]
    assert precommit_fill_ids == [tuple([*prompt_token_ids, *committed_prefix])]
    assert precommit_kv_lengths == [(owned_boundary, owned_boundary)]
    assert precommit_sequence_lengths == [
        (owned_boundary, owned_boundary, owned_boundary)
    ]
    assert precommit_mrope_lengths == [len(prompt_token_ids)]
    assert torch.equal(
        request.multimodal_inputs.mrope_positions,
        torch.arange(len(prompt_token_ids)).expand(3, -1),
    )
    assert schedule_batch.seq_lens_sum is None
    assert scheduler.req_to_token_pool.req_to_token[0, owned_boundary:].tolist() == [
        0
    ] * (len(candidate_token_ids) - len(accepted_prefix))
    if len(candidate_token_ids) == len(accepted_prefix):
        assert allocator.freed == []
    else:
        assert len(allocator.freed) == 1
        assert allocator.freed[0].numel() == len(candidate_token_ids) - len(
            accepted_prefix
        )
    assert batch_result.native_result.next_token_ids.tolist() == [pending_token_id]
    assert temporal_request.cancelled is cancel_before_commit
    assert temporal_request.temporal_commit_in_progress is False

    if cancel_before_commit:
        assert request.output_ids == array("q")
        assert request.finished()
        assert emitted_output_ids == []
        assert published_next_tokens == []
        assert streamed_request_outputs == [(request_id, ())]
        assert released_requests == [(request_id, False, True)]
        assert len(allocator.freed) == 1
        assert allocator.freed[0].numel() == len(candidate_token_ids)
        assert allocator.free_group_depth == 0
        return
    else:
        assert list(request.output_ids) == [*accepted_prefix, pending_token_id]

    if request.finished():
        assert pending_token_id == 2
        assert emitted_output_ids
        assert published_next_tokens == []
        assert streamed_request_outputs == [(request_id, tuple(request.output_ids))]
        assert released_requests == [(request_id, True, True)]
        assert temporal_request.zero_decode
        assert temporal_request.ordinary_decode_rounds == 0
        assert allocator.free_group_depth == 0
        return
    else:
        assert streamed_request_outputs == []
        assert emitted_output_ids == [tuple([*accepted_prefix, pending_token_id])]
        assert published_next_tokens == [[pending_token_id]]

    owned_boundary = len(prompt_token_ids) + len(accepted_prefix)
    allocated_decode_rows: list[tuple[int, int]] = []

    def allocate_decode_row(
        batch: ScheduleBatch, token_per_req: int = 1
    ) -> torch.Tensor:
        assert token_per_req == 1
        assert batch.seq_lens_cpu.tolist() == [owned_boundary]
        allocated_decode_rows.append((int(batch.seq_lens_cpu[0]), token_per_req))
        location = torch.tensor([101], dtype=torch.int64)
        batch.req_to_token_pool.req_to_token[0, owned_boundary] = location[0]
        for batch_request in batch.reqs:
            batch_request.kv.kv_allocated_len += token_per_req
            batch_request.kv.kv_committed_len += token_per_req
        return location

    monkeypatch.setattr(schedule_batch_module, "alloc_for_decode", allocate_decode_row)
    monkeypatch.setattr(
        schedule_batch_module,
        "get_exec",
        lambda: SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer=False)),
    )
    schedule_batch.prepare_for_decode()
    assert allocated_decode_rows == [(owned_boundary, 1)]
    assert schedule_batch.seq_lens.tolist() == [owned_boundary + 1]
    assert schedule_batch.seq_lens_cpu.tolist() == [owned_boundary + 1]
    assert schedule_batch.orig_seq_lens.tolist() == [owned_boundary + 1]
    assert schedule_batch.seq_lens_sum is None
    assert request.multimodal_inputs.mrope_positions.shape[1] == len(prompt_token_ids)
    assert request.kv.kv_allocated_len == owned_boundary + 1
    assert request.kv.kv_committed_len == owned_boundary + 1

    decode_result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=torch.zeros((1, 64))),
        next_token_ids=torch.tensor([2], dtype=torch.long),
    )
    scheduler.process_batch_result(schedule_batch, decode_result)
    assert request.finished()
    assert list(request.output_ids) == [*accepted_prefix, pending_token_id, 2]
    assert released_requests == [(request_id, True, True)]
    assert streamed_request_outputs == [(request_id, tuple(request.output_ids))]
    assert temporal_request.ordinary_decode_rounds == 1
    assert allocator.free_group_depth == 0

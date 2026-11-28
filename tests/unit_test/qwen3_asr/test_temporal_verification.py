# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from array import array
from types import SimpleNamespace, TracebackType

import numpy as np
import pytest
import sglang.srt.managers.schedule_batch as schedule_batch_module
import sglang.srt.managers.scheduler_components.batch_result_processor as batch_result_processor_module
import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.sampling.sampling_params import SamplingParams

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.qwen3_asr.config import Qwen3ASRFactoryArgs
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    Qwen3ASRTemporalRequest,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.models.qwen3_asr.temporal_cuda_graph_runner import (
    Qwen3ASRTemporalPrefillCudaGraphRunner,
)
from sglang_omni.models.qwen3_asr.temporal_runner import Qwen3ASRTemporalModelRunner
from sglang_omni.models.qwen3_asr.temporal_scheduler import Qwen3ASRTemporalScheduler
from sglang_omni.models.qwen3_asr.temporal_verification import (
    Qwen3ASRTemporalBatchResult,
    Qwen3ASRTemporalLogitsProcessor,
    Qwen3ASRTemporalLogitsProcessorOutput,
    Qwen3ASRTemporalVerification,
    attach_temporal_forward_inputs,
    clear_temporal_forward_inputs,
    get_temporal_forward_context,
    verify_greedy_candidates,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.realtime import RealtimeTranscriptionContext
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def make_forward_batch(
    request_id: str,
    input_ids: list[int],
    graph_bucket_size: int | None = None,
) -> ForwardBatch:
    raw_tokens = len(input_ids)
    bucket_size = graph_bucket_size or raw_tokens
    tokens = torch.full((bucket_size,), 997, dtype=torch.long)
    tokens[:raw_tokens] = torch.tensor(input_ids, dtype=torch.long)
    positions = torch.full((bucket_size,), 997, dtype=torch.long)
    positions[:raw_tokens] = torch.arange(raw_tokens, dtype=torch.long)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=tokens,
        req_pool_indices=torch.tensor([0], dtype=torch.long),
        seq_lens=torch.tensor([raw_tokens], dtype=torch.long),
        out_cache_loc=torch.arange(raw_tokens, dtype=torch.long),
        seq_lens_sum=raw_tokens,
        extend_seq_lens=torch.tensor([raw_tokens], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        extend_start_loc=torch.tensor([0], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([raw_tokens], dtype=torch.long),
        extend_seq_lens_cpu=[raw_tokens],
        extend_prefix_lens_cpu=[0],
        positions=positions,
        mrope_positions=positions.unsqueeze(0).expand(3, -1).clone(),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    batch.rids = [request_id]
    return batch


def make_logits_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Qwen3ASRTemporalLogitsProcessor, torch.nn.Module, list[list[int]]]:
    selected_rows: list[list[int]] = []

    class RecordingLMHead(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            rows = [int(row[0]) for row in hidden_states.tolist()]
            selected_rows.append(rows)
            logits = torch.full((len(rows), 32), -100.0)
            for index, row in enumerate(rows):
                token = {2: 20, 3: 9, 4: 2}.get(row, 2)
                logits[index, token] = 100.0
            return logits

    native = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(native)
    native.expected_aux_hidden_states = None
    native.keep_last_hidden_states = False
    monkeypatch.setattr(
        native,
        "_get_logits",
        lambda hidden, lm_head, metadata: lm_head(hidden),
    )
    return Qwen3ASRTemporalLogitsProcessor(native), RecordingLMHead(), selected_rows


@pytest.mark.parametrize(
    ("candidates", "targets", "accepted", "pending"),
    [
        ((20, 21), [20, 21, 2], (20, 21), 2),
        ((20, 21), [20, 9, 2], (20,), 9),
        ((20, 21), [9, 21, 2], (), 9),
    ],
)
def test_greedy_verification_accepts_only_the_contiguous_prefix(
    candidates: tuple[int, ...],
    targets: list[int],
    accepted: tuple[int, ...],
    pending: int,
) -> None:
    result = verify_greedy_candidates("final-1", candidates, targets)
    assert result.request_id == "final-1"
    assert result.accepted_prefix == accepted
    assert result.pending_token_id == pending


@pytest.mark.parametrize("bucket_size", [None, 8])
def test_logits_verification_uses_only_k_plus_one_logical_rows(
    monkeypatch: pytest.MonkeyPatch, bucket_size: int | None
) -> None:
    request_id = "final-1"
    prompt_len = 3
    candidates = (20, 21)
    batch = make_forward_batch(request_id, [10, 11, 12, *candidates], bucket_size)
    processor, lm_head, selected_rows = make_logits_processor(monkeypatch)
    context = attach_temporal_forward_inputs(request_id, prompt_len, candidates)
    if bucket_size is not None:
        context.graph_bucket_size = bucket_size
    else:
        pass
    try:
        output = processor(
            input_ids=None,
            hidden_states=torch.arange(len(batch.input_ids), dtype=torch.float32).view(
                -1, 1
            ),
            lm_head=lm_head,
            logits_metadata=batch,
        )
    finally:
        clear_temporal_forward_inputs(context)

    assert isinstance(output, Qwen3ASRTemporalLogitsProcessorOutput)
    assert output.temporal_verification is not None
    assert output.temporal_verification.accepted_prefix == (20,)
    assert output.temporal_verification.pending_token_id == 9
    assert output.pending_token_ids is not None
    assert output.pending_token_ids.tolist() == [9]
    assert output.graph_bucket_size == bucket_size
    assert selected_rows == [[2, 3, 4]]


def test_context_survives_forward_batch_reconstruction_and_failure_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = "final-1"
    original = make_forward_batch(request_id, [10, 11, 12, 20, 21])
    reconstructed = make_forward_batch(request_id, [10, 11, 12, 20, 21])
    processor, lm_head, _ = make_logits_processor(monkeypatch)
    context = attach_temporal_forward_inputs(request_id, 3, (20, 21))
    runner = object.__new__(Qwen3ASRTemporalModelRunner)
    runner.temporal_context = context
    try:
        assert get_temporal_forward_context(reconstructed) is context
        other_request = make_forward_batch("other", [10, 11, 12, 20, 21])
        with pytest.raises(RuntimeError, match="another request"):
            get_temporal_forward_context(other_request)
        reconstructed.forward_mode = ForwardMode.DECODE
        with pytest.raises(RuntimeError, match="only supports prefill"):
            processor(
                input_ids=None,
                hidden_states=torch.zeros((5, 1)),
                lm_head=lm_head,
                logits_metadata=reconstructed,
            )
    finally:
        runner.cleanup_prefill(original, None, [])

    assert get_temporal_forward_context(original) is None
    assert context.context_token is None


def test_graph_trim_preserves_temporal_result_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification = Qwen3ASRTemporalVerification(
        request_id="final-1", accepted_prefix=(20,), pending_token_id=9
    )
    output = Qwen3ASRTemporalLogitsProcessorOutput(
        next_token_logits=torch.ones((3, 32)),
        temporal_verification=verification,
        pending_token_ids=torch.tensor([9]),
        graph_bucket_size=8,
    )

    def native_trim(
        _runner: PrefillCudaGraphRunner, _output: LogitsProcessorOutput
    ) -> LogitsProcessorOutput:
        return LogitsProcessorOutput(next_token_logits=torch.ones((1, 32)))

    monkeypatch.setattr(PrefillCudaGraphRunner, "_trim_logits_output", native_trim)
    runner = object.__new__(Qwen3ASRTemporalPrefillCudaGraphRunner)
    trim_logits_output = getattr(
        runner, "_trim_logits_output"
    )  # noqa: leading-underscore  # pinned SGLang trim hook
    trimmed = trim_logits_output(output)

    assert isinstance(trimmed, Qwen3ASRTemporalLogitsProcessorOutput)
    assert trimmed.next_token_logits.shape == (1, 32)
    assert trimmed.temporal_verification is verification
    assert trimmed.pending_token_ids.tolist() == [9]
    assert trimmed.graph_bucket_size == 8


@pytest.mark.parametrize(
    (
        "enabled",
        "is_final",
        "context_id",
        "max_tokens",
        "exclude_special_candidate",
        "expected",
    ),
    [
        (False, True, "final-1", 8, False, ()),
        (True, False, "final-1", 8, False, ()),
        (True, True, "other", 8, False, ()),
        (True, True, "final-1", 1, False, ()),
        (True, True, "final-1", 2, False, (24,)),
        (True, True, "final-1", 8, False, (24, 25)),
        (True, True, "final-1", 8, True, ()),
    ],
)
def test_only_enabled_matching_final_requests_get_proposals(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    is_final: bool,
    context_id: str,
    max_tokens: int,
    exclude_special_candidate: bool,
    expected: tuple[int, ...],
) -> None:
    class Tokenizer:
        eos_token_id = 2
        all_special_ids = [2, 42, 24] if exclude_special_candidate else [2, 42]

        def __len__(self) -> int:
            return 102

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|audio_pad|>"
            return 42

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [100, 101] if text == "<asr_text>" else [20, 21, 22, 23, 24, 25]

        def __call__(self, _text: str, *, add_special_tokens: bool = False):
            assert not add_special_tokens
            return SimpleNamespace(input_ids=[11, 42, 12, 13])

        def decode(self, token_ids: list[int], **_: object) -> str:
            return "".join("abcdef"[token_id - 20] for token_id in token_ids)

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda _source, **_kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=Tokenizer(),
        max_new_tokens=max_tokens,
        feature_extractor=lambda *_args, **_kwargs: SimpleNamespace(
            input_features=torch.zeros((1, 128, 100)),
            attention_mask=torch.ones((1, 100), dtype=torch.long),
        ),
        enable_temporal_final_verification=enabled,
    )
    payload = StagePayload(
        request_id="final-1",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={
                "language": "English",
                "_asr_streaming": True,
                "_asr_streaming_prefix_text": "abcdef",
                "_asr_streaming_rollback_tokens": 2,
            },
            realtime_transcription_context=RealtimeTranscriptionContext(
                request_id=context_id,
                is_final=is_final,
                previous_hypothesis="abcdef",
                language="English",
            ),
        ),
        data={},
    )

    request_data = request_builder(payload)
    temporal = request_data.temporal_request
    assert (() if temporal is None else temporal.candidate_token_ids) == expected
    assert request_data.req.sampling_params.temperature == 1


def test_realtime_context_round_trips_inside_the_internal_request() -> None:
    context = RealtimeTranscriptionContext(
        request_id="final-1",
        is_final=True,
        previous_hypothesis="recognized words",
        language="English",
    )
    request = OmniRequest(
        inputs={"audio_bytes": b"wav"}, realtime_transcription_context=context
    )
    assert OmniRequest.from_dict(request.to_dict()) == request
    ordinary = OmniRequest(inputs={"audio_bytes": b"wav"})
    assert "realtime_transcription_context" not in ordinary.to_dict()


def test_feature_option_defaults_off() -> None:
    assert Qwen3ASRFactoryArgs().enable_temporal_final_verification is False


@pytest.mark.parametrize("graph_bucket_size", [None, 8])
def test_prefill_handles_native_eager_and_graph_results(
    graph_bucket_size: int | None,
) -> None:
    request_id = "final-1"
    temporal = Qwen3ASRTemporalRequest(
        candidate_token_ids=(20, 21), prompt_boundary=3, admitted=True
    )
    data = Qwen3ASRRequestData(req=SimpleNamespace(), temporal_request=temporal)
    verification = Qwen3ASRTemporalVerification(
        request_id=request_id, accepted_prefix=(20,), pending_token_id=9
    )
    logits_output = Qwen3ASRTemporalLogitsProcessorOutput(
        next_token_logits=torch.ones((1, 32)),
        temporal_verification=verification,
        pending_token_ids=torch.tensor([9]),
        graph_bucket_size=graph_bucket_size,
    )
    result = GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=None,
        can_run_cuda_graph=graph_bucket_size is not None,
    )
    runner = object.__new__(Qwen3ASRTemporalModelRunner)
    runner.post_prefill(
        result,
        None,
        SimpleNamespace(),
        [SimpleNamespace(request_id=request_id, data=data)],
    )
    assert result.next_token_ids.tolist() == [9]


def test_retracted_request_rebuilds_mrope_from_prompt_and_outputs() -> None:
    request_id = "final-1"
    prompt = array("q", [10, 11, 12, 13])
    outputs = array("q", [20, 21, 22, 23])
    temporal = Qwen3ASRTemporalRequest(
        candidate_token_ids=(20, 21), prompt_boundary=len(prompt)
    )
    request = Req(
        request_id,
        "",
        prompt,
        SamplingParams(max_new_tokens=8, temperature=1, top_p=1, top_k=1),
        eos_token_ids={2},
        vocab_size=64,
    )
    request.output_ids = outputs
    request.full_untruncated_fill_ids = prompt + outputs
    request.input_embeds = None
    request.multimodal_inputs = SimpleNamespace(
        mrope_positions=torch.arange(len(prompt)).expand(3, -1).clone(),
        release_features=lambda: None,
    )
    request.kv.req_pool_idx = None
    request.kv.kv_allocated_len = 0
    request.kv.kv_committed_len = 0
    request.omni_data = Qwen3ASRRequestData(
        req=SimpleNamespace(), temporal_request=temporal
    )

    request.reset_for_retract()
    request.init_next_round_input(tree_cache=None)

    assert request.is_retracted
    assert not request.kv.holds_kv
    assert request.kv.kv_committed_len == 0
    assert request.origin_input_ids == prompt
    assert request.output_ids == outputs
    assert request.full_untruncated_fill_ids == prompt + outputs
    assert request.multimodal_inputs.mrope_positions.shape == (3, 8)
    assert torch.equal(
        request.multimodal_inputs.mrope_positions,
        torch.arange(8).expand(3, -1),
    )
    assert not temporal.admitted


def test_abort_waits_until_temporal_batch_is_visible_to_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Req(
        "final-1",
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
        _scheduler: OmniScheduler, current_batch: ScheduleBatch
    ) -> NextBatchPlan:
        selection_started.set()
        if not finish_selection.wait(timeout=2):
            raise TimeoutError("test did not release native prefill build")
        else:
            return NextBatchPlan(batch_to_run=batch, running_batch=current_batch)

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
    scheduler.mark_running_request_aborted = lambda rid: (
        scheduler.cur_batch is batch and batch.reqs[0].rid == rid
    )
    scheduler.mark_request_finished_immediately = lambda *_args: []
    scheduler.run_abort_callback = lambda _rid: None
    scheduler.emit_model_path_end_once = lambda *_args, **_kwargs: None
    scheduler.drain_inbox_for_request = lambda _rid: None

    def select_batch() -> None:
        try:
            scheduler.build_new_batch_prefill(running_batch)
        except BaseException as error:
            errors.append(error)

    def abort_request() -> None:
        try:
            scheduler.abort(request.rid)
        except BaseException as error:
            errors.append(error)
        finally:
            abort_done.set()

    selection_thread = threading.Thread(target=select_batch)
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
    temporal_request = Qwen3ASRTemporalRequest(
        candidate_token_ids=candidate_token_ids,
        prompt_boundary=len(prompt_token_ids),
        admitted=True,
    )
    verification = verify_greedy_candidates(
        request_id, candidate_token_ids, target_token_ids
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
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=torch.tensor([pending_token_id], dtype=torch.long),
        can_run_cuda_graph=False,
        temporal_verification=verification,
    )
    committed_prefix = () if cancel_before_commit else accepted_prefix
    owned_boundary = len(prompt_token_ids) + len(committed_prefix)

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
    assert batch_result.next_token_ids.tolist() == [pending_token_id]
    assert not temporal_request.admitted

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
    assert allocator.free_group_depth == 0

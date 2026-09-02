# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


class _FakeHiFT(torch.nn.Module):
    # cosyvoice3.yaml: upsample_rates [8, 5, 3], istft_params.hop_len 4.
    upsample_rates: ClassVar[list[int]] = [8, 5, 3]
    istft_params: ClassVar[dict[str, int]] = {"n_fft": 16, "hop_len": 4}

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def inference(self, *, speech_feat, finalize):
        self.calls.append((speech_feat, finalize))
        batch, _, frames = speech_feat.shape
        row = torch.arange(frames * 480, dtype=torch.float32).reshape(1, -1)
        return row.repeat(batch, 1), None


class _FakeEstimator(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("batch adapter should be mocked in vocoder unit tests")


class _BatchCapableFakeFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.output_size = 80
        self.token_mel_ratio = 2
        self.input_embedding = torch.nn.Embedding(32, 80)
        self.spk_embed_affine_layer = torch.nn.Linear(192, 80)
        self.pre_lookahead_layer = torch.nn.Identity()
        self.decoder = SimpleNamespace(
            rand_noise=torch.zeros(1, 80, 1000),
            t_scheduler="cosine",
            inference_cfg_rate=0.7,
            estimator=_FakeEstimator(),
            forward_estimator=lambda *args, **kwargs: None,
        )


def _payload(state: FunCosyVoice3State) -> StagePayload:
    return StagePayload(
        request_id="req-vocoder",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def test_cosyvoice3_vocoder_prepare_and_store_audio_payload() -> None:
    vocoder = stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    state = FunCosyVoice3State(
        text="hello",
        audio_codes=torch.tensor([[1, 2], [3, 4]]),
        flow_prompt_speech_token=torch.tensor([[5]], dtype=torch.int32),
        flow_embedding=torch.ones(1, 192),
    )
    payload = _payload(state)

    restored_state, codes = vocoder.prepare_item(payload)
    assert restored_state.text == "hello"
    assert torch.equal(codes, torch.tensor([1, 2, 3, 4]))

    stored = vocoder.store_result(
        payload, restored_state, torch.tensor([[0.1, 0.2]]), 24000
    )
    assert stored.data["audio_waveform_shape"] == [2]
    assert stored.data["audio_waveform_dtype"] == "float32"
    assert stored.data["sample_rate"] == 24000
    assert stored.data["modality"] == "audio"
    assert "audio_codes" not in stored.data


def test_cosyvoice3_vocoder_rejects_payload_without_audio_codes() -> None:
    vocoder = stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    payload = _payload(FunCosyVoice3State(text="hello"))

    with pytest.raises(RuntimeError, match="requires audio_codes"):
        vocoder.prepare_item(payload)


def test_cosyvoice3_vocoder_rejects_missing_audio_output() -> None:
    vocoder = stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    state = FunCosyVoice3State(text="hello")
    payload = _payload(state)

    with pytest.raises(RuntimeError, match="did not return audio"):
        vocoder.store_result(payload, state, None, 24000)


def test_cosyvoice3_vocoder_decode_batch_uses_state_conditioning(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    state = FunCosyVoice3State(
        speed=1.5,
        flow_prompt_speech_token=torch.tensor([[5]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 1, 80),
        flow_embedding=torch.ones(1, 192),
    )

    results = asyncio.run(vocoder.decode_batch([(state, torch.tensor([1, 2]))]))

    assert len(results) == 1
    assert results[0][1] == 24000
    assert batch_calls[0][0].prompt_token.tolist() == [[5]]


def _state(
    *,
    sample_rate: int = 24000,
    prompt_tokens: int = 1,
    prompt_feat_frames: int | None = None,
) -> FunCosyVoice3State:
    if prompt_feat_frames is None:
        prompt_feat_frames = prompt_tokens * 2
    return FunCosyVoice3State(
        sample_rate=sample_rate,
        flow_prompt_speech_token=torch.arange(prompt_tokens).reshape(1, -1),
        flow_prompt_speech_feat=torch.zeros(1, prompt_feat_frames, 80),
        flow_embedding=torch.ones(1, 192),
    )


def _codes(length: int, value: int = 1) -> torch.Tensor:
    return torch.full((length,), value, dtype=torch.long)


def _install_fake_batch_adapter(monkeypatch, calls: list[list]) -> None:
    def fake_infer(flow, inputs):
        del flow
        calls.append(list(inputs))
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fake_infer)


def _run_decode_with_coalescing(
    monkeypatch,
    items,
    *,
    span_frames: int,
    max_added_padding_pct: float,
    bucket_frames: int = 50,
):
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(
        flow,
        hift,
        flow_batch_bucket_frames=bucket_frames,
        flow_batch_coalesce_span_frames=span_frames,
        flow_batch_coalesce_max_added_padding_pct=max_added_padding_pct,
    )
    results = asyncio.run(vocoder.decode_batch(items))
    return results, batch_calls, hift


def test_decode_batch_size_one_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    results = asyncio.run(vocoder.decode_batch([(_state(), _codes(2))]))

    assert len(results) == 1
    assert [len(call) for call in batch_calls] == [1]
    assert len(hift.calls) == 1


def test_decode_payload_size_one_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    state = _state()
    state.audio_codes = _codes(2)

    result = asyncio.run(vocoder.decode_payload(_payload(state)))

    assert result.data["modality"] == "audio"
    assert [len(call) for call in batch_calls] == [1]


def test_decode_batch_singleton_buckets_use_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    asyncio.run(vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(26))]))

    assert [len(call) for call in batch_calls] == [1, 1]
    assert len(hift.calls) == 2


def test_decode_batch_same_bucket_batches_flow_once(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    asyncio.run(vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))]))

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 2
    # HiFT runs once over the padded batch rather than once per request.
    assert len(hift.calls) == 1
    assert hift.calls[0][0].shape[0] == 2


def test_decode_batch_runs_hift_once_over_padded_mels(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    results = asyncio.run(
        vocoder.decode_batch(
            [(_state(), _codes(9)), (_state(), _codes(10)), (_state(), _codes(11))]
        )
    )

    # 9/10/11 tokens -> 18/20/22 mel frames. Right-zero-padded into one call.
    assert len(hift.calls) == 1
    speech_feat, finalize = hift.calls[0]
    assert finalize is True
    assert speech_feat.shape == (3, 80, 22)
    assert torch.count_nonzero(speech_feat[0, :, 18:]) == 0
    assert torch.count_nonzero(speech_feat[1, :, 20:]) == 0
    # Each request is sliced back to its own true length.
    assert [wav.shape[-1] for wav, _ in results] == [18 * 480, 20 * 480, 22 * 480]


def test_decode_batch_splits_hift_batch_when_padding_waste_is_large(
    monkeypatch,
) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    _install_fake_batch_adapter(monkeypatch, [])
    # max_waste=1.0 only accepts groups that need no padding at all.
    vocoder = stages._CosyVoice3Vocoder(flow, hift, hift_max_padding_waste=1.0)

    asyncio.run(vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))]))

    assert len(hift.calls) == 2


def test_decode_batch_long_singleton_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

    asyncio.run(vocoder.decode_batch([(_state(prompt_tokens=0), _codes(2200, 1))]))

    assert [len(call) for call in batch_calls] == [1]
    assert batch_calls[0][0].token.shape[1] == 2200


def test_decode_batch_different_buckets_do_not_share_padding(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    items = [
        (_state(), _codes(9)),
        (_state(), _codes(10)),
        (_state(), _codes(25)),
        (_state(), _codes(26)),
    ]

    asyncio.run(vocoder.decode_batch(items))

    assert [len(call) for call in batch_calls] == [2, 2]
    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [9, 10],
        [25, 26],
    ]


def test_flow_coalescing_disabled_preserves_current_bucket_groups(monkeypatch) -> None:
    items = [
        (_state(sample_rate=16001, prompt_tokens=0), _codes(27)),
        (_state(sample_rate=16002, prompt_tokens=0), _codes(24)),
        (_state(sample_rate=16003, prompt_tokens=0), _codes(25)),
    ]

    results, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=0,
        max_added_padding_pct=0,
    )

    assert [sample_rate for _, sample_rate in results] == [16001, 16002, 16003]
    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [27],
        [24, 25],
    ]


def test_s64_flow_coalescing_merges_adjacent_baseline_buckets(monkeypatch) -> None:
    items = [
        (_state(prompt_tokens=0), _codes(25)),
        (_state(prompt_tokens=0), _codes(27)),
    ]

    _, batch_calls, hift = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=64,
        max_added_padding_pct=5,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25, 27]
    ]
    assert len(hift.calls) == 2


def test_flow_coalescing_preserves_result_order_across_merged_bucket(
    monkeypatch,
) -> None:
    items = [
        (_state(sample_rate=16001, prompt_tokens=0), _codes(27)),
        (_state(sample_rate=16002, prompt_tokens=0), _codes(25)),
    ]

    results, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=64,
        max_added_padding_pct=5,
    )

    assert [sample_rate for _, sample_rate in results] == [16001, 16002]
    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25, 27]
    ]


def test_flow_coalescing_keeps_baseline_buckets_atomic(monkeypatch) -> None:
    items = [
        (_state(prompt_tokens=0), _codes(24)),  # 48 frames, bucket 1
        (_state(prompt_tokens=0), _codes(25)),  # 50 frames, bucket 1
        (_state(prompt_tokens=0), _codes(27)),  # 54 frames, bucket 2
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=1,
        max_added_padding_pct=5,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [24, 25],
        [27],
    ]


def test_flow_coalescing_merges_only_adjacent_sorted_baseline_buckets(
    monkeypatch,
) -> None:
    # The outer input is intentionally out of bucket order. The adaptive
    # policy sorts atomic buckets, then can merge only neighboring buckets.
    items = [
        (_state(prompt_tokens=0), _codes(52)),  # 104 frames, bucket 3
        (_state(prompt_tokens=0), _codes(25)),  # 50 frames, bucket 1
        (_state(prompt_tokens=0), _codes(27)),  # 54 frames, bucket 2
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=64,
        max_added_padding_pct=30,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25, 27],
        [52],
    ]


def test_flow_coalescing_rejects_merge_over_span_limit(monkeypatch) -> None:
    items = [
        (_state(prompt_tokens=0), _codes(25)),  # 50 frames
        (_state(prompt_tokens=0), _codes(47)),  # 94 frames
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=40,
        max_added_padding_pct=100,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25],
        [47],
    ]


def test_flow_coalescing_applies_padding_cap_to_whole_outer_batch(monkeypatch) -> None:
    # The full 50/54/104-frame merge fits the span limit, but its 50% added
    # work exceeds the 5% cap. Only the first adjacent merge is selected.
    items = [
        (_state(prompt_tokens=0), _codes(25)),
        (_state(prompt_tokens=0), _codes(27)),
        (_state(prompt_tokens=0), _codes(52)),
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=64,
        max_added_padding_pct=5,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25, 27],
        [52],
    ]
    baseline_work = 50 + 54 + 104
    selected_work = 2 * 54 + 104
    assert (selected_work / baseline_work - 1) * 100 <= 5


def test_flow_coalescing_prefers_fewer_solves_then_less_padded_work(
    monkeypatch,
) -> None:
    # One solve is over the global padding cap. Both two-solve partitions are
    # feasible, and merging 50/54 has less padded work than merging 54/104.
    items = [
        (_state(prompt_tokens=0), _codes(25)),
        (_state(prompt_tokens=0), _codes(27)),
        (_state(prompt_tokens=0), _codes(52)),
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=64,
        max_added_padding_pct=30,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25, 27],
        [52],
    ]


def test_flow_coalescing_uses_minimum_maximum_new_merge_span(monkeypatch) -> None:
    # With bucket width 10, the two feasible two-solve partitions have equal
    # padded work. The first merge has span 10, versus 20 for the second.
    items = [
        (_state(prompt_tokens=0), _codes(5)),
        (_state(prompt_tokens=0), _codes(5)),
        (_state(prompt_tokens=0), _codes(10)),
        (_state(prompt_tokens=0), _codes(20)),
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        bucket_frames=10,
        span_frames=40,
        max_added_padding_pct=30,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [5, 5, 10],
        [20],
    ]


def test_flow_coalescing_uses_deterministic_bucket_range_signature(monkeypatch) -> None:
    # The two two-solve candidates tie on work and merge span. The signature
    # ((1, 1), (2, 3)) wins lexicographically over ((1, 2), (3, 3)).
    items = [
        (_state(prompt_tokens=0), _codes(5)),
        (_state(prompt_tokens=0), _codes(10)),
        (_state(prompt_tokens=0), _codes(15)),
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        bucket_frames=10,
        span_frames=20,
        max_added_padding_pct=40,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [5],
        [10, 15],
    ]


def test_flow_coalescing_uses_supplied_non_s64_configuration(monkeypatch) -> None:
    items = [
        (_state(prompt_tokens=0), _codes(25)),
        (_state(prompt_tokens=0), _codes(27)),
    ]

    _, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=40,
        max_added_padding_pct=3,
    )

    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25],
        [27],
    ]


def test_flow_coalescing_allows_positive_span_with_zero_padding() -> None:
    vocoder = stages._CosyVoice3Vocoder(
        _BatchCapableFakeFlow(),
        _FakeHiFT(),
        flow_batch_coalesce_span_frames=64,
        flow_batch_coalesce_max_added_padding_pct=0,
    )

    assert vocoder._flow_batch_coalesce_span_frames == 64
    assert vocoder._flow_batch_coalesce_max_added_padding_pct == 0


def test_decode_batch_preserves_input_order_across_buckets(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    items = [
        (_state(sample_rate=16001), _codes(9, 1)),
        (_state(sample_rate=16002), _codes(25, 2)),
        (_state(sample_rate=16003), _codes(10, 3)),
        (_state(sample_rate=16004), _codes(26, 4)),
    ]

    results = asyncio.run(vocoder.decode_batch(items))

    assert [sample_rate for _, sample_rate in results] == [16001, 16002, 16003, 16004]
    assert [len(call) for call in batch_calls] == [2, 2]


def test_vocoder_rejects_non_pytorch_flow_estimator() -> None:
    flow = _BatchCapableFakeFlow()
    flow.decoder.estimator = object()

    with pytest.raises(RuntimeError, match="PyTorch Flow estimator"):
        stages._CosyVoice3Vocoder(flow, _FakeHiFT())


def test_decode_batch_alignment_mismatch_fails() -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

    with pytest.raises(ValueError, match="prompt feature length"):
        asyncio.run(
            vocoder.decode_batch(
                [
                    (_state(prompt_tokens=1, prompt_feat_frames=1), _codes(2)),
                    (_state(), _codes(3)),
                ]
            )
        )


def test_decode_batch_embedding_width_mismatch_fails() -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    invalid = _state()
    invalid.flow_embedding = torch.ones(1, 191)

    with pytest.raises(ValueError, match="embedding width"):
        asyncio.run(vocoder.decode_batch([(invalid, _codes(2)), (_state(), _codes(3))]))


def test_decode_batch_does_not_retry_after_batch_failure(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

    def fail_batch(flow, inputs):
        del flow, inputs
        raise RuntimeError("batch estimator failed")

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fail_batch)

    with pytest.raises(RuntimeError, match="batch estimator failed"):
        asyncio.run(
            vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))])
        )


def test_vocoder_rejects_non_positive_flow_bucket_size() -> None:
    with pytest.raises(ValueError, match="flow_batch_bucket_frames"):
        stages._CosyVoice3Vocoder(
            _BatchCapableFakeFlow(), _FakeHiFT(), flow_batch_bucket_frames=0
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"flow_batch_coalesce_span_frames": -1},
            "flow_batch_coalesce_span_frames",
        ),
        (
            {"flow_batch_coalesce_max_added_padding_pct": -1},
            "flow_batch_coalesce_max_added_padding_pct",
        ),
        (
            {"flow_batch_coalesce_max_added_padding_pct": float("nan")},
            "flow_batch_coalesce_max_added_padding_pct",
        ),
        (
            {"flow_batch_coalesce_max_added_padding_pct": float("inf")},
            "flow_batch_coalesce_max_added_padding_pct",
        ),
        (
            {
                "flow_batch_coalesce_span_frames": 0,
                "flow_batch_coalesce_max_added_padding_pct": 1,
            },
            "flow_batch_coalesce_max_added_padding_pct",
        ),
    ],
)
def test_vocoder_rejects_invalid_flow_coalescing_configuration(
    kwargs, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT(), **kwargs)


def test_flow_scheduler_cost_rounds_to_bucket() -> None:
    vocoder = stages._CosyVoice3Vocoder(
        _BatchCapableFakeFlow(), _FakeHiFT(), flow_batch_bucket_frames=50
    )
    state = _state(prompt_tokens=1)
    state.audio_codes = _codes(2)

    assert vocoder._flow_scheduler_cost(_payload(state)) == 50


def test_flow_admission_defers_request_after_long_singleton(monkeypatch) -> None:
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16: (_BatchCapableFakeFlow(), _FakeHiFT()),
    )
    # The default admission budget is sized for the real seed-tts-eval length
    # distribution, so pin it here: this test is about admission behaviour, not
    # about the default value.
    scheduler = stages.create_vocoder_executor(
        "model", device="cpu", flow_batch_admission_frames=2000
    )
    long_state = _state(prompt_tokens=0)
    long_state.audio_codes = _codes(2200)
    short_state = _state(prompt_tokens=0)
    short_state.audio_codes = _codes(2)
    first = IncomingMessage("long", "new_request", _payload(long_state))
    second = IncomingMessage("short", "new_request", _payload(short_state))
    scheduler.inbox.put(second)

    assert scheduler._max_batch_cost == 2000
    assert scheduler._collect_batch(first) == [first]
    assert scheduler._next_message() == second


def test_create_vocoder_executor_defaults_batch_for_real_lengths(monkeypatch) -> None:
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16: (_BatchCapableFakeFlow(), _FakeHiFT()),
    )
    scheduler = stages.create_vocoder_executor("model", device="cpu")

    assert scheduler._max_batch_cost == stages._DEFAULT_FLOW_BATCH_ADMISSION_FRAMES
    assert (
        scheduler._max_batch_cost // 713 >= 8
    ), "default admission budget no longer holds a useful batch"
    assert scheduler._max_batch_size == 16
    assert scheduler._max_batch_wait_s == pytest.approx(0.03)


def test_create_vocoder_executor_threads_batch_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_flow = _BatchCapableFakeFlow()
    fake_hift = _FakeHiFT()
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")

    def fake_load(checkpoint_dir, device, fp16):
        captured.update(
            {
                "checkpoint_dir": checkpoint_dir,
                "device": device,
                "fp16": fp16,
            }
        )
        return fake_flow, fake_hift

    monkeypatch.setattr(stages, "_load_cosyvoice3_flow_hift", fake_load)

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        dtype="float16",
        max_batch_size=6,
        max_batch_wait_ms=7,
        flow_batch_bucket_frames=100,
        flow_batch_admission_frames=200,
        flow_batch_coalesce_span_frames=40,
        flow_batch_coalesce_max_added_padding_pct=3,
    )

    assert isinstance(scheduler, stages.SimpleScheduler)
    assert scheduler._max_batch_size == 6
    assert scheduler._max_batch_wait_s == pytest.approx(0.007)
    assert scheduler._max_batch_cost == 200
    assert callable(scheduler._request_cost_fn)
    vocoder = scheduler._request_cost_fn.__self__
    assert vocoder._flow_batch_coalesce_span_frames == 40
    assert vocoder._flow_batch_coalesce_max_added_padding_pct == 3
    state = _state(prompt_tokens=1)
    state.audio_codes = _codes(2)
    assert scheduler._request_cost_fn(_payload(state)) == 100
    assert captured == {
        "checkpoint_dir": "/checkpoint",
        "device": "cpu",
        "fp16": True,
    }


def test_preprocessing_executor_threads_max_concurrency() -> None:
    scheduler = stages.create_preprocessing_executor("model", max_concurrency=11)
    assert scheduler._max_concurrency == 11


def test_preprocessing_executor_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        stages.create_preprocessing_executor("model", max_concurrency=0)


def test_onnx_intra_op_threads_reaches_both_encoders(monkeypatch) -> None:
    from sglang_omni.models.fun_cosyvoice3 import engine_builder, request_builders

    seen: dict[str, int] = {}

    def fake_tokenizer(model_path, device="cpu", intra_op_threads=1):
        seen["speech_tokenizer"] = intra_op_threads
        return object()

    def fake_encoder(model_path, device="cpu", intra_op_threads=1):
        seen["speaker_encoder"] = intra_op_threads
        return object()

    class _StubModel:
        def load_weights(self, weights) -> None:
            del weights

    monkeypatch.setattr(engine_builder, "SpeechTokenizerV3", fake_tokenizer)
    monkeypatch.setattr(engine_builder, "SpeakerEncoder", fake_encoder)
    monkeypatch.setattr(engine_builder, "CosyVoice3Tokenizer", lambda path: object())
    monkeypatch.setattr(engine_builder.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(
        request_builders, "set_cosyvoice3_preprocessing_context", lambda **kwargs: None
    )

    builder = engine_builder.FunCosyVoice3EngineBuilder(onnx_intra_op_threads=6)
    builder._checkpoint_root = "/tmp"
    builder.setup_model(
        model_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=_StubModel(),
                model_config=SimpleNamespace(vocab_size=0),
            )
        ),
        checkpoint_dir="/tmp",
        device="cpu",
        gpu_id=0,
        server_args=object(),
    )

    assert seen == {"speech_tokenizer": 6, "speaker_encoder": 6}


def test_create_vocoder_executor_rejects_large_batch_for_coalescing() -> None:
    with pytest.raises(ValueError, match=r"max_batch_size.*8"):
        stages.create_vocoder_executor(
            "model-that-must-not-load",
            max_batch_size=9,
            flow_batch_coalesce_span_frames=64,
            flow_batch_coalesce_max_added_padding_pct=5,
        )


def test_create_vocoder_executor_allows_large_batch_when_coalescing_disabled(
    monkeypatch,
) -> None:
    fake_flow = _BatchCapableFakeFlow()
    fake_hift = _FakeHiFT()
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16: (fake_flow, fake_hift),
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=16,
        flow_batch_coalesce_span_frames=0,
        flow_batch_coalesce_max_added_padding_pct=0,
    )

    assert scheduler._max_batch_size == 16


def test_create_vocoder_executor_rejects_non_positive_admission_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")

    with pytest.raises(ValueError, match="flow_batch_admission_frames"):
        stages.create_vocoder_executor(
            "model",
            device="cpu",
            flow_batch_admission_frames=0,
        )


def test_pipeline_config_sets_flow_batch_bucket_by_default() -> None:
    vocoder_stage = next(
        stage
        for stage in FunCosyVoice3PipelineConfig(model_path="model").stages
        if stage.name == "vocoder"
    )

    assert vocoder_stage.factory.model_dump(exclude_none=True) == {
        "dtype": "bfloat16",
        "flow_batch_bucket_frames": 50,
        "flow_batch_admission_frames": 8000,
        "max_batch_size": 16,
        "max_batch_wait_ms": 30,
        "flow_batch_coalesce_span_frames": 0,
        "flow_batch_coalesce_max_added_padding_pct": 0.0,
        "enable_dit_torch_compile": False,
    }


def test_vocoder_hift_defaults_to_float32(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

    # bfloat16 gave HiFT no speedup, so the default keeps full precision.
    assert vocoder._hift_compute_dtype is None
    with vocoder._hift_autocast():
        assert not torch.is_autocast_enabled()

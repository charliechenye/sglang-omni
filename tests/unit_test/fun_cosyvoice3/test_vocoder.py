# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        token_count = kwargs["token"].shape[1]
        return torch.ones(1, 80, token_count * 2), None


class _FakeHiFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def inference(self, *, speech_feat, finalize):
        self.calls.append((speech_feat, finalize))
        return torch.arange(speech_feat.shape[-1]).reshape(1, -1).float(), None


class _FakeGraph:
    def __init__(self, static: dict[str, torch.Tensor], output: torch.Tensor) -> None:
        self.static = static
        self.output = output
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.output.copy_(self.static["mu"])


def _payload(state: FunCosyVoice3State) -> StagePayload:
    return StagePayload(
        request_id="req-vocoder",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


@pytest.mark.parametrize(
    ("decoder_t", "bucket_t"),
    [
        (254, None),
        (256, 256),
        (530, 544),
        (544, 544),
        (545, None),
        (546, 576),
        (574, 576),
        (770, 800),
        (994, 1024),
        (1024, 1024),
        (1026, None),
    ],
)
def test_cosyvoice3_flow_cuda_graph_bucket_policy(
    decoder_t: int, bucket_t: int | None
) -> None:
    assert stages._flow_cuda_graph_bucket(decoder_t) == bucket_t


@pytest.mark.parametrize(
    ("decoder_ts", "bucket_t"),
    [
        ((530, 542, 544), 544),
        ((546, 558, 574), 576),
    ],
)
def test_cosyvoice3_flow_cuda_graph_reuses_bucket_with_padding_and_crop(
    decoder_ts: tuple[int, ...], bucket_t: int
) -> None:
    runner = object.__new__(stages._CosyVoice3FlowCudaGraphRunner)
    runner._device = torch.device("cpu")
    static = runner._new_inputs(bucket_t)
    output = torch.empty(1, 80, bucket_t)
    graph = _FakeGraph(static, output)

    def eager_forward(*args, **kwargs):
        raise AssertionError("the bucket should be replayed")

    runner._eager_forward = eager_forward
    runner._graphs = {bucket_t: (graph, static, output)}

    for replay_count, decoder_t in enumerate(decoder_ts, start=1):
        mu = torch.full((1, 80, decoder_t), float(decoder_t))
        mask = torch.ones(1, 1, decoder_t)
        spks = torch.full((1, 80), 7.0)
        cond = torch.full((1, 80, decoder_t), 3.0)

        result, cache = runner.forward(
            mu=mu,
            mask=mask,
            n_timesteps=10,
            spks=spks,
            cond=cond,
            streaming=False,
        )

        assert cache is None
        assert graph.replays == replay_count
        assert result.shape == (1, 80, decoder_t)
        assert torch.equal(result, mu)
        assert torch.equal(static["mu"][..., :decoder_t], mu)
        assert torch.count_nonzero(static["mu"][..., decoder_t:]) == 0
        assert torch.equal(static["mask"][..., :decoder_t], mask)
        assert torch.count_nonzero(static["mask"][..., decoder_t:]) == 0
        assert torch.equal(static["cond"][..., :decoder_t], cond)
        assert torch.count_nonzero(static["cond"][..., decoder_t:]) == 0
        assert torch.equal(static["spks"], spks)


def test_cosyvoice3_vocoder_does_not_pad_or_rescale_short_sequences() -> None:
    flow = _FakeFlow()
    hift = _FakeHiFT()
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    # note: token id 0 is a valid FSQ speech token, not padding — the
    # vocoder must feed exactly what the AR stage generated through
    # untouched, with no minimum-length padding and no speed rescaling
    # (speed is applied once, downstream, on the decoded waveform).
    wav = vocoder._token2wav(
        token=torch.tensor([[0, 2]], dtype=torch.long),
        prompt_token=torch.tensor([[4]], dtype=torch.int32),
        prompt_feat=torch.zeros(1, 2, 80),
        embedding=torch.ones(1, 192),
    )

    flow_call = flow.calls[0]
    assert flow_call["token"].shape == (1, 2)
    assert flow_call["token"].tolist() == [[0, 2]]
    assert flow_call["token_len"].tolist() == [2]
    assert flow_call["prompt_token_len"].tolist() == [1]
    assert flow_call["prompt_feat_len"].tolist() == [2]
    assert flow_call["finalize"] is True
    assert (
        hift.calls[0][0].shape[-1] == 4
    )  # _FakeFlow returns token_count * 2 mel frames
    assert wav.device.type == "cpu"


def test_cosyvoice3_vocoder_raises_on_empty_token_sequence() -> None:
    vocoder = stages._CosyVoice3Vocoder(_FakeFlow(), _FakeHiFT())

    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        vocoder._token2wav(
            token=torch.zeros(1, 0, dtype=torch.long),
            prompt_token=torch.tensor([[4]], dtype=torch.int32),
            prompt_feat=torch.zeros(1, 2, 80),
            embedding=torch.ones(1, 192),
        )


def test_cosyvoice3_vocoder_prepare_and_store_audio_payload() -> None:
    vocoder = stages._CosyVoice3Vocoder(_FakeFlow(), _FakeHiFT())
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
    vocoder = stages._CosyVoice3Vocoder(_FakeFlow(), _FakeHiFT())
    payload = _payload(FunCosyVoice3State(text="hello"))

    with pytest.raises(RuntimeError, match="requires audio_codes"):
        vocoder.prepare_item(payload)


def test_cosyvoice3_vocoder_rejects_missing_audio_output() -> None:
    vocoder = stages._CosyVoice3Vocoder(_FakeFlow(), _FakeHiFT())
    state = FunCosyVoice3State(text="hello")
    payload = _payload(state)

    with pytest.raises(RuntimeError, match="did not return audio"):
        vocoder.store_result(payload, state, None, 24000)


def test_cosyvoice3_vocoder_decode_batch_uses_state_conditioning() -> None:
    flow = _FakeFlow()
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
    assert flow.calls[0]["prompt_token"].tolist() == [[5]]

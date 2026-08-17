# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the Voxtral breakable-prefill adopter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.voxtral_tts import CAPABILITIES
from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner
from sglang_omni.models.voxtral_tts.pipeline.engine_builder import (
    VoxtralTtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)

HIDDEN = 4
VOCAB = 128
AUDIO_TOKEN_ID = 24


def _runner() -> VoxtralTTSModelRunner:
    runner = object.__new__(VoxtralTTSModelRunner)
    embedding = torch.nn.Embedding(VOCAB, HIDDEN)
    runner.model = SimpleNamespace(get_input_embeddings=lambda: embedding)
    return runner


def _request(
    full_ids: list[int],
    *,
    prefix_len: int = 0,
    voice_embedding: torch.Tensor | None = None,
) -> SimpleNamespace:
    req = SimpleNamespace(
        prefix_indices=list(range(prefix_len)),
        extend_range=SimpleNamespace(length=len(full_ids) - prefix_len),
    )
    data = SimpleNamespace(
        req=req,
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        voice_embedding=voice_embedding,
        audio_token_id=AUDIO_TOKEN_ID,
    )
    return SimpleNamespace(data=data)


def _batch(requests: list[SimpleNamespace]) -> SimpleNamespace:
    chunks = []
    for sched_req in requests:
        data = sched_req.data
        prefix_len = len(data.req.prefix_indices)
        req_len = int(data.req.extend_range.length)
        chunks.append(data.input_ids[prefix_len : prefix_len + req_len])
    input_ids = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)
    return SimpleNamespace(
        input_ids=input_ids,
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=object(),
        positions=torch.arange(input_ids.numel(), dtype=torch.long),
        batch_size=len(requests),
        rids=[f"request-{idx}" for idx in range(len(requests))],
    )


def test_before_prefill_attaches_legacy_embeddings_without_mutating_upstream_fields():
    runner = _runner()
    requests = [_request([7, AUDIO_TOKEN_ID, 9, 10])]
    forward_batch = _batch(requests)
    official_mm_inputs = forward_batch.mm_inputs
    official_positions = forward_batch.positions
    expected = runner._build_prefill_input_embeds(forward_batch, requests).clone()

    runner.before_prefill(forward_batch, None, requests)

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    torch.testing.assert_close(sidecar.input_embeds, expected)
    text_rows = (forward_batch.input_ids != AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    text_embeds = runner.model.get_input_embeddings()(forward_batch.input_ids)
    torch.testing.assert_close(sidecar.input_embeds[text_rows], text_embeds[text_rows])
    assert sidecar.input_embeds.shape[0] == len(forward_batch.input_ids)
    assert forward_batch.input_embeds is None
    assert forward_batch.replace_embeds is None
    assert forward_batch.mm_inputs is official_mm_inputs
    assert forward_batch.positions is official_positions
    assert runner.custom_prefill_forward(forward_batch, None, requests) is None


def test_sidecar_preserves_voice_replacement_offsets_across_requests_and_prefixes():
    runner = _runner()
    voice_one = torch.arange(8, dtype=torch.float32).reshape(2, HIDDEN) + 100
    voice_two = torch.arange(16, dtype=torch.float32).reshape(4, HIDDEN) + 200
    requests = [
        _request(
            [7, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 9],
            voice_embedding=voice_one,
        ),
        _request(
            [8, AUDIO_TOKEN_ID, 10, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID],
            prefix_len=2,
            voice_embedding=voice_two,
        ),
    ]
    forward_batch = _batch(requests)
    expected = runner._build_prefill_input_embeds(forward_batch, requests).clone()

    runner.before_prefill(forward_batch, None, requests)

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert sidecar.input_embeds.shape == (7, HIDDEN)

    # Request one starts at row zero and consumes voice rows 0 and 1.
    torch.testing.assert_close(sidecar.input_embeds[1:3], voice_one)
    # Request two starts after its two-token cached prefix. Its first cached
    # token is audio, so the live rows must consume voice rows 1 and 2.
    torch.testing.assert_close(sidecar.input_embeds[5:7], voice_two[1:3])


def test_sidecar_rejects_incompatible_replacement_inputs():
    runner = _runner()
    requests = [_request([7, 8, 9])]
    forward_batch = _batch(requests)
    forward_batch.replace_embeds = object()

    with pytest.raises(RuntimeError, match="replace_embeds"):
        runner.before_prefill(forward_batch, None, requests)

    assert get_omni_prefill_inputs(forward_batch) is None
    assert forward_batch.input_embeds is None


def test_prefill_tail_selects_logical_request_endpoints_from_padded_bcg_output():
    from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel

    model = VoxtralSGLangTTSModel.__new__(VoxtralSGLangTTSModel)
    seen: dict[str, object] = {}
    captured_hidden = torch.arange(8 * HIDDEN, dtype=torch.float32).reshape(8, HIDDEN)

    def fake_language_model(input_ids, positions, forward_batch, input_embeds=None):
        del forward_batch
        assert input_ids.shape == (8,)
        assert positions.shape == (8,)
        assert input_embeds is not None
        assert input_embeds.shape == (5, HIDDEN)
        seen["input_embeds"] = input_embeds
        return captured_hidden

    model.language_model = fake_language_model
    model._decode_input_embed_buffer = torch.zeros((2, HIDDEN))
    forward_batch = SimpleNamespace(
        # The outer BCG replay sees the padded static batch, while the sidecar
        # still carries only the live extend-window embeddings.
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
        extend_seq_lens=torch.tensor([2, 3], dtype=torch.long),
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_extend=lambda: True,
        ),
    )
    input_embeds = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)

    output = model.forward(
        forward_batch.input_ids,
        torch.arange(8, dtype=torch.long),
        forward_batch,
        input_embeds=input_embeds,
        omni_prefill_rids=["request-one", "request-two"],
    )

    expected_hidden = captured_hidden[[1, 4]]
    torch.testing.assert_close(output.hidden_states, expected_hidden)
    assert seen["input_embeds"] is input_embeds
    assert output.next_token_logits.shape == (2, 1)
    assert output.next_token_logits.dtype == output.hidden_states.dtype


def test_voxtral_prefill_sidecar_round_trips_through_eager_forward_kwargs():
    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
    from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel

    runner = _runner()
    requests = [
        _request(
            [7, AUDIO_TOKEN_ID, 9],
            voice_embedding=torch.arange(8, dtype=torch.float32).reshape(2, HIDDEN),
        )
    ]
    forward_batch = _batch(requests)
    expected = runner._build_prefill_input_embeds(forward_batch, requests).clone()

    runner.before_prefill(forward_batch, None, requests)
    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None

    eager_runner = SGLModelRunner.__new__(SGLModelRunner)
    eager_runner.support_pp = False
    eager_runner.is_generation = True
    kwargs = eager_runner._extend_forward_kwargs(forward_batch, object())

    assert kwargs["input_embeds"] is sidecar.input_embeds
    torch.testing.assert_close(kwargs["input_embeds"], expected)
    assert kwargs["omni_prefill_rids"] is forward_batch.rids
    assert forward_batch.input_embeds is None

    model = VoxtralSGLangTTSModel.__new__(VoxtralSGLangTTSModel)
    model._decode_input_embed_buffer = torch.zeros((1, HIDDEN))
    seen: dict[str, object] = {}

    def fake_language_model(input_ids, positions, forward_batch, input_embeds=None):
        del input_ids, positions, forward_batch
        seen["input_embeds"] = input_embeds
        return input_embeds + 1

    model.language_model = fake_language_model
    forward_batch.extend_seq_lens = torch.tensor([3], dtype=torch.long)
    forward_batch.forward_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_extend=lambda: True,
    )
    output = model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
        input_embeds=kwargs["input_embeds"],
        omni_prefill_rids=kwargs["omni_prefill_rids"],
    )

    assert seen["input_embeds"] is sidecar.input_embeds
    torch.testing.assert_close(output.hidden_states, (expected + 1)[-1:])


def test_capability_and_builder_adoption_are_synchronized_without_default_enablement():
    builder = VoxtralTtsEngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")

    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is True
    assert (
        type(builder).supports_breakable_prefill_cuda_graph
        == CAPABILITIES.supports_breakable_prefill_cuda_graph
    )
    assert type(builder).supports_breakable_prefill_cuda_graph is True
    assert "cuda_graph_backend_prefill" not in defaults
    assert "cuda_graph_bs_prefill" not in defaults


def test_explicit_breakable_override_is_accepted_by_shared_policy():
    builder = VoxtralTtsEngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")
    overrides = build_generation_batch_overrides(
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [4, 8, 16],
        },
        **defaults,
    )

    assert overrides["cuda_graph_backend_prefill"] == "breakable"
    assert overrides["cuda_graph_bs_prefill"] == [4, 8, 16]
    assert overrides["cuda_graph_max_bs_prefill"] == 16

    server_args = SimpleNamespace(
        max_running_requests=overrides["max_running_requests"],
        disable_cuda_graph=False,
        enable_torch_compile=overrides["enable_torch_compile"],
        torch_compile_max_bs=overrides["torch_compile_max_bs"],
        chunked_prefill_size=overrides["max_prefill_tokens"],
        max_prefill_tokens=overrides["max_prefill_tokens"],
        attn_cp_size=1,
        dcp_size=1,
        lora_paths=None,
        enable_lora=None,
        moe_a2a_backend="none",
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            ),
            prefill=SimpleNamespace(
                backend=overrides["cuda_graph_backend_prefill"],
                bs=overrides["cuda_graph_bs_prefill"],
                max_bs=overrides["cuda_graph_max_bs_prefill"],
            ),
        ),
        _cuda_graph_config_locked=frozenset({("prefill", "bs")}),
    )

    validate_generation_batch_policy(
        model_name="Voxtral TTS",
        server_args=server_args,
    )

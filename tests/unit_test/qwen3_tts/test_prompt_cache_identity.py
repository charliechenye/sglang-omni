"""CPU-only contract tests for the Qwen3-TTS semantic prompt identity."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.qwen3_tts.prompt_cache_identity import (
    PromptCacheNamespace,
    Qwen3TTSPromptConfig,
    Qwen3TTSPromptRequest,
    build_qwen3_tts_prompt_cache_identity,
    qwen3_tts_reference_source_key,
)
from sglang_omni.models.qwen3_tts.request_builders import build_embedding_cache_key_ids
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

CONFIG = Qwen3TTSPromptConfig(
    tts_bos_token_id=1,
    tts_eos_token_id=2,
    tts_pad_token_id=3,
    codec_bos_token_id=4,
    codec_pad_token_id=5,
    codec_think_token_id=6,
    codec_nothink_token_id=7,
    codec_think_bos_token_id=8,
    codec_think_eos_token_id=9,
    codec_language_ids={"en": 10, "zh": 11, "dialect": 12},
    speaker_ids={"ryan": 20, "alice": 21},
    speaker_dialects={"ryan": "dialect"},
    num_code_groups=3,
)
NAMESPACE = PromptCacheNamespace(
    model_revision="fixture-model-rev",
    model_config_identity="fixture-config-v1",
    reference_encoder_identity="fixture-reference-encoder-v1",
)
ROLE_IDS = (100, 101, 102)
TARGET_IDS = (200, 201, 202)
REFERENCE_AUDIO_KEY = "bytes:fixture-reference"
REFERENCE_TEXT_IDS = (300, 301)
REFERENCE_CODE_FRAMES = ((400, 401, 402), (410, 411, 412))
HIDDEN_SIZE = 24
TEXT_EMBED_OFFSET = 0
MAIN_CODEC_EMBED_OFFSET = 4
PREDICTOR_GROUP_1_EMBED_OFFSET = 8
PREDICTOR_GROUP_2_EMBED_OFFSET = 12
SPEAKER_EMBED_OFFSET = 16


def _base_request(**changes: object) -> Qwen3TTSPromptRequest:
    request = Qwen3TTSPromptRequest(
        task_type="Base",
        role_text_ids=ROLE_IDS,
        target_text_ids=TARGET_IDS,
        reference_audio_key=REFERENCE_AUDIO_KEY,
        reference_text_ids=REFERENCE_TEXT_IDS,
        reference_code_frames=REFERENCE_CODE_FRAMES,
    )
    return replace(request, **changes)


def _xvector_request(**changes: object) -> Qwen3TTSPromptRequest:
    return _base_request(x_vector_only_mode=True, **changes)


def _custom_request(**changes: object) -> Qwen3TTSPromptRequest:
    request = Qwen3TTSPromptRequest(
        task_type="CustomVoice",
        role_text_ids=ROLE_IDS,
        target_text_ids=TARGET_IDS,
        voice="Ryan",
    )
    return replace(request, **changes)


def _design_request(**changes: object) -> Qwen3TTSPromptRequest:
    request = Qwen3TTSPromptRequest(
        task_type="VoiceDesign",
        role_text_ids=ROLE_IDS,
        target_text_ids=TARGET_IDS,
        language="en",
        instruction_text_ids=(500, 501),
    )
    return replace(request, **changes)


def _identity(request: Qwen3TTSPromptRequest):
    return build_qwen3_tts_prompt_cache_identity(request, CONFIG, namespace=NAMESPACE)


def _fill_embedding(embedding: nn.Embedding, *, base: float, offset: int) -> None:
    with torch.no_grad():
        ids = torch.arange(embedding.num_embeddings, dtype=torch.float32).unsqueeze(1)
        embedding.weight.zero_()
        embedding.weight[:, offset] = (base + ids).squeeze(1)
        embedding.weight[:, offset + 1] = ids.squeeze(1)


class _TinyPromptModel(nn.Module):
    """Deterministic CPU embedding modules used by the real talker methods."""

    def __init__(self) -> None:
        super().__init__()
        self.codec_embedding = nn.Embedding(1000, HIDDEN_SIZE)
        self.text_embedding = nn.Embedding(1000, HIDDEN_SIZE)
        _fill_embedding(
            self.codec_embedding,
            base=1_000,
            offset=MAIN_CODEC_EMBED_OFFSET,
        )
        _fill_embedding(
            self.text_embedding,
            base=100,
            offset=TEXT_EMBED_OFFSET,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        return self.text_embedding


class _TinyCodePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.codec_embedding = nn.ModuleList(
            [nn.Embedding(1000, HIDDEN_SIZE), nn.Embedding(1000, HIDDEN_SIZE)]
        )
        _fill_embedding(
            self.model.codec_embedding[0],
            base=10_000,
            offset=PREDICTOR_GROUP_1_EMBED_OFFSET,
        )
        _fill_embedding(
            self.model.codec_embedding[1],
            base=20_000,
            offset=PREDICTOR_GROUP_2_EMBED_OFFSET,
        )


class _TinyQwen3TTSTalker(Qwen3TTSTalker):
    """Use Qwen3TTSTalker's prompt construction with tiny CPU-only modules."""

    def __init__(self) -> None:
        # Bypass the production constructor: it allocates the full transformer,
        # speaker encoder, predictor buffers, and CUDA-specific runtime state.
        nn.Module.__init__(self)
        self.root_config = SimpleNamespace(
            tts_bos_token_id=1,
            tts_eos_token_id=2,
            tts_pad_token_id=3,
        )
        self.config = SimpleNamespace(
            codec_bos_id=4,
            codec_pad_id=5,
            codec_think_id=6,
            codec_nothink_id=7,
            codec_think_bos_id=8,
            codec_think_eos_id=9,
            codec_language_id={"en": 10, "zh": 11, "dialect": 12},
            spk_id={"ryan": 20, "alice": 21},
            spk_is_dialect={"ryan": "dialect"},
            num_code_groups=3,
        )
        self.model = _TinyPromptModel()
        self.text_projection = nn.Identity()
        self.code_predictor = _TinyCodePredictor()


def _speaker_embedding(value: float) -> torch.Tensor:
    embedding = torch.zeros((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    embedding[..., SPEAKER_EMBED_OFFSET] = value
    embedding[..., SPEAKER_EMBED_OFFSET + 1] = value + 1
    return embedding


_SPEAKER_EMBEDS = {
    REFERENCE_AUDIO_KEY: _speaker_embedding(50_000),
    "bytes:other-reference": _speaker_embedding(60_000),
}


def _tokenized_text(
    token_ids: tuple[int, ...], suffix: tuple[int, ...]
) -> torch.Tensor:
    return torch.tensor([[*ROLE_IDS, *token_ids, *suffix]], dtype=torch.long)


def _build_real_prompt_embeddings(request: Qwen3TTSPromptRequest) -> torch.Tensor:
    talker = _TinyQwen3TTSTalker()
    input_id = _tokenized_text(request.target_text_ids, (900, 901, 902, 903, 904))
    instruct_id = (
        torch.tensor([list(request.instruction_text_ids)], dtype=torch.long)
        if request.instruction_text_ids
        else None
    )

    with torch.inference_mode():
        if request.task_type == "Base":
            assert request.reference_audio_key is not None
            ref_id = _tokenized_text(request.reference_text_ids, (703, 704))
            voice_clone_prompt = {
                "ref_spk_embedding": [
                    _SPEAKER_EMBEDS[request.reference_audio_key].clone()
                ],
                "icl_mode": [not request.x_vector_only_mode],
            }
            if not request.x_vector_only_mode:
                voice_clone_prompt["ref_code"] = [
                    torch.tensor(request.reference_code_frames, dtype=torch.long)
                ]
            prompt_embeddings, _, _, _ = talker.build_voice_clone_inputs(
                input_id=input_id,
                ref_id=ref_id,
                voice_clone_prompt=voice_clone_prompt,
                language=request.language,
                non_streaming_mode=request.non_streaming_mode,
                instruct_id=instruct_id,
            )
        elif request.task_type == "CustomVoice":
            assert request.voice is not None
            prompt_embeddings, _, _, _ = talker.build_custom_voice_inputs(
                input_id=input_id,
                voice=request.voice,
                language=request.language,
                non_streaming_mode=request.non_streaming_mode,
                instruct_id=instruct_id,
            )
        else:
            prompt_embeddings, _, _, _ = talker.build_voice_design_inputs(
                input_id=input_id,
                language=request.language,
                non_streaming_mode=request.non_streaming_mode,
                instruct_id=instruct_id,
            )
    return prompt_embeddings.squeeze(0).detach()


def _old_embedding_lcp(
    left_request: Qwen3TTSPromptRequest, right_request: Qwen3TTSPromptRequest
) -> int:
    left_ids = build_embedding_cache_key_ids(
        _build_real_prompt_embeddings(left_request)
    )
    right_ids = build_embedding_cache_key_ids(
        _build_real_prompt_embeddings(right_request)
    )
    for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        if left_id != right_id:
            return index
    return min(len(left_ids), len(right_ids))


def assert_prompt_lcp_equivalent(
    left_request: Qwen3TTSPromptRequest,
    right_request: Qwen3TTSPromptRequest,
    *,
    expected_lcp: int | None = None,
) -> tuple[int, int]:
    """Compare old real-builder and new semantic LCPs against the contract."""

    old_lcp = _old_embedding_lcp(left_request, right_request)
    new_lcp = _identity(left_request).lcp_length(_identity(right_request))
    assert old_lcp == new_lcp, f"old LCP={old_lcp}, new LCP={new_lcp}"
    if expected_lcp is not None:
        assert (
            old_lcp == expected_lcp
        ), f"old LCP={old_lcp}, expected LCP={expected_lcp}"
        assert (
            new_lcp == expected_lcp
        ), f"new LCP={new_lcp}, expected LCP={expected_lcp}"
    return old_lcp, new_lcp


def test_fixture_separates_additive_component_subspaces() -> None:
    talker = _TinyQwen3TTSTalker()
    text_embedding = talker.get_text_embeddings()
    codec_embedding = talker.get_input_embeddings()

    # The removed linear fixture made these two sums equal:
    # (100 + 200 * 10) + (1000 + 5 * 10) ==
    # (100 + 201 * 10) + (1000 + 4 * 10).
    old_linear_left = (100 + 200 * 10) + (1_000 + 5 * 10)
    old_linear_right = (100 + 201 * 10) + (1_000 + 4 * 10)
    assert old_linear_left == old_linear_right

    left = text_embedding(torch.tensor([[200]])) + codec_embedding(torch.tensor([[5]]))
    right = text_embedding(torch.tensor([[201]])) + codec_embedding(torch.tensor([[4]]))
    assert not torch.equal(left, right)


def test_real_builder_has_stable_rows_for_all_task_variants() -> None:
    requests = [
        _base_request(),
        _base_request(non_streaming_mode=True),
        _xvector_request(),
        _xvector_request(non_streaming_mode=True),
        _custom_request(language="en"),
        _design_request(),
    ]

    for request in requests:
        identity = _identity(request)
        prompt_embeddings = _build_real_prompt_embeddings(request)
        assert prompt_embeddings.shape[0] == len(identity.rows)
        assert build_embedding_cache_key_ids(prompt_embeddings) == (
            build_embedding_cache_key_ids(_build_real_prompt_embeddings(request))
        )


@pytest.mark.parametrize(
    ("left", "right", "expected_lcp"),
    [
        pytest.param(
            _base_request(),
            _base_request(reference_text_ids=(999, 301)),
            8,
            id="base-icl-streaming-reference-text",
        ),
        pytest.param(
            _base_request(),
            _base_request(reference_code_frames=((999, 401, 402), (410, 411, 412))),
            9,
            id="base-icl-streaming-reference-code",
        ),
        pytest.param(
            _base_request(non_streaming_mode=True),
            _base_request(
                non_streaming_mode=True,
                reference_text_ids=(999, 301),
            ),
            8,
            id="base-icl-non-streaming-reference-text",
        ),
        pytest.param(
            _base_request(non_streaming_mode=True),
            _base_request(
                non_streaming_mode=True,
                reference_code_frames=((999, 401, 402), (410, 411, 412)),
            ),
            15,
            id="base-icl-non-streaming-reference-code",
        ),
        pytest.param(
            _base_request(),
            _base_request(target_text_ids=(999, 201, 202)),
            10,
            id="base-icl-streaming-target-before-codec-boundary",
        ),
        pytest.param(
            _base_request(),
            _base_request(target_text_ids=(200, 999, 202)),
            11,
            id="base-icl-streaming-target-after-codec-boundary",
        ),
        pytest.param(
            _base_request(non_streaming_mode=True),
            _base_request(
                non_streaming_mode=True,
                target_text_ids=(200, 999, 202),
            ),
            11,
            id="base-icl-non-streaming-target",
        ),
        pytest.param(
            _base_request(non_streaming_mode=True),
            _base_request(
                non_streaming_mode=True,
                target_text_ids=(200, 201),
            ),
            12,
            id="base-icl-non-streaming-target-length",
        ),
        pytest.param(
            _base_request(language="en"),
            _base_request(language="zh"),
            5,
            id="base-icl-language",
        ),
        pytest.param(
            _base_request(instruction_text_ids=(500, 501)),
            _base_request(instruction_text_ids=(500, 999)),
            1,
            id="base-icl-instruction",
        ),
        pytest.param(
            _xvector_request(),
            _xvector_request(target_text_ids=(200, 201, 999)),
            9,
            id="base-x-vector-streaming-trailing-target",
        ),
        pytest.param(
            _xvector_request(),
            _xvector_request(target_text_ids=(999, 201, 202)),
            8,
            id="base-x-vector-streaming-first-target",
        ),
        pytest.param(
            _xvector_request(non_streaming_mode=True),
            _xvector_request(
                non_streaming_mode=True,
                target_text_ids=(200, 999, 202),
            ),
            9,
            id="base-x-vector-non-streaming-target",
        ),
        pytest.param(
            _xvector_request(language="en"),
            _xvector_request(language="zh"),
            5,
            id="base-x-vector-language",
        ),
        pytest.param(
            _xvector_request(instruction_text_ids=(500, 501)),
            _xvector_request(instruction_text_ids=(500, 999)),
            1,
            id="base-x-vector-instruction",
        ),
        pytest.param(
            _xvector_request(non_streaming_mode=True),
            _xvector_request(
                non_streaming_mode=True,
                target_text_ids=(200, 201),
            ),
            10,
            id="base-x-vector-non-streaming-target-length",
        ),
        pytest.param(
            _custom_request(language="en"),
            _custom_request(language="zh"),
            5,
            id="custom-voice-language",
        ),
        pytest.param(
            _custom_request(language="en"),
            _custom_request(language="en", voice="Alice"),
            7,
            id="custom-voice-speaker",
        ),
        pytest.param(
            _custom_request(language="en", instruction_text_ids=(500, 501)),
            _custom_request(language="en", instruction_text_ids=(500, 999)),
            1,
            id="custom-voice-instruction",
        ),
        pytest.param(
            _custom_request(language="en"),
            _custom_request(language="en", target_text_ids=(200, 999, 202)),
            10,
            id="custom-voice-target",
        ),
        pytest.param(
            _design_request(),
            _design_request(instruction_text_ids=(500, 999)),
            1,
            id="voice-design-instruction",
        ),
        pytest.param(
            _design_request(),
            _design_request(target_text_ids=(200, 999, 202)),
            11,
            id="voice-design-target",
        ),
        pytest.param(
            _design_request(language="en"),
            _design_request(language="zh"),
            7,
            id="voice-design-language",
        ),
        pytest.param(
            _custom_request(language="auto"),
            _custom_request(language="dialect"),
            14,
            id="custom-voice-auto-dialect",
        ),
        pytest.param(
            _custom_request(language="auto"),
            _custom_request(language="en"),
            5,
            id="custom-voice-auto-explicit-language",
        ),
        pytest.param(
            _xvector_request(),
            _xvector_request(reference_audio_key="bytes:other-reference"),
            6,
            id="base-x-vector-reference-source",
        ),
    ],
)
def test_semantic_lcp_matches_real_builder_hashes(left, right, expected_lcp) -> None:
    assert_prompt_lcp_equivalent(left, right, expected_lcp=expected_lcp)


def test_icl_reference_code_length_preserves_streaming_prefix_boundary() -> None:
    one_frame = _base_request(reference_code_frames=((400, 401, 402),))
    two_frames = _base_request()

    assert _identity(one_frame).is_prefix_of(_identity(two_frames))
    assert_prompt_lcp_equivalent(one_frame, two_frames, expected_lcp=10)


def test_streaming_and_non_streaming_share_only_conditioned_prefix() -> None:
    streaming = _xvector_request()
    non_streaming = _xvector_request(non_streaming_mode=True)

    assert _identity(streaming).lcp_length(_identity(non_streaming)) == (
        len(_identity(streaming).rows) - 1
    )
    assert_prompt_lcp_equivalent(streaming, non_streaming, expected_lcp=8)


def test_instructions_are_prepended_and_cannot_false_share() -> None:
    without_instruction = _identity(_custom_request())
    with_instruction = _identity(_custom_request(instruction_text_ids=(500,)))

    assert without_instruction.lcp_length(with_instruction) == 0
    assert_prompt_lcp_equivalent(
        _custom_request(),
        _custom_request(instruction_text_ids=(500,)),
        expected_lcp=0,
    )


def test_namespace_changes_disable_cross_model_prefix_reuse() -> None:
    identity = _identity(_xvector_request())
    changed_model = build_qwen3_tts_prompt_cache_identity(
        _xvector_request(),
        CONFIG,
        namespace=replace(NAMESPACE, model_revision="fixture-model-rev-2"),
    )
    changed_encoder = build_qwen3_tts_prompt_cache_identity(
        _xvector_request(),
        CONFIG,
        namespace=replace(
            NAMESPACE, reference_encoder_identity="fixture-reference-encoder-v2"
        ),
    )

    assert identity.lcp_length(changed_model) == 0
    assert identity.lcp_length(changed_encoder) == 0


def test_reference_source_key_is_stable_and_content_based(tmp_path) -> None:
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    other_path = tmp_path / "other.wav"
    first_path.write_bytes(b"same reference bytes")
    second_path.write_bytes(b"same reference bytes")
    other_path.write_bytes(b"different reference bytes")

    first_key = qwen3_tts_reference_source_key(str(first_path))
    second_key = qwen3_tts_reference_source_key(str(second_path))
    assert first_key is not None
    assert second_key is not None
    assert first_key == second_key
    assert _identity(_xvector_request(reference_audio_key=first_key)) == _identity(
        _xvector_request(reference_audio_key=second_key)
    )
    assert qwen3_tts_reference_source_key(
        str(first_path)
    ) != qwen3_tts_reference_source_key(str(other_path))
    assert qwen3_tts_reference_source_key(b"same reference bytes") == (
        qwen3_tts_reference_source_key(b"same reference bytes")
    )
    assert qwen3_tts_reference_source_key("https://example.invalid/ref.wav") is None
    assert qwen3_tts_reference_source_key({"data": "abc", "media_type": "audio/wav"})


def test_uploaded_voice_identity_uses_existing_lifecycle_key_and_artifact() -> None:
    from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
    from sglang_omni.models.qwen3_tts.request_builders import (
        _cacheable_qwen3_tts_voice_prompt,
        _qwen3_tts_uploaded_voice_cache_key,
        _qwen3_tts_voice_prompt_from_cache,
    )

    previous = Qwen3TTSState(
        uploaded_voice_name="guide",
        uploaded_voice_created_at=1,
        x_vector_only_mode=True,
    )
    replacement = replace(previous, uploaded_voice_created_at=2)
    previous_key = _qwen3_tts_uploaded_voice_cache_key(previous)
    replacement_key = _qwen3_tts_uploaded_voice_cache_key(replacement)

    assert previous_key is not None
    assert replacement_key is not None
    assert previous_key != replacement_key
    assert previous_key.model_type == "qwen3_tts_xvec"
    assert previous_key.voice_name == "guide"
    assert previous_key.artifact_kind == "voice_clone_prompt"

    prompt = {
        "ref_spk_embedding": [torch.arange(8, dtype=torch.float32)],
        "icl_mode": [False],
    }
    artifact = _cacheable_qwen3_tts_voice_prompt(prompt, ref_text=None)
    restored_prompt, restored_text = _qwen3_tts_voice_prompt_from_cache(artifact)

    assert restored_text is None
    assert torch.equal(
        restored_prompt["ref_spk_embedding"][0], prompt["ref_spk_embedding"][0]
    )
    assert restored_prompt["icl_mode"] == [False]


def test_semantic_identity_does_not_include_generation_or_weight_state() -> None:
    identity = _identity(_xvector_request())

    assert "weight" not in identity.namespace.to_key()
    assert identity == _identity(_xvector_request())

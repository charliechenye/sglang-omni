"""CPU-only contract tests for the Qwen3-TTS semantic prompt identity."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.qwen3_tts import request_builders
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.prompt_cache_identity import (
    PromptCacheNamespace,
    PromptRowDescriptor,
    Qwen3TTSPromptConfig,
    Qwen3TTSPromptRequest,
    build_qwen3_tts_prompt_cache_identity,
    qwen3_tts_reference_source_key,
)
from sglang_omni.models.qwen3_tts.request_builders import build_embedding_cache_key_ids
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.speaker_cache import (
    SpeakerCacheKey,
    get_speaker_artifact_cache,
)

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
        self._feedback_buffer = torch.empty((1, HIDDEN_SIZE))
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

    def __init__(self, model_type: str = "base") -> None:
        # Bypass the production constructor: it allocates the full transformer,
        # speaker encoder, predictor buffers, and CUDA-specific runtime state.
        nn.Module.__init__(self)
        self.tts_model_type = model_type
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


class _Phase2Processor:
    """Small processor double matching Qwen3TTSModel's processor call shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *, text, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        assert len(text) == 1
        source = text[0]
        self.calls.append(tuple(text))
        if source.startswith("assistant:"):
            ids = [*ROLE_IDS, *TARGET_IDS, 900, 901, 902, 903, 904]
        elif source.startswith("ref:"):
            ids = [*ROLE_IDS, *REFERENCE_TEXT_IDS, 703, 704]
        elif source.startswith("instruct:"):
            ids = [500, 501]
        else:
            raise AssertionError(f"unexpected processor input: {source!r}")
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _Phase2Wrapper:
    def __init__(self) -> None:
        self.processor = _Phase2Processor()

    def _tokenize_texts(self, texts):
        raise AssertionError("production should use processor before device transfer")

    def _build_assistant_text(self, text):
        return f"assistant:{text}"

    def _build_ref_text(self, text):
        return f"ref:{text}"

    def _build_instruct_text(self, text):
        return f"instruct:{text}"

    def _merge_generate_kwargs(self, **kwargs):
        return kwargs


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


def _build_real_prompt_with_semantic_rows(
    request: Qwen3TTSPromptRequest,
) -> tuple[torch.Tensor, tuple[PromptRowDescriptor, ...]]:
    talker = _TinyQwen3TTSTalker(
        model_type={
            "Base": "base",
            "CustomVoice": "custom_voice",
            "VoiceDesign": "voice_design",
        }[request.task_type]
    )
    input_id = _tokenized_text(request.target_text_ids, (900, 901, 902, 903, 904))
    instruct_id = (
        torch.tensor([list(request.instruction_text_ids)], dtype=torch.long)
        if request.instruction_text_ids
        else None
    )
    reference_identity = (
        "adhoc_reference",
        "qwen3_tts",
        "qwen3_tts_voice_clone_prompt",
        "qwen3_tts_voice_clone_prompt_adhoc",
        request.reference_audio_key or "",
        "fixture-model",
        "fixture-encoder",
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
            result = talker.build_voice_clone_inputs(
                input_id=input_id,
                ref_id=ref_id,
                voice_clone_prompt=voice_clone_prompt,
                language=request.language,
                non_streaming_mode=request.non_streaming_mode,
                instruct_id=instruct_id,
                semantic_reference_identity=reference_identity,
                return_semantic_prompt_rows=True,
            )
        elif request.task_type == "CustomVoice":
            assert request.voice is not None
            result = talker.build_custom_voice_inputs(
                input_id=input_id,
                voice=request.voice,
                language=request.language,
                non_streaming_mode=request.non_streaming_mode,
                instruct_id=instruct_id,
                return_semantic_prompt_rows=True,
            )
        else:
            result = talker.build_voice_design_inputs(
                input_id=input_id,
                language=request.language,
                non_streaming_mode=request.non_streaming_mode,
                instruct_id=instruct_id,
                return_semantic_prompt_rows=True,
            )
    prompt_embeddings, _, _, _, semantic_rows = result
    assert semantic_rows is not None
    return prompt_embeddings.squeeze(0).detach(), semantic_rows


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


def _phase2_payload(
    *,
    inputs,
    request_id: str = "phase2-request",
    params: dict | None = None,
    tts_params: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def _put_phase2_uploaded_prompt(*, x_vector_only_mode: bool) -> None:
    cache = get_speaker_artifact_cache()
    cache.put(
        SpeakerCacheKey(
            model_type=("qwen3_tts_xvec" if x_vector_only_mode else "qwen3_tts_icl"),
            voice_name="guide",
            voice_version=7,
            artifact_kind="voice_clone_prompt",
        ),
        {
            "artifact_type": "qwen3_tts_voice_clone_prompt",
            "ref_spk_embedding": [_SPEAKER_EMBEDS[REFERENCE_AUDIO_KEY].clone()],
            "icl_mode": [not x_vector_only_mode],
            **(
                {
                    "ref_code": [torch.tensor(REFERENCE_CODE_FRAMES, dtype=torch.long)],
                    "ref_text": "reference",
                }
                if not x_vector_only_mode
                else {}
            ),
        },
    )


def _phase2_base_inputs(*, x_vector_only_mode: bool) -> dict:
    reference = {"audio_path": "voice.wav"}
    if not x_vector_only_mode:
        reference["text"] = "reference"
    return {"text": "target", "references": [reference]}


def test_phase2_real_preparation_emits_rows_for_all_task_paths() -> None:
    cache = get_speaker_artifact_cache()
    cache.clear()
    _put_phase2_uploaded_prompt(x_vector_only_mode=False)
    _put_phase2_uploaded_prompt(x_vector_only_mode=True)
    try:
        cases = (
            (
                _phase2_base_inputs(x_vector_only_mode=False),
                {
                    "task_type": "Base",
                    "uploaded_voice_name": "guide",
                    "uploaded_voice_created_at": 7,
                },
                "base",
            ),
            (
                _phase2_base_inputs(x_vector_only_mode=True),
                {
                    "task_type": "Base",
                    "x_vector_only_mode": True,
                    "uploaded_voice_name": "guide",
                    "uploaded_voice_created_at": 7,
                },
                "base",
            ),
            (
                "target",
                {
                    "task_type": "CustomVoice",
                    "voice": "Ryan",
                    "instructions": "calm",
                },
                "custom_voice",
            ),
            (
                "target",
                {
                    "task_type": "VoiceDesign",
                    "instructions": "A warm adult voice.",
                },
                "voice_design",
            ),
        )
        for inputs, tts_params, model_type in cases:
            prepared = request_builders._prepare_qwen3_tts_request(
                _phase2_payload(inputs=inputs, tts_params=tts_params),
                model=_TinyQwen3TTSTalker(model_type=model_type),
                wrapper=_Phase2Wrapper(),
            )
            assert prepared.semantic_prompt_rows
            assert (
                len(prepared.semantic_prompt_rows)
                == prepared.prompt_input_embeds.shape[0]
            )
    finally:
        cache.clear()


def test_phase2_provenance_cross_checks_phase1_at_prompt_boundaries() -> None:
    cases = (
        (_base_request(), _base_request(target_text_ids=(999, 201, 202))),
        (_base_request(), _base_request(target_text_ids=(200, 999, 202))),
        (
            _base_request(reference_code_frames=((400, 401, 402),)),
            _base_request(),
        ),
        (_xvector_request(), _xvector_request(target_text_ids=(999, 201, 202))),
        (_custom_request(language="en"), _custom_request(language="en", voice="Alice")),
        (
            _design_request(instruction_text_ids=(500, 501)),
            _design_request(instruction_text_ids=(500, 999)),
        ),
    )
    for left, right in cases:
        _, left_rows = _build_real_prompt_with_semantic_rows(left)
        _, right_rows = _build_real_prompt_with_semantic_rows(right)
        production_lcp = 0
        for production_left, production_right in zip(left_rows, right_rows):
            if production_left != production_right:
                break
            production_lcp += 1
        assert production_lcp == _identity(left).lcp_length(_identity(right))


def test_phase2_generation_and_output_fields_are_prompt_neutral() -> None:
    baseline_params = {
        "task_type": "CustomVoice",
        "voice": "Ryan",
        "instructions": "calm",
    }
    baseline = request_builders._prepare_qwen3_tts_request(
        _phase2_payload(inputs="target", tts_params=baseline_params),
        model=_TinyQwen3TTSTalker(model_type="custom_voice"),
        wrapper=_Phase2Wrapper(),
    ).semantic_prompt_rows
    assert baseline

    variants = (
        {"request_id": "different-request"},
        {"tts_params": {"seed": 123}},
        {"params": {"do_sample": False}},
        {"params": {"temperature": 0.37}},
        {"params": {"top_p": 0.63}},
        {"params": {"top_k": 17}},
        {"params": {"repetition_penalty": 1.23}},
        {"params": {"max_new_tokens": 77}},
        {"params": {"subtalker_dosample": False}},
        {"params": {"subtalker_temperature": 0.41}},
        {"params": {"subtalker_top_p": 0.67}},
        {"params": {"subtalker_top_k": 13}},
        {"params": {"stream_codec_output": False}},
    )
    for variant in variants:
        tts_params = dict(baseline_params)
        tts_params.update(variant.get("tts_params", {}))
        prepared = request_builders._prepare_qwen3_tts_request(
            _phase2_payload(
                inputs="target",
                request_id=variant.get("request_id", "phase2-request"),
                params=variant.get("params"),
                tts_params=tts_params,
            ),
            model=_TinyQwen3TTSTalker(model_type="custom_voice"),
            wrapper=_Phase2Wrapper(),
        )
        assert prepared.semantic_prompt_rows == baseline


def test_phase2_stream_codec_output_does_not_split_base_icl_provenance() -> None:
    cache = get_speaker_artifact_cache()
    cache.clear()
    _put_phase2_uploaded_prompt(x_vector_only_mode=False)
    try:
        rows = []
        for stream_codec_output in (True, False):
            prepared = request_builders._prepare_qwen3_tts_request(
                _phase2_payload(
                    inputs=_phase2_base_inputs(x_vector_only_mode=False),
                    tts_params={
                        "task_type": "Base",
                        "uploaded_voice_name": "guide",
                        "uploaded_voice_created_at": 7,
                        "stream_codec_output": stream_codec_output,
                    },
                ),
                model=_TinyQwen3TTSTalker(model_type="base"),
                wrapper=_Phase2Wrapper(),
            )
            assert prepared.semantic_prompt_rows
            rows.append(prepared.semantic_prompt_rows)
        assert rows[0] == rows[1]
    finally:
        cache.clear()


def test_phase2_non_streaming_mode_changes_base_prompt_rows() -> None:
    cache = get_speaker_artifact_cache()
    cache.clear()
    _put_phase2_uploaded_prompt(x_vector_only_mode=True)
    try:
        rows = []
        for non_streaming_mode in (False, True):
            prepared = request_builders._prepare_qwen3_tts_request(
                _phase2_payload(
                    inputs=_phase2_base_inputs(x_vector_only_mode=True),
                    tts_params={
                        "task_type": "Base",
                        "x_vector_only_mode": True,
                        "uploaded_voice_name": "guide",
                        "uploaded_voice_created_at": 7,
                        "non_streaming_mode": non_streaming_mode,
                    },
                ),
                model=_TinyQwen3TTSTalker(model_type="base"),
                wrapper=_Phase2Wrapper(),
            )
            assert prepared.semantic_prompt_rows
            rows.append(prepared.semantic_prompt_rows)
        assert rows[0] != rows[1]
    finally:
        cache.clear()


def test_phase2_reference_identity_fails_closed_for_unsafe_sources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeReferenceService:
        def get_or_encode(self, state, *, desc):
            del desc
            return (
                {
                    "ref_code": [None],
                    "ref_spk_embedding": [_SPEAKER_EMBEDS[REFERENCE_AUDIO_KEY].clone()],
                    "icl_mode": [False],
                },
                state.ref_text,
            )

    monkeypatch.setattr(
        request_builders,
        "_get_qwen3_tts_adhoc_reference_service",
        lambda model, wrapper: FakeReferenceService(),
    )

    def prepare(ref_audio: str) -> request_builders.Qwen3TTSPreparedRequest:
        return request_builders._prepare_qwen3_tts_request(
            _phase2_payload(
                inputs={
                    "text": "target",
                    "references": [{"audio_path": ref_audio, "text": "reference"}],
                },
                tts_params={
                    "task_type": "Base",
                    "x_vector_only_mode": True,
                },
            ),
            model=_TinyQwen3TTSTalker(model_type="base"),
            wrapper=_Phase2Wrapper(),
        )

    unsafe = prepare("https://example.invalid/reference.wav")
    assert unsafe.semantic_prompt_rows is None
    assert unsafe.input_ids_list == build_embedding_cache_key_ids(
        unsafe.prompt_input_embeds
    )

    safe_path = tmp_path / "reference.wav"
    safe_path.write_bytes(b"content-addressed reference")
    safe = prepare(str(safe_path))
    assert safe.semantic_prompt_rows
    assert safe.input_ids_list == build_embedding_cache_key_ids(
        safe.prompt_input_embeds
    )


def test_phase2_reference_identity_excludes_prompt_options() -> None:
    first = request_builders._qwen3_tts_semantic_reference_identity(
        Qwen3TTSState(
            ref_audio=b"reference",
            ref_text="first transcript",
            x_vector_only_mode=False,
        ),
        model=SimpleNamespace(),
        wrapper=SimpleNamespace(),
    )
    second = request_builders._qwen3_tts_semantic_reference_identity(
        Qwen3TTSState(
            ref_audio=b"reference",
            ref_text="different transcript",
            x_vector_only_mode=True,
        ),
        model=SimpleNamespace(),
        wrapper=SimpleNamespace(),
    )

    assert first == second
    assert first is not None
    assert "first transcript" not in first
    assert "different transcript" not in first


def test_phase2_reference_code_semantics_use_shape_only() -> None:
    from sglang_omni.models.qwen3_tts.sglang_model import (
        _semantic_reference_frame_count,
    )

    class ShapeOnlyReferenceCode:
        ndim = 2
        shape = (3, 3)

        def cpu(self):
            raise AssertionError("semantic identity must not call ref_code.cpu()")

        def tolist(self):
            raise AssertionError("semantic identity must not call ref_code.tolist()")

        def numpy(self):
            raise AssertionError("semantic identity must not call ref_code.numpy()")

    assert _semantic_reference_frame_count(ShapeOnlyReferenceCode()) == 3

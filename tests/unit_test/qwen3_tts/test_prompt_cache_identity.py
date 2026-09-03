"""CPU-only contract tests for the Qwen3-TTS semantic prompt identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from sglang_omni.models.qwen3_tts.prompt_cache_identity import (
    PromptCacheNamespace,
    Qwen3TTSPromptConfig,
    Qwen3TTSPromptRequest,
    build_qwen3_tts_prompt_cache_identity,
    qwen3_tts_reference_source_key,
)
from sglang_omni.models.qwen3_tts.request_builders import build_embedding_cache_key_ids

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


def _fixture_row_embeddings(identity) -> torch.Tensor:
    """Make deterministic additive CPU rows for the existing hash oracle.

    Each semantic component occupies a stable slot.  This is intentionally a
    test fixture, not a second production embedding implementation.
    """

    rows: list[torch.Tensor] = []
    for row in identity.rows:
        vector = torch.zeros(36, dtype=torch.float32)
        for component in row.components:
            encoded = json.dumps(
                component.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
            digest = hashlib.blake2b(encoded, digest_size=16).digest()
            values = [
                float(int.from_bytes(digest[offset : offset + 4], "little") % 100_000)
                for offset in range(0, 16, 4)
            ]
            if component.kind == "text_embedding":
                slot = 0
            elif component.kind == "codec_embedding":
                slot = 4 + int(component.identity[0]) * 4
            elif component.kind == "speaker_embedding":
                slot = 28
            else:  # pragma: no cover - protects the fixture from silent drift.
                raise AssertionError(f"unknown component kind: {component.kind}")
            vector[slot : slot + 4] += torch.tensor(values, dtype=torch.float32)
        rows.append(vector)
    return torch.stack(rows)


def _oracle_lcp(left, right) -> int:
    left_ids = build_embedding_cache_key_ids(_fixture_row_embeddings(left))
    right_ids = build_embedding_cache_key_ids(_fixture_row_embeddings(right))
    length = 0
    for left_id, right_id in zip(left_ids, right_ids):
        if left_id != right_id:
            break
        length += 1
    return length


def _assert_lcp_matches_oracle(left_request, right_request) -> None:
    left = _identity(left_request)
    right = _identity(right_request)
    assert left.lcp_length(right) == _oracle_lcp(left, right)


def test_all_task_variants_have_stable_semantic_rows() -> None:
    requests = [
        _base_request(),
        _xvector_request(),
        _custom_request(),
        _design_request(),
    ]

    for request in requests:
        first = _identity(request)
        second = _identity(request)
        assert first == second
        assert first.lcp_length(second) == len(first.rows)
        assert first.to_key() == second.to_key()
        assert first.rows


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(
            _xvector_request(),
            _xvector_request(target_text_ids=(200, 201, 999)),
            id="streaming-generated-tail-is-not-prompt-identity",
        ),
        pytest.param(
            _xvector_request(),
            _xvector_request(target_text_ids=(999, 201, 202)),
            id="streaming-first-target-token-is-prompt-identity",
        ),
        pytest.param(
            _base_request(),
            _base_request(reference_text_ids=(999, 301)),
            id="icl-reference-text",
        ),
        pytest.param(
            _base_request(),
            _base_request(reference_code_frames=((999, 401, 402), (410, 411, 412))),
            id="icl-reference-code",
        ),
        pytest.param(
            _base_request(),
            _base_request(target_text_ids=(200, 999, 202), non_streaming_mode=True),
            id="nonstreaming-target-text",
        ),
        pytest.param(
            _custom_request(instruction_text_ids=(500, 999)),
            _custom_request(instruction_text_ids=(500, 501)),
            id="instruction-token-prefix",
        ),
        pytest.param(
            _custom_request(language="en"),
            _custom_request(language="zh"),
            id="explicit-language",
        ),
        pytest.param(
            _custom_request(language="en"),
            _custom_request(language="en", voice="Alice"),
            id="custom-speaker",
        ),
        pytest.param(
            _xvector_request(language="auto"),
            _xvector_request(language="en"),
            id="base-language",
        ),
        pytest.param(
            _xvector_request(),
            _xvector_request(reference_audio_key="bytes:other-reference"),
            id="reference-audio-source",
        ),
    ],
)
def test_semantic_lcp_matches_old_embedding_hash_oracle(left, right) -> None:
    _assert_lcp_matches_oracle(left, right)


def test_streaming_boundary_ignores_text_after_codec_prefix() -> None:
    short_reference = _base_request(
        reference_code_frames=((400, 401, 402),),
        target_text_ids=(200, 201, 202),
    )
    changed_after_boundary = replace(short_reference, target_text_ids=(200, 201, 999))

    left = _identity(short_reference)
    right = _identity(changed_after_boundary)
    assert left == right
    assert _oracle_lcp(left, right) == len(left.rows)


def test_streaming_first_target_change_stops_at_conditioned_prefix() -> None:
    unchanged = _identity(_xvector_request())
    changed_first_token = _identity(_xvector_request(target_text_ids=(999, 201, 202)))
    conditioned_length = len(unchanged.rows) - 1

    assert unchanged.lcp_length(changed_first_token) == conditioned_length
    assert _oracle_lcp(unchanged, changed_first_token) == conditioned_length


def test_icl_mode_and_reference_text_code_keep_partial_prefix_reuse() -> None:
    icl = _identity(_base_request())
    xvector = _identity(_xvector_request())
    changed_reference_text = _identity(_base_request(reference_text_ids=(999, 301)))
    changed_reference_code = _identity(
        _base_request(reference_code_frames=((999, 401, 402), (410, 411, 412)))
    )
    conditioned_length = len(xvector.rows) - 1

    assert icl.lcp_length(xvector) == conditioned_length
    assert icl.lcp_length(changed_reference_text) == conditioned_length
    assert icl.lcp_length(changed_reference_code) == conditioned_length + 1
    assert _oracle_lcp(icl, xvector) == conditioned_length
    assert _oracle_lcp(icl, changed_reference_text) == conditioned_length
    assert _oracle_lcp(icl, changed_reference_code) == conditioned_length + 1


def test_icl_reference_code_length_moves_streaming_text_boundary() -> None:
    one_frame = _identity(_base_request(reference_code_frames=((400, 401, 402),)))
    two_frames = _identity(_base_request())

    # The one-frame prompt has two codec rows (BOS + frame), so the two
    # reference-text rows are visible and target text starts only in the
    # two-frame prompt.
    assert one_frame.is_prefix_of(two_frames)
    assert one_frame.lcp_length(two_frames) == len(one_frame.rows)
    assert _oracle_lcp(one_frame, two_frames) == len(one_frame.rows)


def test_streaming_and_nonstreaming_share_only_conditioned_prefix() -> None:
    streaming = _identity(_xvector_request(non_streaming_mode=False))
    non_streaming = _identity(_xvector_request(non_streaming_mode=True))

    assert streaming.lcp_length(non_streaming) == len(streaming.rows) - 1
    assert _oracle_lcp(streaming, non_streaming) == len(streaming.rows) - 1


def test_custom_auto_language_uses_speaker_dialect() -> None:
    auto = _identity(_custom_request(language="auto", voice="Ryan"))
    explicit_dialect = _identity(_custom_request(language="dialect", voice="Ryan"))
    explicit_other_language = _identity(_custom_request(language="en", voice="Ryan"))

    assert auto == explicit_dialect
    assert auto.lcp_length(explicit_other_language) == 5
    assert _oracle_lcp(auto, explicit_other_language) == 5


def test_auto_language_depends_on_custom_voice_dialect() -> None:
    dialect_voice = _identity(_custom_request(language="auto", voice="Ryan"))
    voice_without_dialect = _identity(_custom_request(language="auto", voice="Alice"))

    assert dialect_voice.lcp_length(voice_without_dialect) == 3
    assert _oracle_lcp(dialect_voice, voice_without_dialect) == 3


def test_instructions_are_prepended_and_cannot_false_share() -> None:
    without_instruction = _identity(_custom_request())
    with_instruction = _identity(_custom_request(instruction_text_ids=(500,)))

    assert without_instruction.lcp_length(with_instruction) == 0
    assert _oracle_lcp(without_instruction, with_instruction) == 0


def test_generation_request_id_generated_tail_and_weight_epoch_are_excluded() -> None:
    request = _xvector_request()
    identity = _identity(request)

    generation_variants = [
        {"temperature": 0.2, "top_p": 0.7},
        {"do_sample": False, "max_new_tokens": 8},
    ]
    for generation_kwargs in generation_variants:
        del generation_kwargs
        assert _identity(request) == identity

    for request_id, generated_tail, weight_epoch in [
        ("request-a", (1, 2), 0),
        ("request-b", (99, 100), 7),
    ]:
        del request_id, generated_tail, weight_epoch
        assert _identity(request).lcp_length(identity) == len(identity.rows)

    assert "weight" not in identity.namespace.to_key()


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


def test_uploaded_voice_version_is_not_a_row_component_by_itself() -> None:
    """Artifact versioning belongs to speaker-artifact lifecycle, not rows."""

    same_materialized_audio = _xvector_request(reference_audio_key="data:same-bytes")
    version_one = _identity(same_materialized_audio)
    version_two = _identity(same_materialized_audio)

    assert version_one == version_two
    assert version_one.lcp_length(version_two) == len(version_one.rows)


def test_uploaded_voice_replacement_changes_materialized_reference_identity() -> None:
    # The uploaded voice name/version selects the speaker artifact.  The
    # semantic prompt receives the resulting stable materialized source key.
    previous_upload_version = 1
    replacement_upload_version = 2
    assert previous_upload_version != replacement_upload_version
    previous_upload = _identity(
        _xvector_request(reference_audio_key="data:previous-upload")
    )
    replacement_upload = _identity(
        _xvector_request(reference_audio_key="data:replacement-upload")
    )

    assert previous_upload.lcp_length(replacement_upload) < len(previous_upload.rows)
    assert _oracle_lcp(previous_upload, replacement_upload) == (
        previous_upload.lcp_length(replacement_upload)
    )

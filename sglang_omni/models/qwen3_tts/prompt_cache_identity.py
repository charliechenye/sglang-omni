"""Semantic prompt identity for the Qwen3-TTS prompt-cache contract.

This module describes the rows that can be reused *before* they are turned
into model embeddings.  It is deliberately independent of a model instance,
CUDA, and generated feedback.  The production prompt-cache key remains the
projected-embedding oracle in ``request_builders`` until a later phase wires
this representation into scheduling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

SEMANTIC_PROMPT_IDENTITY_SCHEMA = "qwen3_tts.semantic_prompt.v1"

_TASK_ALIASES = {
    "base": "Base",
    "customvoice": "CustomVoice",
    "voicedesign": "VoiceDesign",
}


def _stable_json(value: Any) -> str:
    """Serialize identity data without process-randomized Python hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _token_tuple(values: Sequence[int] | None, *, field_name: str) -> tuple[int, ...]:
    if values is None:
        return ()
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain integer token IDs") from exc
    return result


def _mapping_tuple(
    values: Mapping[str, int] | Sequence[tuple[str, int]], *, field_name: str
) -> tuple[tuple[str, int], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized: dict[str, int] = {}
    try:
        for key, value in items:
            normalized[str(key).strip().lower()] = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must map strings to integer IDs") from exc
    return tuple(sorted(normalized.items()))


def _lookup(mapping: tuple[tuple[str, int], ...], key: str) -> int | None:
    return dict(mapping).get(key)


@dataclass(frozen=True, slots=True)
class PromptCacheNamespace:
    """Stable namespace inputs that are not represented by prompt rows.

    The namespace intentionally contains model and reference-encoder identity,
    but not request IDs, generation settings, generated output, or weight
    epochs.  Weight epochs remain scheduler-owned invalidation state.
    """

    model_revision: str
    model_config_identity: str
    reference_encoder_identity: str
    schema_version: str = SEMANTIC_PROMPT_IDENTITY_SCHEMA
    model_family: str = "qwen3_tts"

    def __post_init__(self) -> None:
        for field_name in (
            "schema_version",
            "model_family",
            "model_revision",
            "model_config_identity",
            "reference_encoder_identity",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "model_config_identity": self.model_config_identity,
            "reference_encoder_identity": self.reference_encoder_identity,
        }

    def to_key(self) -> str:
        return _stable_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class PromptRowComponent:
    """One additive semantic contribution to a prompt row."""

    kind: str
    identity: tuple[int | str | bool | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("prompt row component kind must be non-empty")
        identity = tuple(self.identity)
        if not all(
            value is None or isinstance(value, (int, str, bool)) for value in identity
        ):
            raise TypeError("prompt row component identity must be scalar")
        object.__setattr__(self, "identity", identity)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "identity": list(self.identity)}


@dataclass(frozen=True, slots=True)
class PromptRowDescriptor:
    """An ordered row made from one or more additive components."""

    components: tuple[PromptRowComponent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        if not self.components:
            raise ValueError("a prompt row must contain at least one component")

    def to_dict(self) -> list[dict[str, Any]]:
        return [component.to_dict() for component in self.components]

    def to_key(self) -> str:
        return _stable_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class PromptCacheIdentity:
    """Semantic prompt rows plus their model/encoder namespace."""

    namespace: PromptCacheNamespace
    rows: tuple[PromptRowDescriptor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))

    def same_namespace(self, other: "PromptCacheIdentity") -> bool:
        return self.namespace == other.namespace

    def lcp_length(self, other: "PromptCacheIdentity") -> int:
        """Return the reusable longest common prefix in semantic rows."""

        if not self.same_namespace(other):
            return 0
        length = 0
        for left, right in zip(self.rows, other.rows):
            if left != right:
                break
            length += 1
        return length

    def is_prefix_of(self, other: "PromptCacheIdentity") -> bool:
        return self.lcp_length(other) == len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_key(self) -> str:
        return _stable_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class Qwen3TTSPromptConfig:
    """Static token and speaker mappings needed to describe Qwen3-TTS rows."""

    tts_bos_token_id: int
    tts_eos_token_id: int
    tts_pad_token_id: int
    codec_bos_token_id: int
    codec_pad_token_id: int
    codec_think_token_id: int
    codec_nothink_token_id: int
    codec_think_bos_token_id: int
    codec_think_eos_token_id: int
    codec_language_ids: Mapping[str, int] | Sequence[tuple[str, int]] = ()
    speaker_ids: Mapping[str, int] | Sequence[tuple[str, int]] = ()
    speaker_dialects: Mapping[str, str] | Sequence[tuple[str, str]] = ()
    num_code_groups: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "codec_language_ids",
            _mapping_tuple(self.codec_language_ids, field_name="codec_language_ids"),
        )
        object.__setattr__(
            self,
            "speaker_ids",
            _mapping_tuple(self.speaker_ids, field_name="speaker_ids"),
        )
        dialect_items = (
            self.speaker_dialects.items()
            if isinstance(self.speaker_dialects, Mapping)
            else self.speaker_dialects
        )
        dialects: dict[str, str] = {}
        try:
            for key, value in dialect_items:
                dialects[str(key).strip().lower()] = str(value).strip().lower()
        except (TypeError, ValueError) as exc:
            raise TypeError("speaker_dialects must map strings to strings") from exc
        object.__setattr__(self, "speaker_dialects", tuple(sorted(dialects.items())))
        if self.num_code_groups is not None and self.num_code_groups <= 0:
            raise ValueError("num_code_groups must be positive")

    def resolve_language_id(
        self, language: str, *, voice: str | None = None
    ) -> int | None:
        language_ids = cast(tuple[tuple[str, int], ...], self.codec_language_ids)
        language_key = str(language).strip().lower()
        if language_key != "auto":
            language_id = _lookup(language_ids, language_key)
            if language_id is None:
                raise ValueError(f"unknown Qwen3-TTS language: {language}")
            return language_id
        if voice is None:
            return None
        dialects = cast(tuple[tuple[str, str], ...], self.speaker_dialects)
        dialect = dict(dialects).get(voice.strip().lower())
        if dialect is None:
            return None
        return _lookup(language_ids, dialect)

    def resolve_speaker_id(self, voice: str) -> int:
        voice_key = voice.strip().lower()
        speaker_ids = cast(tuple[tuple[str, int], ...], self.speaker_ids)
        speaker_id = _lookup(speaker_ids, voice_key)
        if speaker_id is None:
            raise ValueError(f"unknown Qwen3-TTS speaker: {voice}")
        return speaker_id


@dataclass(frozen=True, slots=True)
class Qwen3TTSPromptRequest:
    """Model-visible prompt inputs, represented as token IDs and source keys.

    ``role_text_ids`` and ``target_text_ids`` correspond to the current
    builder's ``input_id[:, :3]`` and ``input_id[:, 3:-5]`` slices.  The
    ``reference_code_frames`` entries are ``[T, Q]`` semantic code rows; all
    quantizer components in one frame contribute to one prompt row.  The
    ``reference_audio_key`` must already be a stable source identity.  This
    request deliberately has no generation parameters, request ID, generated
    tail, or weight epoch: those values must not change semantic prompt
    identity.
    """

    task_type: str
    role_text_ids: Sequence[int]
    target_text_ids: Sequence[int]
    language: str = "auto"
    voice: str | None = None
    instruction_text_ids: Sequence[int] = ()
    reference_audio_key: str | None = None
    reference_text_ids: Sequence[int] = ()
    reference_code_frames: Sequence[Sequence[int]] = ()
    x_vector_only_mode: bool = False
    non_streaming_mode: bool = False

    def __post_init__(self) -> None:
        task_key = str(self.task_type).replace("_", "").replace("-", "").lower()
        try:
            canonical_task = _TASK_ALIASES[task_key]
        except KeyError as exc:
            raise ValueError(
                f"unsupported Qwen3-TTS task type: {self.task_type}"
            ) from exc
        object.__setattr__(self, "task_type", canonical_task)

        role_ids = _token_tuple(self.role_text_ids, field_name="role_text_ids")
        if len(role_ids) != 3:
            raise ValueError("role_text_ids must contain the three role tokens")
        object.__setattr__(self, "role_text_ids", role_ids)
        object.__setattr__(
            self,
            "target_text_ids",
            _token_tuple(self.target_text_ids, field_name="target_text_ids"),
        )
        object.__setattr__(
            self,
            "instruction_text_ids",
            _token_tuple(self.instruction_text_ids, field_name="instruction_text_ids"),
        )
        object.__setattr__(
            self,
            "reference_text_ids",
            _token_tuple(self.reference_text_ids, field_name="reference_text_ids"),
        )

        frames: list[tuple[int, ...]] = []
        try:
            for frame in self.reference_code_frames:
                frames.append(_token_tuple(frame, field_name="reference_code_frames"))
        except TypeError as exc:
            raise TypeError(
                "reference_code_frames must be a sequence of code frames"
            ) from exc
        object.__setattr__(self, "reference_code_frames", tuple(frames))
        if self.voice is not None:
            normalized_voice = self.voice.strip()
            object.__setattr__(self, "voice", normalized_voice or None)
        normalized_language = str(self.language).strip()
        object.__setattr__(self, "language", normalized_language or "auto")
        if canonical_task in {"CustomVoice", "VoiceDesign"}:
            object.__setattr__(self, "non_streaming_mode", True)


def qwen3_tts_reference_source_key(ref_audio: Any) -> str | None:
    """Build the same stable audio-source identity used by reference caching.

    Unresolved strings intentionally return ``None`` so a caller cannot turn a
    mutable remote URL into a reusable semantic prompt without first
    materializing it.  This mirrors the current Qwen3-TTS reference hook.
    """

    if isinstance(ref_audio, str):
        if ref_audio.startswith("data:"):
            return f"data:{hash_bytes(ref_audio.encode())}"
        return reference_path_cache_key(ref_audio, trust_stat=False)
    if isinstance(ref_audio, (bytes, bytearray, memoryview)):
        return f"bytes:{hash_bytes(ref_audio)}"
    if isinstance(ref_audio, dict):
        data_uri = audio_data_uri_from_reference(ref_audio)
        if data_uri is None:
            return None
        return f"data:{hash_bytes(data_uri.encode())}"
    return None


def _text_component(token_id: int) -> PromptRowComponent:
    return PromptRowComponent("text_embedding", (int(token_id),))


def _codec_component(group_index: int, token_id: int) -> PromptRowComponent:
    return PromptRowComponent("codec_embedding", (int(group_index), int(token_id)))


def _speaker_component(source_key: str) -> PromptRowComponent:
    return PromptRowComponent("speaker_embedding", (source_key,))


def _row(
    text_component: PromptRowComponent | None = None,
    codec_components: Sequence[PromptRowComponent] = (),
) -> PromptRowDescriptor:
    components: list[PromptRowComponent] = []
    if text_component is not None:
        components.append(text_component)
    components.extend(codec_components)
    return PromptRowDescriptor(tuple(components))


def _codec_prefill_components(
    request: Qwen3TTSPromptRequest, config: Qwen3TTSPromptConfig
) -> list[PromptRowComponent]:
    language_id = config.resolve_language_id(
        request.language,
        voice=request.voice if request.task_type == "CustomVoice" else None,
    )
    if language_id is None:
        codec_ids = (
            config.codec_nothink_token_id,
            config.codec_think_bos_token_id,
            config.codec_think_eos_token_id,
        )
    else:
        codec_ids = (
            config.codec_think_token_id,
            config.codec_think_bos_token_id,
            language_id,
            config.codec_think_eos_token_id,
        )
    return [_codec_component(0, token_id) for token_id in codec_ids]


def _conditioned_rows_with_config(
    request: Qwen3TTSPromptRequest,
    codec_prefix: Sequence[PromptRowComponent],
    config: Qwen3TTSPromptConfig,
) -> list[PromptRowDescriptor]:
    rows = [_row(_text_component(token_id)) for token_id in request.role_text_ids]
    for index, codec_component in enumerate(codec_prefix[:-1]):
        text_id = (
            config.tts_pad_token_id
            if index < len(codec_prefix) - 2
            else config.tts_bos_token_id
        )
        rows.append(_row(_text_component(text_id), (codec_component,)))
    return rows


def _non_icl_rows(
    request: Qwen3TTSPromptRequest, config: Qwen3TTSPromptConfig
) -> list[PromptRowDescriptor]:
    if not request.target_text_ids:
        raise ValueError("target_text_ids must not be empty")
    if request.non_streaming_mode:
        rows = [
            _row(
                _text_component(token_id),
                (_codec_component(0, config.codec_pad_token_id),),
            )
            for token_id in request.target_text_ids
        ]
        rows.append(
            _row(
                _text_component(config.tts_eos_token_id),
                (_codec_component(0, config.codec_pad_token_id),),
            )
        )
        rows.append(
            _row(
                _text_component(config.tts_pad_token_id),
                (_codec_component(0, config.codec_bos_token_id),),
            )
        )
        return rows
    return [
        _row(
            _text_component(request.target_text_ids[0]),
            (_codec_component(0, config.codec_bos_token_id),),
        )
    ]


def _icl_rows(
    request: Qwen3TTSPromptRequest, config: Qwen3TTSPromptConfig
) -> list[PromptRowDescriptor]:
    if not request.reference_text_ids:
        raise ValueError("ICL mode requires reference_text_ids")
    if not request.reference_code_frames:
        raise ValueError("ICL mode requires reference_code_frames")
    if config.num_code_groups is not None:
        for frame in request.reference_code_frames:
            if len(frame) != config.num_code_groups:
                raise ValueError("each reference code frame must match num_code_groups")

    text_components = [
        _text_component(token_id)
        for token_id in (
            *request.reference_text_ids,
            *request.target_text_ids,
            config.tts_eos_token_id,
        )
    ]
    codec_rows: list[tuple[PromptRowComponent, ...]] = [
        (_codec_component(0, config.codec_bos_token_id),)
    ]
    for frame in request.reference_code_frames:
        codec_rows.append(
            tuple(
                _codec_component(group_index, token_id)
                for group_index, token_id in enumerate(frame)
            )
        )

    if not request.non_streaming_mode:
        rows: list[PromptRowDescriptor] = []
        overlap_length = min(len(text_components), len(codec_rows))
        rows.extend(
            _row(text_components[index], codec_rows[index])
            for index in range(overlap_length)
        )
        rows.extend(
            _row(_text_component(config.tts_pad_token_id), codec_row)
            for codec_row in codec_rows[overlap_length:]
        )
        return rows

    rows = [
        _row(
            text_component,
            (_codec_component(0, config.codec_pad_token_id),),
        )
        for text_component in text_components
    ]
    rows.extend(
        _row(_text_component(config.tts_pad_token_id), codec_row)
        for codec_row in codec_rows
    )
    return rows


def build_qwen3_tts_prompt_cache_identity(
    request: Qwen3TTSPromptRequest,
    config: Qwen3TTSPromptConfig,
    *,
    namespace: PromptCacheNamespace,
) -> PromptCacheIdentity:
    """Build the semantic row sequence for one Qwen3-TTS prompt variant.

    Rows are emitted in the exact order used by the current model builder.
    Instruction rows are prepended last, matching the current instruction
    embedding concatenation.  Streaming-only target text after the initial
    row is intentionally omitted because it is returned as generated-tail
    state rather than part of the reusable prompt.
    """

    if request.task_type == "Base":
        if not request.reference_audio_key:
            raise ValueError("Base mode requires reference_audio_key")
        codec_prefix = _codec_prefill_components(request, config)
        codec_prefix.append(_speaker_component(request.reference_audio_key))
        codec_prefix.extend(
            (
                _codec_component(0, config.codec_pad_token_id),
                _codec_component(0, config.codec_bos_token_id),
            )
        )
        conditioned_rows = _conditioned_rows_with_config(request, codec_prefix, config)
        if request.x_vector_only_mode:
            prompt_rows = conditioned_rows + _non_icl_rows(request, config)
        else:
            prompt_rows = conditioned_rows + _icl_rows(request, config)
    elif request.task_type == "CustomVoice":
        if request.voice is None:
            raise ValueError("CustomVoice mode requires voice")
        if request.x_vector_only_mode:
            raise ValueError("CustomVoice mode does not accept x_vector_only_mode")
        if (
            request.reference_audio_key is not None
            or request.reference_text_ids
            or request.reference_code_frames
        ):
            raise ValueError("CustomVoice mode does not accept reference inputs")
        speaker_id = config.resolve_speaker_id(request.voice)
        codec_prefix = _codec_prefill_components(request, config)
        codec_prefix.extend(
            (
                _codec_component(0, speaker_id),
                _codec_component(0, config.codec_pad_token_id),
                _codec_component(0, config.codec_bos_token_id),
            )
        )
        prompt_rows = _conditioned_rows_with_config(request, codec_prefix, config)
        prompt_rows.extend(_non_icl_rows(request, config))
    elif request.task_type == "VoiceDesign":
        if request.x_vector_only_mode:
            raise ValueError("VoiceDesign mode does not accept x_vector_only_mode")
        if (
            request.reference_audio_key is not None
            or request.reference_text_ids
            or request.reference_code_frames
        ):
            raise ValueError("VoiceDesign mode does not accept reference inputs")
        if request.voice is not None:
            raise ValueError("VoiceDesign mode does not accept a speaker voice")
        if not request.instruction_text_ids:
            raise ValueError("VoiceDesign mode requires instruction_text_ids")
        codec_prefix = _codec_prefill_components(request, config)
        codec_prefix.extend(
            (
                _codec_component(0, config.codec_pad_token_id),
                _codec_component(0, config.codec_bos_token_id),
            )
        )
        prompt_rows = _conditioned_rows_with_config(request, codec_prefix, config)
        prompt_rows.extend(_non_icl_rows(request, config))
    else:  # pragma: no cover - Qwen3TTSPromptRequest canonicalizes task_type.
        raise AssertionError(f"unhandled task type: {request.task_type}")

    if request.instruction_text_ids:
        prompt_rows = [
            _row(_text_component(token_id)) for token_id in request.instruction_text_ids
        ] + prompt_rows

    return PromptCacheIdentity(namespace=namespace, rows=tuple(prompt_rows))


__all__ = [
    "SEMANTIC_PROMPT_IDENTITY_SCHEMA",
    "PromptCacheIdentity",
    "PromptCacheNamespace",
    "PromptRowComponent",
    "PromptRowDescriptor",
    "Qwen3TTSPromptConfig",
    "Qwen3TTSPromptRequest",
    "build_qwen3_tts_prompt_cache_identity",
    "qwen3_tts_reference_source_key",
]

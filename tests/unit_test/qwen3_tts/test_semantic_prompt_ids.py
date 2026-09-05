# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_tts import request_builders as qwen3_request_builders


def _voice_clone_prompt(
    *, ref_code: torch.Tensor | None = None, speaker_value: float = 1.0
) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "ref_spk_embedding": [torch.full((4,), speaker_value)],
        "icl_mode": [ref_code is not None],
    }
    if ref_code is not None:
        prompt["ref_code"] = [ref_code]
    return prompt


def _make_prompt_builder(*, model_type: str = "base") -> Any:
    from torch import nn

    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSPromptBuilderMixin

    class Embeddings(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codec_embedding = nn.Embedding(128, 4)
            self.text_embedding = nn.Embedding(128, 4)
            self.register_buffer(
                "_feedback_buffer", torch.zeros((1, 4)), persistent=False
            )

        def get_input_embeddings(self) -> nn.Embedding:
            return self.codec_embedding

        def get_text_embeddings(self) -> nn.Embedding:
            return self.text_embedding

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.codec_embedding = nn.ModuleList(
                [nn.Embedding(128, 4), nn.Embedding(128, 4)]
            )

    class Builder(Qwen3TTSPromptBuilderMixin, nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.root_config = SimpleNamespace(
                tts_bos_token_id=1,
                tts_eos_token_id=2,
                tts_pad_token_id=3,
            )
            self.config = SimpleNamespace(
                codec_bos_id=4,
                codec_pad_id=5,
                codec_nothink_id=6,
                codec_think_id=7,
                codec_think_bos_id=8,
                codec_think_eos_id=9,
                codec_language_id={"en": 10, "zh": 11},
                spk_id={"vivian": 20, "alice": 21},
                spk_is_dialect={"vivian": "en", "alice": "zh"},
                num_code_groups=3,
            )
            self.model = Embeddings()
            self.code_predictor = Predictor()
            self.text_projection = nn.Identity()
            self.tts_model_type = model_type

    return Builder()


def _semantic_input_id(*target_ids: int) -> torch.Tensor:
    return torch.tensor(
        [[10, 11, 12, *target_ids, 30, 31, 32, 33, 34]], dtype=torch.long
    )


def _semantic_ref_id(*reference_ids: int) -> torch.Tensor:
    return torch.tensor([[40, 41, 42, *reference_ids, 50, 51]], dtype=torch.long)


def _legacy_embedding_cache_key_ids(input_embeds: torch.Tensor) -> list[int]:
    """Frozen pre-semantic production oracle used only for LCP equivalence tests."""
    rows = input_embeds.detach().to(dtype=torch.float32, device="cpu")
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids


def _semantic_base_prompt(
    *, ref_code: torch.Tensor | None, speaker_value: float = 1.0
) -> dict[str, Any]:
    prompt = _voice_clone_prompt(ref_code=ref_code, speaker_value=speaker_value)
    artifact = qwen3_request_builders._cacheable_qwen3_tts_voice_prompt(
        prompt,
        ref_text="reference",
    )
    prompt["speaker_artifact_id"] = artifact["speaker_artifact_id"]
    prompt["ref_code_artifact_id"] = artifact.get("ref_code_artifact_id")
    return prompt


def _build_semantic_prompt(
    kind: str,
    *,
    builder: Any | None = None,
    target_ids: tuple[int, ...] = (20, 21, 22),
    reference_ids: tuple[int, ...] = (50, 51),
    ref_code: torch.Tensor | None = None,
    speaker_value: float = 1.0,
    ref_code_artifact_id: str | None = None,
    language: str = "auto",
    voice: str = "Vivian",
    instruction_ids: tuple[int, ...] = (),
    non_streaming: bool = False,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if builder is None:
        builder = _make_prompt_builder(
            model_type={"custom": "custom_voice", "design": "voice_design"}.get(
                kind, "base"
            )
        )
    input_id = _semantic_input_id(*target_ids)
    instruct_id = (
        torch.tensor([list(instruction_ids)], dtype=torch.long)
        if instruction_ids
        else None
    )
    if kind == "icl":
        prompt = _semantic_base_prompt(
            ref_code=(
                ref_code
                if ref_code is not None
                else torch.tensor([[70, 71, 72], [73, 74, 75]])
            ),
            speaker_value=speaker_value,
        )
        if ref_code_artifact_id is not None:
            prompt["ref_code_artifact_id"] = ref_code_artifact_id
        result = builder.build_voice_clone_inputs(
            input_id=input_id,
            ref_id=_semantic_ref_id(*reference_ids),
            voice_clone_prompt=prompt,
            language=language,
            non_streaming_mode=non_streaming,
            instruct_id=instruct_id,
        )
    elif kind == "xvector":
        result = builder.build_voice_clone_inputs(
            input_id=input_id,
            ref_id=None,
            voice_clone_prompt=_semantic_base_prompt(
                ref_code=None, speaker_value=speaker_value
            ),
            language=language,
            non_streaming_mode=non_streaming,
            instruct_id=instruct_id,
        )
    elif kind == "custom":
        result = builder.build_custom_voice_inputs(
            input_id=input_id,
            voice=voice,
            language=language,
            non_streaming_mode=non_streaming,
            instruct_id=instruct_id,
        )
    else:
        result = builder.build_voice_design_inputs(
            input_id=input_id,
            language=language,
            non_streaming_mode=non_streaming,
            instruct_id=(
                instruct_id
                if instruct_id is not None
                else torch.tensor([[60]], dtype=torch.long)
            ),
        )

    assert len(result) == 5
    embeds, _, _, _, ids = result
    return embeds, tuple(ids)


def _semantic_ids(kind: str, **kwargs: Any) -> tuple[int, ...]:
    return _build_semantic_prompt(kind, **kwargs)[1]


def _lcp(left: tuple[int, ...] | list[int], right: tuple[int, ...] | list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


@pytest.mark.parametrize(
    ("kind", "non_streaming"),
    [
        ("icl", False),
        ("icl", True),
        ("xvector", False),
        ("xvector", True),
        ("custom", False),
        ("design", False),
    ],
)
def test_real_prompt_builder_emits_one_semantic_id_per_row(
    kind: str, non_streaming: bool
) -> None:
    embeds, ids = _build_semantic_prompt(
        kind,
        non_streaming=non_streaming,
        instruction_ids=(60, 61) if kind == "design" else (),
    )
    assert len(ids) == int(embeds.shape[1])
    assert all(-(1 << 63) <= value < (1 << 63) for value in ids)


@pytest.mark.parametrize(
    ("left", "right", "expected_lcp", "compare_legacy"),
    [
        (
            {"kind": "custom", "instruction_ids": (60, 61)},
            {"kind": "custom", "instruction_ids": (60, 99)},
            1,
            True,
        ),
        (
            {"kind": "custom", "language": "en"},
            {"kind": "custom", "language": "zh"},
            5,
            True,
        ),
        (
            {"kind": "custom", "language": "auto"},
            {"kind": "custom", "language": "zh"},
            5,
            True,
        ),
        (
            {"kind": "custom", "language": "en", "voice": "Vivian"},
            {"kind": "custom", "language": "en", "voice": "Alice"},
            7,
            True,
        ),
        (
            {"kind": "xvector", "speaker_value": 1.0},
            {"kind": "xvector", "speaker_value": 2.0},
            6,
            True,
        ),
        (
            {"kind": "icl"},
            {"kind": "icl", "reference_ids": (98, 51)},
            8,
            True,
        ),
        (
            {"kind": "icl"},
            {"kind": "icl", "ref_code_artifact_id": "ref_code:" + "f" * 32},
            9,
            False,
        ),
        (
            {"kind": "icl", "target_ids": (20, 21, 22)},
            {"kind": "icl", "target_ids": (98, 21, 22)},
            10,
            True,
        ),
        (
            {"kind": "xvector"},
            {"kind": "xvector", "target_ids": (98, 21, 22)},
            8,
            True,
        ),
        (
            {"kind": "xvector", "non_streaming": True},
            {"kind": "xvector", "non_streaming": True, "target_ids": (20, 21)},
            10,
            True,
        ),
        (
            {"kind": "icl", "non_streaming": True},
            {"kind": "icl", "non_streaming": True, "target_ids": (20, 21)},
            12,
            True,
        ),
        (
            {"kind": "xvector"},
            {"kind": "xvector", "non_streaming": True},
            8,
            True,
        ),
        (
            {"kind": "xvector"},
            {"kind": "icl"},
            8,
            True,
        ),
    ],
)
def test_semantic_prompt_prefix_boundaries(
    left: dict[str, Any],
    right: dict[str, Any],
    expected_lcp: int,
    compare_legacy: bool,
) -> None:
    kind = str(left["kind"])
    right_kind = str(right.get("kind", kind))
    left_options = {key: value for key, value in left.items() if key != "kind"}
    right_options = {key: value for key, value in right.items() if key != "kind"}
    model_type = (
        "custom_voice"
        if kind == "custom"
        else "voice_design" if kind == "design" else "base"
    )
    builder = _make_prompt_builder(model_type=model_type)
    left_embeds, left_semantic = _build_semantic_prompt(
        kind, builder=builder, **left_options
    )
    right_embeds, right_semantic = _build_semantic_prompt(
        right_kind, builder=builder, **right_options
    )
    assert _lcp(left_semantic, right_semantic) == expected_lcp
    if compare_legacy:
        left_legacy = _legacy_embedding_cache_key_ids(left_embeds.squeeze(0))
        right_legacy = _legacy_embedding_cache_key_ids(right_embeds.squeeze(0))
        assert _lcp(left_semantic, right_semantic) == _lcp(left_legacy, right_legacy)


def test_streaming_target_text_after_visible_boundary_is_prompt_neutral() -> None:
    baseline = _semantic_ids("icl")
    changed_trailing_target = _semantic_ids("icl", target_ids=(20, 98, 22))
    assert baseline == changed_trailing_target


def test_base_xvector_and_icl_share_the_speaker_artifact_row() -> None:
    xvector = _semantic_ids("xvector")
    icl = _semantic_ids("icl")
    assert xvector[6] == icl[6]


def test_custom_voice_auto_language_uses_resolved_codec_language() -> None:
    auto = _semantic_ids("custom", language="auto")
    explicit = _semantic_ids("custom", language="en")
    assert auto == explicit


def test_semantic_prompt_resolves_codec_prefill_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _make_prompt_builder(model_type="custom_voice")
    calls = 0
    original = builder._resolve_language_id

    def counted_resolve_language_id(
        *, language: str, voice: str | None = None
    ) -> int | None:
        nonlocal calls
        calls += 1
        return original(language=language, voice=voice)

    monkeypatch.setattr(builder, "_resolve_language_id", counted_resolve_language_id)
    _semantic_ids("custom", builder=builder, language="auto")
    assert calls == 1


def test_semantic_row_encoders_are_direct_and_domain_separated() -> None:
    from sglang_omni.models.qwen3_tts import sglang_model

    artifact_key = 0x123456789ABCDEF
    text_id = 17
    text_codec = sglang_model._semantic_text_codec_row_id(text_id, 3)
    assert text_codec >= 1 << 61
    prefix = _make_prompt_builder()._semantic_conditioned_prefix_ids(
        role_ids=[text_id], codec_prefill_ids=(3,)
    )
    assert prefix[0] == text_id
    speaker = sglang_model._semantic_speaker_row_id(text_id, artifact_key)
    frame_zero = sglang_model._semantic_frame_row_id(text_id, artifact_key, 0)
    frame_one = sglang_model._semantic_frame_row_id(text_id, artifact_key, 1)
    assert speaker == -1 - (artifact_key ^ (text_id << 31))
    assert frame_zero == -1 - (artifact_key ^ ((text_id << 31) | 1))
    assert speaker < 0 and frame_zero < 0 and frame_one < 0
    assert len({speaker, frame_zero, frame_one}) == 3


def test_base_missing_speaker_artifact_identity_fails() -> None:
    builder = _make_prompt_builder()
    with pytest.raises(RuntimeError, match="speaker artifact identity"):
        builder.build_voice_clone_inputs(
            input_id=_semantic_input_id(20, 21, 22),
            ref_id=None,
            voice_clone_prompt=_voice_clone_prompt(ref_code=None),
            language="auto",
            non_streaming_mode=False,
        )


def test_icl_missing_ref_code_artifact_identity_fails() -> None:
    builder = _make_prompt_builder()
    ref_code = torch.tensor(
        [[70, 71, 72], [73, 74, 75]],
        dtype=torch.long,
    )
    prompt = _semantic_base_prompt(ref_code=ref_code)
    assert prompt["speaker_artifact_id"] is not None
    prompt.pop("ref_code_artifact_id")

    with pytest.raises(RuntimeError, match="ref-code artifact identity"):
        builder.build_voice_clone_inputs(
            input_id=_semantic_input_id(20, 21, 22),
            ref_id=_semantic_ref_id(50, 51),
            voice_clone_prompt=prompt,
            language="auto",
            non_streaming_mode=False,
        )


def test_icl_missing_ref_code_fails() -> None:
    builder = _make_prompt_builder()
    prompt = _semantic_base_prompt(ref_code=None)
    prompt["icl_mode"] = [True]

    with pytest.raises(RuntimeError, match="did not provide ref_code"):
        builder.build_voice_clone_inputs(
            input_id=_semantic_input_id(20, 21, 22),
            ref_id=_semantic_ref_id(50, 51),
            voice_clone_prompt=prompt,
            language="auto",
            non_streaming_mode=False,
        )


@pytest.mark.parametrize(
    ("artifact_field", "artifact_id", "expected_domain"),
    [
        ("speaker_artifact_id", "ref_code:" + "f" * 32, "speaker"),
        ("ref_code_artifact_id", "speaker:" + "f" * 32, "ref_code"),
    ],
)
def test_semantic_prompt_rejects_wrong_artifact_domain(
    artifact_field: str, artifact_id: str, expected_domain: str
) -> None:
    builder = _make_prompt_builder()
    prompt = _semantic_base_prompt(ref_code=torch.tensor([[70, 71, 72], [73, 74, 75]]))
    prompt[artifact_field] = artifact_id

    with pytest.raises(ValueError, match=expected_domain):
        builder.build_voice_clone_inputs(
            input_id=_semantic_input_id(20, 21, 22),
            ref_id=_semantic_ref_id(50, 51),
            voice_clone_prompt=prompt,
            language="auto",
            non_streaming_mode=False,
        )


def test_voice_design_prompt_allows_missing_instruction() -> None:
    builder = _make_prompt_builder(model_type="voice_design")
    embeds, _, _, _, prompt_cache_ids = builder.build_voice_design_inputs(
        input_id=_semantic_input_id(20, 21, 22),
        language="auto",
        non_streaming_mode=True,
        instruct_id=None,
    )

    assert prompt_cache_ids
    assert len(prompt_cache_ids) == int(embeds.shape[1])


def test_negative_reference_ids_survive_array_and_radix_contract() -> None:
    from array import array

    from sglang.srt.mem_cache.radix_cache import RadixKey

    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    first_ids = _semantic_ids("icl")
    second_ids = _semantic_ids("icl", ref_code_artifact_id="ref_code:" + "f" * 32)
    assert any(value < 0 for value in first_ids)
    req = SimpleNamespace(
        origin_input_ids=list(first_ids),
        origin_input_ids_unpadded=list(first_ids),
    )
    OmniScheduler._normalize_req_token_arrays(req)
    assert req.origin_input_ids == array("q", first_ids)
    assert req.origin_input_ids_unpadded == array("q", first_ids)

    first_key = RadixKey(req.origin_input_ids, "qwen3_tts:prompt:v2")
    second_key = RadixKey(array("q", second_ids), "qwen3_tts:prompt:v2")
    assert first_key.match(second_key) == 9
    with pytest.raises(ValueError, match="matching extra_key"):
        first_key.match(RadixKey(req.origin_input_ids, "qwen3_tts:prompt:other"))

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import threading
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_tts import request_builders as qwen3_request_builders
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.scheduling.speaker_cache import SpeakerArtifactCache, SpeakerCacheKey


def _voice_clone_prompt(
    *, ref_code: torch.Tensor | None = None, speaker_value: float = 1.0
) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "ref_spk_embedding": [
            torch.tensor([[speaker_value, speaker_value + 1]], dtype=torch.float32)
        ],
        "icl_mode": [ref_code is not None],
    }
    if ref_code is not None:
        prompt["ref_code"] = [ref_code]
    return prompt


class _QwenArtifactHook(
    KeyedReferenceEncodeHook[str, tuple[dict[str, Any], str | None], dict[str, Any]]
):
    model_id = "test-qwen3-tts"
    model_revision = "test-revision"
    encoder_id = "test-reference-encoder"
    encoder_config_hash = "test-config"
    artifact_kind = "test-voice-clone-prompt"

    def __init__(self) -> None:
        self.encode_calls: Counter[str] = Counter()

    def input_key(self, item: str) -> str | None:
        return item

    def encode_one(self, item: str) -> tuple[dict[str, Any], str | None]:
        self.encode_calls[item] += 1
        value = float(ord(item[0]))
        return (
            _voice_clone_prompt(
                ref_code=torch.tensor([[int(value), int(value) + 1]], dtype=torch.long)
            ),
            "reference text",
        )

    def store_artifact(
        self, artifact: tuple[dict[str, Any], str | None]
    ) -> dict[str, Any]:
        prompt, ref_text = artifact
        return qwen3_request_builders._cacheable_qwen3_tts_voice_prompt(
            prompt,
            ref_text=ref_text,
        )

    def load_artifact(
        self, stored: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        loaded = qwen3_request_builders._qwen3_tts_voice_prompt_from_cache(stored)
        assert loaded is not None
        return loaded


class _QwenRequestWrapper:
    def __init__(self, prompt: dict[str, Any]) -> None:
        self.prompt = prompt
        self.create_calls = 0

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[Any]:
        del kwargs
        self.create_calls += 1
        return [SimpleNamespace(ref_text="uploaded reference")]

    def _prompt_items_to_voice_clone_prompt(
        self, prompt_items: list[Any]
    ) -> dict[str, Any]:
        del prompt_items
        return self.prompt

    def _build_assistant_text(self, text: str) -> str:
        return text

    def _build_ref_text(self, text: str) -> str:
        return text

    def _tokenize_texts(self, texts: list[str]) -> list[torch.Tensor]:
        del texts
        return [torch.tensor([[1, 2, 3, 4]], dtype=torch.long)]


class _QwenVoiceCloneModel:
    def __init__(self) -> None:
        self.captured_prompts: list[dict[str, Any]] = []

    def build_voice_clone_inputs(self, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        self.captured_prompts.append(kwargs["voice_clone_prompt"])
        return (
            torch.zeros((1, 2, 4)),
            torch.ones((1, 2), dtype=torch.long),
            torch.zeros((1, 1, 4)),
            torch.tensor([[10, 20]], dtype=torch.long),
        )


def _prepared_request() -> qwen3_request_builders.Qwen3TTSPreparedRequest:
    return qwen3_request_builders.Qwen3TTSPreparedRequest(
        state=Qwen3TTSState(text="target"),
        input_ids_list=[],
        input_ids=torch.empty(0, dtype=torch.long),
        attention_mask=torch.empty(0, dtype=torch.long),
        trailing_text_hidden=torch.empty(0),
        ref_code=None,
        prompt_input_embeds=torch.empty(0),
        tts_pad_embed=torch.empty(0),
        gen_kwargs={},
    )


def test_artifact_fingerprint_covers_domain_dtype_shape_and_contents() -> None:
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    assert qwen3_request_builders._qwen3_tts_artifact_fingerprint(
        value, domain="speaker"
    ) == qwen3_request_builders._qwen3_tts_artifact_fingerprint(
        value.clone(), domain="speaker"
    )
    assert qwen3_request_builders._qwen3_tts_artifact_fingerprint(
        torch.tensor([[1.0, 3.0]], dtype=torch.float32), domain="speaker"
    ) != qwen3_request_builders._qwen3_tts_artifact_fingerprint(value, domain="speaker")
    assert qwen3_request_builders._qwen3_tts_artifact_fingerprint(
        value.reshape(2, 1), domain="speaker"
    ) != qwen3_request_builders._qwen3_tts_artifact_fingerprint(value, domain="speaker")
    assert qwen3_request_builders._qwen3_tts_artifact_fingerprint(
        value.to(torch.float64), domain="speaker"
    ) != qwen3_request_builders._qwen3_tts_artifact_fingerprint(value, domain="speaker")
    assert qwen3_request_builders._qwen3_tts_artifact_fingerprint(
        value, domain="speaker"
    ) != qwen3_request_builders._qwen3_tts_artifact_fingerprint(
        value, domain="ref_code"
    )


def test_artifact_ids_survive_cache_round_trip_and_separate_ref_code() -> None:
    first = qwen3_request_builders._cacheable_qwen3_tts_voice_prompt(
        _voice_clone_prompt(ref_code=torch.tensor([[1, 2], [3, 4]], dtype=torch.long)),
        ref_text="reference text",
    )
    changed_ref_code = qwen3_request_builders._cacheable_qwen3_tts_voice_prompt(
        _voice_clone_prompt(ref_code=torch.tensor([[1, 2], [3, 5]], dtype=torch.long)),
        ref_text="reference text",
    )

    assert isinstance(first["speaker_artifact_id"], str)
    assert isinstance(first["ref_code_artifact_id"], str)
    assert first["speaker_artifact_id"] == changed_ref_code["speaker_artifact_id"]
    assert first["ref_code_artifact_id"] != changed_ref_code["ref_code_artifact_id"]

    loaded = qwen3_request_builders._qwen3_tts_voice_prompt_from_cache(first)
    assert loaded is not None
    loaded_prompt, loaded_ref_text = loaded
    assert loaded_ref_text == "reference text"
    assert loaded_prompt["speaker_artifact_id"] == first["speaker_artifact_id"]
    assert loaded_prompt["ref_code_artifact_id"] == first["ref_code_artifact_id"]


def test_reference_service_reuses_identity_without_refingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = _QwenArtifactHook()
    service = ReferenceEncodeService(hook, max_items=8, max_bytes=1024 * 1024)
    fingerprint_domains: list[str] = []
    original = qwen3_request_builders._qwen3_tts_artifact_fingerprint

    def counted_fingerprint(value: torch.Tensor, *, domain: str) -> str:
        fingerprint_domains.append(domain)
        return original(value, domain=domain)

    monkeypatch.setattr(
        qwen3_request_builders,
        "_qwen3_tts_artifact_fingerprint",
        counted_fingerprint,
    )
    try:
        first_prompt, _ = service.get_or_encode("same")
        assert fingerprint_domains == ["speaker", "ref_code"]

        second_prompt, _ = service.get_or_encode("same")
        assert fingerprint_domains == ["speaker", "ref_code"]
        assert hook.encode_calls == Counter({"same": 1})
        assert (
            first_prompt["speaker_artifact_id"] == second_prompt["speaker_artifact_id"]
        )
        assert (
            first_prompt["ref_code_artifact_id"]
            == second_prompt["ref_code_artifact_id"]
        )
    finally:
        service.close()


def test_eviction_reencode_reuses_bit_identical_content_identity() -> None:
    hook = _QwenArtifactHook()
    service = ReferenceEncodeService(hook, max_items=1, max_bytes=1024 * 1024)
    try:
        first_prompt, _ = service.get_or_encode("A")
        service.get_or_encode("B")
        reencoded_prompt, _ = service.get_or_encode("A")

        assert hook.encode_calls == Counter({"A": 2, "B": 1})
        assert (
            first_prompt["speaker_artifact_id"]
            == reencoded_prompt["speaker_artifact_id"]
        )
        assert (
            first_prompt["ref_code_artifact_id"]
            == reencoded_prompt["ref_code_artifact_id"]
        )
    finally:
        service.close()


def test_reference_service_reencodes_after_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen3_request_builders.clear_qwen3_tts_preprocessing_context()
    hook = _QwenArtifactHook()
    monkeypatch.setattr(
        qwen3_request_builders,
        "_Qwen3TTSAdhocReferenceHook",
        lambda model, wrapper: hook,
    )
    model = object()
    wrapper = object()
    try:
        first_service = qwen3_request_builders._get_qwen3_tts_adhoc_reference_service(
            model, wrapper
        )
        first_prompt, _ = first_service.get_or_encode("same")

        with qwen3_request_builders.qwen3_tts_weight_update_guard():
            pass

        second_service = qwen3_request_builders._get_qwen3_tts_adhoc_reference_service(
            model, wrapper
        )
        second_prompt, _ = second_service.get_or_encode("same")

        assert first_service is not second_service
        assert hook.encode_calls == Counter({"same": 2})
        assert (
            first_prompt["speaker_artifact_id"] == second_prompt["speaker_artifact_id"]
        )
        assert (
            first_prompt["ref_code_artifact_id"]
            == second_prompt["ref_code_artifact_id"]
        )
    finally:
        with qwen3_request_builders.qwen3_tts_weight_update_guard():
            pass


def test_weight_update_guard_quiesces_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen3_request_builders.clear_qwen3_tts_preprocessing_context()
    entered = threading.Event()
    release = threading.Event()
    new_entered = threading.Event()
    mutation_started = threading.Event()
    mutation_release = threading.Event()
    prepared = _prepared_request()

    def prepare(
        payload: StagePayload, *args: Any, **kwargs: Any
    ) -> qwen3_request_builders.Qwen3TTSPreparedRequest:
        del args, kwargs
        if payload.request_id == "old":
            entered.set()
            if not release.wait(timeout=1.0):
                raise RuntimeError("old preprocessing was not released")
            raise RuntimeError("old preprocessing stopped")
        new_entered.set()
        return prepared

    monkeypatch.setattr(
        qwen3_request_builders,
        "_get_qwen3_tts_adhoc_reference_service_locked",
        lambda model, wrapper: None,
    )
    monkeypatch.setattr(qwen3_request_builders, "_prepare_qwen3_tts_request", prepare)
    qwen3_request_builders.set_qwen3_tts_preprocessing_context(
        model=object(), wrapper=object()
    )
    try:
        old_payload = StagePayload(
            request_id="old", request=OmniRequest(inputs="old"), data={}
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            old_future = executor.submit(
                qwen3_request_builders.preprocess_qwen3_tts_payload, old_payload
            )
            assert entered.wait(timeout=1.0)

            def update() -> None:
                with qwen3_request_builders.qwen3_tts_weight_update_guard():
                    mutation_started.set()
                    if not mutation_release.wait(timeout=1.0):
                        raise RuntimeError("weight mutation was not released")

            update_future = executor.submit(update)
            assert not mutation_started.wait(timeout=0.05)

            release.set()
            with pytest.raises(RuntimeError, match="old preprocessing stopped"):
                old_future.result(timeout=1.0)
            assert mutation_started.wait(timeout=1.0)

            new_payload = StagePayload(
                request_id="new", request=OmniRequest(inputs="new"), data={}
            )
            new_future = executor.submit(
                qwen3_request_builders.preprocess_qwen3_tts_payload, new_payload
            )
            assert not new_entered.wait(timeout=0.05)

            mutation_release.set()
            assert update_future.result(timeout=1.0) is None
            assert new_entered.wait(timeout=1.0)
            new_future.result(timeout=1.0)
    finally:
        release.set()
        mutation_release.set()
        qwen3_request_builders.clear_qwen3_tts_preprocessing_context()


def test_invalidation_clears_qwen_speaker_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SpeakerArtifactCache(max_bytes=1024 * 1024)
    monkeypatch.setattr(
        qwen3_request_builders, "get_speaker_artifact_cache", lambda: cache
    )
    keys = (
        SpeakerCacheKey("qwen3_tts_xvec", "guide", 1, "voice_clone_prompt"),
        SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt"),
    )
    for key in keys:
        cache.put(key, {"key": key.model_type})
        assert cache.get(key) is not None

    with qwen3_request_builders.qwen3_tts_weight_update_guard():
        pass

    for key in keys:
        assert cache.get(key) is None


def test_x_vector_and_icl_share_identical_speaker_identity() -> None:
    icl_artifact = qwen3_request_builders._cacheable_qwen3_tts_voice_prompt(
        _voice_clone_prompt(ref_code=torch.tensor([[10, 20]], dtype=torch.long)),
        ref_text="reference",
    )
    x_vector_artifact = qwen3_request_builders._cacheable_qwen3_tts_voice_prompt(
        _voice_clone_prompt(),
        ref_text=None,
    )

    assert (
        icl_artifact["speaker_artifact_id"] == x_vector_artifact["speaker_artifact_id"]
    )
    assert icl_artifact["ref_code_artifact_id"] is not None
    assert x_vector_artifact["ref_code_artifact_id"] is None


def test_uploaded_voice_miss_and_hit_use_same_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SpeakerArtifactCache(max_bytes=1024 * 1024)
    wrapper = _QwenRequestWrapper(
        _voice_clone_prompt(ref_code=torch.tensor([[10, 20]], dtype=torch.long))
    )
    model = _QwenVoiceCloneModel()
    monkeypatch.setattr(
        qwen3_request_builders,
        "get_speaker_artifact_cache",
        lambda: cache,
    )
    state = Qwen3TTSState(
        text="target",
        ref_audio=b"uploaded audio",
        uploaded_voice_name="Guide",
        uploaded_voice_created_at=7,
        ref_text="uploaded reference",
    )

    first = qwen3_request_builders._prepare_qwen3_tts_base_request(
        state=state, model=model, wrapper=wrapper
    )
    second = qwen3_request_builders._prepare_qwen3_tts_base_request(
        state=state, model=model, wrapper=wrapper
    )

    assert wrapper.create_calls == 1
    assert len(first) == len(second) == 4
    first_prompt, second_prompt = model.captured_prompts
    assert first_prompt["speaker_artifact_id"] == second_prompt["speaker_artifact_id"]
    assert first_prompt["ref_code_artifact_id"] == second_prompt["ref_code_artifact_id"]

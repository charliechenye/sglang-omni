# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import httpx
import pytest
import torch
from huggingface_hub.errors import RepositoryNotFoundError

from sglang_omni.models.moss_transcribe_diarize import stages
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_transcribe_diarize.engine_builder import (
    MossTranscribeDiarizeEngineBuilder,
)
from sglang_omni.models.moss_transcribe_diarize.stages import (
    _missing_additional_chat_templates_compat,
    create_sglang_moss_transcribe_diarize_executor,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.scheduling.generation_batch_policy import (
    build_default_prefill_cuda_graph_bs,
    build_generation_batch_overrides,
)


def _make_moss_engine_builder(
    *, kv_calibration_output_path: str | None = None
) -> MossTranscribeDiarizeEngineBuilder:
    return MossTranscribeDiarizeEngineBuilder(
        max_running_requests=16,
        max_new_tokens=None,
        context_length=None,
        mem_fraction_static=0.8,
        mm_embedding_cache_size_bytes=0,
        encoder_cache_size_bytes=0,
        enable_torch_compile=False,
        torch_compile_max_bs=4,
        enable_async_decode=True,
        async_decode_min_batch_size=2,
        prefill_coalesce_requests=0,
        prefill_coalesce_wait_ms=0.0,
        prefill_coalesce_when_idle=True,
        prefill_coalesce_requires_pending_builds=True,
        prefill_coalesce_after_builds_during_decode=True,
        encoder_chunk_buckets=[1],
        encoder_torch_compile=False,
        encoder_max_batch_size=2,
        request_build_max_workers=8,
        request_build_max_pending=16,
        stream_emit_interval_s=0.05,
        kv_calibration_output_path=kv_calibration_output_path,
        kv_calibration_checkpoint_interval_s=30.0,
    )


def test_moss_transcribe_diarize_config_uses_single_batched_stage() -> None:
    config = MossTranscribeDiarizePipelineConfig(
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory_path.endswith(
        "create_sglang_moss_transcribe_diarize_executor"
    )
    factory = config.stages[0].factory
    engine = config.stages[0].engine
    assert factory.device == "cuda:0"
    assert engine.max_running_requests == 16
    assert engine.enable_torch_compile is True
    assert engine.torch_compile_max_bs == 4
    assert factory.encoder_max_batch_size == 2
    assert factory.request_build_max_workers == 8
    assert factory.request_build_max_pending == 16
    assert factory.prefill_coalesce_requests == 4
    assert factory.prefill_coalesce_wait_ms == 12
    assert factory.prefill_coalesce_when_idle is True
    assert factory.prefill_coalesce_requires_pending_builds is True
    assert factory.prefill_coalesce_after_builds_during_decode is True
    assert factory.kv_calibration_output_path is None
    assert factory.kv_calibration_checkpoint_interval_s is None
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(
            "MossTranscribeDiarizeForConditionalGeneration"
        )
        is MossTranscribeDiarizePipelineConfig
    )
    assert MossTranscribeDiarizePipelineConfig.stage_config_cls("asr").engine_stage


def test_moss_transcribe_diarize_prefill_backend_policy() -> None:
    builder = _make_moss_engine_builder()

    assert builder.kv_calibration_collector is None
    assert builder.prepare_server_args_overrides(None) is None
    assert builder.extra_scheduler_callbacks() == {}
    assert type(builder).supports_breakable_prefill_cuda_graph is True
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["enable_torch_compile"] is False
    assert defaults["torch_compile_max_bs"] == 4
    assert defaults["cuda_graph_backend_prefill"] == "breakable"
    assert defaults["cuda_graph_bs_prefill"] == [
        1,
        2,
        *build_default_prefill_cuda_graph_bs(4096),
    ]
    assert (
        max(defaults["cuda_graph_bs_prefill"])
        == defaults["max_prefill_tokens"]
        == defaults["chunked_prefill_size"]
        == 4096
    )


def test_moss_transcribe_diarize_compile_cap_survives_batch_overrides() -> None:
    """torch_compile_max_bs must bind to the named builder parameter. If it
    ever lands in **stage_defaults instead, the merge silently replaces the
    stage cap with max_running_requests."""
    builder = _make_moss_engine_builder()

    overrides = build_generation_batch_overrides(
        server_args_overrides=None,
        **builder.generation_defaults(dtype="bfloat16"),
    )

    assert overrides["torch_compile_max_bs"] == 4
    assert overrides["max_running_requests"] == 16

    operator = build_generation_batch_overrides(
        server_args_overrides={"torch_compile_max_bs": 8},
        **builder.generation_defaults(dtype="bfloat16"),
    )
    assert operator["torch_compile_max_bs"] == 8


def test_moss_td_calibration_forces_eager_uncompiled_decoder() -> None:
    builder = _make_moss_engine_builder(
        kv_calibration_output_path="results/moss_td.raw.json"
    )

    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False
    assert defaults["cuda_graph_backend_decode"] == "disabled"
    assert defaults["cuda_graph_backend_prefill"] == "disabled"
    assert defaults["cuda_graph_config"] == {
        "decode": {"backend": "disabled"},
        "prefill": {"backend": "disabled"},
    }

    overrides = {
        "disable_cuda_graph": False,
        "enable_torch_compile": True,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "breakable",
        "cuda_graph_config": {
            "decode": {"backend": "full"},
            "prefill": {"backend": "breakable"},
        },
    }
    normalized = builder.prepare_server_args_overrides(overrides)
    assert normalized == {
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "cuda_graph_backend_decode": "disabled",
        "cuda_graph_backend_prefill": "disabled",
        "cuda_graph_config": {
            "decode": {"backend": "disabled"},
            "prefill": {"backend": "disabled"},
        },
    }

    merged = build_generation_batch_overrides(
        server_args_overrides=normalized,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(merged)
    assert merged["disable_cuda_graph"] is True
    assert merged["enable_torch_compile"] is False
    assert merged["cuda_graph_backend_decode"] == "disabled"
    assert merged["cuda_graph_backend_prefill"] == "disabled"
    assert merged["cuda_graph_config"] == {
        "decode": {"backend": "disabled"},
        "prefill": {"backend": "disabled"},
    }


def test_moss_td_calibration_resolves_both_sglang_cuda_graph_phases() -> None:
    """Exercise SGLang's phase resolver after generation-batch merging.

    This catches the precedence bug that a defaults-only assertion misses:
    explicit breakable/full operator settings and decoder compilation must not
    survive calibration normalization.
    """

    from sglang.srt.arg_groups.cuda_graph_hook import handle_cuda_graph_config
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.server_args import ServerArgs

    builder = _make_moss_engine_builder(
        kv_calibration_output_path="results/moss_td.raw.json"
    )
    operator_overrides = {
        "disable_cuda_graph": False,
        "enable_torch_compile": True,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "breakable",
        "cuda_graph_config": {
            "decode": {"backend": "full"},
            "prefill": {"backend": "breakable"},
        },
    }
    normalized = builder.prepare_server_args_overrides(operator_overrides)
    merged = build_generation_batch_overrides(
        server_args_overrides=normalized,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(merged)

    server_args_fields = set(ServerArgs.__dataclass_fields__)
    server_args = ServerArgs(
        model_path="dummy",
        **{
            key: value
            for key, value in merged.items()
            if key in server_args_fields
        },
    )
    # This is the same phase-specific resolver used by ServerArgs.resolve_once;
    # the dummy model keeps the test independent of a model download/config.
    handle_cuda_graph_config(server_args)
    resolved = resolving_view(server_args)

    assert server_args.disable_cuda_graph is True
    assert server_args.enable_torch_compile is False
    assert resolved.cuda_graph_config.prefill.backend == "disabled"
    assert resolved.cuda_graph_config.decode.backend == "disabled"


def test_factory_calibration_wires_collector_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.moss_transcribe_diarize import kv_calibration
    from sglang_omni.scheduling import engine_factory, sglang_backend

    calls = _stub_factory_env(monkeypatch, want_cuda_graph=False)
    captured: dict[str, object] = {}

    def fake_attach(model, **kwargs):
        captured.update(kwargs)
        captured["model"] = model
        return SimpleNamespace(finalize=lambda: None, abort=lambda: None)

    monkeypatch.setattr(kv_calibration, "attach_moss_td_kv_calibration", fake_attach)

    def capture_overrides(**kwargs):
        captured["generation_defaults"] = dict(kwargs)
        return {
            key: value
            for key, value in kwargs.items()
            if key != "server_args_overrides"
        }

    monkeypatch.setattr(
        engine_factory, "build_generation_batch_overrides", capture_overrides
    )

    server_args_kwargs: dict[str, object] = {}

    def capture_server_args(_model_path, **kwargs):
        server_args_kwargs.update(kwargs)
        return SimpleNamespace(
            context_length=4096,
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            enable_torch_compile=False,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="disabled"),
                prefill=SimpleNamespace(backend="disabled"),
            ),
        )

    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", capture_server_args)

    create_sglang_moss_transcribe_diarize_executor(
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        kv_calibration_output_path="results/moss_td.raw.json",
        server_args_overrides={
            "disable_cuda_graph": False,
            "enable_torch_compile": True,
        },
    )

    assert captured["generation_defaults"]["server_args_overrides"] == {
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "cuda_graph_backend_decode": "disabled",
        "cuda_graph_backend_prefill": "disabled",
        "cuda_graph_config": {
            "decode": {"backend": "disabled"},
            "prefill": {"backend": "disabled"},
        },
    }
    assert server_args_kwargs["disable_cuda_graph"] is True
    assert server_args_kwargs["enable_torch_compile"] is False
    assert server_args_kwargs["cuda_graph_backend_decode"] == "disabled"
    assert server_args_kwargs["cuda_graph_backend_prefill"] == "disabled"
    assert server_args_kwargs["cuda_graph_config"] == {
        "decode": {"backend": "disabled"},
        "prefill": {"backend": "disabled"},
    }
    assert captured["model_path"] == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    assert captured["output_path"] == "results/moss_td.raw.json"
    assert calls["init_cuda_graphs"] == 0
    assert "shutdown_callback" not in calls["scheduler_kwargs"][0]
    assert "post_quiescence_callback" in calls["scheduler_kwargs"][0]


def test_moss_td_calibration_build_failure_aborts_collector() -> None:
    from types import SimpleNamespace

    builder = _make_moss_engine_builder(
        kv_calibration_output_path="results/moss_td.raw.json"
    )
    aborted: list[None] = []
    builder.kv_calibration_collector = SimpleNamespace(
        abort=lambda: aborted.append(None)
    )

    builder.cleanup_build_failure()
    builder.cleanup_build_failure()

    assert aborted == [None]
    assert builder.kv_calibration_collector is None


@pytest.mark.parametrize(
    ("worker_override", "expected_workers"),
    [
        (None, 8),
        (4, 4),
    ],
)
def test_moss_transcribe_diarize_omp_default_tracks_request_workers(
    monkeypatch: pytest.MonkeyPatch,
    worker_override: int | None,
    expected_workers: int,
) -> None:
    from sglang_omni.models.moss_transcribe_diarize import config as config_module

    calls: list[tuple[int, int]] = []

    def _bounded_threads(*, worker_count: int, max_threads: int) -> int:
        calls.append((worker_count, max_threads))
        return 3

    monkeypatch.setattr(
        config_module,
        "bounded_intraop_threads",
        _bounded_threads,
    )

    config = config_module.MossTranscribeDiarizePipelineConfig(model_path="dummy")
    if worker_override is not None:
        from sglang_omni.config.manager import ConfigManager

        config = ConfigManager(config).merge_config(
            [("asr.factory.request_build_max_workers", str(worker_override))]
        )

    assert config.resolved_env_defaults()["OMP_NUM_THREADS"] == "3"
    assert calls[-1] == (expected_workers, 8)


def test_moss_transcribe_diarize_preserves_explicit_omp_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_transcribe_diarize import config as config_module

    monkeypatch.setattr(
        config_module,
        "bounded_intraop_threads",
        lambda **_kwargs: 999,
    )

    config = config_module.MossTranscribeDiarizePipelineConfig(
        model_path="dummy",
        env_defaults={"OMP_NUM_THREADS": "3"},
    )

    # The written entry wins over whatever the derivation would produce.
    assert config.resolved_env_defaults()["OMP_NUM_THREADS"] == "3"


def test_moss_transcribe_diarize_stage_reserves_encoder_headroom() -> None:
    signature = inspect.signature(create_sglang_moss_transcribe_diarize_executor)

    assert signature.parameters["max_running_requests"].default == 16
    assert signature.parameters["mem_fraction_static"].default == 0.80
    assert signature.parameters["enable_torch_compile"].default is False
    assert signature.parameters["torch_compile_max_bs"].default == 4
    assert signature.parameters["request_build_max_workers"].default == 8
    assert signature.parameters["request_build_max_pending"].default == 16
    assert signature.parameters["enable_async_decode"].default is True
    assert signature.parameters["async_decode_min_batch_size"].default == 1
    assert signature.parameters["prefill_coalesce_requests"].default == 4
    assert signature.parameters["prefill_coalesce_wait_ms"].default == 12.0
    assert signature.parameters["prefill_coalesce_when_idle"].default is True
    assert (
        signature.parameters["prefill_coalesce_requires_pending_builds"].default is True
    )
    assert (
        signature.parameters["prefill_coalesce_after_builds_during_decode"].default
        is True
    )
    assert signature.parameters["encoder_max_batch_size"].default == 2
    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0
    assert signature.parameters["encoder_chunk_buckets"].default is None
    assert signature.parameters["encoder_torch_compile"].default is False


def test_compile_encoder_sets_runner_and_warms_each_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
        MossTranscribeDiarizeForConditionalGeneration as Model,
    )

    monkeypatch.setattr(
        "sglang_omni.models.moss_transcribe_diarize.sglang_model.set_torch_compile_config",
        lambda: None,
    )
    warmups: list[tuple[int, ...]] = []
    runner = lambda feats, pos, forward_batch: warmups.append(tuple(feats.shape))
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: runner)

    encoder = torch.nn.Linear(4, 4)
    model = SimpleNamespace(
        whisper_encoder=encoder,
        _compiled_encoder=None,
        _compiled_chunk_buckets=frozenset(),
        config=SimpleNamespace(audio_config=SimpleNamespace(num_mel_bins=4)),
    )

    Model.compile_encoder(model, [2, 1, 1], input_feature_len=6)

    assert model._compiled_encoder is runner
    assert model._compiled_chunk_buckets == frozenset({1, 2})
    assert model._compiled_input_feature_len == 6
    assert len(warmups) == 6
    assert {shape[0] for shape in warmups} == {1, 2}
    assert all(shape[1:] == (4, 6) for shape in warmups)


def test_compile_encoder_drops_bucket_whose_warmup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
        MossTranscribeDiarizeForConditionalGeneration as Model,
    )

    monkeypatch.setattr(
        "sglang_omni.models.moss_transcribe_diarize.sglang_model.set_torch_compile_config",
        lambda: None,
    )

    def runner(feats, pos, forward_batch):
        if feats.shape[0] == 2:
            raise RuntimeError("simulated OOM during warmup")

    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: runner)

    model = SimpleNamespace(
        whisper_encoder=torch.nn.Linear(4, 4),
        _compiled_encoder=None,
        _compiled_chunk_buckets=frozenset(),
        _compiled_input_feature_len=0,
        config=SimpleNamespace(audio_config=SimpleNamespace(num_mel_bins=4)),
    )

    Model.compile_encoder(model, [1, 2], input_feature_len=6)

    assert model._compiled_chunk_buckets == frozenset({1})


def _stub_factory_env(monkeypatch: pytest.MonkeyPatch, *, want_cuda_graph: bool):
    from types import SimpleNamespace

    from transformers import AutoProcessor

    from sglang_omni import platforms
    from sglang_omni.models.moss_transcribe_diarize import (
        engine_builder,
        request_builders,
    )
    from sglang_omni.scheduling import (
        bootstrap,
        engine_factory,
        omni_scheduler,
        sglang_backend,
    )

    calls = {
        "init_cuda_graphs": 0,
        "compile_encoder": [],
        "init_encoder_graphs": [],
        "encoder_services": [],
        "scheduler_kwargs": [],
    }
    model = SimpleNamespace(
        compile_encoder=lambda buckets, feat_len: calls["compile_encoder"].append(
            (list(buckets), feat_len)
        ),
        init_encoder_graphs=lambda buckets, feat_len: calls[
            "init_encoder_graphs"
        ].append((list(buckets), feat_len)),
        init_encoder_cache=lambda n: None,
    )

    def _bump_init_cuda_graphs() -> None:
        calls["init_cuda_graphs"] += 1

    model_runner = SimpleNamespace(model=model, init_cuda_graphs=_bump_init_cuda_graphs)
    model_worker = SimpleNamespace(
        gpu_id=0,
        model_runner=model_runner,
        enable_prefill_input_embeds=False,
    )
    infra = (want_cuda_graph, (model_worker, None, None, None, None))

    monkeypatch.setattr(
        platforms.current_platform, "get_device", lambda index: "cpu", raising=False
    )

    processor = SimpleNamespace(
        tokenizer=object(),
        feature_extractor=SimpleNamespace(nb_max_frames=3000),
    )

    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        lambda *a, **k: processor,
    )
    monkeypatch.setattr(stages, "_default_max_new_tokens", lambda path: 100)
    monkeypatch.setattr(stages, "_default_context_length", lambda path: 4096)
    monkeypatch.setattr(
        engine_factory, "build_generation_batch_overrides", lambda **k: {}
    )
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        lambda *a, **k: SimpleNamespace(
            context_length=4096,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            ),
        ),
    )
    monkeypatch.setattr(
        engine_factory, "validate_generation_batch_policy", lambda **k: None
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        lambda *a, **k: infra,
    )
    monkeypatch.setattr(engine_builder, "init_mm_embedding_cache", lambda n: None)

    def _make_encoder_service(model, *, max_batch_size):
        calls["encoder_services"].append((model, max_batch_size))
        return object()

    monkeypatch.setattr(
        engine_builder, "BatchedAudioEncoderService", _make_encoder_service
    )
    monkeypatch.setattr(
        request_builders,
        "make_moss_transcribe_diarize_scheduler_adapters",
        lambda **k: (object(), object()),
    )
    monkeypatch.setattr(
        request_builders,
        "make_moss_transcribe_diarize_stream_output_builder",
        lambda **k: object(),
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", lambda **k: object())
    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        lambda **k: calls["scheduler_kwargs"].append(k) or SimpleNamespace(),
    )
    return calls


def test_factory_compiles_encoder_and_skips_cuda_graph_when_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_factory_env(monkeypatch, want_cuda_graph=True)

    create_sglang_moss_transcribe_diarize_executor(
        "OpenMOSS-Team/MOSS-Transcribe-Diarize", encoder_torch_compile=True
    )

    assert len(calls["compile_encoder"]) == 1
    assert calls["init_encoder_graphs"] == []
    assert calls["init_cuda_graphs"] == 1
    assert len(calls["encoder_services"]) == 1
    assert calls["encoder_services"][0][1] == 2
    scheduler_kwargs = calls["scheduler_kwargs"][0]
    assert scheduler_kwargs["enable_async_decode"] is True
    assert scheduler_kwargs["async_decode_min_batch_size"] == 1
    assert scheduler_kwargs["prefill_coalesce_requests"] == 4
    assert scheduler_kwargs["prefill_coalesce_wait_ms"] == 12.0
    assert scheduler_kwargs["prefill_coalesce_when_idle"] is True
    assert scheduler_kwargs["prefill_coalesce_requires_pending_builds"] is True
    assert scheduler_kwargs["prefill_coalesce_after_builds_during_decode"] is True


def test_factory_context_length_override_uses_final_server_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.moss_transcribe_diarize import request_builders
    from sglang_omni.scheduling import engine_factory, sglang_backend
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
    )

    _stub_factory_env(monkeypatch, want_cuda_graph=False)
    # The shared stub swallows server_args_overrides, but this regression
    # needs the real merge so a context_length key retained in overrides
    # would collide with the explicit keyword and raise TypeError.
    monkeypatch.setattr(
        engine_factory,
        "build_generation_batch_overrides",
        build_generation_batch_overrides,
    )

    server_args_kwargs: dict[str, object] = {}
    adapter_kwargs: dict[str, object] = {}
    final_context_length = 8193

    def capture_server_args(model_path, **kwargs):
        del model_path
        server_args_kwargs.update(kwargs)
        return SimpleNamespace(
            context_length=final_context_length,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            ),
        )

    def capture_adapters(**kwargs):
        adapter_kwargs.update(kwargs)
        return (object(), object())

    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", capture_server_args)
    monkeypatch.setattr(
        request_builders,
        "make_moss_transcribe_diarize_scheduler_adapters",
        capture_adapters,
    )

    create_sglang_moss_transcribe_diarize_executor(
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        server_args_overrides={"context_length": 8192},
    )

    assert server_args_kwargs["context_length"] == 8192
    assert adapter_kwargs["context_length"] == final_context_length


def _repo_not_found(url: str) -> RepositoryNotFoundError:
    response = httpx.Response(404, request=httpx.Request("GET", url))
    return RepositoryNotFoundError(f"missing: {url}", response=response)


def test_processor_compat_ignores_missing_additional_chat_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.processing_utils as processing_utils
    import transformers.utils.hub as hub_utils

    def missing_templates(*_args: object, **_kwargs: object) -> list[str]:
        raise _repo_not_found(
            "https://huggingface.co/api/models/repo/tree/main/"
            "additional_chat_templates"
        )

    monkeypatch.setattr(processing_utils, "list_repo_templates", missing_templates)
    monkeypatch.setattr(hub_utils, "list_repo_templates", missing_templates)

    with _missing_additional_chat_templates_compat():
        assert (
            processing_utils.list_repo_templates("repo", local_files_only=False) == []
        )
        assert hub_utils.list_repo_templates("repo", local_files_only=False) == []


def test_processor_compat_preserves_non_template_repo_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.processing_utils as processing_utils

    def missing_repo(*_args: object, **_kwargs: object) -> list[str]:
        raise _repo_not_found("https://huggingface.co/api/models/missing-repo")

    monkeypatch.setattr(processing_utils, "list_repo_templates", missing_repo)

    with _missing_additional_chat_templates_compat():
        with pytest.raises(RepositoryNotFoundError, match="missing-repo"):
            processing_utils.list_repo_templates("missing-repo", local_files_only=False)

# SPDX-License-Identifier: Apache-2.0
"""CPU-only P22 Serving profiler and P23 feasibility gate regressions."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_tts import (
    code_predictor_spec_p22_profile as prof,
)


def _enable(monkeypatch, tmp_path):
    arm = tmp_path / "P21_ARM"
    cap = tmp_path / "P22_CAPTURE_ARM"
    directory = tmp_path / "p22_profiles"
    monkeypatch.setenv(prof.PROFILE_ENV, "1")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P21_SERVING_K2", "1")
    monkeypatch.setenv("SGLANG_OMNI_QTTS_CP_P21_ARM_FILE", str(arm))
    monkeypatch.setenv(prof.CAPTURE_ARM_ENV, str(cap))
    monkeypatch.setenv(prof.DIR_ENV, str(directory))
    for env in (
        "SGLANG_OMNI_QTTS_CP_P18_DIR",
        "SGLANG_OMNI_QTTS_CP_P19_COST_DIR",
        "SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR",
    ):
        monkeypatch.delenv(env, raising=False)
    return arm, cap, directory


def _totals(count=1):
    return {
        stage: {"calls": count, "cpu_wall_ns": 100000 * count}
        for stage in prof.STAGES
    }


def test_p22_disabled_does_not_touch_cuda_or_model(monkeypatch):
    monkeypatch.delenv(prof.PROFILE_ENV, raising=False)
    m = SimpleNamespace()
    with prof.span(m, "ROUND"):
        pass
    assert not m.__dict__
    assert prof.begin_capture_if_armed(m) is False
    assert prof.finish_request(
        m, request_hash="abc", candidate_frames=7
    ) is None


def test_p22_env_gate_and_nvtx_spans_cpu_only(monkeypatch, tmp_path):
    import torch
    _enable(monkeypatch, tmp_path)
    prof.validate_environment()
    events = []
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_push",
        lambda stage: events.append(("push", stage)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_pop",
        lambda: events.append(("pop",)),
    )
    m = SimpleNamespace()
    with prof.span(m, "FULL_CODE_PREDICTOR"):
        with prof.span(m, "ROUND"):
            pass
    assert events == [
        ("push", "P22_FULL_CODE_PREDICTOR"),
        ("push", "P22_ROUND"),
        ("pop",), ("pop",),
    ]
    assert m._p22_stage_totals["ROUND"]["calls"] == 1
    assert m._p22_stage_totals["ROUND"]["cpu_wall_ns"] > 0
    with pytest.raises(RuntimeError, match="P22_UNKNOWN_STAGE"):
        with prof.span(m, "UNKNOWN"):
            pass


def test_p22_cudaprofiler_only_measured_after_capture_arm(
    monkeypatch, tmp_path,
):
    import torch
    _, cap, directory = _enable(monkeypatch, tmp_path)
    calls = []
    fake_cudart = SimpleNamespace(
        cudaProfilerStart=lambda: calls.append("start") or 0,
        cudaProfilerStop=lambda: calls.append("stop") or 0,
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake_cudart)
    monkeypatch.setattr(
        torch.cuda, "synchronize",
        lambda device: calls.append(("sync", device)),
    )
    m = SimpleNamespace(predictor_device="cpu")
    assert prof.begin_capture_if_armed(m) is False
    assert calls == []
    cap.touch()
    assert prof.begin_capture_if_armed(m) is True
    assert prof.begin_capture_if_armed(m) is True
    assert calls == ["start"]
    m._p22_stage_totals = _totals()
    result = prof.finish_request(
        m, request_hash="abc123", candidate_frames=1
    )
    assert result["cuda_profiler_capture"] is True
    assert calls == ["start", ("sync", "cpu"), "stop"]
    record = directory / "request_abc123.json"
    assert record.is_file()
    assert json.loads(record.read_text())["request_hash"] == "abc123"
    with pytest.raises(RuntimeError, match="ONE_PROFILER_CAPTURE_PER_WORKER"):
        prof.begin_capture_if_armed(m)
    # Never overwrite a completed request's evidence.
    m._p22_stage_totals = _totals()
    with pytest.raises(FileExistsError):
        prof.finish_request(
            m, request_hash="abc123", candidate_frames=1
        )


def test_p22_unarmed_request_produces_no_cuda_capture(monkeypatch, tmp_path):
    import torch
    _, _, directory = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(
        torch.cuda, "cudart",
        lambda: (_ for _ in ()).throw(
            AssertionError("cudaProfilerStart must not run")
        ),
    )
    m = SimpleNamespace(predictor_device="cpu")
    m._p22_stage_totals = _totals()
    p = prof.finish_request(
        m, request_hash="warmup", candidate_frames=1
    )
    assert p["cuda_profiler_capture"] is False
    assert (directory / "request_warmup.json").is_file()


def test_p22_real_stage_count_contract_fail_closed():
    from experiments.qwen3_tts_cp_spec_k2.p22_profile_once import (
        validate_profile,
    )
    worker = {"candidate_frames": 10, "k2_rounds": 87}
    per_frame = {
        "FULL_CODE_PREDICTOR", "INIT_CPU_PARAMS", "INIT_PAIR_FORWARD",
        "INIT_CODE1_SAMPLER",
    }
    stages = {
        stage: {"calls": 10 if stage in per_frame else 87,
                "cpu_wall_ns": 100_000 * (10 if stage in per_frame else 87)}
        for stage in prof.STAGES if stage != "NATIVE_FINAL_AR"
    }
    p = {"cuda_profiler_capture": False, "candidate_frames": 10,
         "stage_totals": stages}
    result = validate_profile(
        p, expected_capture=False, worker=worker
    )
    assert result["WIDE_VERIFIER"]["calls"] == 87
    assert result["ROUND_CPU_ACCEPT_LENGTH"]["cpu_wall_per_k2_round_ms"] > 0
    corrupted = json.loads(json.dumps(p))
    corrupted["stage_totals"]["ROUND"]["calls"] -= 1
    with pytest.raises(RuntimeError, match="STAGE_CALL_COVERAGE_MISMATCH"):
        validate_profile(
            corrupted, expected_capture=False, worker=worker
        )


def test_p22_profile_reads_durable_worker_files(tmp_path):
    from experiments.qwen3_tts_cp_spec_k2.p22_profile_once import (
        worker_profiles,
    )
    d = tmp_path / "profiles"
    d.mkdir()
    for name, ns in (("newer", 200), ("older", 100)):
        (d / f"request_{name}.json").write_text(json.dumps({
            "schema": prof.SCHEMA, "request_hash": name,
            "monotonic_ns": ns,
        }))
    result = worker_profiles(d)
    assert [p["request_hash"] for p in result] == ["older", "newer"]
    (d / "request_newer.json").write_text(json.dumps({
        "schema": "INVALID", "request_hash": "newer",
        "monotonic_ns": 200,
    }))
    with pytest.raises(RuntimeError, match="SCHEMA_OR_FILENAME_MISMATCH"):
        worker_profiles(d)


def test_p23_required_reduction_and_static_device_loop_blockers():
    from experiments.qwen3_tts_cp_spec_k2.p23_full_gpu_feasibility import (
        required_latency_reduction, source_blockers,
    )
    assert required_latency_reduction(.744, 3.691) == pytest.approx(
        1 - .95 * .744 / 3.691
    )
    with pytest.raises(ValueError, match="P23_INVALID_LATENCY"):
        required_latency_reduction(0, 2)
    source_path = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni/models/qwen3_tts/code_predictor_spec_serving_p21.py"
    )
    blockers = source_blockers(source_path.read_text())
    assert "HOST_DYNAMIC_DEPTH_READ" in blockers
    assert "HOST_CODE_VALIDITY_SYNC" in blockers
    assert "PYTHON_DYNAMIC_DEPTH_WHILE" in blockers
    assert "PRIVATE_DRAFT_RNG_NOT_NATIVE_PARITY" in blockers


def test_p22_launch_script_parses_before_any_gpu_activity():
    root = Path(__file__).resolve().parents[3]
    script = (
        root / "experiments/qwen3_tts_cp_spec_k2/"
        "p22_serving_profile_launch.sh"
    )
    p = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=root, capture_output=True, text=True,
        check=False,
    )
    assert p.returncode == 0, p.stderr
    source = script.read_text()
    assert 'BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"' in source
    assert "P22_PROFILE_BRANCH_MISMATCH" in source
    assert "P22_PROFILE_NOT_AT_REPOSITORY_ROOT" in source
    assert 'WORKTREE_PHYSICAL="$(cd "$WORKTREE" && pwd -P)"' in source
    assert "experiment/qwen3-tts-p22-serving-profile" in source
    assert "--capture-range=cudaProfilerApi" in source
    assert "--capture-range-end=stop" in source
    # --target-processes=all is a Nsight Compute (ncu) flag,
    # not supported by the Nsight Systems 2024.6 CLI.
    profile_invocation = source.split("  nsys profile ", 1)[1]
    assert "--target-processes=all" not in profile_invocation
    assert "--capture-range=cudaProfilerApi" in profile_invocation
    assert "--capture-range-end=stop" in profile_invocation
    assert "P22_NSYS_PROFILE_OPTION_UNSUPPORTED" in source
    assert 'NSYS_PROFILE_HELP="$(nsys profile --help 2>&1 || true)"' in source
    assert source.index("NSYS_PROFILE_HELP=") < source.index('mkdir -m 700 -p "$ROOT"')
    assert "SGLANG_OMNI_QTTS_CP_P22_PROFILE_DIR" in source
    assert 'SGLANG_OMNI_QTTS_CP_SPEC_K=0' in source


def test_p22_launcher_wrong_branch_reports_actual_before_GPU(tmp_path):
    """Mismatched local branch must fail with evidence, before GPU setup."""
    repo_root = Path(__file__).resolve().parents[3]
    script = (
        repo_root / "experiments/qwen3_tts_cp_spec_k2/"
        "p22_serving_profile_launch.sh"
    )
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(fake_repo)], check=True
    )
    subprocess.run(
        ["git", "-C", str(fake_repo), "symbolic-ref",
         "HEAD", "refs/heads/other-local-branch"],
        check=True,
    )
    root = tmp_path / "new-absent-output"
    result = subprocess.run(
        ["bash", str(script), str(root)],
        cwd=fake_repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "P22_PROFILE_BRANCH_MISMATCH" in result.stderr
    assert "P22_PROFILE_ACTUAL_BRANCH=other-local-branch" in result.stderr
    assert (
        "P22_PROFILE_EXPECTED_BRANCH=experiment/"
        "qwen3-tts-p22-serving-profile"
    ) in result.stderr
    assert not root.exists()


def test_p22_real_serving_nvtx_marks_host_wait_and_gpu_launches():
    import inspect
    from sglang_omni.models.qwen3_tts import (
        code_predictor_spec_serving_p21 as serving,
        code_predictor_spec_p17_wide as p17,
    )
    real = inspect.getsource(serving.serving_k2)
    wide = inspect.getsource(p17.make_wide_round)
    for name in (
        "FULL_CODE_PREDICTOR", "ROUND", "ROUND_CPU_ACCEPT_LENGTH",
        "ROUND_CPU_CODE_VALIDITY", "ROUND_GPU_COMMIT",
        "NATIVE_FINAL_AR",
    ):
        assert f'with p22_span(model, "{name}")' in real
    for name in (
        "Q_PROPOSALS", "WIDE_VERIFIER", "TARGET_SUPPORT_ACCEPT_BONUS"
    ):
        assert f'with p22_span(model, "{name}")' in wide

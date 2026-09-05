# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the MOSS-TD FP8 KV calibration helper."""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from torch import nn

from sglang_omni.models.moss_transcribe_diarize.kv_calibration import (
    COLLECTOR_VERSION,
    EXPECTED_NUM_LAYERS,
    RAW_ARTIFACT_NAME,
    RAW_SCHEMA_VERSION,
    CalibrationValidationError,
    attach_moss_td_kv_calibration,
    convert_raw_calibration_to_vllm_legacy,
    read_raw_calibration,
    validate_raw_calibration,
)


class _FakeRadixAttention(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.forward_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        _forward_batch=None,
    ) -> torch.Tensor:
        self.forward_inputs.append((key, value))
        return query


class _FakeDecoderLayer(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.attn = _FakeRadixAttention(layer_id)


class _FakeMossModel(nn.Module):
    def __init__(self, num_layers: int = EXPECTED_NUM_LAYERS) -> None:
        super().__init__()
        decoder = nn.Module()
        decoder.layers = nn.ModuleList(
            [_FakeDecoderLayer(layer_id) for layer_id in range(num_layers)]
        )
        language_model = nn.Module()
        language_model.model = decoder
        language_model.anchor = nn.Parameter(torch.zeros(1))
        self.language_model = language_model


def _collect_complete_raw_artifact(tmp_path):
    raw_path = tmp_path / "moss_td.raw.json"
    model = _FakeMossModel()
    collector = attach_moss_td_kv_calibration(
        model,
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        output_path=raw_path,
        checkpoint_interval_s=3600,
        git_metadata=("f62fd76cc9b2cc589db2c2e34f325870c1a5fd73", True),
    )
    query = torch.ones(1, 2)
    for layer_id, layer in enumerate(model.language_model.model.layers):
        key = torch.tensor([[layer_id + 1.0, -(layer_id + 2.0)]])
        value = torch.tensor([[10.0 + layer_id, -11.0 - layer_id]])
        layer.self_attn.attn(query, key, value, None)
        observed_key, observed_value = layer.self_attn.attn.forward_inputs[-1]
        assert observed_key is key
        assert observed_value is value
    collector.finalize()
    return raw_path


def test_collector_hooks_radix_inputs_and_publishes_atomic_raw_artifact(tmp_path):
    raw_path = tmp_path / "moss_td.raw.json"
    model = _FakeMossModel()
    collector = attach_moss_td_kv_calibration(
        model,
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        output_path=raw_path,
        checkpoint_interval_s=3600,
        git_metadata=("deadbeef", False),
    )

    # Every call below goes through the installed pre-hook.  The fake module
    # records the same positional K/V tensors that its RadixAttention forward
    # receives, proving the collector is on the required boundary.
    query = torch.ones(1, 2)
    for layer_id, layer in enumerate(model.language_model.model.layers):
        key = torch.tensor([[layer_id + 1.0, -(layer_id + 2.0)]])
        value = torch.tensor([[10.0 + layer_id, -11.0 - layer_id]])
        layer.self_attn.attn(query, key, value, None)

    checkpoint = json.loads(raw_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "in_progress"
    with pytest.raises(CalibrationValidationError, match="status"):
        read_raw_calibration(raw_path)

    collector.finalize()
    payload = read_raw_calibration(raw_path)
    assert payload["status"] == "complete"
    assert payload["artifact"] == RAW_ARTIFACT_NAME
    assert payload["collector_version"] == COLLECTOR_VERSION
    assert payload["schema_version"] == RAW_SCHEMA_VERSION
    assert payload["model_path"] == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    assert payload["git_head"] == "deadbeef"
    assert payload["git_dirty"] is False
    assert payload["num_layers"] == EXPECTED_NUM_LAYERS
    assert payload["observed_layer_count"] == EXPECTED_NUM_LAYERS
    assert payload["observed_layers"] == list(range(EXPECTED_NUM_LAYERS))
    assert payload["layers"][0]["k_amax"] == 2.0
    assert payload["layers"][0]["v_amax"] == 11.0
    assert payload["layers"][27]["k_amax"] == 29.0
    assert payload["layers"][27]["v_amax"] == 38.0
    assert not list(tmp_path.glob("*.tmp"))


def test_attach_rejects_a_decoder_that_is_not_exactly_28_layers(tmp_path):
    with pytest.raises(CalibrationValidationError, match="exactly 28"):
        attach_moss_td_kv_calibration(
            _FakeMossModel(num_layers=27),
            model_path="model",
            output_path=tmp_path / "raw.json",
            git_metadata=("deadbeef", False),
        )


def test_checkpoint_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        attach_moss_td_kv_calibration(
            _FakeMossModel(),
            model_path="model",
            output_path=tmp_path / "raw.json",
            checkpoint_interval_s=0,
            git_metadata=("deadbeef", False),
        )


def test_abort_detaches_hooks_without_publishing_a_valid_artifact(tmp_path):
    raw_path = tmp_path / "moss_td.raw.json"
    model = _FakeMossModel()
    collector = attach_moss_td_kv_calibration(
        model,
        model_path="model",
        output_path=raw_path,
        git_metadata=("deadbeef", False),
    )
    collector.abort()
    model.language_model.model.layers[0].self_attn.attn(
        torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2), None
    )
    assert json.loads(raw_path.read_text(encoding="utf-8"))["status"] == "in_progress"


def test_finalize_rejects_missing_layer_observation(tmp_path):
    raw_path = tmp_path / "moss_td.raw.json"
    model = _FakeMossModel()
    collector = attach_moss_td_kv_calibration(
        model,
        model_path="model",
        output_path=raw_path,
        checkpoint_interval_s=3600,
        git_metadata=("deadbeef", False),
    )
    for layer_id, module in enumerate(model.language_model.model.layers[:-1]):
        module.self_attn.attn(
            torch.tensor([[layer_id + 1.0]]),
            torch.tensor([[layer_id + 2.0]]),
            torch.ones(1, 1),
            None,
        )

    with pytest.raises(CalibrationValidationError, match="exactly all 28"):
        collector.finalize()

    assert json.loads(raw_path.read_text(encoding="utf-8"))["status"] == "invalid"


@pytest.mark.parametrize(
    ("key_value", "value_value"),
    [(float("nan"), 1.0), (1.0, float("inf"))],
)
def test_finalize_rejects_nonfinite_amax(tmp_path, key_value, value_value):
    raw_path = tmp_path / "moss_td.raw.json"
    model = _FakeMossModel()
    collector = attach_moss_td_kv_calibration(
        model,
        model_path="model",
        output_path=raw_path,
        checkpoint_interval_s=3600,
        git_metadata=("deadbeef", False),
    )
    for layer_id, module in enumerate(model.language_model.model.layers):
        module.self_attn.attn(
            torch.ones(1, 1),
            torch.tensor([[key_value if layer_id == 0 else 1.0]]),
            torch.tensor([[value_value if layer_id == 0 else 2.0]]),
            None,
        )

    with pytest.raises(CalibrationValidationError, match="finite"):
        collector.finalize()

    assert json.loads(raw_path.read_text(encoding="utf-8"))["status"] == "invalid"


@pytest.fixture()
def complete_payload(tmp_path):
    raw_path = _collect_complete_raw_artifact(tmp_path)
    return json.loads(raw_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(num_layers=27), "exactly 28 layers"),
        (
            lambda payload: payload.update(observed_layers=list(range(27))),
            "exactly all 28 layers",
        ),
        (
            lambda payload: payload.update(observed_layer_count=27),
            "observed_layer_count",
        ),
        (
            lambda payload: payload["layers"].pop(4),
            "layer records",
        ),
        (
            lambda payload: payload["layers"][3].update(k_amax=0),
            "k_amax must be > 0",
        ),
        (
            lambda payload: payload["layers"][3].update(v_amax=-1),
            "v_amax must be > 0",
        ),
        (
            lambda payload: payload["layers"][3].update(k_amax=math.nan),
            "k_amax must be finite",
        ),
        (
            lambda payload: payload["layers"][3].update(v_amax=math.inf),
            "v_amax must be finite",
        ),
        (
            lambda payload: payload.update(status="in_progress"),
            "status",
        ),
    ],
)
def test_validation_rejects_invalid_or_incomplete_artifacts(
    complete_payload, mutation, message
):
    payload = copy.deepcopy(complete_payload)
    mutation(payload)
    with pytest.raises(CalibrationValidationError, match=message):
        validate_raw_calibration(payload)


def test_conversion_requires_margin_and_uses_shared_legacy_scale(tmp_path):
    raw_path = _collect_complete_raw_artifact(tmp_path)
    output_path = tmp_path / "moss_td.kv_scales.json"

    with pytest.raises(TypeError):
        convert_raw_calibration_to_vllm_legacy(raw_path, output_path)

    converted = convert_raw_calibration_to_vllm_legacy(
        raw_path,
        output_path,
        margin=2.0,
    )
    assert converted["model_type"] == "qwen3"
    assert converted["kv_cache"]["dtype"] == "float8_e4m3fn"
    scales = converted["kv_cache"]["scaling_factor"]["0"]
    assert math.isclose(scales["0"], 11.0 / 448.0 * 2.0)
    assert math.isclose(scales["27"], 38.0 / 448.0 * 2.0)
    assert json.loads(output_path.read_text(encoding="utf-8")) == converted
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("margin", [0, -1, math.nan, math.inf])
def test_conversion_rejects_invalid_margin(tmp_path, margin):
    raw_path = _collect_complete_raw_artifact(tmp_path)
    with pytest.raises(ValueError, match="margin"):
        convert_raw_calibration_to_vllm_legacy(
            raw_path,
            tmp_path / "scales.json",
            margin=margin,
        )

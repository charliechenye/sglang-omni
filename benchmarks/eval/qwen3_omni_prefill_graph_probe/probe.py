# SPDX-License-Identifier: Apache-2.0
"""Observe upstream prefill graph lifecycle without changing dispatch.

This module is loaded only by :mod:`sitecustomize` in the probe subprocess.
Every wrapped method calls the original upstream method and records its result;
there is no local graph candidate, bucket policy, or fallback dispatcher here.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProbeState:
    requested_prefill_backend: str | None = None
    resolved_prefill_backend: str | None = None
    model_architecture: str | None = None
    is_multimodal: bool | None = None
    compatibility_injection_requested: bool = False
    compatibility_injection_applied: bool = False
    compatibility_injection_note: str = ""
    capture_bucket_list: list[int] = field(default_factory=list)
    capture_time_seconds: float | None = None
    extra_reserved_memory_bytes: int | None = None
    can_run_graph_evaluations: int = 0
    accepted_graph_batches: int = 0
    eager_fallback_count: int = 0
    capture_calls: int = 0
    replay_calls: int = 0
    capture_succeeded: bool = False
    extend_token_counts: list[int] = field(default_factory=list)
    selected_buckets: list[int | None] = field(default_factory=list)
    padding_ratios: list[float | None] = field(default_factory=list)
    live_serving_input_embeds_none_before_eligibility: bool | None = None
    replace_embeds_none_before_eligibility: bool | None = None
    compatibility_error: str | None = None
    instrumentation_error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record_model_runner(self, model_runner: Any) -> None:
        model_config = getattr(model_runner, "model_config", None)
        architectures = getattr(model_config, "architectures", None)
        if architectures:
            if isinstance(architectures, (list, tuple)):
                self.model_architecture = ",".join(map(str, architectures))
            else:
                self.model_architecture = str(architectures)
        if self.model_architecture is None:
            self.model_architecture = type(getattr(model_runner, "model", None)).__name__
        multimodal = getattr(model_config, "is_multimodal", None)
        if multimodal is not None:
            self.is_multimodal = bool(multimodal)

    def _record_none_contract(self, forward_batch: Any) -> None:
        input_embeds_none = getattr(forward_batch, "input_embeds", None) is None
        replace_embeds_none = getattr(forward_batch, "replace_embeds", None) is None
        if self.live_serving_input_embeds_none_before_eligibility is None:
            self.live_serving_input_embeds_none_before_eligibility = input_embeds_none
        else:
            self.live_serving_input_embeds_none_before_eligibility &= input_embeds_none
        if self.replace_embeds_none_before_eligibility is None:
            self.replace_embeds_none_before_eligibility = replace_embeds_none
        else:
            self.replace_embeds_none_before_eligibility &= replace_embeds_none

    def qualified(self) -> bool:
        return bool(
            self.resolved_prefill_backend == "breakable"
            and self.capture_succeeded
            and self.replay_calls > 0
            and self.accepted_graph_batches > 0
            and self.live_serving_input_embeds_none_before_eligibility is True
        )

    def report(self) -> dict[str, Any]:
        data = asdict(self)
        data["c_qualified"] = self.qualified()
        data["qualification_requirements"] = {
            "resolved_backend_is_breakable": self.resolved_prefill_backend
            == "breakable",
            "capture_succeeded": self.capture_succeeded,
            "replay_count_positive": self.replay_calls > 0,
            "accepted_graph_batch_count_positive": self.accepted_graph_batches > 0,
            "live_input_embeds_none": (
                self.live_serving_input_embeds_none_before_eligibility is True
            ),
        }
        return data


def _reserved_memory() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_reserved())
    except Exception:
        return None


def _backend_name(value: Any) -> str:
    return str(getattr(value, "value", value)).lower().split(".")[-1]


def _install_model_config_compatibility(state: ProbeState) -> None:
    """Apply the opt-in 0.5.16 Qwen multimodal classification workaround."""
    if not state.compatibility_injection_requested:
        return
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except Exception as exc:  # pragma: no cover - depends on probe runtime
        state.compatibility_error = f"ModelConfig import failed: {exc}"
        return

    if getattr(ModelConfig.__init__, "_qwen3_omni_probe_wrapped", False):
        state.compatibility_injection_applied = True
        state.compatibility_injection_note = "ModelConfig.__init__ already wrapped"
        return

    original_init = ModelConfig.__init__

    def init_with_qwen_multimodal_flag(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        architectures = getattr(self, "architectures", None) or ()
        if "Qwen3OmniThinkerForCausalLM" in architectures:
            self.is_multimodal = True

    init_with_qwen_multimodal_flag._qwen3_omni_probe_wrapped = True
    ModelConfig.__init__ = init_with_qwen_multimodal_flag
    state.compatibility_injection_applied = True
    state.compatibility_injection_note = (
        "Set ModelConfig.is_multimodal=True only for "
        "Qwen3OmniThinkerForCausalLM"
    )


def install_wrappers(state: ProbeState) -> None:
    """Install observation wrappers around the pinned upstream classes."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    if getattr(PrefillCudaGraphRunner, "_qwen3_omni_probe_wrapped", False):
        return

    original_init = PrefillCudaGraphRunner.__init__
    original_capture = PrefillCudaGraphRunner.capture
    original_capture_one_shape = PrefillCudaGraphRunner.capture_one_shape
    original_can_run_graph = PrefillCudaGraphRunner.can_run_graph
    original_execute = PrefillCudaGraphRunner.execute
    original_model_forward = ModelRunner.forward

    def init_wrapper(self: Any, model_runner: Any) -> None:
        state.record_model_runner(model_runner)
        original_init(self, model_runner)
        state.resolved_prefill_backend = _backend_name(
            getattr(self, "prefill_backend_name", None)
        )
        if state.is_multimodal is None:
            state.is_multimodal = bool(getattr(self, "is_multimodal", False))
        if not state.capture_bucket_list:
            state.capture_bucket_list = [
                int(bucket) for bucket in getattr(self, "capture_num_tokens", ())
            ]

    def capture_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        state.capture_calls += 1
        state.capture_bucket_list = [
            int(bucket) for bucket in getattr(self, "capture_num_tokens", ())
        ]
        reserved_before = _reserved_memory()
        started = time.perf_counter()
        try:
            result = original_capture(self, *args, **kwargs)
        except Exception as exc:
            state.capture_succeeded = False
            state.events.append({"event": "capture_error", "error": str(exc)})
            raise
        else:
            state.capture_succeeded = True
            return result
        finally:
            state.capture_time_seconds = time.perf_counter() - started
            reserved_after = _reserved_memory()
            if reserved_before is not None and reserved_after is not None:
                state.extra_reserved_memory_bytes = reserved_after - reserved_before

    def capture_one_shape_wrapper(self: Any, size: int, *args: Any, **kwargs: Any) -> Any:
        if int(size) not in state.capture_bucket_list:
            state.capture_bucket_list.append(int(size))
        return original_capture_one_shape(self, size, *args, **kwargs)

    def can_run_graph_wrapper(self: Any, forward_batch: Any) -> bool:
        state.can_run_graph_evaluations += 1
        state._record_none_contract(forward_batch)
        token_count = len(getattr(forward_batch, "input_ids", ()))
        state.extend_token_counts.append(token_count)
        result = original_can_run_graph(self, forward_batch)
        if result:
            state.accepted_graph_batches += 1
        state.events.append(
            {
                "event": "can_run_graph",
                "extend_token_count": token_count,
                "accepted": bool(result),
                "input_embeds_none": getattr(forward_batch, "input_embeds", None)
                is None,
                "replace_embeds_none": getattr(forward_batch, "replace_embeds", None)
                is None,
            }
        )
        return result

    def execute_wrapper(self: Any, forward_batch: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = original_execute(self, forward_batch, *args, **kwargs)
        except Exception as exc:
            state.events.append({"event": "prefill_replay_error", "error": str(exc)})
            raise
        state.replay_calls += 1
        selected_bucket = getattr(self, "_static_num_tokens", None)
        raw_tokens = getattr(self, "raw_num_tokens", None)
        state.selected_buckets.append(
            int(selected_bucket) if selected_bucket is not None else None
        )
        state.padding_ratios.append(
            (
                float(selected_bucket) / float(raw_tokens) - 1.0
                if selected_bucket is not None and raw_tokens
                else None
            )
        )
        state.events.append(
            {
                "event": "prefill_replay",
                "extend_token_count": len(getattr(forward_batch, "input_ids", ())),
                "selected_bucket": (
                    int(selected_bucket) if selected_bucket is not None else None
                ),
            }
        )
        return result

    def model_forward_wrapper(self: Any, forward_batch: Any, *args: Any, **kwargs: Any):
        result = original_model_forward(self, forward_batch, *args, **kwargs)
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_extend = getattr(forward_mode, "is_extend", None)
        if callable(is_extend) and is_extend():
            graph_result = bool(getattr(result, "can_run_graph", False))
            if not graph_result:
                state.eager_fallback_count += 1
        return result

    PrefillCudaGraphRunner.__init__ = init_wrapper
    PrefillCudaGraphRunner.capture = capture_wrapper
    PrefillCudaGraphRunner.capture_one_shape = capture_one_shape_wrapper
    PrefillCudaGraphRunner.can_run_graph = can_run_graph_wrapper
    PrefillCudaGraphRunner.execute = execute_wrapper
    ModelRunner.forward = model_forward_wrapper
    PrefillCudaGraphRunner._qwen3_omni_probe_wrapped = True


def install_from_env() -> ProbeState | None:
    output = os.environ.get("QWEN3_OMNI_GRAPH_PROBE_OUTPUT")
    if not output:
        return None
    state = ProbeState(
        requested_prefill_backend=os.environ.get(
            "QWEN3_OMNI_GRAPH_PROBE_REQUESTED_BACKEND"
        ),
        compatibility_injection_requested=(
            os.environ.get("QWEN3_OMNI_GRAPH_PROBE_COMPAT", "0") == "1"
        ),
    )
    _install_model_config_compatibility(state)
    try:
        install_wrappers(state)
    except Exception as exc:  # pragma: no cover - depends on probe runtime
        state.instrumentation_error = str(exc)

    def write_report() -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.report(), indent=2) + "\n")

    atexit.register(write_report)
    return state

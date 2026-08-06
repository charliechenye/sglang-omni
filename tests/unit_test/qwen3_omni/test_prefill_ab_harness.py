# SPDX-License-Identifier: Apache-2.0
"""Independent parity helpers for the Qwen3-Omni prefill A/B/C arms."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import torch

from benchmarks.eval.benchmark_qwen3_omni_prefill_ab import (
    RequestMeasurement,
    _aggregate,
    _arm_config,
    _build_payload,
    _measure_one,
    compare_benchmark_results,
)


def _oracle_text_audio_embeddings(
    input_ids: torch.Tensor,
    embed_tokens: torch.nn.Module,
    audio_positions_cpu: torch.Tensor,
    audio_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Build the expected text/audio rows without calling production helpers."""
    vocab_size = int(embed_tokens.num_embeddings)
    output = embed_tokens(input_ids.clamp(min=0, max=vocab_size - 1)).detach().clone()
    if audio_positions_cpu.numel():
        positions = audio_positions_cpu.to(device=output.device)
        output.index_copy_(0, positions, audio_embeddings.to(output))
    return output


def test_prefill_oracle_handles_text_and_disjoint_audio_spans() -> None:
    torch.manual_seed(0)
    embed_tokens = torch.nn.Embedding(32, 4)
    input_ids = torch.tensor([1, 31, 2, 31, 3], dtype=torch.long)
    audio = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    expected = _oracle_text_audio_embeddings(
        input_ids,
        embed_tokens,
        torch.tensor([1, 3], dtype=torch.long),
        audio,
    )

    assert torch.equal(expected[[1, 3]], audio)
    assert torch.equal(expected[[0, 2, 4]], embed_tokens(input_ids[[0, 2, 4]]))


def test_prefill_oracle_handles_current_chunk_positions() -> None:
    torch.manual_seed(1)
    embed_tokens = torch.nn.Embedding(32, 4)
    chunk_ids = torch.tensor([31, 4, 31], dtype=torch.long)
    audio = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    expected = _oracle_text_audio_embeddings(
        chunk_ids,
        embed_tokens,
        torch.tensor([0, 2], dtype=torch.long),
        audio[[1, 2]],
    )

    assert torch.equal(expected[0], audio[1])
    assert torch.equal(expected[2], audio[2])


def test_benchmark_arms_keep_graph_configuration_external() -> None:
    assert _arm_config("A", None).prefill_backend == "disabled"
    assert _arm_config("B", None).prefill_backend == "disabled"
    assert _arm_config("C", None).prefill_backend == "breakable"
    assert _arm_config("C", "disabled").prefill_backend == "disabled"


def test_benchmark_payload_omits_seed_but_requests_streamed_usage() -> None:
    payload = _build_payload(
        model="qwen3-omni",
        prompt="hello",
        audio_path=None,
        seed=None,
        max_tokens=8,
    )

    assert "seed" not in payload
    assert payload["stream_options"] == {"include_usage": True}

    seeded = _build_payload(
        model="qwen3-omni",
        prompt="hello",
        audio_path=None,
        seed=17,
        max_tokens=8,
    )
    assert seeded["seed"] == 17


def test_benchmark_aggregate_does_not_impute_missing_usage() -> None:
    results = [
        RequestMeasurement(
            arm="B",
            request_id="B-0",
            prompt_id="prompt-0",
            fixture_id="fixture-0",
            success=True,
            completion_tokens=4,
            total_seconds=1.0,
        ),
        RequestMeasurement(
            arm="B",
            request_id="B-1",
            prompt_id="prompt-1",
            fixture_id="fixture-1",
            success=True,
            completion_tokens=None,
            total_seconds=1.0,
        ),
    ]

    aggregate = _aggregate(results, wall_clock=2.0)

    assert aggregate["completion_tokens_observed"] == 1
    assert aggregate["completion_tokens_missing"] == 1
    assert aggregate["output_tokens_per_second"] is None


def test_benchmark_parity_reports_match_mismatch_missing_and_failed() -> None:
    def request(
        fixture_id: str,
        *,
        arm: str,
        output: str,
        success: bool = True,
        finish_reason: str | None = "stop",
        completion_tokens: int | None = 2,
    ) -> dict[str, object]:
        return {
            "request_id": f"{arm}-{fixture_id}",
            "fixture_id": fixture_id,
            "success": success,
            "complete_output_text": output,
            "output_sha256": f"hash-{output}",
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
        }

    report = compare_benchmark_results(
        {
            "requests": [
                request("same", arm="A", output="ok"),
                request("diff", arm="A", output="a"),
            ]
        },
        {
            "requests": [
                request("same", arm="B", output="ok"),
                request("diff", arm="B", output="b"),
                request("failed", arm="B", output="", success=False),
            ]
        },
    )

    assert report["match_field"] == "fixture_id"
    assert report["matched_requests"] == 1
    assert len(report["mismatched_outputs"]) == 1
    assert report["missing_requests"]["left"] == ["failed"]
    assert len(report["failed_requests"]) == 1
    assert report["performance_comparable"] is False


def test_streamed_measurement_collects_text_finish_and_usage() -> None:
    class AsyncLines:
        def __init__(self, lines: list[bytes]) -> None:
            self.lines = lines

        def __aiter__(self):
            self.iterator = iter(self.lines)
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class Response:
        status = 200

        def __init__(self) -> None:
            events = [
                {"choices": [{"delta": {"content": "hel"}}]},
                {"choices": [{"delta": {"content": "lo", "role": "assistant"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"completion_tokens": 2}},
            ]
            self.content = AsyncLines(
                [f"data: {json.dumps(event)}\n".encode() for event in events]
                + [b"data: [DONE]\n"]
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def text(self) -> str:
            return ""

    class Session:
        def __init__(self) -> None:
            self.payload = None

        def post(self, url, *, json):
            del url
            self.payload = json
            return Response()

    session = Session()
    args = SimpleNamespace(
        model="qwen3-omni",
        audio_path=None,
        max_tokens=8,
        base_url="http://localhost:8000",
    )
    result = asyncio.run(
        _measure_one(
            session,
            args=args,
            arm=_arm_config("B", None),
            request_id="B-0",
            prompt_id="prompt-0",
            fixture_id="fixture-0",
            prompt="hello",
            sampling_seed=None,
        )
    )

    assert result.success
    assert result.complete_output_text == "hello"
    assert result.output_sha256 == hashlib.sha256(b"hello").hexdigest()
    assert result.finish_reason == "stop"
    assert result.completion_tokens == 2
    assert session.payload["stream_options"] == {"include_usage": True}
    assert "seed" not in session.payload

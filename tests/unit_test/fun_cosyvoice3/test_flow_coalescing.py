# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages


def _empty_flow_input() -> stages.FlowBatchInput:
    return stages.FlowBatchInput(
        token=torch.empty((1, 0), dtype=torch.int32),
        prompt_token=torch.empty((1, 0), dtype=torch.int32),
        prompt_feat=torch.empty((1, 0, 80)),
        embedding=torch.empty((1, 192)),
    )


def _make_buckets(
    total_mel_frames: list[int], *, bucket_frames: int = 50
) -> dict[int, list[stages._PreparedFlowRequest]]:
    buckets: dict[int, list[stages._PreparedFlowRequest]] = {}
    for index, total in enumerate(total_mel_frames):
        bucket_key = (total + bucket_frames - 1) // bucket_frames
        buckets.setdefault(bucket_key, []).append(
            stages._PreparedFlowRequest(
                index=index,
                sample_rate=24000,
                flow_input=_empty_flow_input(),
                total_mel_frames=total,
                baseline_bucket_key=bucket_key,
            )
        )
    return buckets


def _group_totals(
    groups: list[list[stages._PreparedFlowRequest]],
) -> list[list[int]]:
    return [[request.total_mel_frames for request in group] for group in groups]


def _group_indices(
    groups: list[list[stages._PreparedFlowRequest]],
) -> list[list[int]]:
    return [[request.index for request in group] for group in groups]


def _partition_objective(
    groups: list[list[stages._PreparedFlowRequest]],
) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    work = 0
    maximum_merged_span = 0
    signature: list[tuple[int, int]] = []
    for group in groups:
        totals = [request.total_mel_frames for request in group]
        keys = [request.baseline_bucket_key for request in group]
        work += len(group) * max(totals)
        if len(set(keys)) > 1:
            maximum_merged_span = max(maximum_merged_span, max(totals) - min(totals))
        signature.append((keys[0], keys[-1]))
    return len(groups), work, maximum_merged_span, tuple(signature)


def _brute_force_partition(
    buckets: dict[int, list[stages._PreparedFlowRequest]],
    *,
    coalesce_span_frames: int,
    coalesce_max_added_padding_pct: float,
) -> tuple[
    tuple[int, int, int, tuple[tuple[int, int], ...]],
    list[list[int]],
]:
    """Independent exhaustive oracle over contiguous atomic-bucket partitions."""
    atomic = [(key, tuple(requests)) for key, requests in sorted(buckets.items())]
    baseline_work = sum(
        len(requests) * max(request.total_mel_frames for request in requests)
        for _, requests in atomic
    )
    best_objective = None
    best_groups: list[list[int]] | None = None

    for mask in range(1 << max(0, len(atomic) - 1)):
        ranges: list[tuple[int, int]] = []
        start = 0
        for boundary in range(len(atomic) - 1):
            if mask & (1 << boundary):
                ranges.append((start, boundary + 1))
                start = boundary + 1
        ranges.append((start, len(atomic)))

        groups: list[list[stages._PreparedFlowRequest]] = []
        valid = True
        for start, end in ranges:
            requests = [
                request
                for _, atomic_requests in atomic[start:end]
                for request in atomic_requests
            ]
            totals = [request.total_mel_frames for request in requests]
            if end - start > 1 and max(totals) - min(totals) > coalesce_span_frames:
                valid = False
                break
            groups.append(requests)
        if not valid:
            continue

        objective = _partition_objective(groups)
        added_padding_pct = (objective[1] / baseline_work - 1) * 100
        if added_padding_pct > coalesce_max_added_padding_pct + 1e-9:
            continue
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_groups = _group_indices(groups)

    assert best_objective is not None
    assert best_groups is not None
    return best_objective, best_groups


@pytest.mark.parametrize(
    ("total_mel_frames", "bucket_frames", "span_frames", "padding_pct", "expected"),
    [
        pytest.param(
            [48, 50, 54],
            50,
            1,
            5,
            [[48, 50], [54]],
            id="atomic-baseline-buckets",
        ),
        pytest.param(
            [10, 13, 30, 33],
            10,
            4,
            5,
            [[10], [13], [30, 33]],
            id="whole-outer-padding-cap",
        ),
        pytest.param(
            [10, 10, 20, 40],
            10,
            40,
            30,
            [[10, 10, 20], [40]],
            id="smallest-maximum-merge-span",
        ),
        pytest.param(
            [10, 20, 30],
            10,
            20,
            40,
            [[10], [20, 30]],
            id="bucket-range-signature",
        ),
    ],
)
def test_flow_coalescing_policy_directly(
    total_mel_frames: list[int],
    bucket_frames: int,
    span_frames: int,
    padding_pct: float,
    expected: list[list[int]],
) -> None:
    groups = stages._group_flow_requests(
        _make_buckets(total_mel_frames, bucket_frames=bucket_frames),
        coalesce_span_frames=span_frames,
        coalesce_max_added_padding_pct=padding_pct,
    )

    assert _group_totals(groups) == expected


def test_flow_coalescing_disabled_preserves_bucket_insertion_order() -> None:
    buckets = _make_buckets([104, 50, 54])
    groups = stages._group_flow_requests(
        buckets,
        coalesce_span_frames=0,
        coalesce_max_added_padding_pct=0,
    )

    assert _group_totals(groups) == [[104], [50], [54]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"coalesce_span_frames": -1}, "span_frames"),
        ({"coalesce_max_added_padding_pct": -1}, "padding_pct"),
        ({"coalesce_max_added_padding_pct": float("nan")}, "padding_pct"),
        ({"coalesce_max_added_padding_pct": float("inf")}, "padding_pct"),
        (
            {
                "coalesce_span_frames": 0,
                "coalesce_max_added_padding_pct": 1,
            },
            "padding_pct",
        ),
    ],
)
def test_flow_coalescing_validation(kwargs: dict[str, object], message: str) -> None:
    config: dict[str, object] = {
        "coalesce_span_frames": 64,
        "coalesce_max_added_padding_pct": 5.0,
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        stages._group_flow_requests(_make_buckets([10, 20]), **config)


def test_flow_coalescing_differential_bruteforce_oracle() -> None:
    rng = random.Random(1899)
    span_choices = [1, 5, 10, 20, 50, 100, 1000]
    padding_choices = [0, 1, 2.5, 5, 10, 25, 100]

    for _ in range(500):
        atomic_count = rng.randint(1, 8)
        bucket_frames = rng.choice([1, 5, 10, 50])
        keys = sorted(rng.sample(range(1, atomic_count + 5), atomic_count))
        totals_by_key: dict[int, list[int]] = {}
        for key in keys:
            lower = (key - 1) * bucket_frames + 1
            upper = key * bucket_frames
            totals_by_key[key] = [
                rng.randint(lower, upper) for _ in range(rng.randint(1, 3))
            ]

        buckets: dict[int, list[stages._PreparedFlowRequest]] = {}
        index = 0
        for key in rng.sample(keys, len(keys)):
            for total in totals_by_key[key]:
                buckets.setdefault(key, []).append(
                    stages._PreparedFlowRequest(
                        index=index,
                        sample_rate=24000,
                        flow_input=_empty_flow_input(),
                        total_mel_frames=total,
                        baseline_bucket_key=key,
                    )
                )
                index += 1

        span_frames = rng.choice(span_choices)
        padding_pct = rng.choice(padding_choices)
        expected_objective, expected_indices = _brute_force_partition(
            buckets,
            coalesce_span_frames=span_frames,
            coalesce_max_added_padding_pct=padding_pct,
        )
        actual_groups = stages._group_flow_requests(
            buckets,
            coalesce_span_frames=span_frames,
            coalesce_max_added_padding_pct=padding_pct,
        )

        assert _partition_objective(actual_groups) == expected_objective
        assert _group_indices(actual_groups) == expected_indices


def test_flow_coalescing_handles_sixteen_requests_without_batch_guard() -> None:
    totals = list(range(10, 161, 10))
    groups = stages._group_flow_requests(
        _make_buckets(totals, bucket_frames=10),
        coalesce_span_frames=150,
        coalesce_max_added_padding_pct=100,
    )

    assert _group_indices(groups) == [list(range(16))]

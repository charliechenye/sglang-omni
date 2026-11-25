#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark Qwen3-TTS speculative kernels on frozen checkpoint tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglang_omni.utils.qwen3_tts_speculative_frozen import (
    load_frozen_predictor_artifacts,
)
from sglang_omni.utils.qwen3_tts_speculative_kernel_microbench import (
    run_microbenchmarks,
)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    else:
        pass
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-limit", type=positive_integer, default=16)
    parser.add_argument("--warmup", type=positive_integer, default=5)
    parser.add_argument("--iterations", type=positive_integer, default=30)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="all"
    )
    parser.add_argument("--expected-model-snapshot")
    arguments = parser.parse_args()
    artifacts = load_frozen_predictor_artifacts(
        arguments.capture_root,
        arguments.dataset_root,
        frame_limit=arguments.frame_limit,
        split=arguments.split,
        expected_model_snapshot=arguments.expected_model_snapshot,
    )
    result = run_microbenchmarks(
        artifacts,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

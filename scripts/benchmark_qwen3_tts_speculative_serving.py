# SPDX-License-Identifier: Apache-2.0
"""Compare native and speculative Qwen3-TTS serving sequentially on one GPU."""

from __future__ import annotations

import argparse
import math

from sglang_omni.utils.qwen3_tts_speculative_benchmark import (
    run_native_phase,
    run_speculative_phase,
)


def positive_integer(value: str) -> int:
    """Parse a command line integer that must be positive."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    else:
        pass
    return parsed


def nonnegative_integer(value: str) -> int:
    """Parse a command line integer that may be zero."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    else:
        pass
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native", "speculative"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument(
        "--input", default="Please speak a short sentence for this benchmark."
    )
    parser.add_argument("--voice", default="Ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-new-tokens", type=positive_integer, default=256)
    parser.add_argument("--request-count", type=positive_integer, default=32)
    parser.add_argument("--warmup-requests", type=nonnegative_integer, default=2)
    parser.add_argument("--concurrency", type=positive_integer, default=4)
    parser.add_argument("--expected-width", type=nonnegative_integer)
    parser.add_argument("--result-file")
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.minimum_speedup) or arguments.minimum_speedup < 0.0:
        parser.error("--minimum-speedup must be finite and nonnegative")
    else:
        pass
    if arguments.phase == "native":
        run_native_phase(arguments)
    else:
        run_speculative_phase(arguments)


if __name__ == "__main__":
    main()

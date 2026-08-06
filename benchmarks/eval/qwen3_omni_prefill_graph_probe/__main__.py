# SPDX-License-Identifier: Apache-2.0
"""Launch a target server with benchmark-only graph observation enabled."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requested-prefill-backend", default="breakable")
    parser.add_argument(
        "--compatibility-injection", choices=("on", "off"), default="on"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command or args.command[0] != "--":
        parser.error("append the target command after --")

    env = os.environ.copy()
    probe_dir = str(Path(__file__).resolve().parent)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        probe_dir
        if not existing_pythonpath
        else probe_dir + os.pathsep + existing_pythonpath
    )
    env["QWEN3_OMNI_GRAPH_PROBE_OUTPUT"] = str(args.output.resolve())
    env["QWEN3_OMNI_GRAPH_PROBE_REQUESTED_BACKEND"] = (
        args.requested_prefill_backend
    )
    env["QWEN3_OMNI_GRAPH_PROBE_COMPAT"] = (
        "1" if args.compatibility_injection == "on" else "0"
    )
    completed = subprocess.run(args.command[1:], env=env, check=False)
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "instrumentation_error": (
                        "Target exited before probe sitecustomize wrote a report"
                    ),
                    "target_returncode": completed.returncode,
                },
                indent=2,
            )
            + "\n"
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# P15 Terminal B, while dedicated H200 worker from p15_launch.sh is healthy.
set -euo pipefail
ROOT="$HOME/results/qwen3tts-cpspec-k2-p15-001"
if [[ $# -ge 1 ]]; then ROOT="$1"; fi
if [[ ! -r "$ROOT/source.sha" || ! -r "$ROOT/server.log" ]]; then
  echo "P15 missing source.sha/server.log for dedicated worker: $ROOT" >&2
  exit 2
fi
PINNED_SHA="$(cat "$ROOT/source.sha")"
LOCAL_SHA="$(git rev-parse HEAD)"
if [[ "$LOCAL_SHA" != "$PINNED_SHA" ]]; then
  echo "P15 source drift: worker=$PINNED_SHA local=$LOCAL_SHA" >&2
  exit 2
fi
if [[ -e "$ROOT/smoke" ]]; then
  echo "P15 must use a fresh root, not overwrite previous smoke" >&2
  exit 2
fi
python -m experiments.qwen3_tts_cp_spec_k2.p15_two_round_gate \
  --base-url http://127.0.0.1:8000 \
  --base-arm "$ROOT/ARM_BASE" \
  --spec-arm "$ROOT/ARM_SPEC" \
  --baseline-dir "$ROOT/baseline" \
  --probe-dir "$ROOT/two_round" \
  --server-log "$ROOT/server.log" \
  --expect-source-sha "$PINNED_SHA" \
  --out "$ROOT/smoke"
cat "$ROOT/smoke/summary.md"

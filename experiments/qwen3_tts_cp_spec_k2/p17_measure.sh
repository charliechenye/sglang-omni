#!/usr/bin/env bash
# P17 Terminal B after server /health is ready.
set -euo pipefail
ROOT="$HOME/results/qwen3tts-cpspec-p17-multiwidth-001"
if [[ $# -ge 1 ]]; then ROOT="$1"; fi
if [[ ! -r "$ROOT/source.sha" || ! -r "$ROOT/server.log" ]]; then
  echo "P17 source-locked worker files missing" >&2; exit 2
fi
PIN="$(cat "$ROOT/source.sha")"
test "$PIN" = "$(git rev-parse HEAD)" || {
  echo "P17 worker-source HEAD drift" >&2; exit 2;
}
test ! -e "$ROOT/smoke" || {
  echo "P17 smoke already exists: choose new root" >&2; exit 2;
}
python -m experiments.qwen3_tts_cp_spec_k2.p17_multiwidth_gate \
  --base-url http://127.0.0.1:8000 \
  --base-arm "$ROOT/ARM_BASE" \
  --spec-arm "$ROOT/ARM_SPEC" \
  --baseline-dir "$ROOT/baseline" \
  --probe-dir "$ROOT/wide" \
  --server-log "$ROOT/server.log" \
  --expect-source-sha "$PIN" \
  --out "$ROOT/smoke"
cat "$ROOT/smoke/summary.md"

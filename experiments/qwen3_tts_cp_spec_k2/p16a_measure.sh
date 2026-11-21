#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/results/qwen3tts-cpspec-k2-p16a-001"
if [[ $# -ge 1 ]]; then ROOT="$1"; fi
if [[ ! -r "$ROOT/source.sha" || ! -r "$ROOT/server.log" ]]; then
  echo "P16A dedicated worker evidence missing" >&2; exit 2
fi
PIN="$(cat "$ROOT/source.sha")"
test "$PIN" = "$(git rev-parse HEAD)" || {
  echo "P16A worker/source drift" >&2; exit 2;
}
test ! -e "$ROOT/smoke" || {
  echo "P16A smoke exists: use a new result root" >&2; exit 2;
}
python -m experiments.qwen3_tts_cp_spec_k2.p16a_gate \
  --base-url http://127.0.0.1:8000 \
  --base-arm "$ROOT/ARM_BASE" \
  --spec-arm "$ROOT/ARM_SPEC" \
  --baseline-dir "$ROOT/baseline" \
  --probe-dir "$ROOT/two_round" \
  --server-log "$ROOT/server.log" \
  --expect-source-sha "$PIN" \
  --out "$ROOT/smoke"
cat "$ROOT/smoke/p16a-summary.md"
echo "P16A_NEXT=stop the Nsight worker in Terminal A to finalize .nsys-rep"

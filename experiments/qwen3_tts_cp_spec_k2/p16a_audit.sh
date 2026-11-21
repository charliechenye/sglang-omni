#!/usr/bin/env bash
# Final Nsight node-level audit. Run only AFTER the worker exits.
# Second argument (optional) = real LM-head CUDA kernel regex verified by user.
set -euo pipefail
ROOT="$HOME/results/qwen3tts-cpspec-k2-p16a-001"
if [[ $# -ge 1 ]]; then ROOT="$1"; fi
if [[ ! -s "$ROOT/nsys/p16a.nsys-rep" ||
      ! -s "$ROOT/smoke/report.json" ]]; then
  echo "P16A requires finalized Nsight report + source-locked smoke" >&2; exit 2
fi
test "$(cat "$ROOT/source.sha")" = "$(git rev-parse HEAD)" || {
  echo "P16A source drift" >&2; exit 2;
}
ARGS=()
OUT="$ROOT/trace-audit-markers"
if [[ $# -ge 2 ]]; then
  ARGS=(--bonus-head-regex "$2")
  OUT="$ROOT/trace-audit-head"
fi
test ! -e "$OUT" || {
  echo "P16A audit output exists: $OUT" >&2; exit 2;
}
python -m experiments.qwen3_tts_cp_spec_k2.p16a_trace_audit \
  --smoke-report "$ROOT/smoke/report.json" \
  --nsys-report "$ROOT/nsys/p16a.nsys-rep" \
  --out "$OUT" \
  "${ARGS[@]}"
cat "$OUT/trace-audit.md"

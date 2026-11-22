#!/usr/bin/env bash
# P18 Terminal B: source-locked real native-AR acceptance -> optional P17 cost.
# usage: bash p18_measure.sh ABSOLUTE_NEW_SERVER_ROOT smoke|main [P17_REPORT]
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: bash p18_measure.sh ROOT smoke|main [P17_SMOKE_REPORT]" >&2
  exit 2
fi
ROOT="$(realpath "$1")"
MODE="$2"
if [[ "$MODE" != "smoke" && "$MODE" != "main" ]]; then
  echo "P18 mode must be smoke or main" >&2; exit 2
fi
if [[ $# -eq 3 && "$MODE" != "main" ]]; then
  echo "P18B is forbidden on the smoke cohort" >&2; exit 2
fi
if [[ $# -eq 3 && ! -r "$3" ]]; then
  echo "P18B prior P17 cost report missing or unreadable: $3" >&2; exit 2
fi
if [[ -e "$ROOT/visits.json" || -e "$ROOT/analysis" || -e "$ROOT/cost-model" ]]; then
  echo "P18 requires fresh client evidence under this server root" >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p18-depth-census" ]]; then
  echo "P18 runner branch mismatch" >&2; exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$(cat "$ROOT/source.sha")" ]]; then
  echo "P18 client/worker source SHA mismatch" >&2; exit 2
fi
export PYTHONPATH="$(git rev-parse --show-toplevel)${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.qwen3_tts_cp_spec_k2.p18_run \
  --root "$ROOT" --mode "$MODE"
python -m experiments.qwen3_tts_cp_spec_k2.p18_analyze \
  --root "$ROOT" --out "$ROOT/analysis"
cat "$ROOT/analysis/summary.md"
if [[ $# -eq 3 ]]; then
  python -m experiments.qwen3_tts_cp_spec_k2.p18_cost_model \
    --root "$ROOT" --p17-cost-report "$3" \
    --out "$ROOT/cost-model"
  cat "$ROOT/cost-model/summary.md"
fi

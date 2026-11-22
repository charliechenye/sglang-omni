#!/usr/bin/env bash
# P19 Terminal B: matched P18 native AR+spec census, costs and Bellman DP.
# usage: bash p19_measure.sh ABSENT_CLIENT_ROOT smoke|main
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: bash p19_measure.sh ROOT smoke|main" >&2; exit 2
fi
ROOT="$(realpath "$1")"
MODE="$2"
if [[ "$MODE" != "smoke" && "$MODE" != "main" ]]; then
  echo "P19 mode must be smoke or main" >&2; exit 2
fi
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p19-matched-dp" ]]; then
  echo "P19 runner branch mismatch" >&2; exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$(cat "$ROOT/source.sha")" ]]; then
  echo "P19 client/worker source SHA mismatch" >&2; exit 2
fi
if [[ -e "$ROOT/visits.json" || -e "$ROOT/analysis" || -e "$ROOT/matched_cost/analysis" || -e "$ROOT/dp" ]]; then
  echo "P19 requires fresh evidence; never overwrite an attempt" >&2; exit 2
fi
if [[ ! -f "$ROOT/matched_cost/config.json" ]]; then
  echo "P19 matching cost config missing: did you use p19_launch.sh?" >&2
  exit 2
fi
export PYTHONPATH="$(git rev-parse --show-toplevel)${PYTHONPATH:+:$PYTHONPATH}"
python - "$ROOT" "$MODE" <<'PY'
import json, sys
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
p18 = json.loads((root / "config.json").read_text())
cost = json.loads((root / "matched_cost" / "config.json").read_text())
expected = 2 if mode == "smoke" else 8
if (p18.get("frames_per_request") != expected
        or cost.get("p18_frames_per_request") != expected
        or p18.get("source_sha") != cost.get("source_sha")):
    raise RuntimeError("P19_CLIENT_COHORT_OR_COST_SOURCE_DRIFT")
PY
python -m experiments.qwen3_tts_cp_spec_k2.p18_run \
  --root "$ROOT" --mode "$MODE"
python -m experiments.qwen3_tts_cp_spec_k2.p18_analyze \
  --root "$ROOT" --out "$ROOT/analysis"
python -m experiments.qwen3_tts_cp_spec_k2.p19_analyze \
  --root "$ROOT" --out "$ROOT/matched_cost/analysis"
cat "$ROOT/matched_cost/analysis/summary.md"
python -m experiments.qwen3_tts_cp_spec_k2.p19_dp \
  --root "$ROOT" --out "$ROOT/dp"
cat "$ROOT/dp/summary.md"

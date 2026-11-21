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
# Old captured data stays pinned to its ORIGINAL serving SHA, while
# diagnostics/auditor may be updated later. Do not demand HEAD==serving SHA.
python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
r = Path(sys.argv[1])
pin = (r / "source.sha").read_text().strip()
report = json.loads((r / "smoke/report.json").read_text())
probe = report.get("two_round_probe", {})
assert len(pin) == 40, "P16A invalid original worker source SHA"
assert report.get("source_sha") == pin, "P16A smoke/worker SHA mismatch"
assert probe.get("source_sha") == pin, "P16A probe/worker SHA mismatch"
assert probe.get("pid") == report.get("pid"), "P16A cross-worker PID"
assert probe.get("gpu") == report.get("gpu"), "P16A cross-GPU evidence"
print("P16A_OFFLINE_SOURCE_LOCK_PASS", pin)
PY
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

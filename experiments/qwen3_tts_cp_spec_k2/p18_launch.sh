#!/usr/bin/env bash
# P18A Terminal A: native-AR-EAGER on-policy census; no serving perf claims.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: P18_CAPTURE_FRAMES=4 bash p18_launch.sh ABSENT_ROOT" >&2; exit 2
fi
ROOT="$(realpath -m "$1")"
if [[ -e "$ROOT" ]]; then
  echo "P18 requires NEW absent evidence root: $ROOT" >&2; exit 2
fi
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p18-depth-census" ]]; then
  echo "P18 requires its dedicated experiment branch" >&2; exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "P18 requires a clean tracked worktree" >&2; exit 2
fi
WORKTREE="$(git rev-parse --show-toplevel)"
if [[ "$(pwd -P)" != "$WORKTREE" ]]; then
  echo "P18 launch must run at its worktree root" >&2; exit 2
fi
FRAMES="${P18_CAPTURE_FRAMES:-4}"
if [[ ! "$FRAMES" =~ ^[0-9]+$ ]] || (( FRAMES < 1 || FRAMES > 16 )); then
  echo "P18_CAPTURE_FRAMES must be 1..16" >&2; exit 2
fi

# Force spawn children to import this dedicated worktree. No global pip edits.
export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import importlib
from pathlib import Path
root = Path.cwd().resolve()
for name in (
    "sglang_omni",
    "sglang_omni.models.qwen3_tts.engine_builder",
    "sglang_omni.models.qwen3_tts.model_runner",
    "sglang_omni.models.qwen3_tts.sglang_model",
    "sglang_omni.models.qwen3_tts.code_predictor_spec_p18_census",
):
    module = importlib.import_module(name)
    file = Path(module.__file__).resolve()
    print(f"P18_IMPORT_PREFLIGHT {name}={file}", flush=True)
    if not file.is_relative_to(root / "sglang_omni"):
        raise RuntimeError(f"P18_IMPORT_SOURCE_MISMATCH: {name}")
from sglang_omni.models.qwen3_tts import code_predictor_spec_p18_census as probe
assert probe.feasible_widths(1) == (2, 3, 4)
assert probe.feasible_widths(12) == (2, 3)
assert probe.feasible_widths(13) == (2,)
assert probe.feasible_widths(14) == ()
print("P18_IMPORT_PREFLIGHT_PASS", flush=True)
PY
python - <<'PY'
import torch
if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
    raise RuntimeError("P18 requires one NVIDIA H200")
print("P18_DEVICE:", torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p "$ROOT/census/rows" "$ROOT/census/failures"
SHA="$(git rev-parse HEAD)"
printf '%s\n' "$SHA" > "$ROOT/source.sha"
python - "$ROOT" "$SHA" "$FRAMES" <<'PY'
import json
import sys
from pathlib import Path
root, sha, frames = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
(root / "config.json").write_text(json.dumps({
    "schema": "p18a-native-ar-eager-launch-v1",
    "source_sha": sha, "frames_per_request": frames,
    "gpu": "NVIDIA H200", "single_request": True,
    "native_k2": "OFF", "trajectory": "native-ar-eager-only",
    "sampling": {"temperature": 0.9, "top_k": 50, "top_p": 1.0},
}, indent=2) + "\n")
PY

# Isolation: baseline Graph ON when unarmed; P18 armed intercepts the native
# eager Predictor and restores K/V from every counterfactual. Other probes OFF.
for key in \
  SGLANG_OMNI_QTTS_CP_P4_DIR SGLANG_OMNI_QTTS_CP_P5_DIR \
  SGLANG_OMNI_QTTS_CP_P9_BASE_DIR SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE \
  SGLANG_OMNI_QTTS_CP_P7_DIR SGLANG_OMNI_QTTS_CP_P8_DIR \
  SGLANG_OMNI_QTTS_CP_P9A_DIR SGLANG_OMNI_QTTS_CP_P9B_DIR \
  SGLANG_OMNI_QTTS_CP_P10A_DIR SGLANG_OMNI_QTTS_CP_P10B_DIR \
  SGLANG_OMNI_QTTS_CP_P11A_DIR SGLANG_OMNI_QTTS_CP_P12A_DIR \
  SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR \
  SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR \
  SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR \
  SGLANG_OMNI_QTTS_CP_P16A_DIR \
  SGLANG_OMNI_QTTS_CP_P17_MULTIWIDTH_DIR \
  SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR \
  SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_ONLY \
  SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR \
  SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE \
  SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH \
  SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT \
  SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE; do
  unset "$key"
done
export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_STRICT_PORT=1
export SGLANG_OMNI_QTTS_CP_SPEC_K=0
export SGLANG_OMNI_QTTS_CP_P18_DIR="$ROOT/census"
export SGLANG_OMNI_QTTS_CP_P18_ARM_FILE="$ROOT/ARM_P18"
export SGLANG_OMNI_QTTS_CP_P18_CASE_FILE="$ROOT/ACTIVE_CASE.json"
export SGLANG_OMNI_QTTS_CP_P18_FRAMES_PER_REQUEST="$FRAMES"
export SGLANG_OMNI_QTTS_CP_P18_SOURCE_SHA="$SHA"
test ! -e "$ROOT/ARM_P18"
test ! -e "$ROOT/ACTIVE_CASE.json"
echo "P18_SOURCE_SHA=$SHA"
echo "P18_NATIVE_EAGER_CENSUS_ROOT=$ROOT"
echo "P18_CAPTURE_FRAMES=$FRAMES"
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$ROOT/server.log"

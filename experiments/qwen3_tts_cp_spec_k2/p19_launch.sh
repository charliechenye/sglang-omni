#!/usr/bin/env bash
# P19A Terminal A: matched AR vs wide graph costs on native-AR-eager P18 checkpoints.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: P18_CAPTURE_FRAMES=2|8 P19_COST_FRAMES=1 bash p19_launch.sh ABSENT_ROOT" >&2; exit 2
fi
ROOT="$(realpath -m "$1")"
if [[ -e "$ROOT" ]]; then
  echo "P19 requires NEW absent evidence root: $ROOT" >&2; exit 2
fi
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p19-matched-dp" ]]; then
  echo "P19 requires its dedicated experiment branch" >&2; exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "P19 requires a clean tracked worktree" >&2; exit 2
fi
WORKTREE="$(git rev-parse --show-toplevel)"
if [[ "$(pwd -P)" != "$WORKTREE" ]]; then
  echo "P19 launch must run at its worktree root" >&2; exit 2
fi
FRAMES="${P18_CAPTURE_FRAMES:-2}"
if [[ "$FRAMES" != 2 && "$FRAMES" != 8 ]]; then
  echo "P19 requires P18_CAPTURE_FRAMES=2 (smoke) or 8 (main)" >&2; exit 2
fi
COST_FRAMES="${P19_COST_FRAMES:-1}"
if [[ "$COST_FRAMES" != 1 && "$COST_FRAMES" != 2 ]]; then
  echo "P19_COST_FRAMES must be 1 or 2" >&2; exit 2
fi
if (( COST_FRAMES > FRAMES )); then
  echo "P19_COST_FRAMES must not exceed P18_CAPTURE_FRAMES" >&2; exit 2
fi

# One worker per diagnostic arm, no multi-arm evidence or DP claim.
if [[ -n "${P19_DIAG_ARM:-}" ]]; then
  case "$P19_DIAG_ARM" in
    OFF|AR_EAGER|AR_GRAPH|K2_EAGER|K2_GRAPH|K3_EAGER|K3_GRAPH|K4_EAGER|K4_GRAPH|ALL_GRAPH|ALL_DEPTHS_GRAPH|ALL_DEPTHS_EAGER) ;;
    *) echo "P19_DIAG_ARM invalid: $P19_DIAG_ARM" >&2; exit 2 ;;
  esac
  DEPTH="${P19_DIAG_DEPTH:-1}"
  if [[ ! "$DEPTH" =~ ^[0-9]+$ ]] || (( DEPTH < 1 || DEPTH > 14 )); then
    echo "P19_DIAG_DEPTH must be 1..14" >&2; exit 2
  fi
  if [[ "$P19_DIAG_ARM" == K2_* && "$DEPTH" -gt 13 ]] ||
     [[ "$P19_DIAG_ARM" == K3_* && "$DEPTH" -gt 12 ]] ||
     [[ "$P19_DIAG_ARM" == K4_* && "$DEPTH" -gt 11 ]]; then
    echo "P19 diagnostic requested an unavailable bonus head" >&2; exit 2
  fi
  if [[ "$P19_DIAG_ARM" == ALL_DEPTHS_* && "$COST_FRAMES" != 1 ]]; then
    echo "P19 ALL_DEPTHS_* requires P19_COST_FRAMES=1" >&2; exit 2
  fi
  FENCE="${P19_DIAG_FENCE:-0}"
  if [[ "$FENCE" != 0 && "$FENCE" != 1 ]]; then
    echo "P19_DIAG_FENCE must be 0 or 1" >&2; exit 2
  fi
  export SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM="$P19_DIAG_ARM"
  export SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH="$DEPTH"
  export SGLANG_OMNI_QTTS_CP_P19_DIAG_FENCE="$FENCE"
  NATIVE_FENCE="${P19_DIAG_NATIVE_FENCE:-0}"
  if [[ "$NATIVE_FENCE" != 0 && "$NATIVE_FENCE" != 1 ]]; then
    echo "P19_DIAG_NATIVE_FENCE must be 0 or 1" >&2; exit 2
  fi
  export SGLANG_OMNI_QTTS_CP_P19_NATIVE_FENCE="$NATIVE_FENCE"
  TAIL_FENCE="${P19_DIAG_TAIL_FENCE:-0}"
  if [[ "$TAIL_FENCE" != 0 && "$TAIL_FENCE" != 1 ]]; then
    echo "P19_DIAG_TAIL_FENCE must be 0 or 1" >&2; exit 2
  fi
  export SGLANG_OMNI_QTTS_CP_P19_TAIL_FENCE="$TAIL_FENCE"
else
  unset SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM
  unset SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH
  unset SGLANG_OMNI_QTTS_CP_P19_DIAG_FENCE
  unset SGLANG_OMNI_QTTS_CP_P19_NATIVE_FENCE
  unset SGLANG_OMNI_QTTS_CP_P19_TAIL_FENCE
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
    "sglang_omni.models.qwen3_tts.code_predictor_spec_p19_cost",
    "sglang_omni.models.qwen3_tts.code_predictor_spec_p19_diag",
):
    module = importlib.import_module(name)
    file = Path(module.__file__).resolve()
    print(f"P19_IMPORT_PREFLIGHT {name}={file}", flush=True)
    if not file.is_relative_to(root / "sglang_omni"):
        raise RuntimeError(f"P19_IMPORT_SOURCE_MISMATCH: {name}")
import inspect
from sglang_omni.models.qwen3_tts import code_predictor_spec_p18_census as probe
if "maybe_measure(" not in inspect.getsource(probe.maybe_census):
    raise RuntimeError("P19_HOOK_MISSING_FROM_LIVE_P18_NATIVE_CENSUS")
assert probe.feasible_widths(1) == (2, 3, 4)
assert probe.feasible_widths(12) == (2, 3)
assert probe.feasible_widths(13) == (2,)
assert probe.feasible_widths(14) == ()
print("P19_IMPORT_PREFLIGHT_PASS", flush=True)
PY
python - <<'PY'
import torch
if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
    raise RuntimeError("P19 requires one NVIDIA H200")
print("P19_DEVICE:", torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p "$ROOT/census/rows" "$ROOT/census/failures"
mkdir -p "$ROOT/matched_cost/rows" "$ROOT/matched_cost/failures"
if [[ -n "${P19_DIAG_ARM:-}" ]]; then
  mkdir -p "$ROOT/matched_cost/diagnostic"
fi
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
export SGLANG_OMNI_QTTS_CP_P19_SOURCE_SHA="$SHA"
export SGLANG_OMNI_QTTS_CP_P19_COST_DIR="$ROOT/matched_cost"
export SGLANG_OMNI_QTTS_CP_P19_FRAMES_PER_REQUEST="$COST_FRAMES"
test ! -e "$ROOT/ARM_P18"
test ! -e "$ROOT/ACTIVE_CASE.json"
echo "P18_SOURCE_SHA=$SHA"
echo "P18_NATIVE_EAGER_CENSUS_ROOT=$ROOT"
echo "P18_CAPTURE_FRAMES=$FRAMES"
echo "P19_MATCHED_COST_FRAMES=$COST_FRAMES"
if [[ -n "${P19_DIAG_ARM:-}" ]]; then
  echo "P19_DIAGNOSTIC_ONLY arm=$P19_DIAG_ARM depth=$DEPTH fence=$FENCE native_fence=$NATIVE_FENCE tail_fence=$TAIL_FENCE"
fi
python - "$ROOT" "$SHA" "$FRAMES" "$COST_FRAMES" <<'PY'
import json
import os
import sys
from pathlib import Path
root, sha = Path(sys.argv[1]), sys.argv[2]
frames, cost_frames = int(sys.argv[3]), int(sys.argv[4])
diagnostic = None
if os.environ.get("SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM"):
    diagnostic = {
        "arm": os.environ["SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM"],
        "depth": int(os.environ["SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH"]),
        "fence": int(os.environ["SGLANG_OMNI_QTTS_CP_P19_DIAG_FENCE"]),
        "native_fence": int(
            os.environ["SGLANG_OMNI_QTTS_CP_P19_NATIVE_FENCE"]
        ),
        "tail_fence": int(
            os.environ["SGLANG_OMNI_QTTS_CP_P19_TAIL_FENCE"]
        ),
        "policy": "SINGLE_REQUEST_TRACE_ONLY_NO_COST_SELECTION",
    }
(root / "matched_cost" / "config.json").write_text(json.dumps({
    "diagnostic": diagnostic,
    "schema": "qwen3tts-p19a-matched-cost-config-v1",
    "source_sha": sha, "gpu": "NVIDIA H200",
    "trajectory": "P18_NATIVE_AR_EAGER_SAME_CHECKPOINT",
    "p18_frames_per_request": frames,
    "p19_cost_frames_per_request": cost_frames,
    "arms": ["AR", "2", "3", "4"],
    "native_step_scope": "predictor+lm_head+real_seeded_sampler+embedding",
    "graph_costs": "independently_captured_fixed_state_NOT_FULL_FRAME",
}, indent=2) + "\n")
PY
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$ROOT/server.log"

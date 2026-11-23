#!/usr/bin/env bash
# P22 H200 real-serving K2 profiler: NVTX + post-warmup CUDA capture.
# NEVER use a profiling run for E2E speedup claims.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: bash p22_serving_profile_launch.sh ABSENT_ABSOLUTE_ROOT" >&2
  exit 2
fi
ROOT="$(realpath -m "$1")"
if [[ "$1" != /* || "$ROOT" != "$1" || -e "$ROOT" ]]; then
  echo "P22_PROFILE_REQUIRES_ABSENT_CANONICAL_ABSOLUTE_ROOT" >&2
  exit 2
fi
WORKTREE="$(git rev-parse --show-toplevel)"
BRANCH="$(git branch --show-current)"
if [[ "$(pwd -P)" != "$WORKTREE" || "$BRANCH" != "experiment/qwen3-tts-p22-serving-profile" ]]; then
  echo "P22_PROFILE_REQUIRES_DEDICATED_EXPERIMENT_WORKTREE" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "P22_PROFILE_REQUIRES_CLEAN_WORKTREE" >&2; exit 2
fi
if [[ -n "${SGLANG_OMNI_QTTS_CP_P21_ARM_FILE:-}" ]]; then
  echo "P22_PROFILE_REFUSES_EXTERNAL_ARM_FILE" >&2; exit 2
fi
python - <<'PY'
import socket
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", 8000)) == 0:
        raise SystemExit("P22_PROFILE_PORT_8000_ALREADY_OCCUPIED")
PY
export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
# Pin the same physical H200 for BOTH preflight and the serving process.
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
from pathlib import Path
import importlib
root = Path.cwd().resolve()
for mod in (
    "sglang_omni.models.qwen3_tts.sglang_model",
    "sglang_omni.models.qwen3_tts.model_runner",
    "sglang_omni.models.qwen3_tts.code_predictor_spec_serving_p21",
    "sglang_omni.models.qwen3_tts.code_predictor_spec_p22_profile",
):
    file = Path(importlib.import_module(mod).__file__).resolve()
    if not file.is_relative_to(root / "sglang_omni"):
        raise RuntimeError(f"P21_SOURCE_IMPORT_MISMATCH: {mod}={file}")
    print(f"P21_SOURCE_IMPORT_PASS {mod}={file}", flush=True)
PY
python - <<'PY'
import torch
if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
    raise RuntimeError("P22_PROFILE_REQUIRES_H200")
print(f"P21_H200={torch.cuda.get_device_name(0)}", flush=True)
PY

# Fail closed against leftover prototype/census/Graph-cost interventions
# from previous terminals; do not trust inherited research env settings.
for key in \
 SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE \
 SGLANG_OMNI_QTTS_CP_P18_DIR \
 SGLANG_OMNI_QTTS_CP_P18_ARM_FILE \
 SGLANG_OMNI_QTTS_CP_P18_CASE_FILE \
 SGLANG_OMNI_QTTS_CP_P18_SOURCE_SHA \
 SGLANG_OMNI_QTTS_CP_P19_COST_DIR \
 SGLANG_OMNI_QTTS_CP_P19_SOURCE_SHA \
 SGLANG_OMNI_QTTS_CP_P19_DIAG_ARM \
 SGLANG_OMNI_QTTS_CP_P19_DIAG_DEPTH \
 SGLANG_OMNI_QTTS_CP_P19_DIAG_PROFILE \
 SGLANG_OMNI_QTTS_CP_P19_TAIL_FENCE \
 SGLANG_OMNI_QTTS_CP_P19_NATIVE_FENCE \
 SGLANG_OMNI_QTTS_CP_P19_DIAG_FENCE \
 SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR \
 SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE \
 SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH \
 SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR \
 SGLANG_OMNI_QTTS_CP_P17_MULTIWIDTH_DIR \
 SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR \
 SGLANG_OMNI_QTTS_CP_P16A_DIR \
 SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR \
 SGLANG_OMNI_QTTS_CP_P4_DIR \
 SGLANG_OMNI_QTTS_CP_P5_DIR \
 SGLANG_OMNI_QTTS_CP_P7_DIR \
 SGLANG_OMNI_QTTS_CP_P8_DIR \
 SGLANG_OMNI_QTTS_CP_P9A_DIR \
 SGLANG_OMNI_QTTS_CP_P9B_DIR \
 SGLANG_OMNI_QTTS_CP_P9_BASE_DIR \
 SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE \
 SGLANG_OMNI_QTTS_CP_P10A_DIR \
 SGLANG_OMNI_QTTS_CP_P10B_DIR \
 SGLANG_OMNI_QTTS_CP_P11A_DIR \
 SGLANG_OMNI_QTTS_CP_P12A_DIR \
 SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR \
 SGLANG_OMNI_QTTS_CP_P11B_HYBRID_DIR \
 SGLANG_OMNI_QTTS_CP_P16A_MARKER_DIR \
 SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT \
 SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE \
 SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE; do
  unset "$key"
done

NSYS="${P22_NSYS:-1}"
if [[ "$NSYS" != 0 && "$NSYS" != 1 ]]; then
  echo "P22_NSYS must be 0 or 1" >&2; exit 2
fi
if [[ "$NSYS" == 1 ]]; then
  command -v nsys >/dev/null || {
    echo "P22_NSIGHT_SYSTEMS_CLI_NOT_FOUND" >&2; exit 2;
  }
  nsys --version
fi
mkdir -m 700 -p "$ROOT"
SHA="$(git rev-parse HEAD)"
export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_STRICT_PORT=1
export SGLANG_OMNI_QTTS_CP_SPEC_K=0
export SGLANG_OMNI_QTTS_CP_P21_SERVING_K2=1
export SGLANG_OMNI_QTTS_CP_P21_ARM_FILE="$ROOT/P21_ARM"
export SGLANG_OMNI_QTTS_CP_P21_SOURCE_SHA="$SHA"
export SGLANG_OMNI_QTTS_CP_P22_PROFILE=1
export SGLANG_OMNI_QTTS_CP_P22_CAPTURE_ARM="$ROOT/P22_CAPTURE_ARM"
export SGLANG_OMNI_QTTS_CP_P22_PROFILE_DIR="$ROOT/p22_profiles"
printf '%s\n' "$SHA" > "$ROOT/source.sha"
test ! -e "$SGLANG_OMNI_QTTS_CP_P21_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P22_CAPTURE_ARM"
python - "$ROOT" "$SHA" "$NSYS" <<'PY'
import json,sys
from pathlib import Path
root,sha,nsys = Path(sys.argv[1]),sys.argv[2],bool(int(sys.argv[3]))
(root / "run.json").write_text(json.dumps({
    "schema":"qwen3tts-p22-serving-profile-v1",
    "profiling_enabled":True,
    "nsys_capture_range":nsys,
    "profile_scope":"ONE_WARMUP_B_THEN_ONE_CAPTURED_B",
    "source_sha":sha, "gpu":"NVIDIA H200",
    "model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "baseline":"native CodePredictor CUDA Graph serving (must prove activity)",
    "candidate":"P21 committed K2 wide verifier C1 serving (private draft RNG)",
    "sampling":{"subtalker_dosample":True,
                "temperature":0.9,"top_k":50,"top_p":1.0},
    "P18_P19":"DISABLED", "profiler":"OFF",
    "candidate_default":"OFF_UNTIL_POST_HEALTH_ARM",
    "quality":"NATURAL_STOP_WAV_ONLY_NOT_ASR_OR_SPEAKER_PARITY",
    "performance_claim":"NONE_PROFILER_CONTAMINATED",
},indent=2)+"\n")
PY
echo "P22_PROFILE_SOURCE_SHA=$SHA"
echo "P22_PROFILE_ROOT=$ROOT"
echo "P22_PROFILE_NSYS=$NSYS"
echo "P22_PROFILE_POST_HEALTH_ARM=$SGLANG_OMNI_QTTS_CP_P21_ARM_FILE"
echo "P22_PROFILE_POST_WARMUP_CAPTURE_ARM=$SGLANG_OMNI_QTTS_CP_P22_CAPTURE_ARM"
if [[ "$NSYS" == 1 ]]; then
  # Only the actual B request after B warmup is CUDA-traced. The
  # child TTS worker calls cudaProfilerStart/Stop under a real ARM file.
  nsys profile \
    --trace=cuda,nvtx \
    --target-processes=all \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --output="$ROOT/p22_measured_b" \
    sgl-omni serve \
      --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
      --port 8000 2>&1 | tee "$ROOT/server.log"
else
  # CPU/NVTX attribution only; no GPU profiler API capture.
  sgl-omni serve \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
    --port 8000 2>&1 | tee "$ROOT/server.log"
fi

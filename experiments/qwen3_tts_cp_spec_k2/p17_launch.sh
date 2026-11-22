#!/usr/bin/env bash
# P17 Terminal A: independent H200 process, profiler OFF.
set -euo pipefail
ROOT="$HOME/results/qwen3tts-cpspec-p17-multiwidth-001"
if [[ $# -ge 1 ]]; then ROOT="$1"; fi
if [[ -e "$ROOT" ]]; then
  echo "P17 requires a fresh result directory: $ROOT" >&2; exit 2
fi
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p17-multiwidth" ]]; then
  echo "P17 branch mismatch" >&2; exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "P17 requires clean tracked worktree" >&2; exit 2
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "P17 requires NVIDIA H200"
assert "H200" in torch.cuda.get_device_name(0), "P17 requires H200"
print("P17_CUDA_AVAILABLE", torch.__version__)
PY

mkdir -p "$ROOT"
SHA="$(git rev-parse HEAD)"
printf '%s\n' "$SHA" > "$ROOT/source.sha"

# Absolutely no other experimental probes, hybrid AR or Nsight tracing.
for key in \
  SGLANG_OMNI_QTTS_CP_P4_DIR SGLANG_OMNI_QTTS_CP_P5_DIR \
  SGLANG_OMNI_QTTS_CP_P7_DIR SGLANG_OMNI_QTTS_CP_P8_DIR \
  SGLANG_OMNI_QTTS_CP_P9A_DIR SGLANG_OMNI_QTTS_CP_P9B_DIR \
  SGLANG_OMNI_QTTS_CP_P10A_DIR SGLANG_OMNI_QTTS_CP_P10B_DIR \
  SGLANG_OMNI_QTTS_CP_P11A_DIR SGLANG_OMNI_QTTS_CP_P12A_DIR \
  SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR \
  SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR \
  SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR \
  SGLANG_OMNI_QTTS_CP_P16A_DIR \
  SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR \
  SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE \
  SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH \
  SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT; do
  unset "$key"
done
export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$ROOT/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$ROOT/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$ROOT/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P17_MULTIWIDTH_DIR="$ROOT/wide"
export SGLANG_OMNI_QTTS_CP_P17_MAX_FRAMES_PER_DEPTH=2
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$SHA"
# Never fall back to another port if a stale server owns port 8000.
export SGLANG_OMNI_STRICT_PORT=1
if [[ "${P17_ACTIVATION_DIAG:-0}" == "1" ]]; then
  export SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR="$ROOT/activation"
  export SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_ONLY=1
  mkdir -p "$SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR"
  printf '%s\n' "$SHA" > "$SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR/launcher-source.sha"
  echo "P17_ACTIVATION_ONLY=1 diagnostic_dir=$SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR"
elif [[ "${P17_ACTIVATION_DIAG:-0}" != "0" ]]; then
  echo "P17_ACTIVATION_DIAG must be 0 or 1" >&2
  exit 2
else
  unset SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_DIR
  unset SGLANG_OMNI_QTTS_CP_P17_ACTIVATION_ONLY
fi
test ! -e "$ROOT/ARM_SPEC"
test ! -e "$ROOT/ARM_BASE"
test ! -e "$ROOT/wide"

echo "P17_SOURCE_SHA=$SHA"
echo "P17_OUTPUT_ROOT=$ROOT"
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$ROOT/server.log"

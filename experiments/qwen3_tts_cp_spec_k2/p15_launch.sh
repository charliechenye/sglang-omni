#!/usr/bin/env bash
# P15 Terminal A. Run in its own process so experiment env stays scoped.
set -euo pipefail

ROOT="$HOME/results/qwen3tts-cpspec-k2-p15-001"
if [[ $# -ge 1 ]]; then ROOT="$1"; fi
if [[ -e "$ROOT" ]]; then
  echo "P15 requires absent ROOT to prevent mixed evidence: $ROOT" >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p15-two-round" ]]; then
  echo "P15 requires experiment/qwen3-tts-cpspec-p15-two-round branch" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "P15 requires clean tracked working tree" >&2
  exit 2
fi
mkdir -p "$ROOT"
SHA="$(git rev-parse HEAD)"
printf '%s\n' "$SHA" > "$ROOT/source.sha"

# K2 only becomes active when its post-health ARM file is created by driver.
for key in \
  SGLANG_OMNI_QTTS_CP_P4_DIR \
  SGLANG_OMNI_QTTS_CP_P5_DIR \
  SGLANG_OMNI_QTTS_CP_P7_DIR \
  SGLANG_OMNI_QTTS_CP_P8_DIR \
  SGLANG_OMNI_QTTS_CP_P9A_DIR \
  SGLANG_OMNI_QTTS_CP_P9B_DIR \
  SGLANG_OMNI_QTTS_CP_P10A_DIR \
  SGLANG_OMNI_QTTS_CP_P10B_DIR \
  SGLANG_OMNI_QTTS_CP_P11A_DIR \
  SGLANG_OMNI_QTTS_CP_P12A_DIR \
  SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR \
  SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR \
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
export SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR="$ROOT/two_round"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$SHA"

echo "P15_START_SOURCE_SHA=$SHA"
echo "P15_RESULT_ROOT=$ROOT"
test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$ROOT/server.log"

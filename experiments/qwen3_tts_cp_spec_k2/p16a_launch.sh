#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/results/qwen3tts-cpspec-k2-p16a-001"
if [[ $# -ge 1 ]]; then ROOT="$1"; fi
if [[ -e "$ROOT" ]]; then
  echo "P16A requires a fresh result root: $ROOT" >&2; exit 2
fi
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p16a-branch-trace" ]]; then
  echo "P16A branch mismatch" >&2; exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "P16A requires clean tracked worktree" >&2; exit 2
fi
command -v nsys >/dev/null || { echo "nsys CLI required" >&2; exit 2; }
# Avoid grep -q under pipefail: an early SIGPIPE can give false failure.
NSYS_HELP="$(nsys profile --help 2>&1)"
if [[ "$NSYS_HELP" != *cuda-graph-trace* ]]; then
  echo "nsys node-graph trace unsupported" >&2; exit 2
fi
python - <<'PY'
import torch
from torch._higher_order_ops.cudagraph_conditional_nodes import CUDAGraphCaptureControlFlowOpDispatchMode
assert torch.cuda.is_available() and "H200" in torch.cuda.get_device_name(0)
print("P16A_H200_CONDITIONAL_GRAPH_API_PRESENT", torch.__version__)
PY

mkdir -p "$ROOT/nsys"
SHA="$(git rev-parse HEAD)"
printf '%s\n' "$SHA" > "$ROOT/source.sha"
nsys --version > "$ROOT/nsys/version.txt"

for key in SGLANG_OMNI_QTTS_CP_P4_DIR SGLANG_OMNI_QTTS_CP_P5_DIR SGLANG_OMNI_QTTS_CP_P7_DIR SGLANG_OMNI_QTTS_CP_P8_DIR SGLANG_OMNI_QTTS_CP_P9A_DIR SGLANG_OMNI_QTTS_CP_P9B_DIR SGLANG_OMNI_QTTS_CP_P10A_DIR SGLANG_OMNI_QTTS_CP_P10B_DIR SGLANG_OMNI_QTTS_CP_P11A_DIR SGLANG_OMNI_QTTS_CP_P12A_DIR SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR SGLANG_OMNI_QTTS_CP_P15_TWO_ROUND_DIR SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT; do
  unset "$key"
done

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$ROOT/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$ROOT/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$ROOT/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P16A_DIR="$ROOT/two_round"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$SHA"
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
echo "P16A_SOURCE_SHA=$SHA"
echo "P16A_ROOT=$ROOT"

# Use a bounded NVTX capture. The server remains active after collection;
# stop it only AFTER the ABBA runner finishes to finalize .nsys-rep.
nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --capture-range=nvtx \
  --nvtx-capture='P16A_TRACE@*' \
  --capture-range-end=stop \
  --sample=none --cpuctxsw=none \
  --force-overwrite=false \
  --output="$ROOT/nsys/p16a" \
  sgl-omni serve \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
    --port 8000 \
  2>&1 | tee "$ROOT/server.log"

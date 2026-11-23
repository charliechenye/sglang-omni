#!/usr/bin/env bash
# P22B one REAL native CodePredictor Graph ON request, CUDA Nsight capture.
# No speculative K2/P17/P18/P19 counterfactuals, no production speedup claim.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: bash p22b_native_profile_launch.sh NEW_ABSOLUTE_ROOT" >&2
  exit 2
fi
ROOT="$(realpath -m "$1")"
if [[ "$1" != /* || "$ROOT" != "$1" || -e "$ROOT" ]]; then
  echo "P22B_REQUIRES_ABSENT_CANONICAL_ABSOLUTE_ROOT" >&2; exit 2
fi
WORKTREE="$(git rev-parse --show-toplevel)"
PWD_PHYSICAL="$(pwd -P)"
REPO_PHYSICAL="$(cd "$WORKTREE" && pwd -P)"
BRANCH="$(git symbolic-ref --short -q HEAD || true)"
if [[ "$PWD_PHYSICAL" != "$REPO_PHYSICAL" || "$BRANCH" != "experiment/qwen3-tts-p22b-nvidia-gap" ]]; then
  echo "P22B_WRONG_ROOT_OR_LOCAL_BRANCH" >&2
  echo "PWD=$PWD_PHYSICAL TOP=$REPO_PHYSICAL BRANCH=${BRANCH:-DETACHED_HEAD}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "P22B_DIRTY_WORKTREE" >&2; exit 2
fi
if [[ -n "${SGLANG_OMNI_QTTS_CP_P21_ARM_FILE:-}" || -n "${SGLANG_OMNI_QTTS_CP_P22B_NATIVE_ARM_FILE:-}" ]]; then
  echo "P22B_REFUSES_EXTERNAL_ARM_CONFIG" >&2; exit 2
fi
python - <<'PY'
import socket
with socket.socket() as s:
    if s.connect_ex(("127.0.0.1", 8000)) == 0:
        raise SystemExit("P22B_PORT_8000_OCCUPIED")
PY

NSYS="${P22B_NSYS:-1}"
if [[ "$NSYS" != 0 && "$NSYS" != 1 ]]; then
  echo "P22B_NSYS_MUST_BE_0_OR_1" >&2; exit 2
fi
if [[ "$NSYS" == 1 ]]; then
  command -v nsys >/dev/null || {
    echo "P22B_NSYS_NOT_INSTALLED" >&2; exit 2;
  }
  nsys --version
  HELP="$(nsys profile --help 2>&1 || true)"
  for option in --trace --capture-range --capture-range-end --sample --cpuctxsw --output; do
    if [[ "$HELP" != *"$option"* ]]; then
      echo "P22B_UNSUPPORTED_NSYS_CLI_OPTION=$option" >&2; exit 2
    fi
  done
fi
# Do not inherit old experimental instrumentation or speculative arm flags.
while IFS='=' read -r key _; do
  if [[ "$key" == SGLANG_OMNI_QTTS_CP_* || "$key" == SGLANG_OMNI_STRICT_PORT ]]; then
    unset "$key"
  fi
done < <(env)
export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import importlib
from pathlib import Path
import torch
root = Path.cwd().resolve()
if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
    raise RuntimeError("P22B_REQUIRES_NVIDIA_H200")
for name in (
    "sglang_omni.models.qwen3_tts.sglang_model",
    "sglang_omni.models.qwen3_tts.model_runner",
    "sglang_omni.models.qwen3_tts.code_predictor_spec_p22b_native",
):
    p = Path(importlib.import_module(name).__file__).resolve()
    if not p.is_relative_to(root / "sglang_omni"):
        raise RuntimeError(f"P22B_SOURCE_IMPORT_MISMATCH {name} {p}")
    print(f"P22B_IMPORT_PASS={name}={p}", flush=True)
print("P22B_GPU=" + torch.cuda.get_device_name(0), flush=True)
PY
mkdir -m 700 -p "$ROOT"
SHA="$(git rev-parse HEAD)"
export SGLANG_OMNI_STRICT_PORT=1
export SGLANG_OMNI_QTTS_CP_SPEC_K=0
# P21=1 only enables original request-level activity COUNTERS. No P21 ARM
# is ever created, hence original native CodePredictor Graph always executes.
export SGLANG_OMNI_QTTS_CP_P21_SERVING_K2=1
export SGLANG_OMNI_QTTS_CP_P21_ARM_FILE="$ROOT/P21_ARM_STAYS_ABSENT"
export SGLANG_OMNI_QTTS_CP_P21_SOURCE_SHA="$SHA"
export SGLANG_OMNI_QTTS_CP_P22_PROFILE=0
export SGLANG_OMNI_QTTS_CP_P22B_NATIVE_PROFILE=1
export SGLANG_OMNI_QTTS_CP_P22B_NATIVE_ARM_FILE="$ROOT/P22B_NATIVE_ARM"
export SGLANG_OMNI_QTTS_CP_P22B_NATIVE_PROFILE_DIR="$ROOT/native_profiles"
export SGLANG_OMNI_QTTS_CP_P22B_NATIVE_SOURCE_SHA="$SHA"
printf '%s\n' "$SHA" > "$ROOT/source.sha"
python - "$ROOT" "$SHA" "$NSYS" <<'PY'
import json,sys
from pathlib import Path
root,sha,nsys=Path(sys.argv[1]),sys.argv[2],bool(int(sys.argv[3]))
(root/"run.json").write_text(json.dumps({
    "schema":"qwen3tts-p22b-native-a-profile-v1",
    "source_sha":sha,
    "model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "gpu":"NVIDIA H200",
    "nsys_capture_range":nsys,
    "arm":"P22B_NATIVE_ARM_ONLY_AFTER_HEALTH_AND_A_WARMUP",
    "native_code_predictor_graph":"REQUIRED_ALL_SAMPLE_FRAMES",
    "k2_speculative":"DISABLED",
    "P18_P19":"DISABLED",
    "profiler":"CUDA_API_NVTX_NOT_E2E_PERF",
},indent=2)+"\n")
PY
echo "P22B_NATIVE_SHA=$SHA"
echo "P22B_NATIVE_ROOT=$ROOT"
echo "P22B_NATIVE_NSYS=$NSYS"
if [[ "$NSYS" == 1 ]]; then
  # nsight-systems traces the child TTS process by default on Linux.
  nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --output="$ROOT/p22b_native_a" \
    sgl-omni serve \
      --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
      --port 8000 2>&1 | tee "$ROOT/server.log"
else
  sgl-omni serve \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
    --port 8000 2>&1 | tee "$ROOT/server.log"
fi

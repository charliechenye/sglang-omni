#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${1:?usage: $0 MODEL_PATH OUTPUT_DIR [PORT]}"
OUTPUT_ROOT="${2:?usage: $0 MODEL_PATH OUTPUT_DIR [PORT]}"
PORT="${3:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"
REQUEST_COUNT="${REQUEST_COUNT:-32}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
API_KEY="${API_KEY:-EMPTY}"
INPUT_TEXT="${INPUT_TEXT:-Please speak a short sentence for this benchmark.}"
VOICE="${VOICE:-Ryan}"
LANGUAGE="${LANGUAGE:-English}"
SERVER_PID=""

mkdir -p "$OUTPUT_ROOT"
if [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]; then
  echo "REFUSE_NONEMPTY_OUTPUT_ROOT $OUTPUT_ROOT" >&2
  exit 1
fi
if ss -ltn | grep -qE ":${PORT}[[:space:]]"; then
  echo "PORT_IN_USE $PORT" >&2
  exit 1
fi

stop_server() {
  if [ -z "$SERVER_PID" ]; then
    return
  fi
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 120); do
    if ! pgrep -g "$SERVER_PID" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if pgrep -g "$SERVER_PID" >/dev/null 2>&1; then
    echo "SERVER_PROCESS_GROUP_DID_NOT_EXIT $SERVER_PID" >&2
    return 1
  fi
  for _ in $(seq 1 30); do
    if ! ss -ltn | grep -qE ":${PORT}[[:space:]]"; then
      SERVER_PID=""
      return
    fi
    sleep 1
  done
  echo "PORT_NOT_RELEASED_AFTER_SERVER_EXIT $PORT" >&2
  return 1
}
trap stop_server EXIT INT TERM

start_server() {
  local width="$1"
  local run_dir="$2"
  local config
  case "$width" in
    0) config="$REPO_ROOT/examples/configs/qwen3_tts_1_7b_customvoice.yaml" ;;
    2) config="$REPO_ROOT/examples/configs/qwen3_tts_1_7b_customvoice_speculative_w2.yaml" ;;
    3) config="$REPO_ROOT/examples/configs/qwen3_tts_1_7b_customvoice_speculative_w3.yaml" ;;
    4) config="$REPO_ROOT/examples/configs/qwen3_tts_1_7b_customvoice_speculative.yaml" ;;
    *) echo "UNSUPPORTED_WIDTH $width" >&2; return 2 ;;
  esac
  if ss -ltn | grep -qE ":${PORT}[[:space:]]"; then
    echo "PORT_IN_USE_BEFORE_ARM $PORT" >&2
    return 1
  fi
  setsid sgl-omni serve \
    --model-path "$MODEL_PATH" \
    --config "$config" \
    --port "$PORT" >"$run_dir/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 600); do
    if curl --fail --silent "$BASE_URL/v1/models" >/dev/null; then
      return
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      cat "$run_dir/server.log" >&2
      echo "SERVER_EXITED_DURING_STARTUP width=$width" >&2
      return 1
    fi
    sleep 1
  done
  echo "SERVER_READY_TIMEOUT width=$width" >&2
  return 1
}

run_arm() {
  local width="$1"
  local concurrency="$2"
  local arm_index="$3"
  local run_dir="$OUTPUT_ROOT/c${concurrency}/arm${arm_index}-w${width}"
  local baseline_file="$OUTPUT_ROOT/c${concurrency}/native-baseline.json"
  mkdir -p "$run_dir"
  start_server "$width" "$run_dir"
  local common_args=(
    --base-url "$BASE_URL"
    --baseline-file "$baseline_file"
    --model "$MODEL_PATH"
    --input "$INPUT_TEXT"
    --voice "$VOICE"
    --language "$LANGUAGE"
    --api-key "$API_KEY"
    --request-count "$REQUEST_COUNT"
    --warmup-requests "$WARMUP_REQUESTS"
    --concurrency "$concurrency"
  )
  if [ "$width" -eq 0 ] && [ "$arm_index" -eq 0 ]; then
    python "$REPO_ROOT/scripts/benchmark_qwen3_tts_speculative_serving.py" \
      --phase native "${common_args[@]}"
  elif [ "$width" -eq 0 ]; then
    local repeat_args=("${common_args[@]}")
    repeat_args[3]="$run_dir/native-repeat.json"
    python "$REPO_ROOT/scripts/benchmark_qwen3_tts_speculative_serving.py" \
      --phase native "${repeat_args[@]}"
  else
    python "$REPO_ROOT/scripts/benchmark_qwen3_tts_speculative_serving.py" \
      --phase speculative \
      --expected-width "$width" \
      --minimum-speedup 0.0 \
      --result-file "$run_dir/result.json" \
      "${common_args[@]}"
  fi
  stop_server
  if grep -Eiq 'CUDA error|CUDA_ERROR|illegal memory access|device-side assert|error 700' "$run_dir/server.log"; then
    echo "CUDA_ERROR_IN_SERVER_LOG $run_dir/server.log" >&2
    return 1
  else
    :
  fi
}

for concurrency in 1 8 16; do
  run_arm 0 "$concurrency" 0
  arm_index=1
  for width in 2 3 4 4 3 2; do
    run_arm "$width" "$concurrency" "$arm_index"
    arm_index=$((arm_index + 1))
  done
  run_arm 0 "$concurrency" 7
done

echo "QWEN3_TTS_SPECULATIVE_SWEEP_COMPLETE $OUTPUT_ROOT"

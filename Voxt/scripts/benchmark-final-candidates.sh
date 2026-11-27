#!/usr/bin/env bash
# Interleaved real-model OFF/SHADOW sessions. Requires local model folders;
# the test never downloads weights.
set -euo pipefail

[[ "$(uname -s)" == "Darwin" ]] || { echo "run this benchmark on a Mac" >&2; exit 1; }
: "${VOXTNEW_FINAL_CANDIDATE_BENCH_ASR_MODEL_DIR:?set the local Qwen3-ASR model folder}"
: "${VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_ENDPOINT:?set an OpenAI-compatible LLM API root}"
: "${VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_MODEL:?set the cleanup model id}"
[[ -d "$VOXTNEW_FINAL_CANDIDATE_BENCH_ASR_MODEL_DIR" ]] || { echo "ASR model folder does not exist" >&2; exit 1; }
if [[ -n "${VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS:-}" ]]; then
  [[ -f "$VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS" ]] || { echo "scenario manifest does not exist" >&2; exit 1; }
elif [[ -n "${VOXTNEW_FINAL_CANDIDATE_BENCH_AUDIO:-}" ]]; then
  [[ -f "$VOXTNEW_FINAL_CANDIDATE_BENCH_AUDIO" ]] || { echo "audio fixture does not exist" >&2; exit 1; }
else
  echo "set VOXTNEW_FINAL_CANDIDATE_BENCH_AUDIO or VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS" >&2
  exit 1
fi
runtime="${VOXTNEW_OMNI_RUNTIME:?set the existing local Omni runtime}"
if [[ -d "$runtime" ]]; then runtime="$runtime/bin/qwen3_asr_server"; fi
[[ -x "$runtime" ]] || { echo "Omni runtime server binary does not exist" >&2; exit 1; }
if [[ -n "${VOXTNEW_FINAL_CANDIDATE_BENCH_DRAFT_DIR:-}" ]]; then
  [[ -d "$VOXTNEW_FINAL_CANDIDATE_BENCH_DRAFT_DIR" ]] || { echo "draft model folder does not exist" >&2; exit 1; }
fi

cd "$(dirname "$0")/.."
mkdir -p build
output="${VOXTNEW_FINAL_CANDIDATE_BENCH_OUTPUT:-build/final-candidate-benchmark.log}"
jsonl="${output%.log}.jsonl"
mkdir -p "$(dirname "$output")"
export VOXTNEW_FINAL_CANDIDATE_BENCH=1
export VOXTNEW_FINAL_CANDIDATE_BENCH_ROUNDS="${VOXTNEW_FINAL_CANDIDATE_BENCH_ROUNDS:-5}"
export VOXTNEW_FINAL_CANDIDATE_BENCH_WARMUPS="${VOXTNEW_FINAL_CANDIDATE_BENCH_WARMUPS:-1}"
go test -count=1 -run '^TestMacE2EFinalCandidateBenchmark$' -v -timeout 2h . 2>&1 | tee "$output"
sed -n 's/^FINAL_CANDIDATE_BENCH //p' "$output" > "$jsonl"
printf 'Machine-readable results: %s\n' "$jsonl"

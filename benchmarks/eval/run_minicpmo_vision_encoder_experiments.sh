#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ENV_FILE="${ENV_FILE:-}"
if [[ -n "$ENV_FILE" ]]; then
    [[ -f "$ENV_FILE" ]] || { printf 'Environment file missing: %s\n' "$ENV_FILE" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

MODEL_ID="${MODEL_ID:-${SGLANG_MINICPMO_MODEL_ID:-openbmb/MiniCPM-o-4_5}}"
VIDEOMME_REPO="${VIDEOMME_REPO:-${SGLANG_VIDEOMME_CI_REPO_ID:-zhaochenyang20/Video_MME_ci}}"
RESULT_ROOT="${RESULT_ROOT:-${SGLANG_RESULTS_DIR:-${TMPDIR:-/tmp}}/minicpmo-vision}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
PYTHON="${PYTHON:-python}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
RESAMPLER_WARMUP="${RESAMPLER_WARMUP:-10}"
RESAMPLER_ITERS="${RESAMPLER_ITERS:-50}"
PROFILE_ITERS="${PROFILE_ITERS:-2}"
RESAMPLER_SAMPLE_SELECTION="${RESAMPLER_SAMPLE_SELECTION:-median}"
PROFILE_SAMPLE_SELECTION="${PROFILE_SAMPLE_SELECTION:-high}"
COHORT_FILE="${COHORT_FILE:-}"
PREFERRED_COHORT="${SGLANG_RESULTS_DIR:-$HOME/results}/minicpmo-video-frozen-cohort.json"

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_BASE="a266a0894d8964ca592a86cb198b446acef8908a"
EXPECTED_BRANCH="exp/minicpmo-vision-transformer-perf"
cd "$REPOSITORY"
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
    printf 'Checkout %s before running this driver.\n' "$EXPECTED_BRANCH" >&2; exit 2;
}
git merge-base --is-ancestor "$EXPECTED_BASE" HEAD || {
    printf 'Expected baseline %s is not an ancestor of HEAD.\n' "$EXPECTED_BASE" >&2; exit 2;
}
git diff --quiet "$EXPECTED_BASE" -- sglang_omni || {
    printf 'Production code differs from the validated baseline; restore it before measuring.\n' >&2; exit 2;
}
export PYTHONPATH="$REPOSITORY:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER=torchcodec

FROZEN_IDS=(002-1 003-1 005-1 006-1 007-1 008-1 009-1 010-1
            011-1 012-1 014-1 015-1 017-1 018-1 019-1 020-1)
if [[ -n "$COHORT_FILE" ]]; then
    [[ -f "$COHORT_FILE" ]] || { printf 'Cohort file missing: %s\n' "$COHORT_FILE" >&2; exit 2; }
elif [[ -f "$PREFERRED_COHORT" ]]; then
    COHORT_FILE="$PREFERRED_COHORT"
fi
if [[ -n "$COHORT_FILE" ]]; then
    cohort_arguments=(--cohort-file "$COHORT_FILE")
else
    cohort_arguments=(--sample-ids "${FROZEN_IDS[@]}")
fi

result_ancestor="$RESULT_ROOT"
while [[ ! -d "$result_ancestor" ]]; do
    result_ancestor="$(dirname "$result_ancestor")"
done
if [[ "$(git -C "$result_ancestor" rev-parse --is-inside-work-tree 2>/dev/null || true)" == true ]]; then
    printf 'RESULT_ROOT must be outside Git worktrees: %s\n' "$RESULT_ROOT" >&2; exit 2;
fi
mkdir -p "$RESULT_ROOT"
RUN_DIR="$(mktemp -d "$RESULT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
common_arguments=(--model-id "$MODEL_ID" --videomme-repo "$VIDEOMME_REPO"
                  --device "$DEVICE" --dtype "$DTYPE" --warmup "$WARMUP" --iters "$ITERS"
                  --resampler-warmup "$RESAMPLER_WARMUP" --resampler-iters "$RESAMPLER_ITERS"
                  "${cohort_arguments[@]}")
printf 'base=%s branch=%s HEAD=%s\nresults=%s\n' "$EXPECTED_BASE" "$EXPECTED_BRANCH" "$(git rev-parse HEAD)" "$RUN_DIR"
printf 'Frozen cohort: %s\n' "${COHORT_FILE:-the explicit built-in 16 IDs}"

run_mode() {
    local label="$1" mode="$2"
    shift 2
    local summary="$RUN_DIR/${label#*-}.json"
    printf '\nRunning %s\n' "$label"
    "$PYTHON" -u -m benchmarks.eval.minicpmo_vision_encoder_microbench \
        "${common_arguments[@]}" --mode "$mode" --output-json "$summary" "$@" \
        2>&1 | tee "$RUN_DIR/$label.log"
}

run_mode 01-census census
run_mode 02-decompose-median-bs16 decompose --census-file "$RUN_DIR/census.json" --sample-selection median --vision-batch-size 16
run_mode 03-decompose-median-bs32 decompose --census-file "$RUN_DIR/census.json" --sample-selection median --vision-batch-size 32
run_mode 04-decompose-median-bs64 decompose --census-file "$RUN_DIR/census.json" --sample-selection median --vision-batch-size 64
run_mode 05-decompose-high-bs16 decompose --census-file "$RUN_DIR/census.json" --sample-selection high --vision-batch-size 16
run_mode 06-decompose-high-bs32 decompose --census-file "$RUN_DIR/census.json" --sample-selection high --vision-batch-size 32
run_mode 07-decompose-high-bs64 decompose --census-file "$RUN_DIR/census.json" --sample-selection high --vision-batch-size 64
run_mode 08-vision-batch-median vision-batch --census-file "$RUN_DIR/census.json" --sample-selection median --vision-batch-sizes 16 32 64
run_mode 09-vision-batch-high vision-batch --census-file "$RUN_DIR/census.json" --sample-selection high --vision-batch-sizes 16 32 64
run_mode 10-resampler resampler --census-file "$RUN_DIR/census.json" --sample-selection "$RESAMPLER_SAMPLE_SELECTION"
run_mode 11-profile profile --census-file "$RUN_DIR/census.json" --sample-selection "$PROFILE_SAMPLE_SELECTION" --profile-iters "$PROFILE_ITERS" --profile-dir "$RUN_DIR/profile"
printf '\nDone. Paste the JSON summaries and logs from %s\n' "$RUN_DIR"

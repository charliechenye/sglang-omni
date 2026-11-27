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
repo_root="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
python3 - "$runtime" "$repo_root" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

binary = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
provenance_path = binary.parent.parent / "build-provenance.json"
if not provenance_path.is_file():
    raise SystemExit(f"missing runtime provenance: {provenance_path}")
provenance = json.loads(provenance_path.read_text())
commit = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
source_path = pathlib.Path(provenance["native_source_path"]).resolve()
expected_source = (root / "sglang_omni_mlx" / "native").resolve()
binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
checks = {
    "experiment_commit": provenance.get("experiment_commit") == commit,
    "native_source_commit": provenance.get("native_source_commit") == commit,
    "clean_native_worktree": provenance.get("native_worktree_clean") is True,
    "native_source_path": source_path == expected_source,
    "runtime_binary_path": pathlib.Path(provenance.get("runtime_binary_path", "")).resolve() == binary,
    "runtime_binary_sha256": provenance.get("runtime_binary_sha256") == binary_hash,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("runtime provenance check failed: " + ", ".join(failed))
print(f"Verified native runtime provenance: {provenance_path}")
PY
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
set +e
go test -count=1 -run '^TestMacE2EFinalCandidateBenchmark$' -v -timeout 2h . 2>&1 | tee "$output"
test_status=${PIPESTATUS[0]}
set -e
sed -n 's/^FINAL_CANDIDATE_BENCH //p' "$output" > "$jsonl"
printf 'Machine-readable results: %s\n' "$jsonl"
exit "$test_status"

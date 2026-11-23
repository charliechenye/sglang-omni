#!/usr/bin/env bash
# One fresh H200 worker + ONE request + process-group teardown per P19 leg.
# Never launches the 24-request P19 Main or performs a DP selection.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash p19_lifecycle_leg.sh BASE_EAGER|BASE_GRAPH|MIN_REPLAY|MATCHED_EAGER|GC_GUARD|RETAIN_GRAPH ABSENT_ABSOLUTE_ROOT" >&2
  exit 2
fi
LEG="$1"
ROOT="$(realpath -m "$2")"
if [[ "$2" != /* || "$ROOT" != "$2" ]]; then
  echo "P19_LIFECYCLE_REQUIRES_CANONICAL_ABSOLUTE_RESULT_ROOT" >&2
  exit 2
fi
if [[ -e "$ROOT" || -e "${ROOT}.launcher.log" ]]; then
  echo "P19_LIFECYCLE_REFUSES_OVERWRITE: $ROOT" >&2
  exit 2
fi
case "$LEG" in
  BASE_EAGER) MODE="ALL_DEPTHS_EAGER"; PROFILE="DEFAULT" ;;
  BASE_GRAPH) MODE="ALL_DEPTHS_GRAPH"; PROFILE="DEFAULT" ;;
  MATCHED_EAGER) MODE="ALL_DEPTHS_EAGER"; PROFILE="MATCHED_EAGER" ;;
  MIN_REPLAY) MODE="ALL_DEPTHS_GRAPH"; PROFILE="MIN_REPLAY" ;;
  GC_GUARD) MODE="ALL_DEPTHS_GRAPH"; PROFILE="GC_GUARD" ;;
  RETAIN_GRAPH) MODE="ALL_DEPTHS_GRAPH"; PROFILE="RETAIN_GRAPH" ;;
  *) echo "P19_LIFECYCLE_UNKNOWN_LEG: $LEG" >&2; exit 2 ;;
esac
if [[ "$(git branch --show-current)" != "experiment/qwen3-tts-cpspec-p19-matched-dp" ]] ||
   [[ "$(pwd -P)" != "$(git rev-parse --show-toplevel)" ]]; then
  echo "P19_LIFECYCLE_REQUIRES_PINNED_WORKTREE_ROOT_AND_BRANCH" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "P19_LIFECYCLE_REQUIRES_CLEAN_TRACKED_WORKTREE" >&2; exit 2
fi
python - <<'PY'
import socket
with socket.socket() as sock:
    sock.settimeout(2)
    if sock.connect_ex(("127.0.0.1", 8000)) == 0:
        raise SystemExit("P19_LIFECYCLE_PORT_8000_ALREADY_OCCUPIED")
PY
command -v setsid >/dev/null || { echo "P19 needs setsid" >&2; exit 2; }
command -v curl >/dev/null || { echo "P19 needs curl" >&2; exit 2; }

echo "P19_LIFECYCLE_START leg=$LEG mode=$MODE profile=$PROFILE source_sha=$(git rev-parse HEAD)"
echo "P19_LIFECYCLE_ROOT=$ROOT"
# Only one target. Changes in fence or frame count invalidate pairing and
# are forbidden by p19_launch.sh for non-default profiles.
SERVER_PGID=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PGID" ]]; then
    echo "P19_LIFECYCLE_TERMINATE_WORKER_GROUP=$SERVER_PGID"
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    wait "$SERVER_PGID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

P19_DIAG_ARM="$MODE" \
P19_DIAG_DEPTH=14 \
P19_DIAG_PROFILE="$PROFILE" \
P19_DIAG_FENCE=0 \
P19_DIAG_NATIVE_FENCE=0 \
P19_DIAG_TAIL_FENCE=1 \
P18_CAPTURE_FRAMES=2 \
P19_COST_FRAMES=1 \
  setsid bash experiments/qwen3_tts_cp_spec_k2/p19_launch.sh "$ROOT" \
    > "${ROOT}.launcher.log" 2>&1 &
SERVER_PGID=$!
# A bounded startup probe: do not issue HTTP requests before /health.
healthy=0
for ((i=0; i<180; i++)); do
  if ! kill -0 "$SERVER_PGID" 2>/dev/null; then
    echo "P19_LIFECYCLE_SERVER_EXITED_EARLY: ${ROOT}.launcher.log" >&2
    tail -n 75 "${ROOT}.launcher.log" >&2 || true
    exit 1
  fi
  if [[ -f "$ROOT/source.sha" ]] &&
     curl --connect-timeout 2 --max-time 5 -fsS \
       http://127.0.0.1:8000/health >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if (( i % 15 == 0 )); then
    echo "P19_LIFECYCLE_STARTUP_CHECK=$i"
  fi
  sleep 2
done
if [[ "$healthy" != 1 ]]; then
  echo "P19_LIFECYCLE_HEALTH_NOT_READY: $ROOT" >&2
  exit 1
fi
if [[ "$(cat "$ROOT/source.sha")" != "$(git rev-parse HEAD)" ]]; then
  echo "P19_LIFECYCLE_WORKER_SOURCE_SHA_MISMATCH" >&2; exit 1
fi
echo "P19_LIFECYCLE_HEALTH_PASS"
set +e
python -m experiments.qwen3_tts_cp_spec_k2.p19_diag_once --root "$ROOT" \
  2>&1 | tee "$ROOT/client.console.log"
CLIENT_EXIT=${PIPESTATUS[0]}
set -e
echo "P19_LIFECYCLE_CLIENT_EXIT=$CLIENT_EXIT"
if [[ -f "$ROOT/matched_cost/diagnostic/client.json" ]]; then
  python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(
    (Path(sys.argv[1]) / "matched_cost/diagnostic/client.json").read_text()
)
print("P19_LIFECYCLE_RESULT=" + r.get("status", "UNKNOWN"))
print("P19_LIFECYCLE_PROFILE=" + r.get("diagnostic", {}).get("profile", "UNKNOWN"))
print("P19_LIFECYCLE_COVERAGE=" + str(r.get("trace_coverage", {}).get("graph_arms_verified", 0) + r.get("trace_coverage", {}).get("eager_arms_verified", 0)))
PY
fi
if [[ "$CLIENT_EXIT" -ne 0 ]]; then
  if [[ -f "$ROOT/server.log" ]]; then
    grep -nEm 5 'CUDA_ERROR_ILLEGAL_ADDRESS|illegal memory access|Sticky error' \
      "$ROOT/server.log" || true
  fi
  exit "$CLIENT_EXIT"
fi
echo "P19_LIFECYCLE_ONE_LEG_COMPLETE_NO_COST_OR_SERVING_SELECTION"

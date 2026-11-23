# P19 CUDA700 — matched-workload and Graph lifecycle causal ablations

**Status:** experimental-only, NOT production safety, performance, full serving
Graph-ON parity, quality improvement, P19A matched cost, or P19B DP selection.

## Why these legs

H200 one real native frame at frame 37 passed 14 depths/50 individual AR,
K2, K3 and K4 Graph replay/eager parity checks. The subsequent next-frame
main Talker forward raised CUDA 700 at its FIRST full-device synchronized
return. An isolated 50-arm EAGER control passed, but that control ran each
extra arm only once. The Graph test ran substantially more work.

This ablation changes exactly one independent factor per experiment
and preserves the same prompt/seed, source SHA, native P18 acceptance oracle,
KV and RNG restore, first-frame selection, 14 depths, 50 legal arms, next-frame
trace, HTTP natural-stop requirement and H200 single-request setup.

Original P19 per-arm workload:
- Eager reference: **1** operation
- Eager warmups: **2**
- Graph capture: **1** operation-equivalent (not identical kernel schedule)
- First parity replay: **1**
- 4 timing blocks × (3 warm replay + 12 timed replay) = **60**
- Total: 3 eager + 1 capture + 61 replay = **65 operations-equivalent per arm**
- 50 arms: **150 eager + 50 captures + 3,050 replays = 3,250 equivalents**

New profile variants (all strictly diagnostic-only):

| Leg | Profile | Mode | Per-arm operations | 50-arm calls | Isolated difference |
|---|---|---|---:|---:|---|
| BASE_EAGER | DEFAULT | ALL_DEPTHS_EAGER | 1 eager | 50 | Existing control |
| BASE_GRAPH | DEFAULT | ALL_DEPTHS_GRAPH | 3 eager + 1 capture + 61 replay | 3,250 | Original failing Graph path |
| MIN_REPLAY | MIN_REPLAY | ALL_DEPTHS_GRAPH | 3 eager + 1 capture + 1 replay | 250 | No timing replay blocks |
| MATCHED_EAGER | MATCHED_EAGER | ALL_DEPTHS_EAGER | 65 eager | 3,250 | Same operation count & approximate restore cadence, **not** identical kernel scheduling or allocator |
| GC_GUARD | GC_GUARD | ALL_DEPTHS_GRAPH | 65 | 3,250 | Disable Python GC **during capture**, restore prior GC state |
| RETAIN_GRAPH | RETAIN_GRAPH | ALL_DEPTHS_GRAPH | 65 | 3,250 | Keep all 50 captured Graph objects, streams and output tensors live **until disposable worker exits** |

IMPORTANT: Do not equate graph replay work with eager kernels for an exact
GPU-level A/B. Matching operation-equivalents is a narrower control, not
proof that execution overhead, memory pools or the CUDA scheduler matched.

## New scripts

- `code_predictor_spec_p19_profile.py`: pure validation and exact expected
  workload accounting; prevents profiles from leaking into formal P19A.
- `code_predictor_spec_p19_cost.py`: profile-guarded Graph one-replay,
  GC-disabled capture, temporary Graph keepalive, or 65-eager-control execution;
  DEFAULT code path stays as before.
- `p19_launch.sh`: fail-closed allowed profile/mode/fence/depth checks; hashes
  the source and writes the diagnostic profile into `matched_cost/config.json`.
- `p19_diag_coverage.py`: phase-by-phase, arm-by-arm profile-aware evidence
  validation including retained graph count and actual native frame identity.
- `p19_diag_once.py`: distinguishes HTTP success, missing coverage, client
  failure and CUDA700 evidence; records expected workload counts as **expected**.
- `p19_lifecycle_leg.sh`: executes **ONE leg** in a new `setsid` process
  group, checks source and port 8000, waits for healthy server, sends exactly
  one HTTP request, saves logs and terminates the worker process group.
- `p19_lifecycle_compare.py`: read-only strict comparison; rejects
  cross-source-SHA, mismatched prompt/seed, missing 50-arm coverage,
  invalid launch controls and duplicate profiles.

## Prerequisite: update worktree and test

```bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
git fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-cpspec-p19-matched-dp
git merge --ff-only FETCH_HEAD
git rev-parse --short HEAD

python -m pytest -q -x tests/unit_test/qwen3_tts/test_cp_spec_p19.py \
  -m "not accelerator"
bash -n experiments/qwen3_tts_cp_spec_k2/p19_launch.sh
bash -n experiments/qwen3_tts_cp_spec_k2/p19_lifecycle_leg.sh
```

No H200 claim is allowed until tests pass. Use a **new** root per leg.
Port 8000 must be unused. The helper refuses to kill an already running
server; it only tears down the **new session it started**. Do not share the
GPU with unrelated workloads.

## Recommended experiment order — one process per leg

```bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate

# Re-establish original failure on the NEW identical source SHA.
bash experiments/qwen3_tts_cp_spec_k2/p19_lifecycle_leg.sh \
  BASE_GRAPH "$HOME/results/qwen3tts-p19-p20-base-graph-001"

# Same 50 Graph captures, one parity replay each; no timing replay.
bash experiments/qwen3_tts_cp_spec_k2/p19_lifecycle_leg.sh \
  MIN_REPLAY "$HOME/results/qwen3tts-p19-p20-min-replay-001"

# Same operation-equivalent count as original Graph arm, all eager.
bash experiments/qwen3_tts_cp_spec_k2/p19_lifecycle_leg.sh \
  MATCHED_EAGER "$HOME/results/qwen3tts-p19-p20-matched-eager-001"

# Optional controls, depending on the first three results:
bash experiments/qwen3_tts_cp_spec_k2/p19_lifecycle_leg.sh \
  GC_GUARD "$HOME/results/qwen3tts-p19-p20-gc-guard-001"

# WARNING: retained capture graphs intentionally consume device memory
# through this ONE-use process lifetime. Never use in production.
bash experiments/qwen3_tts_cp_spec_k2/p19_lifecycle_leg.sh \
  RETAIN_GRAPH "$HOME/results/qwen3tts-p19-p20-retain-001"
```

Every command **exits nonzero** on failed HTTP/coverage. That is expected
for a reproducible CUDA700 and must NOT cause a runner to reuse the same
server or result root. Run each next command independently.
`P19_DIAG_TAIL_FENCE=1` is held constant across legs to pinpoint the
first failing main Talker forward; `P19_DIAG_NATIVE_FENCE=0`,
`P19_DIAG_FENCE=0`, P18 frames=2, P19 frames=1.

You can still launch the standard two-terminal flow with
`P19_DIAG_ARM=ALL_DEPTHS_GRAPH|ALL_DEPTHS_EAGER`,
`P19_DIAG_PROFILE=MIN_REPLAY|MATCHED_EAGER|GC_GUARD|RETAIN_GRAPH` and
the same fence values via `p19_launch.sh` + `p19_diag_once.py`.
The one-leg helper merely automates isolation.

## Compare only same-SHA roots

```bash
python -m experiments.qwen3_tts_cp_spec_k2.p19_lifecycle_compare \
  "$HOME/results/qwen3tts-p19-p20-base-graph-001" \
  "$HOME/results/qwen3tts-p19-p20-min-replay-001" \
  "$HOME/results/qwen3tts-p19-p20-matched-eager-001" \
  --out "$HOME/results/qwen3tts-p19-p20-lifecycle-evidence-001"
```

If one root has a different source SHA or request prompt/seed,
comparison MUST fail closed: P18's deterministic uniforms are derived
from source SHA, so cross-SHA comparisons are especially misleading.
The comparator prints expected eager/capture/replay calls, 50-arm
coverage, the latest native tail boundary, HTTP status and independent
server-log CUDA700 detection. It never reports QPS or speedup.

## Interpretation / next action

- BASE_GRAPH fails, MIN_REPLAY passes: suspect sustained Graph replay
  pressure, state accumulation, allocation timing; NOT proof.
- BASE_GRAPH fails, MIN_REPLAY fails: capture/first-replay or
  graph-destruction/lifetime already sufficient. Next examine GC and retention.
- MATCHED_EAGER fails: cumulative workload may be enough even without
  Graph, so do NOT attribute the bug exclusively to graph pools.
- MATCHED_EAGER passes but BASE_GRAPH fails: Graph-specific capture,
  replay, lifetime or graph-only kernel behavior is a stronger explanation.
- GC_GUARD consistently passes where BASE_GRAPH fails: capture-time
  automatic GC becomes a leading candidate. Repeat fresh process,
  do not conflate a single pass with proof.
- RETAIN_GRAPH passes where BASE_GRAPH fails: premature graph object/
  captured-tensor release becomes a leading candidate. A retained-Graph
  memory-footprint/allocator change is itself a confound.
- All profiles fail: preserve first GPU error and use a targeted
  CUDA memory sanitizer. The last synchronized stage identifies an
  interval, not necessarily the writer of the invalid address.

A single leg is a one-request screen, never a production race-freedom
claim. Any current CUDA700 is a **P19A/P19B NO GO**. Do not merge this
diagnostic framework into serving or use it for cost policy selection.

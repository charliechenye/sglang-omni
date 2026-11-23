# P19A/B — Qwen3-TTS matched native AR costs and all-depth Bellman DP (H200)

**Research-only. Not a serving optimization, full native Predictor Graph
benchmark, causal speculative rollout, audio equivalence or production GO.**

## Why P19 exists

Source P17 H200: `de4075501633a4d5054d820ea2eb13039fb5d8dc`.
P17 verified native-head distributions / physical KV of one K2/3/4
wide verifier, but costs were measured at only depths 1 and 9 and were
not a matched single-native-step comparison. P18A main
`86e60743b21db575631d6e9e98d67a413cb997ea` sampled 8 prompts
x 3 seeds x 8 frames, 2688 native-AR-eager depth checkpoints with
independent P17 oracle parity. P18B used EXTRAPOLATED P17 costs and
the local heuristic `round cost / expected committed codes`, which
cannot account for jumping over entire future depth states.

P19 fixes those two gaps, without claiming a full 14-depth native
speculative CUDA Graph or a deployment speedup.

### P19A — same real checkpoint, same GPU worker, full depth graph components

This P19 branch derives from P18's validated native AR source. For
each actual `code_predictor_forward_incremental` checkpoint AFTER
native code[d] was sampled and BEFORE its KV append, P18 still
validates independent q/p sampling distribution, residual correction,
static wide verifier and every physical KV slot.

Then, ONLY for the FIRST selected frame of each request by default,
P19 captures separately from exactly that checkpoint:

- **AR:** one native
  `predictor_forward_tokens(last_embedding,cache_len)`,
  `lm_head[depth]`, genuine native seeded
  `sample_subtalker_embedding` with private same-shape sampler
  buffers, actual codec sum, and next projected token (unless final
  code15).
- **K2/K3/K4:** the exact P17
  `make_wide_round`, with real batched heads, one causal fused wide
  verifier, conditional accept / positive residual and bonus, and
  output code, embedding, state.
- Only legal widths run: depths 1..11 K2/K3/K4, depth12 K2/K3,
  depth13 K2, depth14 native AR only.

Each arm is warmed, independently captured in a CUDA Graph and
checked against an eager invocation of THE SAME arm. Each comparison
includes code outputs and full physical K/V. It then collects four
blocks of twelve graph replays with GPU CUDA Events. It restores
physical KV and CPU/CUDA RNG in all cases, including errors.
A failed capture, incorrect code, mismatched KV, missing depth,
wrong source/PID or non-natural HTTP finish fails CLOSED and retains
the first failure file. Graph events include graph launch gaps;
the arms are separate components, not a fused full-decoder graph.

The experiment uses the same code/prompt/seed inputs across all arms
but **native AR uses the genuine seeded SGLang sampler**, whereas
P17 speculative widths use a predeclared paired uniform tuple.
These are equivalent *distributions* (validated independently in
P18), not identical seeded output trajectories. This is intentionally
not called native seeded full-decoder parity.

P19 records `matched_cost/rows/*.json` one file per actual
(request, measured-frame, depth) and matches each row with the
`census/rows/*.json` P18 row of the SAME source/PID/request/frame.
Source SHA is NEW (P19), but both acceptance and matched costs are
from this same P19 run. No P17 cost extrapolation is used.

### P19B — correct residual-depth transition DP

The terminal state is **last committed code=15**, and d=1..14
represents the last committed residual code. The original first
pair (code0/code1) is NOT priced in P19B.

At state d:
- Native AR cost `AR(d)` advances to d+1.
- K-wide round costs `C(d,K)`; its actual accepted-prefix distribution
  commits j=1..K codes and advances to d+j.
- The model solves the Bellman equation backwards:

`V(d) = min_a [C(d,a) + sum_j P(j | native checkpoint,d,a) * V(d+j)]`

The report includes true native-AR-only residual component cost,
fixed K2/K3/K4 with AR tail fallback, nominal adaptive policy, and
full per-depth Q-values. The CI resamples PROMPTS then SEEDS (paired
across ALL widths and costs); frames stay correlated inside a request.
**The bootstrap evaluates the FROZEN nominal policy**, not a newly
re-optimized policy each resample; separate reselected-optimal
statistics are explicitly marked optimistic.

**Important limitations:** native AR checkpoint distributions are
NOT the distributions after causal speculative commit changes future
state. Separate captured components are NOT a whole sampled Predictor
Graph-ON. The initial pair, integrated graph control, changes in
cross-depth KV/sampling, graph memory, startup, quality and HTTP serving
are unmeasured. The historical 2.23234ms/frame complete native
Predictor Graph ON is CONTEXT ONLY and MUST NOT be compared directly
to P19 residual-component costs.

Even a positive P19B "headroom" is a reason to study causal rollout,
not permission to merge production speculation. The default research
follow-up threshold is >15% lower hierarchical fixed-policy headroom,
NOT a serving GO threshold.

## 0. Create isolated worktree (bootstrap already installed)

Use the existing git worktree as control; do not switch or reset P18.

~~~bash
P18="$HOME/sglang-omni-worktrees/p18-depth-census"
P19="$HOME/sglang-omni-worktrees/p19-matched-dp"

git -C "$P18" fetch \
  https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-cpspec-p19-matched-dp

git -C "$P18" worktree add -b \
  experiment/qwen3-tts-cpspec-p19-matched-dp \
  "$P19" FETCH_HEAD

cd "$P19"
source ~/.venv/bin/activate
git status --short
git rev-parse HEAD

python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p19_cost.py \
  experiments/qwen3_tts_cp_spec_k2/p19_analyze.py \
  experiments/qwen3_tts_cp_spec_k2/p19_dp.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p19.py

bash -n experiments/qwen3_tts_cp_spec_k2/p19_launch.sh
bash -n experiments/qwen3_tts_cp_spec_k2/p19_measure.sh

python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p17.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p18.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p19.py \
  -m "not accelerator"
~~~

If the P19 local branch already exists, add worktree without `-b`;
never force-reset. P19 launch prepends this worktree to PYTHONPATH
BEFORE starting the Python multiprocessing spawned worker. No
editable pip reinstallation or bootstrap is necessary.

## 1. Smoke first — 4 requests, 56 matched depth checkpoints

STOP old P18 or P17 servers and any other large GPU experiment. The
Qwen3-TTS server can reserve ~110 GiB of a single H200 in its KV pool.

Terminal A:

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
P18_CAPTURE_FRAMES=2 P19_COST_FRAMES=1 bash \
  experiments/qwen3_tts_cp_spec_k2/p19_launch.sh \
  "$HOME/results/qwen3tts-p19-smoke-001"
~~~

Terminal B, only after `/health` returns HTTP 200:

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p19-smoke-001"
curl -fsS http://127.0.0.1:8000/health

bash experiments/qwen3_tts_cp_spec_k2/p19_measure.sh "$ROOT" smoke

cat "$ROOT/matched_cost/analysis/summary.md"
cat "$ROOT/dp/summary.md"
~~~

The smoke checks exactly 4 requests * 2 frames * 14 depths =
112 independently validated P18 depth rows; costs attach to the
FIRST frame from each request at all 14 depths (56 matched cost rows).
The DP smoke output is a **software-correctness diagnostic only**,
never width policy selection.

Expected statuses:
- P18A `P18A_ALL_DEPTH_CLUSTERED_ACCEPTANCE_CENSUS_PASS_NO_SPEEDUP`
- P19A `P19A_SAME_NATIVE_CHECKPOINT_GRAPH_COMPONENT_COST_PASS_NOT_E2E`
- P19B `P19B_ALL_DEPTH_MATCHED_COMPONENT_DP_SCREEN_NOT_E2E_GO`

If any first error occurs: DO NOT reuse this root, inspect the earliest
`matched_cost/failures/*.txt` or `census/failures/*.txt`, and the
first server traceback. Never turn off parity or add fallback kernels.

## 2. Main — 24 requests, 336 matched depth costs

ONLY after smoke PASS. Stop the smoke server and use a fresh worker,
same source SHA, with 8 native acceptance frames and 1 cost frame per
request. This produces 2688 native acceptance rows plus
24 requests x 14 depths = 336 matched graph-cost rows.

Terminal A:

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
P18_CAPTURE_FRAMES=8 P19_COST_FRAMES=1 bash \
  experiments/qwen3_tts_cp_spec_k2/p19_launch.sh \
  "$HOME/results/qwen3tts-p19-main-001"
~~~

Terminal B:

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p19-main-001"
curl -fsS http://127.0.0.1:8000/health
bash experiments/qwen3_tts_cp_spec_k2/p19_measure.sh "$ROOT" main
cat "$ROOT/matched_cost/analysis/summary.md"
cat "$ROOT/dp/summary.md"
~~~

No historical P17 report is needed; P19's costs and acceptance are
source-matched from one worker. The P19 DP uses only the measured
P19A costs at every eligible depth, not interpolation.

## 3. Stop/go policy

- If capture or correctness fails: preserve the first exception and
  repair; no model substitution, tolerance relaxation or skipped depth.
- If smoke passes and main has weak fixed-policy hierarchical headroom:
  do not build the full adaptive decoder.
- If main suggests >15% hierarchical lower bound on residual-component
  potential: investigate a minimal causal rollout under native
  Graph-ON and collect real audio-quality and HTTP latency/QPS ABBA
  before committing to fused acceptance kernels.
- Even P19 PASS cannot prove production beats the 2.232ms/frame native
  Graph-ON Predictor. The scopes are different and initial tokens
  and integrated control are unpriced.

No PR merge or performance claim is authorized by P19A/B.

## P19 DP AR-only terminal regression

An early P19 CPU test failed in `test_p19_analysis_smoke_is_not_serving_go`
with `P19_DP_UNPAIRED_REAL_NATIVE_CASES`. The cause was NOT an
incomplete source-locked cost row. At last-committed depth 14,
`widths(14) == ()`: only the terminal-adjacent native AR step exists,
so `native_probs == {}` is the **correct** histogram. The original
`_case_tables` check incorrectly required the set of histogram
value lengths to contain exactly one element, which fails for an
intentionally empty dictionary.

The corrected gate still requires a non-empty measured native AR cost
at depth14, a completely empty speculative histogram there, and
nonempty balanced costs and acceptance histograms at all earlier
eligible depths. Regression tests validate the legitimate empty depth14
histogram and deliberately omit the K2 cost at depth13 to require a
fail-closed error. No correctness requirement was relaxed.

Update the P19 worktree using `git fetch` and `git merge --ff-only`,
then rerun the P19 unit tests BEFORE starting the H200 smoke server.
Prior synthetic test failure does not invalidate P18A/P18B H200 data.


## P19 CUDA700 after first real smoke: isolate the first faulty kernel

An initial P19 smoke crashed its H200 pipeline worker with
`CUDA_ERROR_ILLEGAL_ADDRESS (700)` and `exit=-6`. The visible Python
frame was `p18_census.maybe_census(): semantic_position.item()`, but
CUDA errors may be reported asynchronously at later synchronizations.
The rest of the server traceback is shutdown fallout, not a second
root cause. A sticky CUDA700 requires a **NEW worker/process**.
Do not retry in the failed process, reuse its evidence root, or
treat its partial cost files as valid.

PyTorch's CUDA Graph contract requires stable input addresses and
graph-private output/pool lifetime across each replay. The new
P19 native seeded sampler capture and graph/allocator lifecycle are
suspects, not yet proven causes. P18 native-eager without P19 graph
capture previously passed. We therefore isolate phases rather than
changing `.item()`, skipping parity, or claiming the cause.

### 0. Preserve first crash evidence without changing it

~~~bash
ROOT="$HOME/results/qwen3tts-p19-smoke-001"
test -d "$ROOT"
cat "$ROOT/source.sha"
find "$ROOT/matched_cost/rows" -name '*.json' | wc -l
find "$ROOT/census/rows" -name '*.json' | wc -l
find "$ROOT/matched_cost/failures" "$ROOT/census/failures" \
  -name '*.txt' -print 2>/dev/null
grep -nE 'CUDA_ERROR_ILLEGAL_ADDRESS|illegal memory|CUDA error|P19_' \
  "$ROOT/server.log" | tail -n 40
~~~

Counts establish which complete rows existed BEFORE the crash.
They do not prove which kernel first wrote the invalid address.

### 1. Update branch, run CPU tests, then single-request isolation

Always use a **fresh root per mode**. Stop the old server first. Source
lock and the ordinary native-AR P18 independent oracle remain enabled.

Start with **AR_GRAPH at depth 1**; it is P19's first new graph arm
in the first selected frame. If it fails, compare OFF and AR_EAGER,
each on a fresh worker. If AR_GRAPH passes, test K2_GRAPH then
K3_GRAPH and K4_GRAPH, still one depth and fresh worker per mode.

Terminal A, example AR_GRAPH at depth 1:

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate

P19_DIAG_ARM=AR_GRAPH P19_DIAG_DEPTH=1 P19_DIAG_FENCE=0 \
P18_CAPTURE_FRAMES=2 P19_COST_FRAMES=1 bash \
  experiments/qwen3_tts_cp_spec_k2/p19_launch.sh \
  "$HOME/results/qwen3tts-p19-diag-argraph-d01-001"
~~~

Terminal B, after /health (same root as Terminal A):

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p19-diag-argraph-d01-001"
curl -fsS http://127.0.0.1:8000/health
python -m experiments.qwen3_tts_cp_spec_k2.p19_diag_once --root "$ROOT"
cat "$ROOT/matched_cost/diagnostic/client.json"
~~~

If `python` exits nonzero, preserve the trace; the diagnostic is
expected to stop at the FIRST error, never fall back or run P19B.
Read the durable CPU-only phase events:

~~~bash
ROOT="$HOME/results/qwen3tts-p19-diag-argraph-d01-001"
grep -hE 'P19_ARM_|P19_DIAGNOSTIC_|P18_CENSUS_' \
  "$ROOT"/matched_cost/diagnostic/worker_pid*.jsonl | tail -n 35
find "$ROOT/matched_cost/failures" "$ROOT/census/failures" \
  -name '*.txt' -exec sh -c 'echo "$1"; head -n 60 "$1"' sh {} \;
~~~

Available modes:
- `OFF`: P18 only, no P19 matched-cost operation; isolates native
  P18 path under the P19 source SHA.
- `AR_EAGER`: P19 private native seeded AR operation without graph.
- `AR_GRAPH`: that same operation with capture and repeated replay.
- `K2_EAGER` / `K2_GRAPH`, likewise K3/K4: isolate each real wide arm.
  Use only legal depths (K2 d<=13, K3 d<=12, K4 d<=11).
- `ALL_GRAPH`: after individual graph arms pass, execute **all legal
  arms sequentially at ONE depth** and retain a separate phase trace
  for each arm. This diagnoses cross-arm graph/scratch interactions,
  but still does NOT produce valid P19A matched cost rows.

All modes must run via the **same** `p19_diag_once` client and a
new root/worker. `P19_DIAG_DEPTH` defaults to 1 and can target
other depths, notably 9. `P19_DIAG_FENCE=0` preserves the normal
async schedule; if an arm's last event is ambiguous, repeat with
`P19_DIAG_FENCE=1` to add an explicit device synchronize before the
P18 frame `.item()`. This changes scheduling, so a pass with the
extra fence is **not** evidence of a production-safe kernel.

Success reads
`P19_SINGLE_REQUEST_COMPLETED_DIAGNOSTIC_ONLY_NO_COST_SELECTION`;
failure reads
`P19_CUDA700_OR_OTHER_DIAGNOSTIC_FAILURE_NO_SELECTION`.
Both leave `matched_cost/rows` empty, and P19A/P19B analyzers reject
diagnostic roots. **Neither status is a P19A correctness or speed PASS.**

For `ALL_GRAPH`, make a NEW worker/root with
`P19_DIAG_ARM=ALL_GRAPH P19_DIAG_DEPTH=1`; run the same
`p19_diag_once` command. An individual-arm result and an
ALL_GRAPH result must never share the same server instance or output.

If OFF passes but AR_EAGER fails, focus on the native seeded sampler
and its buffer/shape contract. If AR_EAGER passes but AR_GRAPH fails,
focus on capture safety, warmup stream, graph inputs/lifetime and
allocator behavior. If AR_GRAPH passes but K2_GRAPH fails, isolate
P17 wide verifier/accept kernels under repeated graph replay. If a
targeted one-arm diagnosis passes but full P19 still fails, investigate
cross-arm graph resource reuse and interaction with subsequent native
steps. These are hypotheses, not root-cause proofs.

### 2. Do NOT run P19 main during isolation

A 24-request/336-cost-row run is premature until a diagnostic arm
reliably passes and the full four-arm smoke has fresh-worker correctness
and cost provenance. No DIL/HTTP trace alone proves the fault's first
GPU instruction. Add CUDA_LAUNCH_BLOCKING or Compute Sanitizer as a
subsequent *targeted* step only when the failing arm is identified
and the environment supports it.


## P19 Depth9 ALL_GRAPH false-negative: diagnostic frame not selected

The first `P19_DIAG_ARM=ALL_GRAPH P19_DIAG_DEPTH=9` diagnostic
completed its HTTP request and wrote 28 P18 census depth rows, but
reported `P19_DIAGNOSTIC_HTTP_STOP_WITHOUT_ACTUAL_ARM_EVIDENCE`.
There was **no evidence of CUDA700**. No P19_ARM_SELECTED or
P19_ARM_GRAPH_CAPTURE_BEGIN event at the requested depth existed.

**Confirmed source bug:** `maybe_measure()` returned early at all
depths other than the requested diagnostic depth (9), but its
frame-selection code still required `depth == 1` for the first
selection. This made D9 diagnostics **unreachable** on all frames.
D1 arm passes were real; the D9 result was a no-arm false-negative.

Now `select_cost_frame()` chooses the first frame at the configured
`diagnostic_depth` for isolated diagnostics. In ordinary formal P19A
mode, selection remains fixed at depth1 and subsequent selected-frame
depths 1..14 are unchanged. A CPU regression exercises D9, D14 AR-only,
formal D1 and max-frame gating.

If the request completes without actual arm evidence, the client now
reports `P19_DIAGNOSTIC_ARM_NOT_EXECUTED_NO_SELECTION` rather than
labeling it a possible CUDA700. Such results remain invalid.

**Re-run only after updating source and stopping the old server**:
new HEAD must match new worker source SHA. Use a brand-new directory,
e.g. `$HOME/results/qwen3tts-p19-diag-allgraph-d09-002`,
with `P19_DIAG_ARM=ALL_GRAPH P19_DIAG_DEPTH=9`. Do NOT reuse
`...-d09-001` or pretend its 28 census rows prove P19A Graph safety.
The new trace must contain `P19_COST_TARGET_FRAME_SELECTED` and,
for each of AR/K2/K3/K4, `P19_ARM_REPLAY_EAGER_PARITY_PASS` and
`P19_ARM_GRAPH_RELEASE_AND_SYNC_PASS`, then
`P19_DIAGNOSTIC_FINAL_RESTORE_SYNC_PASS`. Stop on first failure.


## P19 cumulative native-frame lifecycle: ALL_DEPTHS_GRAPH

The isolated D1 and D9 `ALL_GRAPH` diagnostics passed all four
arms, the replay/eager and physical KV checks, and the native request
completed. This **does not** address repeated Graph capture/release
over all depths. The original P19 smoke CUDA700 remains unresolved.

### Single real frame, all 14 depths and 50 legal Graph arms

New mode `ALL_DEPTHS_GRAPH` measures **one selected native frame**
across successive actual native AR checkpoints. It runs all legal arms:
d1..11 AR+K2+K3+K4 (44 arms), d12 AR+K2+K3 (3), d13 AR+K2
(2), d14 AR only (1). Total: **50 Graph arm lifecycles**.

The diagnostic depth flag is now the INCLUSIVE terminal depth;
`P19_DIAG_DEPTH=14` runs the full residual frame. For a bisect
use e.g. `P19_DIAG_DEPTH=8` to run the first 8 depths, not one
isolated D8. This mode requires `P19_COST_FRAMES=1` and uses
`P18_CAPTURE_FRAMES=2` to preserve P18's two independently
validated native frames. Only one of those gets Graph costs.

Before starting, stop the previous server, update P19 worktree and
run the CPU tests.

Terminal A:

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
P19_DIAG_ARM=ALL_DEPTHS_GRAPH \
P19_DIAG_DEPTH=14 \
P19_DIAG_FENCE=0 \
P18_CAPTURE_FRAMES=2 \
P19_COST_FRAMES=1 \
bash experiments/qwen3_tts_cp_spec_k2/p19_launch.sh \
  "$HOME/results/qwen3tts-p19-diag-alldepths-d14-001"
~~~

Terminal B, once healthy:

~~~bash
cd ~/sglang-omni-worktrees/p19-matched-dp
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p19-diag-alldepths-d14-001"
curl -fsS http://127.0.0.1:8000/health
python -m experiments.qwen3_tts_cp_spec_k2.p19_diag_once \
  --root "$ROOT"
~~~

Expected if ALL 50 arm traces and full worker restores are present:

`P19_ALL_DEPTHS_GRAPH_SINGLE_FRAME_LIFECYCLE_PASS_NOT_COST_OR_SERVING`

`P19_DIAG_COVERAGE=depths=[1,...,14] graph_arms=50 frame=37`
(the actual frame index may differ).

If the HTTP request finishes but any phase is missing, it fails closed
with `P19_DIAGNOSTIC_ARM_NOT_EXECUTED_NO_SELECTION`, not a
CUDA700 label. An actual illegal-address error is distinct and
requires a brand-new CUDA worker for a repeat.

The client writes
`matched_cost/diagnostic/client.json` with strict
`trace_coverage`, and phase events are fsync'ed in
`matched_cost/diagnostic/worker_pid*.jsonl`. To see the earliest
incomplete boundary rather than the final P18-only tail:

~~~bash
ROOT="$HOME/results/qwen3tts-p19-diag-alldepths-d14-001"
grep -hE \
  'P19_COST_TARGET_FRAME_SELECTED|P19_ARM_SELECTED|P19_ARM_GRAPH_CAPTURE_BEGIN|P19_ARM_REPLAY_EAGER_PARITY_PASS|P19_ARM_GRAPH_RELEASE_AND_SYNC_PASS|P19_DIAGNOSTIC_FINAL_RESTORE_SYNC_PASS' \
  "$ROOT"/matched_cost/diagnostic/worker_pid*.jsonl \
  | tail -n 70
cat "$ROOT/matched_cost/diagnostic/client.json"
~~~

This intentionally never emits formal
`matched_cost/rows/*.json`. No P19A cost table, P19B DP selection,
native full-frame Graph-ON benchmark, HTTP QPS, or quality improvement
can be claimed even on a complete 50-arm PASS.

If the 1..14 frame FAILS, use a fresh worker/root with an inclusive
`P19_DIAG_DEPTH=7` or `9` to find the first failing prefix; each
prefix run retains all earlier arm interactions, unlike isolated
`ALL_GRAPH`. A pass at D8 plus a fail at D14 can narrow the
source but cannot alone prove the first kernel fault if timing or
memory reuse is nondeterministic. Use targeted synchronization or
sanitizer tools only after the smallest reliable failing prefix
is identified.

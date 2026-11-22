# P18A/P18B — Qwen3-TTS native AR all-depth speculative acceptance census

**Research-only. Native AR EAGER on-policy trajectories, not Graph-ON serving
or a full speculative decoder. No release/merge or throughput GO.**

Base validated P17 SHA: `de4075501633a4d5054d820ea2eb13039fb5d8dc`.
P17 proved exact K2/3/4 wide verification at two checkpoints; early K3/4
had no benefit while later K3 had a small local advantage. The next
question is whether this pattern persists across REAL native-AR sampled
frames, all eligible depths and independently requested prompts/seeds.

## Scope / important semantic distinction

**P18A:** Intercepts `Qwen3TTSTalker.code_predictor_forward` ONLY when an
explicit `ARM_P18` file exists. For that request it executes the native
`code_predictor_forward_incremental` eager path instead of replaying
the existing full Predictor CUDA Graph. Its **native sampler remains
unchanged**. After each native codebook code is sampled, before its
embedding is appended to KV, the observer freezes exactly the
`old_hidden / last_embedding / accumulated / cache_len` state and
runs one paired uniform tuple across all FEASIBLE K2/K3/K4 widths.

At EACH trial it uses the same P17 GPU multi-head proposal, ONE real
wide causal verifier and P17 independent reference, verifying native
LM-head probabilities, CPU accept/residual, all GPU outputs and every
physical K/V slot. It restores physical KV before native AR continues.
All probes must preserve native CPU/CUDA RNG state.

A source-locked one-row-per-real-(request,frame,depth) JSON record
is written under `ROOT/census/rows`. The worker writes its own
`request_hash` from its actual scheduler request ID, while the client
sets a single-case marker for each C1 request. The case marker is only
safe on an ISOLATED port with NO external traffic and exactly one
client request at a time. Do not reuse this mechanism in production.

Eligible widths are determined by available bonus-head indices:
- Last committed depth 1..11: K2,K3,K4
- Depth 12: K2,K3
- Depth 13: K2
- Depth 14: AR only (no legal K2 bonus head)
- The first codebook pair (prior to depth 1) is unchanged native AR.

**P18A sampling design:** one deterministic paired counterfactual
uniform tuple per checkpoint. Smoke uses 2 prompts x 2 seeds x 2
correlated frames = 4 request clusters. Main uses 8 prompts x 3
seeds x 8 correlated frames = **24 unique request clusters** and 192
checkpoints per eligible depth. Frames/widths in the same request
MUST NOT be treated as independent observations. Bootstrap resamples
whole requests, keeping frames and widths paired.

Native codebook sampling remains native AR (eager). The speculative
outcomes are counterfactuals from those native states; they are NOT
on-policy for an actually deployed speculative decoder whose future
states change. Audio finish=stop must be verified. Neither HTTP wall
time nor WAV bytes prove a serving speedup/quality claim.

**P18B:** Conservatively combine per-depth P18A committed-code
histograms with *previous* P17 H200 CUDA Graph full-wide-round costs
at depth1 and depth9. Cost at other depths is clearly tagged as
INTERPOLATED or EXTRAPOLATED, not measured. Compare K2/K3/K4 using
paired request-cluster bootstrap 95% acceptance-only uncertainty.
`AR` is unpriced: the old native Graph-ON full-Predictor 2.232ms/frame
is NOT a matched per-code AR cost. There is no valid
production-wide AR/Spec width policy yet. No 14-depth causal rollout
or end-to-end speedup inference is authorized.

## 0. Prepare a dedicated worktree, no re-bootstrap

Use the already working H200 `~/.venv` and your P17 git worktree
as the control repository. Do not switch the existing P17 branch.

~~~bash
P17="$HOME/sglang-omni-worktrees/p17-multiwidth"
P18="$HOME/sglang-omni-worktrees/p18-depth-census"
git -C "$P17" fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-cpspec-p18-depth-census
git -C "$P17" worktree add -b \
  experiment/qwen3-tts-cpspec-p18-depth-census "$P18" FETCH_HEAD
cd "$P18"
source ~/.venv/bin/activate
git status --short
git rev-parse HEAD

python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p18_census.py \
  experiments/qwen3_tts_cp_spec_k2/p18_run.py \
  experiments/qwen3_tts_cp_spec_k2/p18_analyze.py \
  experiments/qwen3_tts_cp_spec_k2/p18_cost_model.py
bash -n experiments/qwen3_tts_cp_spec_k2/p18_launch.sh
python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p17.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p18.py -m "not accelerator"
~~~

On a machine where the P18 local branch already exists, use
`git worktree add "$P18" experiment/qwen3-tts-cpspec-p18-depth-census`
instead of `-b`; never reset any active worktree.

## 1. H200 smoke, FIRST (4 requests x 2 frames)

Stop the old P17 server. Do not run another GPU-heavy experiment
concurrently. Qwen3-TTS may reserve ~110 GiB of H200 KV cache.

Terminal A (new absent evidence root):

~~~bash
cd ~/sglang-omni-worktrees/p18-depth-census
source ~/.venv/bin/activate
P18_CAPTURE_FRAMES=2 bash \
  experiments/qwen3_tts_cp_spec_k2/p18_launch.sh \
  "$HOME/results/qwen3tts-p18-smoke-001"
~~~

Terminal B (after /health=200):

~~~bash
cd ~/sglang-omni-worktrees/p18-depth-census
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p18-smoke-001"
curl -fsS http://127.0.0.1:8000/health
python -m experiments.qwen3_tts_cp_spec_k2.p18_run \
  --root "$ROOT" --mode smoke
python -m experiments.qwen3_tts_cp_spec_k2.p18_analyze \
  --root "$ROOT" --out "$ROOT/analysis"
cat "$ROOT/analysis/summary.md"
~~~

Expected P18A smoke status:
`P18A_ALL_DEPTH_CLUSTERED_ACCEPTANCE_CENSUS_PASS_NO_SPEEDUP`.
No performance or policy decision is allowed from 4 request clusters.

If the smoke fails, NEVER reuse that root. Inspect the FIRST file under
`ROOT/census/failures`, `ROOT/visits.json`, and the last server
traceback. A real parity failure must be fixed, not weakened.

## 2. Main P18A corpus (24 independent request clusters)

After stopping the smoke server, start a *fresh* main source-locked
worker with **8 frames per request**:

Terminal A:

~~~bash
cd ~/sglang-omni-worktrees/p18-depth-census
source ~/.venv/bin/activate
P18_CAPTURE_FRAMES=8 bash \
  experiments/qwen3_tts_cp_spec_k2/p18_launch.sh \
  "$HOME/results/qwen3tts-p18-main-001"
~~~

Terminal B:

~~~bash
cd ~/sglang-omni-worktrees/p18-depth-census
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p18-main-001"
curl -fsS http://127.0.0.1:8000/health
python -m experiments.qwen3_tts_cp_spec_k2.p18_run \
  --root "$ROOT" --mode main
python -m experiments.qwen3_tts_cp_spec_k2.p18_analyze \
  --root "$ROOT" --out "$ROOT/analysis"
cat "$ROOT/analysis/summary.md"
~~~

The main gate requires 24 distinct worker request IDs,
8 frames/request x 14 depths = **2688 on-policy native depth records**,
the correct eligible width set at EVERY depth, and all independent
P17 oracle parity checks. Case ID alone is insufficient:
a request ID collision or swapped source/seed fails closed.

## 3. P18B paired adaptive width screen (OFFLINE)

No running server/GPU needed. Use the **existing P17 source-locked**
successful report as the previous cost evidence. If its path differs,
locate the real `smoke/report.json` first. Never invent costs.

~~~bash
cd ~/sglang-omni-worktrees/p18-depth-census
ROOT="$HOME/results/qwen3tts-p18-main-001"
P17_REPORT="$HOME/results/qwen3tts-p17-multiwidth-008/smoke/report.json"
test -f "$P17_REPORT"

python -m experiments.qwen3_tts_cp_spec_k2.p18_cost_model \
  --root "$ROOT" \
  --p17-cost-report "$P17_REPORT" \
  --out "$ROOT/cost-model"
cat "$ROOT/cost-model/summary.md"
~~~

Expected status:
`P18B_PAIRED_REQUEST_WIDTH_COST_SCREEN_NOT_NATIVE_AR_OR_E2E_GO`.

The model performs paired request-level bootstrap for each eligible
width against K2 and only flags a width hypothesis if its lower 95%
acceptance-only improvement exceeds 5% in the main cohort. Even a
successful P18B result is NOT a production runtime recommendation.

## 4. Go/No-Go criteria for next engineering investment

- If early depths remain near 1.0 code/round and later widths do not
  show multiple depth-local opportunities, stop SpecDec research.
- If several depths show request-cluster-consistent K3/K4 benefit,
  the **next** task is matched native AR per-step CUDA Graph costs,
  more independent prompts/seeds, and only then a *minimal* real
  adaptive decoder. Measure entire Graph-ON native/E2E ABBA with
  sampling/quality parity BEFORE any speedup claim.
- No per-request refusal override, seeded full-decoder parity,
  listening/ASR assessment, quality or HTTP QPS comparison is
  established by P18A/B. No PR merge from this study.


## Optional one-command Terminal B wrapper

Instead of invoking the client driver, analyzer, and P18B model separately,
you may use the existing fail-closed `p18_measure.sh` wrapper. This still
requires the running matching server from Terminal A and a NEW evidence root:

~~~bash
cd ~/sglang-omni-worktrees/p18-depth-census
source ~/.venv/bin/activate
# SMOKE: Terminal A uses P18_CAPTURE_FRAMES=2
bash experiments/qwen3_tts_cp_spec_k2/p18_measure.sh \
  "$HOME/results/qwen3tts-p18-smoke-001" smoke

# MAIN: Terminal A uses P18_CAPTURE_FRAMES=8, new worker/root
bash experiments/qwen3_tts_cp_spec_k2/p18_measure.sh \
  "$HOME/results/qwen3tts-p18-main-001" main \
  "$HOME/results/qwen3tts-p17-multiwidth-008/smoke/report.json"
~~~

The optional third argument is allowed ONLY for the main cohort and
must be an existing real P17 report; it runs P18B offline after P18A
pass. Do not run this wrapper twice on the same evidence root.

### P18 live native-state invariants

The P18 observer copies the native original hidden, projected embedding,
accumulated code sum, sampled native code, CPU/CUDA RNG state, and all
16 physical KV slots. Every counterfactual width runs on a clone of
the real sampled checkpoint. After the P17 independent oracle, the
probe compares ALL original live tensors and restores them in a
`finally` block, even if a wide-verifier kernel aliases scratch storage.
Any mutation or failure invalidates the experiment; it is never
accepted as speedup evidence.

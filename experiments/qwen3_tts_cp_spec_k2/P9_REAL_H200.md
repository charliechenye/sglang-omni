# P9 — Real H200 four-offset K2 verification + GPU sampler cost screen

**DO NOT CLAIM E2E SPEEDUP FROM THIS REPORT.**
This phase moves from isolated P8 synthetic-control loops toward a measured
real-checkpoint GPU-only K2 implementation. P9A and P9B execute actual
Qwen3-TTS weights/logits, but neither replaces the current CPU-sync K2
serving path; actual full GPU-resident end-to-end K2 is **NOT ready**.

Prior evidence:
- P2: 720 frames, 6974 verify rounds, 49.04% acceptance, 1.445 committed
  residual codes/round; source SHA 398297b1...
- P5: real complete Graph ON predictor ~2.2266ms/frame, 2-token verifier
  graph ~0.139ms/call; source SHA 00858c4...
- P6.1: native CUDA device-WHILE conditional dynamic 7/14 steps PASS.
- P7: 5-layer Predictor GPU conditional KV offset2/6 branch parity 3/3 PASS.
- P8: 5-layer Predictor nested GPU WHILE+IF, 6/6 parity PASS; 1-round
  ~0.184ms, 2-round ~0.356ms. These alone are NOT speedup.

P9A uses ONE CUDA Graph to route actual five-layer Predictor between
FOUR fixed KV offsets 2/6/10/14 under a GPU WHILE, and a GPU decision
mask can skip the next *probe stage*. At proposal depth14 the real
cache already contains 14 committed KV positions. Each graph replay
compares accumulated hidden and ALL 16 model KV slots to independently
run serial references with the identical KV prefix. Tests include 2,
3 and 4 model calls and six different routed paths. **This is not an
on-policy K2 bonus mechanism**: offsets are spaced and frozen inputs
come from actual depth14. The result measures routing cost and backend
compatibility, not K2 quality.

P9B takes the real candidate's target p and head-reuse draft q logits
from one C1 request at residual depths 2 and 10. It captures GPU-only
top-k/top-p normalization, sampling from q, accept min(1,p(x)/q(x)),
and rejection correction from normalized positive residual (p-q)+.
No .item() in the graph; uses a PRIVATE CUDA RNG, checked to advance
across replays. Deterministic draws are compared with the existing
PyTorch sampling_probs reference (4 uniforms per depth). **Not yet
proven identical to the native Omni fused sampler.** P9B does not
commit any code/KV or touch live RNG.

P9 runner records a FRESH existing optimized Graph ON baseline from
the SAME process, SAME code commit, SAME GPU and checkpoint as each
component probe, under dedicated baseline ARM marker. This corrects
the source-SHA mismatch in P4/P5 screens. Both mode A and mode B
perform baseline/spec/spec/baseline sequentially, verifying natural EOS
and candidate-only activity. HTTP timings include deliberate profiling
and are NOT a performance ABBA.

## Step 0 — pull code and run CPU / H200 gates

STOP any previous Omni server first. Use the same branch SHA for ALL
P9A and P9B runs; do not edit or cherry-pick between the two launches.

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git status --short
git rev-parse HEAD

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p9a.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p9b.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p9_budget.py \
  -k 'not h200'

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p9a.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p9b.py \
  -k h200 -s -x
~~~

Do NOT proceed to HTTP if any CUDA graph test fails. A failed capture
may invalidate that process's CUDA stream; use a fresh process.

## Step 1 — P9A model-routing cost in a fresh worker (terminal A)

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p9a-012"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p9a-012/ARM_SPEC"

export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p9a-012/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p9a-012/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P9A_DIR="$HOME/results/qwen3tts-cpspec-k2-p9a-012/four"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9A_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p9a-012/server.log"
~~~

Terminal B (AFTER /health succeeds):

~~~bash
cd ~/sglang-omni

python -m experiments.qwen3_tts_cp_spec_k2.p9_real_probe \
  --mode a \
  --base-url http://localhost:8000 \
  --baseline-arm-file "$HOME/results/qwen3tts-cpspec-k2-p9a-012/ARM_BASE" \
  --spec-arm-file "$HOME/results/qwen3tts-cpspec-k2-p9a-012/ARM_SPEC" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p9a-012/server.log" \
  --baseline-graph-dir "$HOME/results/qwen3tts-cpspec-k2-p9a-012/baseline" \
  --probe-dir "$HOME/results/qwen3tts-cpspec-k2-p9a-012/four" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p9a-smoke-012"
~~~

Expected: 4/4 stop; 1 baseline_graph_pid*.json; 1
four_depth_pid*.json with 6/6 hidden and whole KV PASS. On failure,
preserve server log and the probe directory; DO NOT call it performance.

## Step 2 — P9B GPU sampling cost, NEW worker (terminal A)

STOP the P9A server before starting P9B. SAME code SHA and GPU; NEW
ARM/output dirs, and no competing workloads.

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p9b-013"

unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p9b-013/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p9b-013/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p9b-013/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P9B_DIR="$HOME/results/qwen3tts-cpspec-k2-p9b-013/sampler"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9B_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p9b-013/server.log"
~~~

Terminal B:

~~~bash
cd ~/sglang-omni

python -m experiments.qwen3_tts_cp_spec_k2.p9_real_probe \
  --mode b \
  --base-url http://localhost:8000 \
  --baseline-arm-file "$HOME/results/qwen3tts-cpspec-k2-p9b-013/ARM_BASE" \
  --spec-arm-file "$HOME/results/qwen3tts-cpspec-k2-p9b-013/ARM_SPEC" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p9b-013/server.log" \
  --baseline-graph-dir "$HOME/results/qwen3tts-cpspec-k2-p9b-013/baseline" \
  --probe-dir "$HOME/results/qwen3tts-cpspec-k2-p9b-013/sampler" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p9b-smoke-013"
~~~

Expected: two independently captured real-logit samplers from K2
proposal depths 2 and 10, same source SHA as baseline, four exact
CPU-reference uniform parity checks each, unique independent CUDA
RNG replays, and positive CUDA Graph sampling cost.

## Step 3 — real Graph ON vs components budget screen

Use P9A's FRESH Graph ON baseline, not the old P5 baseline.

~~~bash
cd ~/sglang-omni

python -m experiments.qwen3_tts_cp_spec_k2.p9_budget_screen \
  --p9a-dir "$HOME/results/qwen3tts-cpspec-k2-p9a-012/four" \
  --p9b-dir "$HOME/results/qwen3tts-cpspec-k2-p9b-013/sampler" \
  --baseline-graph-dir "$HOME/results/qwen3tts-cpspec-k2-p9a-012/baseline" \
  --p9b-baseline-graph-dir "$HOME/results/qwen3tts-cpspec-k2-p9b-013/baseline" \
  --p2-acceptance-report "$HOME/results/qwen3tts-cpspec-k2-p2-report-004/report.json" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p9-budget-014"

cat "$HOME/results/qwen3tts-cpspec-k2-p9-budget-014/summary.md"
~~~

The CPU budget analyzer fails on missing P9A parity, missing P9B
real-logit sampling, missing true Graph ON, or nonmatching P9 source.
It additionally compares both fresh-process Graph ON baselines and
rejects an unstable GPU/environment if their median differs by >10%.
The historical P2 acceptance source remains different, and is
explicitly identified as a separate cohort.

## Interpretation / next action

**NO REAL E2E SPEEDUP CAN BE CLAIMED FROM P9.** The final component
screen extrapolates just 2–4 nonadjacent offset graph calls to ~9.686
rounds/frame, then adds independently captured real-logit GPU sampling
graph cost. These are separate graph launch envelopes, not a validated
single composite graph with GPU-controlled true K2 accept/bonus. It
omits the opening pair, dynamic KV, codebook head/bonus and embeddings.

- If the optimistic component cost is *already* greater than full
  Predictor Graph ON baseline, STOP naive four-way GPU cond expansion.
  First consider dynamic KV offsets in one fused verifier kernel, or a
  different speculation strategy; do not blindly build a 14-way tree.
- If the component cost is substantially below Graph ON baseline,
  implement real 14-depth on-policy GPU K2 next, starting with native
  fused sampler parity and dynamic KV ownership, then test with fresh
  source-locked Graph ON HTTP throughput/TTFA vs baseline at C1,
  expanding to C8/C16 only after quality and parity gates.
- Small positive headroom is not a green light: unknown costs could
  erase it. Record per-depth GPU phase costs before committing
  weeks to kernel implementation.

No model retraining, TensorRT or extra GPU is required to run P9. Do
not mutate the source code between Steps 1 and 2.

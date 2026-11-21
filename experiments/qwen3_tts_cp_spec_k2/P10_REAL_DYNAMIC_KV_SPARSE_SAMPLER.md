# P10 H200 — single-graph GPU dynamic KV & sparse fused speculation sampler

**Goal:** two *real-checkpoint* cost/accuracy gates after P9 component
budget rejected naive nested four-way CUDA conditional route:
P9 real model branch-extrapolation ~2.000ms/frame, P9 Torch-op dense
sampler ~1.042ms/frame, full native Graph-ON ~2.240ms/frame, missing
~0.802ms/frame even before codebook heads, bonus and KV management.

Do NOT extend to 14-way CUDA cond; do NOT claim a serving speedup.

P10A tests ONE CUDA graph with an on-device cache_len scalar, reusing
the CURRENT optimized FusedPredictorLayers QKV, norm, GEMV, MLP and
cache allocation. New Triton attention differs only in loading the
effective END from a GPU scalar pointer, while positions are produced
from GPU depth. Static baseline is the ordinary existing five-layer
FusedPredictorLayers model, separately captured for each offset. One
H200 checkpoint, same input, same 16-slot KV prefix; each offset
2,6,10,14 compares hidden and ALL KV against eager and static graph.
**P10A FAILS CLOSED if active model instead uses CuTeDSL or unfused
Predictor.** Never switch the measured production backend silently.
P10A is a verifier component only. It does not dynamically update the
rest of the speculative decoder's K2 depth, bonus, or sampled codes.

P10B tests actual on-policy K2 target/draft p/q logits from depths 2
and 10, vocab 2048 with T=.9, topk50, top-p=1. A sparse Triton
kernel consumes PyTorch topk(50) sorted values/indices for p and q,
then fuses softmax/top-p, draft proposal, accept min(1,p(x)/q(x)),
and correction from normalized positive residual (p-q)+ with
at most 50 nonzero candidates. This eliminates dense vocab-wide
scatter/searchsorted but **still launches two Torch topk kernels**.
Compares p/q probabilities at original 1e-6 tolerance, 4 deterministic
proposal/accept/replacement draws against independent CPU oracle,
checks private CUDA Generator replay advancement, and records SAME
PROCESS full-GPU CUDA Graph launch envelope against the old dense P9B
implementation. It does not change live K2 outputs or production RNG.

The P10 cost gate uses source-identical full native Graph-ON baselines
from each of the two fresh Omni workers and the clearly labeled,
reconstructed historical P2 720-frame aggregate (6974/720 rounds/frame).
It refuses >10% baseline cross-process drift or mixed source SHAs.
A >=0.30ms/frame estimated remaining headroom is a *preliminary*
engineering reserve, NOT proof a full GPU K2 can fit or deliver gains.

## 0. Pull EXACT source and run CPU/H200 tests (stop any old server)

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git status --short
git rev-parse HEAD

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p10a.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p10b.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p10_cost_gate.py \
  -k 'not accelerator'

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p10a.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p10b.py \
  -m accelerator -s -x
~~~

CPU tests: pure input/support/provenance checks. GPU unit tests:
P10A custom Triton device-end decode matches original optimized static
decode exactly for 2/6/10/14 from ONE captured graph; P10B 2048-token
C1 sparse distributions and sampled tokens agree at T=.9/topk50 with
top-p=1.0 and 0.9 and show unique independent random draws.

**If ANY GPU test fails, STOP and share the first traceback, no server.**
Do not alter numeric tolerance to force passing. If a CUDA capture fails,
start a FRESH Python process for the next diagnostic.

## 1. P10A real five-layer dynamic KV probe, fresh H200 server

Terminal A:

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p10a-016"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_P10B_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p10a-016/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p10a-016/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p10a-016/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P10A_DIR="$HOME/results/qwen3tts-cpspec-k2-p10a-016/dynamic_kv"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P10A_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p10a-016/server.log"
~~~

Terminal B, ONLY AFTER /health is healthy:

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p10_real_probe \
  --mode a \
  --base-url http://localhost:8000 \
  --baseline-arm-file "$HOME/results/qwen3tts-cpspec-k2-p10a-016/ARM_BASE" \
  --spec-arm-file "$HOME/results/qwen3tts-cpspec-k2-p10a-016/ARM_SPEC" \
  --baseline-graph-dir "$HOME/results/qwen3tts-cpspec-k2-p10a-016/baseline" \
  --probe-dir "$HOME/results/qwen3tts-cpspec-k2-p10a-016/dynamic_kv" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p10a-016/server.log" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p10a-smoke-016"
~~~

Expected: 4/4 natural stop, 1 complete Graph-ON baseline report
and ONE P10A real five-layer device-offset probe; 4/4 exact hidden
and full K/V parity against same-worker static baseline. Each probe
case outputs both dynamic and fixed graph CUDA event ms.

If it fails with "P10A_BACKEND_OR_SHAPE_UNSUPPORTED", your *active*
runtime did not select FusedPredictorLayers. Preserve this backend
evidence; do NOT switch backend without a new matched reference. Need
a separate CuTeDSL dynamic-KV path rather than asserting that optimized
native kernel is slow.

## 2. P10B real on-policy sparse proposal/accept cost, NEW server

STOP P10A server first. DO NOT change git SHA or GPU between arms.
Use another output root and new ARM markers.

Terminal A:

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p10b-017"

unset SGLANG_OMNI_QTTS_CP_P10A_DIR
unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p10b-017/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p10b-017/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p10b-017/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P10B_DIR="$HOME/results/qwen3tts-cpspec-k2-p10b-017/sparse"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P10B_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p10b-017/server.log"
~~~

Terminal B:

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p10_real_probe \
  --mode b \
  --base-url http://localhost:8000 \
  --baseline-arm-file "$HOME/results/qwen3tts-cpspec-k2-p10b-017/ARM_BASE" \
  --spec-arm-file "$HOME/results/qwen3tts-cpspec-k2-p10b-017/ARM_SPEC" \
  --baseline-graph-dir "$HOME/results/qwen3tts-cpspec-k2-p10b-017/baseline" \
  --probe-dir "$HOME/results/qwen3tts-cpspec-k2-p10b-017/sparse" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p10b-017/server.log" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p10b-smoke-017"
~~~

Expected: 4/4 natural stop; real p/q logits at depths 2 and 10;
sparse fused vs dense P9B matched graph costs, and full numerical
parity, source SHA/PID and independent RNG replay checks.

## 3. P10 source-locked cost gate, NO SERVER / NO GPU

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p10_cost_gate \
  --p10a-dir "$HOME/results/qwen3tts-cpspec-k2-p10a-016/dynamic_kv" \
  --p10b-dir "$HOME/results/qwen3tts-cpspec-k2-p10b-017/sparse" \
  --p10a-baseline "$HOME/results/qwen3tts-cpspec-k2-p10a-016/baseline" \
  --p10b-baseline "$HOME/results/qwen3tts-cpspec-k2-p10b-017/baseline" \
  --p2-aggregate "experiments/qwen3_tts_cp_spec_k2/data/p2_720_historical_aggregate_reconstructed.json" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p10-budget-018"

cat "$HOME/results/qwen3tts-cpspec-k2-p10-budget-018/summary.md"
~~~

P2 input is **historical reconstructed aggregate ONLY**; never report
it as recovered raw traces/parity. P10A and P10B and both Graph-ON
baselines require identical source SHA; the two independent base
worker medians must agree within 10%. If output already exists,
choose a fresh numbered output rather than overwrite any evidence.

## Decision rules

- **Parity FAIL or backend mismatch:** stop and inspect first H200
  traceback; no performance selection. Never silently downgrade backend.
- **P10A device-KV event >= fixed Graph and/or >~0.15ms:** treat single
  dynamic KV kernel as not yet cost competitive; inspect pointer
  indexing/extra ops and PDL scheduling.
- **P10B sparse event >= dense P9B or >~0.03ms:** topk remains a bottleneck;
  next candidate must fuse/block-sort within a custom CUDA kernel
  or reuse existing CuTeDSL sampler, not another full-vocab scatter.
- **P10 component remaining <0.30ms/frame:** stop full 14-depth K2
  integration; the complete K2 heads/bonus/embeddings still need time.
- **P10 component remaining >=0.30ms/frame:** only an exploratory GO
  to integrate full real on-policy GPU K2, with exact speculative
  sampling distribution / native RNG, KV ownership/rollback, natural EOS,
  WER/similarity, and then profiler-OFF fresh-process Graph ON ABBA
  on frozen matched prompts at C1 followed by C8/C16.

No file in P10 should modify the default production inference path;
both probes are opt-in under post-health ARM. No speedup is claimed
until complete native GPU K2 serving is implemented and measured.

# P11 H200 — sampler cost attribution and real hybrid AR/K2

P10 source-locked H200 measurements: native complete Graph ON 2.23632
ms/frame; dynamic 5-layer KV verifier 0.14516 ms/round; sparse topk
and fused speculative sampler 0.06622 ms/round. P10 component budget
2.04745 ms/frame leaves 0.18887 ms/frame, below the 0.30 ms reserve.
Full GPU K2 is NO-GO at current cost, and no actual HTTP speedup is proven.

P11A identifies the actual sampling bottleneck on real p/q states,
without another multi-branch Predictor. It captures four independent
device CUDA graphs: both torch.topk(50) operations, Triton post-topk
accept/residual, private RNG-only, and complete sparse+RNG. It measures
each across four interleaved ABBA-like blocks and checks P10B p/q and
deterministic sampling parity first. Independent timing envelopes are
NOT additive and do not imply native GPU K2 speedup.

P11B runs genuine direct target AR for residual depth 2 and 3 before
the real debug sampled K2 loop at depth 4. The direct AR step consumes
the last committed embedding at its true KV slot, samples the target
head from the new hidden state, commits code/embedding and advances
cache length by exactly one. All subsequent K2 steps use the unchanged
packed verifier/accept/bonus algorithm and the real weights/heads/KV.
The user-controlled HYBRID ARM marker is separate from the regular
K2 ARM: without it, the original K2 runs unchanged in the same worker.

Do not interpret HTTP wall time as throughput: this K2 debug path
still uses Python/GPU synchronizations. AR and K2 use the debug torch
generator, NOT the optimized native fused sampler's RNG; WAV hashes
are not supposed to equal the native baseline.

## 0. Refresh experimental branch and run tests (stop old Omni first)

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git rev-parse HEAD
git status --short

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p11.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p11_runners.py \
  -m "not accelerator"

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p11.py \
  -m accelerator -s -x
~~~

Stop immediately if a numerical, graph, compilation or RNG test fails.
Do not relax probability tolerances. A failed CUDA stream capture may
require a NEW Python process to test again.

## 1. P11A fresh H200 server — terminal A

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p11a-019"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_P10A_DIR
unset SGLANG_OMNI_QTTS_CP_P10B_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p11a-019/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p11a-019/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p11a-019/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P11A_DIR="$HOME/results/qwen3tts-cpspec-k2-p11a-019/attribution"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P11A_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p11a-019/server.log"
~~~

P11A terminal B, after /health succeeds:

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p11a_sampler_live \
  --base-url http://localhost:8000 \
  --base-arm "$HOME/results/qwen3tts-cpspec-k2-p11a-019/ARM_BASE" \
  --spec-arm "$HOME/results/qwen3tts-cpspec-k2-p11a-019/ARM_SPEC" \
  --baseline-dir "$HOME/results/qwen3tts-cpspec-k2-p11a-019/baseline" \
  --attribution-dir "$HOME/results/qwen3tts-cpspec-k2-p11a-019/attribution" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p11a-019/server.log" \
  --expect-source-sha "$(git rev-parse HEAD)" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p11a-019/smoke"

cat "$HOME/results/qwen3tts-cpspec-k2-p11a-019/smoke/summary.md"
~~~

P11A gate: four natural stops, source SHA and PID agreement with the
real full Graph ON baseline, exactly one real on-policy depth 2
and one depth 10 p/q JSON, four timing blocks for each stage,
P10B distribution/sampled-token parity and private RNG advancement.
If any gate fails: no cost selection.

## 2. P11B — STOP P11A; start NEW model worker, terminal A

Do not edit/cherry-pick code or change GPU between P11A and P11B.

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p11b-020"

unset SGLANG_OMNI_QTTS_CP_P11A_DIR
unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_P10A_DIR
unset SGLANG_OMNI_QTTS_CP_P10B_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR
unset SGLANG_OMNI_QTTS_CP_P9_BASE_DIR
unset SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p11b-020/ARM_K2"
export SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p11b-020/ARM_HYBRID"
export SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH=3
export SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT=1
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p11b-020/server.log"
~~~

P11B terminal B:

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p11b_hybrid_smoke \
  --base-url http://localhost:8000 \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p11b-020/server.log" \
  --spec-arm "$HOME/results/qwen3tts-cpspec-k2-p11b-020/ARM_K2" \
  --hybrid-arm "$HOME/results/qwen3tts-cpspec-k2-p11b-020/ARM_HYBRID" \
  --expect-source-sha "$(git rev-parse HEAD)" \
  --ar-through 3 \
  --seeds 20261120,20261121 \
  --out "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke"

cat "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/summary.md"
~~~

P11B gate: for every seed, native/K2/hybrid/hybrid/K2/native
must all naturally stop. Native must have zero K2 frames. K2
must have zero AR steps. Hybrid must generate depth 2 and 3 via
actual target AR and its remaining depths via true K2 proposal/bonus,
with every depth 2..15 committed exactly once. The total target
forward call count and proposed/accepted/bonus frequency are recorded
per generated codec frame, and the source SHA must match.

Use additional seeds with NEW output directories if the first two
seeds pass; a single seed is not a distributional quality study.
Natural stop is not WER, voice similarity, RNG equivalence or speedup.

## Decision rules

- P11A: find whether two torch.topk calls dominate the full sparse graph.
  If yes, prototype a fused dual top-50 selector, preserving ties,
  top-p and native seeded RNG distribution. If post-topk dominates,
  optimize that kernel instead; the top-50 vocabulary-ID CDF currently
  has O(K^2) comparisons.
- P11B: compare actual per-depth acceptance and target forward calls
  per emitted frame. Fewer K2 proposals do not necessarily mean less
  GPU work: direct AR also calls the real target model.
- Quality must be evaluated separately with ASR WER and speaker
  similarity on frozen prompts before any merge. Never use different
  seeded WAV hashes as a strict correctness verdict.
- Actual serving speedup requires a COMPLETE native GPU-resident K2
  implementation, correct real 14-depth state/bonus/KV and profiler-OFF
  fresh-worker Graph ON HTTP ABBA at C1 then C8/C16.

No production merge or speedup claim from P11 component evidence.

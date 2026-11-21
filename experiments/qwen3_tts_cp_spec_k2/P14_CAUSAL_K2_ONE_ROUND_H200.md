# P14 H200 — real causal one-round K2 GPU Graph, exactness then cost

**Research Draft PR #89; NOT a native 14-depth GPU SpecDec integration.**

The most useful P12A H200 result was measured on source
\`ac89ef8a00dc4f94f2c7b1392902ffbfe3b6e501\`: independent real
q-before and p-after per-distribution Triton Top50 candidate Graphs
~0.0088–0.0089ms each vs torch Topk+draw ~0.0267–0.0353ms.
Both depths2/10 showed matching p/q support and deterministic draws
on four actual BF16 on-policy states. However, P12A's q/p Graphs
were *independent component probes*, and the historical P10 component
budget still had a ~0.18887ms/frame remaining reserve; they did NOT
prove K2 GPU acceleration.

P13 Hybrid AR/K2 was not selected: H200 one-token target AR costs
~0.1373ms, almost as much as packed 2-token verifier ~0.1423ms.
One historical seed's Hybrid also generated 181 rather than 142 codec
frames of the same ASR-recognized words with longer low-RMS intervals.
Do NOT extend Hybrid here.

## P14 core contract — ONE complete causal Graph round

Unlike P12's two separately captured q and p selectors, P14
captures *one* CUDA Graph on a **real committed K2 checkpoint**
for each of proposal depths2 and10:

\`\`\`
committed KV prefix + draft hidden + last committed embedding
    |
    v
real q head -> GPU q topk50 -> GPU proposal code
    |
    v
GPU projected proposal embedding (depends on q code)
    |
    v
real packed 2-token 5-layer Predictor verifier (writes true KV suffix)
    |
    v
real p head -> GPU p topk50 support
    |
    v
GPU min(1,p(x)/q(x)) accept or normalized positive (p-q)+ correction
    |
    +---- bonus head on verified proposed hidden, GPU bonus sample
    |
    v
GPU accepted flag, committed code, masked bonus, next depth/offset,
next hidden, projected next embedding and codec embedding sum
\`\`\`

- **Matched A/B:** Torch \`topk(50)\` / softmax / top-p + same
  GPU acceptance/correction and **P12A Triton \`tl.topk(...,64)\`**
  to export exact top50 sparse q/p distributions. Both traverse the
  same actual q head, proposal->embedding, full packed verifier, p
  head, and bonus logic. This is a FULL q→verifier→p causal *round*
  in ONE captured CUDA Graph; p is not available before verifier.
- **No host sync inside Graph.** Four explicit tuples of GPU-resident
  random uniforms control q proposal, accept, correction and bonus.
  All selected tokens, after-top-p p/q categorical distributions,
  hidden, logical next offset/depth, output sum and complete
  physical KV are compared against the matched Torch graph.
  A separate independent \`cpu_reference\` evaluates actual proposal,
  accept and positive residual from dense p/q at the SAME uniforms.
- **KV ownership:** before each graph replay, copy the real live KV
  slots0..15; after the probe, restore all active K/V exactly in a
  finally block. Original K2 request sampler RNG, codec outputs,
  semantic positions and output_embeds are NOT touched. Every P14
  field must be source-locked to the same PID/GPU/native Graph-ON
  baseline to count as valid evidence.
- **Bonus overhead:** this STATIC graph computes bonus head + code
  even for rejected drafts and masks the bonus from the returned
  next depth/state. This preserves the branch-conditional output
  state with **extra unnecessary rejected-round GPU work**. Graph
  cost is the actual unconditional-work candidate envelope, not the
  theoretical conditional-bonus minimum.
- **Private RNG:** a third Graph draws 4 \`torch.rand\` values from
  an independent CUDA Generator and proves five replays advance
  its state. This is NOT the real SGLang native seeded sampler RNG,
  distribution parity or actual request seed coupling.
- **Not a 14-depth loop:** next_cache_len / next_depth are returned
  GPU scalars but are NOT fed into a second dynamically-offset
  verifier; no rejection suffix revalidation/rollback across rounds,
  no conditional GPU WHILE. Therefore **no native GPU serving
  speedup or lossless 14-codebook completion is claimed**.

Even passing all P14 tests is only a conditional GO for researching
the NEXT device-controlled full-14-depth K2 loop. It is not a
production merge gate.

## 0. Refresh branch, validate CPU then H200 unit gates

STOP the old Omni server before starting new pinned measurements.

~~~bash
cd ~/sglang-omni

git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0

git status --short
git rev-parse HEAD

python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p14_causal_round.py \
  experiments/qwen3_tts_cp_spec_k2/p14_live_causal_round.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py

# No GPU needed; checks causal source contract / CPU acceptance.
python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  -m "not accelerator"

# Real H200 is required for CUDA graphs / Triton.
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  -m accelerator -s -x
~~~

The fake predictor H200 test is NOT an on-policy real-model proof:
it checks capture, data dependency, RNG, token/hidden/embedding
parity, physical KV restoration. The real model five-layer and
source/PID checks are performed below. If any unit CUDA/Triton
failure occurs, preserve the **first** traceback; do NOT relax the
1e-6 probability parity tolerance or skip the failed fixture.

## 1. New H200 native+K2 source-locked worker — Terminal A

No simultaneous P11/P12 diagnostics, Hybrid ARM or production traffic.

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_P10A_DIR
unset SGLANG_OMNI_QTTS_CP_P10B_DIR
unset SGLANG_OMNI_QTTS_CP_P11A_DIR
unset SGLANG_OMNI_QTTS_CP_P12A_DIR
unset SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR="$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/causal"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P14_CAUSAL_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/server.log"
~~~

## 2. Same-worker real HTTP natural stop + P14 Graph — Terminal B

Wait until the model \`/health\` is ready. Run once. A failed
CUDA Graph capture or p/q/KV strict parity creates a failure file
in the P14 causal dir; stop and inspect it, do not mask or retry
within the same CUDA-corrupted process.

~~~bash
cd ~/sglang-omni

python -m experiments.qwen3_tts_cp_spec_k2.p14_live_causal_round \
  --base-url http://localhost:8000 \
  --base-arm "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/ARM_BASE" \
  --spec-arm "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/ARM_SPEC" \
  --baseline-dir "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/baseline" \
  --probe-dir "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/causal" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/server.log" \
  --expect-source-sha "$(git rev-parse HEAD)" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/smoke"

cat "$HOME/results/qwen3tts-cpspec-k2-p14-causal-030/smoke/summary.md"
~~~

Requirements: four visits \`base/spec/spec/base\` must all typed
\`finish_reason=stop\`; native visits 0 K2 activity, spec visits
nonzero. The real optimized native full Graph-ON baseline and exactly
two real causal fixed-depth Graph probes (depth2/10) must share
SHA/PID/GPU. Each causal probe must contain 4/4 numerical oracle
cases, full KV parity and restoration, private RNG five unique
replay draws and four ABBA event blocks for reference/candidate/RNG.
An accepted or rejected branch may not be covered at one of the
selected real q/p states (e.g. depth2 low overlap); the actual branch
census is reported rather than falsely claiming both outcomes.

## 3. Interpret results without conflating evidence levels

P14 summary reports:
- **Torch full causal one-round GPU Graph ms** (q head, q proposal,
  proposal embedding, real verifier, p head, accept/residual, ALWAYS
  bonus head/draw, branch-masked embedding sum and next state).
- **Triton full causal one-round GPU Graph ms**: identical full chain
  with fast single-distribution q/p Top-k.
- **Triton + private RNG ms**: same chain with 4 independent Graph
  CUDA randoms.
- **Relative candidate/reference GPU event ratio** and actual
  accepted/rejected branch counts across deterministic uniform
  cases. All are based on the same on-policy checkpoint source.
- **Native full Predictor Graph ON ms per frame** as context, NOT
  as a same-grain baseline for a one-round Graph.

Decision:
1. If ANY p/q support, fixed-uniform proposal, accept, positive
   residual, code-sum, or entire K/V fails parity: **STOP**.
2. If candidate full causal Graph cost >= Torch reference, the
   P12A micro savings do not survive actual integration; do not
   build the 14-depth loop.
3. If candidate is faster but private RNG/always-bonus makes it
   too expensive at historical ~9.68611 rounds/frame, no full
   serving speed claim. The historical P2 rounds/frame are
   reconstructed from older source SHA and do not represent a
   new on-policy production measurement. The actual 14-codebook
   branch/bonus cost and initial pair remain unaccounted.
4. If parity and cost are favorable, NEXT stage must implement
   true device-controlled dynamic KV / cache ownership across
   accepted and rejected rounds, GPU conditional bonus skip,
   correct native seeded sampler RNG and all 14-depth
   on-policy trace correctness. Only THEN attempt fresh-process
   profiler-OFF Graph-ON E2E C1 and C8/C16 ABBA on frozen inputs,
   with audio/stop/WER and speaker-quality gates.
5. Keep PR #89 DRAFT. No default production behavior is changed
   by these post-health ARM experiment hooks.

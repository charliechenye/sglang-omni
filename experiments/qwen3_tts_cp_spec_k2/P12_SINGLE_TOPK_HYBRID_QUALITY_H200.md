# P12 H200 — causal single-distribution selector + Hybrid quality/compute gate

P11 actual real H200 evidence, source \`50dae7a...\`:
- Native optimized Graph ON Predictor ~2.23680ms/frame.
- Sparse sampler full ~0.07040ms at depth2, 0.06172ms at depth10.
- Two torch.topk50 calls, each on frozen p/q logits together, measured
  0.06103/0.05215ms graph envelope (the biggest observed component).
- P11B AR depth2/3 → K2 from depth4: **fewer speculative rounds, but
  total target calls/frame increased** (9.868→9.947, 9.542→9.895).
- Seed 20261121 hybrid WAV lasted 14.40s vs native 11.68s, a
  significant quality/stop-pattern flag, not a proven content bug.
- P10 component gate leaves just 0.18887ms/frame after verifier+sampler.

## What changes in P12 and what DOES NOT

### P12A — separate q-before and p-after selector graphs, not dual-topk

Actual real K2 code computes and samples q FIRST to construct the
proposal embedding. The packed target verifier then runs and computes
p. P11A can receive frozen q+p together only because it is a diagnostic.
P12A instruments the two **actual causal call sites** in
\`code_predictor_spec.py\`: \`maybe_observe_q\` just before proposal
sampling, and \`maybe_observe_p\` only after the verifier target head.

At each on-policy residual depth2 and depth10, it captures separate
CUDA Graphs for two alternative ONE-DISTRIBUTION selectors:

- exact-reference Torch float32/topk(50)/softmax/top-p plus one Triton
  sparse vocab-ID-order inverse-CDF draw, and
- one experimental Triton CTA doing a full 2048-score unsigned-packed
  descending sort, top-50, sparse probabilities, top-p and sampled ID.

Both sides use the SAME frozen on-policy q or p, same seed-independent
fixed uniform draws and matched worker. The selector does NOT replace
live K2 output, change model RNG, alter KV, or fuse across the verifier.
Real q/p source is checked for exact frame+depth identity. Selector
output must match reference selected **IDs**, dense normalized
probabilities (1e-6), and fixed-uniform codes.

**Important BF16 ties:** the custom Triton selector deliberately ties
by smaller token ID; Torch topk may choose another token ID for tied
BF16 scores. Such cases are reported as
\`P12A_TIE_OR_DISTRIBUTION_MISMATCH_NO_SELECTION\`, with CUDA Graph
timings preserved for diagnostics but **NOT eligible for a lossless
performance selection**. Do not weaken tie correctness or hide
mismatch behind a raw speed number. A H200 compile failure is likewise
not a negative speed measurement; preserve the first error.

Even on exact support PASS, P12A measures separate selector graphs,
not one combined q→real Predictor verifier→p→accept CUDA graph.
It cannot establish full K2 E2E speedup.

### P12B — actual one-token AR vs two-token verifier + old WAV audit

A separate probe on an actual C1 K2 committed KV prefix at depths4
and10 captures one-token target AR and two-token target verifier
CUDA Graphs with the SAME loaded model/backend/weights/KV. Each
compares hidden first token and complete K/V against independent eager
and captured references before timing four ABBA blocks. It restores
KV even when post-capture parity fails. No sampling, bonus or code
commits occur in the probe. Two types of forward launches do not
represent identical amounts of GPU work.

The standalone quality tool reads P11B WAV files already saved under
\`...p11b-020/smoke/\`. It verifies each SHA256, PCM format, duration,
silence fraction / trailing silence, and duplicate-visit hashes. Optional
\`faster-whisper\` ASR computes English WER and repeated phrases. ASR
is **NOT_MEASURED** unless an actual model is installed and explicitly
enabled. Speaker cosine is **NOT_MEASURED** unless user supplies
source-attributed speaker embedding vectors from an external model;
waveform correlation is not valid speaker similarity.

The tool can optionally combine P12B GPU target costs with old P11B
on-policy per-frame AR/packed call counts as a **cross-source projection
only**, never as the source-matched live serving speedup.

## 0. Pull exact source and run CPU then H200 gates — STOP old Omni

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git status --short
git rev-parse HEAD

# Syntax gate BEFORE PyTest / GPU server. Do not skip.
python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p12a_selector.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p12b_timing.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec.py \
  experiments/qwen3_tts_cp_spec_k2/p12_live_probe.py \
  experiments/qwen3_tts_cp_spec_k2/p12_quality_audit.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p12.py

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p12.py \
  -m "not accelerator"

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p12.py \
  -m accelerator -s -x
~~~

If the H200 gate fails, preserve the FIRST Triton compilation/numeric
traceback and do not start the live profiler. P12A's GPU test uses
unique fp32 logits; actual BF16 ties are evaluated by the live probe
and may veto selection even if the unique-fp32 unit test passes.

## 1. P12A fresh H200 worker, native Graph ON / K2 q/p selector

Terminal A:

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p12a-021"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_P8_DIR
unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_P10A_DIR
unset SGLANG_OMNI_QTTS_CP_P10B_DIR
unset SGLANG_OMNI_QTTS_CP_P11A_DIR
unset SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p12a-021/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p12a-021/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p12a-021/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P12A_DIR="$HOME/results/qwen3tts-cpspec-k2-p12a-021/selector"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_P12A_DIR"
test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p12a-021/server.log"
~~~

Terminal B, only after \`/health\`:

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p12_live_probe \
  --mode selector \
  --base-url http://localhost:8000 \
  --base-arm "$HOME/results/qwen3tts-cpspec-k2-p12a-021/ARM_BASE" \
  --spec-arm "$HOME/results/qwen3tts-cpspec-k2-p12a-021/ARM_SPEC" \
  --baseline-dir "$HOME/results/qwen3tts-cpspec-k2-p12a-021/baseline" \
  --evidence-dir "$HOME/results/qwen3tts-cpspec-k2-p12a-021/selector" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p12a-021/server.log" \
  --expect-sha "$(git rev-parse HEAD)" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12a-021/smoke"

cat "$HOME/results/qwen3tts-cpspec-k2-p12a-021/smoke/summary.md"
~~~

This report covers actual q-before and p-after events for depths2/10,
with exact static Torch reference, tie count, boundary tie, full Graph
events, status and full native Graph ON baseline. **NO_SELECTION is a
valid diagnostic outcome, not a speedup/accuracy PASS**. Do not select
a faster candidate whose tied token support differs from reference.

## 2. P12B GPU AR one-token vs packed two-token, NEW worker

STOP P12A server. Keep SAME source commit and GPU.

Terminal A:

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p12b-022"

unset SGLANG_OMNI_QTTS_CP_P12A_DIR
unset SGLANG_OMNI_QTTS_CP_P11A_DIR
unset SGLANG_OMNI_QTTS_CP_P10A_DIR
unset SGLANG_OMNI_QTTS_CP_P10B_DIR
unset SGLANG_OMNI_QTTS_CP_P9A_DIR
unset SGLANG_OMNI_QTTS_CP_P9B_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_ARM_FILE
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AUDIT
unset SGLANG_OMNI_QTTS_CP_K2_HYBRID_AR_THROUGH_DEPTH

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p12b-022/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P9_BASE_DIR="$HOME/results/qwen3tts-cpspec-k2-p12b-022/baseline"
export SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p12b-022/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR="$HOME/results/qwen3tts-cpspec-k2-p12b-022/ar_vs_pair"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_P12B_TIMING_DIR"
test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P9_BASE_ARM_FILE"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p12b-022/server.log"
~~~

Terminal B:

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p12_live_probe \
  --mode ar \
  --base-url http://localhost:8000 \
  --base-arm "$HOME/results/qwen3tts-cpspec-k2-p12b-022/ARM_BASE" \
  --spec-arm "$HOME/results/qwen3tts-cpspec-k2-p12b-022/ARM_SPEC" \
  --baseline-dir "$HOME/results/qwen3tts-cpspec-k2-p12b-022/baseline" \
  --evidence-dir "$HOME/results/qwen3tts-cpspec-k2-p12b-022/ar_vs_pair" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p12b-022/server.log" \
  --expect-sha "$(git rev-parse HEAD)" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12b-022/smoke"

cat "$HOME/results/qwen3tts-cpspec-k2-p12b-022/smoke/summary.md"
~~~

This provides matched CUDA-event costs from the existing real model
path and hidden/KV parity at depth4/10. Still not full Hybrid E2E.

## 3. Offline quality audit of the ORIGINAL P11B WAVs — NO GPU

Do NOT regenerate or overwrite original WAV evidence.

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p12_quality_audit \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --p12b-timing-dir "$HOME/results/qwen3tts-cpspec-k2-p12b-022/ar_vs_pair" \
  --asr none \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12-quality-023"

cat "$HOME/results/qwen3tts-cpspec-k2-p12-quality-023/summary.md"
~~~

For real ASR transcript WER/repeated phrases, an explicitly installed
faster-whisper model is necessary; use a different NEW output folder:

~~~bash
python -m experiments.qwen3_tts_cp_spec_k2.p12_quality_audit \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --p12b-timing-dir "$HOME/results/qwen3tts-cpspec-k2-p12b-022/ar_vs_pair" \
  --asr faster-whisper --asr-model small.en --asr-device cuda \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-024"

cat "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-024/summary.md"
~~~

The latter command requires \`faster-whisper\` installed in the current
venv and may require a previously downloaded model or connectivity.
If unavailable, report ASR **NOT_MEASURED**, not an invented WER.
The speaker comparison is also **NOT_MEASURED** without explicit
\`--speaker-embeddings\` JSON from a real external embedding model;
speaker similarity is not PCM waveform correlation.

The historical P11B WAV cohort is from an earlier source SHA than
P12B GPU cost probes. Any combined target-forward cost is explicitly
a CROSS-SOURCE projection, not a source-matched E2E proof.

## Decision and stop gates

- H200 selector Triton compile FAIL: preserve exact first traceback;
  kernel compilation is NOT a latency result.
- BF16 topk tied set/ID mismatch or fixed-uniform code mismatch:
  P12A **NO_SELECTION**, do not replace native/PyTorch topk.
- Exact selector PASS but full Graph cost >= Torch reference:
  no acceleration selection; do not integrate.
- P12B AR one-token/packed prefix parity FAIL: no hybrid cost
  conclusion or performance selection.
- P12B audio truncated/silence excess/WER or repeated phrase issues:
  Hybrid NO-GO, preserve original WAVs; do not infer root cause
  without inspecting actual transcript/audio.
- A positive component projection does NOT overcome the old P10
  full-GPU-K2 reserve NO-GO. Need accurate native seeded RNG,
  correct 14-depth GPU K2, quality, and profiler-OFF Graph ON
  complete HTTP ABBA before any speedup/merge claim.

PR #89 remains Draft. No server-side default inference behavior
changes without explicit env and post-health ARM.

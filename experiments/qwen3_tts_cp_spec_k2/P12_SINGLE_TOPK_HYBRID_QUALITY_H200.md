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
- one experimental Triton CTA doing a packed-score/ID **partial top-64**
  selection with Triton `tl.topk` (power-of-two requirement), exporting
  exactly top-50 probabilities, top-p and a sampled ID.

Both sides use the SAME frozen on-policy q or p, same seed-independent
fixed uniform draws and matched worker. The selector does NOT replace
live K2 output, change model RNG, alter KV, or fuse across the verifier.
Real q/p source is checked for exact frame+depth identity. Selector
output must match the exact **selected token support set**, dense
normalized probabilities (1e-6), and fixed-uniform codes. It also
reports Top-k rank-order equality separately: harmless reordering of
equal-score selected tokens is not itself a distribution mismatch.

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
events, status and full native Graph ON baseline. It also compares the
two INDEPENDENT q+p selector graph envelopes against a provisional
11.47us/round savings threshold from P10's 0.30ms/frame reserve gate.
This stage-sum estimate is **NOT a captured causal combined graph**:
per-stage envelopes need not add. **NO_SELECTION is a valid diagnostic
outcome, not a speedup/accuracy PASS**. Do not select a faster candidate
whose tied token support differs from reference.

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

## 3. Offline quality audit of ORIGINAL P11B WAVs — NO GPU REQUIRED

**Run this step whenever the P11B WAVs exist, including before the P12
H200 probes.** Preserve original WAV evidence. The GPU target-forward
cost is OPTIONAL and must not be invented just to analyze audio.

### 3A. WAV-only analysis NOW, without P12B timing

~~~bash
cd ~/sglang-omni
test -f "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json"

python -m experiments.qwen3_tts_cp_spec_k2.p12_quality_audit \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --asr none \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12-quality-023"

cat "$HOME/results/qwen3tts-cpspec-k2-p12-quality-023/summary.md"
~~~

Expected: verified WAV SHA/format; duration, silence, trailing silence
and repeatability by seed. ASR WER, speaker similarity, and GPU
target-forward projection are **NOT_MEASURED**. If -023 already exists,
choose a NEW output (e.g. -025); do not overwrite prior evidence.

### 3B. Optional GPU cost projection — ONLY after P12B H200 Step 2

First require real parity-passed depth4/depth10 JSON profiles under
`ar_vs_pair`. If the P12B -022 run has not happened, OMIT the
`--p12b-timing-dir` argument; do not create an empty timing directory.

~~~bash
cd ~/sglang-omni
test -d "$HOME/results/qwen3tts-cpspec-k2-p12b-022/ar_vs_pair"

python -m experiments.qwen3_tts_cp_spec_k2.p12_quality_audit \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --p12b-timing-dir "$HOME/results/qwen3tts-cpspec-k2-p12b-022/ar_vs_pair" \
  --asr none \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12-quality-with-gpu-026"

cat "$HOME/results/qwen3tts-cpspec-k2-p12-quality-with-gpu-026/summary.md"
~~~

When optional GPU data is missing or incomplete, the analyzer now
explains `P12B_TIMING_EVIDENCE_MISSING` or
`P12B_TIMING_EVIDENCE_INCOMPLETE`. It must not silently claim the
projection was measured.

### 3C. Optional real ASR transcript WER / repeated phrases

The first attempt at `--asr faster-whisper --asr-device cuda` failed
with `ModuleNotFoundError: No module named 'faster_whisper'`. That
is an **uninstalled OPTIONAL dependency**, not quality/ASR evidence.
The report is not written when ASR initialization raises.

**Safest default on the H200 model host: create a separate
home-scoped ASR virtualenv and run CPU INT8**, leaving Omni's active
PyTorch/CUDA environment untouched. Do NOT downgrade or reinstall
Torch, Triton, cuDNN or CTranslate2 in the model-serving venv.

~~~bash
cd ~/sglang-omni
ASR_VENV="$HOME/.venv-qwen3tts-asr"

# uv already available in the model experimentation environment.
# This is independent of the serving .venv.
uv venv "$ASR_VENV" --python 3.12
uv pip install --python "$ASR_VENV/bin/python" \
  numpy faster-whisper "av==18.1.0"

# This repair also works on an ALREADY created ASR-only venv.
# Old faster-whisper releases still use the av.open metadata_errors
# keyword removed in PyAV 19; av==18.1.0 accepts the old call.
"$ASR_VENV/bin/python" - <<'PY'
import av
import faster_whisper
print("PyAV:", av.__version__)
print("faster-whisper:", faster_whisper.__version__)
assert int(av.__version__.split(".")[0]) < 19, (
    "For legacy faster-whisper file decode, use av==18.1.0"
)
PY

"$ASR_VENV/bin/python" -c \
  "from faster_whisper import WhisperModel; import numpy; print('ASR_ENV_READY')"

# CPU INT8 avoids CTranslate2 CUDA/cuDNN runtime incompatibility.
# Use an output directory that does not yet exist.
"$ASR_VENV/bin/python" \
  -m experiments.qwen3_tts_cp_spec_k2.p12_quality_audit \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --asr faster-whisper \
  --asr-model small.en \
  --asr-device cpu \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-024"

cat "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-024/summary.md"
~~~

**Second independent workaround if you cannot change PyAV:** the
quality analyzer also accepts `--asr-input pcm-array` and reads the
already SHA-verified 24 kHz mono PCM16 WAV through Python `wave`, then
uses SciPy `resample_poly(audio, 2, 3)` to create a contiguous float32
16 kHz NumPy array. Faster-whisper accepts an ndarray without invoking
PyAV decoding. Install SciPy into this isolated ASR venv (not Omni's
serving venv) and use a NEW output directory:

~~~bash
uv pip install --python "$ASR_VENV/bin/python" scipy

"$ASR_VENV/bin/python" \
  -m experiments.qwen3_tts_cp_spec_k2.p12_quality_audit \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --asr faster-whisper --asr-device cpu --asr-model small.en \
  --asr-input pcm-array \
  --out "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-array-027"
~~~

**For the current error the quickest fix is just `av==18.1.0`**;
the PCM-array option avoids PyAV entirely if package compatibility
remains problematic. The two modes use different resampling backends;
do not mix ASR results across them as if they used identical preprocessing.

This uses the original P11B WAVs; no P12B GPU `--timing-dir` and no
server are required. `small.en` may require a model download from
Hugging Face. If outbound network is disabled, replace `small.en`
with a locally cached CTranslate2 Whisper model directory. Do not
treat network/model-download failure as quality failure.

To inspect ACTUAL transcripts and repeated phrases, not just WER:

~~~bash
"$ASR_VENV/bin/python" - <<'PY'
import json
from pathlib import Path
p = (Path.home() / "results/qwen3tts-cpspec-k2-p12-quality-asr-024/report.json")
data = json.loads(p.read_text())
for x in data["items"]:
    # Only first visit of each (seed,mode); the repeats are not
    # statistically independent.
    if x["tag"].split("_")[1] not in ("00", "01", "02"):
        continue
    print(f"\nseed={x['seed']} mode={x['mode']} wer={x.get('wer')}")
    print("transcript:", x.get("transcript", "NOT_MEASURED"))
    print("repeated_trigrams:", x.get("repeated_trigrams", [])[:5])
PY
~~~

ASR WER measures recognition against the fixed English prompt; it
does **not** measure speaker identity, and automatic ASR may be wrong
on long silences. Manual playback remains useful if Hybrid contains
unexpected words or repeated phrases. Speaker similarity is
`NOT_MEASURED` without real source-attributed embedding vectors via
`--speaker-embeddings`; PCM correlation is not identity.

The official faster-whisper README currently specifies CUDA 12
cuBLAS/cuDNN 9 for its CTranslate2 GPU backend. This runbook chooses
CPU INT8 intentionally to isolate the optional ASR from the active
H200 Omni/CUDA runtime.

P11B WAVs are from a prior source SHA; new P12B target event costs
applied to them are a CROSS-SOURCE heuristic only, NOT source-matched
on-policy serving performance.

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

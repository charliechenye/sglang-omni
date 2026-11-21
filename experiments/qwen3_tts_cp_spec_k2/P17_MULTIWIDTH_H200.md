# P17 H200 — NVIDIA-style multi-head drafting, ONE K2/K3/K4 wide verifier

**Research-only source-locked component screen; NOT a production CP SpecDec
implementation, 14-depth decoder, native seeded sampler, acceptance corpus,
or E2E performance result.**

NVIDIA's TensorRT Edge-LLM CP SpecDec does **not** require 14 depths inside
one GPU-conditional graph. It supports verify size 2–8: K-1 future codebook
LM heads reuse the SAME pre-verification hidden to generate drafts, then
one K-token packed verify computes all target p rows and a bonus row. The
accepted prefix is committed, plus exactly one corrected/bonus token.
NVIDIA's C++ loop can synchronize a pinned host accepted length per round.

Relevant NVIDIA source:
- https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/cpp/multimodal/qwen3_omni/codePredictor.cpp
- https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/cpp/kernels/speculative/cpSpecKernels.cu

Our P15/P16A K2+K2 TWO ROUND graph is a different optimization problem.
P16A showed conditional bonus was slower than always-compute in all four
measured branch cases. That does **not** rule out a K3/K4 one-wide verify.
P17 specifically tests the currently missing width advantage.

## P17 implementation and accuracy contract

For each genuine debug K2 on-policy checkpoint with *last committed
depth 1 and 9* (next proposal depths2 and10 respectively):
- Capture 2 separate, distinct codec-frame checkpoints at each depth
  (4 snapshots total) within one worker/source SHA.
- For each checkpoint and K=2/3/4, prepack true loaded
  \`lm_head[depth:depth+K]\` weights, apply a batched GPU \`torch.bmm\`
  to the same old hidden for K-1 distinct draft heads, GPU topk50 sample
  K-1 draft codes, and construct ONE K-row verifier input from the
  last committed embedding followed by ALL draft embeddings.
- Run exactly ONE real 5-layer **FusedPredictorLayers** verifier with
  device-side KV position/attention END, generalized from P10A to
  width2/3/4. A separate STATIC-END fused verifier is the independent
  parity reference. Fail if the current fused backend, shape or fused
  attention tile cannot safely handle K=4 — never silently use unfused
  CUDA/PyTorch or only verify two tokens.
- Batched target heads run on the K verified hidden rows. K-1 q/p pairs
  each get an exact GPU positive-residual accept/correction; commit the
  longest accepted draft prefix plus precisely one corrected code or
  bonus token. Physical accept/residual kernels are still **per draft**;
  this is NOT NVIDIA's optimized fused acceptance CUDA kernel.
- An INDEPENDENT eager reference computes native heads through
  \`module.forward\`, actual static fused wide verifier, native dense
  topk50/top-p probability and CPU accept/residual oracle.
- Every uniform case must match native q/p **probability distribution
  within 1e-6**, draft IDs, committed codes, accepted prefix,
  projected next embedding, next hidden, accumulated codec sum,
  logical next cache depth and **all physical KV slots**. No tolerance
  relaxation, fallback to K2 or hiding bad cases.

**The acceptance sample is deliberately small**:
16 deterministic explicit GPU-uniform tuples for each width at each of
4 checkpoints = 64 frozen-checkpoint draws **per width** (not 64
independent on-policy audio frames). The two checkpoints per depth
come from one natural-stop debug K2 visit. They are correlated; no global
acceptance-rate inference or full-depth speedup claim is allowed.

A CUDA event cost gate captures three UNMARKED graphs per width:
1. **Full round:** batched multihead proposals, one real wide verifier,
   K-1 GPU accept/residual draws, unconditional bonus head, GPU logical
   state and embedding accumulation.
2. **Proposal only:** prepacked batched q heads, topk50 GPU sample,
   projected draft embeddings.
3. **Verifier only:** actual same C1 fused K-width verifier from the
   first real uniform case's packed input (fixed frozen input).

All graph timings exclude CPU oracle/HTTP and alternate four fixed
uniform inputs (indices 0/5/10/15) for the full and proposal cells,
four timed replay blocks per input. Verifier-only stays on frozen case0.
**These are separately captured component envelopes. DO NOT SUM THEM.**
One width must never re-use another width's physical K/V suffix without
restoring the original prefix.

All K=2/3/4 comparisons share source SHA, H200 PID, native Graph-ON
baseline, hidden checkpoint, sampling (T=0.9, topk50, top-p1), and
the very same explicit fixed GPU uniforms. A full-round graph may have
more kernel overhead than NVIDIA's C++ implementation; report it, do
not claim it is NVIDIA performance.

## 0. Pull branch and run local unit gates

Use a new experiment worktree or clean old one. Do not overwrite P16A
results or use a running Nsight-profiler worker.

~~~bash
cd ~/sglang-omni
git fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-cpspec-p17-multiwidth
git switch -c experiment/qwen3-tts-cpspec-p17-multiwidth FETCH_HEAD
git status --short
git rev-parse HEAD

python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p17_wide.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p17_probe.py \
  experiments/qwen3_tts_cp_spec_k2/p17_multiwidth_gate.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p17.py

bash -n \
  experiments/qwen3_tts_cp_spec_k2/p17_launch.sh
bash -n \
  experiments/qwen3_tts_cp_spec_k2/p17_measure.sh

python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p17.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  -m "not accelerator"

CUDA_VISIBLE_DEVICES=0 python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  -m accelerator
~~~

If branch already exists, use \`git switch\` and \`git pull --ff-only\`.
Stop on first compile/test failure and send the first traceback. P17's
independent oracle can legitimately expose batched q/p LM-head
probability differences or a 4-row fused-attention tile deficiency.
Treat this as a **real missing optimization/equivalence problem**, not
permission to skip native-head parity.

## 1. Terminal A — fresh dedicated H200, profiler OFF

~~~bash
cd ~/sglang-omni
bash experiments/qwen3_tts_cp_spec_k2/p17_launch.sh \
  "$HOME/results/qwen3tts-cpspec-p17-multiwidth-001"
~~~

P17 K2 debug ARM starts absent; /health must be ready before Terminal B.
Never enable P14/P15/P16A diagnostics simultaneously. All probing
happens during the *existing live sampled K2* only, with an independent
temporary KV snapshot restored exactly before original K2 continues.

## 2. Terminal B — same-worker native/spec/spec/native ABBA

~~~bash
cd ~/sglang-omni
bash experiments/qwen3_tts_cp_spec_k2/p17_measure.sh \
  "$HOME/results/qwen3tts-cpspec-p17-multiwidth-001"
~~~

All 4 visits must finish naturally (\`finish_reason=stop\`). Native
visits must show 0 debug-K2 frames, while spec visits show real K2,
no hybrid direct AR. P17 captures exactly 4 reports:
\`ROOT/wide/wide_pid*_frame*_depth01.json\` (2 distinct frame IDs) and
\`ROOT/wide/wide_pid*_frame*_depth09.json\` (2 distinct frame IDs).
Any \`failure_pid*.txt\`, missing report, wrong source/PID/GPU,
numerical mismatch, or invalid cost/accept histogram makes it FAIL.

Successful *limited screen* status:

\`P17_K2_K3_K4_REAL_WIDE_CORRECTNESS_AND_COST_SCREEN_NOT_FULL_DECODER\`

Read:
\`ROOT/smoke/summary.md\`
and inspect
\`ROOT/smoke/report.json\` for every absolute cost, stage envelope,
accepted-prefix histogram, and independent oracle evidence.

## 3. Decision gate: do NOT confuse width advantage with E2E GO

Per checkpoint, the meaningful preliminary quantity is:

\`full_round_event_ms / mean_committed_codes_per_round\`

This denominator is empirical across 16 fixed uniform draws at the
SAME frozen checkpoint, NOT a production acceptance estimate.

Compare K2/K3/K4 at BOTH early (last committed depth1) and later
(last committed depth9) checkpoints. The native full Predictor
Graph-ON ~2.2ms/frame is ONLY context; \`native_ms / 14\` can be
printed as a rough orientation but is **not a matched cost baseline**
for this single wide round.

If K3/4 have no meaningful per-code advantage over K2 at these
real checkpoints, do not extend the full 14-depth speculation path.
If a clear K3/4 advantage survives both depths, the NEXT experiment
should collect an on-policy acceptance corpus across all residual
depths and independent prompts/seeds, then integrate optimized wider
accept/sampler kernels and native seeded RNG before performance ABBA.

**No production speedup claim or PR merge is authorized by P17.**
The accurate conclusion can be "K4 wide verifier works but the current
research sampler overhead dominates", which would still identify the
specific optimization gap relative to NVIDIA.

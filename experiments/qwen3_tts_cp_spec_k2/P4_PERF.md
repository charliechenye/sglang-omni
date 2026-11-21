# P4 — Does native Qwen3-TTS CodePredictor K2 have an acceleration budget?

**SGLang-Omni only.** No NVIDIA TensorRT, checkpoint export, model retraining
or extra fork/worktree. Reuses the real Qwen3-TTS model already loaded by the
Omni server. Every profiler is opt-in and ARM-gated.

## Motivation / non-claims

The prior on-policy 720-frame Qwen3-TTS K2 trace showed:

- 10,080 residual codes following the first code;
- 6,974 two-token K2 verify rounds, with 3,420 accepted proposals,
  3,554 rejected proposals and 3,106 bonus tokens;
- 1.445 post-first codes per verify call, or 30.8% fewer decoder calls;
- 49.04% observed acceptance (48.97% selected-state theoretical overlap).

This is **not** a measured speedup: each K2 verify call executes 2 tokens,
not one, and the current prototype serializes Python conditionals,
top-k/top-p, torch.multinomial and host sync. The existing debug HTTP K2
was around 6x slower than Graph-ON baseline on one sample. The four-prompt
P3 ABBA evidence has naturally stopped audio but different output durations
between arms, so its HTTP latencies cannot isolate compute cost.

We first need a fixed-state hardware cost screen: at the same live
checkpoint, with the same committed KV prefix, compare:

- serial_one_stack: 1-token Predictor forward;
- serial_two_stack: 2 successive 1-token forwards for same input codes;
- packed_two_stack: one true 2-token causal Predictor forward;
- draft_head: one extra existing future lm_head;
- draft_select_debug: the current Python/PyTorch head+distribution+sample;
- accept_debug: the current Python rejection/acceptance step.

P4 repeats the *same* selected on-policy tensor state without consuming
the live request's generator, and restores the model K/V state afterward.
It chooses a few residual depths to detect cost dependence on context size.

**Important:** CUDA events measure stream elapsed launch envelopes,
including any host-side gaps. They are **NOT** CUPTI GPU kernel busy
time, a true GPU-only lower bound, or proof of Graph-ON serving speedup.
The result is a go/no-go *screen for more expensive optimization*.
No HTTP QPS claim is allowed.

## 1. Update branch, run CPU and H200 probes

Stop the old Omni server before pulling or switching. Run from your
existing Omni checkout on the experimental branch.

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git status --short

python -m pytest -q tests/unit_test/qwen3_tts/test_cp_spec_p4_perf.py \
  -k 'not gpu_probe'

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p4_perf.py \
  -k gpu_probe -s
~~~

Stop on a failure. The H200 GPU test ensures that profiling fixed states
does not alter the live model KV cache.

## 2. Launch a fresh P4 server (terminal A)

Never reuse the previous P1/P2 diagnostic directory. The old P2 trace's
acceptance figures are used only as a frozen reference cohort.

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p4-005"

unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR
export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p4-005/ARM"
export SGLANG_OMNI_QTTS_CP_P4_DIR="$HOME/results/qwen3tts-cpspec-k2-p4-005/profiles"
export SGLANG_OMNI_QTTS_CP_P4_DEPTHS=2,6,10,14
export SGLANG_OMNI_QTTS_CP_P4_MAX_STATES=4
export SGLANG_OMNI_QTTS_CP_P4_REPEATS=12
export SGLANG_OMNI_QTTS_CP_P4_WARMUP=3
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p4-005/server.log"
~~~

**Do not create ARM before health.** This profiler only executes during
real speculative C1 inference once ARM is present. It profiles the first
on-policy occurrence of each requested residual depth, then stops.

## 3. Trigger the instrumented K2 path (terminal B)

~~~bash
cd ~/sglang-omni
curl -fsS http://localhost:8000/health

python experiments/qwen3_tts_cp_spec_k2/quality_smoke.py \
  --base-url http://localhost:8000 \
  --arm-file "$HOME/results/qwen3tts-cpspec-k2-p4-005/ARM" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p4-005/server.log" \
  --max-new-tokens 512 \
  --out "$HOME/results/qwen3tts-cpspec-k2-p4-smoke-005"

ls -lh "$HOME/results/qwen3tts-cpspec-k2-p4-005/profiles"
~~~

If any visit fails, inspect the server log; **do not treat partial
profiling cells as a completed experiment**. This request is not timed
for actual speedup because the microprobes deliberately add work.

## 4. Analyze the earlier on-policy K2 acceptance cohort

~~~bash
cd ~/sglang-omni
python -m experiments.qwen3_tts_cp_spec_k2.p4_cost_screen \
  --perf-dir "$HOME/results/qwen3tts-cpspec-k2-p4-005/profiles" \
  --acceptance-report "$HOME/results/qwen3tts-cpspec-k2-p2-report-004/report.json" \
  --min-probes 4 \
  --out "$HOME/results/qwen3tts-cpspec-k2-p4-report-005"

cat "$HOME/results/qwen3tts-cpspec-k2-p4-report-005/summary.md"
~~~

The cost screen maps unprofiled residual depths to the nearest measured
depth, and weights serial vs packed forwards using 720-frame proposal/bonus
counts. This interpolation is labeled explicitly: **four sampled states
cannot establish GPU performance for all depths, hardware backends,
concurrencies, or a new selective policy**.

The threshold is 10,080 / 6,974 = approximately 1.445 single-code
forwards per K2 round. If packed2 costs as much as, or more than, that
many serial1 forwards before accounting for draft/accept, we need either
a more efficient packed kernel or a different scheduling strategy.

If fixed-state packed2 looks promising, the next stage is to prototype
GPU-resident acceptance/state update and measure *real* Graph-ON baseline
vs candidate on same locked requests. The existing debug K2 will not
become fast by itself; it is intentionally synchronizing per round.

If fixed-state packed2 looks poor, inspect the CUDA kernel timeline
(Nsight Systems) to distinguish Python launch gaps from GPU work before
deciding the approach is a dead end.

## Explicit next decision

P4 may yield one of three evidence states:

- Forward-cost screen favorable: build a bounded, no-host-roundtrip K2
  prototype, then perform fresh-server Graph-ON ABBA.
- Forward-cost screen unfavorable: identify whether 2-row kernel or host
  launch overhead dominates; consider selective depths only after that.
- Inconclusive (missing probes, unstable/mixed sources, poor parity):
  collect clean evidence; do not make a speedup claim.

Even a favorable microbenchmark cannot establish full K2 speedup, because
reference fused sampler, CUDA Graph replay and production concurrency have
not been evaluated in this phase.

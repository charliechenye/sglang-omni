# P15 — Real two-round acceptance-dependent GPU causal chain (H200)

**Experiment source branch:** experiment/qwen3-tts-cpspec-p15-two-round

**P15 is NOT a complete 14-depth decoder, native RNG proof, an end-to-end
throughput optimization, or a merge-ready change.** It is a fail-closed research
step beyond P14. P14 H200 source 710dfafe1ce17a7868e2ccefe745f9787c9488ae
proved genuine q -> 2-row verifier -> p -> accept/correction -> masked bonus
inside ONE Graph. P14 does NOT feed its GPU next state into a second Predictor.

## P15 contract

On the *real* on-policy C1 sampled K2 depth-10 checkpoint:
- Round 1 uses head[9] and packed two-token P10A dynamic fused Predictor at
  GPU cache offset 10. Proposal embeds depend on a GPU sampled q token.
- Round 1 p/q sparse support and positive-residual acceptance are GPU-generated.
- An actual GPU scalar acceptance determines whether Round 2 uses head[10] or
  head[11]. Round 2 consumes Round 1 next_hidden, next_embedding,
  accumulated code sum, and on-device cache_len; P10A dynamically reads and
  writes the real physical K/V.
- The bonus head is inside a GPU cond_op true branch. Reject path contains
  zeros, not a bonus-head forward. This proves source-level conditional
  structure, but NOT that a particular CUDA kernel was skipped at replay.
  To claim physical kernel skipping: inspect a CUDA/Nsight graph replay trace.
- One CUDA Graph is captured for both rounds. It replays under 16 immutable
  pairs of deterministic uniforms (4 P14 fixed-u cases crossed with 4).
- Every case is compared with an independent HOST-DECIDED eager oracle that
  uses original static-offset Predictor and CPU top-50 dense accept/residual
  logic. Check round1+round2 code, probability distribution (1e-6),
  GPU logical offset, hidden, projected/codec sum, and whole 16-slot K/V
  at the existing P10A fixed-vs-dynamic tolerance.
- All physical K/V are restored to their exact pre-probe values. The live
  K2 request RNG, result_codes and output_embeds are not intentionally changed.

**IMPORTANT:** Four observed transitions AA, AR, RA, RR are required for a
P15 full-coverage GO. The 16 fixed inputs may not exercise all transitions;
missing coverage is an explicit NO GO, not a fake pass or a reason to loosen
acceptance. Rejection-only depth cases are legitimate observations.

## 0. Setup and unit gates

The worktree must have access to the H200 GPU and all normal Omni dependencies.
Ensure a fresh experimental worktree; do not overwrite your previously pinned
P14 evidence directory or start with old P14 environment variables.

~~~bash
cd ~/sglang-omni
git fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-cpspec-p15-two-round
git switch --create experiment/qwen3-tts-cpspec-p15-two-round FETCH_HEAD
git status --short
git rev-parse HEAD

python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p15_two_round.py \
  experiments/qwen3_tts_cp_spec_k2/p15_two_round_gate.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p15.py

python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p15.py \
  -m "not accelerator"

# On H200, also re-run the P14 accelerator tests (P15 must not regress P14):
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  -m accelerator
~~~

If local branch already exists, use git switch + git pull --ff-only rather
than git switch --create. Stop on a failed compile/unit test. No test skip,
numerical tolerance relax, or silent CPU fallback is permitted.

## 1. Terminal A — fresh source-locked dedicated H200 worker

Stop any existing Qwen3-TTS workers first, use no overlapping jobs. This
script requires an absent result directory. The only experimental
difference from your P14 worker is the P15 opt-in probe.

~~~bash
cd ~/sglang-omni

bash experiments/qwen3_tts_cp_spec_k2/p15_launch.sh \
  "$HOME/results/qwen3tts-cpspec-k2-p15-001"
~~~

Wait for the server health endpoint. The worker logs both source SHA and
native Graph-ON performance; the P15 probe does not run during warmup.

## 2. Terminal B — same worker baseline/spec/spec/baseline

~~~bash
cd ~/sglang-omni

bash experiments/qwen3_tts_cp_spec_k2/p15_measure.sh \
  "$HOME/results/qwen3tts-cpspec-k2-p15-001"
~~~

Produced:
- ROOT/source.sha: exact worker source SHA
- ROOT/server.log: all engine logs
- ROOT/baseline/baseline_graph_pid*.json: optimized native Graph-ON context
- ROOT/two_round/two_round_pid*_frame*_depth10.json: exactly one P15 graph
- ROOT/smoke/00_base.wav ... 03_base.wav: real natural-stop audio
- ROOT/smoke/report.json: source/PID/GPU + full cases
- ROOT/smoke/summary.md: branch coverage and limited CUDA graph timing

Required natural-stop: four visits (base/spec/spec/base), all finish_reason
stop; baseline has zero K2 frames, spec has K2 frames; no hybrid AR steps.
The initial native baseline must share PID/GPU/source SHA with P15 graph.

Expected possible status strings:
- P15_TWO_ROUND_GO_TO_SEEDED_RNG_AND_KERNEL_TRACE_NOT_14DEPTH_OR_E2E:
  all four transition classes observed, both rounds passed independent
  numerical and complete physical KV parity.
- P15_TWO_ROUND_COVERAGE_INCOMPLETE_NO_GO:
  parity passed for observed cases but AA/AR/RA/RR incomplete.
- P15_TWO_ROUND_SOURCE_LOCKED_VERIFICATION_FAIL:
  capture, correctness, source, PID/GPU, natural-stop or evidence gate failed.

## 3. Interpretation and next gate

The reported P15 event is a **two-round graph envelope**, not comparable
to the P14 single-round Graph or native full Predictor Graph timing. Both
a valid P15 capture and full four-transition coverage are necessary but
still insufficient for a production decision.

Before any speed claim or 14-depth integration:
1. Show via Nsight Systems/Compute which bonus-head kernels are physically
   absent on rejected graph replays. A returned bonus_executed==0 alone
   does not prove physical kernel skip.
2. Integrate actual request native seeded random draws and consumption
   semantics (not a fixed uniform tuple); compare code sequences for a
   sufficiently large on-policy corpus.
3. Extend to full residual depths with GPU termination, branch dependent
   head/codec offsets, and complete K/V ownership, and only then run
   matching-source serving ABBA QPS/p95/quality.

If any conditional CUDA capture fails or raises CUDA700, keep the FIRST
failure_pid*.txt traceback and RESTART the worker: do not retry within the
same potentially poisoned CUDA process. Preserve failure evidence and
report exact first failing layer/operation. Do not mask the error.

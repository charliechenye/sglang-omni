# P16A — H200 real two-round conditional-bonus GPU kernel trace + branch costs

**Research-only. Source-lock. No production integration.**

P15 H200 on source \`ef1e0946c3cfb4118cd824b4ebfe31266488f4dd\`
passed all 16 cross-product fixed-uniform cases, with AA=10, AR=2,
RA=3, RR=1. That showed real acceptance-dependent Round-2 GPU head,
hidden, embeddings and device KV, but did NOT show physical skipped LM
head kernels or native seeded RNG.

P15's reported 0.49676 ms was a last-uniform-case CUDA event envelope
(not branch-mix average). P16A corrects that attribution.

## Contract: four graphs, two independent evidence levels

The P16A on-policy depth-10 checkpoint reuses the original P15 16-case
CPU/eager reference, with full physical 16-slot K/V parity.

Four captured graphs:

1. **Conditional/unmarked**: original P15 (NO new instrumentation).
2. **Always-compute/unmarked**: matched control; computes bonus heads
   after accepted and rejected proposals, then masks rejected outputs.
   Same q/p, same dynamic fused Predictor and GPU Round-2 routing.
3. **Conditional/marked**: same model code, but inserts the distinct GPU
   kernels \`p16a_round_start_marker_kernel\` at each round start and
   \`p16a_bonus_head_marker_kernel\` immediately before the bonus head
   in the true branch.
4. **Always-compute/marked**: two bonus markers for every replay.

The 3 additional captured graphs MUST match the already validated P15
conditional graph for all 16 fixed-u cases, including code, distributions
(1e-6), hidden, codec sum, depth and entire physical K/V. In failure,
the first exception is recorded; a poisoned CUDA worker is discarded.

Only graphs (1) and (2) are used for **unmarked CUDA Event timing**.
At each transition AA/AR/RA/RR, use the first validated P15 fixed-u
case of that class; perform 4 alternating-order paired blocks per arm,
16 replays per block. Compare Unconditional minus Conditional. A
positive number means the conditional path is faster for THAT input.
No HTTP speedup is implied, and these are not 16-case weighted means.

Only marked graphs (3) and (4) are used for **Nsight node trace**.
A single bounded outer \`P16A_TRACE\` NVTX range contains exactly eight
inner \`P16A_CASE_<AA|AR|RA|RR>_<CONDITIONAL|UNCONDITIONAL>\` ranges.
Each inner range contains exactly one marked graph replay and synchronize.

Expected **physical GPU marker kernel** counts:

| Transition | Round-start, either arm | Bonus conditional | Bonus always-compute |
|---|---:|---:|---:|
| AA | 2 | 2 | 2 |
| AR | 2 | 1 | 2 |
| RA | 2 | 1 | 2 |
| RR | 2 | 0 | 2 |

Both marker controls must hold. But this alone does NOT establish that
a specific LM-head CUDA kernel was skipped. That requires manually
attributing the actual LM-head CUDA kernel in the Nsight node trace and
seeing the expected unconditional-vs-conditional **0/1/1/2** head-kernel
count difference (or an exact integer multiple thereof). The audit
never accepts an instrumentation marker as an LM-head signature.

## 0. Prepare clean branch

Use a separate worktree from your existing pinned P15 environment.
If the branch does not already exist locally:

~~~bash
cd ~/sglang-omni
git fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-cpspec-p16a-branch-trace
git switch -c experiment/qwen3-tts-cpspec-p16a-branch-trace FETCH_HEAD

git status --short
git rev-parse HEAD
python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p15_two_round.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p16a_markers.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p16a_profile.py \
  experiments/qwen3_tts_cp_spec_k2/p16a_gate.py \
  experiments/qwen3_tts_cp_spec_k2/p16a_trace_audit.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p16a.py

# CPU/fail-closed tests. These do not prove CUDA physical execution.
python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p15.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p16a.py \
  -m "not accelerator"

# P14 GPU regression once on H200 before starting production-like worker.
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p14.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p16a.py \
  -m accelerator
~~~

If the local branch exists, use \`git switch\` and \`git pull --ff-only\`,
not \`git switch -c\`. STOP on the first test failure. Keep the original
parity tolerances.

## 1. Terminal A — source-locked worker under bounded Nsight

Stop all other GPU-serving workers; H200 and nsys must be available.
Use a fresh output root (the scripts refuse to overwrite one).

~~~bash
cd ~/sglang-omni
bash experiments/qwen3_tts_cp_spec_k2/p16a_launch.sh \
  "$HOME/results/qwen3tts-cpspec-k2-p16a-001"
~~~

The launcher uses:

~~~bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --capture-range=nvtx --nvtx-capture='P16A_TRACE@*' \
  --capture-range-end=stop ...
~~~

\`NSYS_NVTX_PROFILER_REGISTER_ONLY=0\` is set for dynamic Python NVTX
capture strings. No root privileges, CPU sampling or context-switch
sampling are required. Graph-node trace support can vary with Nsight,
driver and CUDA versions: an unsupported run is a FAIL, not permission
to change to \`--cuda-graph-trace=graph\`. NVIDIA documents node vs
graph granularity and its runtime overhead in Nsight Systems User Guide:
https://docs.nvidia.com/nsight-systems/UserGuide/

The wrapper starts profiling at \`P16A_TRACE\`, after the 16-case parity
and unmarked CUDA-event measurements. At capture-range end profiling
stops, but the worker stays up so the HTTP request can natural-stop.

## 2. Terminal B — real HTTP ABBA and all four branch cost cells

When \`curl -f http://127.0.0.1:8000/health\` succeeds:

~~~bash
cd ~/sglang-omni
bash experiments/qwen3_tts_cp_spec_k2/p16a_measure.sh \
  "$HOME/results/qwen3tts-cpspec-k2-p16a-001"
~~~

This reuses the P15 source SHA/PID/GPU and native optimized Predictor
Graph-ON baseline validation and makes four visits:
base → spec → spec → base. All four must end naturally (\`finish_reason=stop\`).
Baseline must have zero K2 activity, spec must have real K2 activity.

The expected *provisional* success status is:

\`P16A_FOUR_BRANCH_PAIRED_EVENT_TIMING_PASS_NSYS_NODE_TRACE_PENDING\`

Outputs:
- \`ROOT/source.sha\`, \`ROOT/server.log\`
- \`ROOT/baseline/baseline_graph_pid*.json\`
- \`ROOT/two_round/two_round_pid*_depth10.json\` (real two-round P15 + P16A JSON)
- \`ROOT/smoke/p16a-summary.md\` with all 4 timing rows
- \`ROOT/smoke/report.json\` with raw blocks and 8 expected NVTX ranges

If any correctness, source or timing assertion fails, STOP and preserve
\`ROOT/two_round/failure_pid*.txt\` (first traceback). No retries in
the same CUDA process after a graph-capture failure.

## 3. Finalize the trace and audit physical GPU nodes

After Terminal B returns, **stop Terminal A** (Ctrl-C). The target
\`ROOT/nsys/p16a.nsys-rep\` must exist and be nonempty. Nsight may
require worker shutdown to finalize; do not run the trace audit before
the report exists.

~~~bash
cd ~/sglang-omni
bash experiments/qwen3_tts_cp_spec_k2/p16a_audit.sh \
  "$HOME/results/qwen3tts-cpspec-k2-p16a-001"
~~~

The audit runs the documented \`nvtx_kern_sum\` report against the
\`nsys-rep\`, parses physical graph-node CUDA kernels by NVTX range,
requires exact 2/0/1/2 bonus/round GPU marker count gates, and checks
the reported GPU worker PID matches the source-locked smoke worker.

Expected marker-only status:

\`P16A_NODE_TRACE_BRANCH_MARKERS_PASS_BONUS_HEAD_KERNEL_UNATTRIBUTED\`

A marker-only PASS is NOT the P16A physical bonus-head skip gate.

## 4. Only after identifying the true LM-head kernel

Look at:
\`ROOT/trace-audit-markers/trace-audit.json\` →
\`candidate_kernel_signatures_not_proof\` and the raw CSV in
\`ROOT/trace-audit-markers/nvtx_kern_sum.csv\`.
Manually verify the candidate really originates from the LM-head
projection, not from the proposal selector, codec gather or marker
itself. Candidate matching alone is never sufficient.

Then run a *new* audit with a regex for exactly that kernel family:

~~~bash
bash experiments/qwen3_tts_cp_spec_k2/p16a_audit.sh \
  "$HOME/results/qwen3tts-cpspec-k2-p16a-001" \
  'YOUR_MANUALLY_VERIFIED_LM_HEAD_KERNEL_REGEX'
~~~

If real LM-head kernel counts have delta 0/1/1/2 × a positive integer,
the final status may be:

\`P16A_NODE_TRACE_BONUS_HEAD_KERNEL_SKIP_PROVEN_NOT_E2E_OR_SEEDED_RNG\`

If Nsight cannot attribute the nested conditional graph nodes, keep
the trace evidence **INCOMPLETE**. Do not substitute CPU/NVTX-only
ranges, an output \`bonus_executed=0\`, a Graph-level summary or an
invented kernel regex.

## What P16A cannot prove

- Actual production request seed/position-derived Gumbel sampling parity
- Full 14-depth GPU loop with all dynamic KV/head transitions
- Quality/ASR/word timing equivalence
- Real serving HTTP QPS, p95 or end-to-end speedup
- C1/C8/C16 or mixed workload performance

**P16B native seeded RNG** should start only after the trace evidence
is understood, with an independent A/A noise floor and an untouched
source-locked baseline. Do not use the 0.49676ms P15 last-case timing
as a projected 14-depth speedup.

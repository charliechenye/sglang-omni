# P22 — Real Qwen3-TTS CodePredictor K2 profiling / P23 feasibility

**P21 real HTTP serving ABBA already established K2 performance NO GO.**
At P21 source SHA `9abd2ac14122fefaffbad6714dec53f4884e449c`,
native CodePredictor full CUDA Graph HTTP p50 was 0.744s and host-driven
K2 was 3.691s. The paired B/A median was 4.976 (latency +397.6%).
K2 did run real committed audio codebooks on the serving path; it
accepted 45-52% of proposals, but B needed about 9 rounds/codec frame.
B audio was shorter and did not have native seeded bitwise parity.
This is a **performance NO GO**, not a proof of audio quality equivalence.

P22 starts at that SAME proven P21 source and adds opt-in profiling-only
NVTX/cudaProfilerApi hooks. P22 is a **different SHA**; it must NEVER be
used as a paired unprofiled ABBA speed result. P21 is the preserved
unprofiled baseline. P18 and P19 remain disabled. The actual P21
CodePredictor integration, not a P19 counterfactual, is profiled.

## What P22 measures

Each actual integrated K2 codec frame is traced:

- `FULL_CODE_PREDICTOR`: host enclosure over C1 real codec generation.
- `INIT_CPU_PARAMS`, `INIT_PAIR_FORWARD`, `INIT_CODE1_SAMPLER`:
  the first native pair and first residual token.
- `ROUND`: every real K2 round, accepting one or two codes.
- `ROUND_UNIFORMS`: private random generation.
- `Q_PROPOSALS`: P17 multihead proposal, sampled draft, projections.
- `WIDE_VERIFIER`: actual fused five-layer two-row causal verifier.
- `TARGET_SUPPORT_ACCEPT_BONUS`: p target, accept/residual, bonus,
  gathering state/embeddings.
- `ROUND_CPU_ACCEPT_LENGTH`: `committed_count.item()` and Python
  dynamic-depth branch; captures GPU-to-host wait and CPU overhead.
- `ROUND_CPU_CODE_VALIDITY`: `(candidate_codes < 0).any().item()`;
  another mandatory host synchronization in this prototype.
- `ROUND_GPU_COMMIT`: actual committed codes, embedding, next state.
- `NATIVE_FINAL_AR`: terminal code15 when reached.

NVTX labels are CPU-side annotation ranges. `perf_counter_ns` for
these scopes measures **CPU enqueue + blocked wait**. They are NOT
CUDA kernel execution times, and several scopes are NESTED; summing
all stage times double-counts. Use the Nsight CUDA timeline or the
SQLite kernel/launch/NVTX join for GPU-side kernel work.

The worker writes each B request's stage counters and times in a
non-overwriteable `ROOT/p22_profiles/request_<hash>.json`. The client
validates that `FULL_CODE_PREDICTOR` is called exactly once per
real B codec frame and `ROUND`, `Q_PROPOSALS`, `WIDE_VERIFIER`,
both CPU waits, etc. exactly once per real K2 round. It also requires
natural speech EOS, valid WAV, native Graph replay on A, and real
K2 commits on B.

## 0 — Create separate P22 worktree (leave P21/P19 intact)

```bash
P21="$HOME/sglang-omni-worktrees/p21-serving-k2-abba"
P22="$HOME/sglang-omni-worktrees/p22-serving-profile"
git -C "$P21" fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-p22-serving-profile
git -C "$P21" worktree add -b \
  experiment/qwen3-tts-p22-serving-profile "$P22" FETCH_HEAD
cd "$P22"
source ~/.venv/bin/activate
git status --short
git rev-parse HEAD

python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p22_profile.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p22_nsys_sqlite.py \
  tests/unit_test/qwen3_tts/test_cp_spec_serving_p21.py \
  -m "not accelerator"
bash -n experiments/qwen3_tts_cp_spec_k2/p22_serving_profile_launch.sh
python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p22_profile.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_serving_p21.py \
  sglang_omni/models/qwen3_tts/code_predictor_spec_p17_wide.py \
  experiments/qwen3_tts_cp_spec_k2/p22_profile_once.py \
  experiments/qwen3_tts_cp_spec_k2/p22_analyze.py \
  experiments/qwen3_tts_cp_spec_k2/p22_nsys_sqlite.py \
  experiments/qwen3_tts_cp_spec_k2/p23_full_gpu_feasibility.py
```

STOP if any test/compile/preflight fails. `bash -n` is required.
Do not rebase onto latest main while comparing with P21 data.

## 1 — Terminal A: one Nsight Systems session, H200, profiler OFF until armed

Port 8000 must be free. Do NOT share the H200 or use the P21 worker.

```bash
cd ~/sglang-omni-worktrees/p22-serving-profile
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p22-real-serving-profile-001"
test ! -e "$ROOT"
P22_NSYS=1 bash \
  experiments/qwen3_tts_cp_spec_k2/p22_serving_profile_launch.sh "$ROOT"
```

Nsight Systems CLI must be present. The launcher uses:
`--capture-range=cudaProfilerApi --capture-range-end=stop
--trace=cuda,nvtx --target-processes=all --sample=none`.
On a no-root H200 machine disable CPU sampling and CPU context-switch
tracking; this run targets GPU kernels and host launch NVTX.

CUDA capture begins **only** after a post-warmup `P22_CAPTURE_ARM`
appears and the real B request enters its first K2 frame.
It stops when that request finishes and synchronizes the profiled
worker. Nsight recording must NOT start during model startup or JIT.
The underlying test server remains real TTS serving.

If Nsight cannot attach, a separate fresh root with `P22_NSYS=0`
can collect CPU-region evidence only. Do NOT classify that as GPU data.

## 2 — Terminal B: one native A warmup, B warmup, ONE measured B request

```bash
cd ~/sglang-omni-worktrees/p22-serving-profile
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p22-real-serving-profile-001"
curl -fsS http://127.0.0.1:8000/health

python -m experiments.qwen3_tts_cp_spec_k2.p22_profile_once \
  --root "$ROOT" --max-new-tokens 768

cat "$ROOT/p22_client.json"
find "$ROOT/p22_profiles" -maxdepth 1 -type f -name '*.json'
```

Client reports
`P22_ONE_REAL_SERVING_B_REQUEST_PROFILE_CAPTURED_NOT_SPEEDUP`
only if correct A Graph replay, real B K2, natural speech stop,
full per-stage counts and the CUDA profiler capture ARM all pass.

This is one instrumented request, not a throughput/latency A-B-B-A.
Never interpret its HTTP wall time as the speed of candidate K2.

## 3 — Stop profiler/server; export Nsight SQLite

After client finishes, stop Terminal A normally (Ctrl-C). Check
that the real `.nsys-rep` exists and is non-empty. An absent report
is incomplete GPU evidence and must NOT be invented.

```bash
ROOT="$HOME/results/qwen3tts-p22-real-serving-profile-001"
ls -lh "$ROOT/p22_measured_b.nsys-rep"
nsys stats --report cuda_gpu_kern_sum --report nvtx_sum \
  "$ROOT/p22_measured_b.nsys-rep" \
  | tee "$ROOT/p22_nsys_summary.txt"

nsys export --type sqlite \
  --output "$ROOT/p22_measured_b.sqlite" \
  "$ROOT/p22_measured_b.nsys-rep"
```

The optional programmatic SQLite analyzer joins:
`CUPTI_ACTIVITY_KIND_KERNEL.correlationId` to the CUDA
runtime/driver API, scoped by PID; then uses API **CPU launch time**
and thread ID to find the innermost enclosing
`NVTX_EVENTS` eventType=59 range. The GPU kernel time may occur
AFTER CPU NVTX returns, so naively matching GPU timestamps to CPU
NVTX ranges would falsely miss async kernels.

```bash
cd ~/sglang-omni-worktrees/p22-serving-profile
ROOT="$HOME/results/qwen3tts-p22-real-serving-profile-001"

python -m experiments.qwen3_tts_cp_spec_k2.p22_nsys_sqlite \
  --root "$ROOT" \
  --sqlite "$ROOT/p22_measured_b.sqlite" \
  --out "$ROOT/p22_nsys_kernel_work.json"

python -m experiments.qwen3_tts_cp_spec_k2.p22_analyze \
  --root "$ROOT" --out "$ROOT/p22-analysis"

cat "$ROOT/p22-analysis/summary.md"
```

The SQLite script reports **sum of GPU kernel execution durations**
for each launch-attributed NVTX group, with explicit unassigned API
correlation counts. This sum can exceed the GPU timeline elapsed
duration due to concurrent work. It is NOT a critical path or GPU
busy-time estimate; use Nsight UI for overlaps and graph nodes.

## 4 — P23 feasibility gate (NO full GPU code yet)

Once both P21 unprofiled ABBA and P22 actual B profile exist:

```bash
cd ~/sglang-omni-worktrees/p22-serving-profile
ROOT="$HOME/results/qwen3tts-p22-real-serving-profile-001"

python -m experiments.qwen3_tts_cp_spec_k2.p23_full_gpu_feasibility \
  --p21-report \
    "$HOME/results/qwen3tts-p21-real-serving-smoke-001/abba/report.json" \
  --p22-report "$ROOT/p22-analysis/report.json" \
  --p21-candidate-source \
    sglang_omni/models/qwen3_tts/code_predictor_spec_serving_p21.py \
  --out "$ROOT/p23-feasibility"

cat "$ROOT/p23-feasibility/summary.md"
```

P23 gate estimates the necessary *end-to-end* improvement using the
UNPROFILED P21 paired requests. For 5% win against native A, B needs
approximately 80-82% less current HTTP latency in the two existing
cases. It DOES NOT deduct instrumented CPU wait from old P21 E2E
wall time; that would be cross-SHA and profiler-contaminated.

P23 also audits the current host-driven source for:
`committed_count.item()`, second `.any().item()`, Python
`while depth < 15`, host dynamic LM head bank lookup, separate
terminal branch and private RNG. These must be redesigned for a
capture-safe GPU-driven loop. P15's two-round `torch.cond` was
NOT a full 14-depth implementation. Do not fake one by replaying
P17/P19 50 independent counterfactual graphs.

Before authoring full-graph K2 production code, require:
1. Valid 14-depth GPU-local causal transitions and dynamic head choice;
2. Physical K/V ownership and append-end parity for every accept path;
3. True p/q acceptance and native target distribution preservation;
4. Stable fixed-buffer Graph capture/replay with no CPU data dependence;
5. GPU/host critical-path budget that leaves room to beat native Graph;
6. Full E2E natural-stop, audio ASR/speaker check, and repeated ABBA.

**No P22 or P23 result is production GO.** A feature flag and one
natural-stop WAV do not establish the real-time, quality, memory,
concurrency, or safety contract.

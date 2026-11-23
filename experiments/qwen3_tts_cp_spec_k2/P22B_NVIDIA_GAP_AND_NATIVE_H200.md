# P22B — NVIDIA TensorRT Edge-LLM gap vs Omni Qwen3-TTS CP SpecDec

**Research-only. No NVIDIA runtime has been benchmarked on our H200.**
Do not claim "NVIDIA is N times faster" without running both implementations
on matched hardware, model, dtype, sampler, prompts and measurement scope.

## Verified NVIDIA code and design

As reviewed 2026-11-23:
- [NVIDIA CodePredictor.cpp](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/cpp/multimodal/qwen3_omni/codePredictor.cpp), blob
  `59310047ad7bc0a8b5631108e66e07a9940ab2be`
- [NVIDIA cpSpecKernels.cu](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/cpp/kernels/speculative/cpSpecKernels.cu), blob
  `f4550d31c51fdcd145560229e8cc66f9a2d77361`
- [NVIDIA CodePredictor SpecDec design](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/codepredictor-speculative-decoding.md), blob
  `abb63bf65c1cba47f790e8bbe03bfe8daf1623a4`

**NVIDIA's successful design is NOT a 14-depth GPU-only while-loop.**
Its `runSpecDecodeFrame` runs a C++ host while loop, copies accepted
codes/length to host, synchronizes stream once per round, and chooses the next
depth on CPU. It supports K=2..8, documented useful narrow widths K=3/4.
The C++ runtime has preallocated workspaces and a persistent uniform pool.
It captures CodePredictor TensorRT engine graphs for each verify seqLen=1..K
and batch bucket; the verify runs in the generation profile as context-chunk.
Different LM heads are selected with one grouped-head CUDA kernel.

Its `cpSpecKernels.cu` uses codebook-specialized topk/top-p, batched row
sampling, and a *single* probabilistic accept/residual/bonus kernel for
each batch row. The draft is from existing per-depth LM heads; no new model.

Our P21/P22 actual serving uses the same high-level draft+causal verify
algorithm, but only K2; a Python host loop issues many
Tensor/Triton kernels. The P17 fused wide verifier runs EAGER from Python.
One K2 round generates a proposed code and may commit the bonus;
P22 observed ~8.88 rounds per frame and 51.5% draft accept. No native
seeded sampler distribution parity has been certified for full E2E.

| Feature | NVIDIA | Omni P21/P22 | Correct next step |
|---|---|---|---|
| Controller | C++ host loop; one stream sync each round | Python loop, per-frame and per-round ops | Reduce Python dispatch; not necessarily GPU-only |
| Width | K=2..8, K3 in doc, saturation near K4 | K2 serving only, K3/K4 frozen probes | Evaluate *real* K3 serving only after control-path cost gate |
| Decode | TensorRT packed verify, pre-captured seqLen 1..K generation graphs | Fused H200 P17 wide eager launch, 28 kernels/round | Persistent graph per width or reduce calls; preserve KV |
| Head | Grouped linear multiple distinct heads, packed preallocated inputs | torch.bmm + Python gathering | Grouped head and GPU index selection |
| Sampler | Specialized radix/bitonic top-k, row sample, fused accept/residual/bonus | Triton top-k plus separate PyTorch index, cat, stack, scatter, gather | Fuse full target/correction/bonus and preallocate |
| RNG | Counter-seeded uniform pool, host/device staging owned | New generator/torch.rand per frame/round; different native stream | Match distributions and seed policy |
| Precision | FP16 engines; optional FP8/NVFP4 predictor | BF16 H200 native CP | Match numerics before crediting backend |
| Batch | TensorRT C++ batch-aware runtime | K2 C1-only | C8/C16 are subsequent gates |

P22 Nsight measured 100,076 assigned GPU kernels for 143 K2 frames.
Within its 1,270 rounds: 28 wide-verifier kernels, 36 target/accept/
bonus kernels, 5 proposal kernels per round. In the whole captured
request (including Talker) CUDA API summary had >100k launch calls,
25,627 cudaMemcpyAsync and 4,668 cudaStreamSynchronize calls.
**Those whole-trace API counts are not proof that all calls belong to CP.**
P22B isolates API calls using real P22 NVTX CPU launch boundaries.

NVIDIA's public docs explain the architecture and quality semantics.
They do NOT provide a same-model, same-H200, apples-to-apples improvement
number against Omni Graph ON. This project does not invent one.

## Phase A — new worktree, read-only analysis of EXISTING K2 Nsight trace

P22B is an extension of P22 and does NOT rewrite `p22-newhost-001`.
It does not need a running GPU to analyze the existing SQLite.

```bash
P22="$HOME/sglang-omni-worktrees/p22-serving-profile"
P22B="$HOME/sglang-omni-worktrees/p22b-nvidia-gap"

git -C "$P22" fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-p22b-nvidia-gap

# Only if P22B branch and worktree do not already exist:
git -C "$P22" worktree add -b \
  experiment/qwen3-tts-p22b-nvidia-gap "$P22B" FETCH_HEAD

cd "$P22B"
source ~/.venv/bin/activate
git status --short
git rev-parse HEAD

python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_p22b.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p22_profile.py \
  -m "not accelerator"
bash -n experiments/qwen3_tts_cp_spec_k2/p22b_native_profile_launch.sh

K2="$HOME/results/qwen3tts-p22-newhost-001"
test -s "$K2/p22_measured_b.sqlite"
test -s "$K2/p22_client.json"
python -m experiments.qwen3_tts_cp_spec_k2.p22b_cuda_api_nvtx \
  --root "$K2" \
  --sqlite "$K2/p22_measured_b.sqlite" \
  --out "$K2/p22b-cuda-api-001"

cat "$K2/p22b-cuda-api-001/summary.md"
```

P22B queries both CUDA runtime/driver tables, maps each API CPU launch
time to its same-thread innermost NVTX range and avoids double-counting
a driver wrapper of an existing runtime launch. It labels kernel-launch,
graph-launch, memcpy, memset and sync/wait groups by K2 stage and per
real frame. This is **CPU API time**, not device GPU kernel duration.

The client checks original B source SHA from its `run.json` and
`p22_client.json`, but its own analyzer version is newer; do not call
this another P22 execution.

## Phase B — fresh native A same-host one-request Nsight capture

Stop any old P22/K2 server. Port 8000 must be free. This is a
**separate native A worker at a new source SHA**, never a same-worker
HTTP ABBA. B's `P21_ARM` remains absent, and the request-level
counters prove native predictor Graph replay.

Terminal A:
```bash
cd ~/sglang-omni-worktrees/p22b-nvidia-gap
source ~/.venv/bin/activate
unset SGLANG_OMNI_QTTS_CP_P21_ARM_FILE
unset SGLANG_OMNI_QTTS_CP_P22B_NATIVE_ARM_FILE

A="$HOME/results/qwen3tts-p22b-native-a-newhost-001"
test ! -e "$A"
P22B_NSYS=1 bash \
  experiments/qwen3_tts_cp_spec_k2/p22b_native_profile_launch.sh "$A"
```

After health 200, Terminal B:
```bash
cd ~/sglang-omni-worktrees/p22b-nvidia-gap
source ~/.venv/bin/activate
A="$HOME/results/qwen3tts-p22b-native-a-newhost-001"

curl -fsS http://127.0.0.1:8000/health
python -m experiments.qwen3_tts_cp_spec_k2.p22b_native_profile_once \
  --root "$A" --max-new-tokens 768

cat "$A/p22b_native_client.json"
```

Client runs a native Graph A warmup request, then creates an ARM file
and captures exactly one native A request with the same text/seed
as the existing P22 B capture. Every measured frame must be a native
Predictor Graph replay, zero speculative K2 frames. Real HTTP audio
must finish naturally. The request-finish hook synchronizes and stops
the CUDA profiler. The output is a trace **not a throughput benchmark**.

After client completion, stop Terminal A normally; verify the
`p22b_native_a.nsys-rep` was generated, then:

```bash
cd ~/sglang-omni-worktrees/p22b-nvidia-gap
A="$HOME/results/qwen3tts-p22b-native-a-newhost-001"
B="$HOME/results/qwen3tts-p22-newhost-001"
test -s "$A/p22b_native_a.nsys-rep"

nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum \
  "$A/p22b_native_a.nsys-rep" > "$A/nsys_native_summary.txt"

nsys export --type sqlite \
  --output "$A/p22b_native_a.sqlite" \
  "$A/p22b_native_a.nsys-rep"

python -m experiments.qwen3_tts_cp_spec_k2.p22b_native_api_sqlite \
  --root "$A" --sqlite "$A/p22b_native_a.sqlite" \
  --out "$A/p22b_native_api.json"
```

**Do not** claim Native CodePredictor GPU kernel-time isolation from
these numbers: CUDA Graph child kernels may execute outside the CPU
NVTX timestamp range. Native report shows CPU native Graph-launch
API time per native frame plus all-captured-work context (Talker too).
The native Graph child nodes still need a validated graph-ID
correlation before CP-only GPU-time comparison with P22 B.

## Phase C — generate NVIDIA implementation gap report

```bash
cd ~/sglang-omni-worktrees/p22b-nvidia-gap
A="$HOME/results/qwen3tts-p22b-native-a-newhost-001"
B="$HOME/results/qwen3tts-p22-newhost-001"

python -m experiments.qwen3_tts_cp_spec_k2.p22b_nvidia_gap_report \
  --k2-api "$B/p22b-cuda-api-001/report.json" \
  --native-api "$A/p22b_native_api.json" \
  --p22-analysis "$B/p22-analysis/report.json" \
  --out "$A/nvidia-gap-001"

cat "$A/nvidia-gap-001/summary.md"
```

Optional: if the old P21 unprofiled ABBA report was copied onto
the new machine, add `--p21-abba /actual/path/abba/report.json`.
Do not invent missing output files.

## Go / No-Go decision after P22B

- If B has hundreds of host API launch calls per frame but native A has
  ~one Predictor Graph replay/frame, optimize launch/control and buffer
  ownership **before** GPU arithmetic micro-optimizations.
- NVIDIA does NOT use a full 14-depth GPU-conditional while-loop: do
  not build one as a prerequisite to parity with its runtime.
- The smallest realistic production optimization candidate is
  **persistent per-width verifier Graph + grouped LM heads + sampler
  fusion**, with strict KV/RNG/quality parity and actual Serving ABBA.
- K3 is the interesting follow-up width since it can reduce the number
  of rounds, but acceptance and cost must be measured on actual causal
  K3 trajectories, not extrapolated from K2 or frozen P17/P18 states.
- If even the best measured compute/launch budget cannot beat Native
  full-chain Graph, stop speculative decoding engineering.
- Do not merge P22B. It is instrumentation-only.

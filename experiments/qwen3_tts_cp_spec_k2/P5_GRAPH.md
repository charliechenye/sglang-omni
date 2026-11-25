# P5 — Existing full Predictor Graph ON vs native packed-two verifier graphs

Goal: resolve the P4 1.379x optimistic stack-only result against the
actual optimized SGLang-Omni Qwen3-TTS CUDA Graph-ON baseline.

**DRAFT / H200 NOT YET VERIFIED.** These are CUDA-event Graph replay costs,
NOT full K2 production speedup. Before P5, P4 measured ~0.78ms per
serial-one eager stack and ~0.82ms per packed-two eager stack, whereas a
full Graph-ON no-spec HTTP response is only ~0.6s for 137 codec frames.
That mismatch makes P4's extrapolated 1.379x untrustworthy for production.

P5 records two different types of costs in the SAME real running worker:

1. Existing Qwen3-TTS full CodePredictor Graph-ON replay:
   one complete 16-codebook Predictor pass per semantic frame, including
   the opening pair, sampling and embeddings. Uses existing graph and
   requires actual predictor graph replay (fallback to eager fails closed).
2. Isolated CUDA Graph captures/replays of real one- and two-token
   CodePredictor stack at 4 sampled on-policy KV prefixes (depths
   2,6,10,14). Uses static token embeddings and cache; restores the
   original checkpoint K/V afterward. Does not consume the K2 RNG.

**The P5 candidate cost remains a deliberately optimistic LOWER BOUND:**
it excludes the pair/first-code stage, draft head/sampler,
probabilistic acceptance, bonus sampling, KV commits and any host-side
control. It is NOT a full K2 graph and the two costs are not symmetric.
If isolated verifier cost alone exceeds the whole baseline graph cost,
independent per-round graph replay cannot win without substantial
whole-chain fusion. If cost is below, it only licenses the next GPU-only
implementation experiment.

## 1. Fetch P5 and run tests before serving

Stop the previous P4/P3 Omni server before switching code.

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git status --short

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p5_graph.py \
  -k 'not cuda_captured'

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p5_graph.py \
  -k cuda_captured -s
~~~

Stop if any test fails. The GPU test is a *real* CUDA Graph
capture/replay and K/V restoration check. Some CUDA kernels/backends
are not capture-safe in isolation; do NOT silently fall back to eager.

## 2. Fresh P5 server (terminal A)

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p5-006"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p5-006/ARM_SPEC"
export SGLANG_OMNI_QTTS_CP_P5_DIR="$HOME/results/qwen3tts-cpspec-k2-p5-006/graph_reports"
export SGLANG_OMNI_QTTS_CP_P5_BASE_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p5-006/ARM_BASE"
export SGLANG_OMNI_QTTS_CP_P5_GRAPH_DEPTHS=2,6,10,14
export SGLANG_OMNI_QTTS_CP_P5_REPEATS=64
export SGLANG_OMNI_QTTS_CP_P5_BASE_FRAMES=32
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P5_BASE_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P5_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p5-006/server.log"
~~~

Do not create either ARM file before /health succeeds; this avoids
capturing startup JIT/graph warmup. The actual baseline ARM and K2
ARM markers are different and never coexist.

## 3. Run the P5 cost screen (terminal B)

~~~bash
cd ~/sglang-omni
curl -fsS http://localhost:8000/health

python -m experiments.qwen3_tts_cp_spec_k2.p5_graph_smoke \
  --base-url http://localhost:8000 \
  --baseline-arm-file "$HOME/results/qwen3tts-cpspec-k2-p5-006/ARM_BASE" \
  --spec-arm-file "$HOME/results/qwen3tts-cpspec-k2-p5-006/ARM_SPEC" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p5-006/server.log" \
  --profiles-dir "$HOME/results/qwen3tts-cpspec-k2-p5-006/graph_reports" \
  --out "$HOME/results/qwen3tts-cpspec-k2-p5-smoke-006"

python -m experiments.qwen3_tts_cp_spec_k2.p5_graph_screen \
  --perf-dir "$HOME/results/qwen3tts-cpspec-k2-p5-006/graph_reports" \
  --acceptance-report "$HOME/results/qwen3tts-cpspec-k2-p2-report-004/report.json" \
  --min-probes 4 \
  --out "$HOME/results/qwen3tts-cpspec-k2-p5-report-006"

cat "$HOME/results/qwen3tts-cpspec-k2-p5-report-006/summary.md"
~~~

Expect one baseline_graph_pid....json plus 4 captured
packed_graph_pid..._depth{02,06,10,14}.json. All are from the same worker
SHA and GPU. The previous 720-frame P2 cohort is source-sha mismatched,
so weighted proposal counts are a historical approximation, not a
new matched P5 on-policy acceptance census.

If capture produces a runtime exception, preserve server.log and the
failing GPU unit test; do not treat this as negative K2 performance.

## 4. Decision gates

P5 compares an **actual full Predictor Graph ON baseline event duration**
against a **minimum K2 cost consisting solely of isolated 2-row
verifier graph replay events** weighted by past proposal counts.
This is intentionally optimistic toward K2. A value above 1 is
not an E2E acceleration claim.

- If the optimistic K2 lower bound is already >= full Graph-ON AR cost,
  do not implement naive per-round graph replays. A more advanced whole
  chain/single-graph design is needed before seeking production speedup.
- If lower bound is comfortably below the Graph-ON baseline, a
  GPU-resident sampler/accept/control kernel may be worth prototyping.
  It must then pass correctness, normal quality, and real fresh-process
  Graph-ON ABBA with the same workload and observed output length.
- If kernel capture is unsupported, the next measurement is a tightly
  scoped Nsight Systems timeline, not more eager CUDA-event estimates.

All P5 results are a one-GPU C1 screen. No C8/C16, production concurrency,
quality, TTFA or throughput gain can be claimed from this experiment.

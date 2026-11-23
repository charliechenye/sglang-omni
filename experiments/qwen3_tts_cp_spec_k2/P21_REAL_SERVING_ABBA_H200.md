# P21 — Qwen3-TTS real serving K2 C1 A-B-B-A on H200

**State: implementation committed; CPU/H200 tests NOT yet run. No speedup
has been demonstrated. Nothing is approved for production merge.**

## What changed vs P19

P19 was a read-only native EAGER checkpoint observer: capture and replay
50 independently created temporary graphs, restore K/V/RNG, let native AR
serve. Its temporary graph lifetime triggered CUDA700 upon the next main
Talker forward. P19 Graph cost and its failure are NOT a serving speedup.

P21 is a NEW isolated branch
`experiment/qwen3-tts-serving-k2-abba-p21`. For this first
experimental integration it starts from the known P19 research source
to reuse validated P17 fused-wide kernels. It is **NOT a clean
main-based/upstream-ready PR**. All P18/P19 probes and their arm files
are explicitly disabled by the P21 launcher.

At a real `Qwen3TTSTalker.code_predictor_forward` call:

- A (no P21 ARM file): original native CodePredictor graph/eager dispatch,
  unchanged. The server must log actual native Predictor Graph replays.
- B (P21 ARM exists): actually writes K2 committed codes and codec
  embeddings returned by the live TTS HTTP request. Reuses P17's real
  fused 2-token causal wide verifier and q/p residual sampler at each
  continuation depth. On rejection commits exactly one target correction;
  on acceptance commits one draft and one target bonus.
- Codebook 0/1 and the last unspeculatable depth use actual native
  CodePredictor operations, using their genuine native sampler.
- The main Talker stays on its normal SGLang serving forward.
- The B CodePredictor is an experimental HOST-DRIVEN K2 loop, with a
  private seeded proposal stream and a host accepted-length synchronization
  each round. **Not full-chain CUDA Graph, not C8/C16, not native seeded
  bitwise parity, not deployment quality equivalence.**

An unsupported batch, sampler setting, CUDA Graph capture state,
missing H200 fused backend, unsupported top-k, or stale P18/P19
diagnostic environment must fail CLOSED. No silent K2-to-AR fallback.

This is a useful direct E2E feasibility screen, not a finished
production acceleration. Native Graph ON was roughly 2.2ms/frame in
historical research; K2 may be slower, especially with host sync.
Report any regression.

## Critical acceptance gates

1. GPU correctness: real H200 C1 16 codec groups, per-frame 15 residual
   codes committed, valid dimensions, correct K2 accept/reject counts.
   P17's frozen-checkpoint parity is supporting evidence but does NOT
   prove this integrated 14-depth causal rollout.
2. **A really replays native Predictor CUDA Graph; B really runs K2.**
   Per-request worker log `P21_REAL_SERVING_REQUEST ...` records
   candidate frames, K2 rounds, accepted/rejected, native graph replays.
   A with zero graph replays is an invalid baseline.
3. All sampled HTTP responses terminate naturally
   `finish_reason=stop`, valid 24kHz mono PCM16 WAV.
4. Same seed + prompt within each A and each B must produce repeatable
   WAV SHA. A vs B hashes are NOT expected to match, because B's RNG is
   explicitly different.
5. The supplied ABBA driver computes E2E full HTTP wall latency, audio
   RTF and prompt/seed paired B/A ratio. It does NOT compute CUDA
   graph-event cost, streaming TTFC, concurrent C8/C16 QPS, ASR WER/CER,
   speaker similarity or MOS.
6. Quality inspection with an independent ASR and listening tests is
   REQUIRED before calling the candidate quality-safe or merge-ready,
   even if the smoke is fast. This script does not invent a quality score.
7. Do NOT merge unless correctness and reproducible performance beat
   the native Graph baseline. After that, port ONLY essential serving
   changes onto a clean current-main branch without P17/P19 harness.

## New worktree, do NOT modify P19

```bash
P19="$HOME/sglang-omni-worktrees/p19-matched-dp"
P21="$HOME/sglang-omni-worktrees/p21-serving-k2-abba"
git -C "$P19" fetch https://github.com/charliechenye/sglang-omni.git \
  experiment/qwen3-tts-serving-k2-abba-p21
git -C "$P19" worktree add -b \
  experiment/qwen3-tts-serving-k2-abba-p21 "$P21" FETCH_HEAD

cd "$P21"
source ~/.venv/bin/activate
git status --short
git rev-parse HEAD

python -m pytest -q -x \
  tests/unit_test/qwen3_tts/test_cp_spec_serving_p21.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p17.py \
  -m "not accelerator"
bash -n experiments/qwen3_tts_cp_spec_k2/p21_serving_launch.sh
python -m py_compile \
  sglang_omni/models/qwen3_tts/code_predictor_spec_serving_p21.py \
  experiments/qwen3_tts_cp_spec_k2/p21_serving_abba.py
```

If the branch already exists as a local worktree, DO NOT use `worktree
add -b` again. Switch to that existing worktree and update by FF-only.
Do not reset/rebase an active P19 worktree. Run a dedicated H200 with
port 8000 free; CUDA errors require a NEW worker.

## Terminal A: fresh P21 server, ARM absent at startup

```bash
cd ~/sglang-omni-worktrees/p21-serving-k2-abba
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p21-real-serving-smoke-001"
test ! -e "$ROOT"
bash experiments/qwen3_tts_cp_spec_k2/p21_serving_launch.sh "$ROOT"
```

Expect `P21_SERVING_SOURCE_SHA`, an import preflight from the P21
worktree, and healthy `http://127.0.0.1:8000/health`. The standalone
P21_ARM marker remains ABSENT until the client changes it.

## Terminal B: C1 real production-request A-B-B-A smoke

```bash
cd ~/sglang-omni-worktrees/p21-serving-k2-abba
source ~/.venv/bin/activate
ROOT="$HOME/results/qwen3tts-p21-real-serving-smoke-001"

curl -fsS http://127.0.0.1:8000/health

python -m experiments.qwen3_tts_cp_spec_k2.p21_serving_abba \
  --root "$ROOT" --mode smoke --max-new-tokens 768

cat "$ROOT/abba/report.json"
grep 'P21_REAL_SERVING_REQUEST' "$ROOT/server.log" | tail -n 15
```

The smoke has 2 distinct prompts, one configured seed per prompt, and
A-B-B-A with a separate untimed real-request warmup for EACH visit.
Any candidate failure is a fail-closed error and writes a partial
`abba/report.json`. NO velocity/quality result may be selected.

Do NOT automatically run the main benchmark if smoke fails.

## Main C1 ABBA — new worker and new root

Once smoke reaches a full natural stop and the hardware backend works,
STOP the smoke server and start a fresh Terminal A with:

```bash
ROOT="$HOME/results/qwen3tts-p21-real-serving-main-001"
bash experiments/qwen3_tts_cp_spec_k2/p21_serving_launch.sh "$ROOT"
```

Terminal B after fresh health:

```bash
ROOT="$HOME/results/qwen3tts-p21-real-serving-main-001"
python -m experiments.qwen3_tts_cp_spec_k2.p21_serving_abba \
  --root "$ROOT" --mode main --max-new-tokens 768
cat "$ROOT/abba/report.json"
```

Main includes eight distinct prompts in English/Mandarin and the A-B-B-A
order, yielding 32 timed C1 requests + 4 untimed warmups.
It reports paired median B/A latency and pooled HTTP p50/p95.
Only one ABBA block and eight prompts, so no robust production CI.

There is no need to run the P19 lifecycle legs; they only diagnose
temporary Graph ownership and are not comparable to this E2E experiment.

## Decision and stop conditions

- If B cannot complete all natural-stop audios, the codec backend is
  unsupported, or the source/Graph/K2 activity gate fails: **NO GO**.
  Share the first traceback, not aggregate latency.
- If B is >5% slower or roughly the same as A: **performance NO GO** for
  this integrated host-driven K2 candidate. Do not keep optimizing the
  P19 harness to claim speedup.
- If B looks >5% faster, it is **only preliminary C1 HTTP latency
  screening**. Follow with a repeat fresh-worker experiment, full
  causal correctness/sampler-distribution parity, independent ASR
  WER/CER and speaker/quality checks.
- To support production deployment: native-seeded sampler parity or
  explicitly justified distribution preservation, C8/C16 support,
  all-depth GPU execution, memory/Graph safety, startup/memory,
  streaming TTFC, concurrency and quality regression testing.
- Only after every gate should a clean-main, code-only PR be proposed.

**A verified server request is not equivalent to production-ready code.**

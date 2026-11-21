# P8 — One CUDA Graph with device WHILE + real-model IF (H200)

P7 real-model IF conditional graph PASSED at true KV offset 2, false KV
offset 6, then true 2 again, with exact hidden/KV parity, immutable input,
no inactive-branch cache modification on the actual 5-layer checkpoint.
Conditional graph replay envelopes: 0.1570 / 0.1593 ms.

P6.1 device WHILE primitive also PASSED on torch 2.14.1+cu130/H200.

**P8 is the first NESTED gate:** can an on-device WHILE body invoke a
2-way IF that executes the actual loaded Predictor stack at static
KV offsets 2 or 6, and does this preserve both GPU loop counters and
the complete model K/V state?

This probe is intentionally *not* the full train-free K2 algorithm:
it uses real on-policy two-token embedding from a residual depth-6
proposal but replays those same frozen embeddings for 1 or 2 model
calls, selecting offset 2/6 by GPU branch selectors. It neither
samples real draft codes on GPU nor changes the residual codebook
depth according to accepted tokens. Running the embedding at offset
2 is OFF-policy (as in P7). Graph performance is NOT serving speedup.

The single captured CUDA Graph uses:
- a dynamic CUDA WHILE node with a GPU tensor loop bound 1 or 2;
- a nested CUDA IF node selecting actual Predictor fixed-offset forward;
- both branches are independently input-cloned (BF16 in-place safe);
- accumulated hidden output and all first 8 K/V slots compared to a
  separately eager-executed serial reference, with identical starting
  K/V cache for each mask;
- six cases, switching 1- and 2-round sequences and branch directions
  across repeated replays; original live K/V restored in a finally block.

For real serving, it runs once when residual proposal depth 6 is reached,
using exactly the same already-loaded 5-layer CodePredictor weights and
KV workspace, then resumes the original K2 path.

**This is an isolation and feasibility gate only, NOT a QPS/perf test.**

## 1. Refresh experimental branch and run CPU/GPU tests

STOP the old serving process first. P8 uses private PyTorch 2.14
CUDA conditional graph dispatch; if capture fails, use a NEW process
rather than continuing in a potentially poisoned CUDA context.

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git status --short

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p8_nested.py \
  -k 'not h200'

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p8_nested.py \
  -k h200 -s -x
~~~

The GPU unit fixture has two lightweight Predictor layers; the
real serving path has five. A passing fixture is necessary but
not sufficient.

## 2. Start fresh P8 server (terminal A)

Use a completely fresh -011 results directory. Never share P7
graph capture pools or old ARM markers with P8.

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p8-011"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_P7_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p8-011/ARM"
export SGLANG_OMNI_QTTS_CP_P8_DIR="$HOME/results/qwen3tts-cpspec-k2-p8-011/nested"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P8_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p8-011/server.log"
~~~

Do not enable P1/P4/P5/P7 with P8, and do not create ARM before /health.

## 3. Trigger C1 sampled real target and check six nested cases (terminal B)

~~~bash
cd ~/sglang-omni

python experiments/qwen3_tts_cp_spec_k2/quality_smoke.py \
  --base-url http://localhost:8000 \
  --arm-file "$HOME/results/qwen3tts-cpspec-k2-p8-011/ARM" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p8-011/server.log" \
  --max-new-tokens 512 \
  --out "$HOME/results/qwen3tts-cpspec-k2-p8-smoke-011"

python - <<'PY'
import json
from pathlib import Path
root = Path.home() / "results/qwen3tts-cpspec-k2-p8-011/nested"
files = list(root.glob("nested_pid*_frame*.json"))
if len(files) != 1:
    raise SystemExit(f"P8 INCOMPLETE: expected 1 report, found {len(files)}")
r = json.loads(files[0].read_text())
print("STATUS:", r["status"])
print("SOURCE SHA:", r["source_sha"])
print("REAL LAYERS:", r["predictor_layers"])
for row in r["cases"]:
    print(row["label"], row["actual_offsets"], "PASS", row["pass"],
          "failed", {k:v["max_abs"] for k,v in row["checks"].items()
                     if not v["close"]})
print("GRAPH EVENT MS:", r.get("cuda_graph_event_ms_per_case"))
if not r["status"].startswith("P8_NESTED_WHILE_IF_REAL_PREDICTOR_MODEL_PASS"):
    raise SystemExit("P8 REAL MODEL NOT PASSED")
if r["predictor_layers"] != 5:
    raise SystemExit("P8 EXPECTED FIVE REAL CHECKPOINT PREDICTOR LAYERS")
if len(r["cases"]) != 6 or not all(x["pass"] for x in r["cases"]):
    raise SystemExit("P8 CASE COUNT / PARITY NOT PASSED")
PY
~~~

If GPU test or HTTP fails, preserve the first exception. During serving
the P8 hook writes capture_fail_pid....txt in the nested directory.
Do not retry a failed conditional capture in the same process.

## 4. Decision after P8

- If FAIL: inspect failure type before optimizing. A nested conditional
  may expose BF16 scratch aliasing, graph output ownership, or CUDA
  memory-pool lifetime behavior not covered by separate P6/P7 tests.
  Fix with strict hidden+whole-KV parity; never relax tolerance.
- If PASS: device WHILE and real conditional model branches compose on
  actual checkpoint/H200. Only then implement a true on-device
  K2 accumulator: 14 depth indices, dynamic accepted/bonus transition,
  exact p/q sampling and residual correction, reusable fixed-offset model
  graphs or a safe equivalent, and device KV validity/fencing.
- Bench FINAL candidate vs original full Graph-ON Predictor baseline with
  profiler OFF, matched prompts and observed frame counts, C1 then C8/C16.
  Do not label P8 event times an E2E speedup.

Critical: the 2.2266 ms/frame P5 baseline is already a very optimized
whole-Predictor graph. Adding conditional graph nodes can make P8
much slower than separate P7 branches; this gate measures that risk.

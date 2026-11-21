# P7 — Real Qwen3-TTS Predictor CUDA conditional two-depth dispatch

P6.1 PASS on H200 / torch 2.14.1+cu130 proved a GPU-controlled CUDA
WHILE graph with 7–14 residual depth iterations. Its toy depth-control
cost was ~0.165–0.316 ms/graph. That is not an actual model.

P7 is the first HARD model-integration gate: can an explicit PyTorch
CUDA IF conditional node select the **actual loaded Qwen3-TTS Predictor**
with fixed KV offsets 2 vs 6 and preserve output/KV parity?

PyTorch 2.14.1 explicitly does NOT support torch.switch with CUDA Graph
(Inductor lowers it with disable_cudagraphs_reason). Therefore start with
a two-way cond_op under CUDAGraphCaptureControlFlowOpDispatchMode.

This experimental P7 path uses the actual existing 5-layer model loaded
by Omni; not fake weights, not TensorRT, and no extra checkpoint. At the
first on-policy K2 proposal for residual depth 6, it:

1. Freezes the real model's K/V prefix and packed verifier input.
2. Captures a single conditional CUDA Graph with TRUE branch using
   cache_len=2 and FALSE branch cache_len=6; both branches use the
   same actual Predictor stack, but different static KV offsets.
3. Changes a GPU Boolean tensor between replays; runs TRUE, FALSE,
   TRUE and compares each result with separate normal Predictor calls
   starting from the SAME KV prefix. Compares both hidden and K/V,
   including non-selected suffix, and restores original K/V.
4. Records C1 Graph replay launch envelopes for each GPU branch.
5. Resumes the original K2 sample/accept path afterward with no
   extra token draws.

**Important:** the offset=2 branch uses the input taken at depth 6,
and thus is NOT an on-policy sample at depth 2. This P7 is only a
conditional graph execution and K/V ownership gate; do not infer
acceptance or distribution from its output. It is NOT the full 14-depth
dispatch, not a WHILE+real model graph, not GPU sampler/bonus control,
and not a speedup claim.

## 1. Update branch and run CPU + CUDA checks

Stop any old server before pulling, and do not run multiple capture
experiments in one Python process.

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p7_dispatch.py \
  -k 'not h200'

CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p7_dispatch.py \
  -k h200 -s
~~~

If H200 CUDA Graph capture fails, preserve the complete first traceback;
do NOT continue to HTTP. Because PyTorch CUDA conditional nodes can have
allocator lifetime issues, use a fresh subprocess (pytest above already
runs separately from server).

## 2. Start a fresh P7 model server (terminal A)

~~~bash
cd ~/sglang-omni
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-p7-009"

unset SGLANG_OMNI_QTTS_CP_P4_DIR
unset SGLANG_OMNI_QTTS_CP_P5_DIR
unset SGLANG_OMNI_QTTS_CP_K2_DIAG_DIR

export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-p7-009/ARM"
export SGLANG_OMNI_QTTS_CP_P7_DIR="$HOME/results/qwen3tts-cpspec-k2-p7-009/conditional"
export SGLANG_OMNI_QTTS_CP_K2_SOURCE_SHA="$(git rev-parse HEAD)"

test ! -e "$SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE"
test ! -e "$SGLANG_OMNI_QTTS_CP_P7_DIR"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 \
  2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-p7-009/server.log"
~~~

## 3. Trigger the actual model (terminal B)

~~~bash
cd ~/sglang-omni
curl -fsS http://localhost:8000/health

python experiments/qwen3_tts_cp_spec_k2/quality_smoke.py \
  --base-url http://localhost:8000 \
  --arm-file "$HOME/results/qwen3tts-cpspec-k2-p7-009/ARM" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-p7-009/server.log" \
  --max-new-tokens 512 \
  --out "$HOME/results/qwen3tts-cpspec-k2-p7-smoke-009"

python - <<'PY'
import json
from pathlib import Path
p = Path.home() / "results/qwen3tts-cpspec-k2-p7-009/conditional"
files = list(p.glob("real_cond_pid*.json"))
if len(files) != 1:
    raise SystemExit(f"INVALID P7 EVIDENCE: expected one report, got {len(files)}")
r = json.loads(files[0].read_text())
print("P7 status:", r["status"])
print("Source SHA:", r["source_sha"])
print("Actual model predictor layers:", r["num_predictor_layers"])
print("Branch replay ms:", r.get("conditional_cuda_event_ms_by_offset"))
for cell in r["cases"]:
    print("offset", cell["offset"], "pass", cell["pass"],
          "checks", {name: v["max_abs"] for name, v in cell["checks"].items()})
if not r["status"].startswith("P7_REAL_MODEL_TWO_FIXED_DEPTH_CUDA_COND_PASS"):
    raise SystemExit("P7 REAL MODEL DISPATCH NOT PASS")
if r["num_predictor_layers"] != 5:
    raise SystemExit("P7 WRONG MODEL SHAPE: expected 5 layers")
if len(r["cases"]) != 3 or not all(x["pass"] for x in r["cases"]):
    raise SystemExit("P7 FAILED PARITY")
PY
~~~

## Next engineering decision

- P7 capture/parity FAIL: our model kernel backend, CUDA conditional
  capture mode or graph buffer ownership is incompatible. Diagnose
  first failure; no speculative E2E performance claim.
- P7 PASS: next P8 should nest the real 2-way model branches INSIDE the
  GPU WHILE, then expand to four fixed offsets using a balanced binary
  cond tree, and only then add GPU-side p/q sampler/rejection/bonus.
- Even after P8 functional PASS, finish by comparing a FULL device-
  resident K2 HTTP implementation vs the existing Graph-ON baseline
  on frozen prompts, natural EOS and profiler-OFF fresh-process ABBA.

# Qwen3-TTS CodePredictor Speculative Decoding K2 — Omni-native P0

**Experimental:** Implements a real K=2 draft -> packed verification ->
accept/reject/residual -> bonus-code loop in SGLang-Omni Qwen3-TTS.
No TensorRT, model exports, additional draft checkpoints or training.

Base: charliechenye/sglang-omni main at
3b73fdf389d265384db7cd603666c65861f5dd84
(same Omni SHA as A0 and A1 p/q captures).

## Scope / limitations

- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, 16 RVQ groups.
- Sampled, CUDA, batch=1, one semantic frame per Predictor call.
- K=2: use the existing head for one future code as proposal, do a two-token
  causal target verification at nonzero KV offset, accept min(1,p/q) or
  residual-resample, emit a bonus from trailing verified state on acceptance.
- Invalid speculative KV suffix is excluded from the logical cache and
  overwritten at the next round.
- Two-token packed verify supports the existing Triton decode_attention as
  well as explicit prefix-aware SDPA, not naive is_causal=True for Q_len<KV_len.
- Enabled only when SGLANG_OMNI_QTTS_CP_SPEC_K=2 AND the ARM marker exists.
  Before ARM, baseline is the original graph/eager path, including warmup.
- Debug PyTorch implementation uses repeated FP32 top-k/p, random draws,
  per-round .item() and Python control flow. **It is not optimized**.
- Existing A1 overlap was only 10.2% at T=.9, topk50, topp1.0; this is
  unlikely to be fast without dramatically better later-depth acceptance.
- Seeded output bytes will not match baseline exactly: experiments use a
  separate sampling RNG stream. Distribution/quality and KV parity must
  be verified before any "lossless" production claim.
- H200 smoke and real-model parity are NOT yet tested as of initial PR.

## 1. Checkout the fork branch in your existing Omni clone

Ensure previous A0/A1 probes are fully restored (clean tracked worktree).

~~~bash
cd ~/sglang-omni
git status --short
git remote get-url charlie-fork >/dev/null 2>&1 || \
  git remote add charlie-fork https://github.com/charliechenye/sglang-omni.git
git fetch charlie-fork experiment/qwen3-tts-codepredictor-spec-k2-p0
git switch -c experiment/qwen3-tts-codepredictor-spec-k2-p0 \
  --track charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git rev-parse HEAD
~~~

This experiment is source-pinned. Do not rebase onto newer upstream before
re-running source and kernel parity gates.

## Fix after the first HTTP 500: terminal-codebook projection

The initial H200 smoke reached K2 but returned
`index 14 is out of bounds for dimension 0 with size 14`.
This was a prototype bug at residual depth 15: the codec embedding has
15 tables, but Omni deliberately precomputes projection tables for only
the first 14. For the last code, the revised code uses the ordinary
embedding and projects it directly rather than indexing a nonexistent
precomputed table.

The CPU regression test covers both projected depth 14 and the final
depth 15, and the extra CUDA test executes the full K2 loop against a
fixture with **16 code groups, 15 codec tables, and 14 projected tables**.
This guards against another terminal-depth crash before HTTP.
Do not interpret the old 500 as a sampled-acceptance failure.

Stop any old server, pull this branch, rerun correctness tests and restart
the server so the worker loads the fixed Python code. The old smoke results
are partial evidence: use fresh `-002` directories, never overwrite `-001`.

## 2. CPU and H200 GPU correctness tests

~~~bash
cd ~/sglang-omni
python -m pytest -q \
  tests/unit_test/qwen3_tts/test_code_predictor_spec_k2.py \
  -k 'not packed_two_token'
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_code_predictor_spec_k2.py \
  -k 'packed_two_token or k2_real_16_group' -s
~~~

**Stop on a failure.** The GPU test compares the nonzero-offset two-row
causal verifier with the same two target calls executed sequentially,
including hidden and KV values under existing BF16 tolerances.

## 3. Start one isolated server (terminal A)

~~~bash
cd ~/sglang-omni
export CUDA_VISIBLE_DEVICES=0
export SGLANG_OMNI_QTTS_CP_SPEC_K=2
export SGLANG_OMNI_QTTS_CP_SPEC_ARM_FILE="$HOME/results/qwen3tts-cpspec-k2-002/ARM"
mkdir -p "$HOME/results/qwen3tts-cpspec-k2-002"

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000 2>&1 | tee "$HOME/results/qwen3tts-cpspec-k2-002/server.log"
~~~

**Do not ARM until health succeeds.** Server warmup runs baseline.
Removing the ARM file returns subsequent requests to the existing normal
serving path. K values other than 0 and 2 explicitly fail closed.

## 4. C1 live output smoke (terminal B)

~~~bash
cd ~/sglang-omni
python experiments/qwen3_tts_cp_spec_k2/smoke.py \
  --base-url http://localhost:8000 \
  --arm-file "$HOME/results/qwen3tts-cpspec-k2-002/ARM" \
  --out "$HOME/results/qwen3tts-cpspec-k2-smoke-002"
~~~

This runs baseline/spec/spec/baseline on ONE server, saving WAVs and
report.json. This is *not* a real performance ABBA. CPU sync and independent
sampling streams contaminate timing and hash comparisons.

Inspect the server log for lines beginning QTTS_CP_SPEC_K2. Confirm every
sampled frame reports rounds, proposed, accepted, bonus, rejected and
committed_residual=15. The smoke script prints QTTS_K2_VISIT_BEGIN and
QTTS_K2_VISIT_DONE for each arm, and on HTTP 500 writes failure.json naming
the failed cell. No such server-side K2 log means the candidate was not activated.
On CUDA failures or poor audio, stop instead of generating throughput claims.

## 4B. Full-output C1 gate: finish_reason + live K2 activity

First four-request smoke on Nov 21 produced HTTP 200 on all visits
(00_base, 01_spec, 02_spec, 03_base), with deterministic matching WAV SHA
within each arm. The debug K2 arm took ~3.3-3.6s versus baseline ~0.54-0.57s;
**these are debug-only timings, not real speedup numbers**.

Critically every output was **491564 bytes**. For a 24kHz 16-bit mono PCM
WAV with 44-byte header, that is 10.24 seconds = exactly 128 80ms codec
frames, and the original smoke had max_new_tokens=128. This strongly suggests
all four visits were length-capped; the raw audio endpoint does NOT expose
the terminal finish_reason to the old smoke. **Never infer a clean natural
EOS just from HTTP 200 or stable WAV SHA.**

The new quality_smoke.py uses the built-in single-item
/v1/audio/speech/batch endpoint, which still invokes the normal client.speech
generation path, but returns the authoritative per-item finish_reason and
base64 WAV. It requires all four to finish with "stop", correct 24kHz mono
PCM16 WAV, and actual candidate-only QTTS_CP_SPEC_K2 log lines.
Length/unknown/missing-activity are failed gates. It writes complete results
even when the final gate fails. No latency/throughput gate is attempted.

### CPU tests first

~~~bash
cd ~/sglang-omni
python -m pytest -q tests/unit_test/qwen3_tts/test_cp_spec_quality_smoke.py
~~~

### Use an already-healthy -002 server (if still running)

No server restart is needed merely to run this *new client script*: the
speculative decoder itself is unchanged by the quality harness. Do NOT run
an unrelated load test concurrently. Ensure the original ARM file is absent.

~~~bash
cd ~/sglang-omni
curl -fsS http://localhost:8000/health
test ! -e "$HOME/results/qwen3tts-cpspec-k2-002/ARM"

python experiments/qwen3_tts_cp_spec_k2/quality_smoke.py \
  --base-url http://localhost:8000 \
  --arm-file "$HOME/results/qwen3tts-cpspec-k2-002/ARM" \
  --server-log "$HOME/results/qwen3tts-cpspec-k2-002/server.log" \
  --max-new-tokens 512 \
  --out "$HOME/results/qwen3tts-cpspec-k2-quality-001"
~~~

If the server is no longer running, start a **fresh** server using separate
-003 ARM and server.log paths and pass those same paths here. Never overwrite
an earlier server.log or output directory. The quality harness intentionally
exits nonzero if any visit has finish_reason=length, if the K2 candidate
did not run (missing log frames), if a baseline unexpectedly runs K2, or if
the audio response is malformed. Inspect report.json in all cases.

### Reuse old -002 server logs without generating audio again

~~~bash
cd ~/sglang-omni
python - <<'PY'
from pathlib import Path
from experiments.qwen3_tts_cp_spec_k2.quality_smoke import parse_k2_log_entries
log = Path.home() / "results/qwen3tts-cpspec-k2-002/server.log"
events = parse_k2_log_entries(log.read_text(errors="replace"))
print("K2 logged frames:", len(events))
for key in ("rounds", "proposed", "accepted", "bonus", "rejected"):
    print(key, sum(event[key] for event in events))
proposed = sum(event["proposed"] for event in events)
accepted = sum(event["accepted"] for event in events)
print("observed draft acceptance:", accepted / proposed if proposed else None)
PY
~~~

Do not mix acceptance percentages from multiple model revisions or sampling
configs; this coarse parser does not identify the request or model SHA.
It only summarizes the log files you explicitly select.

## 5. Gates before an upstream optimization

1. Test live target per-depth logits and two-row causal verification vs
   sequential verification, BF16 KV ownership and cache rollback after
   rejection, codec embedding sums and output lengths.
2. Ensure torch top-k/top-p reference matches the actual fused target
   distribution including near-ties and rounding; check randomized output
   distribution equality rather than requiring identical seeded WAVs.
3. Multiple languages, speakers, seeds and model revisions, followed by
   TTFA, WER, latency, QPS and GPU occupancy. Current K2 branch intentionally
   does not enable C8/C16.
4. Only after correctness: remove per-round host sync, fuse proposal/accept
   on device, then run new-process ABBA against fully optimized Graph-ON
   no-spec baseline. An initial slow K2 smoke is not evidence the algorithm
   cannot work with a fused kernel.

## Sources

NVIDIA design:
https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/codepredictor-speculative-decoding.md

NVIDIA runtime:
https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/cpp/multimodal/qwen3_omni/codePredictor.cpp

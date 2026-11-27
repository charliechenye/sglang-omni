# Authoritative ASR Final candidate experiment

## Status

`SHADOW_FIXES_COMPLETE_READY_FOR_NATIVE_SCREENING`

No real-model screening has been run. No model weights were downloaded. This status means the deterministic fixes and screening tools are ready; it says nothing about candidate coverage, transcript quality, or latency opportunity.

Production is OFF unless `VOXT_EXPERIMENT_FINAL_CANDIDATES` is explicitly set. `instrumented` records the baseline without requesting candidate transport. `shadow` enables bounded candidate observation; `shadow_dense` uses the existing dense diagnostic cadence. Shadow arms retain ordinary WebSocket HeadStart dispatch and issue no candidate-triggered LLM requests. There is no active mode.

## Findings and fixes

| Finding | Fix and code | Evidence |
| --- | --- | --- |
| A late WebSocket generation could change the apparent Final-entry budget. | Record the initial retained generation count, late allowance, and effective limits at `waitFinal` entry; replay uses the recorded state. Unknown budgets and incomplete comparison pairs remain unresolved. [run_recognition.go](../internal/session/run_recognition.go), [headstart.go](../internal/session/headstart.go), [main.go](../cmd/final-candidate-replay/main.go), [policies.go](../cmd/final-candidate-replay/policies.go) | Replay tests cover 0/2/3 generations, a real late WebSocket dispatch, failed first attempts, missing budgets, multipart reuse, and earlier/later paired selections. |
| OFF, SHADOW, and SHADOW_DENSE could be pooled, and missing sessions could disappear from denominators. | Replay preserves arm/scenario/run identity, filters by arm, keeps warmup/cold-start/failed sessions in coverage, and excludes warmup/cold-start from steady-state scoring. | Synthetic three-arm JSONL tests cover filtering, grouping, failed/warmup/cold-start runs, and zero-candidate sessions. |
| Baseline actual adoption and alternative oracle compatibility were not comparable. | Replay keeps actual adoption/failure outcomes separate from retrospective compatibility, reports unresolved pairs, and records paired dispatch-ready lead including negative differences. [policies.go](../cmd/final-candidate-replay/policies.go), [outcomes.go](../cmd/final-candidate-replay/outcomes.go) | Tests cover both-compatible earlier/later cases, actual adoption, failed/missing outcomes, and stable request-ID attribution. |
| Early exits could omit the terminal experiment summary. | The manager's common terminal boundary finalizes once after publishing the terminal snapshot; terminal status distinguishes cancellation, no speech, empty Final, ASR failure, salvage, cleanup-not-requested, and missing plan. [manager.go](../internal/session/manager.go), [headstart_shadow_report.go](../internal/session/headstart_shadow_report.go) | Deterministic tests cover exactly-once cancellation reporting and terminal-status classification. The full benchmark waits for the terminal summary. |
| Experiment capture could run in production OFF. | OFF does not allocate candidate channels, capture plans, hash plans, or emit experiment events. Instrumented baseline captures common traces only; shadow arms additionally enable candidate transport. [headstart.go](../internal/session/headstart.go), [headstart_shadow.go](../internal/session/headstart_shadow.go), [asr.go](../internal/asr/asr.go) | Mode tests verify OFF, instrumented, shadow, and dense channel/trace behavior; the existing shadow session test verifies no candidate LLM dispatch. |
| Candidate support and decode path could be inferred from requested settings. | The native SSE protocol reports support, cadence, published/coalesced counts, and actual decode paths, guess/draft/reuse counters, batching, and chunking. [asr_service.cpp](../../sglang_omni_mlx/native/src/common/asr_service.cpp), [qwen3_asr_transcriber.cpp](../../sglang_omni_mlx/native/src/qwen3_asr/qwen3_asr_transcriber.cpp) | Weight-free protocol tests distinguish unsupported transport from supported transport with zero publishable candidates. |

## Native SSE feasibility screen

This local-only screen uses the native Qwen3-ASR HTTP transcription endpoint. It does not use a remote LLM, cleanup, HeadStart execution, Voxt UI, or fake capture. The runner tests local English ~3 s, English ~10 s, and Chinese ~10 s fixtures in automatic and explicit language modes, with one warmup and five interleaved OFF/SHADOW/SHADOW_DENSE measurements by default.

Build the runtime from a committed, clean worktree. This uses the existing build procedure and does not download model weights:

```sh
cd Voxt
OMNI_NATIVE=../sglang_omni_mlx/native scripts/build-omni.sh --out "$PWD/build/native-final-candidate-runtime"
```

Run from the repository root with a model directory already present locally:

```sh
python3 -m sglang_omni_mlx.native.ci.screen_qwen3_asr_candidates \
  --runtime-bin Voxt/build/native-final-candidate-runtime/bin/qwen3_asr_server \
  --model-dir /path/to/local/qwen3_asr \
  --model-id qwen3-asr-local \
  --output-dir Voxt/build/native-final-candidate-screen
```

Optional local draft weights can be supplied with `--draft-model-dir` and `--draft-bits`. `--oracle-guess-diagnostic` adds an explicitly oracle-derived `guessed_text` diagnostic from the OFF warmup; it is not a production preview estimate. The runner labels realtime-session Final Prefill Reuse unvalidated and defers it to the full Voxt benchmark.

The output directory contains `requests.jsonl` and `summary.json`. Records contain monotonic request/header/candidate/Final/[DONE] times, hashes and lengths instead of transcript text, per-candidate prefix/exact metrics, native generation metadata, paired latency and transcript-parity summaries, protocol status, and failed samples. `--fixture-debug-text` is opt-in and writes fixture transcript text into those artifacts. Unsupported protocol is reported as unsupported/unresolved; it is never scored as zero opportunity.

## Full Voxt screening

Run this only after native screening demonstrates a worthwhile lead opportunity. It reuses the existing benchmark, manifest, runtime provenance checks, and causal replay. It requires the configured cleanup endpoint; it does not download model weights.

```sh
cd Voxt
VOXTNEW_OMNI_RUNTIME="$PWD/build/native-final-candidate-runtime" \
VOXTNEW_FINAL_CANDIDATE_BENCH_ASR_MODEL_DIR=/path/to/local/qwen3_asr \
VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_ENDPOINT=https://llm.example/v1 \
VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_MODEL=cleanup-model \
VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS="$PWD/testdata/final-candidate-screening.example.json" \
VOXTNEW_FINAL_CANDIDATE_BENCH_WARMUPS=1 \
VOXTNEW_FINAL_CANDIDATE_BENCH_ROUNDS=5 \
scripts/benchmark-final-candidates.sh
```

Expected artifacts are `Voxt/build/final-candidate-benchmark.log` and `Voxt/build/final-candidate-benchmark.jsonl`. The script extracts JSONL even when the benchmark exits with a failure, then returns the original failure status. Replay the full report or filter one shadow cadence:

```sh
go run ./cmd/final-candidate-replay -input build/final-candidate-benchmark.jsonl > build/final-candidate-replay.json
go run ./cmd/final-candidate-replay -input build/final-candidate-benchmark.jsonl -arm shadow > build/final-candidate-replay-shadow.json
go run ./cmd/final-candidate-replay -input build/final-candidate-benchmark.jsonl -arm shadow_dense > build/final-candidate-replay-shadow-dense.json
```

The native screen is diagnostic only. A large standalone HTTP lead does not establish a Voxt opportunity because Final Prefill Reuse and target-verified guesses can shorten production Final. Stop this direction if realistic native leads are consistently only a few milliseconds; treat ~50 ms as a useful exploratory signal, not an SLA. Do not enable an active trigger in this revision.

## Validation boundary

The checked-in tests can validate parsing, eligibility, replay, reporting, and native SSE protocol semantics without model weights. Real-model candidate coverage, transcript parity, draft behavior, and latency remain unmeasured until a local runtime and Qwen3-ASR weights are available. Thirty-second fixtures are optional and do not block the initial screen.

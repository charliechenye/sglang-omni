# Authoritative ASR Final candidate experiment

## Status

`SHADOW_CORRECTNESS_COMPLETE_READY_FOR_NATIVE_SCREENING`

No real-model screening has been run. No model weights were downloaded. This status means the deterministic fixes and screening tools are ready; it says nothing about candidate coverage, transcript quality, or latency opportunity.

Production is OFF unless `VOXT_EXPERIMENT_FINAL_CANDIDATES` is explicitly set. `instrumented` records the baseline without requesting candidate transport. `shadow` enables bounded candidate observation; `shadow_dense` uses the existing dense diagnostic cadence. Shadow arms retain ordinary WebSocket HeadStart dispatch and issue no candidate-triggered LLM requests. There is no active mode.

## Findings and fixes

| Finding | Fix and code | Evidence |
| --- | --- | --- |
| A late WebSocket generation could change the apparent Final-entry budget. | Record the initial retained generation count, late allowance, and effective limits at `waitFinal` entry; replay uses the recorded state. Unknown budgets and incomplete comparison pairs remain unresolved. [run_recognition.go](../internal/session/run_recognition.go), [headstart.go](../internal/session/headstart.go), [main.go](../cmd/final-candidate-replay/main.go), [policies.go](../cmd/final-candidate-replay/policies.go) | Replay tests cover 0/2/3 generations, a real late WebSocket dispatch, failed first attempts, missing budgets, multipart reuse, and earlier/later paired selections. |
| OFF, SHADOW, and SHADOW_DENSE could be pooled, and missing sessions could disappear from denominators. | Replay preserves arm/scenario/run identity, filters by arm, keeps warmup/cold-start/failed sessions in coverage, and excludes warmup/cold-start from steady-state scoring. | Synthetic three-arm JSONL tests cover filtering, grouping, failed/warmup/cold-start runs, and zero-candidate sessions. |
| Baseline actual adoption and alternative oracle compatibility were not comparable. | Replay keeps actual adoption/failure outcomes separate from retrospective compatibility, reports unresolved pairs, and records paired dispatch-ready lead including negative differences. It counts a HeadStart miss only for a dispatched WebSocket decision with observed request outcomes and zero adopted parts; incomplete dispatches stay unresolved. Fallback cleanup request results (nil error or error) are reported separately, with attempts lacking a result kept unresolved. [policies.go](../cmd/final-candidate-replay/policies.go), [outcomes.go](../cmd/final-candidate-replay/outcomes.go), [cleanup.go](../internal/session/cleanup.go) | Tests cover both-compatible earlier/later cases, actual adoption, failed/missing outcomes, stable request-ID attribution, HeadStart misses, and completed/unresolved fallback attempts. |
| Early exits could omit the terminal experiment summary. | The manager's common terminal boundary finalizes once after publishing the terminal snapshot; terminal status distinguishes cancellation, no speech, empty Final, ASR failure, salvage, cleanup-not-requested, and missing plan. [manager.go](../internal/session/manager.go), [headstart_shadow_report.go](../internal/session/headstart_shadow_report.go) | Deterministic tests cover exactly-once cancellation reporting and terminal-status classification. The full benchmark waits for the terminal summary. |
| Experiment capture could run in production OFF. | OFF does not allocate candidate channels, capture plans, hash plans, or emit experiment events. Instrumented baseline captures common traces only; shadow arms additionally enable candidate transport. [headstart.go](../internal/session/headstart.go), [headstart_shadow.go](../internal/session/headstart_shadow.go), [asr.go](../internal/asr/asr.go) | Mode tests verify OFF, instrumented, shadow, and dense channel/trace behavior; the existing shadow session test verifies no candidate LLM dispatch. |
| Candidate support and decode path could be inferred from requested settings. | The native SSE protocol reports support, cadence, published/coalesced counts, and actual decode paths, guess/draft/reuse counters, batching, and chunking. [asr_service.cpp](../../sglang_omni_mlx/native/src/common/asr_service.cpp), [qwen3_asr_transcriber.cpp](../../sglang_omni_mlx/native/src/qwen3_asr/qwen3_asr_transcriber.cpp) | Weight-free protocol tests distinguish unsupported transport from supported transport with zero publishable candidates. |
| Optional generation counters made metadata parity all-or-nothing. | The runner compares fields independently and reports present, protocol-known zero, not-applicable, and unresolved states. Failed and empty transcripts are excluded from parity. [screen_qwen3_asr_candidates.py](../../sglang_omni_mlx/native/ci/screen_qwen3_asr_candidates.py) | Focused runner tests cover standalone, explicit zero, draft statistics, mismatch, missing metadata, and unsuccessful pairs. |
| A stale or dirty native runtime could contaminate standalone screening. | Before starting the server, the runner validates experiment/source commits, current source commit and cleanliness, source and binary paths, binary SHA-256, and the build provenance file. It preserves both artifacts on validation failure. | Focused runner tests cover valid and missing provenance, commit/path/hash mismatch, dirty source, and artifact preservation. |
| A late WebSocket generation could inherit adoption credit from a reused recording-time request, and one available multipart result could resolve the entire generation. | Decisions now record generation identity plus reused and fresh request IDs. Replay joins outcomes by stable event/request IDs, waits for every fresh result, and requires terminal evidence before counting a miss. It reports recording reuse, fresh late adoption, partial adoption, misses, failures, and cancellations separately. [headstart_late.go](../internal/session/headstart_late.go), [run_recognition.go](../internal/session/run_recognition.go), [headstart_shadow_report.go](../internal/session/headstart_shadow_report.go), [outcomes.go](../cmd/final-candidate-replay/outcomes.go), [policies.go](../cmd/final-candidate-replay/policies.go) | Replay tests cover reused versus fresh adoption, partial multipart results, missing terminal evidence, provider failures, and cancellation. |
| Candidate serialization before Final could inflate client lead. | Qwen3-ASR records authoritative completion against each path's candidate clock origin. SSE sends Final before any pending terminal mailbox candidate or candidate summary; candidate events identify terminal-drain delivery. The runner reports native candidate-to-completion separately from client event arrival and Final-to-DONE timings. [transcription.h](../../sglang_omni_mlx/native/src/common/transcription.h), [qwen3_asr_transcriber.cpp](../../sglang_omni_mlx/native/src/qwen3_asr/qwen3_asr_transcriber.cpp), [asr_service.cpp](../../sglang_omni_mlx/native/src/common/asr_service.cpp) | Weight-free protocol tests cover active candidate publication, terminal drain ordering, a slow SSE consumer, zero candidates, and error ordering. |
| Automatic-language oracle guesses were not representative of the token sequence used by the realtime Final path. | Oracle guess runs are retained only for explicit-language mode. Automatic-language oracle simulation is reported unsupported because the standalone `guessed_text` field cannot carry the detected-language header and ASR marker. Actual verified and accepted guess counts remain separate. Realtime Final Prefill Reuse stays unvalidated by this runner. | No production API was added. |
| Failed, unsupported, or timing-incomplete requests could appear as negative opportunity data. | Screening requires HTTP 200, one non-empty authoritative Final, terminal `[DONE]`, supported candidate protocol, matching cadence, summary, and native completion timing for shadow rows. Supported zero-candidate transport remains valid coverage. Missing pairs and invalid rows remain in denominators; opportunity distributions use only successful supported timing-valid rows. | Focused runner and protocol tests cover unsupported protocol, empty/failed parity, and zero-candidate support. |

## Native SSE feasibility screen

This local-only screen uses the native Qwen3-ASR HTTP transcription endpoint. It does not use a remote LLM, cleanup, HeadStart execution, Voxt UI, or fake capture. The runner tests local English ~3 s, English ~10 s, and Chinese ~10 s fixtures in automatic and explicit language modes, with one warmup and five interleaved OFF/SHADOW/SHADOW_DENSE measurements by default.

Build only after committing the experiment revision. The build script writes the existing provenance record. The runner checks that record against the current experiment commit, clean native source, exact source/runtime paths, and runtime binary hash before starting inference. No model weights are downloaded:

```sh
cd /path/to/sglang-omni
git status --short --branch
git rev-parse HEAD
test -z "$(git status --porcelain --untracked-files=all)"
cd Voxt
OMNI_NATIVE=../sglang_omni_mlx/native scripts/build-omni.sh --out "$PWD/build/native-final-candidate-runtime"
```

Run from the repository root with a Qwen3-ASR model directory already present locally. The default is one warmup and five measured rounds per arm and scenario, in automatic and explicit language modes, for the existing ~3.86 s English, ~9.68 s English, and ~10.30 s Chinese fixtures:

```sh
python3 -m sglang_omni_mlx.native.ci.screen_qwen3_asr_candidates \
  --runtime-bin Voxt/build/native-final-candidate-runtime/bin/qwen3_asr_server \
  --model-dir /path/to/local/qwen3_asr \
  --model-id qwen3-asr-local \
  --rounds 5 \
  --warmups 1 \
  --output-dir Voxt/build/native-final-candidate-screen
```

Optional local draft weights can be supplied with `--draft-model-dir` and `--draft-bits`; the runner never downloads them. To add oracle diagnostics, repeat the command with `--oracle-guess-diagnostic`. Only explicit-language oracle guesses are executed; automatic-language requests are recorded as unsupported because the standalone API cannot reproduce the realtime language header and ASR-text marker. Oracle runs are not production preview estimates.

Exact optional oracle diagnostic command:

```sh
python3 -m sglang_omni_mlx.native.ci.screen_qwen3_asr_candidates \
  --runtime-bin Voxt/build/native-final-candidate-runtime/bin/qwen3_asr_server \
  --model-dir /path/to/local/qwen3_asr \
  --model-id qwen3-asr-local \
  --rounds 5 \
  --warmups 1 \
  --oracle-guess-diagnostic \
  --output-dir Voxt/build/native-final-candidate-screen-oracle
```

The output directory contains `requests.jsonl` and `summary.json`; with the command above, they are `Voxt/build/native-final-candidate-screen/requests.jsonl` and `Voxt/build/native-final-candidate-screen/summary.json`. The summary shows the provenance record and validation checks. JSONL records contain HTTP/header/candidate/Final/[DONE] timings, native candidate-to-completion time, terminal-drain delivery, generation metadata states, hashes and lengths, and failures. Transcript text is omitted unless `--fixture-debug-text` is set. Read the run gate and candidate coverage first:

```sh
jq '{run_status, validation_errors, runtime_provenance, runtime_candidate_protocol_status, groups}' \
  Voxt/build/native-final-candidate-screen/summary.json
jq -c 'select(.event == "native_final_candidate_request" and .arm == "shadow") |
  {fixture, language_mode, round, status, candidate_protocol_supported,
   observed_candidate_cadence, native_published_candidate_count,
   native_sse_candidate_received_count, candidates}' \
  Voxt/build/native-final-candidate-screen/requests.jsonl
```

`run_status: validation_blocked` is a nonzero result and cannot be interpreted as no opportunity. Supported transport with zero published candidates is valid protocol coverage. Candidate lead distributions include only successful, supported, timing-valid shadow requests. The summary separates native generation overlap from client candidate-to-Final arrival, Final-to-DONE, and post-Final summary delivery. Standalone realtime Final Prefill Reuse remains explicitly unvalidated.

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

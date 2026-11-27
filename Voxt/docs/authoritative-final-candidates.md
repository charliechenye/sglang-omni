# Authoritative Final candidate experiment

## Status

`SHADOW_FIXES_COMPLETE_READY_FOR_SCREENING`

The experiment is OFF by default. `VOXT_EXPERIMENT_FINAL_CANDIDATES=shadow` enables bounded candidate observations; `shadow_dense` enables the denser diagnostic cadence. Both retain existing WebSocket HeadStart behavior and send no candidate-triggered LLM request. There is no active mode. Real-model screening has not been run, so this status makes no claim about opportunity, ASR quality, or latency improvement.

No model weights were downloaded. Local fixtures cover about 3 seconds of English and about 10 seconds of English and Chinese. This checkout has no suitable 30-second fixtures, so the 30-second cells remain for the Mac operator to supply.

## Findings and fixes

| Finding | Fix | Regression evidence |
| --- | --- | --- |
| Explicit-language decoding already has the ASR marker in the prompt, so requiring it in generated tokens suppresses candidates. | Candidate extraction accepts committed transcript tokens in explicit mode, requires a complete marker plus transcript tokens in automatic mode, and rejects unfinished markers, special spellings, invalid UTF-8, controls, EOS, loops, and the final budget snapshot. Authoritative `FinishedResult` is unchanged. | Native weight-free test `qwen3_asr_candidate_text_test`; Go event tests in `Voxt/internal/omni/final_candidates_test.go`. |
| Replay selected candidates using eventual outcome and could erase a failed first attempt. | Replay makes causal selections first using only fields known at dispatch time; retrospective outcome joins by stable request/candidate identity after selection. A failed or unresolved selection stays in history and keeps its simulated budget. | `TestReplayDoesNotSkipFailedFirstAttemptUsingFutureOutcome`; cancellation, Final-boundary, and missing-outcome cases in `Voxt/cmd/final-candidate-replay/main_test.go`. |
| HTTP and WebSocket candidates were modeled in separate budgets, and late WebSocket observations could misstate production eligibility. | Production WebSocket decision events record arrival, processing, preparation, dispatch, gates, budget, request plan, request IDs, and request starts. Baseline replay uses actual decisions. Combined policies put HTTP and WebSocket candidates through one policy-specific budget. | Shared-budget, production-event, baseline-adoption, recording-budget, and Final-boundary tests in `Voxt/cmd/final-candidate-replay/main_test.go`. |
| Retrospective hashing and compatibility analysis ran before Final returned to cleanup. | Final handling retains bounded immutable observations. Candidate processing records its causal timing before Final; summary, hashing, and oracle comparison run after normal delivery. The benchmark waits for its own report only after terminal delivery. | Session shadow test and the reporting wait in `TestMacE2EFinalCandidateBenchmark`; replay tests use separate eligibility and outcome events. |
| Shadow plans could differ from the request Cleanup actually sends after normalization and preparation; completed early-cleanup pieces could be counted as requests sent again. | Cleanup captures only unanswered production request parts at construction, preserves original part indices, and compares them with candidate parts through existing request compatibility checks. Completed pieces remain in their earlier cleanup records and are not counted as fresh requests to avoid. | `TestShadowPlanCompatibilityUsesPendingCleanupPartIndices` plus cleanup/request-plan and replay compatibility tests. |
| Candidate arrival time was mistaken for dispatch readiness. | Logs keep generated, received, processing, plan-ready, dispatch-ready, Final-available, and actual request-start times separately. A candidate processed after observable Final is dropped. Arrival and dispatch-ready lead are reported separately. | Session candidate-order and replay Final-during-preparation tests. |
| Provider first-token time was overwritten by later deltas, and requests starting before Stop were omitted. | The benchmark records the first nonempty provider delta once and retains negative Stop-relative request/token offsets with validity fields. Request kinds, request IDs, and adoption outcomes are recorded separately. | First-token and pre-Stop timing tests in `Voxt/mac_e2e_final_candidates_report_test.go`. |
| Intended HeadStart scenario labels were being confused with measured adoption. | Each request has an observed outcome keyed by stable request ID; exact adoption, repair adoption, partial multipart adoption, miss, failure, fallback, and no-cleanup are distinct from `headstart_case`. | `measuredHeadstartOutcomes` report tests and replay actual-adoption tests. |
| Fake text I/O could be mistaken for macOS insertion time. | Output distinguishes cleanup completion, result readiness, delivery request, simulated delivery completion, and session terminal. Delivery is labeled `simulated_fake_text_io`. | Benchmark pair/run schema and fake text I/O timestamp capture. |
| Warmups, cold starts, failed sessions, or empty transcripts could pollute steady-state latency/parity. | Replay excludes warmups and cold starts; aggregate coverage labels cold, warmup, and steady-state samples. Pair summaries exclude cold starts and failed/degraded sessions from successful latency/parity samples. Empty transcript/result pairs are invalid. | `TestReplayKeepsRecordingTimeHeadStartBudgetAndFiltersWarmups` covers warmup and cold-start exclusion; `benchRunSuccessful` and parity validity gate aggregates. |
| A binary hash alone could not prove the experimental source revision. | `scripts/build-omni.sh` writes provenance tying the runtime path and SHA-256 to a clean experiment/native source commit; both runner and shell gate verify it. Run rows include model IDs, fixture hash, language hint, draft/piece configuration, cleanup model, and provenance. | Provenance validation is performed before the benchmark starts. No stale or unknown runtime is treated as negative evidence. |
| A late request from one run could be attributed to the next run through a mutable active pointer. | Runtime events carry run/session IDs and LLM requests carry stable request IDs; the log collector resolves by those IDs and does not use the next session as the request owner. | Benchmark attribution is exercised by the fake LLM/event harness; request outcome joins use request IDs. |
| A run could report unsupported candidate transport as a no-opportunity result. | A manifest must include an explicit-language protocol smoke fixture. Each shadow arm must observe at least one HTTP candidate across the fixture's sessions; a single session with zero candidates remains a valid coverage outcome. | `TestValidateBenchScenariosRequiresExplicitProtocolSmoke`; benchmark support gate in `TestMacE2EFinalCandidateBenchmark`. |

The runner cannot currently expose authoritative generated-token counts or detected language from both OFF and shadow arms through the public session snapshot. It reports transcript/result parity and requested language; generation-metadata parity remains unavailable to this harness. Delivery completion is simulated, not a measurement of insertion into a macOS application.

## Screening matrix

The runner executes OFF, SHADOW, and SHADOW_DENSE for every scenario, with one warmup and five interleaved measured repetitions per arm. This is an exploratory screen, not statistically conclusive validation. Each treatment pair is compared to OFF on the same fixture and settings.

| Duration | Automatic language | Explicit language | Local fixture status |
| --- | --- | --- | --- |
| About 3 seconds | English `eval/en-2.wav` | English `eval/en-2.wav` | Available; 3.86 seconds. |
| About 10 seconds | English `eval/en-1.wav`; Chinese `eval/zh-filler-1.wav` | English `eval/en-1.wav`; Chinese `eval/zh-filler-1.wav` | Available; 9.68 and 10.30 seconds. |
| About 30 seconds | English fixture required | Chinese fixture required | None available locally; add local 16 kHz mono PCM WAV files before screening these cells. |

`Voxt/testdata/final-candidate-screening.example.json` is a runnable manifest for the available 3- and 10-second cells. `protocol_smoke` is set on an explicit English scenario. The `headstart_case` values are labels only; observed adoption is measured from request events.

## Build and run on the Mac

Commit the experiment first so the provenance record names the exact source revision. Build the native runtime from this worktree; the script does not download model weights, although its pinned build dependencies must be available to the Mac:

```sh
cd Voxt
OMNI_NATIVE=../sglang_omni_mlx/native scripts/build-omni.sh --out "$PWD/build/final-candidate-runtime"
```

Then run the available screening matrix against existing local model directories and a configured OpenAI-compatible cleanup endpoint:

```sh
cd Voxt
VOXTNEW_OMNI_RUNTIME="$PWD/build/final-candidate-runtime" \
VOXTNEW_FINAL_CANDIDATE_BENCH_ASR_MODEL_DIR=/path/to/local/qwen3_asr \
VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_ENDPOINT=https://llm.example/v1 \
VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_MODEL=cleanup-model \
VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS="$PWD/testdata/final-candidate-screening.example.json" \
VOXTNEW_FINAL_CANDIDATE_BENCH_WARMUPS=1 \
VOXTNEW_FINAL_CANDIDATE_BENCH_ROUNDS=5 \
scripts/benchmark-final-candidates.sh
```

The script verifies native source commit, clean worktree, runtime path, and binary SHA-256 before running. It never downloads ASR or draft weights. To add 30-second cases, put local 16 kHz mono WAVs beside the manifest or use absolute fixture paths and add automatic/explicit English and Chinese scenarios.

Expected outputs are `Voxt/build/final-candidate-benchmark.log` and `Voxt/build/final-candidate-benchmark.jsonl`. JSONL events are `final_candidate_benchmark_run`, `final_candidate_benchmark_pair`, `final_candidate_benchmark_summary`, `final_candidate_benchmark_coverage_summary`, plus content-free `voxt_*` experiment events. Run the causal replay and retain its JSON report:

```sh
cd Voxt
go run ./cmd/final-candidate-replay -input build/final-candidate-benchmark.jsonl > build/final-candidate-replay.json
```

Replay reports candidate coverage, unresolved outcomes, actual WebSocket decisions/adoptions, causal compatible selections, displaced baseline opportunities, wasted-generation estimates, and arrival/dispatch-ready lead distributions. Oracle-compatible lead remains retrospective and is not a projected end-to-end latency saving.

## Validation boundary

The deterministic tests establish protocol parsing, candidate eligibility/extraction, session fencing, shared-budget replay, request attribution, and benchmark record semantics. They do not establish real Qwen3-ASR candidate coverage, ASR quality/parity for real models, or end-to-end latency savings. Do not report a screening result until the real-model matrix runs. Native model-path and 30-second scenarios remain outstanding until local weights/runtime and fixtures are supplied.

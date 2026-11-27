# Authoritative Final candidate experiment

## Status

`SHADOW_READY_FOR_MODEL_VALIDATION`

The experiment defaults OFF. Set `VOXT_EXPERIMENT_FINAL_CANDIDATES=shadow` to observe candidates for Qwen3-ASR live Final requests. There is no `active` mode: the first stage must establish a causal opportunity with real measurements before an LLM trigger is considered.

This checkout has no local Omni runtime, Qwen3-ASR weights, draft weights, benchmark WAVs or benchmark endpoint configured. Mocked tests do not establish ASR quality or a latency benefit. The real-model run remains outstanding.

## Code path audit

1. The native transcription handler accepts `include_transcript_candidates=true` only with streaming, calls the model's optional candidate preparation hook, and uses a request-local one-slot coalescing mailbox. The inference worker publishes into the mailbox; only the HTTP handler writes SSE. Candidate events are drained before `transcript.text.done` or the error event, then `[DONE]`. See `sglang_omni_mlx/native/src/common/asr_service.cpp` (`HandleTranscriptions`, `CandidateMailbox`, `CandidateEvent`) and `sglang_omni_mlx/native/src/common/asr_service.h` (`ServedModel::PrepareWithCandidates`).
2. Qwen3-ASR wires the publisher into its normal transcriber options. Snapshots are taken from `commit` after plain decode or target verification, and from `AcceptToken` in batched decoding. Rejected guesses and draft proposals never reach these callbacks; accepted target-verified tokens do. Final Prefill Reuse keeps the same commit callback. Large native audio requests join completed window text with the current snapshot. See `sglang_omni_mlx/native/src/qwen3_asr/qwen3_asr_server.cpp` and `qwen3_asr_transcriber.cpp` (`Transcribe`, `TranscribeChunk`, `AcceptToken`, `PublishCandidate`, `SplitOutput`).
3. Candidate text passes through `SplitOutput`, prefix reconstruction and conservative checks for ASR markers, control bytes, invalid UTF-8 and special-token strings. The authoritative `FinishedResult` path is unchanged. Candidate snapshots begin at four committed tokens, then use a bounded cadence, with at most 64 snapshots per native request. Per-request counters retain the total build time and publication count when the mailbox coalesces older snapshots.
4. Voxt requests candidates only for the default-off shadow switch, Qwen3-ASR, and non-`LiveResult` streams. `final.go` parses candidates separately from `transcript.text.done` and ignores any candidate after Final. `live_final.go` joins prior recording pieces to each whole-request candidate and fences each request attempt. `asr.go` normalization and dictionary-echo checks suppress candidates that cannot safely be compared. Other providers and multiple Voxt Final HTTP chunks are excluded with a reason log; this does not serialize their existing batch path.
5. Session shadow candidates use a separate event and channel. `headstart_shadow.go` prepares a measurement-only cleanup plan through the existing compatibility code and records hashes, lengths, timing, gates and compatibility. It does not call `submitLate`, spend the late-generation allowance, mutate visible text or alter delivery. Candidate events that cannot enter the session event queue are recorded as suppressed.

Native counters are per HTTP request. Each candidate carries its Go request ID and attempt; the authoritative Final carries the session ID and attempt used to fence candidate adoption. Durations are monotonic offsets within their owning process; the benchmark does not subtract unrelated native and Go clock timestamps.

## Measurements and replay

Voxt emits `voxt_final_candidate_observed`, `voxt_final_candidate_eligibility`, `voxt_final_candidate_outcome` and `voxt_final_candidate_summary` JSON log events. They contain hashes and counts instead of transcript text. For each candidate, the outcome records arrival before Final, lead time, completeness, exact transcript match, exact/near cleanup-plan compatibility, repair eligibility, existing equivalent requests, gates, attempt/session match and whether a compatible incremental request was possible.

`earliest_oracle_compatible_lead_ms` is retrospective: it uses the authoritative Final to find an upper bound and is not a causal trigger or a measured latency improvement. `cmd/final-candidate-replay` evaluates causal policies from fields known at candidate time: first eligible HTTP snapshot, latest-preview match, text boundary, hypothesis extension, token/time cadence, and the existing WebSocket candidate control. Replay also filters out candidates that arrived after Final became available.

```sh
cd Voxt
go run ./cmd/final-candidate-replay -input build/final-candidate-benchmark.jsonl
```

The replay output reports per-policy selected sessions, compatible selections, wasted-request estimates and p50/p95 lead. Cadence thresholds are offline parameters, not production settings.

## Real-model benchmark

The Mac runner pairs `off` and `shadow` on the same commit, ASR model, cleanup model, fixture and settings. It runs one warmup per arm and five interleaved pairs per scenario by default, reversing arm order on alternating rounds. The first model startup is labeled separately. Each run emits transcript/result SHA-256 and lengths, Stop-relative Final/LLM/cleanup/delivery timings, request timing and token usage, candidate counters and content-free experiment events. The JSONL output is suitable for the replay command above.

Provide a JSON manifest with paths relative to the manifest, for example:

```json
[
  {
    "name": "short-en-auto-headstart-hit",
    "audio": "short-en.wav",
    "language": "",
    "headstart_case": "hit",
    "dictionary_terms": [],
    "draft": false
  }
]
```

For a useful validation set, include short, medium and long English, Chinese and mixed-language recordings; explicit and automatic language detection; dictionary-heavy cases; HeadStart hit and miss cases; piece/multi-chunk cases; and draft on/off cases where available. Supply local 16 kHz mono PCM WAV files. Use `VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_REMOTE=0` for a local cleanup endpoint; the current HeadStart eligibility gate is remote-only, so that arm measures ASR contention and shadow overhead rather than a local HeadStart trigger.

Example invocation (use existing local paths; the script never downloads weights):

```sh
cd Voxt
VOXTNEW_OMNI_RUNTIME=/path/to/qwen3_asr_server \
VOXTNEW_FINAL_CANDIDATE_BENCH_ASR_MODEL_DIR=/path/to/local/qwen3_asr \
VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_ENDPOINT=https://llm.example/v1 \
VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_MODEL=cleanup-model \
VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS=/path/to/scenarios.json \
scripts/benchmark-final-candidates.sh
```

The script writes `build/final-candidate-benchmark.log` and `build/final-candidate-benchmark.jsonl` by default. Set `VOXTNEW_FINAL_CANDIDATE_BENCH_OUTPUT` to change the log path. Local LLM peak memory and Metal contention are not currently sampled; transcript quality beyond authoritative transcript hashes requires review of the selected fixture results.

## Current validation boundary

The fake SSE/native protocol tests establish request gating, event order, candidate/final separation, error handling, session isolation and that shadow mode sends no extra LLM requests. They do not prove model accuracy, candidate coverage on actual Qwen3-ASR output, or latency improvement. Real-model results must be reported by scenario with sample counts, paired deltas and p50/p95 before considering P1.

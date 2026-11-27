# Qwen3-ASR Final Temporal Verification (P1B)

Status: `P1B_STATIC_FIXES_COMPLETE_PINNED_RUNTIME_PENDING`.

Branch: `perf/qwen3-asr-temporal-final-verification`.

Implementation base: `12ee8c6d49ee2dfe814571596d0b4d7825837bc2` (`upstream/main`).

This is an opt-in correctness prototype and remains disabled by default. P1A found argmax drift between fused-prefill verification and sequential decode for one refresh request, so refresh stays on its original path and exact token parity across numerical execution paths is not claimed. This branch has not been run with the pinned runtime or on CUDA hardware.

## Runtime path

The Qwen3-ASR factory option `enable_temporal_final_verification` defaults to `false`. Trusted realtime context is attached only to eligible final requests with a retained streaming prefix. Refresh requests, ordinary HTTP requests, stale segment attempts, invalid candidates, and incomplete histories use standard prefill.

The Qwen3-ASR builder selects its temporal scheduler only when the option is enabled. The inherited runtime builder resolves `extra_scheduler_kwargs` once and the model-specific override forwards those same arguments in either branch. No shared scheduler factory or SGLang upstream method is patched; the shared `OmniScheduler` prefill hook delegates to the upstream call unchanged by default.

The scheduler admits one isolated, whole-prompt request with greedy sampling, a page-size-one MHA allocator, no prefix or host cache hit, no chunking, no session or retraction, no custom logits, no logprobs, and no active speculative algorithm. The temporal scheduler keeps coalescing and request-build waits outside the cancellation lock, then locks native queue-to-batch construction and publishes the resulting `cur_batch` before releasing it. The native prefill selector checks the candidate KV and output budget before admission. Candidate IDs use a separate `array("q")` fill; failed preflight and native rejection restore the canonical request fill and MRoPE positions; native admission runs once.

The model runner attaches request-scoped metadata. The Qwen3-ASR logits adapter accepts the multimodal forward contract where `input_ids=None`, validates token and position metadata from the `ForwardBatch`, checks the request identity and selected hidden-state rows, then invokes the native logits processor once over the prompt-final row and candidate rows. Non-temporal forwards delegate directly to the native processor. No prompt retokenization or manual LM head is used.

The verifier keeps the longest matching candidate prefix and the first corrective or bonus target token. Before native one-token result processing, the scheduler validates the candidate/KV boundary and prepares replacement fill, MRoPE, and sequence metadata. It truncates KV and sequence lengths to `P + A`, where `P` is the prompt length and `A` is the accepted prefix length, while persistent MRoPE positions return to `P`. Forward-time verification still uses MRoPE positions through `P + K`, where `K` is the candidate count. The scheduler frees the rejected KV suffix through the existing allocator and invalidates `seq_lens_sum`. The native result processor then handles the pending token and normal finish/release lifecycle. A terminal release still uses the native `checkpoint` value; `skip_radix_cache_insert` independently prevents speculative KV publication. The FutureMap bridge receives the pending token only after native processing. Cancellation commits no candidate output and suppresses the pending token.

## Pinned interface and lifecycle limits

Admission preflight uses private SGLang 0.5.23 `PrefillAdder` sizing helpers. Commit handling edits `Req`, `ScheduleBatch`, and request-token-pool state before invoking the native result processor. This narrow adapter must be revalidated for every SGLang upgrade. A stable upstream integration should expose a public prefill admission plan and a native verified-prefill commit operation that owns KV truncation, sequence metadata, token append, finish/EOS handling, cache release, and publication as one operation.

The admission tests invoke `Qwen3ASRTemporalPrefillAdder.add_one_req()` with native `Req`, `SamplingParams`, and `array("q")` tokens. They replace the native sizing selector and base admission boundary to verify candidate isolation, fallback restoration, one admission call, and admission verdict handling; they do not exercise actual native allocator allocation or budget charging. The continuation test uses native `Req`, `ScheduleBatch`, `GenerationBatchResult`, and `SchedulerBatchResultProcessor` contracts. It checks canonical fill, accepted KV boundary, sequence lengths, prompt-only persistent MRoPE, reclaimed suffix, pending-token commit, native `prepare_for_decode()` boundary, terminal release arguments, and cache insertion suppression. Allocator operations remain recording doubles. A native retraction contract test checks prompt-only MRoPE extension from retained output IDs. The cancellation/admission test uses events to verify that abort cannot inspect the selected request until `cur_batch` has been published. These tests remain unexecuted here because the pinned runtime dependencies are unavailable.

## Supported configuration and graph scope

The feature requires CUDA, TP=1, PP=1, DP=1, CP=1, no DP attention or disaggregation, no DFlash or other SGLang speculation, async decode disabled, non-overlap scheduling, torch compile disabled, and a page-size-one MHA allocator. Temporal admission supports disabled or Breakable prefill graphs. Eligible temporal requests use eager prefill under Breakable graphs; ordinary requests retain their graph eligibility and decode graphs remain unchanged. Full and piecewise prefill graph modes fall back before candidate mutation. Full-graph temporal verification is out of scope.

Mixed or already-running batches, cache hits, chunking, insufficient output budget, non-greedy or transformed sampling, control/special tokens, stale metadata, and incomplete history fall back to standard prefill. The feature is not approved for a default change or a performance claim.

## Matched launch commands

Run from the repository root with the model already available locally. The two commands below differ only in the temporal flag.

Temporal off:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --asr.factory.enable_temporal_final_verification false \
  --asr.factory.enable_async_decode false \
  --asr.engine.disable_speculative_decoding true \
  --asr.engine.disable_overlap_schedule true \
  --asr.engine.enable_torch_compile false \
  --asr.engine.cuda_graph_backend_prefill breakable \
  --asr.engine.page_size 1 \
  --port 8000
```

Temporal on:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --asr.factory.enable_temporal_final_verification true \
  --asr.factory.enable_async_decode false \
  --asr.engine.disable_speculative_decoding true \
  --asr.engine.disable_overlap_schedule true \
  --asr.engine.enable_torch_compile false \
  --asr.engine.cuda_graph_backend_prefill breakable \
  --asr.engine.page_size 1 \
  --port 8000
```

## Validation

The implementation environment does not have `torch`, `sglang`, `pytest`, or `transformers`, and has no CUDA device or local model checkpoint, so the Python unit tests and H200 smoke cannot run here. `pre-commit run --all-files` passes. Static syntax, whitespace, format, and repository checks do not replace pinned-runtime admission, result-processing, decode-continuation, KV-release, or H200 smoke validation.

The first H200 smoke is C1 with the existing real-audio workload and a 500 ms refresh interval. Accept it only if temporal admission and verification are nonzero, candidates are accepted at least once, zero-decode completion occurs, at least one rejection continues through ordinary decode, A/B final token and EOS outputs match exactly, refresh requests stay on their ordinary path, every request completes, allocator ownership remains valid, no CUDA error occurs, and ordinary graph execution still works. Then run the Qwen3-ASR streaming, temporal admission, temporal verification, shared engine factory, scheduler, protocol, config, realtime-session, and unaffected-model regression suites in the pinned environment. Performance qualification is later work: use the same local checkpoint and 64 real audio samples for matched temporal-off/on C1/C4/C16/C32/C64 runs, warmup, repeated ABBA visits, and separate correctness/performance traces. Record final latency p50/p95/p99, absolute milliseconds saved, refresh latency, throughput, admitted batch sizes, eligibility/acceptance/rejection, zero-decode completions, graph replay, decode rounds, exact output token/EOS parity, WER/CER, utilization, errors, cancellations, and allocator health. P1A component measurements do not establish an end-to-end serving speedup.

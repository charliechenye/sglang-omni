# Qwen3-ASR Final Temporal Verification (P1B)

Status: `P1B_REVIEW_FIXES_INCOMPLETE`.

Branch: `perf/qwen3-asr-temporal-final-verification`.

Implementation base: `d1e437f52878e37fc22749e659a1fc58955d5db5` (`upstream/main`).

This is an opt-in correctness prototype and remains disabled by default. P1A found argmax drift between fused-prefill verification and sequential decode for one refresh request, so refresh stays on its original path and exact token parity across numerical execution paths is not claimed. This branch has not been run with the pinned runtime or on CUDA hardware.

## Runtime path

The Qwen3-ASR factory option `enable_temporal_final_verification` defaults to `false`. Trusted realtime context is attached only to eligible final requests with a retained streaming prefix. Refresh requests, ordinary HTTP requests, stale segment attempts, invalid candidates, and incomplete histories use standard prefill.

The Qwen3-ASR builder selects its temporal scheduler only when the option is enabled. The inherited runtime builder resolves `extra_scheduler_kwargs` once and the model-specific override forwards those same arguments in either branch. No shared scheduler factory or upstream method is patched.

The scheduler admits one isolated, whole-prompt request with greedy sampling, a page-size-one MHA allocator, no prefix or host cache hit, no chunking, no session or retraction, no custom logits, no logprobs, and no active speculative algorithm. The native prefill selector checks the candidate KV and output budget before admission. Failed preflight restores the canonical request fill and MRoPE positions; native admission runs once.

The model runner attaches request-scoped metadata. The Qwen3-ASR logits adapter accepts the multimodal forward contract where `input_ids=None`, validates token and position metadata from the `ForwardBatch`, checks the request identity and selected hidden-state rows, then invokes the native logits processor once over the prompt-final row and candidate rows. Non-temporal forwards delegate directly to the native processor. No prompt retokenization or manual LM head is used.

The verifier keeps the longest matching candidate prefix and the first corrective or bonus target token. Before native one-token result processing, the scheduler validates the candidate/KV boundary and prepares replacement fill, MRoPE, and sequence metadata. It truncates to `P + A`, where `P` is the prompt length and `A` is the accepted prefix length, frees the rejected KV suffix through the existing allocator, and invalidates `seq_lens_sum`. The native result processor then handles the pending token and normal finish/cache lifecycle. The FutureMap bridge receives the pending token only after native processing. Cancellation commits no candidate output and suppresses the pending token.

## Pinned interface and lifecycle limits

Admission preflight uses private SGLang 0.5.23 `PrefillAdder` sizing helpers. Commit handling edits `Req`, `ScheduleBatch`, and request-token-pool state before invoking the native result processor. This narrow adapter must be revalidated for every SGLang upgrade. A stable upstream integration should expose a public prefill admission plan and a native verified-prefill commit operation that owns KV truncation, sequence metadata, token append, finish/EOS handling, cache release, and publication as one operation.

The automated continuation test now uses native `Req`, `ScheduleBatch`, `GenerationBatchResult`, and `SchedulerBatchResultProcessor` contracts. It checks the canonical fill, accepted boundary, CPU/GPU/original sequence lengths, MRoPE boundary, reclaimed suffix, pending-token commit, native `prepare_for_decode()` boundary, terminal decode processing, and the native release call boundary. Its allocator and cache operations are recording test doubles; it does not exercise actual `PrefillAdder` allocation, the allocator implementation, or KV release implementation. The admission tests still use fixtures and do not establish every native admission and budget behavior described in the review request.

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

The implementation environment does not have `torch`, `sglang`, `pytest`, or `transformers`, and has no CUDA device or local model checkpoint. The unit tests and `pre-commit run --all-files` therefore cannot run here. Static syntax, whitespace, format, and repository checks are reported with the branch handoff; they do not replace pinned-runtime admission, result-processing, decode-continuation, KV-release, or H200 smoke validation.

Before any performance qualification, run the Qwen3-ASR streaming, temporal admission, temporal verification, shared engine factory, scheduler, protocol, config, realtime-session, and unaffected-model regression suites in the pinned environment. On H200, run matched temporal-off/on arms with the same local checkpoint, 64 real audio samples, a 500 ms refresh interval, concurrency C1/C4/C16/C32/C64, warmup, repeated ABBA visits, and separate correctness and performance traces. Record final latency p50/p95/p99, absolute milliseconds saved, refresh latency, throughput, admitted batch sizes, eligibility/acceptance/rejection, zero-decode completions, graph replay, decode rounds, exact output token/EOS parity, WER/CER, utilization, errors, cancellations, and allocator health. P1A component measurements do not establish an end-to-end serving speedup.

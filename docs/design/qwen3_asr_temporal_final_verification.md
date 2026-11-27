# Qwen3-ASR Final Temporal Verification (P1B)

Status: `P1B_IMPLEMENTED_GPU_VALIDATION_PENDING`.

Branch: `perf/qwen3-asr-temporal-final-verification`
Implementation base: `0f70d9c7c41c3e239dc129cb7b2c321e815c92e4` (`upstream/main`).

This is an opt-in prototype. The default remains off. It is not production-ready: P0 observed argmax drift between fused-prefill verification and sequential decode, and this branch has not been validated on a GPU.

## Runtime path

`enable_temporal_final_verification` is a Qwen3-ASR factory option, defaulting to `false`. The realtime session attaches trusted request metadata only to a final request that has a retained streaming prefix. Refresh requests do not receive the metadata. The request builder checks request/segment/attempt identity, prefix and language consistency, token validity, and output budget. It tokenizes the previous hypothesis, reuses `retained_streaming_prefix()`, and takes at most five tokens after the retained token prefix. The normal prompt remains the same.

The scheduler admits only an isolated, whole-prompt request with greedy sampling, page-size-one MHA allocation, no prefix/host cache hit, no chunking, session, retraction, logprobs, custom logits, or active speculative algorithm. The existing SGLang prefill selector checks the extra KV and output budget before mutation; native admission and allocation run once. Rejected or unsupported requests use the ordinary path and emit one fallback reason.

The model runner attaches request-scoped forward metadata. The Qwen3-ASR logits adapter first runs the native processor unchanged for the ordinary sampled output, then uses the same native processor and LM head on only the prompt-final and candidate-position hidden rows. Temporal requests require an exact unpadded eager prefill shape. Full and piecewise prefill graph configurations fall back before candidate mutation. With Breakable prefill graphs configured, only the temporal prefill runs eagerly; ordinary Breakable prefill replay and decode graphs remain available.

The verifier accepts the longest matching candidate prefix and returns the first corrective prediction or the bonus prediction. Before calling SGLang's ordinary one-token result processor, the scheduler records accepted tokens, frees the rejected KV suffix once through the native allocator, truncates speculative fill and MRoPE positions, and updates logical KV lengths. The native result processor then commits the single pending token and handles finish, release, cache checkpoint, and publication. Cache insertion is skipped for this request so the speculative suffix is not published. Verification is one-shot; a later re-prefill after retraction follows the normal path.

## Pinned SGLang interface caveat

The prefill admission preflight calls private SGLang 0.5.23 `PrefillAdder` sizing helpers, and the result adapter trims `Req` and token-pool state before invoking native one-token result handling. This narrow path must be revalidated on any SGLang change. A stable upstream integration should expose a public prefill admission-plan API and a typed verified-prefill commit API that owns accepted token appends, page-aware KV truncation, sequence lengths, stop/EOS handling, cache/release, and publication in one native transaction.

## Supported configuration and fallbacks

The flag currently requires CUDA, TP=1, PP=1, DP=1, CP=1, no DP attention or disaggregation, no DFlash/other SGLang speculation, non-overlap scheduling, and torch compile disabled. Temporal admission also requires the Breakable or disabled prefill graph backend, page size one, no chunked prefill, and the supported MHA allocator. Full/piecewise graph mode, mixed or already-running batches, cache hits, insufficient output budget, non-greedy or transformed sampling, control/special tokens, stale metadata, and incomplete history fall back to standard prefill.

The CUDA graph is not globally disabled. Breakable prefill graphs remain enabled for ordinary prefill, and the normal decode graph path is unchanged. This branch has not verified graph replay, KV ownership, or token parity on hardware.

## Opt-in launch

From the repository root, with the model already available locally:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --asr.factory.enable_temporal_final_verification true \
  --asr.engine.disable_overlap_schedule true \
  --asr.engine.enable_torch_compile false \
  --asr.engine.cuda_graph_backend_prefill breakable \
  --port 8000
```

Breakable prefill graphs keep ordinary graph replay available while temporal requests use the eager fallback. If the selected attention backend or graph configuration cannot support this setup, startup validation or per-request admission must fail closed. Do not enable the flag for HTTP transcriptions or refresh requests.

## Validation

CPU/unit coverage is in `tests/unit_test/qwen3_asr/test_streaming.py`, `test_temporal_verification.py`, and `test_temporal_admission.py`: retained-prefix and UTF-8 rollback, final/refresh and request identity, output-budget and special-token filters, realtime metadata serialization, greedy alignment, EOS bonus and EOS at mismatch, first/middle rejection, no-EOS full acceptance, admission fallbacks, and temporal KV precommit/reclamation including cancellation. Run with:

```bash
pytest tests/unit_test/qwen3_asr/test_streaming.py tests/unit_test/qwen3_asr/test_temporal_verification.py tests/unit_test/qwen3_asr/test_temporal_admission.py
```

Also run `pre-commit run --all-files`. GPU validation is pending: this development environment has no CUDA device, checkpoint, or installed `torch`, `sglang`, and `pytest` packages.

## P1B H200 handoff

Compare matched arms: A) NoSpec/temporal off, B) NoSpec/temporal on, and C) existing DFlash/temporal off. Use the same Qwen3-ASR checkpoint, 64 real audio samples, 500 ms refresh interval, encoder and scheduler settings, and GPU allocation. Test client concurrency C1/C4/C16/C32/C64 with warmup and repeated ABBA visits; keep correctness traces separate from clean performance runs.

Record final latency p50/p95/p99 and absolute milliseconds saved; refresh latency; session throughput; actual admitted batch sizes; eligibility, acceptance and rejection; zero-decode completions; actual prefill graph replay; ordinary decode rounds; exact output token/EOS parity; WER/CER; utilization; errors, cancellations, and allocator health. Require zero observed A/B token/EOS drift on the agreed validation set, no refresh regression, no lifecycle/cache failures, and no material throughput/tail regression before considering any default change. The P1A fixed-batch results do not establish an end-to-end serving speedup.

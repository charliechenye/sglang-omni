# Qwen3-ASR Final Temporal Verification

Final realtime transcription can reuse up to five rollback tokens from the
previous streaming hypothesis as greedy proposals. The target model verifies
those tokens in one prefill, commits the matching prefix, and continues with
SGLang's normal decoding and completion path. When native Breakable CUDA Graph
eligibility allows it, the prefill uses the native graph runner.

The feature is off by default. Enable it on the Qwen3-ASR stage with:

```text
--asr.factory.enable_temporal_final_verification true
```

It currently requires CUDA, TP/PP/DP/attention-CP size one, disabled overlap,
async decoding, Torch compile, plain fallback, and SGLang speculative decoding.
Prefill graphs must be disabled or use the Breakable backend; enabling the
feature selects Breakable as the CUDA default. The temporal prefill also needs
a page-size-one allocator and an isolated, whole-prompt greedy request with no
prefix-cache or host-cache ownership. Hybrid KV execution is unsupported.

Only a trusted final realtime request with a matching retained hypothesis and
language, valid rollback tokens, and enough output budget is eligible. Other
requests use ordinary Qwen3-ASR prefill and decoding. A temporal request that
cannot use native graph replay runs eagerly through the native model path.

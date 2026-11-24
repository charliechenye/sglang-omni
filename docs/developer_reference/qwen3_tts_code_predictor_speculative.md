# Qwen3-TTS CodePredictor speculative decoding

## Status

The model-connected frame runner, grouped LM-head projection, production Top-K probability output, request-isolated RNG, packed fused verification, logical KV commitment, and persistent verifier CUDA Graph capture are implemented. The feature remains opt-in and GPU correctness and speed are unverified on this development machine.

The runner is reachable through `Qwen3TTSTalker.code_predictor_forward`. It first processes the opening pair and samples codebook 1, then generates future-head proposals, verifies the pending code plus drafts in one packed `FusedPredictorLayers` call, applies the target heads once to verified hidden rows, accepts the draft prefix, samples a residual correction or bonus, commits outputs and embeddings, and repeats until all Predictor codebooks are complete. The normal carry-forward path has no standalone correction forward.

Backend `cuda_graph` is the default when a nonzero width is configured. It captures a persistent verifier for supported batch buckets during model setup. Backend `eager` is an explicit CUDA correctness and debugging path; it still uses the production fused Predictor and GPU sampling kernels. Width 0 remains the default and allocates no speculative seed, head, metadata, or graph buffers. Width 1 is rejected; widths 2 through 8 are requested verify widths.

An example opt-in configuration is [qwen3_tts_1_7b_customvoice_speculative.yaml](../../examples/configs/qwen3_tts_1_7b_customvoice_speculative.yaml). The native-off configuration remains [qwen3_tts_1_7b_customvoice.yaml](../../examples/configs/qwen3_tts_1_7b_customvoice.yaml).

## State and cache contract

Let `m` be the number of committed Predictor codes after codebook 0. At frame start, codebook 1 has been committed to output but is pending in the Predictor input; the opening talker-hidden/codebook-0 pair occupies two KV slots, so the logical KV length is `m + 1 = 2`. The carried hidden row is immediately before the pending code.

Each round verifies the pending input and up to `verify_width - 1` drafts. After accepting `k` drafts and committing one correction or bonus, `m` and the logical KV length both advance by `k + 1`. The last new code becomes pending, and its preceding hidden row is selected from the verifier output at offset `k`. The invariant remains `logical_kv_length = m + 1`.

Rows close to the end of a frame can have zero proposals while other rows verify a full window. They still receive one valid query if active; finished rows have zero valid queries. Inactive embeddings, LM-head selectors, positions, and output stores use safe staged values or masks before access. Positions at `-1` do not write KV and do not participate in attention. Rejected physical KV suffixes may remain in cache but are outside the logical prefix and are overwritten before reuse.

The fixed-width policy reduces width to fit `min(batch_size * verify_width, MAX_FUSED_ROWS)`, the graph bucket, and `verify_width * GQA_group <= ATTENTION_ROWS`. A failed graph capture retries narrower widths before decoding. If no supported width remains, the request uses native decoding with a reason in model info. Generic attention is not used for packed verification.

## Sampling, RNG, and precision

The speculative probability builder reuses Omni's score packing, temperature scaling, top-k ordering, FP32 normalization, and top-p boundary rule. It returns candidate probabilities and IDs for proposal, target, and residual calculations. On SM90 it uses the existing split chunk/merge selection infrastructure. The NVIDIA-style radix selector is present as a parity candidate for larger top-k values; it is not the default sampler. An accelerator parity test compares it with the production split path.

For top-k values up to 32, the production probability kernel reconstructs the native PyTorch gather order and bitonic sort order. For larger top-k values, packed score keys order score ties by token ID, matching the split selector contract. The grouped Triton LM-head kernel accumulates BF16 products in FP32 and rounds logits to BF16. Its output may differ from the native GEMM at rounding boundaries, so exact target-distribution parity must be established by the H200 tests before claiming mathematical output equivalence.

Ordinary sampled requests are eligible even though the request builder generated internal sampler seeds. Explicit user seeds, deterministic mode, greedy or mixed-sampling batches, unbounded or unsupported top-k settings, missing request identity, unsupported vocabulary or dtype, tensor parallelism, and missing fused capacity use native decoding. A batch containing any explicit-seed or greedy row falls back as a whole. Speculative uniforms are namespaced by request seed, semantic position, round, and draw purpose; GPU generation is stateless and graph-safe.

## Metrics

`/model_info` includes `predictor_speculative` metrics. `frames` counts successful request-row frames; `committed_codes` includes every Predictor code committed in those frames; `round_committed_codes` counts codes committed after the initial codebook-1 sample. Draft, accepted, bonus, rejection, packed forward, valid query, and padded query counts are separate. The effective-width histogram counts batch-rounds. `generic_predictor_forwards` and `standalone_correction_forward_count` must remain zero for completed speculative frames.

`configuration_time_ms` measures speculative setup, including opt-in buffers, head packing, and graph capture; the serving benchmark prints this field for the native width-0 server and the enabled server so their feature setup costs can be compared. It does not measure total process launch-to-ready time. `graph_capture_time_ms` isolates verifier capture wall time, and `graph_capture_memory_delta_bytes` is the approximate CUDA allocator increase during capture, not total model memory. `native_fallback_fraction` is computed over attempted frame rows. A successful performance run must show positive frame and packed-forward deltas and zero fallback delta for its unseeded workload. The benchmark reports full fused stack traversals per frame and packed query padding fraction.

## H200 test commands

Run CPU reference, acceptance-state, and eligibility tests:

```bash
pytest -q tests/unit_test/qwen3_tts/test_predictor_speculation.py
```

Run packed attention positions, C1/C8/C16 capacity, and tail-mask checks:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_kernels.py
```

Run grouped LM-head and target-distribution parity:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_grouped_heads.py
```

Run native probability, split Top-K, radix-candidate, and tie-order tests:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_sampling_kernels.py -k "speculative_probabilities or speculative_radix"
```

Run a complete unseeded eager CUDA frame and verify all codebooks and accumulated codec embeddings:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k unseeded_eager
```

Force each rejection position in a three-proposal window, then replay committed codes through the sequential native Predictor to compare target logits, probabilities, carried hidden states, and valid KV prefixes across heterogeneous tail depths:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k forced_code_replay
```

Run persistent verifier graph capture/replay correctness:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k verifier_graph
```

Run the synthetic fixed-frame smoke test. It checks that the speculative path completes frames without falling back; it is not a performance comparison.

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k frozen_frame_smoke
```

Run explicit-seed fallback followed by an unseeded speculative frame:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k explicit_seed_fallback
```

Run the mixed explicit-seed batch safety case, then verify that a later fully unseeded batch enters speculation:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k mixed_explicit_seed_batch
```

Run graph replay and ragged cache accesses under Compute Sanitizer:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py
```

## H200 single-GPU serving comparison

Use one H200 for both runs. Keep the request payload, model revision, concurrency, clocks, and process environment fixed. The native phase requires positive replays of the full Predictor CUDA Graph. The speculative phase requires positive verifier graph replays, positive speculative frame counts, and zero native fallbacks. The current speculative graph captures the packed verifier; proposal generation, acceptance, and commitment still run outside that graph.

Start the native server with speculative decoding disabled:

```bash
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --config examples/configs/qwen3_tts_1_7b_customvoice.yaml --port 8000
```

In a second terminal, warm up and measure the fixed payload. This writes the native baseline and graph replay counters to `/tmp/qwen3-tts-native-baseline.json`.

```bash
python scripts/benchmark_qwen3_tts_speculative_serving.py --phase native --base-url http://127.0.0.1:8000 --baseline-file /tmp/qwen3-tts-native-baseline.json --request-count 32 --warmup-requests 2 --concurrency 4
```

Stop the native server before starting the speculative server on the same GPU and port:

```bash
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --config examples/configs/qwen3_tts_1_7b_customvoice_speculative.yaml --port 8000
```

Run the same workload against that process. The benchmark omits the `seed` request field so speculative decoding remains eligible, reads actual model counters around the measured requests, and exits nonzero if speculative frames or graph replays are missing, if any frame falls back, or if the configured serving speedup is missed.

```bash
python scripts/benchmark_qwen3_tts_speculative_serving.py --phase speculative --base-url http://127.0.0.1:8000 --baseline-file /tmp/qwen3-tts-native-baseline.json --request-count 32 --warmup-requests 2 --concurrency 4 --minimum-speedup 1.0
```

The output includes request latency, committed codes, accepted drafts, bonus tokens, effective widths, valid and padded query counts, head evaluations, generic forward counts, standalone correction forward counts, and graph replay counts. The API benchmark measures full speech-request latency; it does not substitute a verifier-only timing for full-frame serving latency. The current speculative CUDA Graph is verifier-only, so this benchmark does not yet meet a round-level graph comparison gate.

## Push commands for a personal fork

The configured `origin` remote is the personal fork. Push the current branch without opening a PR:

```bash
git push origin HEAD:refs/heads/feat/qwen3-tts-codepredictor-specdec
```

## NVIDIA parity reference

The parity audit was anchored to NVIDIA TensorRT-Edge-LLM revision `513dcd18291e587c80928911a480761352f3c2d8`. Its frame loop applies future RVQ heads to the carried hidden state, verifies the pending token plus drafts in one pass, and commits the correction or bonus as a pending token. NVIDIA's public overview describes the same shared-residual-stream rationale and round semantics in its [CodePredictor speculative decoding guide](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/codepredictor-speculative-decoding.md).

This implementation does not establish NVIDIA-matched performance. The NVIDIA exporter uses a different FP16 model and TensorRT kernel stack, while this Omni path currently accepts single-rank BF16 heads and uses Omni's fused Predictor. H200 correctness, output-distribution parity, graph lifecycle, serving speedup, and audio quality remain required validation gates.

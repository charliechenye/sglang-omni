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

The speculative probability builder reuses Omni's score packing, temperature scaling, top-k ordering, FP32 normalization, and top-p boundary rule. It returns candidate probabilities and IDs for proposal, target, and residual calculations. On SM90 it uses the existing split chunk/merge selection infrastructure. The NVIDIA-style radix selector is present as a parity candidate for larger top-k values; it is not the default sampler. The candidate is tested against the production split path on SM90.

For top-k values up to 32, the production probability kernel reconstructs the native PyTorch gather order and bitonic sort order. For larger top-k values, packed score keys order score ties by token ID, matching the split selector contract. The grouped Triton LM-head kernel accumulates BF16 products in FP32 and rounds logits to BF16. Its output may differ from the native GEMM at rounding boundaries, so exact target-distribution parity must be established by the H200 tests before claiming mathematical output equivalence.

Ordinary sampled requests are eligible even though the request builder generated internal sampler seeds. Explicit user seeds, deterministic mode, greedy or mixed-sampling batches, unbounded or unsupported top-k settings, missing request identity, unsupported vocabulary or dtype, tensor parallelism, and missing fused capacity use native decoding. A batch containing any explicit-seed or greedy row falls back as a whole. Speculative uniforms are namespaced by request seed, semantic position, round, and draw purpose; GPU generation is stateless and graph-safe.

## Metrics

`/model_info` includes `predictor_speculative` metrics. `frames` counts successful request-row frames; `committed_codes` includes every Predictor code committed in those frames; `round_committed_codes` counts codes committed after the initial codebook-1 sample. Draft, accepted, bonus, rejection, packed forward, valid query, and padded query counts are separate. The effective-width histogram counts batch-rounds. `generic_predictor_forwards` and `standalone_correction_forward_count` must remain zero for completed speculative frames.

`configuration_time_ms` measures speculative setup, including opt-in buffers, head packing, and graph capture; the serving benchmark prints this field for the native width-0 server and the enabled server so their feature setup costs can be compared. It does not measure total process launch-to-ready time. `graph_capture_time_ms` isolates verifier capture wall time, and `graph_capture_memory_delta_bytes` is the approximate CUDA allocator increase during capture, not total model memory. `native_fallback_fraction` is computed over attempted frame rows. A successful performance run must show positive frame and packed-forward deltas and zero fallback delta for its unseeded workload. The benchmark reports full fused stack traversals per frame and packed query padding fraction.

## H200 test commands

Run CPU reference and RNG tests:

```bash
pytest -q tests/unit_test/qwen3_tts/test_predictor_speculation.py tests/unit_test/qwen3_tts/test_speculative_rng.py
```

Run packed attention positions, C1/C8/C16 capacity, and tail-mask checks:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_kernels.py
```

Run grouped LM-head and target-distribution parity:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_grouped_heads.py
```

Run native probability, SM90 split Top-K, radix-candidate, and tie-order tests:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_sampling_kernels.py -k speculative
```

Run a complete unseeded eager CUDA frame and verify all codebooks and accumulated codec embeddings:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k unseeded_eager
```

Run persistent verifier graph capture/replay correctness:

```bash
pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k verifier_graph
```

Run the checkpoint-shaped frozen-frame timing gate. It compares four fixed-input frames through the five-layer Predictor, asserts a positive speculative frame count, and fails if any frame falls back. The weights are synthetic; use the serving benchmark below for real-checkpoint timing.

```bash
pytest -q -s -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k frozen_frame_benchmark
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

## H200 serving comparison

Start a native server with speculation disabled:

```bash
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --config examples/configs/qwen3_tts_1_7b_customvoice.yaml --port 8000
```

Start a second server with verifier width 4:

```bash
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --config examples/configs/qwen3_tts_1_7b_customvoice_speculative.yaml --port 8001
```

Compare fixed, unseeded serving requests and require at least parity with native latency:

```bash
python scripts/benchmark_qwen3_tts_speculative_serving.py --native-url http://127.0.0.1:8000 --speculative-url http://127.0.0.1:8001 --request-count 32 --warmup-requests 2 --concurrency 4 --minimum-speedup 1.0
```

The benchmark omits the `seed` request field, reads speculative counters before and after the measured workload, and exits nonzero if the speculative frame delta or packed verifier count is zero, if any measured frame falls back, or if the configured speedup gate is missed. Run the two servers on matched hardware and with the same software, model revision, GPU clocks, and concurrency. End-to-end audio latency includes preprocessing and vocoding; the frozen-frame test isolates the Predictor path.

## Push commands for a personal fork

After adding your fork as a Git remote, push the current branch without opening a PR:

```bash
git remote add fork git@github.com:<your-user>/sglang-omni.git
git push fork HEAD:refs/heads/feat/qwen3-tts-codepredictor-specdec
```

If the `fork` remote already exists, omit the `git remote add` command.

## NVIDIA parity reference

The parity audit was anchored to NVIDIA TensorRT-Edge-LLM revision `513dcd18291e587c80928911a480761352f3c2d8`. Its frame loop applies future RVQ heads to the carried hidden state, verifies the pending token plus drafts in one pass, and commits the correction or bonus as a pending token. NVIDIA's public overview describes the same shared-residual-stream rationale and round semantics in its [CodePredictor speculative decoding guide](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/codepredictor-speculative-decoding.md).

This implementation does not establish NVIDIA-matched performance. The NVIDIA exporter uses a different FP16 model and TensorRT kernel stack, while this Omni path currently accepts single-rank BF16 heads and uses Omni's fused Predictor. H200 correctness, output-distribution parity, graph lifecycle, serving speedup, and audio quality remain required validation gates.

# Qwen3-TTS CodePredictor speculative decoding

## Status

The opt-in model-connected frame runner now has dynamic tail widths, target-only W1 rounds, compact in-kernel acceptance, fused verification input preparation, fused state transition, and a histogram/compaction radix probability candidate. Warp-oriented grouped GEMV, warp-local accumulation GEMV, and frame-level RNG pooling are selectable research candidates. The required persistent full-round graph and real frozen-frame runner are still missing. GPU compilation, correctness, generated-code structure, and speed remain unverified here.

The runner is reachable through `Qwen3TTSTalker.code_predictor_forward`. It first processes the opening pair and samples codebook 1, then generates future-head proposals, verifies the pending code plus drafts in one packed `FusedPredictorLayers` call, applies the target heads once to verified hidden rows, accepts the draft prefix, samples a residual correction or bonus, commits outputs and embeddings, and repeats until all Predictor codebooks are complete. The normal carry-forward path has no standalone correction forward.

Backend `cuda_graph` captures the packed verifier and target heads for supported batch buckets and widths 1 through the backend capacity. It does not capture a complete speculative round. Backend `eager` is an explicit CUDA correctness and debugging path; it still uses the fused Predictor and GPU sampling kernels. Width 0 remains the default and allocates no speculative seed, head, metadata, or graph buffers. Configuration accepts widths 2 through 8; W1 is selected internally for the final target-only tail.

An example opt-in configuration is [qwen3_tts_1_7b_customvoice_speculative.yaml](../../examples/configs/qwen3_tts_1_7b_customvoice_speculative.yaml). The native-off configuration remains [qwen3_tts_1_7b_customvoice.yaml](../../examples/configs/qwen3_tts_1_7b_customvoice.yaml).

## State and cache contract

Let `m` be the number of committed Predictor codes after codebook 0. At frame start, codebook 1 has been committed to output but is pending in the Predictor input; the opening talker-hidden/codebook-0 pair occupies two KV slots, so the logical KV length is `m + 1 = 2`. The carried hidden row is immediately before the pending code.

Each round verifies the pending input and up to `verify_width - 1` drafts. After accepting `k` drafts and committing one correction or bonus, `m` and the logical KV length both advance by `k + 1`. The last new code becomes pending, and its preceding hidden row is selected from the verifier output at offset `k`. The invariant remains `logical_kv_length = m + 1`.

Rows close to the end of a frame can have zero proposals while other rows verify a full window. They still receive one valid query if active; finished rows have zero valid queries. Inactive embeddings, LM-head selectors, positions, and output stores use safe staged values or masks before access. Positions at `-1` do not write KV and do not participate in attention. Rejected physical KV suffixes may remain in cache but are outside the logical prefix and are overwritten before reuse.

Each round uses the already-staged host metadata to choose `min(configured_width, backend_capacity, max_remaining_depth)` without another synchronization. The batch width shrinks as active rows reach the frame tail; finished rows are masked. If the exact-width graph is unavailable, that round runs eagerly at the safe width. Graph keys are bounded by batch bucket and width. Generic attention is not used for packed verification.

## Sampling, RNG, and precision

The speculative probability builder reuses Omni's score packing, temperature scaling, top-k ordering, FP32 normalization, and top-p boundary rule. The production split chunk/merge path remains the default. The selectable `radix_compact` candidate uses 8-bit MSB histogram passes over the full 64-bit score/token key, compacts the boundary set, and sorts only that compacted set. Its CUDA test covers the full Top-K 50/64/128 by C1/C8/C16 matrix; it still requires H200 compilation and runtime validation.

For top-k values up to 32, the production probability kernel reconstructs the native PyTorch gather order and bitonic sort order. For larger top-k values, packed score keys order score ties by token ID, matching the split selector contract. Compact acceptance matches supports inside one Triton program and sorts only the selected draft row for residual lookup; bonus sampling keeps the target candidate order. Invalid mass is reported through the batched round metadata. The grouped Triton LM-head remains the default; `warp_gemv` selects an eight-output CTA candidate, while `warp_accumulate` assigns four output channels per four-warp CTA and accumulates BF16 products into FP32 lane-local sums before a final reduction and BF16 conversion. These candidates may differ from the native GEMM at rounding boundaries. The Triton source shape does not prove vectorized BF16 loads, cross-warp hidden reuse, reduction lowering, shared-memory traffic, or spill behavior; inspect generated code on H200 before making those claims.

To inspect the `warp_accumulate` candidate on H200, call `grouped_lm_head_warp_accumulate_kernel.warmup` with device tensors shaped like the serving checkpoint and the launch values `source_rows`, `head_count`, `vocab_size`, `hidden_size`, `outputs_per_cta=4`, `block_hidden=64`, `grid=(row_count, triton.cdiv(vocab_size, 4))`, and `num_warps=4`. Save the returned compiled specialization's `asm["ptx"]`, `metadata.shared`, `n_regs`, and specialization key. Check whether BF16 loads use vector forms, each output warp keeps accumulation local until the final reduction, shared-memory instructions are limited to layout conversion, and local-memory traffic indicates spills. Inspect the assembled device code with `cuobjdump --dump-sass` and profile global/shared transactions and local-memory traffic with Nsight Compute. Save the PTX, SASS, register count, shared-memory bytes, and profiler report with the checkpoint revision. None of these generated-code checks has been run in this environment.

Ordinary sampled requests are eligible even though the request builder generated internal sampler seeds. Explicit user seeds, deterministic mode, greedy or mixed-sampling batches, unbounded or unsupported top-k settings, missing request identity, unsupported vocabulary or dtype, tensor parallelism, and missing fused capacity use native decoding. A batch containing any explicit-seed or greedy row falls back as a whole. `per_round` RNG remains the default; `frame_pool` precomputes all per-request draw-purpose domains for the frame and indexes them by round and proposal offset. Neither backend uses a shared fixed seed.

## Metrics

`/model_info` includes `predictor_speculative` metrics, including the requested probability, LM-head, and RNG backends. `probability_backend_execution_counts` counts actual probability-builder calls by implementation, and `probability_backend_fallbacks` counts requested radix calls that ran split Top-K instead. `graph_mode` distinguishes `disabled`, `eager`, and `verifier_only`; `verifier_graph_replays` and `eager_verifier_rounds` expose actual round coverage, while the serving benchmark classifies each run as all-graph, mixed, or eager-only. Verifier graph coverage never implies full-round capture. `frames` counts successful request-row frames; `output_code_frames` counts all native and speculative output rows; `committed_codes` includes every Predictor code committed in speculative frames; `round_committed_codes` counts codes committed after the initial codebook-1 sample. Draft, accepted, bonus, rejection, packed forward, valid query, and padded query counts are separate. The effective-width histogram counts batch-rounds. `generic_predictor_forwards` and `standalone_correction_forward_count` must remain zero for completed speculative frames.

`configuration_time_ms` measures speculative setup, including opt-in buffers, head packing, and verifier capture. `graph_capture_time_ms` isolates verifier capture wall time, and `graph_capture_memory_delta_bytes` is the approximate CUDA allocator increase during capture, not total model memory. `native_fallback_fraction` is computed over attempted frame rows. These counters do not establish a full-round graph replay or performance gain.

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

Run the narrow acceptance padding case under Compute Sanitizer before the larger CUDA suite:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k compact_acceptance_top_k50_padding
```

Then run the fused input and state memory checks:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest -q -m accelerator tests/unit_test/qwen3_tts/test_predictor_speculative_cuda.py -k 'fused_verifier_preparation or fused_state_transition or fused_commit'
```

## H200 single-GPU serving comparison

Use one H200 and one local checkpoint snapshot. The sweep starts and stops one fresh server process group at a time, checks port release, runs the mirrored width order `0,2,3,4,4,3,2,0` at concurrency 1, 8, and 16, omits explicit seed fields, and refuses to overwrite a nonempty output directory. Each response is parsed as PCM WAV and checked for nonempty, nonsilent samples. Results include E2E latency, QPS, audio duration, real-time factor, PCM integrity measurements, output code-frame counts, Predictor counters, and per-request measurements. TTFA is explicitly unmeasured because these requests are nonstreaming.

Run from a Linux host with `sgl-omni`, `curl`, `ss`, `pgrep`, `setsid`, and `nvidia-smi` installed:

```bash
MODEL_PATH="$HOME/hf-cache/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe"
RUN="$HOME/results/qtts-specdec-serving-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/benchmark_qwen3_tts_speculative_serving_sweep.sh "$MODEL_PATH" "$RUN" 51231
```

The output root contains one `server.log` and one result JSON per arm under `c1`, `c8`, and `c16`, plus the first native baseline for each concurrency. The repeated final native arm is retained separately for drift inspection. The sweep requires every verifier round to use its width-specific graph and records actual graph/eager coverage. A nonzero exit means an arm failed validation; do not report the partial matrix as a pass. The speculative CUDA graph in this branch is verifier-only, so this sweep does not satisfy the requested full-round graph replay gate.

## Frozen-frame H200 artifacts

The referenced pilot dataset is expected at `~/results/qtts-specdec-r2e0-pilot-001/{capture,dataset}`. Its dataset contract uses `r2e0-depth-transitions-v1` shards with `hidden[N,15,H]`, `projected_embed[N,14,H]`, `logits[N,15,V]`, `codes[N,16]`, `semantic_positions[N,1]`, and six named sampling tensors. The upstream diagnostic kit is pinned to Omni commit `caf13d4d4bd0eab7edb7186675fb7307a5a0ecdc` and checkpoint snapshot `0c0e3051f131929182e2c023b9537f8b1c68adfe`; its [README](https://github.com/charliechenye/sglang-scripts/blob/exp/qwen3-tts-spec-r2e/experiments/qwen3_tts_spec_r2e/README.md) documents capture/extraction and its source manifest.

Audit the existing dataset on H200 and require the pilot marker before using it:

```bash
source "$HOME/sglang-omni-env.sh"
KIT="$HOME/sglang-scripts/experiments/qwen3_tts_spec_r2e"
RUN="$HOME/results/qtts-specdec-r2e0-pilot-001"
python -u "$KIT/audit_dataset.py" "$RUN/dataset" | tee "$RUN/dataset/dataset-audit.log"
grep -q 'R2E1_PILOT_DATASET_PASS' "$RUN/dataset/dataset-audit.log"
```

The full frozen-frame correctness and benchmark runner is not implemented in this branch. Dataset audit is only an artifact integrity gate; it does not compare this branch's native/eager/verifier/full-round paths. The current checkout has no pilot data at the referenced user path, and this branch has no full-round graph backend, so no full-frame H200 result can be claimed yet.

## Push commands for a personal fork

The configured `origin` remote is the personal fork. Push the current branch without opening a PR:

```bash
git push origin HEAD:refs/heads/feat/qwen3-tts-codepredictor-specdec
```

## NVIDIA parity reference

The parity audit was anchored to NVIDIA TensorRT-Edge-LLM revision `513dcd18291e587c80928911a480761352f3c2d8`. Its frame loop applies future RVQ heads to the carried hidden state, verifies the pending token plus drafts in one pass, and commits the correction or bonus as a pending token. NVIDIA's public overview describes the same shared-residual-stream rationale and round semantics in its [CodePredictor speculative decoding guide](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/codepredictor-speculative-decoding.md).

This implementation does not establish NVIDIA-matched performance. The NVIDIA exporter uses a different FP16 model and TensorRT kernel stack, while this Omni path currently accepts single-rank BF16 heads and uses Omni's fused Predictor. H200 correctness, output-distribution parity, graph lifecycle, serving speedup, and audio quality remain required validation gates.

# Qwen3-TTS CodePredictor speculative decoding

## Current implementation status

The configuration, capability resolver, per-row CPU reference state machine, opt-in fused scratch sizing, explicit-seed request marker, observability shape, and row-position attention interface are present on this branch. The speculative CUDA round runner is not implemented. A nonzero request therefore logs `speculative-graph-unavailable`, reports effective width 0, and uses the native Predictor path. Unseeded requests do not yet enter speculation because no speculative execution path exists.

The default remains disabled. Width 0 allocates no speculative buffers. Width 1 is invalid. Widths 2 through 8 are accepted as requests, but currently fall back before speculative frame state is created.

Example pipeline YAML:

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
code_predictor_spec_verify_size: 4
```

The current model-info payload includes `predictor_speculative.requested_width`, `effective_width`, `fallback_reason`, counters, and `native_fallback_count`. With width 4 today, expect effective width 0 and native output.

## State and cache invariants in the CPU reference

The initial state contains codebook 0 and the natively sampled codebook 1; the Predictor KV prefix contains the opening pair and codebook 1 is pending. For proposal count `n`, the packed verifier receives the pending token followed by up to `n` drafts and returns target distributions for all `n + 1` query rows.

Let `m` count committed Predictor codes after codebook 0. The opening state has `m = 1` because codebook 1 is committed but pending, and its valid KV length is `m + 1 = 2` for the talker-hidden/codebook-0 opening pair. After accepting `k` drafts, the next state commits `k + 1` Predictor codes and advances the valid KV prefix by `k + 1`; the invariant remains `valid_kv_length = m + 1`. The correction or bonus token is committed to output but remains pending. The carried hidden state is the verifier hidden row at index `k`, immediately before that pending token. Active rows with no proposals have one valid query row. Finished rows have zero proposals and zero valid queries and do not advance.

The CPU reference implements inverse-CDF sampling, first-rejection acceptance, normalized `max(p-q, 0)` correction, the trailing bonus, a deterministic lower-token-ID tie policy, and the current nucleus-filter boundary rule. That lower-ID policy is a generic CPU reference; it is not evidence of CUDA native tie parity. The reference also accepts a caller-supplied native candidate order so a CUDA selection result can be checked against the CPU FP64 probability calculation. No production CUDA probability output is wired to supply that order yet. An empty residual falls back to `p`; under valid uniforms that case cannot be reached by a mathematically valid rejection when `p == q`.

## Supplemental request-seed and capacity audit

`Qwen3TTSSGLangRequestData.user_seed_provided` preserves `Qwen3TTSState.seed is not None` through request construction. Internal semantic and subtalker seeds remain present for ordinary unseeded requests, so their presence is not used to classify explicit-seed requests. Decode-buffer staging keeps this flag aligned with request rows and restages it on reorder or request-ID reuse. `derive_qwen3_tts_speculative_draw_seed` creates a stable draw key from the internal row seed, request ID, semantic position, RVQ depth, round, purpose, and draw index. The helper is not connected to a graph-safe GPU RNG; all requests still use native decoding because there is no speculative runner.

`fused_predictor_scratch_rows` sizes opt-in packed scratch to `min(max_batch_size * verify_width, MAX_FUSED_ROWS)`. The default width of two preserves the existing opening-pair capacity, and width zero creates no additional scratch. A nonzero requested width can expand scratch before native Predictor graph capture. `resolve_fused_predictor_width` uses the largest configured graph bucket, the candidate allocation, the GQA group, and the attention tile to reduce unsupported widths before allocating. Register pressure and architecture-specific resource limits have not been validated. These sizing helpers do not establish a production effective width until a speculative graph runner consumes them. Since the runner is absent, an opt-in width can allocate scratch and still fall back to native decoding.

## Existing native sampler and probability parity

The Qwen3-TTS native path in `sglang_omni/models/qwen3_tts/sampling_kernels.py` first tries `sample_from_logits_with_seed_top_k_top_p`. Its single-kernel path packs FP32-scaled score bits with token IDs and selects candidates with `tl.topk`; for `max_top_k <= 32`, it reconstructs the selected order used by the pinned PyTorch `topk` gather and the 32-entry bitonic sort. The SM90 split path uses `seeded_top_k_chunk_kernel` followed by `seeded_top_k_merge_sample_kernel`. The selected scores are normalized in FP32, nucleus filtering removes a token when `cdf - probability >= top_p`, and seeded Gumbel sampling hashes the row seed, semantic position, and candidate rank. `sample_codes_and_gather_codec_embedding` fuses embedding accumulation only for its supported split-sampler contract.

These kernels return sampled IDs and optionally gather embeddings; they do not expose the row distribution needed for `p(x)`, `q(x)`, or `max(p-q, 0)`. The current CPU lower-ID tie policy is therefore insufficient for exact speculative parity. A speculative probability kernel must reuse the native selection order, FP32 temperature/filtering arithmetic, and top-p boundary rule. A separate radix selector is not implemented or compared with the native split top-k path. No exact distribution-equivalence claim is made.

## NVIDIA-to-Omni parity matrix

The reference is NVIDIA TensorRT-Edge-LLM revision `513dcd18291e587c80928911a480761352f3c2d8`. Source paths below are relative to that repository.

| NVIDIA implementation | Omni status on this branch | Notes |
|---|---|---|
| `cpp/multimodal/qwen3_omni/codePredictor.cpp:resolveSpeculation` | Implemented but needs extension | Width validation and a startup capability resolver exist; GPU round resources and effective execution are still missing. |
| `runSpecDecodeFrame` | Implemented but needs extension | `sglang_omni/utils/predictor_speculation.py` defines the CPU round and frame reference; there is no model-connected CUDA runner. |
| `uploadRowSelectors`, `applyRowLmHeads` | Missing | Per-row hidden/head selector buffers and grouped LM-head projection are not implemented. |
| `buildVerifyInput`, `invokeGatherCodecEmbedRows` | Implemented but needs extension | Existing Qwen3-TTS has codebook-specific embedding gathers and preprojected tables. There is no packed per-depth input gather or committed-prefix output kernel. |
| `executeVerifyStep` | Implemented but needs extension | `FusedPredictorLayers` now accepts absolute per-query positions and masks padded rows; no speculative call site stages those positions. CUDA parity is unverified. |
| `cpSpecTopKTopPProbs`, `cpSpecSampleRows` | Missing | The CPU FP64 probability reference exists. No production p/q distribution or independent speculative RNG kernel exists. |
| `cpSpecProbabilisticAccept` | Implemented but needs extension | The CPU reference covers first rejection, correction, bonus, zero proposals, and edge cases. There is no GPU implementation. |
| `storeCodecHiddensForAccepted`, `commitSequenceLength` | Implemented but needs extension | The CPU state machine commits only accepted outputs and tracks logical prefix length. Persistent GPU commit buffers and KV-length staging are absent. |
| `dsparkFillUniforms` | Missing | No graph-safe request-isolated speculative RNG stream is integrated. |
| `captureDecodeGraphs` speculative verify graph ladder | Missing | Native graph capture is unchanged. There is no speculative graph owner or capture/replay path. |
| Generation-profile widening for verify windows | Not applicable to Omni | Omni uses a frame-local Predictor cache with a fixed code-group bound, not TensorRT generation token profiles. |

NVIDIA call-site map at the pinned revision: `runSpecDecodeFrame` calls `buildVerifyInput`, `executeVerifyStep`, `applyRowLmHeads`, `cpSpecTopKTopPProbs`, `cpSpecSampleRows`, `cpSpecProbabilisticAccept`, `storeCodecHiddensForAccepted`, and `commitSequenceLength`. `resolveSpeculation` configures widths and buffers. `uploadRowSelectors` prepares grouped head selectors. `invokeGroupedHeadLinear` is defined in `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu` and declared in `cpp/kernels/talkerMLPKernels/talkerMLPKernels.h`. `invokeGatherCodecEmbedRows` is in the same kernel pair. `dsparkFillUniforms` is in `cpp/kernels/speculative/dsparkKernels.cu` and declared in its header. Sampling kernels are in `cpp/kernels/speculative/cpSpecKernels.cu` and `.h`.

The reference execution also widens CodePredictor generation profiles in `cpp/builder/llmBuilder.cpp` and captures decode graphs from the builder's `captureDecodeGraphs` path. The two model export files inspected are `tensorrt_edgellm/builder/models/qwen3_tts/modeling_qwen3_tts_code_predictor.py` and `tensorrt_edgellm/builder/models/qwen3_omni/modeling_qwen3_omni_code_predictor.py`.

## Quantization and hardware gap

The pinned NVIDIA Qwen3-TTS export keeps the stacked codec embeddings and LM heads in FP16 in `tensorrt_edgellm/builder/models/qwen3_tts/embeddings.py`. The Qwen3-TTS `weights.py` loader discards its `quant_type` argument for this component. The Predictor model files declare FP16 LM-head inputs and select FP8 KV cache only when `kv_cache_quant == "fp8"`; otherwise KV is FP16. The model emits logits as FP32 after its selected LM-head projection.

The inspected Qwen3-TTS CodePredictor files do not establish an NVFP4 CodePredictor path. The Qwen3-Omni talker has separate MoE quantization/export logic; it is not evidence that Qwen3-TTS Predictor heads or embeddings run NVFP4. NVFP4 kernels and TensorRT's grouped GEMV/attention implementation remain backend-specific. H200 must not be described as providing Blackwell NVFP4 acceleration.

This branch makes no quantization change. Its existing fused Predictor path is single-rank BF16 and has backend-specific rounding behavior documented in `sglang_omni/utils/predictor_layers.py`. The new position-mask behavior has not been validated on H200. FP8 KV and NVFP4 remain independent work with separate output-quality and performance gates.

## H200 validation commands

Run the CPU reference tests and packed attention kernel tests from the repository root:

```bash
pytest -q tests/unit_test/qwen3_tts/test_predictor_speculation.py
pytest -q tests/unit_test/qwen3_tts/test_predictor_kernels.py -k packed_predictor_attention_masks_per_row_absolute_positions
```

The first command is CPU-only. The second requires CUDA and Triton; it explicitly skips on unsupported hosts. A skip is not a GPU pass.

To check the requested-width plumbing and current fallback on an H200, start a server from a YAML file containing the example above:

```bash
sgl-omni serve --config /tmp/qwen3-tts-spec-width-4.yaml --port 8000
```

The log and `/model_info` should report effective width 0 and `speculative-graph-unavailable`. This validates fallback only; it is not a speculative benchmark.

No matched speculative H200 benchmark command is ready because the round executor is missing. The existing SeedTTS serving benchmark can compare a future completed implementation at C1/C8/C16 with identical model and sampling settings, but running it against this branch now would measure the native fallback in every case. Do not report those timings as speculative performance.

## Known blockers

The remaining P0 item is the missing CUDA execution path: grouped future/target heads, compact native-parity probability construction and speculative sampling, request-isolated graph-safe RNG, packed codec gathers, logical KV commit staging, and a graph-owned speculative round runner are not connected to `code_predictor_forward`. Ordinary unseeded requests therefore cannot enter a speculative path yet. The H200 hidden/logit/KV parity, graph lifecycle, mixed native/speculative stress, full serving benchmark, compute-sanitizer checks, and audio-quality checks have not run. No production-readiness or speedup claim is justified.

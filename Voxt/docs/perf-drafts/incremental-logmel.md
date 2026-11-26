# Qwen3-ASR incremental LogMel

Status: FRONTEND_VALIDATED_MODEL_VALIDATION_PENDING.

The cache reuses FFT power rows only. Each growing snapshot reruns the complete mel projection, logarithm, global clipping and normalization because suffix-shaped MLX matrix multiplication differed from the full reference by one or two ULPs. An identical PCM snapshot returns cached normalized features.

Reuse requires an exact PCM prefix match, the same final-frame layout, compatible cached shapes and float32 tensors. Frames are reused only when their centered window's right half lies inside the PCM. Audio shorter than the 400-sample FFT window has no stable frames. A changed recording start clears Final's cache; other incompatible inputs fall back to full extraction.

The parity test passed on pinned MLX 0.32.5 CPU for both layouts and all deterministic append, padding, normalization, reset and 8-second boundary cases: maximum absolute error 0, unequal elements 0. The native Qwen3-ASR library and parity target compile against the CPU build. Metal parity, GPU timing and model-level validation have not run.

The paired CPU benchmark covered 1, 5, 10, 20, 30 and 60 seconds, 100/500/1000 ms appends, synthetic speech-like/quiet/changing-amplitude audio, and the local English fixture repeated to each duration. Median candidate-to-baseline cumulative refresh time across audio types and append intervals was 0.95x, 0.70x, 0.62x, 0.57x, 0.55x and 0.54x by duration. At 60 seconds, median P50 and P95 refresh latency were both about 0.53x baseline.

For 60-second speech-like audio with 100 ms appends, cumulative frontend time was 1.123 s baseline and 0.588 s candidate; final refresh was 4.26 ms and 1.94 ms. The candidate evaluated 6,600 FFT frames versus 1,803,600. A repeated-snapshot call took 0.238 ms, including a 0.228 ms exact PCM comparison. Retained cache data was 11,736,000 bytes (PCM, power spectrum and normalized features); transient peak memory was not measured. In that 60-second case, candidate host preparation P50 was 0.207 ms and MLX evaluation/synchronization P50 was 0.759 ms. Individual lazy FFT, mel and normalization stages were not isolated.

## Model-level commands (not run)

Set `NATIVE_RUNTIME_BIN`, `CI_DATA_ROOT`, and `OUTPUT_DIR` for one arm; run once using binaries built from base `da2cadcf72b818073f9911e14744f487bdb8d662`, then again using candidate binaries. Keep the data root, hardware, and pinned model files the same, and use separate output directories for the two arms. The existing native golden runner records text, language, generated token count, finish reason and corpus WER/CER/mixed error metrics. These cases cover 0.6B and 1.7B, greedy and fixed-policy DFlash, using the frozen English/Chinese/mixed corpus:

```sh
for golden in \
  sglang_omni_mlx/native/ci/golden/qwen3-asr-0.6b-4bit.json \
  sglang_omni_mlx/native/ci/golden/qwen3-asr-0.6b-4bit-dflash.json \
  sglang_omni_mlx/native/ci/golden/qwen3-asr-1.7b-8bit.json \
  sglang_omni_mlx/native/ci/golden/qwen3-asr-1.7b-8bit-dflash.json
do
  python sglang_omni_mlx/native/ci/check_golden.py \
    --runtime-bin "$NATIVE_RUNTIME_BIN" \
    --data-root "$CI_DATA_ROOT" \
    --golden "$golden" \
    --outputs "$OUTPUT_DIR"
done
```

Compare the recorded outputs with `diff -ru "$BASELINE_OUTPUT_DIR" "$CANDIDATE_OUTPUT_DIR"` after both runs.

The realtime and authoritative HTTP Final path is exercised by the existing 0.6B API tests. Run on both builds:

```sh
NATIVE_RUNTIME_BIN="$NATIVE_RUNTIME_BIN" CI_DATA_ROOT="$CI_DATA_ROOT" \
  pytest -q -s sglang_omni_mlx/native/ci/test_server_api.py \
  -k 'voxt_final_request_streams_text_and_generation_metadata or segment_final_reuses_its_live_session_cache or louder_frame_after_the_last_window_rebuilds_the_prefill'

NATIVE_RUNTIME_BIN="$NATIVE_RUNTIME_BIN" CI_DATA_ROOT="$CI_DATA_ROOT" \
  pytest -q -s sglang_omni_mlx/native/ci/test_server_api.py \
  -k a_draft_model_serves_the_voxt_final_request

cmake --build "$BUILD_DIR" --target qwen3_asr_auto_language_test
NATIVE_RUNTIME_BIN="$NATIVE_RUNTIME_BIN" CI_DATA_ROOT="$CI_DATA_ROOT" \
  NATIVE_QWEN3_ASR_AUTO_LANGUAGE_TEST="$NATIVE_RUNTIME_BIN/qwen3_asr_auto_language_test" \
  pytest -q -s sglang_omni_mlx/native/ci/test_server_api.py \
  -k 'prefixed_segment_final_preserves_output_with_rollback_guesses or incorrect_auto_language_guess_does_not_replace_final_output or auto_language_final_guess_requires_an_eligible_live_preview'
```

The API harness does not parameterize its realtime server target to 1.7B; the golden runner covers 1.7B offline decoding but does not exercise incremental refresh or Final reuse. Dictionary/no-dictionary and a complete fixed/automatic-language realtime matrix also need additional model-capable coverage. No model-level test was executed here.

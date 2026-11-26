# Qwen3-ASR incremental LogMel

Status: FRONTEND_VALIDATED_MODEL_VALIDATION_PENDING.

The cache reuses FFT power rows only. Each growing snapshot reruns the complete mel projection, logarithm, global clipping and normalization because suffix-shaped MLX matrix multiplication differed from the full reference by one or two ULPs. An identical PCM snapshot returns cached normalized features.

Reuse requires an exact PCM prefix match, the same final-frame layout, compatible cached shapes and float32 tensors. Frames are reused only when their centered window's right half lies inside the PCM. Audio shorter than the 400-sample FFT window has no stable frames. A changed recording start clears Final's cache; other incompatible inputs fall back to full extraction.

The parity test passed on pinned MLX 0.32.5 CPU for both layouts and all deterministic append, padding, normalization, reset and 8-second boundary cases: maximum absolute error 0, unequal elements 0. The native Qwen3-ASR library and parity target compile against the CPU build. Metal parity, GPU timing and model-level validation have not run.

The paired CPU benchmark covered 1, 5, 10, 20, 30 and 60 seconds, 100/500/1000 ms appends, synthetic speech-like/quiet/changing-amplitude audio, and the local English fixture repeated to each duration. Median candidate-to-baseline cumulative refresh time across audio types and append intervals was 0.95x, 0.70x, 0.62x, 0.57x, 0.55x and 0.54x by duration. At 60 seconds, median P50 and P95 refresh latency were both about 0.53x baseline.

For 60-second speech-like audio with 100 ms appends, cumulative frontend time was 1.123 s baseline and 0.588 s candidate; final refresh was 4.26 ms and 1.94 ms. The candidate evaluated 6,600 FFT frames versus 1,803,600. A repeated-snapshot call took 0.238 ms, including a 0.228 ms exact PCM comparison. Retained cache data was 11,736,000 bytes (PCM, power spectrum and normalized features); transient peak memory was not measured. Host preparation and MLX evaluation/synchronization were timed separately, but individual lazy FFT, mel and normalization stages were not isolated.

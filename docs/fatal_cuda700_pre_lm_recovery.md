# P0 — Pre-LM encoder fatal CUDA700 fail-fast

**Production recovery-safety change. This does not repair the initial GPU fault.**

## Reproduction and rationale

P39 H200 Nemotron Full C16, Graph OFF: W (Runtime flags=0) passed, V
(flags=1) raised CUDA700 in both uninstrumented and CPU-flight runs.

The LM scheduler first reported CUDA700 at output token ids.tolist()
(cuMemcpyDtoHAsync_v2). The audio encoder reported a second CUDA700
at mel_lengths.max().item() (cuStreamSynchronize). The inherited
Qwen3ASRPreLMEncoderService then logged "retrying per item" for a two
item batch after the fatal GPU error. The stage later SIGSEGVed in
CUDACachingAllocator::process_events while attempting Tensor.to().

Neither the earliest host detection site nor the later allocator
SIGSEGV proves the faulting CUDA kernel or a use-after-free root cause.

## Scope of fix

1. Recognize known illegal-memory-access CUDA error 700 markers before
   traceback detachment and before batch -> per-item retry.
2. Do not run any further CUDA operation with the poisoned context.
3. Emit a low-level stderr marker; terminate the STAGE process with
   os._exit(70), allowing its external supervisor/coordinator to detect
   stage death and fail in-flight requests.
4. Preserve existing retry for ordinary audio/validation errors and
   existing cleanup behavior for ordinary OOM.

No changes to STFT, CUDA streams, allocator policy, cuFFT plan cache,
model weights, numerical output, request routing, or decoder behavior.
Nemotron inherits the Qwen3ASR encoder-service class and gets the same
fatal-error guard.

This is a process failure policy, NOT in-process recovery and NOT a
promise of automatic CUDA restart. The deployment's supervisor remains
responsible for a fresh process after the stage exits.

## Changed files

- sglang_omni/scheduling/fatal_gpu_context.py
- sglang_omni/models/qwen3_asr/encoder_service.py
- tests/unit_test/qwen3_asr/test_fatal_cuda700_failfast.py

## CPU-only regression tests

~~~bash
git fetch origin fix/prelm-fatal-cuda700-failfast
git switch -c fix/prelm-fatal-cuda700-failfast \
  --track origin/fix/prelm-fatal-cuda700-failfast

python -m pytest -q \
  tests/unit_test/qwen3_asr/test_fatal_cuda700_failfast.py \
  tests/unit_test/qwen3_asr/test_encoder_service.py \
  tests/unit_test/nemotron_transcribe/test_encoder_service.py
~~~

The tests monkeypatch process termination during in-process service
checks, and separately verify real exit code 70 in a subprocess.
No GPU or root privileges required.

## Safeguards

- Only narrow CUDA illegal-address error markers in runtime failures
  are fatal; invalid input strings and OOM remain nonfatal.
- Fatal CUDA700 is handled before original exception detachment.
- No per-item retry, GPU cleanup, synchronization or further use of
  the CUDA context once fatal has been detected.
- This branch is deliberately independent from P40A encoder-only
  root-cause experiments.
- Require CI and code review before merging; do not merge based solely
  on CPU tests.

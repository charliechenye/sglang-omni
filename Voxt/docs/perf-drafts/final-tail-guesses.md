# Draft scope: verified HTTP Final tail guesses

Coordination only; this is not an implementation or a performance claim. Replace this temporary scope note with the implementation and tests before marking the PR ready.

Related: charliechenye/sglang-omni#94 (A3), sgl-project/sglang-omni#4650 and #4707.

## Ownership

- Default authoritative HTTP Final **after committed Voxt pieces** only. The existing `done == 0` guard in `Voxt/internal/omni/live_final.go` excludes this path from preview guesses.
- Inspect native Qwen3-ASR HTTP request preparation and compatible realtime preview snapshots. Require recording/session identity, exact audio sample range, language/header and tokenization compatibility before proposing guesses.
- Verify every proposed token with the existing target-model path, including DFlash on/off. Fall back to the existing Final when provenance is unavailable or mismatched.
- Do not change Segment Final ownership/cache snapshot behavior (#4707), LogMel, worker scheduling, refresh cadence, or segmentation. No string-prefix heuristic for audio alignment.

## Acceptance

- Exact authoritative transcript, language, generated token count and finish reason under fixed deterministic settings; wrong/stale guesses, failed piece, mixed language, multiple pieces and session replacement.
- A/B real Mac: HTTP Final latency and guess acceptance by audio range; Stop-to-visible TTLC and no quality loss. No measured gain yet.

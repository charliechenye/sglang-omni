# Draft scope: Final-aware native ASR scheduling

Coordination only; not an implementation or performance claim. Replace this temporary scope note with the implementation and tests before marking the PR ready.

Related: charliechenye/sglang-omni#94 (A2/A4), sgl-project/sglang-omni#4700 and #4707.

## Ownership

- Native MLX worker queue and Qwen3-ASR realtime/authoritative Final dependency handling. Attribute queue wait and active preview/prefill work by session and job type.
- Cancel only obsolete queued work after a Final request. Preserve useful encoder/KV/guess-producing work, active claimed authoritative pieces, session isolation and exact audio coverage. Do not interrupt an active MLX operation without proving safe cancellation.
- No frontend/LogMel computation change, cache ownership redesign, tail guess policy, preview cadence, or Go EarlyClean/LLM changes. Coordinate overlapping realtime call sites with #4707 and the other drafts.

## Acceptance

- Queue fixtures: Final vs obsolete preview, useful prefill, active authoritative piece, cancellation/Close, multiple sessions and starvation prevention.
- Deterministic native transcript/language/finish/token parity, cache invariants, real Mac queue and execution attribution and Stop-to-visible TTLC. No speedup claimed yet.

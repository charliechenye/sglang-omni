# Draft scope: HeadStart trigger sensitivity

Coordination only; not an implementation or performance claim. Replace this temporary scope note with experiment code/tests before marking the PR ready.

Related: charliechenye/sglang-omni#94 (H2/P0-L1), sgl-project/sglang-omni#4618 and #4655.

## Ownership

- Voxt session-layer speculative remote cleanup trigger only. Keep the current 500 ms silence trigger as the default control.
- Experiment with 300/150 ms, stable-text event trigger and optionally adaptive threshold. Use a bounded opt-in configuration and existing generation/request limits.
- Preserve the current exact/near-match, repair, prompt, thinking, fallback, multipart, rate-limit and final delivery semantics.
- Do not change ASR preview cadence, native scheduler, EarlyClean drain or cleanup matching. Keep #4655 late-candidate behavior intact.

## Acceptance

- Deterministic fixtures covering silence/resumed speech, self-corrections, short recordings, caps, duplicate requests, cancellation and stale candidate fences.
- Replay causal ASR events and real Mac E2E. Compare Stop-to-visible and speech-to-visible TTLC, reuse, requests/tokens, 429s, and cleanup quality. Include speech-end-to-Stop 0/100/300/800 ms where actual traces allow. No default policy change without evidence.

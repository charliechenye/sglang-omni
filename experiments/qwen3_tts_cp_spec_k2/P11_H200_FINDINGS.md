# P11 H200 findings — sampler attribution and on-policy Hybrid AR/K2

Source SHA of both user-reported H200 experiments: `50dae7a07484dd82c6d200a818569be705ea89e7`.
GPU: NVIDIA H200. This is an **evidence summary of user-provided Markdown outputs**; original H200 raw JSON, server logs and WAV bytes were not fetched into this repository during this update. Do not treat this file as replacement raw evidence. PR remains draft.

## P11A — real p/q sparse sampler attribution

Reported gate: **`P11A_REAL_K2_PQ_SAMPLER_ATTRIBUTION_PASS_NOT_E2E`**.

Existing complete optimized Graph ON Predictor: **2.23680 ms/frame**.

| Real on-policy residual depth | 2× Torch topk + float/temperature (ms) | Triton post-topk (ms) | private RNG (ms) | complete sparse Graph (ms) | same-worker P10B sparse (ms) | P9B dense (ms) | selected-state p/q overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.06103 | 0.00484 | 0.01007 | 0.07040 | 0.07040 | 0.11125 | 0.00% |
| 10 | 0.05215 | 0.00481 | 0.01004 | 0.06172 | 0.06236 | 0.10219 | 73.42% |

Topk-pair independent graph time / complete graph time = **86.69% (depth2)**, **84.49% (depth10)**. This is *only a ratio of independently captured CUDA Graph launch envelopes*; components overlap and their timings cannot be added to reconstruct kernel busy time. The topk-pair cell includes dtype cast to FP32 and temperature scaling; do not present these numbers as pure PyTorch topk kernel busy time.

**Critical causal-path correction for any P12 design:** In `code_predictor_spec.py:speculative_k2()`, q logits are computed and sampled **before** packed target verification, because the proposal embedding is required to feed the verifier. Target p logits only exist **after** the verifier. Hence a single fused dual-topk operation consuming both logits simultaneously is **NOT feasible on the real K2 dependency path**. P11A's simultaneous frozen p/q input is an attribution-only micrograph. Future P12 must optimize *one distribution at a time*: q selection/proposal before the verifier, p topk and acceptance/residual after; persist q's top-50 probabilities/IDs for the second stage, without CPU sync. Existing `cutedsl/sampling.py` and `sampling_kernels.py` can inform fast topk approaches, but the native sampler's single-token output is insufficient for exact p/q acceptance unless sparse mass and IDs are exported. Preserve topk tie handling, top-p, distribution parity, vocab-ID inverse CDF coupling, and native seeded RNG semantics.

As a rough *historical P2-weighted component screen* with unchanged P10A: P10 sampler measured 0.06622 ms/round and would need **<=0.05475 ms/round** to reach the existing 0.30 ms/frame engineering reserve at historical 9.68611 rounds/frame. This is only ~0.01147 ms/round saved and does not cover unknown initial/heads/bonus/KV transition costs or imply an E2E speedup. 0.50 ms/frame reserve would require ~0.03410 ms/round.

## P11B — actual target AR depths 2/3, then K2 from depth 4

Reported gate: **`P11B_REAL_ON_POLICY_HYBRID_AR_K2_ALL_STOP_ACCOUNTING_PASS_NOT_PARITY_OR_SPEED`**.

Actual experiment was **two distinct seeds**, `20261120` and `20261121`. Each pure K2 and each hybrid was repeated **twice at the same seed**, producing identical summary metrics. These repeats verify same-process reproducibility but are **NOT four independent seeds**.

| Seed | Native audio (s) | Pure K2 audio (s) | Hybrid audio (s) | Pure K2 spec rounds/frame | Hybrid spec rounds/frame | Hybrid direct AR/frame | Pure K2 total target calls/frame | Hybrid total target calls/frame |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261120 | 10.960 | 11.440 | 10.400 | 9.868 | 7.947 | 2 | 9.868 | 9.947 |
| 20261121 | 11.680 | 11.280 | 14.400 | 9.542 | 7.895 | 2 | 9.542 | 9.895 |

On-policy proposal acceptance measured in the two **distinct**, repeated visits: pure K2 **46.16%, 50.26%** vs hybrid **56.39%, 56.82%**, but this comparison is **selection-biased**: hybrid removes low-acceptance depths 2/3 from the proposal denominator. It does **not** demonstrate improved acceptance on matched states at later depths. Counts show per-frame total target forward **calls increase** by **0.8%** and **3.7%** respectively; one-token AR vs two-token verifier calls have unequal GPU cost, so this *does not* prove a GPU latency regression or improvement.

The hybrid changes generated audio duration significantly: seed `20261121` **14.40 s** vs native **11.68 s** and pure K2 **11.28 s**. Hybrid increase is **23.3% vs native**, **27.7% vs pure K2** on that seed, despite all outputs naturally stopping. Seed `20261120` hybrid **10.40 s**, shorter than native **10.96 s**. The higher hybrid duration may be caused by different RNG trajectories, a content repetition, prosody/silence, or a decoding/state error; do **NOT** assert which without listening to the audio and ASR/quality evidence.

Reported per-depth aggregate 572 pure K2 frames vs 624 hybrid frames counts **both repeated visits**, not distinct samples; do not treat 572/624 as independent inference observations. Depth2 pure K2 acceptance was 9.4%; depth3 23.9%, consistent with earlier P2 low-early acceptance trend, but replacing those depths with AR is not a free performance optimization.

**P11B decision: NO HYBRID PERFORMANCE SELECTION YET.** Do not use the debug Python-sync HTTP wall times as production latency or QPS. Before adding hybrid to GPU K2, require (a) matched GPU event costs of actual one-token target AR vs two-token verifier, (b) lossless target-distribution / KV continuation parity for the hybrid transition, and (c) transcript WER / repeated-phrase / speaker similarity and natural-stop quality over more independent prompts/seeds. If target calls/per-frame and content quality remain unfavorable, retire the hybrid branch.

## P12 recommendation

1. **P12A — causal single-distribution topk+sampling.** On real p/q states, measure and prototype a fast one-distribution top-50 selection that exports **IDs + normalized probabilities**, separately before and after the packed target verifier, and plugs into the existing sparse Triton accept/residual. Use a strict comparison to the existing source-locked Torch topk output, including ties/top-p, fixed-uniform inverse CDF, residual and native sampler RNG. Benchmark the *causally ordered full sampler path* rather than a simultaneous p/q frozen-input graph.
2. **P12B — hybrid quality/compute gate (only if continued).** First analyze existing 4 unique WAV outcomes per seed (native, K2, hybrid), count repetitions/silence, compare transcript WER and speaker similarity. Independently time real single-token AR and two-token verifier under identical backend, and recompute per-frame/utterance cost. More seeds are necessary; repeated seeded visits alone do not establish distributional quality.

**No complete GPU-resident K2 E2E speedup has been demonstrated.** Keep experimental PR #89 Draft and preserve original GPU artifacts on the H200 host.

# P13 offline acoustic/Whisper timeline — H200-generated original WAVs

**Source:** User-reported P13 output from
\`~/results/qwen3tts-cpspec-k2-p13-pauses-seed20261121-028/summary.md\`.
Original **P11B source SHA**:
\`50dae7a07484dd82c6d200a818569be705ea89e7\`.
Status **\`P13_RMS_SWEEP_AND_WHISPER_WORD_TIMESTAMPS_MEASURED_NOT_CAUSAL_PROOF\`**.
Only first visit per mode for seed \`20261121\`. P13 ran **offline**
on immutable P11B WAVs. The raw CSVs, audio and word-time JSON remain
on the H200 host; this findings file was prepared from the user's
summary, not from independently downloaded or listened-to WAV content.
Do not present it as proof of a particular root cause or speech identity.

## Strong acoustic observation: +3.12s is mainly low RMS

Same prompt / seed: pure K2 **11.28s (142 codec frames)** vs Hybrid AR
depths2+3 then K2 **14.40s (181 codec frames)**. Extra 39 codec frames
at 80ms/frame generate **+3.12s** of real audio, not WAV tail padding.
Whisper transcripts previously matched ignoring capitalization/punctuation,
both **9.52% WER**, with no detected repeated trigrams. That limits
ASR-detectable lexical errors but does NOT prove perfect content,
prosody or speaker similarity.

| RMS threshold | Additional Hybrid WAV | Additional below-threshold audio | Additional above-threshold audio | Additional >=160ms quiet regions duration | Hybrid long regions |
|---:|---:|---:|---:|---:|---:|
| -35 dBFS | +3.12s | +2.94s (94%) | +0.18s | +2.54s | 9 |
| -40 dBFS | +3.12s | +2.58s (83%) | +0.54s | +2.62s | 9 |
| -45 dBFS | +3.12s | +2.32s (74%) | +0.80s | +2.18s | 7 |
| -50 dBFS | +3.12s | +2.30s (74%) | +0.82s | +1.98s | 6 |

Robust under four RMS cutoffs: **at least ~2.30s** of the 3.12s
extension is additional low-amplitude samples at the tested cutoffs.
This is low RMS, not necessarily speech silence/VAD. At -45dBFS,
K2 has three long low-RMS regions, Hybrid has seven. Region length
threshold requested 150ms but quantized to 8 x 20ms = **160ms**.

Longest reported -45dBFS Hybrid intervals:

| Time in Hybrid WAV | Duration |
|---|---:|
| 5.18–5.96s | 0.78s |
| 9.54–10.26s | 0.72s |
| 1.20–1.66s | 0.46s |
| 3.10–3.50s | 0.40s |
| 7.96–8.28s | 0.32s |
| 11.86–12.18s | 0.32s |
| 11.36–11.52s | 0.16s |

The low-amplitude intervals are distributed through the utterance,
not only at its end. The reported longest K2 intervals are
8.06–8.42s (0.36s), 6.60–6.92s (0.32s) and 4.38–4.68s (0.30s).
Do NOT compare same absolute times as equivalent words because audio
durations diverge.

## Whisper word-timestamp decomposition is NOT a trustworthy pause locator

P13 aligned **22 Whisper-estimated words**, obtaining the algebraic
audio-duration delta decomposition:

| Component (Hybrid - K2) | Seconds |
|---|---:|
| Leading time | +0.000 |
| Signed interword gaps | -0.020 |
| Estimated word spans | +3.040 |
| Trailing time | +0.100 |
| Sum | +3.120 |

At first sight +3.04s could appear to be elongation of spoken words,
not pauses. **That conclusion is NOT supported**. Every long low-RMS
interval shown in the P13 summary has context
\`OVERLAPS_ESTIMATED_WORD_TIME\`. Whisper cross-attention/DTW word
boundaries can include quiet acoustic regions inside estimated word
ranges; it reports almost no interword gaps even though thresholded
PCM shows substantial long low-amplitude intervals. The decomposition
is an arithmetic identity of *Whisper's estimated boundaries*, not
a segmentation into objectively speech-active and paused time.
Top reported inflated word-span estimates include:
\`speculative\` +0.60s, \`on\` +0.44s, \`including\` +0.40s,
\`natural\` +0.40s; these cannot be interpreted as verified articulation
elongation without a separate VAD/forced aligner and auditory review.

## Decision: no Hybrid integration; redirect engineering effort to P12A

The Hybrid has **not** shown performance benefit even before sound
length is considered: P11B per-frame forward calls did not decline,
and P12B static five-layer H200 graph events showed one-token AR
**0.13728–0.13761ms** versus packed 2-token verifier
**0.14232–0.14238ms**, only ~3.4% difference per model call.
Additionally, generating 181 rather than 142 codec frames is
~27.5% more frames for the same input prompt in seed20261121.

Illustrative cross-source, **forward-only** total-cost scenario:
applying the mean P12B graph event times for one-token AR and packed
verifier to P11B seed20261121 per-frame call frequencies gives
pure K2 ~**192.9ms/utterance** and Hybrid ~**253.2ms/utterance**
(~**31% higher**), excluding heads, sampler, embeddings, initial
pair, bonus and actual overlapping GPU work. This is NOT a
source-matched true measured E2E cost; do not quote as a benchmark.

**No more Hybrid optimization work is warranted from the present
evidence.** Keep it a debug research branch, not production:
- P11B 2 independent seeds only; Whisper ASR shows no extra words
  for these two seeds, but no audio-quality, pause or speaker parity.
- More Hybrid samples are not justified solely to chase a possible
  speedup while the core compute/call gate is unfavorable.
- If needing to characterize the remaining *audio quality* concern,
  listen to original-PCM excerpts, particularly
  \`seed20261121_hybrid_quiet01_from5.18_to5.96.wav\` and
  \`seed20261121_hybrid_quiet02_from9.54_to10.26.wav\`.
  Optionally use a separate speech-VAD segmentation for truly
  speech-active vs nonspeech intervals before investigating word
  alignment. Check against the *corresponding words*, not fixed
  timestamps. No audio-listening outcome has been supplied yet.
- Continue **P12A single-distribution selector** work: real causal
  q proposal → target verifier → p support → residual accept/bonus,
  native RNG behavior, KV ownership and actual Graph ON E2E cost.
  The optimistic P12A independent q/p stage savings are NOT a
  full causal Graph result.

**PR #89 remains Draft.** Neither Hybrid nor full K2 is approved
for production merge based on the P13 result.

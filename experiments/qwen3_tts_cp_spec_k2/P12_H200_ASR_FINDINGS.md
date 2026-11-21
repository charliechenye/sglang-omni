# P12 H200 — real ASR audit of original P11B Hybrid AR/K2 WAVs

**Evidence scope:** This memo transcribes the user's supplied `p12-quality-asr-024/summary.md` results, not independently downloaded audio or per-utterance `report.json`. The original WAV cohort was generated from source `50dae7a07484dd82c6d200a818569be705ea89e7`; P12 selector/AR GPU profiling was performed on source `ac89ef8a00dc4f94f2c7b1392902ffbfe3b6e501`. Do NOT combine these as a source-matched serving benchmark.

## ASR finally measured

User-observed gate: `P12B_ASR_WER_AND_WAV_STATS_MEASURED_SPEAKER_EXTERNAL_OR_UNMEASURED_NOT_A_HYBRID_SELECTION`.

ASR: **faster-whisper**, English model configured as `small.en`, audio validated against original saved hashes. **Speaker similarity: NOT_MEASURED.** **Target GPU cost projection in this report: NOT_MEASURED.**

Ground truth text from `quality_smoke.py` has **21 English word tokens** under the evaluator's `[a-z0-9']+` regex. WER values of 14.29%, 19.05%, 9.52% are respectively **3, 4, and 2 word-level edit errors** on this short sentence; 1 error is ~4.76 percentage points.

| Seed | Mode | WAV seconds | 20ms RMS silence fraction (-45dBFS) | trailing silence | English ASR WER | Word edit distance of 21 |
|---:|---|---:|---:|---:|---:|---:|
| 20261120 | native | 10.960 | 22.8% | 0.08s | 14.29% | 3 |
| 20261120 | pure K2 | 11.440 | 16.4% | 0.04s | 14.29% | 3 |
| 20261120 | AR depths2/3 + K2 | 10.400 | 19.8% | 0.02s | 14.29% | 3 |
| 20261121 | native | 11.680 | 24.5% | 0.04s | 19.05% | 4 |
| 20261121 | pure K2 | 11.280 | 20.0% | 0.02s | 9.52% | 2 |
| 20261121 | AR depths2/3 + K2 | **14.400** | **31.8%** | 0.08s | **9.52%** | **2** |

**Interpretation:** ASR does NOT show worsened *word-level edit counts* from Hybrid relative to pure K2 in these two seeds; this is a narrower positive result than speech quality parity. Equal WER does NOT prove equal transcripts, no repeated phrases, stable prosody, a stable speaker, identical native sampling distribution, or correct E2E K2 state progression. The baseline native condition has WER 19.05% while K2 and Hybrid have 9.52% for seed20261121; this is not sufficient evidence of a statistically meaningful quality improvement because there are only **two independent seeds**, a very short reference, and same-seed visits are repetitions.

Seed20261121 hybrid audio is **3.12s longer than pure K2** and has **39 more generated codec frames** (181 vs 142 in the P11B run), which at ~80ms/frame accounts for 3.12s. The extension is from longer generation, not WAV trailing padding (tail 0.08 vs 0.02s). Relative to native (11.68s), Hybrid's +2.72s audio duration contains about +1.72s of additional quiet-window RMS time and about +1.00s of non-quiet time. These are threshold-based acoustic approximations, not verified voiced/unvoiced speech regions. Longer pauses, prosody shifts, or late stopping are plausible but remain hypotheses until inspecting transcripts, aligned segment intervals, and listening to WAVs.

### Immediate evidence from existing ASR results — no GPU or model rerun

The source `p12-quality-asr-024/report.json` includes per-visit `transcript`, `wer`, and `repeated_trigrams`. Extract **only first of two repeated visits for each seed/mode**, compare strings and short repeated phrases:

```bash
ASR_VENV="$HOME/.venv-qwen3tts-asr"
"$ASR_VENV/bin/python" - <<'PY'
from pathlib import Path
import json
p=Path.home()/"results/qwen3tts-cpspec-k2-p12-quality-asr-024/report.json"
d=json.loads(p.read_text())
for x in d["items"]:
    if x["tag"].split("_")[1] not in ("00","01","02"):
        continue
    print(f"\\nseed={x['seed']} mode={x['mode']} wer={x.get('wer')}")
    print("transcript:",x.get("transcript","NOT_MEASURED"))
    print("repeated_trigrams:",x.get("repeated_trigrams",[])[:5])
PY
```

If the actual transcript matches closely but the longer Hybrid audio has extended low-RMS intervals, examine timestamped speech regions and pause placement. If real repeated or extra words occur, audit hybrid AR→K2 state/caches and RNG, but do not infer a specific bug from aggregate WER alone.

**Decision:** P12B WAV integrity + ASR WER MEASURED, no obvious WER degradation vs pure K2; speech content/pacing/speaker parity NOT PROVEN. **Hybrid performance remains NO-GO**: source-mismatched P12B H200 one-token AR cost ~0.1373ms vs packed 2-token verifier ~0.1423ms, and two-seed P11B total target calls/frame did not decline. Continue P12A single-distribution selector research and prove fully causal q→verifier→p→accept/RNG/KV behavior before any E2E speedup claim. Keep PR #89 Draft.

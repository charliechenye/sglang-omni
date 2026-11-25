# P13 — Offline acoustic pause and Whisper word-timestamp diagnosis

**Inputs:** The exact immutable P11B native/K2/Hybrid WAVs in
\`~/results/qwen3tts-cpspec-k2-p11b-020/smoke/\`, recorded at original
source SHA \`50dae7a07484dd82c6d200a818569be705ea89e7\`.
The P12 ASR report (if available) under
\`~/results/qwen3tts-cpspec-k2-p12-quality-asr-024/report.json\` is used
only as a same-source transcript consistency reference. P13 does not
mutate audio or run any model generation.

**Reason:** For seed 20261121, K2=11.28s and Hybrid=14.40s, a +3.12s
difference with nearly identical English ASR transcripts. The prior
-45dBFS 20ms RMS audit found 20.0% low-amplitude in K2 vs 31.8% in
Hybrid; this suggests +~2.32s of extra low-amplitude intervals, but
does not identify their exact positions or prove linguistic silence.
Hybrid also had 181 codec frames vs 142 K2 (39*0.08s = 3.12s). Native
11.68s acts as a context control.

**What P13 does:**
- Select exactly the FIRST native/K2/hybrid visit per requested seed
  from the existing 6-visit P11B report. Verify original report PASS
  and source SHA, per-visit typed natural stop, SHA256 and 24k PCM16 WAV
  sample counts/durations before any analysis. Repeated visits are
  NOT independent data.
- Scan **exact non-overlapping 20ms RMS windows** at -35/-40/-45/-50
  dBFS. Group contiguous low-amplitude windows, keeping those with
  at least 150ms requested length (**160ms effective**, i.e., 8 x
  20ms). Report total low-RMS duration, long-region counts,
  long-region duration and exact start/end seconds per mode.
- Optionally run \`faster-whisper small.en\` with
  \`word_timestamps=True\`, \`language=en\`, \`beam_size=5\`,
  \`vad_filter=False\` and \`condition_on_previous_text=False\`.
  The old -024 report does not include word timestamps, so P13 must
  run the **ASR model on the same original WAV again** for timestamps;
  NO TTS generation rerun is needed. Timestamps are cross-attention
  and DTW *estimates*, not an independent forced aligner.
- If K2 and Hybrid have the same normalized ASR word sequence, split
  the +3.12s WAV difference into (a) estimated leading gap, (b)
  signed interword gaps, (c) estimated word spans, (d) estimated
  trailing gap. Signed gaps preserve the exact time identity even
  where word timestamps overlap. A change in Whisper word timestamps
  is **not** proof that speech actually slowed or stopped there.
  If ASR words differ, report \`NO_POSITIONAL_ATTRIBUTION\`;
  do NOT zip misaligned words or invent a time explanation.
- Export CSVs, report.json, summary.md and short **original-PCM**
  clips around the longest -45dBFS quiet intervals, padded by 0.25s
  on either side so they can be listened to locally. Clips are from
  each mode separately; identical absolute time across two modes
  is not guaranteed to correspond to the same word.
- The script does not rank serving QPS, claim Hybrid losslessness,
  estimate speaker similarity or modify P12A source.

## 0. Refresh and run CPU unit tests

No H200 GPU/Omni server is required. Stop any unnecessary TTS serving
work if desired, but the original P11B WAV files remain immutable.

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
git rev-parse HEAD
git status --short

python -m py_compile \
  experiments/qwen3_tts_cp_spec_k2/p13_acoustic_timeline.py \
  tests/unit_test/qwen3_tts/test_cp_spec_p13.py

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p13.py \
  -m "not accelerator"
~~~

CPU tests do not import/require a live faster-whisper model; a fake
Whisper timestamp stub tests source/word alignment. A successful unit
test is NOT proof that real Whisper word estimates are correct.

## 1. Main P13 real-word-timestamp audit, seed 20261121

Use the *independent* \`~/.venv-qwen3tts-asr\` with previously
installed \`numpy\`, \`faster-whisper\`, and pinned PyAV 18.1.0.
The source server's Torch/Triton venv must not be changed.
CPU INT8 is enough; no new GPU benchmark or model generation.

~~~bash
cd ~/sglang-omni
ASR_VENV="$HOME/.venv-qwen3tts-asr"
P13_OUT="$HOME/results/qwen3tts-cpspec-k2-p13-pauses-seed20261121-028"

"$ASR_VENV/bin/python" - <<'PY'
import av
from faster_whisper import WhisperModel
print("ASR_READY", av.__version__)
PY

test -f "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json"
test -f "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-024/report.json"
test ! -e "$P13_OUT"

"$ASR_VENV/bin/python" \
  -m experiments.qwen3_tts_cp_spec_k2.p13_acoustic_timeline \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --p12-asr-report "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-024/report.json" \
  --seed 20261121 \
  --asr faster-whisper \
  --asr-model small.en --asr-device cpu --asr-input file \
  --thresholds=-35,-40,-45,-50 \
  --min-pause-ms 150 \
  --clip-count 5 \
  --out "$P13_OUT"

cat "$P13_OUT/summary.md"
~~~

If \`-028\` already exists, select a NEW output path. The analyzer
fails before creating output if the ASR package/model cannot load,
original WAV SHA mismatches, source report lacks PASS or source
word timings are missing. Do not replace old WAVs or fake the
missing data.

**Alternative when PyAV file decode is incompatible:** install
\`scipy\` only in the independent ASR virtualenv and use
\`--asr-input pcm-array\`. This reads 24k WAV via Python
\`wave\` and converts it into a 16k float32 NumPy array, bypassing
PyAV. This changes the ASR resampling path, so do not silently
merge timestamp results from both decode methods.

~~~bash
uv pip install --python "$ASR_VENV/bin/python" scipy

# In the main command replace only:
#   --asr-input file
# with:
#   --asr-input pcm-array
# Use a NEW P13_OUT directory.
~~~

## 2. Read detailed pause locations and listen to clips

~~~bash
P13_OUT="$HOME/results/qwen3tts-cpspec-k2-p13-pauses-seed20261121-028"
cat "$P13_OUT/summary.md"

# Every 20ms RMS observation per original mode
head -n 12 "$P13_OUT/rms_20ms.csv"

# Exact times of all >=160ms low-RMS regions for every threshold
head -n 40 "$P13_OUT/long_quiet_regions.csv"

# Real model's estimated word timings
head -n 40 "$P13_OUT/whisper_words.csv"

# Excerpts for manual playback, including selected Hybrid pause spans
find "$P13_OUT/clips" -type f -name '*.wav' | sort
~~~

For listening, copy the selected WAV clips from the H200 host to a
local machine using your usual SSH transfer workflow, then listen
to the K2 and Hybrid excerpts around the same *words* using the
Whisper estimated timestamps. Absolute seconds differ when the
utterances stretch; do not assume equal wall-clock offsets align.

The JSON includes exact original source and excerpt SHA256, estimated
word-gap context for each low-RMS region and a word-by-word timing
difference list. Use \`long_quiet_regions.csv\` for threshold sensitivity
and \`whisper_words.csv\` for acoustic/timing inspection.

## 3. Optional second seed 20261120 as a contrast

Keep the first result directory and its source evidence unchanged.
Use a distinct output path (e.g. \`-029\`).

~~~bash
P13_OUT="$HOME/results/qwen3tts-cpspec-k2-p13-pauses-seed20261120-029"

"$ASR_VENV/bin/python" \
  -m experiments.qwen3_tts_cp_spec_k2.p13_acoustic_timeline \
  --p11b-report "$HOME/results/qwen3tts-cpspec-k2-p11b-020/smoke/report.json" \
  --p12-asr-report "$HOME/results/qwen3tts-cpspec-k2-p12-quality-asr-024/report.json" \
  --seed 20261120 --asr faster-whisper --asr-device cpu \
  --asr-model small.en --asr-input file --clip-count 5 \
  --out "$P13_OUT"

cat "$P13_OUT/summary.md"
~~~

If ASR is temporarily unavailable, run \`--asr none\` to get the
RMS/interval/clip analysis ONLY. The summary explicitly returns
\`P13_RMS_SWEEP_ONLY_WORD_TIMESTAMPS_NOT_MEASURED\` and does not
claim word alignment.

## 4. Decision rules

- **Large quiet delta persistent across thresholds:** robust evidence
  for additional low-amplitude audio (not necessarily speech silence).
  Inspect the reported long regions and clips for mid-utterance pauses.
- **Whisper interword gaps grow while word-spans are mostly stable:**
  a *hypothesis* of changed pause timing; compare locations and
  listen to segments. Word times are not ground-truth alignment.
- **Whisper word spans expand:** possible prosody/duration change;
  listen for slow/elongated words. Whisper timestamp uncertainty can
  also explain the apparent difference.
- **Long quiet intervals after final ASR word:** possible delayed
  acoustic stopping; verify with raw excerpts; natural typed EOS and
  codec frame totals alone cannot prove early/late stop mechanism.
- **ASR transcript tokens changed:** stop automatic word-by-word
  attribution; preserve the independent RMS evidence and review
  transcripts manually.
- **No obvious pause/voice difference:** don't invent a causal
  explanation for audio duration. P13 has only two source seeds with
  same-seed repetitions, no population-level quality parity.
- **Regardless of P13 outcome:** Hybrid AR/K2 still lacks a
  performance advantage; keep Hybrid unselected. P12A selector
  and a complete GPU causal q→verifier→p→accept/bonus/RNG/KV path
  remain the separate performance engineering track.

P13 is offline research. No serving-code changes, no new Qwen3-TTS
generation, and no new E2E speedup claim. PR #89 remains Draft.

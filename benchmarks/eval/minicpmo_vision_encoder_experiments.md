# MiniCPM-o vision encoder experiment harness

This branch starts at `a266a0894d8964ca592a86cb198b446acef8908a` on
`perf/minicpm-o-video-resize-threadpool-pr-ready`. All changes are benchmark
utilities. Production preprocessing and encoder code stay unchanged.

## Run on the H200

These commands load real model weights and execute CUDA experiments. They are
for Charlie's manual run, after installing the baseline's pinned dependencies.

```bash
source ~/sglang-scripts/sglang-omni-env.sh
cd "$SGLANG_OMNI_ROOT"
git fetch origin
git switch exp/minicpmo-vision-transformer-perf
git pull --ff-only origin exp/minicpmo-vision-transformer-perf
bash benchmarks/eval/run_minicpmo_vision_encoder_experiments.sh
```

Alternatively let the driver source the environment:

```bash
ENV_FILE="$HOME/sglang-scripts/sglang-omni-env.sh" \
  bash benchmarks/eval/run_minicpmo_vision_encoder_experiments.sh
```

Select an idle GPU using your existing `CUDA_VISIBLE_DEVICES`. The driver uses
the current environment unless `ENV_FILE` is supplied. Its editable variables
include `MODEL_ID`, `VIDEOMME_REPO`, `RESULT_ROOT`, `DEVICE`, `DTYPE`, `PYTHON`,
`WARMUP`, `ITERS`, `RESAMPLER_WARMUP`, `RESAMPLER_ITERS`, `PROFILE_ITERS`,
`RESAMPLER_SAMPLE_SELECTION`, `PROFILE_SAMPLE_SELECTION`, and `COHORT_FILE`.

The model defaults to `openbmb/MiniCPM-o-4_5`; the dataset defaults to
`$SGLANG_VIDEOMME_CI_REPO_ID` or `zhaochenyang20/Video_MME_ci`. To use the full
dataset instead, set `VIDEOMME_REPO="$SGLANG_VIDEOMME_REPO_ID"`. Existing production
loaders resolve the model and dataset; uncached resources may download when
Charlie runs these commands. Set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` before
running if resources must already be cached.

Results go to a unique UTC timestamp directory below
`$SGLANG_RESULTS_DIR/minicpmo-vision` (otherwise the system temporary directory).
There are eleven named logs and JSON summaries plus a profiler table and a Chrome
trace. Output paths inside Git worktrees are rejected.
No tensor caches are written. Each process reuses its real CPU encoder inputs
in memory; preprocessing is outside all timed regions. The saved census lets
later processes decode only the selected video instead of all sixteen.

## Frozen cohort

The harness requires exactly these IDs, with no replacements or duplicates:

```text
002-1 003-1 005-1 006-1 007-1 008-1 009-1 010-1
011-1 012-1 014-1 015-1 017-1 018-1 019-1 020-1
```

The driver prefers
`$SGLANG_RESULTS_DIR/minicpmo-video-frozen-cohort.json` if present. Otherwise it
passes the explicit IDs above. Missing dataset IDs or media files are errors.
An explicitly supplied missing `COHORT_FILE` also fails.

`--cohort-file` accepts a JSON list of IDs, `{"sample_ids": [...]}`, a list of
records, `{"samples": [...]}`, or `{"per_sample": [...]}`. Records contain
`sample_id`, and may contain `video_path` and `prompt`. Extra record fields are
ignored. Complete records bypass dataset loading. Missing paths or prompts are
resolved through `load_videomme_samples()`. Relative manifest video paths resolve
against the manifest's directory. `--sample-ids` is the alternative to a manifest.

All modes call `MiniCPMOPreprocessor` with `video_fps=2`, `video_max_frames=128`,
`video_max_pixels=401408`, and video audio disabled. The entry point forces
`FORCE_QWENVL_VIDEO_READER=torchcodec` before the preprocessing stack is imported.
It consumes the production payload's `pixel_values` and `tgt_sizes` directly.

## Individual modes

Run these from the experiment checkout with the environment already activated.
If you use a custom model, dataset, or manifest, pass the same options to every
command. Without `--census-file`, each measurement mode preprocesses all sixteen
requests to select its representative. With it, file identity and selected
preprocessing shapes are checked before measuring.

```bash
OUT="$(mktemp -d /tmp/minicpmo-vision-XXXXXX)"
BENCH=benchmarks.eval.minicpmo_vision_encoder_microbench

python -m "$BENCH" --mode census --output-json "$OUT/census.json"

python -m "$BENCH" --mode decompose --census-file "$OUT/census.json" \
  --sample-selection median --vision-batch-size 16 \
  --output-json "$OUT/decompose-median-bs16.json"
python -m "$BENCH" --mode decompose --census-file "$OUT/census.json" \
  --sample-selection median --vision-batch-size 32 \
  --output-json "$OUT/decompose-median-bs32.json"
python -m "$BENCH" --mode decompose --census-file "$OUT/census.json" \
  --sample-selection median --vision-batch-size 64 \
  --output-json "$OUT/decompose-median-bs64.json"
python -m "$BENCH" --mode decompose --census-file "$OUT/census.json" \
  --sample-selection high --vision-batch-size 16 \
  --output-json "$OUT/decompose-high-bs16.json"
python -m "$BENCH" --mode decompose --census-file "$OUT/census.json" \
  --sample-selection high --vision-batch-size 32 \
  --output-json "$OUT/decompose-high-bs32.json"
python -m "$BENCH" --mode decompose --census-file "$OUT/census.json" \
  --sample-selection high --vision-batch-size 64 \
  --output-json "$OUT/decompose-high-bs64.json"

python -m "$BENCH" --mode vision-batch --census-file "$OUT/census.json" \
  --sample-selection median --vision-batch-sizes 16 32 64 --warmup 5 --iters 20 \
  --output-json "$OUT/vision-batch-median.json"
python -m "$BENCH" --mode vision-batch --census-file "$OUT/census.json" \
  --sample-selection high --vision-batch-sizes 16 32 64 --warmup 5 --iters 20 \
  --output-json "$OUT/vision-batch-high.json"

python -m "$BENCH" --mode resampler --census-file "$OUT/census.json" \
  --sample-selection median --resampler-warmup 10 --resampler-iters 50 \
  --output-json "$OUT/resampler.json"

python -m "$BENCH" --mode profile --census-file "$OUT/census.json" \
  --sample-selection high --profile-iters 2 --profile-dir "$OUT/profile" \
  --output-json "$OUT/profile.json"
```

Use `--sample-id 002-1` instead of `--sample-selection` to select an explicit
frozen request. Automatic selection minimizes distance to the cohort's p50 or
p95 **attention work proxy**, defined as `sum(patch_count ** 2)` across slices;
this is only an approximate proxy for the quadratic self-attention component
across independently segmented slices, not a FLOPs or runtime claim. Ties use
sample ID. `total_patches` remains in every census entry and summary for context.
The selected ID, metric, target work, actual proxy, total patches, and reason
print before the measurements.

Read `total_patches` as a token-linear volume proxy and
`attention_work_proxy` as the attention-quadratic proxy; neither is a FLOPs or
runtime measurement.

- `census`: CPU preprocessing only. Reports all target sizes, per-slice patches,
  total patches, the integer attention work proxy, their distributions, VPM chunk counts for 16/32/64, and the five most
  frequent geometries for each arm, including their frequency/share. Geometry
  includes chunk slice count, request-wide padding, and ordered segment lengths.
  Prints checkpoint configuration; backend and strided QKV require an initialized
  encoder and are explicitly reported as uninspected here.
- `decompose`: Executes the production forward with temporary module hooks and a
  wrapper around `run_vpm`. CUDA events split input preparation (H2D, padding,
  packing, metadata, and concatenating VPM chunks), VPM embeddings, Transformer,
  post-LayerNorm/unpack, and resampler (including final concatenation). Chunk
  spans are summed by phase. Only the whole-forward end event is waited on.
  Reports phase distributions, whole-forward GPU and CPU wall times, memory,
  and parity against an uninstrumented forward. `--vision-batch-size` records
  the selected 8/16/32/64 value in the heading, trial, and JSON report.
- `vision-batch`: Visits the same cached input in the order
  `bs16-A, bs32-A, bs64-A, bs64-B, bs32-B, bs16-B`. Every visit has independent
  warmup, timed iterations, and peak-memory reset. JSON includes each visit,
  pair averages for bs16/32/64, and repeat spread; candidate deltas use the bs16
  pair average.
- `resampler`: Captures every real bs16 VPM-output chunk using a temporary
  resampler input hook. R0/R1 measure the **complete request's resampler stage**,
  including all production chunks and output concatenation, with VPM and H2D
  excluded. Visits are `R0-A, R1-A, R1-B, R0-B`; pair averages, repeat spread,
  and candidate deltas are reported. R1 adds only `need_weights=False` via a
  reversible MHA pre-hook; weights, masks, and the resampler implementation are
  unchanged. If the **paired** R1 result passes parity and improves paired median
  latency, `V0-A, V1-A, V1-B, V0-B` measure the complete encoder at bs16.
  Frozen GPU VPM inputs are released before this full-encoder A/B.
- `profile`: Warms up then profiles 1–3 baseline forwards with CPU/CUDA activities
  and `record_shapes=True`. Saves a table sorted by CUDA time and, by default,
  a Chrome trace (`--no-chrome-trace` disables it). Module ranges attribute CUDA
  time to QKV, the actual attention backend, output projection, norms, fc1,
  GELU, fc2, pointwise work, patch embedding, and resampler components. Contributor
  times are kernel time per forward; inspect the trace for exact kernel names.

GPU-mode startup prints effective and configured layer counts, hidden/intermediate
sizes, heads, patch size, queries, dtype/device, actual `qkv_backend_name`,
`pass_strided_qkv`, fused-QKV status/shape, and PyTorch/SGLang versions. It does
not select an attention backend.

## Reading the measurements

`mean_ms`, `median_ms`, `p95_ms`, and `min_ms` are whole-forward CUDA-event elapsed
times. They include stream idle time while CPU code submits work; they are not
the sum of kernel durations. CPU wall totals include the final event wait.
Decomposition adds event/hook overhead; use the uninstrumented sweeps for deltas.
Peak allocated/reserved fields use MiB, include resident model/input tensors,
and cover warmup plus timed forwards. Peak counters and unused cached allocations
are reset between arms. Deltas use median GPU latency; negative delta % is faster.

Parity preserves original shape/dtype/equality, then computes max/mean absolute
error, relative L2 against the baseline, and cosine on CPU float64 tensors.
Bitwise inequality alone passes. Default gates are max_abs <= 0.1, rel_l2 <= 0.01,
and cosine >= 0.9999; nonfinite output or shape/dtype differences fail. These are
explicit initial screening limits, not a model-quality guarantee. Adjust only
after reviewing observed error magnitudes using `--parity-max-abs`,
`--parity-max-rel-l2`, and `--parity-min-cosine`.

Substantial drift saves the summary and exits 2, stopping the driver. A batch
alternative OOM is recorded without shrinking its workload and the sweep
continues after cleanup. A baseline/resampler/decomposition OOM exits 2. Unexpected
failures remain visible as exceptions. If the paired isolated resampler latency
does not improve, or parity fails, the full-encoder comparison is explicitly
marked skipped.

No server, serving ABBA, compilation, CUDA Graph integration, attention rewrite,
or custom kernel is part of this harness. Run on an idle GPU before using its
numbers to decide the next experiment.

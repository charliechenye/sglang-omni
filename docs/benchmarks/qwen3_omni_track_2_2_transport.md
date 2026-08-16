# Qwen3-Omni Track 2.2 transport probe

This probe measures the economic value of the approved semantic Track 2.2
choices. It does not load Qwen3-Omni weights or any Hugging Face model, and it
does not modify production SGLang-Omni code.

The implementation is
[`benchmarks/qwen3_omni_track_2_2_transport.py`](../../benchmarks/qwen3_omni_track_2_2_transport.py).

## Exact architecture

Each direction is a fresh two-process sequence. The sender is a Thinker-like
process pinned to GPU 0; the receiver is a Talker-like process pinned to GPU
1. Both processes reuse the real `CommEngine.send_stream_chunk()` and
`stage_io.write_stream_chunk()` / `read_stream_chunk()` path. The edge control
messages use the repository's real `PushSocket` / `PullSocket` and msgpack
serialization. There is no synthetic `torch.copy_` loop.

The primary producer is one task, matching `Stage._drain_outbox_external`:
request IDs are interleaved round-robin for c1/c8/c32. A send waits only for
`CommEngine.send_stream_chunk()` to return after `DataReady` publication. It
then marks that object ready for the ACK loop and immediately produces the next
chunk. Warmup transfers are fully drained before measurement starts. Measured
transfers are fully drained after production ends.

The JSON reports these distinct timing domains:

- `publish_latency_ms`: call start through `send_stream_chunk()` return;
- `ack_release_latency_ms`: call start through ACK plus relay-resource release;
- `publish_chunks_per_sec`: measured chunks divided by the publication wall
  interval;
- `end_to_end_drain_chunks_per_sec`: measured chunks divided by measurement
  start through the final ACK/resource drain;
- `final_ack_drain_ms`: final publication return through the final completion;
- `maximum_outstanding_transfer_count`: peak published-but-not-retired objects.

The sender keeps one persistent source tensor per arm for the whole process. In
particular, C reuses one CUDA `uint8[1]` carrier for every chunk. Receiver
allocations are the normal destination allocations performed by
`stage_io.read_tensor()`.

| Arm | Primary data | Metadata | Logical tensor bytes/chunk | Tensor transfers/chunk | Expected route |
| --- | --- | --- | ---: | ---: | --- |
| A | CUDA BF16 `[2048]` | Python `int` token ID + CUDA BF16 `layer_hidden[2048]` | 8,192 | 2 | pooled CUDA IPC |
| B | CUDA BF16 `[2048]` | Python `int` token ID | 4,096 | 1 | pooled CUDA IPC |
| C | persistent CUDA `uint8[1]` | Python `int` token ID | 1 | 1 | pooled CUDA IPC |
| D | CPU `torch.long[1]` | Python `int` token ID | 8 | 1 | host SHM |

The probe asserts the route on the sender, the `DataRef.transport` on the
receiver, the concrete relay class, tensor transfer count, shapes/dtypes, and
Python type of `token_id`. It also performs a content check during warmup. It
explicitly checks that the two-GPU topology does not qualify for the direct
same-placement CUDA IPC shortcut; A/B/C therefore exercise pooled CUDA IPC.

The benchmark inserts the checked-out repository root into `sys.path` before
importing `sglang_omni`, asserts that `sglang_omni.__file__` is inside that
checkout, and records both paths in the JSON output.

## Primary command

Run from the repository root on a host with two visible same-node GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_OMNI_COMM_TRACE=0 \
python benchmarks/qwen3_omni_track_2_2_transport.py \
  --mode primary \
  --json-output /tmp/qwen3-omni-track-2-2-transport-primary.json
```

Primary defaults are 100 warmup chunks, 2,000 measured chunks, five rounds,
request counts 1/8/32, and both arm orders `A B C D` and `D C B A`. Chunk
counts are aggregate per arm/cell and are distributed across request IDs. The
single producer interleaves those request IDs without sleeps or per-request
producer tasks.

The CUDA IPC pool size is intentionally unset by default. That leaves
`cuda_ipc_pool_size_mb` absent from the router config, so the production relay
default applies: 1 GiB with the production 512 MiB legacy slot-size default
and two credits. Use `--cuda-ipc-pool-mb 1024` only when an explicit equivalent
is needed; any override is recorded in JSON.

## Smaller trace diagnostic

Trace logging is excluded from the primary timing comparison. Run the smaller
diagnostic separately:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_OMNI_COMM_TRACE=1 \
python benchmarks/qwen3_omni_track_2_2_transport.py \
  --mode trace \
  --json-output /tmp/qwen3-omni-track-2-2-transport-trace.json
```

Trace mode defaults to 64 warmup chunks, 100 measured chunks, one round, the
same three request counts, and both arm orders. Existing communication trace
events are aggregated by run and process. Where emitted, JSON reports:

- logical bytes and event counts;
- CUDA IPC slot count and slot allocation wait;
- sender copy enqueue and CUDA event-handle time;
- receiver event import and copy enqueue time;
- receiver GPU copy/wait time;
- sender ACK/resource-hold time.

SHM has no CUDA-event fields; those fields are absent rather than fabricated.

## Modal smoke and full commands

`modal shell --add-local .` mounts the current directory at
`/mnt/{local-directory-basename}`. These commands derive that basename instead
of assuming the checkout is named `sglang-omni`. They keep the pinned CI image,
request two same-node H100s, and leave the CUDA IPC pool at its production
default.

Smoke run (trace mode, no production result):

```bash
WORKTREE_BASENAME="$(basename "$PWD")"
modal shell \
  --gpu H100!:2 \
  --image hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101 \
  --add-local . \
  -c "cd \"/mnt/${WORKTREE_BASENAME}\" && CUDA_VISIBLE_DEVICES=0,1 SGLANG_OMNI_COMM_TRACE=1 python benchmarks/qwen3_omni_track_2_2_transport.py --mode trace --json-output /tmp/qwen3-omni-track-2-2-transport-trace.json"
```

Full primary run:

```bash
WORKTREE_BASENAME="$(basename "$PWD")"
modal shell \
  --gpu H100!:2 \
  --image hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101 \
  --add-local . \
  -c "cd \"/mnt/${WORKTREE_BASENAME}\" && CUDA_VISIBLE_DEVICES=0,1 SGLANG_OMNI_COMM_TRACE=0 python benchmarks/qwen3_omni_track_2_2_transport.py --mode primary --json-output /tmp/qwen3-omni-track-2-2-transport-primary.json"
```

These are paid-GPU commands and are not run by CPU qualification.

## Paired attribution and decision semantics

The JSON and human-readable table report aggregate timing plus paired
statistics. Arms are paired within identical `(direction, round, concurrency)`
cells. Each paired metric reports median percentage delta, p25/p75, MAD,
improvement count/total, improvement fraction, and stable-effect status.
Latency deltas are positive when the candidate is faster; throughput deltas are
positive when the candidate is faster.

Incremental attribution remains:

- **A → B:** value of removing the second tensor while keeping BF16 `[2048]`;
- **B → C:** value of shrinking the remaining primary tensor while retaining
  pooled CUDA IPC and a persistent CUDA carrier;
- **C → D:** value or cost of changing from pooled CUDA IPC to host SHM.

Direct baseline comparisons are also reported:

- **A → B**;
- **A → C**;
- **A → D**.

The recommendation ranks candidates against A using direct evidence. A stable
`C → D` regression means “prefer C over D”; it does not make the overall Track
2.2 result NO-GO if direct `A → C` is a stable GO.

The rubric is deliberately variability-aware:

- **GO:** paired median publication and ACK/resource latencies improve by at
  least 5%, at least 60% of paired observations improve, the median effect is
  larger than MAD, and neither throughput metric materially regresses;
- **MAYBE:** effects are marginal, mixed, or ordinary paired variation is
  comparable to the nominal effect;
- **NO-GO:** the compared candidate has a stable material regression. This
  rejects that candidate relative to its stated baseline, not every other
  candidate.

Overall status is based on the direct A-baseline recommendation, not on every
incremental transition improving.

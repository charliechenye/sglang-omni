# Qwen3-Omni Track 2.2 transport probe

This probe measures the economic value of the semantic Track 2.2 choices. It
does not load Qwen3-Omni weights or any Hugging Face model, and it does not
modify production transport code.

The implementation is
[`benchmarks/qwen3_omni_track_2_2_transport.py`](../../benchmarks/qwen3_omni_track_2_2_transport.py).

## Exact architecture

Each direction is a fresh two-process sequence. The sender is a Thinker-like
process pinned to GPU 0; the receiver is a Talker-like process pinned to GPU
1. Both processes reuse the real `CommEngine.send_stream_chunk()` and
`stage_io.write_stream_chunk()` / `read_stream_chunk()` path. The edge control
messages use the repository's real `PushSocket` / `PullSocket` and msgpack
serialization. There is no synthetic `torch.copy_` loop.

The sender keeps one persistent source tensor per arm for the whole process.
In particular, C reuses one CUDA `uint8[1]` carrier for every chunk. The
receiver allocations are the normal destination allocations performed by
`stage_io.read_tensor()`.

| Arm | Primary data | Metadata | Logical tensor bytes/chunk | Tensor transfers/chunk | Expected route |
| --- | --- | --- | ---: | ---: | --- |
| A | CUDA BF16 `[2048]` | Python `int` token ID + CUDA BF16 `layer_hidden[2048]` | 8,192 | 2 | pooled CUDA IPC |
| B | CUDA BF16 `[2048]` | Python `int` token ID | 4,096 | 1 | pooled CUDA IPC |
| C | persistent CUDA `uint8[1]` | Python `int` token ID | 1 | 1 | pooled CUDA IPC |
| D | CPU `torch.long[1]` | Python `int` token ID | 8 | 1 | host SHM |

The probe asserts the route on the sender, the `DataRef.transport` on the
receiver, the concrete relay class, the tensor transfer count, shapes/dtypes,
and the Python type of `token_id`. It also performs a content check during
warmup. It explicitly checks that the two-GPU topology does not qualify for
the direct same-placement CUDA IPC shortcut; A/B/C therefore exercise the
pooled CUDA IPC relay.

## Primary measurement command

Run from the repository root on a host with two visible same-node GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_OMNI_COMM_TRACE=0 \
python benchmarks/qwen3_omni_track_2_2_transport.py \
  --mode primary \
  --json-output /tmp/qwen3-omni-track-2-2-transport-primary.json
```

The primary defaults are 100 warmup chunks, 2,000 measured chunks, five
rounds, request counts 1/8/32, and both arm orders `A B C D` and `D C B A`.
The chunk counts are aggregate per arm/cell and are distributed across the
active request IDs. Each request ID has at most one outstanding chunk, while
all request IDs run concurrently; the benchmark introduces no sleeps.

The pooled CUDA IPC slot size is the production default of 64 KiB. The pool
defaults to 64 MiB to avoid reserving a 1 GiB pool for a model-free probe; the
transport class and slot allocator are unchanged. Override it explicitly if
the target deployment uses another pool size.

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
events are aggregated by run and process. Where emitted, the JSON reports:

- logical bytes and event counts;
- CUDA IPC slot count and slot allocation wait;
- sender copy enqueue and CUDA event-handle time;
- receiver event import and copy enqueue time;
- receiver GPU copy/wait time;
- sender ACK/resource-hold time.

SHM has no CUDA-event fields; those fields are absent rather than fabricated.

## Modal / two-H100 command

From the repository root, this opens a two-GPU same-node Modal shell using the
pinned CUDA image used by the repository's GPU CI. The `!` prevents Modal's
automatic H100-to-H200 upgrade, and `:2` requests two GPUs in one container.
The local checkout is mounted at `/mnt/sglang-omni`.

```bash
modal shell \
  --gpu H100!:2 \
  --image hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101 \
  --add-local . \
  -c 'cd /mnt/sglang-omni && CUDA_VISIBLE_DEVICES=0,1 SGLANG_OMNI_COMM_TRACE=0 python benchmarks/qwen3_omni_track_2_2_transport.py --mode primary --json-output /tmp/qwen3-omni-track-2-2-transport-primary.json'
```

Run the trace command in the same shell with `--mode trace` and a different
JSON output path. This is a paid-GPU command and is intentionally not run by
the development validation here.

## Attribution and rubric

The JSON and human-readable table report median and p95 per-chunk wall
latency, aggregate chunks/sec, logical tensor bytes, tensor transfer count,
backend byte lengths, relay classes, and transport checks. The intended
attribution is:

- **A → B:** value of removing the second tensor while leaving the primary
  tensor at BF16 `[2048]`;
- **B → C:** value of shrinking the remaining primary tensor while retaining
  pooled CUDA IPC and a persistent CUDA carrier;
- **C → D:** value or cost of changing from pooled CUDA IPC to host SHM. This
  is a transport-class comparison, not a claim that 1 byte and 8 bytes have
  equivalent movement costs.

The generated JSON applies this rubric separately at request counts 1, 8, and
32, then gives a transition decision:

- **GO:** at least two of the three cells improve median and p95 by at least
  5%, with no throughput regression of at least 5%;
- **MAYBE:** mixed, marginal, or insufficiently repeatable evidence;
- **NO-GO:** at least two cells regress or show no material transport benefit.

No production Qwen code is implied by a GO result. The result only justifies
the corresponding transport/representation optimization for a follow-up
change.

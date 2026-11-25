# P6.1 — Diagnose and repair H200 CUDA conditional-WHILE capture

## Why P6.0 was not an unsupported-hardware verdict

On H200 / torch 2.14.1+cu130, the first P6.0 prototype reported:

P6_DYNAMIC_DEPTH_CUDA_GRAPH_UNSUPPORTED_OR_INCORRECT
AcceleratorError: CUDA error: operation failed due to a previous error during capture
cudaErrorStreamCaptureInvalidated

The earlier script entered a plain torch.cuda.graph with torch.while_loop,
but did **not** install the special CUDA control-flow dispatch mode used
by PyTorch 2.14's own Inductor cudagraph implementation. This is a bug
in the experiment setup; the observed error does not prove lack of
CUDA conditional WHILE hardware support.

Upstream PyTorch 2.14:
- torch/_higher_order_ops/cudagraph_conditional_nodes.py defines
  CUDAGraphCaptureControlFlowOpDispatchMode. It maps while_loop_op to
  while_loop_node using begin_capture_to_while_node.
- torch/_inductor/cudagraph_trees.py uses BOTH
  torch.cuda.graph(... capture_error_mode="thread_local") and
  CUDAGraphCaptureControlFlowOpDispatchMode() for actual conditional nodes.
- PyTorch issue #201567 documents a direct, minimal capture reproducer
  using while_loop_op and this mode, along with a CUDA graph private
  memory-pool cleanup issue in 2.14.0+cu130. Avoid repeatedly trying
  failed captures in one long-lived process.

References:
https://github.com/pytorch/pytorch/blob/v2.14.1/torch/_higher_order_ops/cudagraph_conditional_nodes.py
https://github.com/pytorch/pytorch/blob/v2.14.1/torch/_inductor/cudagraph_trees.py
https://github.com/pytorch/pytorch/issues/201567
https://pytorch.org/blog/pytorch-2-14-release-blog/

## P6.1 fix

Instead of the public torch.while_loop called directly inside a plain
graph, P6.1 uses the higher-order while_loop_op under the same explicit
PyTorch CUDA conditional-node dispatch mode that Inductor uses. This is
a private experimental API, not production-approved code.

P6.1 executes THREE STAGES in THREE FRESH Python/CUDA processes:

1. plain_graph — normal simple CUDA Graph capture/replay on changing
   input values. Isolates generic GPU/capture/driver problems.
2. while_minimal — capture/replay a real conditional WHILE with a
   device-controlled trip count 0/3/7. Verifies one captured graph
   responds to NEW GPU inputs on subsequent replays.
3. depth_k2 — capture the 16-codegroup residual-depth progression and
   accepted bonus skip with five different GPU decision masks. Expected
   all-reject=14 rounds, all-accept=7; compare all outputs to CPU oracle.

No sequential GPU->CPU .item() occurs inside graph replay. Device outputs
are copied to the host ONLY after a completed graph replay for validation.

Each stage reports the first Python exception and its full traceback
in its OWN stage directory. After the first stage failure, subsequent
stages are NOT run. This avoids a poisoned CUDA capture context masking
the actual cause and limits known pool-lifetime issues. A crashing
subprocess is also recorded as a nonzero/signal exit without claiming
a supported feature.

## Commands on H200 (NO Omni server required)

Use a NEW evidence directory; preserve the failed P6.0 -007 results.

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0

python -m pytest -q \
  tests/unit_test/qwen3_tts/test_cp_spec_p6_1_isolation.py

CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_tts_cp_spec_k2.p6_1_conditional_isolation \
  --out "$HOME/results/qwen3tts-cpspec-k2-p6-1-008"

cat "$HOME/results/qwen3tts-cpspec-k2-p6-1-008/summary.md"
~~~

If FAIL, the useful error is now under the first failed stage:

~~~bash
for f in "$HOME/results/qwen3tts-cpspec-k2-p6-1-008"/*/traceback.txt; do
  [ -f "$f" ] && { echo "===== $f"; sed -n '1,140p' "$f"; }
done
~~~

Do not retry a failed stage in the same Python process or reuse the old
evidence directory. No CUDA package upgrade, root, profiler or server is
required.

## Decision

- plain_graph FAIL: ordinary CUDA Graph/stream or driver problem; fix
  that FIRST, not higher-order loop semantics.
- plain_graph PASS, while_minimal FAIL: higher-order conditional node
  integration/capture problem in the current PyTorch+CUDA runtime.
  Preserve the very first traceback and compare upstream 2.14 implementation.
- while_minimal PASS, depth_k2 FAIL: our shape/dynamic gather/bonus loop
  implementation or conditional-node body restriction needs repair.
- all three PASS: dynamic GPU-driven codebook depth progression is
  **supported in this isolated H200 test**, not a working TTS model
  graph or an acceleration proof.

The report includes 0/3/7 and 7..14 round CUDA-event launch envelopes
as a rough conditional-control overhead screen. The P5 actual full
Graph-ON Predictor baseline of 2.2266 ms/frame remains the real target.

After success, the hard integration work remains: replace the Python
integer KV cache offset and Python sampler with GPU-state-compatible
fixed-depth subgraphs, preserve exact p/q and residual correction,
and compare full Graph-ON Omni HTTP serving with quality guardrails.

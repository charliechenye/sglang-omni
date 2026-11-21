# P6.0 — H200 device-resident dynamic K2 graph compatibility gate

**Run this short feasibility test before implementing a GPU-resident K2 scheduler. No server required.**

Prior P5 on the same Qwen3-TTS model:

- Full existing Predictor CUDA Graph ON baseline = 2.2266 ms/frame.
- Static 2-token verifier CUDA Graph = 0.1386–0.1395 ms/round.
- Historical K2 ~9.686 rounds/frame gives a verifier-only lower bound
  1.3478 ms/frame, leaving **0.8788 ms/frame** for the opening pair,
  sampling, rejection/residual correction, bonus, embedding and dynamic
  scheduling. This is a cost budget, **not an actual speedup**.
- P2 and P5 were at different experiment SHAs. The 720 P2 frames included
  deterministic repeated arms, not independent utterances.

PyTorch 2.14 (Omni pins torch==2.14.1) added CUDA Graph capture for
torch.while_loop using CUDA conditional WHILE nodes. Source:
https://pytorch.org/blog/pytorch-2-14-release-blog/
It is a prototype API; runtime/backend compatibility on our H200 has
NOT been validated yet.

## 1. Fetch scripts, CPU oracle test

~~~bash
cd ~/sglang-omni
git fetch charlie-fork
git switch experiment/qwen3-tts-codepredictor-spec-k2-p0
git merge --ff-only \
  charlie-fork/experiment/qwen3-tts-codepredictor-spec-k2-p0
python -m pytest -q tests/unit_test/qwen3_tts/test_cp_spec_p6_graph.py
~~~

## 2. Test a GPU-driven conditional graph (without loading TTS)

~~~bash
cd ~/sglang-omni
CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_tts_cp_spec_k2.p6_device_conditional_gate \
  --repeats 128 \
  --out "$HOME/results/qwen3tts-cpspec-k2-p6-control-007"

cat "$HOME/results/qwen3tts-cpspec-k2-p6-control-007/summary.md"
~~~

The script creates ONE CUDA Graph with a device-driven dynamic while
loop. The same captured graph must replay five GPU acceptance masks
with different true iterations (all reject=14, all accept=7, others
intermediate) and return exactly the CPU oracle's depth, proposal round,
accept and bonus counts. The graph does not use CPU .item() to branch
during replay. It reports both CUDA-event launch envelope and host
wall time at 7, 14 and mixed iterations.

PASS establishes device-driven graph control feasibility ONLY, not
model correctness or GPU K2 speedup. If capture fails, report.json
contains the real exception; fail closed, do not silently use eager.

## Critical obstacle to actual model integration

The current Omni predictor_forward_tokens takes cache_len as a Python
integer and slices KV cache / position buffers at a fixed offset. It
cannot simply accept a GPU tensor as a dynamic cache offset. After
this control-flow gate, design a conditional selection of depth-specific
fixed-offset verifier subgraphs, with GPU-maintained depth, draft,
accept/reject, bonus, valid KV length and RNG. PyTorch 2.14 also added
torch.switch for multiway branches, but nested graph compatibility
with our model still needs explicit testing.

Before any claimed acceleration:
- Preserve truncated p/q sampler distribution and residual correction.
- Demonstrate live H200 KV and exact-model target parity in the new graph.
- Compare entire Graph-ON serving E2E on matched workloads, same
  concurrency, no debug instrumentation and natural EOS.
- Do not infer real performance from this toy device-control graph.

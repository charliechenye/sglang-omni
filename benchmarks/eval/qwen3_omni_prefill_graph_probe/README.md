# Qwen3-Omni prefill graph probe

This directory is benchmark-only. It observes the upstream
`PrefillCudaGraphRunner` and never implements a local graph candidate or
graph/eager dispatcher.

Run a server command through the probe wrapper:

```bash
python -m benchmarks.eval.qwen3_omni_prefill_graph_probe \
  --output /tmp/qwen3-omni-prefill-probe.json \
  --requested-prefill-backend breakable \
  --compatibility-injection on \
  -- python -m sglang.launch_server ...
```

The optional compatibility injection is restricted to
`Qwen3OmniThinkerForCausalLM` model configuration classification. It sets
`ModelConfig.is_multimodal` for that architecture in the probe subprocess only;
it does not change upstream eligibility checks.

The JSON report includes capture buckets and memory, upstream eligibility
evaluation/acceptance counts, eager fallbacks, replay calls, selected buckets,
padding ratios, and the live `input_embeds`/`replace_embeds` contract before
eligibility. `c_qualified` is true only when the resolved backend is breakable,
capture succeeded, replay completed, an upstream graph batch was accepted, and
the live `input_embeds` contract was empty.

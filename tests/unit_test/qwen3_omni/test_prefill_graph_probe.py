# SPDX-License-Identifier: Apache-2.0
"""Pure accounting checks for the benchmark-only upstream graph probe."""

from benchmarks.eval.qwen3_omni_prefill_graph_probe.probe import ProbeState


def test_probe_qualification_requires_upstream_replay_evidence() -> None:
    state = ProbeState(
        resolved_prefill_backend="breakable",
        capture_succeeded=True,
        replay_calls=1,
        accepted_graph_batches=1,
        live_serving_input_embeds_none_before_eligibility=True,
    )

    report = state.report()

    assert report["c_qualified"] is True
    assert report["qualification_requirements"]["live_input_embeds_none"] is True


def test_probe_does_not_qualify_from_requested_backend_label_alone() -> None:
    state = ProbeState(requested_prefill_backend="breakable")

    report = state.report()

    assert report["c_qualified"] is False
    assert report["capture_succeeded"] is False
    assert report["replay_calls"] == 0

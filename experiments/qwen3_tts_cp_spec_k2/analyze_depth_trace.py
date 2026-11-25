# SPDX-License-Identifier: Apache-2.0
"""CPU-only P1/P2 report from real C1 K2 per-depth and parity traces.

Conditional depth acceptance is measured only *when a proposal was made*.
Bonus depths are explicitly separated; deeper proposal cohorts are selected
by earlier accept/reject events, not 14 independent i.i.d. Bernoulli trials.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sglang_omni.models.qwen3_tts.code_predictor_spec_diagnostics import (
    SCHEMA, validate_depth_events,
)

STATUS = "P1_P2_K2_DEPTH_DIAGNOSTIC_COMPLETE_NOT_FULL_MODEL_DISTRIBUTION_PARITY_OR_SPEED"
STATUS_PARTIAL = "P1_P2_K2_TRACE_PARTIAL_OR_PARITY_NOT_CHECKED"


def analyze(folder: Path, *, expect_source_sha: str | None, min_frames: int, min_parity_frames: int = 0) -> dict:
    folder = folder.expanduser().resolve(strict=True)
    paths = sorted(folder.glob("frame_pid*_*.json"))
    failed = sorted(folder.glob("frame_pid*_*_FAIL.json"))
    if failed:
        raise RuntimeError(
            f"{len(failed)} P1 parity failures in trace dir; inspect {failed[0]}"
        )
    if len(paths) < min_frames:
        raise RuntimeError(f"only {len(paths)} frames, require >= {min_frames}")
    rows = [json.loads(p.read_text()) for p in paths]
    if len({r.get("pid") for r in rows}) != 1:
        raise RuntimeError("mixed worker PIDs in trace directory")
    frames = [r.get("frame") for r in rows]
    if sorted(frames) != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"missing/duplicate frame numbers: {frames[:12]}")
    recorded_shas = {row.get("source_sha") for row in rows}
    if len(recorded_shas) != 1:
        raise RuntimeError("source revision mixed across trace rows")
    revision = next(iter(recorded_shas))
    if expect_source_sha is not None and revision != expect_source_sha:
        raise RuntimeError(
            f"source SHA mismatch: traces {revision!r} expected {expect_source_sha!r}"
        )
    per_depth = defaultdict(lambda: {
        "proposed": 0, "accepted": 0, "rejected": 0, "bonuses": 0,
        "expected_overlap_sum": 0.0, "target_on_proposal_support_sum": 0.0,
        "proposal_on_target_support_sum": 0.0,
    })
    total_proposals = total_accepted = total_rejected = total_bonus = 0
    parity_checked = 0
    worst_abs: dict[str, float] = {}
    for row in rows:
        if row.get("schema") != SCHEMA:
            raise RuntimeError("trace schema mismatch")
        if row.get("total_residual_codes") != 15:
            raise RuntimeError("not a 16-code-group full-frame trace")
        events = row.get("per_depth")
        if not isinstance(events, list):
            raise RuntimeError("missing per-depth proposals")
        validate_depth_events(events)
        if any(len(event) < 5 for event in events):
            raise RuntimeError("incomplete proposal event")
        proposals = len(events)
        accepted = sum(int(event["accepted"]) for event in events)
        rejected = proposals - accepted
        bonuses = sum(event.get("bonus_depth") is not None for event in events)
        if (
            row.get("proposed") != proposals
            or row.get("accepted") != accepted
            or row.get("rejected") != rejected
            or row.get("rounds") != proposals
            or row.get("bonus") != bonuses
            or accepted + rejected + bonuses != 14
        ):
            raise RuntimeError(f"K2 counters contradict codec depth coverage frame={row.get('frame')}")
        if not isinstance(row.get("output_codes"), list) or len(row["output_codes"]) != 16:
            raise RuntimeError("output code length is not 16")
        emb = row.get("embedding_sum")
        if not isinstance(emb, dict) or emb.get("close") is not True:
            raise RuntimeError("embedding sum did not pass real-code parity")
        parity = row.get("target_parity")
        if not isinstance(parity, dict) or parity.get("pass") is not True:
            raise RuntimeError("target parity is explicitly failed")
        if parity.get("executed"):
            if len(parity.get("checks", [])) != proposals:
                raise RuntimeError("partial parity check of a frame is not a pass")
            parity_checked += 1
            for check in parity["checks"]:
                if check.get("pass") is not True:
                    raise RuntimeError("target parity check failed")
                for name, measured in check["checks"].items():
                    if measured.get("close") is not True:
                        raise RuntimeError(f"KV/logits/hidden parity failed: {name}")
                    worst_abs[name] = max(worst_abs.get(name, 0.0), measured["max_abs"])
        for event in events:
            depth = event["proposal_depth"]
            info = per_depth[depth]
            info["proposed"] += 1
            info["accepted"] += int(event["accepted"])
            info["rejected"] += int(not event["accepted"])
            info["expected_overlap_sum"] += event["expected_overlap"]
            info["target_on_proposal_support_sum"] += event["target_mass_on_q_support"]
            info["proposal_on_target_support_sum"] += event["proposal_mass_on_p_support"]
            bonus_depth = event.get("bonus_depth")
            if bonus_depth is not None:
                per_depth[bonus_depth]["bonuses"] += 1
        total_proposals += proposals
        total_accepted += accepted
        total_rejected += rejected
        total_bonus += bonuses
    table = []
    for depth in range(2, 16):
        data = per_depth[depth]
        n = data["proposed"]
        table.append({
            "residual_depth": depth,
            "proposals": n,
            "accepted": data["accepted"],
            "rejected": data["rejected"],
            "bonus_skipped_proposals": data["bonuses"],
            "accept_fraction_conditional_on_proposal": (
                data["accepted"] / n if n else None
            ),
            "expected_overlap_conditional_on_proposal": (
                data["expected_overlap_sum"] / n if n else None
            ),
            "target_mass_on_q_support": (
                data["target_on_proposal_support_sum"] / n if n else None
            ),
            "q_mass_on_target_support": (
                data["proposal_on_target_support_sum"] / n if n else None
            ),
        })
    if not 0 <= min_parity_frames <= len(rows):
        raise ValueError("min_parity_frames must fit in selected frame count")
    if parity_checked < min_parity_frames:
        raise RuntimeError(
            f"insufficient fully-verified real target frames: "
            f"{parity_checked} < {min_parity_frames}"
        )
    expected = sum(event["expected_overlap_sum"] for event in per_depth.values())
    result = {
        "status": STATUS if parity_checked else STATUS_PARTIAL,
        "source_sha": revision,
        "frames": len(rows),
        "parity_checked_frames": parity_checked,
        "parity_max_abs_by_tensor": worst_abs,
        "total_proposals": total_proposals,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_bonus": total_bonus,
        "observed_acceptance": total_accepted / total_proposals,
        "theoretical_selected_state_overlap": expected / total_proposals,
        "depths": table,
        "limits": [
            "Proposals at later depths are a selection-biased conditional cohort; bonus depths were skipped.",
            "Theoretical overlap uses the PyTorch sampled distribution, not a proven bit-identical fused GPU sampler.",
            "P1 compares packed vs serial using the same K2 proposal+prefix, not whole-utterance seeded distribution equality.",
            "Frames from one utterance are correlated. Do not treat proposal count as independent samples.",
            "No WER, speaker similarity, Graph-ON QPS or E2E speedup claim.",
        ],
    }
    return result


def write_report(out: Path, data: dict) -> None:
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    values = [
        "# Omni Qwen3-TTS CP Spec K2: P1 target parity + P2 depth acceptance",
        "",
        "**" + data["status"] + "**",
        "",
        f"Source: {data['source_sha']}",
        f"Frames: {data['frames']}",
        f"Fully parity-checked frames: {data['parity_checked_frames']}",
        f"Proposed/accepted/rejected/bonus: "
        f"{data['total_proposals']}/{data['total_accepted']}/"
        f"{data['total_rejected']}/{data['total_bonus']}",
        f"Observed proposal acceptance: {data['observed_acceptance']:.4%}",
        f"Mean theoretical selected-state overlap: {data['theoretical_selected_state_overlap']:.4%}",
        f"Worst max abs by tensor: {json.dumps(data['parity_max_abs_by_tensor'], sort_keys=True)}",
        "",
        "## Per-depth conditional proposal acceptance (bonus depths excluded)",
        "",
        "| Depth | Proposed | Accepted | Bonus skips | Observed | Expected overlap | Target mass on q support |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["depths"]:
        observed = row["accept_fraction_conditional_on_proposal"]
        expected = row["expected_overlap_conditional_on_proposal"]
        support = row["target_mass_on_q_support"]
        values.append(
            f"| {row['residual_depth']} | {row['proposals']} | "
            f"{row['accepted']} | {row['bonus_skipped_proposals']} | "
            f"{observed:.2%} | {expected:.2%} | {support:.2%} |"
            if observed is not None and expected is not None and support is not None
            else f"| {row['residual_depth']} | 0 | 0 | {row['bonus_skipped_proposals']} | — | — | — |"
        )
    values.extend(["", "## Limits", ""] + [
        f"- {line}" for line in data["limits"]
    ])
    (out / "summary.md").write_text("\n".join(values) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--expect-source-sha", default=None)
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--min-parity-frames", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.min_frames < 1:
        p.error("min-frames must be positive")
    report = analyze(
        args.frames_dir, expect_source_sha=args.expect_source_sha,
        min_frames=args.min_frames, min_parity_frames=args.min_parity_frames,
    )
    write_report(args.out, report)
    print(f"{report['status']}: {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()

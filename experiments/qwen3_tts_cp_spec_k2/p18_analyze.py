# SPDX-License-Identifier: Apache-2.0
"""P18A: fail-closed all-depth acceptance census; cluster bootstrap by request.

NO frame-level independence. One paired uniform draw per real checkpoint.
The P17 independent q/p + static verifier + full-KV oracle must PASS for
every recorded width. This reader never uses serving perf or audio quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "qwen3tts-p18a-native-ar-eager-all-depth-v1"
SOURCE_SCHEMA = "p18a-native-eager-http-driver-v1"
RESULT = "P18A_ALL_DEPTH_CLUSTERED_ACCEPTANCE_CENSUS_PASS_NO_SPEEDUP"
DEPTHS = tuple(range(1, 15))
BOOTSTRAPS = 1000


def widths(depth: int) -> tuple[int, ...]:
    if depth not in DEPTHS:
        raise ValueError("P18_INVALID_DEPTH")
    return tuple(w for w in (2, 3, 4) if depth + w - 1 <= 14)


def seed_for(sha: str, case: str, frame: int, depth: int) -> int:
    text = f"p18a-v1|{sha}|{case}|{frame}|{depth}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2**63)


def _ci(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return (ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))])


def clustered_ci(per_request: dict[str, list[float]],
                 *, seed: int, repeats: int = BOOTSTRAPS) -> dict:
    """Each request is one resampling unit; included frames stay clustered."""
    if len(per_request) < 2 or repeats < 100:
        raise ValueError("P18_MIN_TWO_REQUEST_CLUSTERS_AND_100_RESAMPLES")
    means = [
        statistics.mean(values)
        for _, values in sorted(per_request.items())
    ]
    rng = random.Random(seed)
    population = len(means)
    bootstrap = [
        statistics.mean(
            means[rng.randrange(population)] for _ in range(population)
        ) for _ in range(repeats)
    ]
    lo, hi = _ci(bootstrap)
    return {
        "mean": statistics.mean(means),
        "cluster_bootstrap_95pct": [lo, hi],
        "independent_request_clusters": population,
        "frames_within_clusters_NOT_independent": sum(
            len(v) for v in per_request.values()
        ),
    }


def validate_rows(root: Path) -> tuple[dict, list[dict]]:
    config = json.loads((root / "config.json").read_text())
    driver = json.loads((root / "visits.json").read_text())
    sha = (root / "source.sha").read_text().strip()
    if (config.get("source_sha") != sha or len(sha) != 40
            or config.get("native_k2") != "OFF"
            or config.get("trajectory") != "native-ar-eager-only"
            or driver.get("schema") != SOURCE_SCHEMA
            or driver.get("source_sha") != sha
            or driver.get("status")
            != "P18A_NATIVE_AR_ALL_DEPTH_CENSUS_EVIDENCE_CAPTURED"
            or driver.get("trajectory")
            != "NATIVE_AR_EAGER_NOT_GRAPH_SERVING"):
        raise RuntimeError("P18A_SOURCE_OR_FINISH_GATE_FAILED")
    mode = driver.get("mode")
    expected_visits = {"smoke": 4, "main": 24}.get(mode)
    frames = {"smoke": 2, "main": 8}.get(mode)
    visits = driver.get("visits")
    if (expected_visits is None or not isinstance(visits, list)
            or len(visits) != expected_visits
            or config.get("frames_per_request") != frames
            or driver.get("frames_per_request") != frames):
        raise RuntimeError("P18A_MODE_OR_REQUEST_COUNT_MISMATCH")
    known_cases = {}
    for visit in visits:
        case = visit.get("case_id")
        if (not isinstance(case, str) or case in known_cases
                or visit.get("finish_reason") != "stop"
                or visit.get("frames") != frames
                or visit.get("depth_rows") != frames * len(DEPTHS)
                or visit.get("source_sha") != sha
                or not isinstance(visit.get("request_hash"), str)):
            raise RuntimeError("P18A_INVALID_CASE_OR_NON_NATURAL_STOP")
        known_cases[case] = visit

    if ((root / "ARM_P18").exists() or (root / "ACTIVE_CASE.json").exists()
            or any((root / "census" / "failures").glob("*"))):
        raise RuntimeError("P18A_STALE_ARM_CASE_OR_WORKER_FAILURE")
    paths = sorted((root / "census" / "rows").glob("*.json"))
    expected_count = expected_visits * frames * len(DEPTHS)
    if len(paths) != expected_count:
        raise RuntimeError(
            f"P18A_INCOMPLETE_ROW_COUNT {len(paths)}/{expected_count}"
        )
    rows, unique = [], set()
    case_frames = defaultdict(set)
    case_requests = defaultdict(set)
    case_native_seeds = defaultdict(set)
    pids = set()
    for path in paths:
        row = json.loads(path.read_text())
        case = row.get("case_id")
        depth = row.get("last_committed_depth")
        frame = row.get("frame")
        if case not in known_cases:
            raise RuntimeError("P18A_UNMATCHED_CASE_ID")
        v = known_cases[case]
        if (row.get("schema") != SCHEMA
                or row.get("source_sha") != sha
                or row.get("trajectory") != "NATIVE_AR_EAGER_CODEBOOK_SAMPLING"
                or row.get("probe") != "P17_INDEPENDENT_ORACLE_COUNTERFACTUAL"
                or row.get("gpu") != "NVIDIA H200"
                or type(row.get("pid")) is not int or row["pid"] <= 0
                or type(row.get("native_sub_sampling_seed")) is not int
                or type(row.get("native_committed_code")) is not int
                or not 0 <= row["native_committed_code"] < 2048
                or type(frame) is not int or frame < 0
                or type(depth) is not int or depth not in DEPTHS
                or row.get("cache_len") != depth + 1
                or row.get("request_hash") != v["request_hash"]
                or row.get("prompt_id") != v["prompt_id"]
                or row.get("requested_seed") != v["seed"]
                or row.get("request_nonce") != hashlib.sha256(
                    f"{sha}|{case}|p18a-v1".encode()
                ).hexdigest()
                or row.get("uniform_seed") != seed_for(
                    sha, case, frame, depth
                )
                or row.get("sampling", {}).get("top_k") != 50
                or abs(row.get("sampling", {}).get("temperature", 0) - 0.9) > 1e-5
                or abs(row.get("sampling", {}).get("top_p", 0) - 1.0) > 1e-5
                or set(row.get("widths", {})) != {
                    str(w) for w in widths(depth)
                }):
            raise RuntimeError(f"P18A_BAD_ROW_PROVENANCE: {path}")
        marker = (case, frame, depth)
        if marker in unique:
            raise RuntimeError("P18A_DUPLICATE_NATIVE_CHECKPOINT")
        unique.add(marker)
        case_frames[case].add(frame)
        case_requests[case].add(row["request_hash"])
        case_native_seeds[case].add(row["native_sub_sampling_seed"])
        pids.add(row["pid"])
        for sw, evidence in row["widths"].items():
            width = int(sw)
            prefix = evidence.get("accepted_prefix")
            flags = evidence.get("accepted_flags")
            if (evidence.get("status")
                != "INDEPENDENT_P17_DISTRIBUTION_AND_KV_PARITY_PASS"
                or type(prefix) is not int or not 0 <= prefix < width
                or type(flags) is not list or len(flags) != width - 1
                or any(type(flag) is not bool for flag in flags)
                or evidence.get("committed_count") != prefix + 1
                or evidence.get("bonus_committed") is not (prefix == width - 1)
                or evidence.get("first_rejection") != (
                    prefix if prefix < width - 1 else None
                )
                or prefix != sum(1 for flag in flags[:prefix] if flag)
                or (prefix < width - 1 and flags[prefix])):
                raise RuntimeError(
                    f"P18A_BAD_INDEPENDENT_WIDTH_ACCOUNTING: {path}, K={width}"
                )
        rows.append(row)
    if (len(pids) != 1 or
            any(len(case_frames[c]) != frames
                or len(case_requests[c]) != 1
                or {
                    (r["frame"], r["last_committed_depth"])
                    for r in rows if r["case_id"] == c
                } != {(f, d) for f in case_frames[c] for d in DEPTHS}
                for c in known_cases)):
        raise RuntimeError("P18A_PID_OR_REQUEST_FRAME_DEPTH_PROVENANCE_BAD")
    hashes = [v["request_hash"] for v in known_cases.values()]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("P18A_REQUEST_IDS_NOT_INDEPENDENT")
    if any(len(case_native_seeds[c]) != 1 for c in known_cases):
        raise RuntimeError("P18A_NATIVE_SUBSAMPLING_SEED_DRIFT_WITHIN_REQUEST")
    per_prompt = defaultdict(dict)
    for case, visit in known_cases.items():
        per_prompt[visit["prompt_id"]][visit["seed"]] = next(
            iter(case_native_seeds[case])
        )
    if any(len(observed) != len(set(observed.values()))
           for observed in per_prompt.values()):
        raise RuntimeError(
            "P18A_REQUESTED_SEEDS_DID_NOT_PRODUCE_DISTINCT_NATIVE_SAMPLER_SEEDS"
        )
    return driver, rows


def analyze(root: Path, out: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    out = out.expanduser().resolve()
    if out.exists():
        raise FileExistsError("P18A analysis must use an absent output dir")
    driver, rows = validate_rows(root)
    groups = defaultdict(list)
    for row in rows:
        groups[row["last_committed_depth"]].append(row)
    result = {
        "status": RESULT, "source_sha": driver["source_sha"],
        "mode": driver["mode"],
        "trajectory": "NATIVE_AR_EAGER_NOT_GRAPH_ON",
        "independent_units": "REQUEST (prompt_id + seed)",
        "independent_requests": len(driver["visits"]),
        "frames_per_request_correlated": driver["frames_per_request"],
        "total_native_checkpoint_rows": len(rows),
        "bootstrap": {"method": "cluster resample request IDs",
                      "resamples": BOOTSTRAPS, "confidence": 0.95},
        "depths": {},
    }
    for d in DEPTHS:
        drows = groups[d]
        if len(drows) != len(driver["visits"]) * driver["frames_per_request"]:
            raise RuntimeError("P18A_UNBALANCED_DEPTH_COHORT")
        result["depths"][str(d)] = {}
        for w in widths(d):
            cases = [r["widths"][str(w)] for r in drows]
            first = sum(bool(e["accepted_flags"][0]) for e in cases)
            full = sum(e["accepted_prefix"] == w - 1 for e in cases)
            hist = Counter(e["accepted_prefix"] for e in cases)
            per_request = defaultdict(list)
            for row in drows:
                per_request[row["case_id"]].append(
                    row["widths"][str(w)]["committed_count"]
                )
            result["depths"][str(d)][str(w)] = {
                "checkpoints_NOT_independent": len(cases),
                "independent_requests": len(per_request),
                "first_accept_rate_per_checkpoint": first / len(cases),
                "full_draft_block_accept_rate_per_checkpoint": full / len(cases),
                "prefix_histogram": {
                    str(n): hist[n] for n in range(w)
                },
                "mean_committed": clustered_ci(
                    per_request, seed=20261123 + 101*d + 17*w,
                ),
            }
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(
        result, indent=2, sort_keys=True
    ) + "\n")
    table = [
        "# P18A — Native AR on-policy all-depth acceptance census",
        "",
        f"**{RESULT}**",
        "",
        f"Source SHA: {result['source_sha']}",
        f"Mode: {result['mode']}; independent request clusters: "
        f"{result['independent_requests']}; correlated frames/request: "
        f"{result['frames_per_request_correlated']}",
        "Native AR EAGER to expose true checkpoints; no Graph-ON serving cost.",
        "",
        "| Last committed depth | Width | Mean committed | "
        "Request-cluster bootstrap 95% CI | First accept | "
        "Full block accept |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for d in DEPTHS:
        if not widths(d):
            table.append(f"| {d} | AR only | — | — | — | — |")
        for w in widths(d):
            v = result["depths"][str(d)][str(w)]
            mean = v["mean_committed"]
            lo, hi = mean["cluster_bootstrap_95pct"]
            table.append(
                f"| {d} | K={w} | {mean['mean']:.3f} | "
                f"[{lo:.3f}, {hi:.3f}] | "
                f"{v['first_accept_rate_per_checkpoint']:.1%} | "
                f"{v['full_draft_block_accept_rate_per_checkpoint']:.1%} |"
            )
    table += [
        "",
        "## Interpretation guardrails",
        "",
        "- Each row is a real native-AR-EAGER sampled checkpoint. Widths share "
        "the same checkpoint and one predeclared uniform tuple.",
        "- Every K2/K3/K4 case individually passed P17's native LM-head, "
        "probability, static verifier, CPU accept/residual and physical KV oracle.",
        "- Resampling is BY REQUEST. Multiple acoustic frames and widths "
        "from the same request are correlated, not independent samples.",
        "- Tail depth 14 has no legal K2 bonus head and must remain AR.",
        "- These are native-AR counterfactual outcomes, not the distribution "
        "under a deployed speculative decoding policy after its state shifts.",
        "- No native Graph-ON step costs, full-frame speedup, HTTP QPS, "
        "or speech quality is measured. Do not ship or merge based on this.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(table))
    print(f"P18A_STATUS={RESULT}")
    print(f"P18A_REPORT={out/'summary.md'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    analyze(a.root, a.out)


if __name__ == "__main__":
    main()

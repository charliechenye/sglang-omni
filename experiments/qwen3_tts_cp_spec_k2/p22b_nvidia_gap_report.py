# SPDX-License-Identifier: Apache-2.0
"""P22B evidence-locked Nvidia implementation-gap report; NO Nvidia H200 speedup.

NVIDIA source facts are pinned to public repo paths + content blob hashes
verified at design time. Our perf data comes from source-pinned P21/P22/P22B
reports; native profile uses a different instrumentation SHA, hence these
are architecture comparisons, NOT matched CUDA kernel speed benchmarks.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OUR_API_STATUS = "P22B_CUDA_API_NVTX_ATTRIBUTION_NOT_CRITICAL_PATH"
NATIVE_API_STATUS = "P22B_NATIVE_GRAPH_LAUNCH_API_CAPTURED_GPU_WORK_NOT_CODEPREDICTOR_ISOLATED"
NVIDIA_CPP_URL = (
    "https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/"
    "cpp/multimodal/qwen3_omni/codePredictor.cpp"
)
NVIDIA_CUDA_URL = (
    "https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/"
    "cpp/kernels/speculative/cpSpecKernels.cu"
)
NVIDIA_DOC_URL = (
    "https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/"
    "docs/source/user_guide/features/codepredictor-speculative-decoding.md"
)
STATUS = "P22B_NVIDIA_IMPLEMENTATION_GAP_IDENTIFIED_NO_NVIDIA_H200_SPEEDUP_PROOF"


def _load(path: Path) -> dict:
    d = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(d, dict):
        raise RuntimeError("P22B_REPORT_NOT_OBJECT")
    return d


def report(k2_api: dict, native_api: dict, p22: dict, p21: dict | None, k2_gpu: dict | None = None) -> dict:
    if (k2_api.get("status") != OUR_API_STATUS
            or native_api.get("status") != NATIVE_API_STATUS
            or p22.get("status") !=
               "P22_ATTRIBUTION_COMPLETE_NOT_PERFORMANCE_BENCHMARK"):
        raise RuntimeError("P22B_REQUIRED_PROFILE_GATES_MISSING")
    if k2_api.get("source_sha") != p22.get("source_sha"):
        raise RuntimeError("P22B_K2_API_AND_GPU_SOURCE_SHA_MISMATCH")
    frames = k2_api.get("real_frames")
    rounds = k2_api.get("real_k2_rounds")
    if (type(frames) is not int or type(rounds) is not int
            or frames != p22.get("candidate_frames")
            or rounds != p22.get("k2_rounds")):
        raise RuntimeError("P22B_REAL_K2_FRAME_COVERAGE_FAIL")
    nframes = native_api.get("native_graph_frames")
    if type(nframes) is not int or nframes < 2:
        raise RuntimeError("P22B_NATIVE_GRAPH_FRAME_COVERAGE_FAIL")
    bstage = k2_api["stages"]
    def sum_stage(category):
        return {
            "calls": sum(
                groups.get(category, {}).get("calls", 0)
                for groups in bstage.values()
            ),
            "cpu_ms": sum(
                groups.get(category, {}).get("summed_cpu_api_ms", 0)
                for groups in bstage.values()
            ),
        }
    b_counts = {}
    for category in ("kernel_launch", "graph_launch",
                     "memcpy", "sync_or_wait", "memset"):
        counted = sum_stage(category)
        counted["calls_per_frame"] = counted["calls"] / frames
        counted["cpu_ms_per_frame"] = counted["cpu_ms"] / frames
        b_counts[category] = counted
    native_graph = {
        "launch_calls": native_api["native_graph_launch_api_calls"],
        "calls_per_frame":
            native_api["native_graph_launch_api_calls"] / nframes,
        "launch_cpu_ms_per_frame":
            native_api["native_graph_launch_api_cpu_per_frame_ms"],
        "host_predictor_cpu_ms_per_frame":
            native_api["native_graph_cpu_predictor_per_frame_ms"],
    }
    k2_gpu_summary = None
    if k2_gpu is not None:
        if (k2_gpu.get("source_sha") != k2_api.get("source_sha")
                or k2_gpu.get("profiled_candidate_frames") != frames
                or k2_gpu.get("profiled_k2_rounds") != rounds
                or k2_gpu.get("status") !=
                   "P22_NSYS_SOURCE_SCOPED_KERNEL_WORK_ATTRIBUTION_NOT_GPU_ELAPSED"):
            raise RuntimeError("P22B_K2_GPU_KERNEL_SOURCE_OR_SCOPE_MISMATCH")
        stages = k2_gpu.get("stage_sums", {})
        if not all(name in stages for name in (
            "P22_Q_PROPOSALS", "P22_WIDE_VERIFIER",
            "P22_TARGET_SUPPORT_ACCEPT_BONUS"
        )):
            raise RuntimeError("P22B_K2_GPU_REQUIRED_STAGES_MISSING")
        k2_gpu_summary = {
            "assigned_kernel_work_ms_per_k2_frame": sum(
                x["summed_gpu_work_ms_per_frame"] for x in stages.values()
            ),
            "stage_kernel_work_ms_per_frame": {
                name: row["summed_gpu_work_ms_per_frame"]
                for name, row in stages.items()
            },
            "scope": "SUMMED_ASSIGNED_GPU_KERNEL_DURATIONS_NOT_GPU_ELAPSED",
        }
    p21_summary = None
    if p21 is not None:
        if (p21.get("status") !=
              "P21_REAL_SERVING_C1_ABBA_SCREEN_NOT_QUALITY_OR_PRODUCTION_GO"
                or not p21.get("paired")
                or p21.get("quality_certificate") !=
                   "NONE_NOT_ASR_OR_SPEAKER_EQUIVALENCE"):
            raise RuntimeError("P22B_P21_ABBA_EVIDENCE_UNQUALIFIED")
        vals = [r["b_over_a"] for r in p21["paired"]]
        p21_summary = {
            "source_sha": p21["source_sha"],
            "median_case_b_over_a": statistics.median(vals),
            "A_native_http_p50_s": p21["overall"]["A"]["e2e_wall_p50_s"],
            "B_k2_http_p50_s": p21["overall"]["B"]["e2e_wall_p50_s"],
            "not_same_source_as_new_P22B": True,
        }
    return {
        "schema": "qwen3tts-p22b-nvidia-implementation-gap-v1",
        "status": STATUS,
        "evidence": {
            "k2_source_sha": k2_api["source_sha"],
            "native_source_sha": native_api["source_sha"],
            "native_vs_k2_is_not_same_sha": (
                native_api["source_sha"] != k2_api["source_sha"]
            ),
            "real_k2_frames": frames,
            "real_k2_rounds": rounds,
            "rounds_per_frame": rounds / frames,
            "k2_gpu_kernel_work": k2_gpu_summary,
            "native_gpu_kernel_work_scope":
                "WHOLE_CAPTURE_NOT_ISOLATED_TO_CODEPREDICTOR",
            "P21_historical_unprofiled_http": p21_summary,
        },
        "our_k2_cpu_api_per_real_frame": b_counts,
        "native_graph_cpu_api_per_real_frame": native_graph,
        "nvidia_source": {
            "cpp_url": NVIDIA_CPP_URL,
            "cpp_blob_sha": "59310047ad7bc0a8b5631108e66e07a9940ab2be",
            "cuda_url": NVIDIA_CUDA_URL,
            "cuda_blob_sha": "f4550d31c51fdcd145560229e8cc66f9a2d77361",
            "doc_url": NVIDIA_DOC_URL,
            "doc_blob_sha": "abb63bf65c1cba47f790e8bbe03bfe8daf1623a4",
            "benchmarked_on_our_H200": False,
        },
        "source_verified_differences": [
            {
                "feature": "control_loop",
                "nvidia": "C++ host loop, one pinned host accept-length sync per round",
                "omni": "Python dynamic-depth loop, multiple Python tensor operations",
            },
            {
                "feature": "width",
                "nvidia": "K 2..8, K3 in docs, utility saturates beyond ~4",
                "omni": "real serving K2 only; K3/K4 counterfactual probes",
            },
            {
                "feature": "verifier",
                "nvidia": "TensorRT generation-profile context-chunk and captured Graph for seqLen 1..K",
                "omni": "5-layer fused eager multi-row verifier, many launches each round",
            },
            {
                "feature": "head_and_sampler",
                "nvidia": "grouped LM-head linear + CUDA top-k/top-p + fused accept residual/bonus",
                "omni": "torch.bmm and multiple Triton/PyTorch sampler/embedding/gather launches",
            },
            {
                "feature": "scratch_and_rng",
                "nvidia": "preallocated device/host buffers and counter-based uniform pool",
                "omni": "per-round allocations and private RNG; native distribution parity not proven",
            },
            {
                "feature": "precision",
                "nvidia": "TensorRT FP16, optional FP8/NVFP4 CodePredictor",
                "omni": "H200 BF16 fused predictor in P22; not precision-matched",
            },
        ],
        "decisions": [
            "NVIDIA K3 width and C++ host loop can be tested before full-GPU dynamic-depth Graph",
            "First target launch count/CPU API overhead, grouped heads and sampler fusion",
            "Measure native Graph GPU child-node scope before GPU-work speedup inference",
            "No numerical NVIDIA/Omni speedup without identical model/hardware/sampling/precision"
        ],
    }


def markdown(d):
    rows = [
        "# P22B — Why NVIDIA's CodePredictor SpecDec architecture differs",
        "",
        "**Source-audited implementation gap; NOT a direct NVIDIA H200 benchmark.**",
        "",
        f"Omni K2 profile SHA: `{d['evidence']['k2_source_sha']}`",
        f"Native A profile SHA: `{d['evidence']['native_source_sha']}`",
        f"K2 rounds/frame: **{d['evidence']['rounds_per_frame']:.3f}**",
        "",
        ("K2 attributed GPU kernel work/frame (summed, NOT elapsed): "
         + f"**{d['evidence']['k2_gpu_kernel_work']['assigned_kernel_work_ms_per_k2_frame']:.3f}ms**"
         if d['evidence']['k2_gpu_kernel_work'] is not None
         else "No separate K2 GPU kernel-work JSON was supplied."),
        "",
        "| Feature | NVIDIA TensorRT Edge-LLM | Omni P21/P22 |",
        "|---|---|---|",
    ]
    for row in d["source_verified_differences"]:
        rows.append(f"| {row['feature']} | {row['nvidia']} | {row['omni']} |")
    rows += [
        "",
        "## Our K2 CUDA APIs (CPU-side summed duration, NOT critical path)",
        "",
        "| Category | Calls / real K2 frame | CPU API ms / real K2 frame |",
        "|---|---:|---:|",
    ]
    for name, row in d["our_k2_cpu_api_per_real_frame"].items():
        rows.append(f"| {name} | {row['calls_per_frame']:.2f} "
                    f"| {row['cpu_ms_per_frame']:.4f} |")
    rows += [
        "",
        "## Native graph (separate fresh-worker profiled run)",
        "",
        f"Graph launch API calls per native frame: "
        f"**{d['native_graph_cpu_api_per_real_frame']['calls_per_frame']:.3f}**",
        f"Graph launch CPU API ms/frame: "
        f"**{d['native_graph_cpu_api_per_real_frame']['launch_cpu_ms_per_frame']:.4f}**",
        "",
        "**Do not compare trace-wide native GPU kernel work to B's "
        "CodePredictor-scoped kernel work as if they were matched.**",
        "",
        "## NVIDIA source references",
        "",
        f"- [CodePredictor C++]({NVIDIA_CPP_URL})",
        f"- [Speculative CUDA kernels]({NVIDIA_CUDA_URL})",
        f"- [Official algorithm / limitations]({NVIDIA_DOC_URL})",
        "",
        "## Decision",
        "",
    ]
    rows.extend("- " + item for item in d["decisions"])
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k2-api", type=Path, required=True)
    ap.add_argument("--native-api", type=Path, required=True)
    ap.add_argument("--p22-analysis", type=Path, required=True)
    ap.add_argument("--k2-gpu", type=Path)
    ap.add_argument("--p21-abba", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError("P22B_GAP_REFUSES_OVERWRITE")
    result = report(
        _load(args.k2_api), _load(args.native_api), _load(args.p22_analysis),
        _load(args.p21_abba) if args.p21_abba is not None else None,
        _load(args.k2_gpu) if args.k2_gpu is not None else None,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output / "summary.md").write_text(markdown(result))
    print("P22B_NVIDIA_GAP_STATUS=" + result["status"])
    print("P22B_NVIDIA_GAP_REPORT=" + str(output / "summary.md"))


if __name__ == "__main__":
    main()

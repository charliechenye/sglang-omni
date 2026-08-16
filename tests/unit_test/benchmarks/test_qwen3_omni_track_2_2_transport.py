import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = REPOSITORY_ROOT / "benchmarks" / "qwen3_omni_track_2_2_transport.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_track_2_2_transport_test_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def _metric(median: float) -> dict[str, float]:
    return {
        "count": 10,
        "median": median,
        "p25": median,
        "p75": median,
        "p95": median,
        "mad": 0.0,
        "mean": median,
    }


def _fake_runs() -> list[dict[str, object]]:
    values = {
        "A": (100.0, 100.0, 100.0, 100.0),
        "B": (90.0, 90.0, 100.0, 100.0),
        "C": (80.0, 80.0, 100.0, 100.0),
        "D": (120.0, 120.0, 100.0, 100.0),
    }
    transports = {"A": "cuda_ipc", "B": "cuda_ipc", "C": "cuda_ipc", "D": "shm"}
    runs: list[dict[str, object]] = []
    for direction in ("forward", "reverse"):
        for round_index in range(5):
            for concurrency in (1, 8, 32):
                for arm, (publish, ack, publish_rate, drain_rate) in values.items():
                    runs.append(
                        {
                            "run_key": (
                                f"{direction}-round{round_index}-c{concurrency}-{arm}"
                            ),
                            "arm": arm,
                            "direction": direction,
                            "round": round_index,
                            "concurrency": concurrency,
                            "publish_latency_ms": _metric(publish),
                            "ack_release_latency_ms": _metric(ack),
                            "publish_chunks_per_sec": publish_rate,
                            "end_to_end_drain_chunks_per_sec": drain_rate,
                            "final_ack_drain_ms": 1.0,
                            "maximum_outstanding_transfer_count": 10,
                            "logical_tensor_bytes_per_chunk": 1,
                            "tensor_transfers_per_chunk": 1,
                            "transport": transports[arm],
                            "producer_mode": "single_round_robin",
                            "transport_check_passed": True,
                        }
                    )
    return runs


class Track22TransportBenchmarkTest(unittest.TestCase):
    def test_checkout_provenance_and_help(self) -> None:
        source = Path(BENCHMARK._SGLANG_OMNI_SOURCE)
        self.assertTrue(source.is_relative_to(REPOSITORY_ROOT / "sglang_omni"))
        self.assertEqual(BENCHMARK._REPOSITORY_ROOT, REPOSITORY_ROOT)
        result = subprocess.run(
            [sys.executable, str(BENCHMARK_PATH), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cuda-ipc-pool-mb", result.stdout)

    def test_round_robin_planning(self) -> None:
        self.assertEqual(BENCHMARK._distribute_chunks(10, 3), [4, 3, 3])
        self.assertEqual(
            BENCHMARK._round_robin_chunk_plan([2, 2, 1]),
            [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
        )
        specs = BENCHMARK._build_run_specs(
            direction="forward",
            arm_order=("A", "B", "C", "D"),
            concurrency_values=(1, 8, 32),
            rounds=2,
            warmup_chunks=100,
            measure_chunks=2_000,
        )
        self.assertEqual(len(specs), 24)
        self.assertEqual([spec.arm for spec in specs[:4]], ["A", "B", "C", "D"])

    def test_aggregation_and_direct_recommendation(self) -> None:
        runs = _fake_runs()
        aggregates = BENCHMARK._aggregate_runs(runs)
        self.assertEqual(len(aggregates), 12)
        arm_a_c1 = next(
            cell
            for cell in aggregates
            if cell["arm"] == "A" and cell["concurrency"] == 1
        )
        self.assertEqual(arm_a_c1["publish_latency_ms"]["median"], 100.0)
        self.assertEqual(arm_a_c1["maximum_outstanding_transfer_count"], 10)

        comparisons = BENCHMARK._build_comparisons(runs)
        self.assertEqual(comparisons["incremental"]["C->D"]["decision"], "NO-GO")
        self.assertEqual(comparisons["direct_from_A"]["A->C"]["decision"], "GO")
        self.assertEqual(comparisons["recommendation"]["recommended_candidate"], "C")
        self.assertEqual(comparisons["overall"], "GO")

    def test_variability_blocks_nominal_effect(self) -> None:
        pairs = []
        for index in range(10):
            baseline = _fake_runs()[index]
            candidate = dict(baseline)
            candidate["publish_latency_ms"] = _metric(94.0 if index % 2 == 0 else 106.0)
            pairs.append((baseline, candidate))
        metrics = BENCHMARK._paired_metrics(pairs)
        self.assertNotEqual(
            BENCHMARK._status_for_paired_metrics(metrics),
            "GO",
        )


if __name__ == "__main__":
    unittest.main()

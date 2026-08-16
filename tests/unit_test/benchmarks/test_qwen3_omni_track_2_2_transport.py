import asyncio
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        "B": (90.0, 99.0, 100.0, 100.0),
        "C": (80.0, 99.0, 100.0, 100.0),
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
        self.assertEqual(comparisons["direct_from_A"]["A->D"]["decision"], "NO-GO")

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

    def test_non_regression_threshold_is_not_mad_gated(self) -> None:
        def stats(delta: float, mad: float) -> dict[str, float]:
            return {
                "median_paired_percentage_delta": delta,
                "mad_paired_percentage_delta": mad,
            }

        self.assertTrue(BENCHMARK._stable_non_regression(stats(-1.0, 100.0)))
        self.assertTrue(BENCHMARK._stable_non_regression(stats(-4.0, 100.0)))
        self.assertFalse(BENCHMARK._stable_non_regression(stats(-6.0, 100.0)))

    def test_bf16_expected_pattern_uses_bf16_construction(self) -> None:
        class FakeTensor:
            def __init__(self, values: list[float], dtype: object, device: str):
                self.values = values
                self.dtype = dtype
                self.device = device

        class FakeTorch:
            bfloat16 = object()

            def __init__(self) -> None:
                self.calls: list[tuple[int, object, str]] = []

            def arange(self, length: int, *, dtype: object, device: str) -> FakeTensor:
                self.calls.append((length, dtype, device))
                values = [float(index) for index in range(length)]
                values[-1] = 2048.0
                return FakeTensor(values, dtype, device)

        fake_torch = FakeTorch()
        original_torch = BENCHMARK.torch
        original_load_runtime = BENCHMARK._load_runtime
        BENCHMARK.torch = fake_torch
        BENCHMARK._load_runtime = lambda: None
        try:
            expected = BENCHMARK._expected_bf16_pattern(device="cpu")
        finally:
            BENCHMARK.torch = original_torch
            BENCHMARK._load_runtime = original_load_runtime

        self.assertEqual(
            fake_torch.calls,
            [(2048, fake_torch.bfloat16, "cpu")],
        )
        self.assertEqual(expected.values[-1], 2048.0)
        self.assertNotEqual(expected.values[-1], 2047.0)

    def test_numeric_stats_use_conventional_quantiles_and_mad(self) -> None:
        two_values = BENCHMARK._numeric_stats([0.2, 5.0])
        assert two_values is not None
        self.assertEqual(two_values["median"], 2.6)
        self.assertEqual(two_values["mad"], 2.4)

        one_value = BENCHMARK._numeric_stats([7.25])
        assert one_value is not None
        for field in ("median", "p25", "p75", "p95"):
            self.assertEqual(one_value[field], 7.25)
        self.assertEqual(one_value["mad"], 0.0)

        even_values = BENCHMARK._numeric_stats([1.0, 2.0, 3.0, 4.0])
        assert even_values is not None
        self.assertEqual(even_values["median"], 2.5)
        self.assertEqual(even_values["p25"], 1.75)
        self.assertEqual(even_values["p75"], 3.25)

        odd_values = BENCHMARK._numeric_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert odd_values is not None
        self.assertEqual(odd_values["median"], 3.0)

    def test_paired_effect_stats_interpolate_two_observations(self) -> None:
        pairs = [
            (
                {"publish_latency_ms": _metric(100.0)},
                {"publish_latency_ms": _metric(90.0)},
            ),
            (
                {"publish_latency_ms": _metric(100.0)},
                {"publish_latency_ms": _metric(80.0)},
            ),
        ]
        stats = BENCHMARK._paired_metric_stats(pairs, "publish_latency_ms")
        self.assertAlmostEqual(stats["median_paired_percentage_delta"], 15.0)
        self.assertAlmostEqual(stats["mad_paired_percentage_delta"], 5.0)
        self.assertAlmostEqual(stats["p25_paired_percentage_delta"], 12.5)
        self.assertAlmostEqual(stats["p75_paired_percentage_delta"], 17.5)


class Track22TransportAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def _bare_control(self) -> tuple[object, asyncio.Future[int]]:
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        completion: asyncio.Future[int] = loop.create_future()
        control = object.__new__(BENCHMARK._BenchmarkControlPlane)
        control._object_ids = {("run:req0", 0): "object-0"}
        control._object_keys = {"object-0": ("run:req0", 0)}
        control._ready = {"object-0": ready}
        control._complete = {"object-0": completion}
        control._outstanding = set()
        control._max_outstanding = 0
        control._fatal = None
        return control, completion

    async def test_max_outstanding_resets_at_measurement_boundary(self) -> None:
        loop = asyncio.get_running_loop()
        control = object.__new__(BENCHMARK._BenchmarkControlPlane)
        control._object_ids = {}
        control._object_keys = {}
        control._ready = {}
        control._complete = {}
        control._outstanding = set()
        control._max_outstanding = 0
        control._fatal = None

        def add_object(index: int) -> str:
            object_id = f"object-{index}"
            key = ("run:req0", index)
            control._object_ids[key] = object_id
            control._object_keys[object_id] = key
            control._ready[object_id] = loop.create_future()
            control._complete[object_id] = loop.create_future()
            return object_id

        def release_and_retire(object_id: str) -> None:
            control._outstanding.remove(object_id)
            control._complete[object_id].set_result(1)
            control.retire(object_id)

        warmup_objects = [add_object(index) for index in range(32)]
        for object_id in warmup_objects:
            control.mark_ready(object_id)
        self.assertEqual(control.max_outstanding, 32)
        with self.assertRaisesRegex(RuntimeError, "live objects"):
            control.reset_max_outstanding()

        for object_id in warmup_objects:
            release_and_retire(object_id)
        self.assertEqual(control._outstanding, set())

        control.reset_max_outstanding()
        self.assertEqual(control.max_outstanding, 0)
        measured_objects = [add_object(index) for index in range(3)]
        for object_id in measured_objects:
            control.mark_ready(object_id)
        reported_peak = control.max_outstanding
        self.assertEqual(reported_peak, 3)

        for object_id in measured_objects:
            release_and_retire(object_id)

    async def test_ack_release_timestamp_and_retirement_are_distinct(self) -> None:
        control, completion = await self._bare_control()
        release_gate = asyncio.Event()

        async def pending_task() -> None:
            await release_gate.wait()

        engine = SimpleNamespace(
            _pending={
                "object-0": SimpleNamespace(task=asyncio.create_task(pending_task()))
            },
            acked=None,
        )

        def ack_transfer(message: object) -> None:
            engine.acked = message

        engine.ack_transfer = ack_transfer
        resolve_task = asyncio.create_task(
            control._resolve_ack(SimpleNamespace(object_id="object-0"), engine)
        )
        await asyncio.sleep(0)
        self.assertFalse(resolve_task.done())
        self.assertIsNone(engine.acked)

        self.assertIs(control.mark_ready("object-0"), completion)
        self.assertEqual(control.max_outstanding, 1)
        self.assertIn("object-0", control._outstanding)
        release_gate.set()
        release_ns = 1_234_567_890
        with patch.object(BENCHMARK.time, "perf_counter_ns", return_value=release_ns):
            await resolve_task

        self.assertEqual(completion.result(), release_ns)
        self.assertNotIn("object-0", control._outstanding)
        self.assertIn("object-0", control._complete)
        control.retire("object-0")
        self.assertNotIn("object-0", control._object_keys)
        self.assertNotIn("object-0", control._ready)
        self.assertNotIn("object-0", control._complete)
        self.assertEqual(control.max_outstanding, 1)

    async def test_drain_uses_release_timestamp_without_observing_completion_time(
        self,
    ) -> None:
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[int] = loop.create_future()
        release_ns = 1_002_000_000
        completion.set_result(release_ns)
        record = BENCHMARK._PendingSend(
            object_id="object-0",
            start_ns=1_000_000_000,
            publish_end_ns=1_000_500_000,
            phase="measure",
            completion=completion,
        )

        class FakeControl:
            def fatal_future(self) -> None:
                return None

            def retire(self, object_id: str) -> None:
                self.retired = object_id

        control = FakeControl()
        ack_latencies_ms: list[float] = []
        with patch.object(
            BENCHMARK.time,
            "perf_counter_ns",
            side_effect=AssertionError("drain must not sample observation time"),
        ):
            latest_release_ns = await BENCHMARK._drain_completion_records(
                [record], control, ack_latencies_ms
            )

        self.assertEqual(latest_release_ns, release_ns)
        self.assertEqual(ack_latencies_ms, [2.0])
        self.assertEqual(control.retired, "object-0")


if __name__ == "__main__":
    unittest.main()

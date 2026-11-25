"""Check the EdgeCrafter speed adapter's inputs and receipt on CPU."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkCase,
    BenchmarkProtocol,
    BenchmarkReport,
    Status,
)
from src.utils.inference_benchmark_inputs import TimingSchedule


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    runs = Path(__file__).resolve().parents[3] / "knowledge/runs"
    comparison = runs / "run-meiji-player-model-comparison"
    monkeypatch.syspath_prepend(str(runs / "run-person-model-speed-baseline"))
    monkeypatch.syspath_prepend(str(comparison))
    return runpy.run_path(str(comparison / "edgecrafter_speed.py"))


def test_case_uses_ordered_full_images_and_explicit_parity_opt_out(
    script: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    one_case = script["one_case"]
    scope = one_case.__globals__
    threads = {"torch": 2, "opencv": 8}
    monkeypatch.setitem(
        scope,
        "torch",
        SimpleNamespace(
            device=scope["torch"].device,
            set_num_threads=lambda count: threads.__setitem__("torch", count),
            get_num_threads=lambda: threads["torch"],
        ),
    )
    monkeypatch.setitem(
        scope,
        "cv2",
        SimpleNamespace(
            setNumThreads=lambda count: threads.__setitem__("opencv", count),
            getNumThreads=lambda: threads["opencv"],
        ),
    )
    monkeypatch.setitem(scope, "_require_harness_revision", lambda revision: None)
    for key, value in (("first", 12), ("second", 87)):
        np.save(tmp_path / f"{key}.npy", np.full((2, 3, 3), value, dtype=np.uint8))
    keys = ("second", "first")
    schedule = TimingSchedule(BenchmarkProtocol((1, 2), 2, 3, 1), (keys,), "c" * 64)
    pins = SimpleNamespace(
        manifest_sha256="a" * 64,
        timing_sha256="b" * 64,
        manifest={"id": "fixed-speed-fixture"},
        schedule=lambda *_: schedule,
        rows=lambda _: [{"id": key, "shape": [2, 3, 3]} for key in keys],
        verified_asset=lambda row: tmp_path / f"{row['id']}.npy",
    )
    monkeypatch.setitem(scope, "PinnedBenchmarkInputs", lambda *a, **kw: pins)
    adapter = object()
    monkeypatch.setitem(scope, "make_benchmark_adapter", lambda *a, **kw: adapter)
    monkeypatch.setitem(scope, "_adapter_with_receipt", lambda value: value)

    def run_benchmark(
        case: BenchmarkCase,
        batches: list[Batch[Any]],
        backends: dict[Backend, object],
        **settings: Any,
    ) -> BenchmarkReport:
        assert case.backend is Backend.PT16
        assert case.input_manifest_sha256 == "a" * 64
        assert case.settings["timed_input_ids_sha256"] == "c" * 64
        assert threads == {"torch": 4, "opencv": 1}
        assert backends == {Backend.PT16: adapter}
        assert [batch.ids for batch in batches] == [("second", "first")]
        assert [image[0, 0, 0] for image in batches[0].data] == [87, 12]
        assert settings["protocol"] is schedule.protocol
        assert settings["dedicated_stream"]
        assert settings.get("check_parity") is None
        assert settings["parity_not_evaluated_reason"] == "PT32未計測"
        return BenchmarkReport(
            case,
            schedule.protocol,
            (),
            parity_not_evaluated_reason=settings["parity_not_evaluated_reason"],
        )

    monkeypatch.setitem(scope, "benchmark", run_benchmark)
    assert (
        one_case(
            argparse.Namespace(
                backend="pt16",
                parity="not_measured",
                harness_commit="d" * 40,
                model="ecdet-l-coco",
                device="cuda:0",
                assets=tmp_path,
                engine=None,
                inputs=tmp_path,
                inputs_sha256="a" * 64,
                timing_protocol=tmp_path / "timing.json",
                timing_protocol_sha256="b" * 64,
                batch_size=2,
                pose_box_margin=None,
                conversion=[],
                output=tmp_path,
            )
        )
        is Status.OK
    )
    report = json.loads((tmp_path / "speed-b2.json").read_text(encoding="utf-8"))
    assert report["case"]["settings"]["harness_commit"] == "d" * 40
    assert report["case"]["settings"]["host_threads"] == 4
    assert report["case"]["settings"]["opencv_threads"] == 1
    assert report["parity_status"] == "not_evaluated"
    assert report["parity_not_evaluated_reason"] == "PT32未計測"
    assert report["parity"] is None


def test_speed_refuses_a_changed_harness(
    script: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    check = script["_require_harness_revision"]

    def dirty_diff(command: list[str], *, check: bool) -> SimpleNamespace:
        assert command[:5] == ["git", "diff", "--quiet", "--no-ext-diff", "a" * 40]
        assert command[-2:] == [
            "src/utils/inference_benchmark.py",
            "src/utils/inference_benchmark_inputs.py",
        ]
        return SimpleNamespace(returncode=1)

    monkeypatch.setitem(
        check.__globals__, "subprocess", SimpleNamespace(run=dirty_diff)
    )

    with pytest.raises(ValueError, match="The benchmark harness differs"):
        check("a" * 40)

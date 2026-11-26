from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pmpose_speed
import pytest
from numpy.typing import NDArray
from pose_contract import PoseBatch, PoseInput

from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkProtocol,
    Status,
)
from src.utils.inference_benchmark_inputs import TimingSchedule


def _options(output: Path, backend: str) -> list[str]:
    return [
        "--inputs",
        "inputs",
        "--assets",
        "assets",
        "--output",
        str(output),
        "--model",
        "pmpose-l",
        "--backend",
        backend,
        "--batch",
        "1",
    ]


@pytest.mark.parametrize("backend", [Backend.PT16, Backend.TRT16])
def test_cli_binds_pinned_xyxy_and_selected_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: Backend
) -> None:
    protocol = BenchmarkProtocol(batch_sizes=(1, 4, 8), warmup=30)
    schedule = TimingSchedule(protocol, (("b",), ("a",)), "timed_order")
    frames: dict[str, NDArray[np.uint8]] = {
        "a-frame": np.zeros((16, 10, 3), dtype=np.uint8),
        "b-frame": np.ones((16, 10, 3), dtype=np.uint8),
    }
    a_box, b_box = [1.0, 2.0, 3.0, 8.0], [1.5, 1.0, 6.0, 7.0]
    crops = (
        {
            "id": "a",
            "frame_id": "a-frame",
            "xyxy": a_box,
            "xys": [0, 0, 0],
        },
        {
            "id": "b",
            "frame_id": "b-frame",
            "xyxy": b_box,
            "xys": [0, 0, 0],
        },
    )

    def pinned_schedule(index: str, batch_size: int) -> TimingSchedule:
        assert (index, batch_size) == ("crops", 1)
        return schedule

    fixed = SimpleNamespace(
        sha256="a" * 64,
        manifest={"id": "test-crops", "pose_sweep_tail_ids": []},
        pins=SimpleNamespace(timing_sha256="timing_protocol", schedule=pinned_schedule),
        rows={"crops": crops}.__getitem__,
        frame=frames.__getitem__,
    )
    monkeypatch.setattr(pmpose_speed, "InputSet", lambda _: fixed)
    monkeypatch.setattr(pmpose_speed.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(pmpose_speed.cv2, "setNumThreads", lambda _: None)

    def benchmark(
        case: BenchmarkCase,
        batches: Sequence[Batch[tuple[PoseInput, ...]]],
        adapters: Mapping[Backend, BenchmarkAdapter[tuple[PoseInput, ...], PoseBatch]],
        **options: object,
    ) -> SimpleNamespace:
        assert set(adapters) == {backend}
        assert adapters[backend].backend is backend
        assert bool(adapters[backend].engines) == (backend is Backend.TRT16)
        assert case.backend is backend
        assert case.input_manifest_sha256 == fixed.sha256
        assert case.settings["timed_input_ids_sha256"] == schedule.timed_ids_sha256
        assert options["protocol"] is protocol
        assert options["dedicated_stream"] is True
        assert "check_parity" not in options
        assert options["parity_not_evaluated_reason"] == (
            "PT32計測はスコープ外（ユーザー判断 2026-11-25）"
        )
        assert [batch.ids for batch in batches] == [("b",), ("a",)]
        for batch, frame, box in zip(
            batches, (frames["b-frame"], frames["a-frame"]), (b_box, a_box), strict=True
        ):
            assert batch.item_count == batch.processed_items == 1
            (row,) = batch.data
            assert row.frame_bgr is frame
            np.testing.assert_array_equal(row.bbox_xyxy, box)
        return SimpleNamespace(
            status=Status.OK, failure=None, to_dict=lambda: {"status": "ok"}
        )

    monkeypatch.setattr(pmpose_speed, "benchmark", benchmark)
    pmpose_speed.main(_options(tmp_path, backend))
    report = tmp_path / "pmpose-l" / f"{backend}-b1" / "report.json"
    assert json.loads(report.read_text()) == {"status": "ok"}
    with pytest.raises(FileExistsError, match=f"{backend}-b1"):
        pmpose_speed.main(_options(tmp_path, backend))


def test_cli_rejects_pt32_before_loading_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        pmpose_speed.main(_options(tmp_path, "PT32"))
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "invalid choice" in message and "PT32" in message
    assert not list(tmp_path.iterdir())

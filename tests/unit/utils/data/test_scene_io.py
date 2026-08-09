"""Tests for the shared task-local scene-directory reader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.utils.data import load_scene_payload


def test_load_scene_payload_reads_sidecars_and_memory_maps_arrays(
    tmp_path: Path,
) -> None:
    (tmp_path / "scalars.json").write_text(
        json.dumps({"fps": 30, "scene_id": "sample"}),
        encoding="utf-8",
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"camera_count": 2}),
        encoding="utf-8",
    )
    expected: npt.NDArray[np.float32] = np.arange(
        12, dtype=np.float32
    ).reshape(3, 4)
    np.save(tmp_path / "positions.npy", expected)

    payload = load_scene_payload(tmp_path)

    assert payload["fps"] == 30
    assert payload["scene_id"] == "sample"
    assert payload["meta"] == {"camera_count": 2}
    assert isinstance(payload["positions"], np.memmap)
    np.testing.assert_array_equal(payload["positions"], expected)


def test_load_scene_payload_keeps_missing_optional_sidecars_explicitly_absent(
    tmp_path: Path,
) -> None:
    expected = np.array([1, 2, 3], dtype=np.int64)
    np.save(tmp_path / "frame_ids.npy", expected)

    payload = load_scene_payload(tmp_path)

    assert set(payload) == {"frame_ids"}
    np.testing.assert_array_equal(payload["frame_ids"], expected)

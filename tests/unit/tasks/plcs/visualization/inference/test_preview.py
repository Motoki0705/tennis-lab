"""GPU-free scene preview: camera geometry plus ground-truth tracks."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.plcs.visualization.inference.service import (
    InferenceService,
    SceneCatalogError,
)
from src.utils.paths import PROJECT_ROOT

_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "plcs",
    Path("/home/kamimura/projects/tennis-lab/data/plcs"),
)
DATA_ROOT = next((path for path in _DATA_CANDIDATES if path.is_dir()), None)

FORMS = (
    "single_object",
    "single_object_broadcast",
    "single_object_camera_view_v2",
    "multi_object",
    "multi_object_broadcast",
    "multi_object_camera_view_v2",
)
EXPECTED_CAMERAS = {
    "single_object": 6,
    "single_object_broadcast": 2,
    "single_object_camera_view_v2": 4,
    "multi_object": 6,
    "multi_object_broadcast": 2,
    "multi_object_camera_view_v2": 6,
}


def _write_scene_root(data_root: Path, family: str, cameras: int) -> None:
    root = data_root / family
    (root / "scenes" / "scene_000000").mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "court_keypoints": {"selector": "physical_v1"},
                    "generation": {"mode": "single_object"},
                    "camera": {"layout": "fixed"},
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "val.txt").write_text("scene_000000\n", encoding="utf-8")
    (root / "train.txt").write_text("", encoding="utf-8")
    (root / "test.txt").write_text("", encoding="utf-8")
    scene = root / "scenes" / "scene_000000"
    (scene / "meta.json").write_text(
        json.dumps(
            {
                "scene_id": "scene_000000",
                "fps": 120.0,
                "num_frames": 12,
                "court_keypoint_views": [
                    {"camera_id": f"camera_{index}"} for index in range(cameras)
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene / "scalars.json").write_text(
        json.dumps({"num_cameras": cameras, "num_persons": 1}), encoding="utf-8"
    )


def _synthetic_service(tmp_path: Path) -> InferenceService:
    data_root = tmp_path / "data" / "plcs"
    checkpoint_root = tmp_path / "outputs" / "plcs"
    checkpoint_root.mkdir(parents=True)
    _write_scene_root(data_root, "single_object", cameras=6)
    return InferenceService(
        data_root=data_root, checkpoint_root=checkpoint_root, device="cpu"
    )


def test_preview_rejects_bad_camera_window_and_reference(tmp_path: Path) -> None:
    service = _synthetic_service(tmp_path)
    with pytest.raises(SceneCatalogError, match="out of range"):
        service.scene_preview("single_object", "scene_000000", cameras=(0, 9))
    with pytest.raises(SceneCatalogError, match="distinct"):
        service.scene_preview("single_object", "scene_000000", cameras=(1, 1))
    with pytest.raises(SceneCatalogError, match="num_frames"):
        service.scene_preview(
            "single_object", "scene_000000", window_start=99, window_length=4
        )
    with pytest.raises(SceneCatalogError, match="exceeds num_frames"):
        service.scene_preview(
            "single_object", "scene_000000", window_start=10, window_length=10
        )
    with pytest.raises(SceneCatalogError, match="reference_camera_id"):
        service.scene_preview(
            "single_object", "scene_000000", reference_camera_id="camera_9"
        )
    with pytest.raises(SceneCatalogError, match="selected cameras"):
        service.scene_preview(
            "single_object",
            "scene_000000",
            cameras=(0,),
            reference_camera_id="camera_3",
        )


def test_preview_rejects_unknown_scene(tmp_path: Path) -> None:
    service = _synthetic_service(tmp_path)
    with pytest.raises(SceneCatalogError, match="not listed"):
        service.scene_preview("single_object", "scene_999999")
    with pytest.raises(SceneCatalogError, match="unknown scene family"):
        service.scene_preview("nope", "scene_000000")


def _decode(result_bytes: bytes) -> tuple[dict[str, Any], np.ndarray]:
    (header_length,) = struct.unpack("<I", result_bytes[:4])
    header = cast(
        "dict[str, Any]",
        json.loads(result_bytes[4 : 4 + header_length].decode("utf-8")),
    )
    payload = np.frombuffer(result_bytes[4 + header_length :], dtype="<f4")
    return header, payload


def _dataset_root() -> Path:
    assert DATA_ROOT is not None
    return DATA_ROOT


def _repo_root() -> Path:
    # ``DATA_ROOT`` is ``<repo>/data/plcs``; the worktree omits the data tree.
    return _dataset_root().parents[1]


@pytest.mark.skipif(DATA_ROOT is None, reason="PLCS dataset is unavailable")
@pytest.mark.parametrize("form", FORMS)
def test_preview_exposes_gt_and_cameras_for_every_form(form: str) -> None:
    service = InferenceService(
        data_root=_dataset_root(),
        checkpoint_root=_repo_root() / "outputs" / "plcs",
        device="cpu",
        project_root=PROJECT_ROOT,
    )
    header, payload = _decode(
        service.scene_preview(
            form, "scene_000000", window_start=4, window_length=6
        ).to_bytes()
    )
    assert header["mode"] == "preview"
    assert header["checkpoint"] is None
    scene = header["scene"]
    assert scene["family"] == form
    cameras = header["cameras"]
    assert len(cameras) == EXPECTED_CAMERAS[form]
    for camera in cameras:
        assert len(camera["center"]) == 3
        assert np.asarray(camera["frustum"]).shape == (5, 3)
    request = header["request"]
    assert request["window_start"] == 4
    assert request["window_length"] == 6

    tracks = header["tracks"]
    assert len(tracks) == int(scene["num_persons"]) + (
        1 if form.startswith("multi_object") else 0
    )
    for track in tracks:
        assert track["position"]["shape"] == [6, 3]
        assert track["joints"]["shape"] == [6, 17, 3]
        assert track["rotation"]["shape"] == [6, 2]
        assert track["presence"]["shape"] == [6]
    assert int(header["payload_elements"]) == payload.size
    assert np.isfinite(payload).all()


@pytest.mark.skipif(DATA_ROOT is None, reason="PLCS dataset is unavailable")
def test_preview_window_slices_ground_truth_on_real_scene() -> None:
    service = InferenceService(
        data_root=_dataset_root(),
        checkpoint_root=_repo_root() / "outputs" / "plcs",
        device="cpu",
        project_root=PROJECT_ROOT,
    )
    full, _ = _decode(service.scene_preview("multi_object", "scene_000000").to_bytes())
    windowed, _ = _decode(
        service.scene_preview(
            "multi_object", "scene_000000", window_start=10, window_length=5
        ).to_bytes()
    )
    assert full["request"]["window_length"] == full["scene"]["num_frames"]
    assert windowed["request"]["window_start"] == 10
    assert (
        full["tracks"][0]["position"]["shape"]
        != windowed["tracks"][0]["position"]["shape"]
    )
    assert any("のみを表示" in warning for warning in windowed["warnings"])

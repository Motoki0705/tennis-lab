"""Tests for durable court-line model output reuse."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.line_inference_cache import (
    load_or_predict_line_probabilities,
)
from src.synthetic_data_generation.alignment.line_inputs import (
    AlignmentLineInputBatch,
    AlignmentLineInputView,
    CourtLineInputSource,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def test_raw_probabilities_are_reused_and_mirrored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_root = tmp_path / "scenes/B01"
    export_root = scene_root / "reconstruction/export"
    images_root = export_root / "images"
    images_root.mkdir(parents=True)
    cameras: list[SceneCamera] = []
    images: list[NDArray[np.uint8]] = []
    for index in range(2):
        image_path = images_root / f"frame-{index:03d}.png"
        Image.fromarray(
            np.full((4, 6, 3), 20 + index, dtype=np.uint8), mode="RGB"
        ).save(image_path)
        image_rgb: NDArray[np.uint8] = np.full((4, 6, 3), 20 + index, dtype=np.uint8)
        images.append(image_rgb)
        cameras.append(
            SceneCamera(
                camera_id=f"camera-{index}",
                source_frame_index=index,
                width=6,
                height=4,
                intrinsics=(4.0, 0.0, 3.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0),
                camera_to_scene=RigidTransform.identity(),
                image_path=str(image_path),
            )
        )
    scene = SimpleNamespace(scene_id="B01", export_root=export_root)
    inputs = AlignmentLineInputBatch(
        source=CourtLineInputSource.NHT_RENDERED_RGB,
        provenance={"schema": "test_nht_render_v1"},
        views=tuple(
            AlignmentLineInputView(camera=camera, image_rgb=image)
            for camera, image in zip(cameras, images, strict=True)
        ),
    )
    mirror = tmp_path / "drive/B01/court-line-inference"
    monkeypatch.setenv("TENNIS_LAB_ALIGNMENT_INFERENCE_MIRROR_ROOT", str(mirror))
    calls = 0

    def predict(_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        nonlocal calls
        calls += 1
        return np.full((3, 5), calls / 10.0, dtype=np.float32)

    first = load_or_predict_line_probabilities(
        scene=cast(Any, scene),
        inputs=inputs,
        detector_identity={"model": "fixed", "preprocessing": 256},
        predict_probability=predict,
    )

    assert calls == 2
    assert tuple(first) == ("camera-0", "camera-1")
    cache_versions = list((scene_root / "court-line-inference").iterdir())
    assert len(cache_versions) == 1
    manifest = json.loads((cache_versions[0] / "manifest.json").read_text())
    assert manifest["completed_view_count"] == 2
    assert manifest["schema"] == "court_line_inference_cache_v2"
    assert manifest["inference_identity"]["input"]["source"] == "nht_rendered_rgb"
    assert manifest["views"][0]["input_file"].endswith("-input.png")
    mirror_manifest = mirror / cache_versions[0].name / "manifest.json"
    assert json.loads(mirror_manifest.read_text()) == manifest
    assert len(list((mirror_manifest.parent / "views").glob("*.png"))) == 4
    mirrored_probabilities = sorted((mirror_manifest.parent / "views").glob("*.npy"))
    mirrored_probability_mtimes = tuple(
        path.stat().st_mtime_ns for path in mirrored_probabilities
    )

    def reject_regeneration(_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        raise AssertionError("cached court detection must not be regenerated")

    second = load_or_predict_line_probabilities(
        scene=cast(Any, scene),
        inputs=inputs,
        detector_identity={"model": "fixed", "preprocessing": 256},
        predict_probability=reject_regeneration,
    )

    for camera_id in first:
        assert np.array_equal(second[camera_id], first[camera_id])
    assert tuple(path.stat().st_mtime_ns for path in mirrored_probabilities) == (
        mirrored_probability_mtimes
    )


def test_corrupt_probability_cache_fails_without_regeneration(tmp_path: Path) -> None:
    export_root = tmp_path / "B01/reconstruction/export"
    image_path = export_root / "images/frame.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), mode="RGB").save(image_path)
    scene = SimpleNamespace(scene_id="B01", export_root=export_root)
    camera = SceneCamera(
        camera_id="camera-0",
        source_frame_index=0,
        width=6,
        height=4,
        intrinsics=(4.0, 0.0, 3.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path=str(image_path),
    )
    inputs = AlignmentLineInputBatch(
        source=CourtLineInputSource.NHT_RENDERED_RGB,
        provenance={"schema": "test_nht_render_v1"},
        views=(
            AlignmentLineInputView(
                camera=camera,
                image_rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            ),
        ),
    )
    arguments = {
        "scene": cast(Any, scene),
        "inputs": inputs,
        "detector_identity": {"model": "fixed"},
    }
    load_or_predict_line_probabilities(
        **arguments,
        predict_probability=lambda _image: np.ones((3, 5), dtype=np.float32),
    )
    array_path = next((tmp_path / "B01/court-line-inference").glob("*/views/*.npy"))
    array_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="unreadable"):
        load_or_predict_line_probabilities(
            **arguments,
            predict_probability=lambda _image: pytest.fail(
                "corrupt cache must not trigger model inference"
            ),
        )


def test_rendered_inputs_never_reuse_captured_rgb_cache(tmp_path: Path) -> None:
    export_root = tmp_path / "B01/reconstruction/export"
    image_path = export_root / "images/frame.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), mode="RGB").save(image_path)
    scene = SimpleNamespace(scene_id="B01", export_root=export_root)
    camera = SceneCamera(
        camera_id="camera-0",
        source_frame_index=0,
        width=6,
        height=4,
        intrinsics=(4.0, 0.0, 3.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path=str(image_path),
    )
    rendered = AlignmentLineInputBatch(
        source=CourtLineInputSource.NHT_RENDERED_RGB,
        provenance={"schema": "test_source_v1"},
        views=(
            AlignmentLineInputView(
                camera=camera,
                image_rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            ),
        ),
    )
    captured = replace(rendered, source=CourtLineInputSource.CAPTURED_RGB)
    calls = 0

    def predict(_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        nonlocal calls
        calls += 1
        return np.full((3, 5), calls / 10.0, dtype=np.float32)

    for inputs in (captured, rendered):
        load_or_predict_line_probabilities(
            scene=cast(Any, scene),
            inputs=inputs,
            detector_identity={"model": "fixed"},
            predict_probability=predict,
        )

    assert calls == 2
    assert len(list((tmp_path / "B01/court-line-inference").iterdir())) == 2

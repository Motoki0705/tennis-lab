from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.reconstruction.scene_export import (
    NHT_ALPHA_OUTPUT_SEMANTICS,
    NHT_DEPTH_OUTPUT_SEMANTICS,
    NHT_IMAGE_RESOLUTION_SEMANTICS,
    NHT_PIXEL_COORDINATE_CONVENTION,
    NHT_RGB_OUTPUT_SEMANTICS,
    NHT_SCENE_COORDINATE_CONVENTION,
    validate_standard_scene_export,
)


def _write_valid_export(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(root / "images/frame_000000.jpg")
    (root / "model/ckpts").mkdir(parents=True)
    (root / "model/ckpts/model.pt").write_bytes(b"checkpoint")
    (root / "model/runtime-config.json").write_text(
        json.dumps(
            {
                "schema": "nht_runtime_config_v1",
                "camera_model": "pinhole",
                "pose_opt": False,
                "primitive_type": "3dgs",
                "antialiased": False,
                "packed": False,
                "tile_size": 16,
                "with_ut": True,
                "with_eval3d": True,
                "near_plane": 0.01,
                "far_plane": 100.0,
                "deferred_opt_feature_dim": 48,
                "deferred_opt_enable_view_encoding": True,
                "deferred_opt_view_encoding_type": "sh",
                "deferred_mlp_hidden_dim": 128,
                "deferred_mlp_num_layers": 3,
                "deferred_opt_sh_degree": 3,
                "deferred_opt_sh_scale": 3.0,
                "deferred_opt_fourier_num_freqs": 4,
                "deferred_opt_center_ray_encoding": False,
                "deferred_decode_activation": "sigmoid",
                "post_processing": None,
            }
        ),
        encoding="utf-8",
    )
    np.save(
        root / "points_scene.npy",
        np.asarray([[0.0, 0.0, 0.0, 1.0, 0.5, 0.0]], dtype=np.float32),
    )
    identity = np.eye(4).tolist()
    cameras: dict[str, object] = {
        "schema": "nht_standard_cameras_v1",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "transform_semantics": (
            "camera_to_scene maps homogeneous camera coordinates to scene coordinates"
        ),
        "cameras": [
            {
                "camera_id": "frame_000000",
                "source_frame_index": 0,
                "time_seconds": 0.0,
                "split": "train",
                "image": "images/frame_000000.jpg",
                "width": 16,
                "height": 12,
                "intrinsics": {
                    "model": "PINHOLE",
                    "distortion_model": "NONE",
                    "params": [10.0, 10.0, 8.0, 6.0],
                    "matrix": [[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]],
                },
                "camera_to_scene": identity,
                "source_image_processing": {
                    "source_resolution": [16, 12],
                    "crop_xywh": [0, 0, 16, 12],
                    "undistorted": True,
                    "data_factor": 1,
                },
                "diagnostics": {
                    "sfm_camera_id": 1,
                    "sfm_camera_to_world": identity,
                },
                "group": "default",
            }
        ],
    }
    scene: dict[str, object] = {
        "schema": "nht_standard_scene_v1",
        "scene_id": "B00",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "scene_coordinate_convention": NHT_SCENE_COORDINATE_CONVENTION,
        "pixel_coordinate_convention": NHT_PIXEL_COORDINATE_CONVENTION,
        "image_resolution_semantics": NHT_IMAGE_RESOLUTION_SEMANTICS,
        "camera_count": 1,
        "cameras": "cameras.json",
        "point_cloud": {
            "path": "points_scene.npy",
            "shape": [1, 6],
            "dtype": "float32",
            "columns": ["x", "y", "z", "red", "green", "blue"],
            "color_range": [0.0, 1.0],
        },
        "image_root": "images",
        "model_root": "model",
        "scene_from_sfm": identity,
        "sfm_from_scene": identity,
        "normalization": {
            "applied": True,
            "camera_similarity": identity,
            "principal_axis_alignment": identity,
            "upside_down_correction": identity,
        },
        "renderer": {
            "command": "nht-render",
            "model": "model",
            "runtime_config": "model/runtime-config.json",
            "checkpoint": "model/ckpts/model.pt",
            "outputs": {
                "rgb": NHT_RGB_OUTPUT_SEMANTICS,
                "alpha": NHT_ALPHA_OUTPUT_SEMANTICS,
                "depth": NHT_DEPTH_OUTPUT_SEMANTICS,
            },
        },
        "sfm_summary": {},
        "nht_training_summary": {},
        "capabilities": ["nht_rendering_model"],
    }
    (root / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
    (root / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    return scene, cameras


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    return value


def test_validates_complete_standard_export(tmp_path: Path) -> None:
    _write_valid_export(tmp_path)

    export = validate_standard_scene_export(tmp_path / "scene.json")

    assert export.scene_id == "B00"
    assert export.camera_ids == ("frame_000000",)
    assert export.point_count == 1
    assert export.points_scene.dtype == np.float32
    assert not export.points_scene.flags.writeable
    assert export.cameras[0].image_path == str(
        (tmp_path / "images/frame_000000.jpg").resolve()
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("unknown_scene_key", "schema mismatch"),
        ("bad_convention", "scene_coordinate_convention"),
        ("bad_intrinsics", "params disagree"),
        ("improper_rotation", "proper rotation"),
        ("camera_id_escape", "portable identifier"),
        ("missing_checkpoint", "does not exist"),
    ],
)
def test_rejects_structural_geometry_and_file_violations(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    scene, cameras = _write_valid_export(tmp_path)
    camera_records = _list(cameras["cameras"], name="cameras")
    camera = _mapping(camera_records[0], name="camera")
    if mutation == "unknown_scene_key":
        scene["unknown"] = True
        _write_payload(tmp_path / "scene.json", scene)
    elif mutation == "bad_convention":
        scene["scene_coordinate_convention"] = "left-handed"
        _write_payload(tmp_path / "scene.json", scene)
    elif mutation == "bad_intrinsics":
        intrinsics = _mapping(camera["intrinsics"], name="camera.intrinsics")
        params = _list(intrinsics["params"], name="camera.intrinsics.params")
        params[0] = 11.0
        _write_payload(tmp_path / "cameras.json", cameras)
    elif mutation == "improper_rotation":
        camera_to_scene = _list(
            camera["camera_to_scene"], name="camera.camera_to_scene"
        )
        first_row = _list(camera_to_scene[0], name="camera.camera_to_scene[0]")
        first_row[0] = -1.0
        _write_payload(tmp_path / "cameras.json", cameras)
    elif mutation == "camera_id_escape":
        camera["camera_id"] = "../escape"
        _write_payload(tmp_path / "cameras.json", cameras)
    else:
        (tmp_path / "model/ckpts/model.pt").unlink()

    with pytest.raises((FileNotFoundError, TypeError, ValueError), match=match):
        validate_standard_scene_export(tmp_path / "scene.json")


def test_rejects_point_dtype_and_nonfinite_values(tmp_path: Path) -> None:
    _write_valid_export(tmp_path)
    np.save(tmp_path / "points_scene.npy", np.zeros((1, 6), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype float32"):
        validate_standard_scene_export(tmp_path / "scene.json")


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("tile_size", 0, "runtime.tile_size"),
        ("antialiased", 1, "runtime.antialiased"),
        ("deferred_opt_view_encoding_type", "unknown", "view_encoding_type"),
        ("deferred_decode_activation", "unknown", "decode_activation"),
        ("deferred_opt_sh_scale", float("inf"), "finite"),
    ],
)
def test_rejects_invalid_public_renderer_runtime_fields(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    _write_valid_export(tmp_path)
    runtime_path = tmp_path / "model/runtime-config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[key] = value
    _write_payload(runtime_path, runtime)

    with pytest.raises((TypeError, ValueError), match=match):
        validate_standard_scene_export(tmp_path / "scene.json")


def test_rejects_unsatisfied_public_renderer_runtime_dependency(
    tmp_path: Path,
) -> None:
    _write_valid_export(tmp_path)
    runtime_path = tmp_path / "model/runtime-config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["with_eval3d"] = False
    _write_payload(runtime_path, runtime)

    with pytest.raises(ValueError, match="with_ut requires"):
        validate_standard_scene_export(tmp_path / "scene.json")

    points: NDArray[np.float32] = np.zeros((1, 6), dtype=np.float32)
    points[0, 0] = np.nan
    np.save(tmp_path / "points_scene.npy", points)
    with pytest.raises(ValueError, match="finite"):
        validate_standard_scene_export(tmp_path / "scene.json")


def test_rejects_symlink_escape_for_export_references(tmp_path: Path) -> None:
    export_root = tmp_path / "export"
    _write_valid_export(export_root)
    outside = tmp_path / "outside-cameras.json"
    outside.write_text(
        (export_root / "cameras.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (export_root / "cameras.json").unlink()
    (export_root / "cameras.json").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes the export root"):
        validate_standard_scene_export(export_root / "scene.json")

"""BLCS residual I/O preserves physical units, identity, timing and missingness."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.base.generate_dataset import (
    CourtKeypointArtifactMetadata,
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.triangulation_residual.contracts import CameraRig
from src.tasks.blcs.triangulation_residual import data as residual_data
from src.tasks.blcs.triangulation_residual import real_clip
from src.tasks.blcs.triangulation_residual.data import (
    list_scene_paths,
    load_clean_scene,
)
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
    normalize_court_position,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _change_json(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text())
    change(value)
    _write_json(path, value)


@pytest.fixture
def clean_scene(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic"
    scene = root / "scenes" / "scene_000001"
    scene.mkdir(parents=True)
    contract = resolve_court_keypoint_contract("camera_view_v2")
    artifact = CourtKeypointArtifactMetadata.from_contract(
        contract, dataset_schema_id="blcs_generated_dataset_v2"
    ).to_dict()
    centers = [[-3.0, 20.0, 5.0], [3.0, -20.0, 5.0]]
    rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    _write_json(root / "meta.json", {"court_keypoints": artifact})
    _write_json(
        scene / "meta.json",
        {
            "scene_id": scene.name,
            "num_frames": 3,
            "fps_out": 30,
            "num_cameras": 2,
            "court_coordinate_normalization": court_coordinate_normalization_metadata(),
            "court_keypoints": artifact,
            "court_keypoint_views": [
                build_court_view_record(
                    camera_id=f"cam_{index}",
                    camera_center_court_m=center,
                    contract=contract,
                ).to_dict()
                for index, center in enumerate(centers)
            ],
        },
    )
    _write_json(
        scene / "scalars.json",
        {
            "num_balls": 1,
            "num_cameras": 2,
            **{
                f"cam_{index}_params": {
                    "R": rotation,
                    "C": center,
                    "f": 200.0,
                    "cx": 160.0,
                    "cy": 120.0,
                    "w": 320,
                    "h": 240,
                }
                for index, center in enumerate(centers)
            },
        },
    )
    world = np.asarray([[-2, 3, 1], [0, -4, 2], [4, 5, 3]], dtype=np.float32)
    np.save(scene / "ball_pos_world.npy", world)
    np.save(scene / "ball_pos_norm.npy", normalize_court_position(world))
    (root / "train.txt").write_text(f"{scene.name}\n")
    return scene


def test_clean_scene_has_physical_metres_and_world_to_camera_translation(
    clean_scene: Path,
) -> None:
    scene = load_clean_scene(clean_scene)
    expected_world = np.load(clean_scene / "ball_pos_world.npy")
    np.testing.assert_allclose(scene.world_m[:, 0], expected_world, atol=1e-6)
    assert scene.world_m.dtype == np.float32
    assert scene.world_m.shape == (3, 1, 3)
    assert scene.fps == 30
    assert scene.source_group == clean_scene.name
    np.testing.assert_allclose(scene.rig.centers, [[-3, 20, 5], [3, -20, 5]])
    np.testing.assert_allclose(scene.rig.t, [[20, 3, -5], [-20, -3, -5]])
    assert scene.rig.image_size.dtype == np.int64
    assert scene.rig.image_size.tolist() == [[320, 240], [320, 240]]


def test_fixed_splits_preserve_order_and_reject_duplicate_ids(
    clean_scene: Path,
) -> None:
    root = clean_scene.parent.parent
    another = clean_scene.parent / "scene_000000"
    another.mkdir()
    (root / "train.txt").write_text(f"{clean_scene.name}\n\n{another.name}\n")
    assert list_scene_paths(root, "train") == [clean_scene, another]
    (root / "train.txt").write_text(f"{clean_scene.name}\n{clean_scene.name}\n")
    with pytest.raises(ValueError, match="unique"):
        list_scene_paths(root, "train")
    with pytest.raises(ValueError, match="split"):
        list_scene_paths(root, "validation")


@pytest.mark.parametrize(
    ("filename", "change", "message"),
    [
        (
            "meta.json",
            lambda value: value.pop("court_coordinate_normalization"),
            "normalization",
        ),
        ("meta.json", lambda value: value.update(num_frames=4), "GT"),
        ("meta.json", lambda value: value.update(fps_out=0), "fps_out"),
        ("meta.json", lambda value: value.update(scene_id="another"), "scene_id"),
        ("scalars.json", lambda value: value.update(num_balls=2), "one ball"),
        (
            "meta.json",
            lambda value: value["court_keypoint_views"].reverse(),
            "stable ID",
        ),
    ],
)
def test_clean_scene_rejects_invalid_metadata(
    clean_scene: Path,
    filename: str,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _change_json(clean_scene / filename, change)
    with pytest.raises(ValueError, match=message):
        load_clean_scene(clean_scene)


def test_clean_scene_rejects_disagreeing_units_and_missing_targets(
    clean_scene: Path,
) -> None:
    original = np.load(clean_scene / "ball_pos_world.npy")
    np.save(clean_scene / "ball_pos_world.npy", original / 10)
    with pytest.raises(ValueError, match="disagree"):
        load_clean_scene(clean_scene)
    np.save(clean_scene / "ball_pos_world.npy", original)
    np.save(clean_scene / "ball_present.npy", np.asarray([True, False, True]))
    with pytest.raises(ValueError, match="Inactive or padded"):
        load_clean_scene(clean_scene)


def test_clean_scenes_share_one_root_metadata_read(
    clean_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    another = clean_scene.with_name("scene_000002")
    shutil.copytree(clean_scene, another)
    _change_json(another / "meta.json", lambda meta: meta.update(scene_id=another.name))
    root_path = clean_scene.parent.parent / "meta.json"
    reads: list[Path] = []
    original_read = residual_data._read_object

    def read(path: Path) -> dict[str, Any]:
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(residual_data, "_read_object", read)
    for scene_path in [clean_scene, another, clean_scene, another]:
        load_clean_scene(scene_path)
    assert reads.count(root_path) == 1
    assert reads.count(clean_scene / "meta.json") == 2
    assert reads.count(another / "scalars.json") == 2


@pytest.mark.parametrize("changed_stat", ["mtime_ns", "size"])
def test_root_contract_cache_invalidates_on_metadata_revision(
    clean_scene: Path, monkeypatch: pytest.MonkeyPatch, changed_stat: str
) -> None:
    root_path = clean_scene.parent.parent / "meta.json"
    original_stat = root_path.stat()
    reads: list[Path] = []
    original_read = residual_data._read_object

    def read(path: Path) -> dict[str, Any]:
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(residual_data, "_read_object", read)
    load_clean_scene(clean_scene)
    if changed_stat == "mtime_ns":
        root_path.write_text(
            root_path.read_text().replace(
                "blcs_generated_dataset_v2", "blcs_generated_dataset_v3"
            )
        )
        os.utime(
            root_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1),
        )
        assert root_path.stat().st_size == original_stat.st_size
    else:
        _change_json(root_path, lambda meta: meta.update(court_keypoint_views=[]))
        os.utime(
            root_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        assert root_path.stat().st_size != original_stat.st_size
    with pytest.raises(ValueError, match="court keypoint metadata|per-camera"):
        load_clean_scene(clean_scene)
    assert reads.count(root_path) == 2


@pytest.mark.parametrize(
    ("filename", "change", "message"),
    [
        (
            "meta.json",
            lambda meta: meta["court_keypoint_views"].reverse(),
            "stable ID",
        ),
        (
            "scalars.json",
            lambda scalars: scalars["cam_0_params"].update(C=[-4, 20, 5]),
            "camera center",
        ),
        (
            "scalars.json",
            lambda scalars: scalars.update(cam_2_params=scalars["cam_0_params"]),
            "parameter slots",
        ),
    ],
)
def test_cached_root_does_not_hide_scene_or_scalar_changes(
    clean_scene: Path,
    filename: str,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    load_clean_scene(clean_scene)
    _change_json(clean_scene / filename, change)
    with pytest.raises(ValueError, match=message):
        load_clean_scene(clean_scene)


@pytest.fixture
def annotated_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    clip = tmp_path / "clip"
    (clip / "media").mkdir(parents=True)
    (clip / "outsource").mkdir()
    camera_ids = ["cam2", "cam0"]
    _write_json(
        clip / "clip.json",
        {"video_paths": [f"media/{camera}.mp4" for camera in camera_ids]},
    )
    rig = CameraRig(
        K=np.asarray([[[200, 0, 160], [0, 200, 120], [0, 0, 1]]] * 2, dtype=np.float64),
        R=np.asarray([np.eye(3)] * 2, dtype=np.float64),
        t=np.asarray([[0, 0, 5], [2, 0, 5]], dtype=np.float64),
        image_size=np.asarray([[320, 240]] * 2, dtype=np.int64),
    )
    court: np.ndarray = np.arange(56, dtype=np.float64).reshape(2, 14, 2)

    def calibration(
        path: Path,
    ) -> tuple[CameraRig, np.ndarray, np.ndarray, float, dict[str, Any]]:
        assert path == clip
        return (
            rig,
            court,
            np.ones((2, 14)),
            2.0,
            {
                "camera_ids": camera_ids,
                "num_frames": 4,
                "width": 320,
                "height": 240,
                "clip_id": "video/clip",
                "calibration_source": "fixture",
            },
        )

    monkeypatch.setattr(real_clip, "load_clip_calibration", calibration)
    for view, camera in enumerate(camera_ids):
        video_bytes = f"fixture video {camera}".encode()
        (clip / "media" / f"{camera}.mp4").write_bytes(video_bytes)
        records = []
        for index, status in enumerate(
            ["observed", "interpolated", "occlusion_estimated", "unresolved"]
        ):
            x, y = 100.25 + view * 20 + index, 60.5 + index
            records.append(
                {
                    "frame_index": index,
                    "pts": index,
                    "timestamp_seconds": index / 2,
                    "track_id": 1,
                    "label": "tennis_ball",
                    "status": status,
                    "center_px": {"x": x, "y": y} if status != "unresolved" else None,
                    "center_normalized": {"x": x / 320, "y": y / 240}
                    if status != "unresolved"
                    else None,
                    "image_score": 9999,
                }
            )
        _write_json(
            clip / "outsource" / f"{camera}_annotations.json",
            {
                "schema_version": "video_ball_annotation.v2",
                "source": {
                    "file_name": f"{camera}.mp4",
                    "sha256": hashlib.sha256(video_bytes).hexdigest(),
                    "width": 320,
                    "height": 240,
                    "frame_count": 4,
                    "fps_numerator": 2,
                    "fps_denominator": 1,
                    "constant_pts_step": True,
                    "pts_step": 1,
                    "time_base": "1/2",
                    "rotation": 0,
                },
                "coordinate_system": {
                    "origin": "top_left",
                    "x_axis": "right",
                    "y_axis": "down",
                    "frame_index": "zero_based",
                    "pixel_coordinates": "Original decoded source pixels; no crop or rescaling.",
                },
                "target": {"track_id": 1, "label": "tennis_ball"},
                "review": {"reviewer": "fixture; no ground truth"},
                "frames": records,
            },
        )
    return clip


def test_real_clip_preserves_pixels_statuses_camera_and_physical_court_order(
    annotated_clip: Path,
) -> None:
    scene = real_clip.load_real_clip(annotated_clip)
    assert scene.observations_px.shape == (2, 4, 1, 2)
    assert scene.observations_px.dtype == np.float64
    assert scene.metadata["camera_ids"] == ["cam2", "cam0"]
    np.testing.assert_allclose(scene.observations_px[:, 0, 0, 0], [100.25, 120.25])
    np.testing.assert_allclose(scene.scores[:, :, 0], [[1, 0.5, 0.25, 0]] * 2)
    assert np.isnan(scene.observations_px[:, 3]).all()
    np.testing.assert_array_equal(scene.court_px, np.arange(56).reshape(2, 14, 2))
    assert scene.metadata["frame_status"][0] == [
        "observed",
        "interpolated",
        "occlusion_estimated",
        "unresolved",
    ]
    assert scene.metadata["has_ground_truth_3d"] is False
    assert scene.metadata["status_counts"]["cam2"]["unresolved"] == 1
    assert scene.metadata["excluded_at_default_min_score"] == [
        "occlusion_estimated",
        "unresolved",
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda value: value["source"].update(file_name="cam0.mp4"),
            "camera video name",
        ),
        (lambda value: value["source"].update(fps_numerator=3), "FPS"),
        (lambda value: value["source"].update(width=640), "dimensions"),
        (
            lambda value: value["coordinate_system"].update(origin="bottom_left"),
            "original decoded",
        ),
        (lambda value: value["frames"][1].update(frame_index=0), "frame IDs"),
        (lambda value: value["frames"][1].update(pts=0), "PTS"),
        (
            lambda value: value["frames"][1].update(status="new_state"),
            "unsupported ball status",
        ),
        (lambda value: value["frames"][1].update(center_px=None), "localized center"),
        (
            lambda value: value["frames"][3].update(center_px={"x": 0, "y": 0}),
            "must be null",
        ),
    ],
)
def test_real_clip_rejects_misaligned_or_ambiguous_annotations(
    annotated_clip: Path,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _change_json(annotated_clip / "outsource" / "cam2_annotations.json", change)
    with pytest.raises(ValueError, match=message):
        real_clip.load_real_clip(annotated_clip)


def test_real_clip_rejects_changed_source_video(annotated_clip: Path) -> None:
    (annotated_clip / "media" / "cam2.mp4").write_bytes(b"other video")
    with pytest.raises(ValueError, match="SHA256"):
        real_clip.load_real_clip(annotated_clip)

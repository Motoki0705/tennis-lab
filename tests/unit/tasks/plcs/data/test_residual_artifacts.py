"""Physical-coordinate and published-scene boundaries for PLCS residual inputs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.base.generate_dataset import (
    CourtKeypointContractMetadata,
    build_court_view_record,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import plcs_artifact_metadata
from src.tasks.plcs.data.residual_dataset import (
    audit_source_splits,
    list_scene_paths,
    load_clean_scene,
)
from src.tasks.plcs.data.residual_types import CameraRig
from src.tasks.plcs.inference import residual_clip_io as real_clip
from src.tasks.plcs.inference.residual_clip_io import (
    load_clip_calibration,
    load_real_clip,
)
from src.tennis_scene.schema import SceneResult
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)
from tests.support.tennis_scene.annotations import publish_scene_to_clip_store

Mutation = Callable[[dict[str, Any]], None]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _scene(root: Path, name: str, source: str = "Male1Walking_c3d/walk.npz") -> Path:
    path = root / "scenes" / name
    path.mkdir(parents=True)
    contract = resolve_court_keypoint_contract("camera_view_v2")
    centers = [[-4.0, 18.0, 3.0], [8.0, -17.0, 5.0]]
    meta = {
        "scene_id": name,
        "motion_source": f"/original/data/ACCAD/{source}",
        "num_frames": 3,
        "fps": 120.0,
        "num_cameras": 2,
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
        "court_keypoints": plcs_artifact_metadata(contract).to_dict(),
        "court_keypoint_views": [
            build_court_view_record(
                camera_id=f"camera_{index}",
                camera_center_court_m=center,
                contract=contract,
            ).to_dict()
            for index, center in enumerate(centers)
        ],
    }
    scalars: dict[str, Any] = {"num_persons": 1, "num_cameras": 2}
    for index, center in enumerate(centers):
        scalars[f"cam_{index}_params"] = {
            "C": center,
            "R": (
                np.eye(3)
                if index == 0
                else np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            ).tolist(),
            "f": 500.0 + index * 50,
            "cx": 320.0,
            "cy": 240.0,
            "w": 640,
            "h": 480,
        }
    world = np.arange(3 * 17 * 3, dtype=np.float32).reshape(3, 17, 3) / 10
    world[..., 2] = 1.25
    np.save(path / "human_kp_3d.npy", world)
    # These existing-format payloads must never be needed by this loader.
    (path / "position.npy").write_text("not an NPY file")
    (path / "canonical_pose_3d.npy").write_text("not an NPY file")
    _write_json(path / "meta.json", meta)
    _write_json(path / "scalars.json", scalars)
    return path


def _dataset(root: Path) -> dict[str, Path]:
    scenes = {
        "train": _scene(root, "scene_train", "Male1Walking_c3d/walk.npz"),
        "val": _scene(root, "scene_val", "Male1Running_c3d/run.npz"),
        "test": _scene(root, "scene_test", "Female1Walking_c3d/walk.npz"),
    }
    for split, scene in scenes.items():
        (root / f"{split}.txt").write_text(scene.name + "\n", encoding="utf-8")
    return scenes


def _clip(root: Path, *, meta: Mutation | None = None, arrays: Mutation | None = None) -> Path:
    """Publish a SceneResult v2 through the clip store, as the pipeline does.

    ``meta``/``arrays`` edit the metadata or SceneResult fields before
    publication; the immutable export cannot be edited afterwards.
    """
    annotation = root / "annotations" / "tennis_scene"
    frames, width, height = 3, 640, 480
    camera_ids = ["cam2", "cam0", "cam1"]
    fits = []
    for index in range(3):
        center = np.array([index * 4.0, [18.0, -18.0, -17.0][index], 3.0])
        fits.append(
            {
                "K": [[500.0 + index * 10, 0, 320], [0, 500.0, 240], [0, 0, 1]],
                "R": np.eye(3).tolist(),
                "t": (-center).tolist(),
                "camera_center_court_m": center.tolist(),
                "calibration": {"source": "unit_fixture"},
                "calibration_frame_index": 0,
            }
        )
    contract = resolve_court_keypoint_contract("camera_view_v2")
    contract_metadata = CourtKeypointContractMetadata.from_contract(contract).to_dict()
    court_views = [
        build_court_view_record(
            camera_id=camera_id,
            camera_center_court_m=fit["camera_center_court_m"],
            contract=contract,
        )
        for camera_id, fit in zip(camera_ids, fits, strict=True)
    ]
    provenance = build_reference_frame_provenance(
        court_views, reference_camera_id="cam0"
    ).to_dict()
    metadata: dict[str, Any] = {
        "scene_schema_version": 2,
        "camera_ids": camera_ids,
        "num_cameras": 3,
        "sync_assumption": "preprocessed",
        "track_ids": [4, 8],
        "court_keypoints": contract_metadata,
        "court_observation_order": "camera_local_v1",
        "court_reference_provenance": provenance,
        "court_reference": {
            "camera_ids": camera_ids,
            "camera_fits": fits,
            "court_keypoints": contract_metadata,
            "court_observation_order": "camera_local_v1",
            "court_reference_provenance": provenance,
            "reference_camera": "cam0",
            "view_half_turns": [True, False, False],
            "court_keypoint_views": [view.to_dict() for view in court_views],
        },
    }
    if meta is not None:
        meta(metadata)
    _write_json(root / "clip.json", {
        "clip_id": "test_clip", "camera_ids": camera_ids, "num_frames": frames,
        "width": width, "height": height, "fps": 60.0,
    })
    uv = (np.arange(2 * 3 * frames * 17 * 2, dtype=np.float32).reshape(2, 3, frames, 17, 2) / 1000)
    scores = np.full(uv.shape[:-1], 0.75, dtype=np.float32)
    scores[0, 2, 1, 7] = 1.02
    scores[1, 1, 2, 4] = 0
    players: np.ndarray = np.ones((2, frames), bool)
    rejected: np.ndarray = np.zeros((2, frames), bool)
    fields: dict[str, Any] = dict(
        num_frames=frames, fps=60.0, width=width, height=height,
        court_kp=np.repeat(np.arange(3 * 14 * 2, dtype=np.float32).reshape(3, 1, 14, 2) / 100, frames, axis=1),
        court_vis=np.ones((3, frames, 14), dtype=np.float32),
        # The residual reader never uses reconstructed geometry: all rejected.
        player_position=np.zeros((2, frames, 3), np.float32), player_yaw=np.zeros((2, frames), np.float32),
        ball_uv=np.zeros((3, frames, 2), np.float32), ball_vis=np.zeros((3, frames), bool),
        ball_3d=np.zeros((frames, 3), np.float32),
        human_kp_2d=uv, human_kp_vis=scores, player_track_ids=np.array([4, 8], dtype=np.int32),
        player_kp_3d=np.zeros((2, frames, 17, 3), np.float32),
        player_observed=players, player_valid=rejected, player_heading_valid=rejected.copy(),
        player_kp_3d_vis=np.zeros((2, frames, 17), bool), player_smpl_valid=rejected.copy(),
        ball_3d_valid=np.zeros(frames, bool), player_rejection_code=np.ones((2, frames), np.uint8),
        player_kp_3d_rejection_code=np.ones((2, frames, 17), np.uint8), ball_rejection_code=np.ones(frames, np.uint8),
        metadata=metadata,
    )
    if arrays is not None:
        arrays(fields)
    export = publish_scene_to_clip_store(root, "test_clip", SceneResult(**fields))
    _write_json(annotation / "annotation.json", {"scene_index": "scene.json", "scene_result": str(export.relative_to(annotation))})
    return root


def _published(clip: Path) -> SceneResult:
    from src.tennis_scene.archive import load_scene_result

    return load_scene_result(clip / "annotations/tennis_scene/scene.json")


def test_clean_scene_preserves_metres_axes_cameras_and_native_timing(
    tmp_path: Path,
) -> None:
    path = _scene(tmp_path, "scene_000")
    result = load_clean_scene(path)
    np.testing.assert_array_equal(result.world_m, np.load(path / "human_kp_3d.npy"))
    assert result.world_m.dtype == np.float32
    assert result.world_m[2, 16, 0] > 10
    assert result.world_m[0, 0, 2] == 1.25
    assert result.fps == 120.0
    assert result.scene_id == "scene_000"
    assert result.source_group == "ACCAD/Male1Walking_c3d/walk.npz"
    np.testing.assert_allclose(result.rig.centers, [[-4, 18, 3], [8, -17, 5]])
    np.testing.assert_allclose(result.rig.t[1], [-17, -8, -5])
    np.testing.assert_array_equal(result.rig.image_size, [[640, 480], [640, 480]])


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((3, 1, 17, 3), dtype=np.float32),
        np.zeros((2, 17, 3), dtype=np.float32),
        np.zeros((3, 17, 3), dtype=np.int32),
        np.full((3, 17, 3), np.nan, dtype=np.float32),
    ],
)
def test_clean_scene_rejects_invalid_world_arrays(
    tmp_path: Path, array: np.ndarray
) -> None:
    path = _scene(tmp_path, "scene_000")
    np.save(path / "human_kp_3d.npy", array)
    with pytest.raises(ValueError, match="human_kp_3d"):
        load_clean_scene(path)


@pytest.mark.parametrize(
    "field,value", [("fps", 0), ("fps", float("nan")), ("num_frames", 3.0)]
)
def test_clean_scene_rejects_invalid_timing(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _scene(tmp_path, "scene_000")
    meta = _read_json(path / "meta.json")
    meta[field] = value
    _write_json(path / "meta.json", meta)
    with pytest.raises(ValueError, match=field):
        load_clean_scene(path)


def test_clean_scene_rejects_reference_frame_or_camera_order_mismatch(
    tmp_path: Path,
) -> None:
    path = _scene(tmp_path, "scene_000")
    meta = _read_json(path / "meta.json")
    meta["court_keypoints"]["coordinate_frame"] = "reference_camera"
    _write_json(path / "meta.json", meta)
    with pytest.raises(ValueError, match="physical_court"):
        load_clean_scene(path)
    meta["court_keypoints"]["coordinate_frame"] = "physical_court"
    meta["court_keypoint_views"].reverse()
    _write_json(path / "meta.json", meta)
    with pytest.raises(ValueError, match="Camera order"):
        load_clean_scene(path)


def test_clean_scene_rejects_multi_person_and_inactive_gt(tmp_path: Path) -> None:
    path = _scene(tmp_path, "scene_000")
    scalars = _read_json(path / "scalars.json")
    scalars["num_persons"] = 2
    _write_json(path / "scalars.json", scalars)
    with pytest.raises(ValueError, match="exactly one person"):
        load_clean_scene(path)
    scalars["num_persons"] = 1
    _write_json(path / "scalars.json", scalars)
    np.save(path / "person_present.npy", [True, False, True])
    with pytest.raises(ValueError, match="Inactive"):
        load_clean_scene(path)


def test_split_audit_reports_source_disjoint_but_shared_subjects(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path)
    report = audit_source_splits(tmp_path)
    assert report["scene_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["source_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["source_motion_disjoint"] is True
    assert report["subject_disjoint"] is False
    assert report["subject_overlap"]["train/val"] == ["Male1"]


def test_split_list_preserves_order_and_rejects_duplicates(tmp_path: Path) -> None:
    scenes = _dataset(tmp_path)
    (tmp_path / "train.txt").write_text("scene_test\nscene_train\n", encoding="utf-8")
    assert list_scene_paths(tmp_path, "train") == [scenes["test"], scenes["train"]]
    (tmp_path / "train.txt").write_text("scene_train\nscene_train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        list_scene_paths(tmp_path, "train")


def test_split_audit_rejects_scene_overlap(tmp_path: Path) -> None:
    _dataset(tmp_path)
    (tmp_path / "val.txt").write_text("scene_train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Scene overlap"):
        audit_source_splits(tmp_path)


def test_split_audit_rejects_source_overlap_across_dataset_prefixes(
    tmp_path: Path,
) -> None:
    scenes = _dataset(tmp_path)
    meta = _read_json(scenes["val"] / "meta.json")
    meta["motion_source"] = "/relocated/dataset/ACCAD/Male1Walking_c3d/walk.npz"
    _write_json(scenes["val"] / "meta.json", meta)
    with pytest.raises(ValueError, match="Source motion overlap"):
        audit_source_splits(tmp_path)


def test_non_accad_source_is_rejected_by_audit_and_loader(tmp_path: Path) -> None:
    scenes = _dataset(tmp_path)
    meta = _read_json(scenes["train"] / "meta.json")
    meta["motion_source"] = "/data/gvhmr/Male1/walk.npz"
    _write_json(scenes["train"] / "meta.json", meta)
    with pytest.raises(ValueError, match="only ACCAD"):
        load_clean_scene(scenes["train"])
    with pytest.raises(ValueError, match="only ACCAD"):
        audit_source_splits(tmp_path)


def test_calibration_preserves_already_aligned_cam2_court_points(
    tmp_path: Path,
) -> None:
    clip = _clip(tmp_path)
    rig, court, scores, fps, info = load_clip_calibration(clip)
    expected = _published(clip).court_kp[:, 0].astype(np.float64) * [640, 480]
    # cam2 is the first camera and has a camera-local half-turn in metadata.
    # The serialized SceneResult has applied it already, so every point index
    # and its fitted physical camera must survive unchanged.
    np.testing.assert_array_equal(court, expected)
    np.testing.assert_array_equal(scores, np.ones((3, 14)))
    np.testing.assert_allclose(rig.centers, [[0, 18, 3], [4, -18, 3], [8, -17, 3]])
    assert fps == 60
    assert info["calibration_frame_index"] == 0
    assert info["court_coordinate_frame"] == "physical_court"
    assert info["court_reference_provenance"]["reference_camera_local_index"] == 1
    assert (
        info["court_reference_provenance"]["reference_from_physical"]
        == np.eye(3).tolist()
    )


def test_calibration_rejects_consistent_half_turn_reference(tmp_path: Path) -> None:
    def half_turn_reference(meta: dict[str, Any]) -> None:
        contract = resolve_court_keypoint_contract("camera_view_v2")
        views = [
            build_court_view_record(
                camera_id=camera_id,
                camera_center_court_m=fit["camera_center_court_m"],
                contract=contract,
            )
            for camera_id, fit in zip(
                meta["camera_ids"], meta["court_reference"]["camera_fits"], strict=True
            )
        ]
        half_turn = build_reference_frame_provenance(
            views, reference_camera_id="cam2"
        ).to_dict()
        meta["court_reference_provenance"] = half_turn
        meta["court_reference"]["court_reference_provenance"] = half_turn
        meta["court_reference"]["reference_camera"] = "cam2"

    clip = _clip(tmp_path, meta=half_turn_reference)
    with pytest.raises(ValueError, match="identity reference transforms"):
        load_clip_calibration(clip)


@pytest.mark.parametrize("field", ["court_keypoints", "court_reference_provenance"])
def test_calibration_rejects_conflicting_root_and_nested_metadata(
    tmp_path: Path, field: str
) -> None:
    def conflict(meta: dict[str, Any]) -> None:
        if field == "court_keypoints":
            meta["court_reference"][field] = CourtKeypointContractMetadata.from_contract(
                resolve_court_keypoint_contract("physical_v1")
            ).to_dict()
        else:
            # A copy: root and nested provenance share one object before publication.
            meta["court_reference"][field] = {**meta["court_reference"][field], "reference_camera_local_index": 2}

    clip = _clip(tmp_path, meta=conflict)
    with pytest.raises(ValueError, match=f"Root and nested {field}"):
        load_clip_calibration(clip)


@pytest.mark.parametrize(
    "nested,field",
    [
        (False, "court_keypoints"),
        (True, "court_keypoints"),
        (False, "court_reference_provenance"),
        (True, "court_reference_provenance"),
        (True, "court_keypoint_views"),
        (True, "reference_camera"),
        (True, "view_half_turns"),
    ],
)
def test_calibration_rejects_missing_court_metadata(
    tmp_path: Path, nested: bool, field: str
) -> None:
    clip = _clip(tmp_path, meta=lambda meta: (meta["court_reference"] if nested else meta).pop(field))
    with pytest.raises(ValueError, match=field):
        load_clip_calibration(clip)


@pytest.mark.parametrize(
    ("indices", "match"),
    [([0, 0, 3], "calibration_frame_index"), ([0, -1, 0], "calibration_frame_index"),
     ([True, 0, 0], "calibration_frame_index"), ([0, 1, 0], "same frame")],
)
def test_calibration_rejects_invalid_or_disagreeing_calibration_frames(
    tmp_path: Path, indices: list[object], match: str
) -> None:
    def set_frames(meta: dict[str, Any]) -> None:
        for fit, index in zip(meta["court_reference"]["camera_fits"], indices, strict=True):
            fit["calibration_frame_index"] = index

    clip = _clip(tmp_path, meta=set_frames)
    with pytest.raises(ValueError, match=match):
        load_clip_calibration(clip)


def test_calibration_rejects_time_varying_court(tmp_path: Path) -> None:
    def vary(fields: dict[str, Any]) -> None:
        fields["court_kp"] = fields["court_kp"].copy()
        fields["court_kp"][:, 2] += 0.01

    clip = _clip(tmp_path, arrays=vary)
    with pytest.raises(ValueError, match="Time-varying calibration"):
        load_clip_calibration(clip)


@pytest.mark.parametrize("field", ["camera_ids", "court_keypoint_views", "camera_fits"])
def test_calibration_rejects_disagreeing_camera_order(
    tmp_path: Path, field: str
) -> None:
    clip = _clip(tmp_path, meta=lambda meta: meta["court_reference"][field].reverse())
    with pytest.raises(ValueError, match="camera order|ordered court_keypoint_views"):
        load_clip_calibration(clip)


def test_calibration_rejects_inconsistent_half_turn_flags(tmp_path: Path) -> None:
    clip = _clip(tmp_path, meta=lambda meta: meta["court_reference"].__setitem__("view_half_turns", [False] * 3))
    with pytest.raises(ValueError, match="view_half_turns"):
        load_clip_calibration(clip)


def test_real_clip_preserves_associated_players_camera_order_and_raw_scores(
    tmp_path: Path,
) -> None:
    clip = _clip(tmp_path)
    result = load_real_clip(clip)
    assert len(result) == 2
    published = _published(clip)
    assert published.human_kp_2d is not None and published.human_kp_vis is not None
    expected_uv = published.human_kp_2d.astype(np.float64)
    expected_scores = published.human_kp_vis.astype(np.float64)
    for index, scene in enumerate(result):
        np.testing.assert_allclose(
            scene.observations_px, expected_uv[index] * [640, 480]
        )
        np.testing.assert_array_equal(scene.scores, expected_scores[index])
        assert scene.observations_px.shape == (3, 3, 17, 2)
        assert scene.metadata["camera_ids"] == ["cam2", "cam0", "cam1"]
        assert scene.metadata["player_id"] == [4, 8][index]
        assert scene.metadata["association_already_applied"] is True
        assert scene.metadata["association_source"]["axis"] == "player_track_ids"
        assert scene.metadata["independent_3d_ground_truth"] is False
    assert result[0].scores.max() > 1
    assert result[1].scores[1, 2, 4] == 0
    np.testing.assert_allclose(result[0].rig.centers[:, 0], [0, 4, 8])


def test_real_clip_scales_uv_using_each_camera_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path)
    load = real_clip._load_calibration
    rig, court, scores, fps, info, path, scene = load(clip)
    varied_rig = CameraRig(
        rig.K, rig.R, rig.t, np.array([[640, 480], [1280, 720], [1920, 1080]])
    )
    monkeypatch.setattr(
        real_clip,
        "_load_calibration",
        lambda _: (varied_rig, court, scores, fps, info, path, scene),
    )
    result = load_real_clip(clip)
    assert scene.human_kp_2d is not None
    np.testing.assert_allclose(
        result[0].observations_px,
        scene.human_kp_2d[0].astype(np.float64) * varied_rig.image_size[:, None, None, :],
    )


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("human_kp_vis", np.full((2, 3, 3, 17), -1.0, dtype=np.float32), "nonnegative"),
        ("player_track_ids", np.array([8, 4], dtype=np.int32), "track_ids must match"),
        ("player_track_ids", np.array([4, 4], dtype=np.int32), "unique nonnegative"),
    ],
)
def test_real_clip_rejects_invalid_observation_arrays(
    tmp_path: Path, key: str, value: np.ndarray, match: str
) -> None:
    clip = _clip(tmp_path, arrays=lambda fields: fields.__setitem__(key, value))
    with pytest.raises(ValueError, match=match):
        load_real_clip(clip)


def test_calibration_requires_the_published_scene_marker(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    (clip / "annotations/tennis_scene/annotation.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_clip_calibration(clip)


def test_v1_layout_marker_is_rejected(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    _write_json(clip / "annotations/tennis_scene/annotation.json", {"scene_result": "scene.npz"})
    with pytest.raises(ValueError, match="v1 layout is not readable"):
        load_clip_calibration(clip)

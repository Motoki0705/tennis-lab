from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.tasks.base.data.scene_dataset import Scene, SceneDatasetConfig
from src.tasks.base.generate_dataset import (
    DatasetCourtKeypointContract,
    SceneCourtViewRecords,
    apply_court_view_record,
    build_court_view_record,
    court_points_physical_to_target,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.data.dataset import SceneDataset
from src.tasks.plcs.data.tracking_dataset import PLCSTrackingDataset
from src.utils.schema.court_normalization import normalize_court_position


def _dataset_and_scene() -> tuple[SceneDataset, Scene]:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="camera_0",
            camera_center_court_m=(2.0, -12.0, 4.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_1",
            camera_center_court_m=(-3.0, 12.0, 5.0),
            contract=contract,
        ),
    )
    frames = 2
    physical_court = np.stack(
        [
            np.linspace(0.01, 0.2, 20, dtype=np.float32),
            np.linspace(0.21, 0.4, 20, dtype=np.float32),
        ],
        axis=-1,
    )
    data: dict[str, Any] = {
        "position": np.repeat(
            np.asarray(
                normalize_court_position(np.array([1.0, 2.0, 0.5], dtype=np.float32))
            )[None].astype(np.float32),
            frames,
            axis=0,
        ),
        "rotation": np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), frames, axis=0),
        "human_kp_3d": np.full((frames, 17, 3), [1.0, 2.0, 0.5], np.float32),
        "canonical_pose_3d": np.full((frames, 17, 3), [0.25, 0.0, 0.0], np.float32),
    }
    for index, view in enumerate(views):
        disk_court = apply_court_view_record(
            physical_court,
            view,
            keypoint_axis=0,
        )
        data[f"cam_{index}_human_kp_uv"] = np.full(
            (frames, 17, 2), 0.3 + index * 0.1, dtype=np.float32
        )
        data[f"cam_{index}_human_kp_vis"] = np.ones((frames, 17), dtype=np.bool_)
        data[f"cam_{index}_court_kp_uv"] = np.repeat(disk_court[None], frames, axis=0)
        data[f"cam_{index}_court_kp_vis"] = np.ones((frames, 20), dtype=np.bool_)
        data[f"cam_{index}_params"] = {
            "C": list(view.camera_center_court_m),
            "R": np.eye(3, dtype=np.float32).tolist(),
            "f": 800.0,
            "cx": 640.0,
            "cy": 360.0,
            "w": 1280,
            "h": 720,
        }
    scene = Scene(
        path=Path("/dataset/scenes/scene_000000"),
        data=data,
        meta={"scene_id": "scene_000000", "num_frames": frames},
        num_frames=frames,
        num_cameras=2,
    )
    dataset = object.__new__(SceneDataset)
    dataset.rng = np.random.default_rng(0)
    dataset.augment = False
    dataset.court_keypoint_contract = contract
    dataset.court_keypoint_validation = DatasetCourtKeypointContract(
        contract=contract,
        metadata=None,
        legacy_metadata_free=False,
        scenes=(
            SceneCourtViewRecords(
                scene_id="scene_000000",
                court_views=views,
            ),
        ),
    )
    dataset._plcs_num_views_range = (2, 2)
    dataset._plcs_seq_len_range = (2, 2)
    dataset.camera_mode_plcs = "random"
    dataset.num_court_kp = 20
    dataset.config = SceneDatasetConfig(
        scene_dir=Path("/dataset"),
        split_file=Path("train.txt"),
        seq_len_range=(2, 2),
        num_views_range=(2, 2),
        camera_mode="random",
        crop_mode="center",
        min_num_frames=2,
        min_num_cameras=2,
    )
    return dataset, scene


def test_standard_dataset_aligns_before_first20_and_rotates_court_targets() -> None:
    dataset, scene = _dataset_and_scene()
    sample = dataset.build_sample(scene)
    provenance = sample["court_reference_provenance"]

    assert sample["court_kp"].shape == (2, 2, 20, 2)
    torch.testing.assert_close(sample["court_kp"][0], sample["court_kp"][1])
    assert sample["selected_camera_ids"] == ("camera_0", "camera_1") or sample[
        "selected_camera_ids"
    ] == ("camera_1", "camera_0")
    # Reference selection is identity-based; camera ordering is independently random.
    assert provenance.reference_camera_id == "camera_0"
    assert sample["selected_camera_ids"][provenance.reference_camera_local_index] == (
        provenance.reference_camera_id
    )

    expected_physical = torch.tensor([[1.0, 2.0, 0.5]]).expand(2, 3)
    expected_target_m = court_points_physical_to_target(
        expected_physical,
        provenance,
    )
    expected_position = normalize_court_position(expected_target_m)
    torch.testing.assert_close(sample["position"], expected_position)
    torch.testing.assert_close(
        sample["human_kp_3d"][:, 0],
        expected_target_m,
    )
    expected_heading = torch.tensor(
        [-1.0, 0.0] if provenance.reference_camera_id == "camera_1" else [1.0, 0.0]
    )
    torch.testing.assert_close(sample["rotation"][0], expected_heading)

    selected_ids = cast("tuple[str, ...]", sample["selected_camera_ids"])
    expected_centers = {
        "camera_0": torch.tensor([2.0, -12.0, 4.0]),
        "camera_1": torch.tensor([-3.0, 12.0, 5.0]),
    }
    for local_index, camera_id in enumerate(selected_ids):
        transformed = court_points_physical_to_target(
            expected_centers[camera_id],
            provenance,
        )
        torch.testing.assert_close(sample["camera_C"][local_index], transformed)


def test_object_uv_is_invariant_under_reference_transform() -> None:
    dataset, scene = _dataset_and_scene()
    sample = dataset.build_sample(scene)
    for local_index, camera_id in enumerate(sample["selected_camera_ids"]):
        physical_index = int(camera_id.rsplit("_", 1)[1])
        expected = torch.full((2, 17, 2), 0.3 + physical_index * 0.1)
        torch.testing.assert_close(sample["human_kp"][local_index], expected)


def test_tracking_aligns_before_first14_and_keeps_canonical_pose_local() -> None:
    standard, scene = _dataset_and_scene()
    dataset = object.__new__(PLCSTrackingDataset)
    dataset.rng = np.random.default_rng(0)
    dataset.augment = False
    dataset.court_keypoint_contract = standard.court_keypoint_contract
    dataset.court_keypoint_validation = standard.court_keypoint_validation
    dataset.num_queries = 1
    dataset.min_reuse_gap_frames = 0
    dataset.randomize_slots_train = False
    dataset.config = SceneDatasetConfig(
        scene_dir=Path("/dataset"),
        split_file=Path("train.txt"),
        seq_len_range=(2, 2),
        num_views_range=(2, 2),
        camera_mode="random",
        crop_mode="center",
        min_num_frames=2,
        min_num_cameras=2,
    )

    sample = dataset.build_sample(scene)
    provenance = sample["court_reference_provenance"]
    assert sample["court_kp"].shape == (2, 2, 14, 2)
    torch.testing.assert_close(sample["court_kp"][0], sample["court_kp"][1])
    torch.testing.assert_close(
        sample["target_canonical_pose_3d"][:, 0],
        torch.from_numpy(scene.data["canonical_pose_3d"]),
    )
    physical_world = torch.from_numpy(scene.data["human_kp_3d"])
    expected_world = court_points_physical_to_target(physical_world, provenance)
    torch.testing.assert_close(
        sample["target_human_kp_3d"][:, 0],
        expected_world,
    )

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.base.data import ReferenceViewSelectionError
from src.tasks.base.data.scene_dataset import Scene, SceneDatasetConfig
from src.tasks.base.generate_dataset import (
    DatasetCourtKeypointContract,
    SceneCourtViewRecords,
    apply_court_view_record,
    build_court_view_record,
    court_points_physical_to_target,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import (
    choose_reference_selection,
    track_query_reference_contract_document,
)
from src.tasks.plcs.data.dataset import SceneDataset
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)
from src.tasks.plcs.data.tracking_dataset import (
    PLCSTrackingDataset,
    collate_plcs_tracking_batch,
)
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
            "f": 800.0 + 100.0 * index,
            "cx": 640.0 + 10.0 * index,
            "cy": 360.0 + 5.0 * index,
            "w": 1280 + 640 * index,
            "h": 720 + 360 * index,
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
    dataset.reference_camera_id = "camera_1"
    dataset.track_query_reference_document = None
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


def _tracking_dataset(
    standard: SceneDataset,
    *,
    augment: bool,
    track_query_reference_document: dict[str, object] | None = None,
) -> PLCSTrackingDataset:
    dataset = object.__new__(PLCSTrackingDataset)
    dataset.rng = np.random.default_rng(0)
    dataset.augment = augment
    dataset.reference_camera_id = "camera_1"
    dataset.track_query_reference_document = track_query_reference_document
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
    return dataset


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
    assert provenance.reference_camera_id == "camera_1"
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
    dataset = _tracking_dataset(standard, augment=False)

    sample = dataset.build_sample(scene)
    provenance = sample["court_reference_provenance"]
    selection = sample["reference_view_selection"]
    assert selection.provenance is provenance
    assert sample["reference_view_index"].dtype == torch.int64
    assert int(sample["reference_view_index"]) == (
        sample["selected_camera_ids"].index("camera_1")
    )
    assert int(sample["reference_camera_id"]) == 1
    assert sample["view_camera_ids"].tolist() in ([0, 1], [1, 0])
    torch.testing.assert_close(
        sample["physical_from_reference"],
        sample["reference_from_physical"].T,
    )
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
    assert sample["clean_human_kp"].shape == (2, 2, 1, 17, 2)
    assert sample["clean_human_vis"].shape == (2, 2, 1, 17)
    torch.testing.assert_close(sample["human_kp_target"], sample["clean_human_kp"])
    torch.testing.assert_close(
        sample["human_vis_target"], sample["clean_human_vis"]
    )
    assert sample["human_kp_target"].data_ptr() != sample["clean_human_kp"].data_ptr()
    assert (
        sample["human_vis_target"].data_ptr()
        != sample["clean_human_vis"].data_ptr()
    )

    expected_intrinsics = {
        "camera_0": (800.0, 640.0, 360.0, 1280.0, 720.0),
        "camera_1": (900.0, 650.0, 365.0, 1920.0, 1080.0),
    }
    for local_index, camera_id in enumerate(sample["selected_camera_ids"]):
        actual = torch.stack(
            [
                sample["camera_f"][local_index],
                sample["camera_cx"][local_index],
                sample["camera_cy"][local_index],
                sample["camera_w"][local_index],
                sample["camera_h"][local_index],
            ]
        )
        torch.testing.assert_close(
            actual,
            torch.tensor(expected_intrinsics[camera_id]),
        )


def test_tracking_build_augment_collate_preserves_reference_contract() -> None:
    standard, scene = _dataset_and_scene()
    document = track_query_reference_contract_document(
        {
            "model": {
                "name": "plcs_track_query_reference",
                "target_frame_contract": "reference_camera_court_rzpi_v1",
                "track_query_rope_contract": (
                    "time_camera_reference_selector_v1"
                ),
                "reference_selector_mode": "reference",
            }
        },
        standard.court_keypoint_contract,
    )
    assert document is not None
    dataset = _tracking_dataset(
        standard,
        augment=True,
        track_query_reference_document=document,
    )
    augmentation_config = OmegaConf.load(
        Path(__file__).resolve().parents[5]
        / "src/tasks/plcs/configs/data/_augmentation.yaml"
    ).augmentation
    augmentation_config.enabled = False
    dataset.tracking_augmentation = PLCSTrackingDetectionAugmentation(
        augmentation_config
    )

    sample = dataset.build_sample(scene)
    augmented = dataset.augment_sample(sample)

    assert set(augmented) == set(sample)
    for key in (
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
    ):
        torch.testing.assert_close(augmented[key], sample[key])
        assert augmented[key].data_ptr() != sample[key].data_ptr()
    assert augmented["reference_view_selection"].provenance is augmented[
        "court_reference_provenance"
    ]
    assert augmented["reference_view_selection"].stable_camera_id_table is augmented[
        "stable_camera_id_table"
    ]
    assert augmented["reference_camera_id_string"] == sample[
        "reference_camera_id_string"
    ]
    assert augmented["track_query_reference"] == sample["track_query_reference"]
    assert augmented["track_query_reference"] is not sample["track_query_reference"]
    assert augmented["court_keypoint_metadata"] is not sample[
        "court_keypoint_metadata"
    ]

    augmented_document = cast("dict[str, object]", augmented["track_query_reference"])
    sample_document = cast("dict[str, object]", sample["track_query_reference"])
    schema_version = augmented_document["schema_version"]
    augmented_document["schema_version"] = -1
    assert sample_document["schema_version"] == schema_version
    augmented_document["schema_version"] = schema_version

    batch = collate_plcs_tracking_batch([augmented])

    assert batch["reference_view_selection"] == (
        augmented["reference_view_selection"],
    )
    assert batch["stable_camera_id_table"] == (
        augmented["stable_camera_id_table"],
    )
    assert batch["reference_camera_id_string"] == (
        augmented["reference_camera_id_string"],
    )
    assert batch["track_query_reference"] == augmented["track_query_reference"]
    torch.testing.assert_close(
        batch["reference_from_physical"][0],
        augmented["reference_from_physical"],
    )


def test_evaluation_reference_selection_fails_closed_for_multiview() -> None:
    dataset, _ = _dataset_and_scene()
    views = dataset.court_keypoint_validation.scenes[0].court_views

    with pytest.raises(
        ReferenceViewSelectionError,
        match="requires an explicit canonical reference camera ID",
    ):
        choose_reference_selection(
            dataset.court_keypoint_contract,
            views,
            views,
            rng=None,
            requested_camera_id=None,
        )


def test_single_view_and_seeded_reordering_resolve_reference_by_identity() -> None:
    dataset, _ = _dataset_and_scene()
    views = dataset.court_keypoint_validation.scenes[0].court_views
    single = choose_reference_selection(
        dataset.court_keypoint_contract,
        views,
        (views[1],),
        rng=None,
        requested_camera_id=None,
    )
    assert single is not None
    assert single.reference_camera_id == "camera_1"
    assert single.reference_view_index == 0

    ordered = choose_reference_selection(
        dataset.court_keypoint_contract,
        views,
        views,
        rng=np.random.default_rng(9),
        requested_camera_id=None,
    )
    reversed_order = choose_reference_selection(
        dataset.court_keypoint_contract,
        views,
        tuple(reversed(views)),
        rng=np.random.default_rng(9),
        requested_camera_id=None,
    )
    assert ordered is not None and reversed_order is not None
    assert ordered.reference_camera_id == reversed_order.reference_camera_id
    assert (
        ordered.selected_camera_ids[ordered.reference_view_index]
        == reversed_order.selected_camera_ids[reversed_order.reference_view_index]
        == ordered.reference_camera_id
    )

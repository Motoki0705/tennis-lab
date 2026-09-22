"""Association samples need 2D observations and labels, not world-pose targets."""

from pathlib import Path
from types import MethodType

import numpy as np
import pytest

from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.data.scene_dataset import Scene, SceneDatasetConfig
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.data.association_dataset import BLCSAssociationDataset
from src.tasks.plcs.data.association_dataset import PLCSAssociationDataset
from src.tasks.plcs.data.frame_rate_augmentation import PLCSFrameRateSampler


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_sample_does_not_require_three_dimensional_arrays(task, monkeypatch):
    kind = PLCSAssociationDataset if task == "plcs" else BLCSAssociationDataset
    d = object.__new__(kind)
    d.augment = False
    d.rng = np.random.default_rng(0)
    d.num_slots = 4
    d.max_identities = 10
    d.config = SceneDatasetConfig(
        scene_dir=Path("/data"),
        split_file=Path("val.txt"),
        seq_len_range=(2, 2),
        num_views_range=(2, 2),
        camera_mode="first",
        crop_mode="center",
        min_num_frames=1,
        min_num_cameras=1,
    )
    d.observation_tracking_config = ObservationTrackingConfig(
        max_distance=0.08,
        max_missed_frames=2,
        min_reuse_gap_frames=4,
        use_velocity_prediction=True,
        min_common_keypoints=1,
        cost_reduction="mean",
        overflow_policy="error",
    )
    court = resolve_court_keypoint_contract("camera_view_v2")
    views = tuple(
        build_court_view_record(
            camera_id=f"cam_{i}",
            camera_center_court_m=(0.0, -12.0 if i == 0 else 12.0, 3.0),
            contract=court,
        )
        for i in range(2)
    )
    monkeypatch.setattr(d, "selected_views", MethodType(lambda self, scene: ((0, 1), views, 1), d))
    payload: dict[str, np.ndarray] = {}
    for i in range(2):
        if task == "plcs":
            payload[f"cam_{i}_human_kp_uv"] = np.full((2, 1, 17, 2), 0.3, np.float32)
            payload[f"cam_{i}_human_kp_vis"] = np.ones((2, 1, 17), bool)
            payload[f"cam_{i}_court_kp_uv"] = np.full((2, 20, 2), 0.4, np.float32)
            payload[f"cam_{i}_court_kp_vis"] = np.ones((2, 20), bool)
        else:
            payload[f"cam_{i}_ball_uv"] = np.full((2, 1, 2), 0.3, np.float32)
            payload[f"cam_{i}_ball_vis"] = np.ones((2, 1), bool)
            payload[f"cam_{i}_court_kp_uv"] = np.full((20, 2), 0.4, np.float32)
            payload[f"cam_{i}_court_kp_vis"] = np.ones(20, bool)
    payload["person_present" if task == "plcs" else "ball_present"] = np.ones(
        (2, 1), bool
    )
    if task == "plcs":
        d.frame_rate_sampler = PLCSFrameRateSampler(
            {
                "enabled": False,
                "frame_rate": {"enabled": False, "prob": 1.0, "choices_hz": [60.0]},
            }
        )
    scene = Scene(
        path=Path("/data/scenes/s"),
        data=payload,
        meta={"scene_id": "s", "fps": 60.0},
        num_frames=2,
        num_cameras=2,
    )
    sample = d.build_sample(scene)
    assert set(sample) == {
        "object_uv",
        "object_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "reference_view_index",
        "object_id_target",
        "side_target",
    }
    assert sample["side_target"].tolist() == [True, False]
    assert sample["object_id_target"][
        sample["object_vis"].any(-1)
    ].unique().tolist() == [0]
    assert sample["object_uv"].shape == (2, 2, 4, 17 if task == "plcs" else 1, 2)

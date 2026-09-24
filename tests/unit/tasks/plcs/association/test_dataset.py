"""Track grouping precedes crops; synthetic GT numbers remain teacher-only."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.tasks.base.data.scene_dataset import Scene, SceneDatasetConfig
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.data.association_collate import collate_association
from src.tasks.plcs.data.association_dataset import PLCSAssociationDataset


def fixture(rosters: list[list[int]], *, frames: int = 2, capacity: int = 4):
    count = max(max(row) for row in rosters) + 1
    dataset = object.__new__(PLCSAssociationDataset)
    dataset.augment = False
    dataset.rng = np.random.default_rng(42)
    dataset.num_slots = 4
    dataset.reference_camera_id = "camera_0"
    dataset.hydra_cfg = SimpleNamespace(data=SimpleNamespace(target_fps=30.))
    dataset.config = SceneDatasetConfig(Path("/data"), Path("test.txt"), (capacity, capacity), (len(rosters), len(rosters)), "first", "center", 1, 2)
    court = resolve_court_keypoint_contract("camera_view_v2")
    dataset.court_views = {"s": tuple(build_court_view_record(camera_id=f"camera_{v}", camera_center_court_m=(0., -12. if v == 0 else 12., 3.), contract=court) for v in range(len(rosters)))}
    payload: dict[str, np.ndarray] = {"person_present": np.ones((frames, count), bool)}
    for v, roster in enumerate(rosters):
        values = np.broadcast_to((.05 + np.arange(count, dtype=np.float32) * .05)[None, :, None, None], (frames, count, 17, 2)).copy()
        mask: np.ndarray = np.zeros((frames, count, 17), bool)
        mask[:, roster] = True
        payload.update({f"cam_{v}_human_kp_uv": values, f"cam_{v}_human_kp_vis": mask,
            f"cam_{v}_court_kp_uv": np.full((frames, 20, 2), .4, np.float32), f"cam_{v}_court_kp_vis": np.ones((frames, 20), bool)})
    return dataset, Scene(Path("/data/scenes/s"), payload, {"scene_id": "s", "fps": 30.}, frames, len(rosters))


def test_only_2d_inputs_short_scene_padding_and_teacher_correspondence():
    dataset, scene = fixture([[0, 1, 2, 3]] * 3)
    sample = dataset.build_sample(scene)
    assert sample["human_kp"].shape == (3, 4, 4, 17, 2)
    assert sample["padding_mask"].tolist() == [[False, False, True, True]] * 3
    assert "local_track_ids" not in sample and "detection_gt_index" not in sample
    for view in range(3):
        for slot, identity in enumerate(sample["track_person_id"][view]):
            torch.testing.assert_close(sample["human_kp"][view, :2, slot], torch.full((2, 17, 2), .05 + float(identity) * .05))
    assert not torch.equal(sample["track_person_id"][0], sample["track_person_id"][1])
    assert sample["side_target"].tolist() == [False, True, True]


def test_capacity_counts_full_scene_before_temporal_crop():
    dataset, scene = fixture([[0, 1, 2, 3, 4]] * 2, frames=10, capacity=2)
    for view in range(2):
        mask = scene.data[f"cam_{view}_human_kp_vis"]
        mask[:] = False
        for person in range(5):
            mask[person * 2:person * 2 + 2, person] = True
    with pytest.raises(ValueError, match="before cropping"):
        dataset.build_sample(scene)


def test_camera_capacity_does_not_limit_scene_identity_count():
    dataset, scene = fixture([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]])
    sample = dataset.build_sample(scene)
    assert len(sample["track_person_id"].unique()) == 8
    sample["sample_index"] = torch.tensor(0)
    batch = collate_association([sample], pad_views_to=5)
    assert batch["track_person_id"].shape == (1, 5, 4)
    assert batch["track_person_id"][0, 3:].eq(-1).all()
    assert batch["padding_mask"][0, 3:].all()

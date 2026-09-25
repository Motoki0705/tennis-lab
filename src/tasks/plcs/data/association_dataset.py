"""PLCS fixed camera-local tracks with teacher-only cross-view person IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data.rng import require_run_seed
from src.tasks.base.data.scene_dataset import (
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
)
from src.tasks.base.data.track_query_reference import (
    include_evaluation_reference_camera,
    resolve_evaluation_reference_camera_id,
    select_seeded_training_reference_camera_id,
)
from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.court_keypoint_contract import validate_plcs_dataset_court_keypoints
from src.tasks.plcs.data.tracked_slots import FixedTrackRegistry
from src.utils.data.camera_sampling import camera_candidate_indices


class PLCSAssociationDataset(SceneDatasetBase[dict[str, Tensor]]):
    def __init__(self, *, scene_dir: str | Path, split_file: str | Path, config: Any,
                 seed: int | None = None, augment: bool = False, reference_camera_id: str | None = None) -> None:
        self.augment, self.hydra_cfg = augment, config
        self.reference_camera_id = reference_camera_id
        self.num_slots = int(config.model.num_slots)
        court = resolve_court_keypoint_contract(str(config.court_keypoints.selector))
        if court.selector != "camera_view_v2":
            raise ValueError("PLCS association requires raw camera-local courts")
        contract = validate_plcs_dataset_court_keypoints(Path(scene_dir), Path(split_file), court)
        self.court_views = {scene.scene_id: scene.court_views for scene in contract.scenes}
        self._cache: dict[int, dict[str, Tensor]] = {}
        self.cache_size = int(config.data.validation_cache_size) if not augment else 0
        data = config.data
        super().__init__(config=SceneDatasetConfig(scene_dir=Path(scene_dir), split_file=Path(split_file),
            seq_len_range=tuple(data.seq_len_range), num_views_range=tuple(data.num_views_range),
            camera_mode=data.camera_mode, crop_mode="random" if augment else "center",
            min_num_frames=1, min_num_cameras=2, camera_candidates=camera_candidate_indices(data.camera_candidates)),
            seed=require_run_seed(config) if seed is None else seed, sample_local_rng=not augment)

    def _required_min_frames(self) -> int:
        # Short complete source motions are retained and explicitly padded.
        return 1

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        data = self.hydra_cfg.data
        complete = self.court_views[scene.path.name]
        selected = self.select_cameras(scene).indices
        if not self.augment:
            selected = include_evaluation_reference_camera(tuple(v.camera_id for v in complete), selected,
                requested_camera_id=self.reference_camera_id, candidate_camera_indices=self.config.camera_candidates, rng=self.rng)
        views = tuple(complete[i] for i in selected)
        camera_ids = tuple(v.camera_id for v in views)
        reference = (select_seeded_training_reference_camera_id(camera_ids, rng=self.rng) if self.augment else
            resolve_evaluation_reference_camera_id(camera_ids, requested_camera_id=self.reference_camera_id))
        fps = float(scene.meta["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Scene fps must be positive and finite")
        target_fps = float(data.target_fps)
        positions = np.arange(int(np.floor((scene.num_frames - 1) * target_fps / fps)) + 1) * fps / target_fps
        capacity = int(self.rng.integers(self.config.seq_len_range[0], self.config.seq_len_range[1] + 1))
        start_max = max(0, len(positions) - capacity)
        start = int(self.rng.integers(start_max + 1)) if self.augment else start_max // 2
        # All cameras use the same nearest source timestamps at the declared FPS.
        indices = np.rint(positions[start:start + capacity]).astype(np.int64).clip(0, scene.num_frames - 1)
        length = len(indices)
        presence = scene.get_array("person_present").astype(bool)
        humans, masks, courts, court_masks, local_ids, teachers = [], [], [], [], [], []
        for camera in selected:
            full_vis = scene.get_camera_array(camera, "human_kp_vis").astype(bool) & presence[..., None]
            roster = np.flatnonzero(full_vis.any(axis=(0, 2)))
            if len(roster) > self.num_slots:
                raise ValueError(f"{scene.path.name}/camera_{camera} exceeds cumulative track capacity {self.num_slots} before cropping")
            # Simulate a correct upstream tracker; local numbers are independent per camera.
            local = np.full(presence.shape[1], -1, np.int64)
            local[roster] = self.rng.choice(1_000_000, len(roster), replace=False)
            target = {int(local[person]): int(person) for person in roster}
            humans.append(torch.from_numpy(scene.get_camera_array(camera, "human_kp_uv")[indices]).float())
            masks.append(torch.from_numpy(full_vis[indices]))
            local_ids.append(torch.from_numpy(local))
            teachers.append(target)
            courts.append(torch.from_numpy(scene.get_camera_array(camera, "court_kp_uv")[indices, :14]).float())
            court_masks.append(torch.from_numpy(scene.get_camera_array(camera, "court_kp_vis")[indices, :14]).bool())
        packed = FixedTrackRegistry(camera_ids, num_slots=self.num_slots).pack(torch.stack(humans), torch.stack(masks), torch.stack(local_ids))
        kp, visible = packed.keypoints, packed.visibility
        court, cv = torch.stack(courts), torch.stack(court_masks)
        target = torch.full(packed.local_track_ids.shape, -1, dtype=torch.int64)
        for view, labels in enumerate(teachers):
            for slot, identity in enumerate(packed.local_track_ids[view].tolist()):
                if identity >= 0:
                    target[view, slot] = labels[identity]
        if self.augment:
            cfg = data.augmentation
            kp = kp + torch.from_numpy(self.rng.normal(0, float(cfg.pose_noise), kp.shape)).float()
            court = court + torch.from_numpy(self.rng.normal(0, float(cfg.court_noise), court.shape)).float()
            visible &= torch.from_numpy(self.rng.random(visible.shape) >= float(cfg.joint_dropout))
            cv &= torch.from_numpy(self.rng.random(cv.shape) >= float(cfg.court_dropout))
            for view in range(len(views)):
                for slot in range(self.num_slots):
                    if target[view, slot] >= 0 and self.rng.random() < float(cfg.track_dropout):
                        first = int(self.rng.integers(length))
                        last = min(length, first + int(self.rng.integers(1, max(2, length // 3))))
                        visible[view, first:last, slot] = False
            visible &= ((kp >= 0) & (kp <= 1)).all(-1)
            cv &= ((court >= 0) & (court <= 1)).all(-1)
        v, p = len(views), self.num_slots
        output = {
            "human_kp": torch.zeros(v, capacity, p, 17, 2),
            "human_vis": torch.zeros(v, capacity, p, 17, dtype=torch.bool),
            "court_kp": torch.zeros(v, capacity, 14, 2),
            "court_vis": torch.zeros(v, capacity, 14, dtype=torch.bool),
            "padding_mask": torch.ones(v, capacity, dtype=torch.bool),
            "track_person_id": target,
            "reference_view_index": torch.tensor(camera_ids.index(reference)),
        }
        output["human_kp"][:, :length] = kp.masked_fill(~visible[..., None], 0)
        output["human_vis"][:, :length] = visible
        output["court_kp"][:, :length] = court.masked_fill(~cv[..., None], 0)
        output["court_vis"][:, :length] = cv
        output["padding_mask"][:, :length] = False
        absolute = torch.tensor([view.camera_center_court_m[1] > 0 for view in views])
        output["side_target"] = absolute ^ absolute[camera_ids.index(reference)]
        return output

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index in self._cache:
            return self._cache[index]
        sample: dict[str, Tensor] = super().__getitem__(index)
        sample["sample_index"] = torch.tensor(index)
        if self.cache_size:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = sample
        return sample

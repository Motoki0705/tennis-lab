"""BLCS camera-local observations and side/identity teachers, without 3D packing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.association_dataset import AssociationSceneDataset
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.base.generate_dataset import CourtViewRecord
from src.tasks.blcs.data.court_view import (
    court_views_by_scene,
    validate_blcs_dataset_court_keypoints,
)
from src.tasks.blcs.data.observation_candidates import PhysicalObservationCandidates
from src.tasks.blcs.data.tracking_augmentation import BLCSTrackingDetectionAugmentation


class BLCSAssociationDataset(AssociationSceneDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.augmentation = BLCSTrackingDetectionAugmentation(
            self.hydra_cfg.data.augmentation, num_slots=self.num_slots
        )

    def load_court_views(
        self, root: Path, split: Path
    ) -> dict[str, tuple[CourtViewRecord, ...]]:
        result: dict[str, tuple[CourtViewRecord, ...]] = court_views_by_scene(
            validate_blcs_dataset_court_keypoints(
                scene_dir=root, split_file=split, contract=self.court_contract
            )
        )

        return result

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        presence = torch.from_numpy(scene.get_array("ball_present")).bool()
        window = self.select_window(scene, full_len=presence.shape[0])
        presence = presence[window.sl]
        indices, views, reference = self.selected_views(scene)
        uv_rows, vis_rows, courts, court_masks = [], [], [], []
        for i in indices:
            uv = torch.from_numpy(
                scene.get_camera_array(i, "ball_uv", window=window)
            ).float()
            vis = (
                torch.from_numpy(
                    scene.get_camera_array(i, "ball_vis", window=window)
                ).bool()
                & presence
            )
            uv_rows.append(uv.masked_fill(~vis[..., None], 0))
            vis_rows.append(vis)
            court = torch.from_numpy(scene.get_camera_array(i, "court_kp_uv")).float()
            mask = torch.from_numpy(scene.get_camera_array(i, "court_kp_vis")).bool()
            if court.ndim == 2:
                court = court[:14][None].expand(window.seq_len, -1, -1)
                mask = mask[:14][None].expand(window.seq_len, -1)
            else:
                court, mask = court[window.sl, :14], mask[window.sl, :14]
            courts.append(court)
            court_masks.append(mask)
        uv, visible, court, mask = (
            torch.stack(uv_rows),
            torch.stack(vis_rows),
            torch.stack(courts),
            torch.stack(court_masks),
        )
        ids = torch.arange(presence.shape[1])[None, None].expand(uv.shape[:3])
        candidates = PhysicalObservationCandidates(
            uv=uv, vis=visible, gt_index=torch.where(visible, ids, -1)
        )
        if self.augment:
            candidates = self.augmentation(
                candidates,
                court_kp=court.masked_fill(~mask[..., None], 0),
                court_vis=mask,
            )
        return self.finish_sample(
            uv=candidates.uv.unsqueeze(-2),
            visible=candidates.vis.unsqueeze(-1),
            court=court,
            court_visible=mask,
            identity=candidates.gt_index,
            camera_indices=indices,
            views=views,
            reference=reference,
        )

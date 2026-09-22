"""PLCS camera-local COCO17 association observations; no 3D targets are loaded."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.association_dataset import AssociationSceneDataset
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.base.generate_dataset import CourtViewRecord
from src.tasks.plcs.court_keypoint_contract import validate_plcs_dataset_court_keypoints
from src.tasks.plcs.data.frame_rate_augmentation import PLCSFrameRateSampler
from src.tasks.plcs.data.tracking_augmentation import PLCSTrackingDetectionAugmentation


class PLCSAssociationDataset(AssociationSceneDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.augmentation = PLCSTrackingDetectionAugmentation(
            self.hydra_cfg.data.augmentation, num_slots=self.num_slots
        )
        self.frame_rate_sampler = PLCSFrameRateSampler(self.hydra_cfg.data.augmentation)

    def load_court_views(
        self, root: Path, split: Path
    ) -> dict[str, tuple[CourtViewRecord, ...]]:
        contract = validate_plcs_dataset_court_keypoints(
            root, split, self.court_contract
        )
        return {scene.scene_id: scene.court_views for scene in contract.scenes}

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        presence = torch.from_numpy(scene.get_array("person_present")).bool()
        plan = self.frame_rate_sampler.plan(
            source_fps=scene.meta.get("fps"),
            full_len=presence.shape[0],
            seq_len_range=self.config.seq_len_range,
            crop_mode="random" if self.augment else "center",
            augment=self.augment,
            rng=self.rng,
        )
        window = plan.source_window
        presence = presence[window.sl]
        indices, views, reference = self.selected_views(scene)
        uv_rows, vis_rows, courts, masks = [], [], [], []
        for i in indices:
            uv = torch.from_numpy(
                scene.get_camera_array(i, "human_kp_uv", window=window)
            ).float()
            vis = (
                torch.from_numpy(
                    scene.get_camera_array(i, "human_kp_vis", window=window)
                ).bool()
                & presence[..., None]
            )
            uv, vis = plan.masked_linear(uv, vis, axis=0)
            uv_rows.append(uv.masked_fill(~vis[..., None], 0))
            vis_rows.append(vis)
            court = torch.from_numpy(
                scene.get_camera_array(i, "court_kp_uv", window=window)[:, :14]
            ).float()
            mask = torch.from_numpy(
                scene.get_camera_array(i, "court_kp_vis", window=window)[:, :14]
            ).bool()
            court, mask = plan.masked_linear(court, mask, axis=0)
            courts.append(court)
            masks.append(mask)
        uv, vis, court, mask = (
            torch.stack(uv_rows),
            torch.stack(vis_rows),
            torch.stack(courts),
            torch.stack(masks),
        )
        identity = torch.arange(presence.shape[1])[None, None].expand(uv.shape[:3])
        packet = {
            "human_kp": uv,
            "human_vis": vis,
            "court_kp": court.masked_fill(~mask[..., None], 0),
            "court_vis": mask,
            "detection_gt_index": torch.where(vis.any(-1), identity, -1),
        }
        if self.augment:
            packet = self.augmentation(packet)
        return self.finish_sample(
            uv=packet["human_kp"],
            visible=packet["human_vis"],
            court=packet["court_kp"],
            court_visible=packet["court_vis"],
            identity=packet["detection_gt_index"],
            camera_indices=indices,
            views=views,
            reference=reference,
        )

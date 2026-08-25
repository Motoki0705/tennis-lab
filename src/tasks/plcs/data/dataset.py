"""Unified dataset for PLCS frame/sequence/single/multiview modes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.tasks.base.data.scene_dataset import (
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
)
from src.tasks.plcs.data.augmentation import PLCSObservationAugmentation
from src.tasks.plcs.data.targets import build_coco17_world_targets
from src.tasks.plcs.data.types import PLCSBatch
from src.utils.schema.court_normalization import (
    validate_court_coordinate_normalization,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneDataset(SceneDatasetBase[dict[str, Tensor]]):
    """Unified PLCS dataset – scene-level indexing.

    Returns per-sample tensors with camera-time ordering:
    - human_kp: (N, T, 17, 2)
    - court_kp: (N, T, 20, 2)
    - human_vis: (N, T, 17)
    - court_vis: (N, T, 20)
    - padding_mask: (N, T), True for padding
    - position: (T, 3)
    - rotation: (T, 2)
    """

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: DictConfig,
        augment: bool = True,
    ) -> None:
        self.hydra_cfg = config
        self.augment = augment
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self._configure_task(data_cfg)
        super().__init__(
            config=self._build_scene_dataset_config(
                scene_dir=scene_dir,
                split_file=split_file,
                data_cfg=data_cfg,
            )
        )

    # -- Composed-method hooks ------------------------------------------

    def _configure_task(self, data_cfg: dict) -> None:
        from omegaconf import DictConfig

        self.camera_mode_plcs = str(data_cfg["camera_mode"])
        self.is_multiview = str(data_cfg["mode"]) in {
            "multiview",
            "multiview_sequence",
        }

        r = data_cfg["num_views_range"]
        self._plcs_num_views_range: tuple[int, int] = (int(r[0]), int(r[1]))
        r = data_cfg["seq_len_range"]
        self._plcs_seq_len_range: tuple[int, int] = (int(r[0]), int(r[1]))

        augmentation_cfg = data_cfg["augmentation"]
        if not isinstance(augmentation_cfg, (dict, DictConfig)):
            raise ValueError("data.augmentation must be a mapping-like config.")
        self.augmentation = PLCSObservationAugmentation(augmentation_cfg)
        # Number of court keypoints to use (first N from the canonical order)
        self.num_court_kp = int(data_cfg["num_court_kp"])

    def _build_scene_dataset_config(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        data_cfg: dict,
    ) -> SceneDatasetConfig:
        return SceneDatasetConfig(
            scene_dir=Path(scene_dir),
            split_file=Path(split_file),
            seq_len_range=self._plcs_seq_len_range,
            num_views_range=self._plcs_num_views_range,
            camera_mode=self.camera_mode_plcs,
            crop_mode=("random" if self.augment else "center"),
            min_num_frames=self._plcs_seq_len_range[0],
            min_num_cameras=self._plcs_num_views_range[0],
        )

    def _validate_scene_metadata(self, meta: dict, *, path: Path) -> None:
        validate_court_coordinate_normalization(meta, artifact=f"Scene {path}")

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        cams = self.select_cameras(
            scene,
            num_views_range=self._plcs_num_views_range,
            camera_mode=self.camera_mode_plcs,
        )
        # Resolve effective frame count from arrays
        pos_len = int(scene.data["position"].shape[0])
        rot_len = int(scene.data["rotation"].shape[0])
        primary_len = int(scene.get_camera_array(cams.primary, "human_kp_uv").shape[0])
        full_len = scene.effective_num_frames(primary_len, pos_len, rot_len)
        window = self.select_window(scene, full_len=full_len)

        human_kp_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        human_vis_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []

        for cam_idx in cams.indices:
            human_kp = torch.from_numpy(
                scene.get_camera_array(cam_idx, "human_kp_uv", window=window)
            ).float()
            court_kp = torch.from_numpy(
                scene.get_camera_array(cam_idx, "court_kp_uv", window=window)
            ).float()
            human_vis = torch.from_numpy(
                scene.get_camera_array(cam_idx, "human_kp_vis", window=window)
            ).float()
            court_vis = torch.from_numpy(
                scene.get_camera_array(cam_idx, "court_kp_vis", window=window)
            ).float()
            court_kp = court_kp[..., : self.num_court_kp, :]
            court_vis = court_vis[..., : self.num_court_kp]

            human_kp = human_kp * human_vis.unsqueeze(-1)
            court_kp = court_kp * court_vis.unsqueeze(-1)

            human_kp_list.append(human_kp)
            court_kp_list.append(court_kp)
            human_vis_list.append(human_vis)
            court_vis_list.append(court_vis)

        position = torch.from_numpy(scene.get_array("position", window=window)).float()
        rotation = torch.from_numpy(scene.get_array("rotation", window=window)).float()

        sample: dict[str, Tensor] = {
            "human_kp": torch.stack(human_kp_list, dim=0),
            "court_kp": torch.stack(court_kp_list, dim=0),
            "human_vis": torch.stack(human_vis_list, dim=0),
            "court_vis": torch.stack(court_vis_list, dim=0),
            "padding_mask": torch.zeros(
                len(cams.indices),
                window.seq_len,
                dtype=torch.bool,
            ),
            "position": position,
            "rotation": rotation,
        }

        # Build COCO17 world targets from raw NPZ data
        # scene.data["meta"] is a raw numpy scalar (JSON string); use scene.meta
        # (already parsed dict) so build_coco17_world_targets gets a plain dict.
        payload_for_targets = {**scene.data, "meta": scene.meta}
        human_kp_3d = build_coco17_world_targets(payload_for_targets)
        sample["human_kp_3d"] = torch.from_numpy(human_kp_3d[window.sl].copy()).float()

        return sample

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        augmented: dict[str, Tensor] = self.augmentation.forward(sample)
        return augmented


def collate_plcs_batch(batch: list[dict[str, Tensor]]) -> PLCSBatch | dict[str, Tensor]:
    """Collate variable-view/variable-length PLCS samples into a padded batch."""
    max_views = max(int(sample["human_kp"].shape[0]) for sample in batch)
    max_seq_len = max(int(sample["human_kp"].shape[1]) for sample in batch)

    human_kp_batch = []
    court_kp_batch = []
    human_vis_batch = []
    court_vis_batch = []
    padding_mask_batch = []
    position_batch = []
    rotation_batch = []
    human_kp_3d_batch = []

    for sample in batch:
        n_views = int(sample["human_kp"].shape[0])
        seq_len = int(sample["human_kp"].shape[1])
        pad_views = max_views - n_views
        pad_seq = max_seq_len - seq_len

        human_kp = sample["human_kp"]
        court_kp = sample["court_kp"]
        n_kp = int(court_kp.shape[2])
        human_vis = sample["human_vis"]
        court_vis = sample["court_vis"]
        padding_mask = sample["padding_mask"]
        position = sample["position"]
        rotation = sample["rotation"]
        human_kp_3d = sample.get("human_kp_3d")

        if pad_seq > 0:
            human_kp = torch.cat(
                [human_kp, torch.zeros(n_views, pad_seq, 17, 2)], dim=1
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(n_views, pad_seq, n_kp, 2)], dim=1
            )
            human_vis = torch.cat([human_vis, torch.zeros(n_views, pad_seq, 17)], dim=1)
            court_vis = torch.cat(
                [court_vis, torch.zeros(n_views, pad_seq, n_kp)], dim=1
            )
            padding_mask = torch.cat(
                [padding_mask, torch.ones(n_views, pad_seq, dtype=torch.bool)], dim=1
            )
            position = torch.cat([position, torch.zeros(pad_seq, 3)], dim=0)
            rotation = torch.cat([rotation, torch.zeros(pad_seq, 2)], dim=0)
            if human_kp_3d is not None:
                human_kp_3d = torch.cat(
                    [human_kp_3d, torch.zeros(pad_seq, 17, 3)], dim=0
                )

        if pad_views > 0:
            human_kp = torch.cat(
                [human_kp, torch.zeros(pad_views, max_seq_len, 17, 2)], dim=0
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(pad_views, max_seq_len, n_kp, 2)], dim=0
            )
            human_vis = torch.cat(
                [human_vis, torch.zeros(pad_views, max_seq_len, 17)], dim=0
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(pad_views, max_seq_len, n_kp)], dim=0
            )
            padding_mask = torch.cat(
                [
                    padding_mask,
                    torch.ones(pad_views, max_seq_len, dtype=torch.bool),
                ],
                dim=0,
            )

        human_kp_batch.append(human_kp)
        court_kp_batch.append(court_kp)
        human_vis_batch.append(human_vis)
        court_vis_batch.append(court_vis)
        padding_mask_batch.append(padding_mask)
        position_batch.append(position)
        rotation_batch.append(rotation)
        if human_kp_3d is not None:
            human_kp_3d_batch.append(human_kp_3d)

    collated: dict[str, Tensor] = {
        "human_kp": torch.stack(human_kp_batch, dim=0),
        "court_kp": torch.stack(court_kp_batch, dim=0),
        "human_vis": torch.stack(human_vis_batch, dim=0),
        "court_vis": torch.stack(court_vis_batch, dim=0),
        "padding_mask": torch.stack(padding_mask_batch, dim=0),
        "position": torch.stack(position_batch, dim=0),
        "rotation": torch.stack(rotation_batch, dim=0),
    }
    if human_kp_3d_batch:
        collated["human_kp_3d"] = torch.stack(human_kp_3d_batch, dim=0)

    return collated

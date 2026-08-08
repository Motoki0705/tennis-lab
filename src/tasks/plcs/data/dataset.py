"""Unified dataset for PLCS frame/sequence/single/multiview modes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from src.synthetic_data_generation.dataset.plcs.validation import (
    PLCSCompactDatasetReader,
    PLCSTrackIndex,
)
from src.tasks.base.data.canonical_dataset import CanonicalDataset
from src.tasks.plcs.data.augmentation import PLCSObservationAugmentation
from src.tasks.plcs.data.types import PLCSBatch

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneDataset(CanonicalDataset[dict[str, Tensor]]):
    """Materialize one person interval across every generated camera."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        split: str,
        config: DictConfig,
        augment: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(config=config, augment=augment, rng=rng)
        self.reader = PLCSCompactDatasetReader(Path(dataset_dir))
        self.index: tuple[PLCSTrackIndex, ...] = self.reader.split_tracks(split)
        if not self.index:
            raise ValueError(f"Canonical PLCS split {split!r} is empty.")
        num_court_kp = self.data_config["num_court_kp"]
        if not isinstance(num_court_kp, int):
            raise TypeError("data.num_court_kp must be an integer.")
        self.num_court_kp = num_court_kp
        if not 1 <= self.num_court_kp <= 20:
            raise ValueError("data.num_court_kp must be within [1, 20].")
        augmentation = self.data_config["augmentation"]
        if not isinstance(augmentation, Mapping):
            raise TypeError("data.augmentation must be a mapping.")
        self.augmentation = PLCSObservationAugmentation(augmentation)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        track = self.index[item]
        scene = self.reader.materialize_all_views(track.scene_id).supervision
        local_window = self.contiguous_window(track.stop_frame - track.start_frame)
        start = track.start_frame + int(local_window.start or 0)
        stop = track.start_frame + int(local_window.stop or 0)
        frames = slice(start, stop)
        person = track.object_index
        sample: dict[str, Tensor] = {
            "human_kp": torch.from_numpy(scene.human_kp[frames, :, person]).permute(
                1, 0, 2, 3
            ),
            "court_kp": torch.from_numpy(
                scene.court_kp[frames, :, : self.num_court_kp]
            ).permute(1, 0, 2, 3),
            "human_vis": torch.from_numpy(scene.human_vis[frames, :, person]).permute(
                1, 0, 2
            ),
            "court_vis": torch.from_numpy(
                scene.court_vis[frames, :, : self.num_court_kp]
            ).permute(1, 0, 2),
            "human_mask": torch.from_numpy(scene.human_mask[frames, :, person]).permute(
                1, 0
            ),
            "position": torch.from_numpy(scene.position[frames, person]),
            "rotation": torch.from_numpy(scene.rotation[frames, person]),
            "human_kp_3d": torch.from_numpy(scene.human_kp_3d[frames, person]),
        }
        return self.augmentation.forward(sample) if self.augment else sample


def collate_plcs_batch(batch: list[dict[str, Tensor]]) -> PLCSBatch | dict[str, Tensor]:
    """Collate variable-view/variable-length PLCS samples into a padded batch."""
    max_views = max(int(sample["human_kp"].shape[0]) for sample in batch)
    max_seq_len = max(int(sample["human_kp"].shape[1]) for sample in batch)

    human_kp_batch = []
    court_kp_batch = []
    human_vis_batch = []
    court_vis_batch = []
    human_mask_batch = []
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
        human_mask = sample["human_mask"]
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
            human_mask = torch.cat([human_mask, torch.zeros(n_views, pad_seq)], dim=1)
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
            human_mask = torch.cat(
                [human_mask, torch.zeros(pad_views, max_seq_len)], dim=0
            )

        human_kp_batch.append(human_kp)
        court_kp_batch.append(court_kp)
        human_vis_batch.append(human_vis)
        court_vis_batch.append(court_vis)
        human_mask_batch.append(human_mask)
        position_batch.append(position)
        rotation_batch.append(rotation)
        if human_kp_3d is not None:
            human_kp_3d_batch.append(human_kp_3d)

    collated: dict[str, Tensor] = {
        "human_kp": torch.stack(human_kp_batch, dim=0),
        "court_kp": torch.stack(court_kp_batch, dim=0),
        "human_vis": torch.stack(human_vis_batch, dim=0),
        "court_vis": torch.stack(court_vis_batch, dim=0),
        "human_mask": torch.stack(human_mask_batch, dim=0),
        "position": torch.stack(position_batch, dim=0),
        "rotation": torch.stack(rotation_batch, dim=0),
    }
    if human_kp_3d_batch:
        collated["human_kp_3d"] = torch.stack(human_kp_3d_batch, dim=0)

    return collated

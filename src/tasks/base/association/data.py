"""Promote synthetic provenance to supervised labels AFTER local 2D tracking."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.base.data.rng import require_worker_seeded_dataset


class AssociationDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, source: Dataset, *, task: str) -> None:
        self.source = source
        self.task = task

    def __len__(self) -> int:
        return len(self.source)

    def seed_worker(self, *, worker_seed: int, worker_id: int) -> None:
        require_worker_seeded_dataset(self.source).seed_worker(
            worker_seed=worker_seed, worker_id=worker_id
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.source[index]
        selection = sample["reference_view_selection"]
        absolute = torch.tensor(
            [v.camera_center_court_m[1] > 0 for v in selection.selected_views],
            dtype=torch.bool,
        )
        side = absolute ^ absolute[selection.reference_view_index]
        if self.task == "plcs":
            uv, visible = sample["human_kp"], sample["human_vis"]
            identity = sample["detection_gt_index"]
        else:
            uv, visible = (
                sample["ball_uv"].unsqueeze(-2),
                sample["ball_vis"].unsqueeze(-1),
            )
            identity = sample["candidate_gt_index"]
        # These are explicit teacher-only fields. Never pass provenance/3D targets to forward.
        return {
            "object_uv": uv,
            "object_vis": visible,
            "court_kp": sample["court_kp"],
            "court_vis": sample["court_vis"],
            "padding_mask": sample["padding_mask"],
            "reference_view_index": sample["reference_view_index"],
            "object_id_target": identity.clone(),
            "side_target": side,
        }


def collate_association(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("Association batch must not be empty")
    max_v = max(s["object_uv"].shape[0] for s in samples)
    max_t = max(s["object_uv"].shape[1] for s in samples)
    output: dict[str, Tensor] = {}
    for key in samples[0]:
        if key == "reference_view_index":
            output[key] = torch.stack([s[key] for s in samples])
            continue
        rows = []
        for sample in samples:
            value = sample[key]
            shape = (
                (max_v,) if key == "side_target" else (max_v, max_t, *value.shape[2:])
            )
            fill = (
                True
                if key == "padding_mask"
                else (-1 if key == "object_id_target" else 0)
            )
            row = value.new_full(shape, fill)
            row[tuple(slice(0, n) for n in value.shape)] = value
            rows.append(row)
        output[key] = torch.stack(rows)
    return output


class AssociationDataModule(SceneDirectoryDataModule):
    def __init__(
        self, config: Any, *, task: str, dataset_factory: Callable[..., Dataset]
    ) -> None:
        self.task = task
        self.dataset_factory = dataset_factory
        super().__init__(config)

    def _build_collate_fn(self) -> Callable[..., Any]:
        return collate_association

    def _dataset_name(self) -> str:
        return self.task

    def _build_dataset(
        self, scene_dir: Path, split_file: str, augment: bool, seed: int | None = None
    ) -> Dataset:
        source = self.dataset_factory(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.config,
            augment=augment,
            seed=self._dataset_seed(scene_dir, split_file) if seed is None else seed,
            reference_camera_id=None
            if augment
            else str(self.config.data.evaluation_reference_camera_id),
        )
        return AssociationDataset(source, task=self.task)

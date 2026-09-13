"""Unified PyTorch Lightning DataModule for PLCS."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.base.data.rng import seed_scene_dataset_worker
from src.tasks.base.generate_dataset import CAMERA_VIEW_V2_SELECTOR
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch


class PLCSDataModule(SceneDirectoryDataModule):
    """Lightning DataModule for unified PLCS frame/sequence/multiview training."""

    def __init__(self, config: object) -> None:
        self.plcs_runtime = PLCSTrainingConfig.from_config(config)
        super().__init__(config)

    def _build_collate_fn(self) -> Callable[..., Any] | None:
        return cast(Callable[..., Any], collate_plcs_batch)

    def _build_dataset(
        self,
        scene_dir: Path,
        split_file: str,
        augment: bool,
        seed: int | None = None,
    ) -> Dataset:
        return SceneDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.plcs_runtime.raw,
            seed=(self._dataset_seed(scene_dir, split_file) if seed is None else seed),
            augment=augment,
            reference_camera_id=(
                self.plcs_runtime.data.evaluation_reference_camera_id
                if (
                    not augment
                    and self.plcs_runtime.court_keypoint_contract.selector
                    == CAMERA_VIEW_V2_SELECTOR
                )
                else None
            ),
        )

    def _dataset_name(self) -> str:
        return "plcs"

    def train_dataloader(self) -> DataLoader:
        weights_name = self.plcs_runtime.data.values.get("sampling_weights")
        if weights_name is None:
            return super().train_dataloader()
        if not isinstance(self.train_dataset, SceneDataset):
            raise RuntimeError("Weighted PLCS training requires setup('fit') first.")
        path = self.scene_dir / str(weights_name)
        weights_by_scene = json.loads(path.read_text())
        if not isinstance(weights_by_scene, dict) or set(weights_by_scene) != {
            p.name for p in self.train_dataset.scenes
        }:
            raise ValueError(
                "Sampling weights must match the complete filtered train split exactly."
            )
        weights = [weights_by_scene[p.name] for p in self.train_dataset.scenes]
        if any(
            type(w) not in (int, float) or not math.isfinite(w) or w <= 0
            for w in weights
        ):
            raise ValueError("Sampling weights must be finite and positive.")
        sampler = WeightedRandomSampler(
            weights,
            len(weights),
            replacement=True,
            generator=self._loader_generators["train"],
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn,
            generator=self._loader_generators["train"],
            worker_init_fn=seed_scene_dataset_worker,
        )

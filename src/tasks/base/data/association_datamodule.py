"""Loader lifecycle for association datasets with explicit prefetch/cache settings."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.tasks.base.data.association_collate import collate_association
from src.tasks.base.data.association_dataset import AssociationSceneDataset
from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.base.data.rng import seed_scene_dataset_worker


class AssociationDataModuleBase(SceneDirectoryDataModule):
    dataset_type: type[AssociationSceneDataset]
    config: Any

    def _build_collate_fn(self) -> Any:
        return partial(
            collate_association, pad_views_to=int(self.config.data.num_views_range[1])
        )

    def _build_dataset(
        self, scene_dir: Path, split_file: str, augment: bool, seed: int | None = None
    ) -> Dataset:
        return self.dataset_type(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.config,
            augment=augment,
            seed=self._dataset_seed(scene_dir, split_file) if seed is None else seed,
            reference_camera_id=None
            if augment
            else str(self.config.data.evaluation_reference_camera_id),
        )

    def _build_loader(self, dataset: Dataset, *, stage: str) -> DataLoader:
        extras = (
            {}
            if self.num_workers == 0
            else {
                "persistent_workers": bool(self.config.data.persistent_workers),
                "prefetch_factor": int(self.config.data.prefetch_factor),
            }
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=stage == "train",
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=stage == "train",
            collate_fn=self.collate_fn,
            generator=self._loader_generators[stage],
            worker_init_fn=seed_scene_dataset_worker,
            **extras,
        )

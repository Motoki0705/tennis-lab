"""Fixed source-disjoint splits with deterministic epoch-dependent corruption."""

from __future__ import annotations

import multiprocessing
from collections import OrderedDict
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.tasks.base.triangulation_residual.configuration import ResidualConfig
from src.tasks.base.triangulation_residual.contracts import CleanResidualScene
from src.tasks.base.triangulation_residual.corruption import corrupt_observations
from src.tasks.base.triangulation_residual.geometry import (
    InsufficientGeometryError,
    prepare_geometry,
)
from src.utils.configuration import PathRole


class ResidualDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        scenes: list[Path],
        loader: Callable[[Path], CleanResidualScene],
        config: ResidualConfig,
        split: str,
    ) -> None:
        self.scenes, self.loader, self.config, self.split = (
            scenes,
            loader,
            config,
            split,
        )
        self.epoch = multiprocessing.Value("q", 0)
        self.cache: OrderedDict[Path, CleanResidualScene] = OrderedDict()

    def __len__(self) -> int:
        return len(self.scenes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.value = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.scenes[index]
        if path in self.cache:
            scene = self.cache.pop(path)
            self.cache[path] = scene
        else:
            scene = self.loader(path)
            if self.config.data.cache_scenes:
                self.cache[path] = scene
                while len(self.cache) > self.config.data.cache_scenes:
                    self.cache.popitem(last=False)
        if scene.world_m.shape[1:] != (self.config.joints, 3):
            raise ValueError("Scene joints do not match selected residual profile")
        epoch = self.epoch.value if self.split == "train" else 0
        seed = (
            self.config.runtime.run.seed
            + index * 101
            + epoch * 1000003
            + {"train": 0, "val": 100000000, "test": 200000000}[self.split]
        )
        rng = np.random.default_rng(seed)
        length = self.config.data.sequence_length
        stride = max(1, round(scene.fps / self.config.data.target_fps))
        fps = scene.fps / stride
        span = (length - 1) * stride + 1
        max_start = max(0, len(scene.world_m) - span)
        start = (
            int(rng.integers(max_start + 1))
            if self.split == "train"
            else max_start // 2
        )
        indices: np.ndarray = start + np.arange(length) * stride
        frame_valid = indices < len(scene.world_m)
        world = np.asarray(
            scene.world_m[np.minimum(indices, len(scene.world_m) - 1)], dtype=np.float32
        ).copy()
        maximum = min(len(scene.rig.K), self.config.data.max_views)
        if maximum < self.config.data.min_views:
            raise ValueError(f"{scene.scene_id} has insufficient cameras")
        n_views = (
            int(rng.integers(self.config.data.min_views, maximum + 1))
            if self.split == "train"
            else maximum
        )
        candidates = list(combinations(range(len(scene.rig.K)), n_views))
        rng.shuffle(candidates)
        geometry = None
        draw_count = 0
        # Explicit observation-only feasibility sampling: a fixed scene/window
        # can be invisible in one camera subset but visible in another. Try all
        # subsets in randomized order before another corruption round. Never
        # replace the scene, GT, time window or split, or select on target error.
        for attempt in range(8):
            if attempt:
                rng.shuffle(candidates)
            for candidate in candidates:
                selected = rng.permutation(candidate).astype(np.int64)
                rig = scene.rig.subset(selected)
                draw_count += 1
                noisy = corrupt_observations(world, rig, rng, self.config.corruption)
                noisy.scores[:, ~frame_valid] = 0
                noisy.observations_px[:, ~frame_valid] = np.nan
                try:
                    geometry = prepare_geometry(
                        noisy.observations_px,
                        noisy.scores,
                        noisy.court_px,
                        noisy.court_scores,
                        noisy.estimated_rig,
                        root_indices=self.config.root_indices,
                        fps=fps,
                        min_score=self.config.initializer.min_score,
                        refinement_steps=self.config.initializer.refinement_steps,
                    )
                except InsufficientGeometryError:
                    continue
                break
            if geometry is not None:
                break
        if geometry is None:
            raise InsufficientGeometryError(
                f"No root geometry in any {n_views}-camera subset after 8 corruption rounds: {scene.scene_id}"
            )
        true_p = noisy.true_rig.matrices.copy()
        true_p[:, 0] /= rig.image_size[:, 0, None]
        true_p[:, 1] /= rig.image_size[:, 1, None]
        values: dict[str, Any] = {
            "features": geometry.features,
            "view_valid": geometry.view_valid & frame_valid[None],
            "time_positions": geometry.time_positions,
            "root_init": geometry.root_init_m,
            "relative_init": geometry.relative_init_m,
            "init_world": geometry.init_world_m,
            "init_valid": geometry.init_valid & frame_valid[:, None],
            "target_world": world,
            "frame_valid": frame_valid,
            "true_projection": true_p.astype(np.float32),
            "clean_uv": noisy.clean_uv,
            "clean_visible": noisy.clean_visible & frame_valid[None, :, None],
            "fps": np.array(fps, np.float32),
            "severity": np.array(noisy.severity, np.float32),
            "geometry_attempts": np.array(draw_count, np.int64),
            "corruption_rounds": np.array(attempt + 1, np.int64),
            "selected_camera_indices": selected,
        }
        tensors: dict[str, Any] = {
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in values.items()
        }
        tensors["scene_id"] = scene.scene_id
        return tensors


VIEW_FIELDS = frozenset(
    {
        "features",
        "view_valid",
        "true_projection",
        "clean_uv",
        "clean_visible",
        "selected_camera_indices",
    }
)


def collate_residual(samples: list[dict[str, Any]]) -> dict[str, Any]:
    views = max(sample["features"].shape[0] for sample in samples)
    batch: dict[str, Any] = {}
    for key in samples[0]:
        if key == "scene_id":
            batch[key] = [s[key] for s in samples]
            continue
        values: list[Tensor] = []
        for sample in samples:
            value = sample[key]
            if key in VIEW_FIELDS and value.shape[0] < views:
                padding = torch.full(
                    (views - value.shape[0], *value.shape[1:]),
                    -1 if key == "selected_camera_indices" else 0,
                    dtype=value.dtype,
                )
                value = torch.cat((value, padding))
            values.append(value)
        batch[key] = torch.stack(values)
    return batch


class ResidualDataModule(pl.LightningDataModule):
    def __init__(self, config: ResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.datasets: dict[str, ResidualDataset] = {}
        self.split_audit: dict[str, Any] = {}

    def setup(self, stage: str | None = None) -> None:
        if self.datasets:
            return
        root = self.config.runtime.resolver.resolve(
            PathRole.DATA, self.config.data.scene_dir
        )
        if self.config.task == "plcs":
            from src.tasks.plcs.triangulation_residual.data import (
                audit_source_splits,
                list_scene_paths,
                load_clean_scene,
            )

            self.split_audit = audit_source_splits(root)
        else:
            from src.tasks.blcs.triangulation_residual.data import (
                list_scene_paths,
                load_clean_scene,
            )

            groups = {
                split: {p.name for p in list_scene_paths(root, split)}
                for split in ("train", "val", "test")
            }
            if (
                groups["train"] & groups["val"]
                or groups["train"] & groups["test"]
                or groups["val"] & groups["test"]
            ):
                raise ValueError("BLCS scene split overlap")
            self.split_audit = {
                "scene_overlap": 0,
                "dataset": str(root),
                "source": "physics simulation",
            }
        for split in ("train", "val", "test"):
            paths = list_scene_paths(root, split)
            limit = getattr(self.config.data, f"{split}_limit")
            if limit:
                paths = paths[:limit]
            self.datasets[split] = ResidualDataset(
                paths, load_clean_scene, self.config, split
            )
        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.datasets[k] for k in ("train", "val", "test")
        )

    def _loader(self, split: str) -> DataLoader[dict[str, Any]]:
        cfg = self.config.data
        return DataLoader(
            self.datasets[split],
            batch_size=cfg.batch_size,
            shuffle=split == "train",
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=collate_residual,
            generator=torch.Generator().manual_seed(self.config.runtime.run.seed),
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("test")

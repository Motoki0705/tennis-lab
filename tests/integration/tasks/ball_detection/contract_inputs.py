"""Small, real frame stores and approved pose manifests for Ball contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.manifest import (
    build_coordinate_manifest,
    shared_pose_reference,
)
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from tests.integration.tasks.ball_detection.test_coordinate_training import (
    pose_manifest,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def write_windows(root: Path, *, pose: bool) -> Path:
    store_path = root / "store"
    for split in ("train", "val", "test"):
        labels = [
            frame(i, ball("unresolved", None))
            if i == 5
            else frame(i)
            if i % 8 == 3
            else frame(i, ball(xy=(8.0 + i / 16, 9.0 + i / 32)))
            for i in range(32)
        ]
        write_store_clip(
            store_path, f"{split}/pose", labels, split=split, size=(32, 32)
        )
    store = BallFrameStore(store_path)
    poses = pose_manifest(store_path, root / "poses")
    common, approved = shared_pose_reference(store, poses)
    manifest = build_coordinate_manifest(
        store,
        PlayIntervalConfig(),
        TemporalSamplingConfig((1,)),
        input_kind="mdd_pose" if pose else "mdd_only",
        pose_directory=poses if pose else None,
        approved_poses=approved if pose else None,
        common_reference=common,
    )
    path = root / "windows.json"
    path.write_text(json.dumps(manifest))
    return path


def window_loader(manifest: Path, *, pose: bool = False) -> Any:
    dataset = FrozenBallWindowDataset(manifest, split="train", requires_pose=pose)
    return CoordinateRuntime(cpu_threads=1).loader(dataset, batch_size=1)


def write_model_config(config: DictConfig, root: Path) -> Path:
    path = root / "model.yaml"
    OmegaConf.save(config.model, path)
    return path


def assert_window_batch(batch: Any) -> None:
    assert batch["rgb"].shape == (1, 32, 3, 32, 32)
    assert batch["rgb"].dtype == torch.uint8
    assert batch["position_valid"].dtype == batch["heatmap_valid"].dtype == torch.bool
    assert not batch["heatmap_valid"][0, 5] and not batch["position_valid"][0, 3]
    assert batch["heatmap_valid"][0, 3]  # confirmed empty, distinct from unresolved

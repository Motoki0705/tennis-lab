"""Small composed CPU settings and asymmetric camera observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_tiny(task: str, root: Path, gan: bool, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(Path(f"src/tasks/{task}/configs").resolve()), version_base="1.3"
    ):
        return compose(
            config_name="train",
            overrides=[
                f"training={'gan_small' if gan else 'default'}",
                f"paths.data_root={root.as_posix()}",
                "data.scene_dir=sceneset",
                "data.batch_size=2",
                "data.num_workers=0",
                "data.pin_memory=false",
                "data.seq_len_range=[3,5]",
                "data.num_views_range=[1,2]",
                "data.camera_mode=first",
                "data.augmentation.enabled=false",
                "model.hidden_dim=16",
                "model.num_layers=1",
                "model.num_heads=2",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.max_seq_len=8",
                "model.dropout=0.0",
                "training.compile.enabled=false",
                "training.learning_rate=0.001",
                "training.weight_decay=0.0",
                "training.warmup_steps=0",
                "training.steps_per_epoch=1",
                "training.trainer.max_epochs=2",
                "training.qualitative_logging.enabled=false",
                *(
                    [
                        "training.gan.discriminator.hidden_dim=16",
                        "training.gan.discriminator.num_layers=1",
                        "training.gan.discriminator.num_heads=2",
                        "training.gan.discriminator.ffn_dim=32",
                        "training.gan.discriminator.rope_dim=4",
                        "training.gan.discriminator.max_seq_len=8",
                        "training.gan.discriminator.dropout=0.0",
                        "training.gan.transition.start_epoch=1",
                        "training.gan.target_weight=0.1",
                        "training.gan.warmup_epochs=1",
                    ]
                    if gan
                    else []
                ),
                *overrides,
            ],
        )


def camera_observations(frames: int, index: int, joints: int) -> dict[str, Any]:
    """One view, with an all-object-missing frame and partial court visibility."""
    uv: np.ndarray = np.linspace(
        0.2, 0.8, frames * joints * 2, dtype=np.float32
    ).reshape(frames, joints, 2)
    uv += index * 0.01
    vis: np.ndarray = np.ones((frames, joints), dtype=np.bool_)
    vis[1] = False
    uv[~vis] = 0
    court: np.ndarray = np.linspace(0.1, 0.9, 40, dtype=np.float32).reshape(20, 2)
    court_vis: np.ndarray = np.ones(20, dtype=np.bool_)
    court_vis[2] = False
    court[2] = 0
    return {
        "camera_params": {
            "R": np.eye(3).tolist(),
            "C": [float(index), -12.0, -30.0],
            "f": 800.0,
            "cx": 640.0,
            "cy": 360.0,
            "w": 1280,
            "h": 720,
        },
        "uv": uv,
        "vis": vis,
        "court_kp_uv": court,
        "court_kp_vis": court_vis,
    }


def write_splits(root: Path) -> None:
    # Both examples must reach collate together; split identity isn't under test.
    for split in ("train", "val", "test"):
        (root / f"{split}.txt").write_text("scene_0\nscene_1\n")

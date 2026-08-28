"""Base-specific fixtures: dummy scenes, tiny configs, minimal subclasses.

These fixtures support unit tests for ``src.tasks.base`` abstractions. They are
intentionally lightweight (no real models, no real datasets) so the tests stay
fast and CPU-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.base.data.scene_dataset import (
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
)


@pytest.fixture
def make_training_config(tmp_path: Path):
    """Build a complete shared training config with explicit test values."""

    def _factory(
        *,
        run: dict[str, Any] | None = None,
        trainer: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_config: dict[str, Any] = {
            "output_dir": "run",
            "seed": 1,
            "gpus": 0,
            "resume": None,
            "init_weights": None,
            "fast_dev_run": False,
            "dry_run": False,
            "test_after_fit": False,
        }
        run_config.update(run or {})
        trainer_config: dict[str, Any] = {
            "max_epochs": 2,
            "gradient_clip_val": None,
            "deterministic": True,
            "precision": "32-true",
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 1,
            "accumulate_grad_batches": 1,
            "reload_dataloaders_every_n_epochs": 0,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "benchmark": False,
        }
        trainer_config.update(trainer or {})
        training_config: dict[str, Any] = {
            "trainer": trainer_config,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "warmup_epochs": None,
            "min_lr": 0.0,
            "steps_per_epoch": 1,
            "optimizer": {"betas": [0.9, 0.999]},
            "checkpoint": {
                "enabled": False,
                "filename": "model-{epoch}",
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": False,
            },
            "early_stopping": {
                "enabled": False,
                "monitor": "val/loss",
                "mode": "min",
                "patience": 1,
                "min_delta": 0.0,
                "check_on_train_epoch_end": False,
            },
            "lr_monitor": {"enabled": False, "interval": "step"},
            "qualitative_logging": {
                "enabled": False,
                "every_n_epochs": 1,
                "num_samples": 1,
                "selection_mode": "random",
                "selected_indices": None,
            },
            "gan": {
                "enabled": False,
                "target_weight": 0.0,
                "warmup_epochs": 1,
                "generator_gradient_clip_val": None,
                "discriminator_gradient_clip_val": None,
                "transition": {"start_epoch": 0},
            },
            "compile": {
                "enabled": True,
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": False,
            },
            "matmul_precision": "high",
            "allow_tf32": False,
        }
        training_config.update(training or {})
        return {
            "paths": {
                "project_root": str(tmp_path),
                "data_root": "data",
                "checkpoint_root": str(tmp_path),
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "external",
            },
            "run": run_config,
            "training": training_config,
        }

    return _factory


def write_scene_dir(
    scene_dir: Path,
    *,
    num_frames: int = 8,
    num_cameras: int = 1,
    arrays: dict[str, np.ndarray] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal on-disk scene directory (meta.json, scalars.json, npy).

    Returns the scene directory path.
    """
    scene_dir.mkdir(parents=True, exist_ok=True)
    meta_payload: dict[str, Any] = {"num_frames": num_frames}
    if meta:
        meta_payload.update(meta)
    (scene_dir / "meta.json").write_text(json.dumps(meta_payload), encoding="utf-8")
    (scene_dir / "scalars.json").write_text(
        json.dumps({"num_cameras": num_cameras}), encoding="utf-8"
    )
    arr = (
        arrays
        if arrays is not None
        else {"position": np.zeros((num_frames, 3), np.float32)}
    )
    for key, value in arr.items():
        np.save(scene_dir / f"{key}.npy", value)
    return scene_dir


@pytest.fixture
def scene_writer():
    """Expose ``write_scene_dir`` as a fixture (importlib mode hides the module)."""
    return write_scene_dir


class _ConcreteSceneDataset(SceneDatasetBase[dict]):
    """Minimal concrete dataset whose sample is just scene metadata."""

    def build_sample(self, scene: Scene) -> dict:
        random_draw = int(self.rng.integers(0, 2**31))
        cameras = self.select_cameras(scene)
        window = self.select_window(scene)
        return {
            "path": str(scene.path),
            "num_frames": scene.num_frames,
            "random_draw": random_draw,
            "camera_indices": cameras.indices,
            "window_start": window.start,
            "window_length": window.seq_len,
        }


@pytest.fixture
def make_scene_dataset(tmp_path: Path):
    """Factory building a concrete ``SceneDatasetBase`` over on-disk scenes.

    With ``n_scenes > 0`` it writes that many scene directories plus a
    ``train.txt`` split and returns an initialized dataset. With ``n_scenes == 0``
    nothing is written; the caller is expected to supply a ``config`` and to have
    laid out the scenes/split themselves (used for filtering/error tests).
    """

    def _factory(
        *,
        n_scenes: int = 3,
        num_frames: int = 8,
        num_cameras: int = 1,
        config: SceneDatasetConfig | None = None,
        seed: int = 0,
        rng: np.random.Generator | None = None,
        sample_local_rng: bool = False,
        root: Path | None = None,
    ) -> SceneDatasetBase:
        root = root or (tmp_path / f"ds_{n_scenes}_{num_frames}_{num_cameras}")
        cfg = config
        if n_scenes > 0:
            scenes_dir = root / "scenes"
            names = []
            for i in range(n_scenes):
                name = f"scene_{i:04d}"
                write_scene_dir(
                    scenes_dir / name,
                    num_frames=num_frames,
                    num_cameras=num_cameras,
                )
                names.append(name)
            split_file = root / "train.txt"
            split_file.write_text("\n".join(names) + "\n", encoding="utf-8")
            cfg = config or SceneDatasetConfig(
                scene_dir=root,
                split_file=split_file,
                seq_len_range=(1, num_frames),
                num_views_range=(1, num_cameras),
                camera_mode="random",
                crop_mode="random",
                min_num_frames=1,
                min_num_cameras=1,
            )
        if cfg is None:
            raise ValueError("config is required when n_scenes == 0")
        return _ConcreteSceneDataset(
            config=cfg,
            seed=seed,
            rng=rng,
            sample_local_rng=sample_local_rng,
        )

    return _factory


@pytest.fixture
def make_scene():
    """Factory building an in-memory ``Scene`` (no disk IO)."""

    def _factory(
        *,
        num_frames: int = 8,
        num_cameras: int = 2,
        data: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> Scene:
        payload = (
            data
            if data is not None
            else {
                "cam_0_ball_uv": np.zeros((num_frames, 2), np.float32),
                "position": np.zeros((num_frames, 3), np.float32),
            }
        )
        return Scene(
            path=path or Path("/tmp/scene_x"),
            data=payload,
            meta=meta or {},
            num_frames=num_frames,
            num_cameras=num_cameras,
        )

    return _factory

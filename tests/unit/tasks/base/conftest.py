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
    arr = arrays if arrays is not None else {"position": np.zeros((num_frames, 3), np.float32)}
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
        return {"path": str(scene.path), "num_frames": scene.num_frames}


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
        rng: np.random.Generator | None = None,
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
            )
        if cfg is None:
            raise ValueError("config is required when n_scenes == 0")
        return _ConcreteSceneDataset(config=cfg, rng=rng or np.random.default_rng(0))

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
        payload = data if data is not None else {
            "cam_0_ball_uv": np.zeros((num_frames, 2), np.float32),
            "position": np.zeros((num_frames, 3), np.float32),
        }
        return Scene(
            path=path or Path("/tmp/scene_x"),
            data=payload,
            meta=meta or {},
            num_frames=num_frames,
            num_cameras=num_cameras,
        )

    return _factory

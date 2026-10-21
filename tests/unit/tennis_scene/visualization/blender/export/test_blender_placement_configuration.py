"""Strict configuration boundary for the Blender placement exporter."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tennis_scene.configuration import parse_blender_placement_config
from src.utils.configuration import ConfigurationError, SemanticConfigurationError
from src.utils.paths import PROJECT_ROOT

_CONFIG_DIR = (PROJECT_ROOT / "src/tennis_scene/configs").resolve()


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    store = root / "outputs/stores/clip"
    store.mkdir(parents=True, exist_ok=True)
    video = root / "data/videos/cam0.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    faces = root / "ckpt/body_models/smplh/neutral/model.npz"
    faces.parent.mkdir(parents=True, exist_ok=True)
    faces.write_bytes(b"faces")
    return store, video, faces


def _config(root: Path, *overrides: str, omit: str | None = None) -> DictConfig:
    _inputs(root)
    base = {
        "store": "stores/clip",
        "identity": "0",
        "frames": "2:12",
        "t0": "4",
        "plate_camera_id": "cam0",
        "plate_video_path": "videos/cam0.mp4",
        "human_vis_threshold": "0.15",
    }
    if omit is not None:
        base.pop(omit)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="blender_placement",
            overrides=[
                f"paths.project_root={root.resolve()}",
                *(f"{key}={value}" for key, value in base.items()),
                *overrides,
            ],
        )


def test_shipped_config_resolves_roles_and_explicit_choices(tmp_path: Path) -> None:
    runtime = parse_blender_placement_config(_config(tmp_path))

    assert runtime.store_path == (tmp_path / "outputs/stores/clip").resolve()
    assert runtime.export.identity == 0
    assert runtime.export.frames == (2, 12)
    assert runtime.export.t0 == 4
    assert runtime.export.plate_camera_id == "cam0"
    assert runtime.export.plate_video_path == (tmp_path / "data/videos/cam0.mp4")
    assert runtime.export.human_vis_threshold == 0.15
    assert runtime.smpl_faces_path == (
        tmp_path / "ckpt/body_models/smplh/neutral/model.npz"
    )
    assert runtime.output_path.name == "bundle"
    assert runtime.output_path.is_relative_to(
        tmp_path / "outputs/tennis_scene/visualize/blender_placement"
    )


@pytest.mark.parametrize(
    "override",
    [
        "mode=rank",
        "identity=-1",
        "frames=5:5",
        "frames=0",
        "frames=0:abc",
        "t0=1",
        "t0=99",
        "human_vis_threshold=1.5",
        "plate_jpeg_quality=0",
        "plate_camera_id=",
        "+unexpected_key=1",
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, override: str) -> None:
    with pytest.raises(ConfigurationError):
        parse_blender_placement_config(_config(tmp_path, override))


def test_missing_identity_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SemanticConfigurationError, match="identity is required"):
        parse_blender_placement_config(_config(tmp_path, omit="identity"))


@pytest.mark.parametrize("field", ["frames", "t0", "plate_video_path"])
def test_missing_required_keys_are_rejected(tmp_path: Path, field: str) -> None:
    with pytest.raises(SemanticConfigurationError, match=f"{field} is required"):
        parse_blender_placement_config(_config(tmp_path, omit=field))


def test_missing_store_and_plate_video_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SemanticConfigurationError, match="not a directory"):
        parse_blender_placement_config(_config(tmp_path, "store=stores/missing"))

    with pytest.raises(SemanticConfigurationError, match="not a file"):
        parse_blender_placement_config(
            _config(tmp_path, "plate_video_path=videos/missing.mp4")
        )

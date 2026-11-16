"""Validated model assets and their content identities (never loaded models)."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.submodules.configuration import SmplAssetPaths, SubmoduleRuntimeConfig
from src.utils.checksum import dual_sha256


@lru_cache(maxsize=128)
def _asset_digest(path: Path, stat_identity: tuple[int, ...]) -> str:
    digest: str = dual_sha256(path)
    return digest


def file_identity(path: Path) -> dict[str, Any]:
    """Content identity of a required asset; missing assets stop the build."""
    if not path.is_file():
        raise FileNotFoundError(f"Required pipeline asset is missing: {path}")
    stat = path.stat()
    return {"path": str(path), "sha256": _asset_digest(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))}


def asset_identities(enabled: bool, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: file_identity(path) for name, path in paths.items()} if enabled else {"enabled": False}


@dataclass(frozen=True)
class PeopleModelConfig:
    dino_checkpoint: Path
    dino_repository: Path
    vitpose_checkpoint: Path
    hmr2_checkpoint: Path
    gvhmr_checkpoint: Path
    body_models_dir: Path
    smpl_assets: SmplAssetPaths
    runtime: SubmoduleRuntimeConfig

    def __post_init__(self) -> None:
        if not self.runtime.static_cam:
            raise ValueError("Automatic reconstruction currently requires static cameras")

    def body_assets(self) -> dict[str, Path]:
        """Every file the GVHMR body recovery and SMPL placement read."""
        return {
            "hmr2": self.hmr2_checkpoint,
            "gvhmr": self.gvhmr_checkpoint,
            "body_model": self.body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz",
            **{f"smpl_{field.name}": getattr(self.smpl_assets, field.name) for field in fields(self.smpl_assets)},
        }

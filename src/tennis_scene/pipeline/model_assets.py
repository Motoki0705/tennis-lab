"""Validated model assets and their content identities (never loaded models)."""

from __future__ import annotations

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


def asset_hashes(settings: Mapping[str, Any]) -> dict[Path, str]:
    """Read file_identity snapshots already bound to an artifact's settings."""
    hashes: dict[Path, str] = {}
    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                path, digest = Path(value["path"]), value["sha256"]
                if path in hashes and hashes[path] != digest:
                    raise ValueError(f"Conflicting artifact asset identities: {path}")
                hashes[path] = digest
            for child in value.values():
                collect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                collect(child)
    collect(settings)
    return hashes


@dataclass(frozen=True)
class ModelAssetSnapshot:
    """Content versions shared by a publication identity and its execution plan."""

    files: tuple[tuple[Path, str], ...]

    @classmethod
    def capture(cls, identity: Mapping[str, Any]) -> ModelAssetSnapshot:
        return cls(tuple(asset_hashes(identity).items()))

    def require_settings(self, settings: Mapping[str, Any]) -> None:
        expected = dict(self.files)
        for path, digest in asset_hashes(settings).items():
            if expected.get(path) != digest:
                raise RuntimeError(f"Plan disagrees with publication asset snapshot: {path}")

    def assert_unchanged(self) -> None:
        for path, digest in self.files:
            try:
                current = file_identity(path)["sha256"]
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Cannot verify publication asset snapshot: {path}") from error
            if current != digest:
                raise RuntimeError(f"Changed publication asset snapshot: {path}")


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

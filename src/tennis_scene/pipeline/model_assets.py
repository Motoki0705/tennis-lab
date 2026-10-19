"""Validated model assets; model execution belongs to individual components."""

from dataclasses import dataclass, fields
from pathlib import Path

from src.submodules.configuration import BundledModelAssetPaths, SubmoduleRuntimeConfig


@dataclass(frozen=True)
class PeopleModelConfig:
    dino_checkpoint: Path
    dino_repository: Path
    vitpose_checkpoint: Path
    hmr2_checkpoint: Path
    gvhmr_checkpoint: Path
    body_models_dir: Path
    bundled_assets: BundledModelAssetPaths
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
            **{f"bundled_{field.name}": getattr(self.bundled_assets, field.name) for field in fields(self.bundled_assets)},
        }

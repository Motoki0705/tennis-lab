"""Validated model assets; model execution belongs to individual components."""

from dataclasses import dataclass
from pathlib import Path

from src.submodules.configuration import BundledModelAssetPaths, SubmoduleRuntimeConfig


@dataclass(frozen=True)
class PeopleModelConfig:
    detector: str
    dino_checkpoint: Path
    dino_repository: Path
    yolo_checkpoint: Path
    vitpose_checkpoint: Path
    hmr2_checkpoint: Path
    gvhmr_checkpoint: Path
    body_models_dir: Path
    bundled_assets: BundledModelAssetPaths
    runtime: SubmoduleRuntimeConfig

    def __post_init__(self) -> None:
        if self.detector not in {"dino", "yolo"}:
            raise ValueError("Person detector must be dino or yolo")
        if not self.runtime.static_cam:
            raise ValueError("Automatic reconstruction currently requires static cameras")


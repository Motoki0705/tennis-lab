"""Validated model assets and their content identities (never loaded models)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from src.submodules.configuration import (
    BodyRuntimeConfig,
    DinoDetectorRuntimeConfig,
    RFDetrRuntimeConfig,
    RTMPoseRuntimeConfig,
    SmplAssetPaths,
    SubmoduleRuntimeConfig,
    ViTPoseRuntimeConfig,
)
from src.utils.checksum import dual_sha256

DetectorName = Literal["dino", "rfdetr-l"]
PoseName = Literal["vitpose-h", "rtmpose-x"]


def people_model_names(values: Mapping[str, object]) -> tuple[DetectorName, PoseName]:
    # Older saved pipeline settings predate the two selectors.
    effective = {"detector": "dino", "pose": "vitpose-h", **values}
    detector, pose = effective["detector"], effective["pose"]
    if detector not in ("dino", "rfdetr-l"):
        raise ValueError("people_models.detector must be dino or rfdetr-l")
    if pose not in ("vitpose-h", "rtmpose-x"):
        raise ValueError("people_models.pose must be vitpose-h or rtmpose-x")
    return detector, pose


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
    detector: DetectorName
    pose: PoseName
    rfdetr_checkpoint: Path | None
    rtmpose_checkpoint: Path | None

    def __post_init__(self) -> None:
        people_model_names({"detector": self.detector, "pose": self.pose})
        if self.detector == "rfdetr-l":
            self.rfdetr_settings()
        if self.pose == "rtmpose-x":
            self.rtmpose_settings()
        if not self.runtime.static_cam:
            raise ValueError("Automatic reconstruction currently requires static cameras")

    def rfdetr_settings(self) -> tuple[Path, RFDetrRuntimeConfig]:
        if self.rfdetr_checkpoint is None or self.runtime.rfdetr_l is None:
            raise ValueError("RF-DETR-L requires people_models.rfdetr_checkpoint and runtime.rfdetr_l")
        return self.rfdetr_checkpoint, self.runtime.rfdetr_l

    def rtmpose_settings(self) -> tuple[Path, RTMPoseRuntimeConfig]:
        if self.rtmpose_checkpoint is None or self.runtime.rtmpose_x is None:
            raise ValueError("RTMPose-X requires people_models.rtmpose_checkpoint and runtime.rtmpose_x")
        return self.rtmpose_checkpoint, self.runtime.rtmpose_x

    @property
    def detector_runtime(self) -> DinoDetectorRuntimeConfig | RFDetrRuntimeConfig:
        return self.runtime.dino_detector if self.detector == "dino" else self.rfdetr_settings()[1]

    @property
    def pose_runtime(self) -> ViTPoseRuntimeConfig | RTMPoseRuntimeConfig:
        return self.runtime.vitpose if self.pose == "vitpose-h" else self.rtmpose_settings()[1]

    @property
    def uses_tensorrt(self) -> bool:
        return bool(self.detector_runtime.engines or self.pose_runtime.engines
                    or self.runtime.hmr2.backend.engines or self.runtime.clip_reid.backend.engines
                    or self.runtime.gvhmr.backend.engines)

    @property
    def uses_alternative_models(self) -> bool:
        return (self.detector, self.pose) != ("dino", "vitpose-h")

    def detection_assets(self) -> dict[str, Path]:
        return {"checkpoint": self.dino_checkpoint if self.detector == "dino" else self.rfdetr_settings()[0]}

    def pose_assets(self) -> dict[str, Path]:
        return {"pose": self.vitpose_checkpoint if self.pose == "vitpose-h" else self.rtmpose_settings()[0]}

    def body_assets(self) -> dict[str, Path]:
        return _body_assets(self)


@dataclass(frozen=True)
class BodyModelConfig:
    """Asset and runtime inputs for the HMR2, GVHMR, and placement stages."""

    hmr2_checkpoint: Path
    gvhmr_checkpoint: Path
    body_models_dir: Path
    smpl_assets: SmplAssetPaths
    runtime: BodyRuntimeConfig

    def __post_init__(self) -> None:
        if not self.runtime.static_cam:
            raise ValueError("Automatic reconstruction currently requires static cameras")

    def body_assets(self) -> dict[str, Path]:
        return _body_assets(self)


def _body_assets(config: BodyModelConfig | PeopleModelConfig) -> dict[str, Path]:
    """Every file the GVHMR body recovery and SMPL placement read."""
    return {
        "hmr2": config.hmr2_checkpoint,
        "gvhmr": config.gvhmr_checkpoint,
        "body_model": config.body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz",
        **{f"smpl_{field.name}": getattr(config.smpl_assets, field.name) for field in fields(config.smpl_assets)},
    }

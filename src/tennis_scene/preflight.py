"""Collect installation failures without constructing inference models or using CUDA."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.submodules.models import validate_dino_extension
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.installation_assets import INSTALLATION_ASSETS
from src.tennis_scene.pipeline.definition import enabled_model_assets
from src.utils.checksum import FileIntegrityError, dual_sha256
from src.utils.configuration import PathRole
from src.utils.sample_dataset import PUBLISHED_FILES


@dataclass(frozen=True)
class AssetRequirement:
    name: str
    path: Path
    sha256: str | None = None


@dataclass(frozen=True)
class AssetCheck:
    name: str
    path: str
    status: str
    detail: str
    sha256: str | None = None


def standard_requirements(cfg: PipelineRuntimeConfig) -> tuple[AssetRequirement, ...]:
    """Every bound model asset plus the released court/visualization dependencies.

    Digests qualify the published standard recipe, including its separate ViT-B
    backbone. Alternate model releases require their own qualification pins;
    this command never selects another checkpoint on a mismatch.
    """
    pins = {asset.name: asset for asset in INSTALLATION_ASSETS}
    pins["association_encoder"] = pins["tracking_encoder"]
    requirements = [AssetRequirement(name, path, pins[name].sha256)
                    for name, path in enabled_model_assets(cfg).items()]
    for name in ("dinov3", "smpl_faces"):
        asset = pins[name]
        requirements.append(AssetRequirement(name, cfg.resolver.resolve(PathRole.CHECKPOINT, asset.filename), asset.sha256))
    if "smpl_smpl_neutral_joint_regressor" not in {item.name for item in requirements}:
        asset = pins["smpl_smpl_neutral_joint_regressor"]
        requirements.append(AssetRequirement(asset.name, cfg.resolver.resolve(PathRole.CHECKPOINT, asset.filename), asset.sha256))
    source_files = {
        "dino_models": cfg.people.dino_repository / "models/__init__.py",
        "dino_util": cfg.people.dino_repository / "util/__init__.py",
        "dino_ops": cfg.people.dino_repository / "models/dino/ops/src/vision.cpp",
        "dinov3_hub": cfg.resolver.resolve(PathRole.EXTERNAL_ASSET, "dinov3/hubconf.py"),
        "dinov3_backbones": cfg.resolver.resolve(PathRole.EXTERNAL_ASSET, "dinov3/dinov3/hub/backbones.py"),
    }
    requirements.extend(AssetRequirement(name, path) for name, path in source_files.items())
    sample_pins = {cfg.roots.data_root / "samples" / item.filename: item.sha256 for item in PUBLISHED_FILES}
    requirements.extend(AssetRequirement(f"video_{camera}", path, sample_pins.get(path))
                        for camera, path in zip(cfg.camera_ids, cfg.video_paths, strict=True))
    # The sample manifest and license accompany its videos; custom inputs do not
    # claim to be the published sample and are checked for presence only.
    if cfg.video_paths and all(path in sample_pins for path in cfg.video_paths):
        requirements.extend(AssetRequirement(f"sample_{path.name}", path, digest)
                            for path, digest in sample_pins.items() if path.suffix != ".mp4")
    return tuple(requirements)


def check_assets(
    requirements: Sequence[AssetRequirement],
    *,
    extension_validator: Callable[[], Path] = validate_dino_extension,
) -> tuple[AssetCheck, ...]:
    """Report every missing/corrupt/unreadable file and the CPU-only op check.

    Hash results are shared only within this call (CLIP serves two components).
    A read/hash failure is a failed check, never an instruction to skip it.
    """
    checks = []
    digests: dict[Path, str] = {}
    for requirement in requirements:
        name, path, expected = requirement.name, requirement.path, requirement.sha256
        try:
            if not path.is_file():
                checks.append(AssetCheck(name, str(path), "missing", "Required file is absent"))
                continue
            if expected is None:
                checks.append(AssetCheck(name, str(path), "ok", "File exists; no byte pin for source/custom input"))
                continue
            if path not in digests:
                digests[path] = dual_sha256(path)
            actual = digests[path]
            status = "ok" if actual == expected else "mismatch"
            checks.append(AssetCheck(name, str(path), status, f"Expected SHA-256 {expected}", actual))
        except (OSError, RuntimeError, ValueError, FileIntegrityError) as error:
            checks.append(AssetCheck(name, str(path), "error", str(error)))
    try:
        binary = extension_validator()
        checks.append(AssetCheck("dino_extension", str(binary), "ok", "CPU forward/backward dispatch validated; CUDA kernels untested"))
    except (ImportError, OSError, RuntimeError) as error:
        checks.append(AssetCheck("dino_extension", "PYTHONPATH:MultiScaleDeformableAttention", "error", str(error)))
    return tuple(checks)

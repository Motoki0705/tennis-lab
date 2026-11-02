"""Collect installation failures without constructing inference models or using CUDA."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from omegaconf import DictConfig
from yaml import YAMLError

from src.submodules.models import validate_dino_extension
from src.tasks.player_association.appearance.encoders import encoder_weights
from src.tasks.player_association.association.config import (
    DEFAULT_CONFIG,
    load_association_config,
)
from src.tennis_scene.configuration import (
    PipelineRuntimeConfig,
    pipeline_asset_configuration,
)
from src.tennis_scene.installation_assets import INSTALLATION_ASSETS
from src.tennis_scene.pipeline.definition import enabled_model_assets
from src.utils.checksum import FileIntegrityError, dual_sha256
from src.utils.configuration import PathContractError, PathRole
from src.utils.paths import PROJECT_ROOT
from src.utils.sample_dataset import PUBLISHED_FILES


@dataclass(frozen=True)
class AssetRequirement:
    name: str
    path: Path
    sha256: str | None = None
    path_error: str | None = None
    error_status: str = "invalid_path"


@dataclass(frozen=True)
class AssetCheck:
    name: str
    path: str
    status: str
    detail: str
    sha256: str | None = None


def configured_requirements(cfg: DictConfig) -> tuple[AssetRequirement, ...]:
    """Build a diagnostic inventory, retaining rejected paths without reading them.

    A rejected path is display-only. No substitute path or runnable configuration
    is produced. The typed-runtime inventory is checked for parity in tests.
    """
    value, roots, resolver = pipeline_asset_configuration(cfg)
    pins = {asset.name: asset for asset in INSTALLATION_ASSETS}
    pins["association_encoder"] = pins["tracking_encoder"]
    requirements: list[AssetRequirement] = []

    def section(name: str) -> Mapping[str, object]:
        return cast(Mapping[str, object], value[name])

    def add(name: str, role: PathRole, *parts: str | Path, digest: str | None = None) -> None:
        # Keep the configured entry for diagnostics; only the resolver's accepted
        # result may reach check_assets' file inspection or hashing.
        entry = roots.root(role).joinpath(*parts)
        try:
            path = resolver.resolve(role, *parts)
        except (PathContractError, OSError, RuntimeError, ValueError) as error:
            requirements.append(AssetRequirement(name, entry, digest, str(error)))
        else:
            requirements.append(AssetRequirement(name, path, digest))

    def checkpoint(name: str, path: object) -> None:
        add(name, PathRole.CHECKPOINT, cast(str, path), digest=pins[name].sha256)

    models = section("people_models")
    checkpoint("court", section("court_kp")["checkpoint"])
    checkpoint("ball", section("ball_detection")["checkpoint"])
    if section("person_observations")["enabled"]:
        checkpoint("detector", models["dino_checkpoint"])
        checkpoint("vitpose", models["vitpose_checkpoint"])
        tracking = section("person_tracking")
        tracking_path = encoder_weights(cast(str, tracking["encoder"]), checkpoint_root=roots.checkpoint_root,
                                        external_root=roots.external_asset_root)
        checkpoint("tracking_encoder", tracking_path.relative_to(roots.checkpoint_root))
        checkpoint("aflink", tracking["aflink_checkpoint"])
        # The association recipe is itself an input. Validate its authority
        # before loading YAML, even when another asset is already invalid.
        try:
            association_path = resolver.resolve(PathRole.PROJECT, DEFAULT_CONFIG.relative_to(PROJECT_ROOT))
            association = load_association_config(association_path,
                players_per_side=cast(int, section("player_association")["players_per_side"]))
        except (OSError, RuntimeError, TypeError, ValueError, YAMLError) as error:
            requirements.append(AssetRequirement("association_configuration", roots.project_root / DEFAULT_CONFIG.relative_to(PROJECT_ROOT),
                path_error=f"Association encoder inventory unavailable: {error}",
                error_status="invalid_path" if isinstance(error, PathContractError) else "error"))
        else:
            if association.appearance is not None:
                association_path = encoder_weights(association.appearance.encoder, checkpoint_root=roots.checkpoint_root,
                                                    external_root=roots.external_asset_root)
                checkpoint("association_encoder", association_path.relative_to(roots.checkpoint_root))
    if section("gvhmr")["enabled"]:
        checkpoint("hmr2", models["hmr2_checkpoint"])
        checkpoint("gvhmr", models["gvhmr_checkpoint"])
        add("body_model", PathRole.CHECKPOINT, cast(str, models["body_models_dir"]), "smplx/SMPLX_NEUTRAL.npz",
            digest=pins["body_model"].sha256)
        for name, path in cast(Mapping[str, str], models["smpl_assets"]).items():
            checkpoint(f"smpl_{name}", path)
    for name in ("dinov3", "smpl_faces", "smpl_smpl_neutral_joint_regressor"):
        if name not in {item.name for item in requirements}:
            checkpoint(name, pins[name].filename)
    for name, suffix in (("dino_models", "models/__init__.py"), ("dino_util", "util/__init__.py"),
                         ("dino_ops", "models/dino/ops/src/vision.cpp")):
        add(name, PathRole.EXTERNAL_ASSET, cast(str, models["dino_repository"]), suffix)
    for name, path in (("dinov3_hub", "dinov3/hubconf.py"), ("dinov3_backbones", "dinov3/dinov3/hub/backbones.py")):
        add(name, PathRole.EXTERNAL_ASSET, path)
    sample_pins = {roots.data_root / "samples" / item.filename: item.sha256 for item in PUBLISHED_FILES}
    videos = cast(Sequence[str], value["video_paths"])
    cameras = cast(Sequence[str], value["camera_ids"])
    if len(videos) != len(cameras):
        raise ValueError("Video paths and camera IDs must have the same length; asset-to-camera mapping is unavailable")
    video_requirements = []
    for camera, path in zip(cameras, videos, strict=True):
        add(f"video_{camera}", PathRole.DATA, path)
        # Accepted paths have already been canonicalized by the resolver. Match
        # those paths, not caller spelling, so ./ and contained ../ cannot turn
        # a pinned sample into an unpinned custom input.
        requirement = requirements[-1]
        requirements[-1] = replace(requirement, sha256=sample_pins.get(requirement.path))
        video_requirements.append(requirements[-1])
    if video_requirements and all(item.path in sample_pins for item in video_requirements):
        for path, digest in sample_pins.items():
            if path.suffix != ".mp4":
                add(f"sample_{path.name}", PathRole.DATA, path.relative_to(roots.data_root), digest=digest)
    return tuple(requirements)


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
        if requirement.path_error is not None:
            checks.append(AssetCheck(name, str(path), requirement.error_status, requirement.path_error))
            continue
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

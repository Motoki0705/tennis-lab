"""Strict runtime configuration and asset paths for external model submodules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.utils.configuration import (
    ConfigField,
    ConfigFieldContract,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)

__all__ = [
    "SUBMODULE_RUNTIME_SCHEMA",
    "BUNDLED_MODEL_ASSET_SCHEMA",
    "BundledModelAssetPaths",
    "DinoDetectorRuntimeConfig",
    "ExternalModelAssetPaths",
    "GvhmrDemoConfig",
    "Hmr2RuntimeConfig",
    "SubmoduleRuntimeConfig",
    "TrackingRuntimeConfig",
    "ViTPoseRuntimeConfig",
    "ViTPoseHeadConfig",
    "inspect_gvhmr_demo_schema",
    "require_absolute_path",
]


@dataclass(frozen=True, slots=True)
class ViTPoseHeadConfig:
    """Complete released heatmap-head architecture with no inferred fields."""

    in_channels: int
    out_channels: int
    num_deconv_layers: int
    num_deconv_filters: tuple[int, ...]
    num_deconv_kernels: tuple[int, ...]
    final_conv_kernel: int
    num_conv_layers: int
    num_conv_kernels: tuple[int, ...]

    def __post_init__(self) -> None:
        integer_fields = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "num_deconv_layers": self.num_deconv_layers,
            "final_conv_kernel": self.final_conv_kernel,
            "num_conv_layers": self.num_conv_layers,
        }
        for name, value in integer_fields.items():
            if type(value) is not int:
                raise TypeError(f"ViTPose head {name} must be exactly int.")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("ViTPose head channel counts must be positive.")
        if self.num_deconv_layers < 0 or self.num_conv_layers < 0:
            raise ValueError("ViTPose head layer counts must be non-negative.")
        sequences = {
            "num_deconv_filters": self.num_deconv_filters,
            "num_deconv_kernels": self.num_deconv_kernels,
            "num_conv_kernels": self.num_conv_kernels,
        }
        for name, values in sequences.items():
            if type(values) is not tuple or any(
                type(value) is not int for value in values
            ):
                raise TypeError(f"ViTPose head {name} must be a tuple of exact ints.")
        if len(self.num_deconv_filters) != self.num_deconv_layers:
            raise ValueError("num_deconv_filters length must equal num_deconv_layers.")
        if len(self.num_deconv_kernels) != self.num_deconv_layers:
            raise ValueError("num_deconv_kernels length must equal num_deconv_layers.")
        if len(self.num_conv_kernels) != self.num_conv_layers:
            raise ValueError("num_conv_kernels length must equal num_conv_layers.")
        if any(value <= 0 for value in self.num_deconv_filters):
            raise ValueError("ViTPose deconvolution filters must be positive.")
        if any(value not in {2, 3, 4} for value in self.num_deconv_kernels):
            raise ValueError("ViTPose deconvolution kernels must be 2, 3, or 4.")
        if self.final_conv_kernel not in {0, 1, 3}:
            raise ValueError("ViTPose final_conv_kernel must be 0, 1, or 3.")
        if any(value not in {1, 3} for value in self.num_conv_kernels):
            raise ValueError("ViTPose convolution kernels must be 1 or 3.")
        if self.final_conv_kernel == 0 and self.num_conv_layers:
            raise ValueError(
                "Identity final mapping forbids preceding convolution layers."
            )


_CHECKPOINT_SCHEMA = StrictConfigSchema(
    name="submodules.assets.checkpoints",
    fields={
        "yolo": ConfigField.of(str),
        "dino": ConfigField.of(str),
        "vitpose": ConfigField.of(str),
        "hmr2": ConfigField.of(str),
        "gvhmr": ConfigField.of(str),
    },
)
BUNDLED_MODEL_ASSET_SCHEMA = StrictConfigSchema(
    name="submodules.assets.bundled",
    fields={
        "hmr2_mean_params": ConfigField.of(str),
        "smplx_to_smpl": ConfigField.of(str),
        "smpl_coco17_regressor": ConfigField.of(str),
        "smplx_verts437": ConfigField.of(str),
        "smpl_neutral_joint_regressor": ConfigField.of(str),
    },
)
_ASSET_SCHEMA = StrictConfigSchema(
    name="submodules.assets",
    fields={
        "checkpoints": ConfigField.mapping(_CHECKPOINT_SCHEMA),
        "dino_repository": ConfigField.of(str),
        "body_models_dir": ConfigField.of(str),
        "smpl_faces": ConfigField.of(str),
        "bundled": ConfigField.mapping(BUNDLED_MODEL_ASSET_SCHEMA),
    },
)
_PATH_ROOT_SCHEMA = StrictConfigSchema(
    name="submodules.paths",
    fields={
        "project_root": ConfigField.of(str),
        "data_root": ConfigField.of(str),
        "checkpoint_root": ConfigField.of(str),
        "artifact_root": ConfigField.of(str),
        "output_root": ConfigField.of(str),
        "cache_root": ConfigField.of(str),
        "external_asset_root": ConfigField.of(str),
    },
)


def _validate_tracking_runtime(value: Mapping[str, object]) -> None:
    confidence = cast(float, value["yolo_confidence"])
    bbox_enlarge = cast(float, value["bbox_enlarge"])
    if not 0.0 < confidence <= 1.0:
        raise SemanticConfigurationError(
            "submodules.runtime.tracking.yolo_confidence must be in (0, 1]."
        )
    if bbox_enlarge <= 0.0:
        raise SemanticConfigurationError(
            "submodules.runtime.tracking.bbox_enlarge must be positive."
        )


def _validate_dino_runtime(value: Mapping[str, object]) -> None:
    confidence = cast(float, value["confidence"])
    short_side = cast(int, value["short_side"])
    max_long_side = cast(int, value["max_long_side"])
    if not 0.0 < confidence < 1.0:
        raise SemanticConfigurationError(
            "submodules.runtime.dino_detector.confidence must be in (0, 1)."
        )
    if short_side <= 0 or max_long_side < short_side:
        raise SemanticConfigurationError(
            "submodules.runtime.dino_detector requires 0 < short_side <= max_long_side."
        )


def _validate_vitpose_runtime(value: Mapping[str, object]) -> None:
    if cast(int, value["batch_size"]) <= 0:
        raise SemanticConfigurationError(
            "submodules.runtime.vitpose.batch_size must be positive."
        )


def _validate_vitpose_head(value: Mapping[str, object]) -> None:
    ViTPoseHeadConfig(
        in_channels=cast(int, value["in_channels"]),
        out_channels=cast(int, value["out_channels"]),
        num_deconv_layers=cast(int, value["num_deconv_layers"]),
        num_deconv_filters=cast(tuple[int, ...], value["num_deconv_filters"]),
        num_deconv_kernels=cast(tuple[int, ...], value["num_deconv_kernels"]),
        final_conv_kernel=cast(int, value["final_conv_kernel"]),
        num_conv_layers=cast(int, value["num_conv_layers"]),
        num_conv_kernels=cast(tuple[int, ...], value["num_conv_kernels"]),
    )


def _validate_hmr2_runtime(value: Mapping[str, object]) -> None:
    if cast(int, value["batch_size"]) <= 0:
        raise SemanticConfigurationError(
            "submodules.runtime.hmr2.batch_size must be positive."
        )


_TRACKING_RUNTIME_SCHEMA = StrictConfigSchema(
    name="submodules.runtime.tracking",
    fields={
        "yolo_confidence": ConfigField.of(float),
        "bbox_enlarge": ConfigField.of(float),
    },
    semantic_checks=(_validate_tracking_runtime,),
)
_DINO_RUNTIME_SCHEMA = StrictConfigSchema(
    name="submodules.runtime.dino_detector",
    fields={
        "confidence": ConfigField.of(float),
        "short_side": ConfigField.of(int),
        "max_long_side": ConfigField.of(int),
    },
    semantic_checks=(_validate_dino_runtime,),
)
_VITPOSE_HEAD_SCHEMA = StrictConfigSchema(
    name="submodules.runtime.vitpose.head",
    fields={
        "in_channels": ConfigField.of(int),
        "out_channels": ConfigField.of(int),
        "num_deconv_layers": ConfigField.of(int),
        "num_deconv_filters": ConfigField.sequence(ConfigField.of(int)),
        "num_deconv_kernels": ConfigField.sequence(ConfigField.of(int)),
        "final_conv_kernel": ConfigField.of(int),
        "num_conv_layers": ConfigField.of(int),
        "num_conv_kernels": ConfigField.sequence(ConfigField.of(int)),
    },
    semantic_checks=(_validate_vitpose_head,),
)
_VITPOSE_RUNTIME_SCHEMA = StrictConfigSchema(
    name="submodules.runtime.vitpose",
    fields={
        "flip_test": ConfigField.of(bool),
        "batch_size": ConfigField.of(int),
        "head": ConfigField.mapping(_VITPOSE_HEAD_SCHEMA),
    },
    semantic_checks=(_validate_vitpose_runtime,),
)
_HMR2_RUNTIME_SCHEMA = StrictConfigSchema(
    name="submodules.runtime.hmr2",
    fields={"batch_size": ConfigField.of(int)},
    semantic_checks=(_validate_hmr2_runtime,),
)
SUBMODULE_RUNTIME_SCHEMA = StrictConfigSchema(
    name="submodules.runtime",
    fields={
        "device": ConfigField.of(str),
        "allow_device_fallback": ConfigField.of(bool),
        "tracking": ConfigField.mapping(_TRACKING_RUNTIME_SCHEMA),
        "dino_detector": ConfigField.mapping(_DINO_RUNTIME_SCHEMA),
        "vitpose": ConfigField.mapping(_VITPOSE_RUNTIME_SCHEMA),
        "hmr2": ConfigField.mapping(_HMR2_RUNTIME_SCHEMA),
        "static_cam": ConfigField.of(bool),
    },
)
_DEMO_SCHEMA = StrictConfigSchema(
    name="submodules.demo_gvhmr",
    fields={
        "paths": ConfigField.mapping(_PATH_ROOT_SCHEMA),
        "assets": ConfigField.mapping(_ASSET_SCHEMA),
        "video": ConfigField.of(str),
        "output_directory": ConfigField.of(str),
        "output_name": ConfigField.of(str),
        "num_tracks": ConfigField.of(int),
        "interactive_tracks": ConfigField.of(bool),
        "max_frames": ConfigField.of(int, type(None)),
        "runtime": ConfigField.mapping(SUBMODULE_RUNTIME_SCHEMA),
        "mesh_alpha": ConfigField.of(float),
        "video_crf": ConfigField.of(int),
    },
)


def _nested_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"Strict schema did not return a mapping for {name}.")
    return value


@dataclass(frozen=True, slots=True)
class TrackingRuntimeConfig:
    """Validated person-tracking settings selected by composition."""

    yolo_confidence: float
    bbox_enlarge: float


@dataclass(frozen=True, slots=True)
class DinoDetectorRuntimeConfig:
    """Validated DINO detector threshold and resize settings."""

    confidence: float
    short_side: int
    max_long_side: int


@dataclass(frozen=True, slots=True)
class ViTPoseRuntimeConfig:
    """Validated ViTPose inference settings."""

    flip_test: bool
    batch_size: int
    head: ViTPoseHeadConfig


@dataclass(frozen=True, slots=True)
class Hmr2RuntimeConfig:
    """Validated HMR2 feature-extraction settings."""

    batch_size: int


@dataclass(frozen=True, slots=True)
class SubmoduleRuntimeConfig:
    """Single typed runtime contract shared by submodule entrypoints."""

    device: str
    allow_device_fallback: bool
    tracking: TrackingRuntimeConfig
    dino_detector: DinoDetectorRuntimeConfig
    vitpose: ViTPoseRuntimeConfig
    hmr2: Hmr2RuntimeConfig
    static_cam: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SubmoduleRuntimeConfig:
        """Validate exact runtime keys, types, and inference semantics."""
        validated = SUBMODULE_RUNTIME_SCHEMA.validate(value)
        tracking = _nested_mapping(validated["tracking"], name="runtime.tracking")
        dino = _nested_mapping(validated["dino_detector"], name="runtime.dino_detector")
        vitpose = _nested_mapping(validated["vitpose"], name="runtime.vitpose")
        vitpose_head = _nested_mapping(vitpose["head"], name="runtime.vitpose.head")
        hmr2 = _nested_mapping(validated["hmr2"], name="runtime.hmr2")
        device = cast(str, validated["device"])
        if not device.strip():
            raise SemanticConfigurationError(
                "submodules.runtime.device must be a non-empty device specification."
            )
        if device != "auto":
            import torch

            try:
                torch.device(device)
            except (RuntimeError, ValueError) as error:
                raise SemanticConfigurationError(
                    f"submodules.runtime.device is invalid: {device!r}."
                ) from error
        return cls(
            device=device,
            allow_device_fallback=cast(bool, validated["allow_device_fallback"]),
            tracking=TrackingRuntimeConfig(
                yolo_confidence=cast(float, tracking["yolo_confidence"]),
                bbox_enlarge=cast(float, tracking["bbox_enlarge"]),
            ),
            dino_detector=DinoDetectorRuntimeConfig(
                confidence=cast(float, dino["confidence"]),
                short_side=cast(int, dino["short_side"]),
                max_long_side=cast(int, dino["max_long_side"]),
            ),
            vitpose=ViTPoseRuntimeConfig(
                flip_test=cast(bool, vitpose["flip_test"]),
                batch_size=cast(int, vitpose["batch_size"]),
                head=ViTPoseHeadConfig(
                    in_channels=cast(int, vitpose_head["in_channels"]),
                    out_channels=cast(int, vitpose_head["out_channels"]),
                    num_deconv_layers=cast(int, vitpose_head["num_deconv_layers"]),
                    num_deconv_filters=cast(
                        tuple[int, ...], vitpose_head["num_deconv_filters"]
                    ),
                    num_deconv_kernels=cast(
                        tuple[int, ...], vitpose_head["num_deconv_kernels"]
                    ),
                    final_conv_kernel=cast(int, vitpose_head["final_conv_kernel"]),
                    num_conv_layers=cast(int, vitpose_head["num_conv_layers"]),
                    num_conv_kernels=cast(
                        tuple[int, ...], vitpose_head["num_conv_kernels"]
                    ),
                ),
            ),
            hmr2=Hmr2RuntimeConfig(batch_size=cast(int, hmr2["batch_size"])),
            static_cam=cast(bool, validated["static_cam"]),
        )


def require_absolute_path(value: str | Path, *, name: str) -> Path:
    """Reject paths which did not pass through the shared runtime resolver."""
    path = Path(value)
    if not path.is_absolute():
        raise SemanticConfigurationError(
            f"{name} must be an absolute path resolved by PathResolver; got {value!r}."
        )
    return path.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class BundledModelAssetPaths:
    """Repository-owned GVHMR assets resolved through the PROJECT role."""

    hmr2_mean_params: Path
    smplx_to_smpl: Path
    smpl_coco17_regressor: Path
    smplx_verts437: Path
    smpl_neutral_joint_regressor: Path

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        resolver: PathResolver,
    ) -> BundledModelAssetPaths:
        """Reject unknown bundled assets and resolve every file as PROJECT-owned."""
        validated = BUNDLED_MODEL_ASSET_SCHEMA.validate(value)
        return cls(
            hmr2_mean_params=resolver.resolve(
                PathRole.PROJECT, cast(str, validated["hmr2_mean_params"])
            ),
            smplx_to_smpl=resolver.resolve(
                PathRole.PROJECT, cast(str, validated["smplx_to_smpl"])
            ),
            smpl_coco17_regressor=resolver.resolve(
                PathRole.PROJECT, cast(str, validated["smpl_coco17_regressor"])
            ),
            smplx_verts437=resolver.resolve(
                PathRole.PROJECT, cast(str, validated["smplx_verts437"])
            ),
            smpl_neutral_joint_regressor=resolver.resolve(
                PathRole.PROJECT,
                cast(str, validated["smpl_neutral_joint_regressor"]),
            ),
        )

    def require_files(self) -> None:
        """Fail before model construction when a bundled source asset is absent."""
        for name in (
            "hmr2_mean_params",
            "smplx_to_smpl",
            "smpl_coco17_regressor",
            "smplx_verts437",
            "smpl_neutral_joint_regressor",
        ):
            path = getattr(self, name)
            if not path.is_file():
                raise FileNotFoundError(f"Bundled GVHMR asset is missing: {path}")


@dataclass(frozen=True, slots=True)
class ExternalModelAssetPaths:
    """Absolute external-model paths derived from the shared root roles."""

    yolo_checkpoint: Path
    dino_checkpoint: Path
    vitpose_checkpoint: Path
    hmr2_checkpoint: Path
    gvhmr_checkpoint: Path
    dino_repository: Path
    body_models_dir: Path
    smpl_faces: Path
    bundled: BundledModelAssetPaths

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        resolver: PathResolver,
    ) -> ExternalModelAssetPaths:
        """Validate exact asset keys and resolve every path under its role root."""
        validated = _ASSET_SCHEMA.validate(value)
        checkpoints = validated["checkpoints"]
        bundled = validated["bundled"]
        if not isinstance(checkpoints, Mapping):
            raise AssertionError("Strict checkpoint schema did not return a mapping.")
        if not isinstance(bundled, Mapping):
            raise AssertionError(
                "Strict bundled-asset schema did not return a mapping."
            )
        return cls(
            yolo_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, checkpoints["yolo"])
            ),
            dino_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, checkpoints["dino"])
            ),
            vitpose_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, checkpoints["vitpose"])
            ),
            hmr2_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, checkpoints["hmr2"])
            ),
            gvhmr_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, checkpoints["gvhmr"])
            ),
            dino_repository=resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                cast(str, validated["dino_repository"]),
            ),
            body_models_dir=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, validated["body_models_dir"])
            ),
            smpl_faces=resolver.resolve(
                PathRole.DATA, cast(str, validated["smpl_faces"])
            ),
            bundled=BundledModelAssetPaths.from_mapping(bundled, resolver=resolver),
        )


@dataclass(frozen=True, slots=True)
class GvhmrDemoConfig:
    """Validated GVHMR demo settings with all runtime paths already absolute."""

    roots: RuntimePathRoots
    assets: ExternalModelAssetPaths
    video_path: Path
    output_path: Path
    num_tracks: int
    interactive_tracks: bool
    max_frames: int | None
    runtime: SubmoduleRuntimeConfig
    mesh_alpha: float
    video_crf: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        repository_root: Path,
    ) -> GvhmrDemoConfig:
        """Validate the complete boundary and deterministically resolve its paths."""
        validated = _DEMO_SCHEMA.validate(value)
        paths = validated["paths"]
        assets = validated["assets"]
        runtime = validated["runtime"]
        if (
            not isinstance(paths, Mapping)
            or not isinstance(assets, Mapping)
            or not isinstance(runtime, Mapping)
        ):
            raise AssertionError("Strict demo schema did not return nested mappings.")
        roots = RuntimePathRoots.from_mapping(paths, repository_root=repository_root)
        resolver = PathResolver(roots)
        video_path = resolver.resolve(PathRole.DATA, cast(str, validated["video"]))
        output_name = cast(str, validated["output_name"])
        if not output_name or Path(output_name).name != output_name:
            raise SemanticConfigurationError(
                "submodules.demo_gvhmr.output_name must be a non-empty filename, "
                "not a path."
            )
        output_path = resolver.resolve(
            PathRole.OUTPUT,
            cast(str, validated["output_directory"]),
            output_name,
        )
        num_tracks = validated["num_tracks"]
        max_frames = validated["max_frames"]
        mesh_alpha = validated["mesh_alpha"]
        video_crf = validated["video_crf"]
        if not isinstance(num_tracks, int) or isinstance(num_tracks, bool):
            raise AssertionError(
                "Strict demo schema did not return an integer num_tracks."
            )
        if not isinstance(mesh_alpha, float):
            raise AssertionError(
                "Strict demo schema did not return a float mesh_alpha."
            )
        if not isinstance(video_crf, int) or isinstance(video_crf, bool):
            raise AssertionError(
                "Strict demo schema did not return an integer video_crf."
            )
        if max_frames is not None and (
            not isinstance(max_frames, int) or isinstance(max_frames, bool)
        ):
            raise AssertionError(
                "Strict demo schema did not return an optional integer max_frames."
            )
        if num_tracks <= 0:
            raise SemanticConfigurationError("num_tracks must be positive.")
        if max_frames is not None and max_frames <= 0:
            raise SemanticConfigurationError(
                "max_frames must be null or a positive integer."
            )
        if not 0.0 < mesh_alpha <= 1.0:
            raise SemanticConfigurationError("mesh_alpha must be in (0, 1].")
        if not 0 <= video_crf <= 51:
            raise SemanticConfigurationError("video_crf must be in [0, 51].")
        return cls(
            roots=roots,
            assets=ExternalModelAssetPaths.from_mapping(assets, resolver=resolver),
            video_path=video_path,
            output_path=output_path,
            num_tracks=num_tracks,
            interactive_tracks=cast(bool, validated["interactive_tracks"]),
            max_frames=max_frames,
            runtime=SubmoduleRuntimeConfig.from_mapping(runtime),
            mesh_alpha=mesh_alpha,
            video_crf=video_crf,
        )


def inspect_gvhmr_demo_schema() -> tuple[ConfigFieldContract, ...]:
    """Expose required/default/precedence policy for the composed demo schema."""
    contracts: tuple[ConfigFieldContract, ...] = _DEMO_SCHEMA.inspect()
    return contracts

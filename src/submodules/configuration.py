"""Strict runtime configuration and asset paths for external model submodules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES, require_smpl_asset
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    SemanticConfigurationError,
    StrictConfigSchema,
)

__all__ = [
    "SUBMODULE_RUNTIME_SCHEMA",
    "SMPL_ASSET_SCHEMA",
    "SmplAssetPaths",
    "DinoDetectorRuntimeConfig",
    "Hmr2RuntimeConfig",
    "InferencePrecision",
    "SubmoduleRuntimeConfig",
    "TrackingRuntimeConfig",
    "ViTPoseRuntimeConfig",
    "ViTPoseHeadConfig",
    "require_absolute_path",
]

InferencePrecision = Literal["fp32", "bf16", "fp16"]


@dataclass(frozen=True, slots=True)
class BackendConfig:
    precision: InferencePrecision = "fp32"
    engines: tuple[Path, ...] = ()


_BACKEND_FIELDS = {
    "backend": ConfigField.of(str, required=False),
    "engines": ConfigField.sequence(ConfigField.of(str), required=False),
}


def _backend_config(
    value: Mapping[str, object],
    *,
    model: str,
    device: str,
    resolver: PathResolver | None,
) -> BackendConfig:
    backend = value.get("backend", "pytorch")
    precision = cast(InferencePrecision, value.get("precision", "fp32"))
    roots = cast(tuple[str, ...], value.get("engines", ()))
    if backend not in {"pytorch", "tensorrt"}:
        raise SemanticConfigurationError(f"{model} backend must be pytorch or tensorrt")
    if backend == "pytorch":
        if roots:
            raise SemanticConfigurationError(f"{model} engines require the TensorRT backend")
        if model not in {"vitpose", "dino_detector"} and precision != "fp32":
            raise SemanticConfigurationError(f"{model} PyTorch inference requires fp32")
        return BackendConfig(precision)
    if precision != "fp16" or not roots or (model != "dino_detector" and len(roots) != 1):
        raise SemanticConfigurationError(
            f"{model} TensorRT requires fp16 and explicit engine directories "
            "(one per input shape for DINO, exactly one for other models)"
        )
    if device != "auto" and not device.startswith("cuda"):
        raise SemanticConfigurationError(f"{model} TensorRT requires CUDA")
    if model == "gvhmr" and value.get("batch_size", 1) != 1:
        raise SemanticConfigurationError("GVHMR TensorRT requires batch_size=1")
    engines = tuple(
        require_absolute_path(path, name=f"{model} engine")
        if resolver is None
        else resolver.resolve(PathRole.ARTIFACT, path)
        for path in roots
    )
    if len(set(engines)) != len(engines):
        raise SemanticConfigurationError(f"{model} TensorRT engines must be distinct")
    return BackendConfig(precision, engines)


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


SMPL_ASSET_SCHEMA = StrictConfigSchema(
    name="submodules.assets.smpl",
    fields={
        "hmr2_mean_params": ConfigField.of(str),
        "smplx_to_smpl": ConfigField.of(str),
        "smpl_coco17_regressor": ConfigField.of(str),
        "smplx_verts437": ConfigField.of(str),
        "smpl_neutral_joint_regressor": ConfigField.of(str),
    },
)


def _validate_tracking_runtime(value: Mapping[str, object]) -> None:
    bbox_enlarge = cast(float, value["bbox_enlarge"])
    if bbox_enlarge <= 0.0:
        raise SemanticConfigurationError(
            "submodules.runtime.tracking.bbox_enlarge must be positive."
        )


def _validate_precision(value: Mapping[str, object], *, model: str) -> None:
    if value["precision"] not in get_args(InferencePrecision):
        raise SemanticConfigurationError(
            f"submodules.runtime.{model}.precision must be fp32, bf16, or fp16."
        )


def _validate_dino_runtime(value: Mapping[str, object]) -> None:
    _validate_precision(value, model="dino_detector")
    _validate_batch_size(value, model="dino_detector")
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
    _validate_precision(value, model="vitpose")
    _validate_batch_size(value, model="vitpose")


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


def _validate_batch_size(value: Mapping[str, object], *, model: str) -> None:
    if cast(int, value.get("batch_size", 1)) <= 0:
        raise SemanticConfigurationError(
            f"submodules.runtime.{model}.batch_size must be positive."
        )


_TRACKING_RUNTIME_SCHEMA = StrictConfigSchema(
    name="submodules.runtime.tracking",
    fields={
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
        "precision": ConfigField.of(str),
        "batch_size": ConfigField.of(int, required=False),
        **_BACKEND_FIELDS,
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
        "precision": ConfigField.of(str),
        **_BACKEND_FIELDS,
    },
    semantic_checks=(_validate_vitpose_runtime,),
)
def _batched_runtime_schema(model: str) -> StrictConfigSchema:
    return StrictConfigSchema(
        name=f"submodules.runtime.{model}",
        fields={
            "batch_size": ConfigField.of(int),
            "precision": ConfigField.of(str, required=False),
            **_BACKEND_FIELDS,
        },
        semantic_checks=(lambda value: _validate_batch_size(value, model=model),),
    )


SUBMODULE_RUNTIME_SCHEMA = StrictConfigSchema(
    name="submodules.runtime",
    fields={
        "device": ConfigField.of(str),
        "tracking": ConfigField.mapping(_TRACKING_RUNTIME_SCHEMA),
        "dino_detector": ConfigField.mapping(_DINO_RUNTIME_SCHEMA),
        "vitpose": ConfigField.mapping(_VITPOSE_RUNTIME_SCHEMA),
        "hmr2": ConfigField.mapping(_batched_runtime_schema("hmr2")),
        "clip_reid": ConfigField.mapping(_batched_runtime_schema("clip_reid"), required=False),
        "gvhmr": ConfigField.mapping(_batched_runtime_schema("gvhmr"), required=False),
        "static_cam": ConfigField.of(bool),
    },
)


def _nested_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"Strict schema did not return a mapping for {name}.")
    return value


@dataclass(frozen=True, slots=True)
class TrackingRuntimeConfig:
    """Validated person-tracking settings selected by composition."""

    bbox_enlarge: float


@dataclass(frozen=True, slots=True)
class DinoDetectorRuntimeConfig:
    """Validated DINO detector threshold, resize, and precision settings."""

    confidence: float
    short_side: int
    max_long_side: int
    precision: InferencePrecision
    batch_size: int = 1
    engines: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ViTPoseRuntimeConfig:
    """Validated ViTPose inference settings."""

    flip_test: bool
    batch_size: int
    head: ViTPoseHeadConfig
    precision: InferencePrecision
    engines: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchedRuntimeConfig:
    """Validated batch size and backend for an inference model."""

    batch_size: int
    backend: BackendConfig = BackendConfig()


Hmr2RuntimeConfig = BatchedRuntimeConfig


@dataclass(frozen=True, slots=True)
class SubmoduleRuntimeConfig:
    """Single typed runtime contract shared by submodule entrypoints."""

    device: str
    tracking: TrackingRuntimeConfig
    dino_detector: DinoDetectorRuntimeConfig
    vitpose: ViTPoseRuntimeConfig
    hmr2: Hmr2RuntimeConfig
    static_cam: bool
    clip_reid: BatchedRuntimeConfig = BatchedRuntimeConfig(batch_size=64)
    gvhmr: BatchedRuntimeConfig = BatchedRuntimeConfig(batch_size=1)

    @property
    def uses_tensorrt(self) -> bool:
        return bool(
            self.dino_detector.engines or self.vitpose.engines or self.hmr2.backend.engines
            or self.clip_reid.backend.engines or self.gvhmr.backend.engines
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, resolver: PathResolver | None = None
    ) -> SubmoduleRuntimeConfig:
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
                device_type = torch.device(device).type
            except (RuntimeError, ValueError) as error:
                raise SemanticConfigurationError(
                    f"submodules.runtime.device is invalid: {device!r}."
                ) from error
            if device_type != "cuda":
                if dino["precision"] != "fp32":
                    raise SemanticConfigurationError("Mixed precision DINO requires a CUDA device.")
                if vitpose["precision"] != "fp32" and (device_type, vitpose["precision"]) != ("cpu", "bf16"):
                    raise SemanticConfigurationError("Mixed precision ViTPose requires CUDA or bf16 on CPU.")
        dino_backend = _backend_config(dino, model="dino_detector", device=device, resolver=resolver)
        pose_backend = _backend_config(vitpose, model="vitpose", device=device, resolver=resolver)

        def simple(section: str, default: int) -> BatchedRuntimeConfig:
            options = _nested_mapping(validated.get(section, {}), name=f"runtime.{section}")
            return BatchedRuntimeConfig(
                batch_size=cast(int, options.get("batch_size", default)),
                backend=_backend_config(options, model=section, device=device, resolver=resolver),
            )

        return cls(
            device=device,
            tracking=TrackingRuntimeConfig(
                bbox_enlarge=cast(float, tracking["bbox_enlarge"]),
            ),
            dino_detector=DinoDetectorRuntimeConfig(
                confidence=cast(float, dino["confidence"]),
                short_side=cast(int, dino["short_side"]),
                max_long_side=cast(int, dino["max_long_side"]),
                precision=dino_backend.precision,
                engines=dino_backend.engines,
                batch_size=cast(int, dino.get("batch_size", 1)),
            ),
            vitpose=ViTPoseRuntimeConfig(
                flip_test=cast(bool, vitpose["flip_test"]),
                batch_size=cast(int, vitpose["batch_size"]),
                precision=pose_backend.precision,
                engines=pose_backend.engines,
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
            hmr2=Hmr2RuntimeConfig(
                batch_size=cast(int, hmr2["batch_size"]),
                backend=_backend_config(hmr2, model="hmr2", device=device, resolver=resolver),
            ),
            static_cam=cast(bool, validated["static_cam"]),
            clip_reid=simple("clip_reid", 64),
            gvhmr=simple("gvhmr", 1),
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
class SmplAssetPaths:
    """User-acquired SMPL auxiliary files resolved through the CHECKPOINT role."""

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
    ) -> SmplAssetPaths:
        """Reject unknown assets and resolve every file under checkpoint storage."""
        validated = SMPL_ASSET_SCHEMA.validate(value)
        return cls(
            hmr2_mean_params=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, validated["hmr2_mean_params"])
            ),
            smplx_to_smpl=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, validated["smplx_to_smpl"])
            ),
            smpl_coco17_regressor=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, validated["smpl_coco17_regressor"])
            ),
            smplx_verts437=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, validated["smplx_verts437"])
            ),
            smpl_neutral_joint_regressor=resolver.resolve(
                PathRole.CHECKPOINT,
                cast(str, validated["smpl_neutral_joint_regressor"]),
            ),
        )

    def require_files(self) -> None:
        """Fail with acquisition instructions before constructing any body model."""
        for name in SMPL_ASSET_SOURCES:
            require_smpl_asset(getattr(self, name), name)

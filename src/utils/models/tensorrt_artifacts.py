"""File, profile and environment contracts for explicitly built TensorRT plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    PositiveInt,
    StringConstraints,
    model_validator,
)

from src.utils.checksum import dual_sha256

EXPORT_MANIFEST = "export.json"
ENGINE_MANIFEST = "engine.json"
ONNX_FILE = "model.onnx"
PLAN_FILE = "engine.plan"

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
TensorDtype = Literal["float16", "float32", "int32", "int64", "bool"]
Shape = tuple[int, ...]
AssetDigest = Callable[[Path], str]
_Model = TypeVar("_Model")


class _AfterValidator(Protocol):
    def __call__(self, function: Callable[[_Model], _Model]) -> Callable[[_Model], _Model]: ...


_after_validator = cast(_AfterValidator, model_validator(mode="after"))


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TensorSpec(_Manifest):
    dtype: TensorDtype
    shape: Shape

    @_after_validator
    def positive_or_dynamic(self) -> TensorSpec:
        if not self.shape or any(dim < -1 or dim == 0 for dim in self.shape):
            raise ValueError(f"Invalid tensor dimensions: {self.shape}")
        return self


class InputProfile(_Manifest):
    minimum: tuple[PositiveInt, ...]
    optimum: tuple[PositiveInt, ...]
    maximum: tuple[PositiveInt, ...]

    @property
    def shapes(self) -> tuple[Shape, Shape, Shape]:
        return self.minimum, self.optimum, self.maximum

    @_after_validator
    def ordered(self) -> InputProfile:
        if not self.minimum or any(
            not lo <= opt <= hi for lo, opt, hi in zip(*self.shapes, strict=True)
        ):
            raise ValueError(f"Invalid min/opt/max profile: {self.shapes}")
        return self

    def require(self, shape: Shape, *, tensor: str) -> None:
        if len(shape) != len(self.minimum) or any(
            not lo <= dim <= hi
            for lo, dim, hi in zip(self.minimum, shape, self.maximum, strict=True)
        ):
            raise ValueError(
                f"TensorRT {tensor} shape {shape} outside profile "
                f"[{self.minimum}, {self.maximum}]"
            )


class ExportManifest(_Manifest):
    version: Literal[1] = 1
    model: str
    precision: Literal["fp16"] = "fp16"
    settings: dict[str, JsonValue]
    sources: dict[str, Digest]
    onnx_files: dict[str, Digest]
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]

    @_after_validator
    def check_inventory(self) -> ExportManifest:
        if ONNX_FILE not in self.onnx_files or any(
            name not in {ONNX_FILE, f"{ONNX_FILE}.data"} for name in self.onnx_files
        ):
            raise ValueError("Unexpected TensorRT ONNX file inventory")
        if (
            not self.sources
            or not self.inputs
            or not self.outputs
            or self.inputs.keys() & self.outputs.keys()
        ):
            raise ValueError("Incomplete TensorRT export sources or tensor bindings")
        return self


class EngineEnvironment(_Manifest):
    tensorrt_version: str
    device_name: str
    compute_capability: tuple[int, int]


class EngineManifest(_Manifest):
    version: Literal[1] = 1
    export_sha256: Digest
    plan_sha256: Digest
    environment: EngineEnvironment
    profiles: dict[str, InputProfile]


def require_profiles(
    profiles: Mapping[str, InputProfile], specs: Mapping[str, TensorSpec]
) -> None:
    if profiles.keys() != specs.keys():
        raise ValueError("TensorRT profile bindings differ from the ONNX inputs")
    for name, spec in specs.items():
        for shape in profiles[name].shapes:
            if len(shape) != len(spec.shape) or any(
                expected != -1 and expected != actual
                for expected, actual in zip(spec.shape, shape, strict=True)
            ):
                raise ValueError(
                    f"TensorRT profile {name}={shape} differs from ONNX shape {spec.shape}"
                )


def require_digests(
    root: Path, expected: Mapping[str, str], digest: AssetDigest
) -> None:
    for name, sha256 in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing TensorRT artifact: {path}")
        if digest(path) != sha256:
            raise ValueError(f"TensorRT SHA-256 mismatch: {path}")


@dataclass(frozen=True)
class EngineBundle:
    root: Path
    export: ExportManifest
    engine: EngineManifest

    @classmethod
    def read(cls, root: Path, *, digest: AssetDigest = dual_sha256) -> EngineBundle:
        engine = EngineManifest.model_validate_json(
            (root / ENGINE_MANIFEST).read_bytes()
        )
        require_digests(root, {EXPORT_MANIFEST: engine.export_sha256}, digest)
        export = ExportManifest.model_validate_json(
            (root / EXPORT_MANIFEST).read_bytes()
        )
        require_digests(
            root, {**export.onnx_files, PLAN_FILE: engine.plan_sha256}, digest
        )
        require_profiles(engine.profiles, export.inputs)
        return cls(root, export, engine)

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(
            self.root / name
            for name in (
                ENGINE_MANIFEST,
                EXPORT_MANIFEST,
                PLAN_FILE,
                *sorted(self.export.onnx_files),
            )
        )

    def require_model(
        self,
        *,
        model: str,
        settings: Mapping[str, JsonValue],
        sources: Mapping[str, str],
        inputs: Mapping[str, TensorSpec],
        outputs: Mapping[str, TensorSpec],
    ) -> None:
        for name, actual, expected in (
            ("model", self.export.model, model),
            ("export settings", self.export.settings, settings),
            ("source assets", self.export.sources, sources),
            ("inputs", self.export.inputs, inputs),
            ("outputs", self.export.outputs, outputs),
        ):
            if actual != expected:
                raise ValueError(f"TensorRT {name} mismatch for {self.root}")

    def require_environment(self, current: EngineEnvironment) -> None:
        if self.engine.environment != current:
            raise ValueError(
                f"TensorRT/GPU environment mismatch for {self.root}: "
                f"built for {self.engine.environment}, running on {current}"
            )

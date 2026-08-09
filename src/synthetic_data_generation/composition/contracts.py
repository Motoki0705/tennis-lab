"""Semantic contracts for composing Gaussian assets in canonical scene space.

The contracts in this module describe observable geometry, coordinate systems,
object identity, and frame membership.  They intentionally contain no artifact
digests, repository revisions, renderer revisions, or publication behavior.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self, TypeVar, cast

import numpy as np

from src.synthetic_data_generation.scene_contract import RigidTransform

GAUSSIAN_ASSET_SCHEMA = "tennis_gaussian_asset_semantic_v1"
GAUSSIAN_FOREGROUND_SCHEMA = "tennis_gaussian_foreground_composition_semantic_v1"
GAUSSIAN_SCENE_SCHEMA = "tennis_gaussian_scene_composition_semantic_v1"
GAUSSIAN_TENSOR_ENCODING = "raw_gaussian_parameters_v1"
SCENE_COORDINATE_CONVENTION = "right_handed_normalized_scene"
ASSET_COORDINATE_CONVENTION = "right_handed_asset_local_metres"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ParsedT = TypeVar("_ParsedT")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class GaussianAssetRole(StrEnum):
    """How an asset participates in a composed scene."""

    BACKGROUND = "background"
    MOVABLE = "movable"


class GaussianCoordinateFrame(StrEnum):
    """The two coordinate frames accepted by the compositor."""

    SCENE = "scene"
    ASSET_LOCAL = "asset_local"


class GaussianUnit(StrEnum):
    """Units associated with a Gaussian coordinate frame."""

    SCENE_UNIT = "scene_unit"
    METRE = "metre"


class GaussianDeformationKind(StrEnum):
    """Observable local-geometry behavior expected for one scene object."""

    RIGID = "rigid"
    ARTICULATED = "articulated"


@dataclass(frozen=True, slots=True)
class GaussianCoordinates:
    """Explicit coordinate frame, unit, and handedness convention."""

    frame: GaussianCoordinateFrame
    unit: GaussianUnit
    convention: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame, GaussianCoordinateFrame):
            raise TypeError("coordinate frame must be GaussianCoordinateFrame.")
        if not isinstance(self.unit, GaussianUnit):
            raise TypeError("coordinate unit must be GaussianUnit.")
        convention = _string(self.convention, name="coordinate convention")
        expected = {
            GaussianCoordinateFrame.SCENE: (
                GaussianUnit.SCENE_UNIT,
                SCENE_COORDINATE_CONVENTION,
            ),
            GaussianCoordinateFrame.ASSET_LOCAL: (
                GaussianUnit.METRE,
                ASSET_COORDINATE_CONVENTION,
            ),
        }
        expected_unit, expected_convention = expected[self.frame]
        if self.unit != expected_unit or convention != expected_convention:
            raise ValueError(
                f"{self.frame.value} Gaussian coordinates require unit "
                f"{expected_unit.value!r} and convention {expected_convention!r}."
            )
        object.__setattr__(self, "convention", convention)

    @classmethod
    def scene(cls) -> Self:
        """Return the one canonical scene-space coordinate contract."""
        return cls(
            frame=GaussianCoordinateFrame.SCENE,
            unit=GaussianUnit.SCENE_UNIT,
            convention=SCENE_COORDINATE_CONVENTION,
        )

    @classmethod
    def asset_local_metres(cls) -> Self:
        """Return the one canonical movable-asset coordinate contract."""
        return cls(
            frame=GaussianCoordinateFrame.ASSET_LOCAL,
            unit=GaussianUnit.METRE,
            convention=ASSET_COORDINATE_CONVENTION,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON-compatible coordinate record."""
        return {
            "frame": self.frame.value,
            "unit": self.unit.value,
            "convention": self.convention,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse the current semantic coordinate record."""
        raw = _strict_mapping(
            value,
            name="Gaussian coordinates",
            keys={"frame", "unit", "convention"},
        )
        return cls(
            frame=_enum_value(
                GaussianCoordinateFrame,
                raw["frame"],
                name="coordinate frame",
            ),
            unit=_enum_value(GaussianUnit, raw["unit"], name="coordinate unit"),
            convention=_string(raw["convention"], name="coordinate convention"),
        )


@dataclass(frozen=True, slots=True)
class GaussianTransform:
    """A positive uniform scale followed by a proper rigid transform.

    For row-vector points the represented mapping is
    ``x_scene = scale * x_asset @ R.T + t``.
    """

    scale: float
    rigid: RigidTransform

    def __post_init__(self) -> None:
        if not isinstance(self.rigid, RigidTransform):
            raise TypeError("Gaussian transform rigid must be a RigidTransform.")
        scale = _positive_float(self.scale, name="Gaussian transform scale")
        object.__setattr__(self, "scale", scale)

    @classmethod
    def identity(cls) -> Self:
        """Return an identity asset-to-scene transform."""
        return cls(scale=1.0, rigid=RigidTransform.identity())

    @property
    def rotation(self) -> tuple[float, ...]:
        """Return the proper row-major rotation matrix."""
        matrix = self.rigid.matrix()
        return tuple(float(value) for value in matrix[:3, :3].ravel())

    @property
    def translation(self) -> tuple[float, float, float]:
        """Return the finite scene-space translation."""
        matrix = self.rigid.matrix()
        return cast(
            tuple[float, float, float],
            tuple(float(value) for value in matrix[:3, 3]),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the semantic transform record."""
        return {"scale": self.scale, "rigid": self.rigid.to_list()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict positive-scale, proper-transform record."""
        raw = _strict_mapping(
            value,
            name="Gaussian transform",
            keys={"scale", "rigid"},
        )
        rigid = raw["rigid"]
        if not isinstance(rigid, Sequence) or isinstance(rigid, (str, bytes)):
            raise TypeError("Gaussian transform rigid must be an array.")
        matrix_values = _finite_tuple(rigid, size=16, name="Gaussian transform rigid")
        matrix = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4)
        return cls(
            scale=_positive_float(raw["scale"], name="Gaussian transform scale"),
            rigid=RigidTransform.from_matrix(matrix),
        )


@dataclass(frozen=True, slots=True)
class GaussianAsset:
    """Typed semantic metadata for one validated Gaussian tensor set."""

    asset_id: str
    asset_class: str
    role: GaussianAssetRole
    coordinates: GaussianCoordinates
    gaussian_count: int
    feature_dim: int
    floating_dtype: Literal["float32", "float64"]
    appearance_model: str
    appearance_space: str
    tensor_encoding: str = GAUSSIAN_TENSOR_ENCODING
    schema: str = GAUSSIAN_ASSET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GAUSSIAN_ASSET_SCHEMA:
            raise ValueError(
                f"Unsupported Gaussian asset schema {self.schema!r}; "
                f"expected {GAUSSIAN_ASSET_SCHEMA!r}."
            )
        _validate_id(self.asset_id, name="asset_id")
        _validate_id(self.asset_class, name="asset_class")
        if not isinstance(self.role, GaussianAssetRole):
            raise TypeError("role must be GaussianAssetRole.")
        if not isinstance(self.coordinates, GaussianCoordinates):
            raise TypeError("coordinates must be GaussianCoordinates.")
        if self.tensor_encoding != GAUSSIAN_TENSOR_ENCODING:
            raise ValueError(f"Unsupported tensor encoding: {self.tensor_encoding!r}.")
        _positive_integer(self.gaussian_count, name="gaussian_count")
        _positive_integer(self.feature_dim, name="feature_dim")
        if self.floating_dtype not in {"float32", "float64"}:
            raise ValueError("floating_dtype must be 'float32' or 'float64'.")
        appearance_model = _string(self.appearance_model, name="appearance_model")
        appearance_space = _string(self.appearance_space, name="appearance_space")
        expected_coordinates = (
            GaussianCoordinates.scene()
            if self.role == GaussianAssetRole.BACKGROUND
            else GaussianCoordinates.asset_local_metres()
        )
        if self.coordinates != expected_coordinates:
            raise ValueError(
                f"{self.role.value} assets must use {expected_coordinates.frame.value}/"
                f"{expected_coordinates.unit.value} coordinates."
            )
        object.__setattr__(self, "appearance_model", appearance_model)
        object.__setattr__(self, "appearance_space", appearance_space)

    def to_dict(self) -> dict[str, object]:
        """Return the current semantic asset record."""
        return {
            "schema": self.schema,
            "asset_id": self.asset_id,
            "asset_class": self.asset_class,
            "role": self.role.value,
            "coordinates": self.coordinates.to_dict(),
            "gaussian_count": self.gaussian_count,
            "feature_dim": self.feature_dim,
            "floating_dtype": self.floating_dtype,
            "tensor_encoding": self.tensor_encoding,
            "appearance_model": self.appearance_model,
            "appearance_space": self.appearance_space,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse only the current semantic asset schema."""
        raw = _strict_mapping(
            value,
            name="Gaussian asset",
            keys={
                "schema",
                "asset_id",
                "asset_class",
                "role",
                "coordinates",
                "gaussian_count",
                "feature_dim",
                "floating_dtype",
                "tensor_encoding",
                "appearance_model",
                "appearance_space",
            },
        )
        dtype = _string(raw["floating_dtype"], name="floating_dtype")
        if dtype not in {"float32", "float64"}:
            raise ValueError("floating_dtype must be 'float32' or 'float64'.")
        return cls(
            schema=_string(raw["schema"], name="schema"),
            asset_id=_string(raw["asset_id"], name="asset_id"),
            asset_class=_string(raw["asset_class"], name="asset_class"),
            role=_enum_value(GaussianAssetRole, raw["role"], name="asset role"),
            coordinates=GaussianCoordinates.from_dict(raw["coordinates"]),
            gaussian_count=_integer(raw["gaussian_count"], name="gaussian_count"),
            feature_dim=_integer(raw["feature_dim"], name="feature_dim"),
            floating_dtype=cast(Literal["float32", "float64"], dtype),
            tensor_encoding=_string(raw["tensor_encoding"], name="tensor_encoding"),
            appearance_model=_string(raw["appearance_model"], name="appearance_model"),
            appearance_space=_string(raw["appearance_space"], name="appearance_space"),
        )


@dataclass(frozen=True, slots=True)
class GaussianSceneObject:
    """Stable object identity and deformation contract across all frames."""

    object_id: str
    instance_id: int
    asset_id: str
    deformation_kind: GaussianDeformationKind

    def __post_init__(self) -> None:
        _validate_id(self.object_id, name="object_id")
        _positive_integer(self.instance_id, name="instance_id")
        _validate_id(self.asset_id, name="asset_id")
        if not isinstance(self.deformation_kind, GaussianDeformationKind):
            raise TypeError("deformation_kind must be GaussianDeformationKind.")

    def to_dict(self) -> dict[str, object]:
        """Return the stable scene-object record."""
        return {
            "object_id": self.object_id,
            "instance_id": self.instance_id,
            "asset_id": self.asset_id,
            "deformation_kind": self.deformation_kind.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict stable scene-object record."""
        raw = _strict_mapping(
            value,
            name="Gaussian scene object",
            keys={"object_id", "instance_id", "asset_id", "deformation_kind"},
        )
        return cls(
            object_id=_string(raw["object_id"], name="object_id"),
            instance_id=_integer(raw["instance_id"], name="instance_id"),
            asset_id=_string(raw["asset_id"], name="asset_id"),
            deformation_kind=_enum_value(
                GaussianDeformationKind,
                raw["deformation_kind"],
                name="deformation_kind",
            ),
        )


@dataclass(frozen=True, slots=True)
class GaussianInstance:
    """One object's source-frame mapping and transform in a global frame."""

    object_id: str
    source_frame_index: int
    scene_from_asset: GaussianTransform

    def __post_init__(self) -> None:
        _validate_id(self.object_id, name="object_id")
        _nonnegative_integer(self.source_frame_index, name="source_frame_index")
        if not isinstance(self.scene_from_asset, GaussianTransform):
            raise TypeError("scene_from_asset must be GaussianTransform.")

    def to_dict(self) -> dict[str, object]:
        """Return the per-frame instance record."""
        return {
            "object_id": self.object_id,
            "source_frame_index": self.source_frame_index,
            "scene_from_asset": self.scene_from_asset.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict per-frame instance record."""
        raw = _strict_mapping(
            value,
            name="Gaussian instance",
            keys={"object_id", "source_frame_index", "scene_from_asset"},
        )
        return cls(
            object_id=_string(raw["object_id"], name="object_id"),
            source_frame_index=_integer(
                raw["source_frame_index"],
                name="source_frame_index",
            ),
            scene_from_asset=GaussianTransform.from_dict(raw["scene_from_asset"]),
        )


@dataclass(frozen=True, slots=True)
class GaussianFrame:
    """All movable object placements for one global frame."""

    frame_index: int
    instances: tuple[GaussianInstance, ...]

    def __post_init__(self) -> None:
        _nonnegative_integer(self.frame_index, name="frame_index")
        instances = tuple(self.instances)
        if any(not isinstance(instance, GaussianInstance) for instance in instances):
            raise TypeError("frame instances must be GaussianInstance values.")
        _require_unique(
            [instance.object_id for instance in instances],
            name=f"frame {self.frame_index} object ids",
        )
        object.__setattr__(self, "instances", instances)

    def to_dict(self) -> dict[str, object]:
        """Return the global-frame record."""
        return {
            "frame_index": self.frame_index,
            "instances": [instance.to_dict() for instance in self.instances],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict global-frame record."""
        raw = _strict_mapping(
            value,
            name="Gaussian frame",
            keys={"frame_index", "instances"},
        )
        return cls(
            frame_index=_integer(raw["frame_index"], name="frame_index"),
            instances=_sequence_of(
                raw["instances"],
                GaussianInstance.from_dict,
                name="instances",
            ),
        )


@dataclass(frozen=True, slots=True)
class GaussianSceneComposition:
    """Complete semantic Gaussian composition over one global timeline."""

    scene_id: str
    composition_id: str
    background: GaussianAsset
    assets: tuple[GaussianAsset, ...]
    objects: tuple[GaussianSceneObject, ...]
    frames: tuple[GaussianFrame, ...]
    schema: str = GAUSSIAN_SCENE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GAUSSIAN_SCENE_SCHEMA:
            raise ValueError(
                f"Unsupported Gaussian scene schema {self.schema!r}; "
                f"expected {GAUSSIAN_SCENE_SCHEMA!r}."
        )
        _validate_id(self.scene_id, name="scene_id")
        _validate_id(self.composition_id, name="composition_id")
        if not isinstance(self.background, GaussianAsset):
            raise TypeError("Scene background must be a GaussianAsset.")
        if self.background.role != GaussianAssetRole.BACKGROUND:
            raise ValueError("Scene background must be a background Gaussian asset.")

        assets = tuple(self.assets)
        objects = tuple(self.objects)
        frames = tuple(self.frames)
        if any(not isinstance(asset, GaussianAsset) for asset in assets):
            raise TypeError("Composition assets must be GaussianAsset values.")
        if any(not isinstance(item, GaussianSceneObject) for item in objects):
            raise TypeError("Composition objects must be GaussianSceneObject values.")
        if any(not isinstance(frame, GaussianFrame) for frame in frames):
            raise TypeError("Composition frames must be GaussianFrame values.")
        if any(asset.role != GaussianAssetRole.MOVABLE for asset in assets):
            raise ValueError("Composition assets must all be movable assets.")
        asset_ids = [self.background.asset_id, *(asset.asset_id for asset in assets)]
        _require_unique(asset_ids, name="Gaussian asset ids")
        _require_unique([item.object_id for item in objects], name="object ids")
        _require_unique([item.instance_id for item in objects], name="instance ids")

        movable_by_id = {asset.asset_id: asset for asset in assets}
        unknown_assets = sorted(
            {item.asset_id for item in objects}.difference(movable_by_id)
        )
        if unknown_assets:
            raise ValueError(f"Objects reference unknown movable assets: {unknown_assets}.")
        for asset in assets:
            _validate_render_compatibility(self.background, asset)

        if not frames:
            raise ValueError("A Gaussian scene composition requires at least one frame.")
        frame_indices = tuple(frame.frame_index for frame in frames)
        expected_indices = tuple(range(len(frames)))
        if frame_indices != expected_indices:
            raise ValueError(
                "Gaussian frame indices must exactly equal 0..T-1 in order; "
                f"got {frame_indices}."
            )

        objects_by_id = {item.object_id: item for item in objects}
        placements: dict[str, list[tuple[int, GaussianInstance]]] = {
            object_id: [] for object_id in objects_by_id
        }
        for frame in frames:
            for instance in frame.instances:
                if instance.object_id not in objects_by_id:
                    raise ValueError(
                        f"Frame {frame.frame_index} references unknown object "
                        f"{instance.object_id!r}."
                    )
                placements[instance.object_id].append((frame.frame_index, instance))

        unused = sorted(object_id for object_id, values in placements.items() if not values)
        if unused:
            raise ValueError(f"Declared Gaussian objects never appear: {unused}.")
        for object_id, values in placements.items():
            source_indices = [instance.source_frame_index for _, instance in values]
            expected_sources = list(
                range(source_indices[0], source_indices[0] + len(source_indices))
            )
            if source_indices != expected_sources:
                raise ValueError(
                    f"Object {object_id!r} source frames must be consecutive in global "
                    f"appearance order; got {source_indices}."
                )
            scene_object = objects_by_id[object_id]
            if (
                scene_object.deformation_kind == GaussianDeformationKind.ARTICULATED
                and len(values) < 2
            ):
                raise ValueError(
                    f"Articulated object {object_id!r} requires at least two frames."
                )

        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "frames", frames)

    def asset(self, asset_id: str) -> GaussianAsset:
        """Return one movable asset or fail instead of selecting a fallback."""
        matches = [asset for asset in self.assets if asset.asset_id == asset_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown movable Gaussian asset: {asset_id!r}.")
        return matches[0]

    def scene_object(self, object_id: str) -> GaussianSceneObject:
        """Return one stable object record or fail closed."""
        matches = [item for item in self.objects if item.object_id == object_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown Gaussian object: {object_id!r}.")
        return matches[0]

    def frame(self, frame_index: int) -> GaussianFrame:
        """Return one exact global frame or reject an out-of-range index."""
        if frame_index < 0 or frame_index >= len(self.frames):
            raise KeyError(f"Unknown Gaussian frame index: {frame_index}.")
        return self.frames[frame_index]

    def active_frame_indices(self, object_id: str) -> tuple[int, ...]:
        """Return every global frame containing the requested object."""
        self.scene_object(object_id)
        return tuple(
            frame.frame_index
            for frame in self.frames
            if any(instance.object_id == object_id for instance in frame.instances)
        )

    def to_dict(self) -> dict[str, object]:
        """Return the current strict semantic scene record."""
        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "composition_id": self.composition_id,
            "background": self.background.to_dict(),
            "assets": [asset.to_dict() for asset in self.assets],
            "objects": [item.to_dict() for item in self.objects],
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse only the current semantic composition schema."""
        raw = _strict_mapping(
            value,
            name="Gaussian scene composition",
            keys={
                "schema",
                "scene_id",
                "composition_id",
                "background",
                "assets",
                "objects",
                "frames",
            },
        )
        return cls(
            schema=_string(raw["schema"], name="schema"),
            scene_id=_string(raw["scene_id"], name="scene_id"),
            composition_id=_string(raw["composition_id"], name="composition_id"),
            background=GaussianAsset.from_dict(raw["background"]),
            assets=_sequence_of(raw["assets"], GaussianAsset.from_dict, name="assets"),
            objects=_sequence_of(
                raw["objects"],
                GaussianSceneObject.from_dict,
                name="objects",
            ),
            frames=_sequence_of(raw["frames"], GaussianFrame.from_dict, name="frames"),
        )


@dataclass(frozen=True, slots=True)
class GaussianForegroundComposition:
    """Complete articulated foreground timeline with no background authority.

    Every object is a positive-identity movable asset.  Empty-space pixels are
    represented only by raster output masks; instance ID 0 is never a scene
    object in this contract.
    """

    scene_id: str
    composition_id: str
    assets: tuple[GaussianAsset, ...]
    objects: tuple[GaussianSceneObject, ...]
    frames: tuple[GaussianFrame, ...]
    schema: str = GAUSSIAN_FOREGROUND_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GAUSSIAN_FOREGROUND_SCHEMA:
            raise ValueError(
                f"Unsupported Gaussian foreground schema {self.schema!r}; "
                f"expected {GAUSSIAN_FOREGROUND_SCHEMA!r}."
            )
        _validate_id(self.scene_id, name="scene_id")
        _validate_id(self.composition_id, name="composition_id")
        assets = tuple(self.assets)
        objects = tuple(self.objects)
        frames = tuple(self.frames)
        if not assets or any(not isinstance(asset, GaussianAsset) for asset in assets):
            raise TypeError("Foreground assets must be non-empty GaussianAsset values.")
        if not objects or any(
            not isinstance(item, GaussianSceneObject) for item in objects
        ):
            raise TypeError(
                "Foreground objects must be non-empty GaussianSceneObject values."
            )
        if not frames or any(not isinstance(frame, GaussianFrame) for frame in frames):
            raise TypeError("Foreground frames must be non-empty GaussianFrame values.")
        if any(asset.role != GaussianAssetRole.MOVABLE for asset in assets):
            raise ValueError("Foreground assets must all be movable assets.")
        if any(
            item.deformation_kind != GaussianDeformationKind.ARTICULATED
            for item in objects
        ):
            raise ValueError("Foreground objects must all be declared articulated.")

        _require_unique([asset.asset_id for asset in assets], name="foreground asset ids")
        _require_unique([item.object_id for item in objects], name="foreground object ids")
        _require_unique(
            [item.instance_id for item in objects],
            name="foreground instance ids",
        )
        reference = assets[0]
        for asset in assets[1:]:
            _validate_render_compatibility(reference, asset)

        assets_by_id = {asset.asset_id: asset for asset in assets}
        unknown_assets = sorted(
            {item.asset_id for item in objects}.difference(assets_by_id)
        )
        if unknown_assets:
            raise ValueError(
                f"Foreground objects reference unknown assets: {unknown_assets}."
            )
        _validate_object_timeline(objects=objects, frames=frames)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "frames", frames)

    def asset(self, asset_id: str) -> GaussianAsset:
        """Return one movable foreground asset or fail closed."""
        matches = [asset for asset in self.assets if asset.asset_id == asset_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown foreground Gaussian asset: {asset_id!r}.")
        return matches[0]

    def scene_object(self, object_id: str) -> GaussianSceneObject:
        """Return one stable foreground object or fail closed."""
        matches = [item for item in self.objects if item.object_id == object_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown foreground Gaussian object: {object_id!r}.")
        return matches[0]

    def frame(self, frame_index: int) -> GaussianFrame:
        """Return one exact foreground frame."""
        if frame_index < 0 or frame_index >= len(self.frames):
            raise KeyError(f"Unknown Gaussian foreground frame index: {frame_index}.")
        return self.frames[frame_index]

    def active_frame_indices(self, object_id: str) -> tuple[int, ...]:
        """Return all global frames containing one foreground object."""
        self.scene_object(object_id)
        return tuple(
            frame.frame_index
            for frame in self.frames
            if any(instance.object_id == object_id for instance in frame.instances)
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict foreground-only semantic record."""
        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "composition_id": self.composition_id,
            "assets": [asset.to_dict() for asset in self.assets],
            "objects": [item.to_dict() for item in self.objects],
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse only the current foreground-only semantic schema."""
        raw = _strict_mapping(
            value,
            name="Gaussian foreground composition",
            keys={"schema", "scene_id", "composition_id", "assets", "objects", "frames"},
        )
        return cls(
            schema=_string(raw["schema"], name="schema"),
            scene_id=_string(raw["scene_id"], name="scene_id"),
            composition_id=_string(raw["composition_id"], name="composition_id"),
            assets=_sequence_of(raw["assets"], GaussianAsset.from_dict, name="assets"),
            objects=_sequence_of(
                raw["objects"],
                GaussianSceneObject.from_dict,
                name="objects",
            ),
            frames=_sequence_of(raw["frames"], GaussianFrame.from_dict, name="frames"),
        )


def _validate_render_compatibility(
    background: GaussianAsset,
    asset: GaussianAsset,
) -> None:
    if asset.appearance_model != background.appearance_model:
        raise ValueError(
            f"Asset {asset.asset_id!r} uses a different appearance model."
        )
    if asset.appearance_space != background.appearance_space:
        raise ValueError(
            f"Asset {asset.asset_id!r} uses a different appearance space."
        )
    if asset.feature_dim != background.feature_dim:
        raise ValueError(f"Asset {asset.asset_id!r} has a different feature dimension.")
    if asset.floating_dtype != background.floating_dtype:
        raise ValueError(f"Asset {asset.asset_id!r} has a different floating dtype.")


def _validate_object_timeline(
    *,
    objects: tuple[GaussianSceneObject, ...],
    frames: tuple[GaussianFrame, ...],
) -> None:
    frame_indices = tuple(frame.frame_index for frame in frames)
    expected_indices = tuple(range(len(frames)))
    if frame_indices != expected_indices:
        raise ValueError(
            "Gaussian foreground frame indices must exactly equal 0..T-1 in order; "
            f"got {frame_indices}."
        )
    objects_by_id = {item.object_id: item for item in objects}
    placements: dict[str, list[GaussianInstance]] = {
        object_id: [] for object_id in objects_by_id
    }
    for frame in frames:
        for instance in frame.instances:
            if instance.object_id not in objects_by_id:
                raise ValueError(
                    f"Frame {frame.frame_index} references unknown foreground object "
                    f"{instance.object_id!r}."
                )
            placements[instance.object_id].append(instance)
    unused = sorted(object_id for object_id, values in placements.items() if not values)
    if unused:
        raise ValueError(f"Declared foreground objects never appear: {unused}.")
    for object_id, values in placements.items():
        if len(values) < 2:
            raise ValueError(
                f"Articulated foreground object {object_id!r} requires at least two frames."
            )
        source_indices = [instance.source_frame_index for instance in values]
        expected_sources = list(
            range(source_indices[0], source_indices[0] + len(source_indices))
        )
        if source_indices != expected_sources:
            raise ValueError(
                f"Foreground object {object_id!r} source frames must be consecutive; "
                f"got {source_indices}."
            )


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}."
        )
    return cast(Mapping[str, object], value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    result = _integer(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _finite_tuple(
    value: Sequence[object],
    *,
    size: int,
    name: str,
) -> tuple[float, ...]:
    if len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} numeric values.")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only numeric values.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values.")
        result.append(number)
    return tuple(result)


def _validate_id(value: str, *, name: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable identifier: {value!r}.")


def _require_unique(values: Sequence[object], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates.")


def _sequence_of(
    value: object,
    parser: Callable[[object], _ParsedT],
    *,
    name: str,
) -> tuple[_ParsedT, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return tuple(parser(item) for item in value)


def _enum_value(
    enum_type: type[_EnumT],
    value: object,
    *,
    name: str,
) -> _EnumT:
    text = _string(value, name=name)
    try:
        return enum_type(text)
    except ValueError as error:
        supported = [item.value for item in enum_type]
        raise ValueError(f"Unsupported {name} {text!r}; expected one of {supported}.") from error


__all__ = [
    "ASSET_COORDINATE_CONVENTION",
    "GAUSSIAN_ASSET_SCHEMA",
    "GAUSSIAN_FOREGROUND_SCHEMA",
    "GAUSSIAN_SCENE_SCHEMA",
    "GAUSSIAN_TENSOR_ENCODING",
    "SCENE_COORDINATE_CONVENTION",
    "GaussianAsset",
    "GaussianAssetRole",
    "GaussianCoordinateFrame",
    "GaussianCoordinates",
    "GaussianDeformationKind",
    "GaussianFrame",
    "GaussianForegroundComposition",
    "GaussianInstance",
    "GaussianSceneComposition",
    "GaussianSceneObject",
    "GaussianTransform",
    "GaussianUnit",
]

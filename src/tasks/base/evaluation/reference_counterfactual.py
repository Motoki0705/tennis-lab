"""Strict paired-reference counterfactual evaluation shared by BLCS and PLCS.

The module deliberately keeps three concerns together because they form one
fail-closed artifact contract: deterministic side selection from persisted
#799 geometry, a strict reference-only pair join, and recomputable NPZ/JSON
reports.  Camera names, indices, and centres are never used to infer a side.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.evaluation.track_query_reference import (
    PairedReferenceEvaluationError,
    PairedReferenceKey,
    PairedReferencePositionMetrics,
    ReferenceTransformQuantity,
    compute_heading_error_radians,
    compute_paired_reference_position_metrics,
    compute_reference_transform_consistency_error,
)
from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_V2_SELECTOR,
    IDENTITY_ROTATION_3D,
    RZ_PI_ROTATION_3D,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
    validate_dataset_court_keypoint_contract_documents,
)

REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION: Final = 1
REFERENCE_COUNTERFACTUAL_NUM_VIEWS: Final = 6
REFERENCE_COUNTERFACTUAL_REPORT_STEM: Final = "reference_counterfactual"
REFERENCE_COUNTERFACTUAL_PREDICTIONS_FILENAME: Final = "pred_test.npz"
REFERENCE_COUNTERFACTUAL_METRICS_FILENAME: Final = "metrics.json"

ReferenceCounterfactualTask: TypeAlias = Literal["blcs", "plcs"]
ReferenceCounterfactualSide: TypeAlias = Literal["same_side", "opposite_side"]
ReferenceCounterfactualSelectorMode: TypeAlias = Literal["reference", "selector_zero"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_SIDES: Final[tuple[ReferenceCounterfactualSide, ...]] = (
    "same_side",
    "opposite_side",
)
_PARITY_FIELDS: Final[tuple[str, ...]] = (
    "frame_digest",
    "lifecycle_digest",
    "observation_digest",
    "target_digest",
)


class ReferenceCounterfactualError(PairedReferenceEvaluationError):
    """Raised when a counterfactual is not an exact reference-only pair."""


def _require_exact_fields(
    value: object,
    expected: set[str] | frozenset[str],
    *,
    location: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ReferenceCounterfactualError(
            f"{location} must be a string-keyed mapping."
        )
    mapping = cast("Mapping[str, object]", value)
    if set(mapping) != set(expected):
        raise ReferenceCounterfactualError(
            f"{location} must have exactly fields {sorted(expected)!r}; got "
            f"{sorted(mapping)!r}."
        )
    return mapping


def _require_non_empty_string(value: object, *, location: str) -> str:
    if type(value) is not str or not value.strip():
        raise ReferenceCounterfactualError(
            f"{location} must be a non-empty exact string."
        )
    return value


def _require_digest(value: object, *, location: str) -> str:
    digest = _require_non_empty_string(value, location=location)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ReferenceCounterfactualError(
            f"{location} must be a lowercase SHA-256 digest."
        )
    return digest


def _normalize_json(value: object, *, location: str) -> JsonValue:
    if value is None or type(value) in (str, bool, int):
        return cast("JsonScalar", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ReferenceCounterfactualError(
                f"{location} cannot contain non-finite JSON numbers."
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ReferenceCounterfactualError(
                    f"{location} must use only string mapping keys."
                )
            result[key] = _normalize_json(item, location=f"{location}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ReferenceCounterfactualError(
        f"{location} contains unsupported JSON value {type(value).__name__}."
    )


def canonical_json_sha256(value: object) -> str:
    """Return a deterministic digest for a finite JSON-compatible value."""
    normalized = _normalize_json(value, location="digest input")
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one existing regular file."""
    source = Path(path)
    if not source.is_file():
        raise ReferenceCounterfactualError(
            f"SHA-256 source is not an existing regular file: {source}."
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_mapping(path: Path) -> dict[str, object]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReferenceCounterfactualError(
            f"Required counterfactual metadata file is missing: {path}."
        ) from error
    except json.JSONDecodeError as error:
        raise ReferenceCounterfactualError(
            f"Invalid JSON in counterfactual metadata file {path}: {error}."
        ) from error
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ReferenceCounterfactualError(
            f"Counterfactual metadata must be a JSON object: {path}."
        )
    return cast("dict[str, object]", value)


@dataclass(frozen=True, slots=True)
class ReferenceSideSelection:
    """One persisted-transform side's deterministic camera selection."""

    side: ReferenceCounterfactualSide
    camera_id: str
    local_index: int
    provenance: CourtReferenceFrameProvenance

    def __post_init__(self) -> None:
        if self.side not in _SIDES:
            raise ReferenceCounterfactualError(f"Unknown reference side {self.side!r}.")
        _require_non_empty_string(self.camera_id, location="side camera_id")
        if type(self.local_index) is not int or self.local_index < 0:
            raise ReferenceCounterfactualError(
                "side local_index must be a non-negative exact integer."
            )
        if self.provenance.reference_camera_id != self.camera_id:
            raise ReferenceCounterfactualError(
                "Side camera ID does not match persisted reference provenance."
            )
        if self.provenance.reference_camera_local_index != self.local_index:
            raise ReferenceCounterfactualError(
                "Side local index does not match persisted reference provenance."
            )
        expected = (
            IDENTITY_ROTATION_3D if self.side == "same_side" else RZ_PI_ROTATION_3D
        )
        if self.provenance.reference_from_physical != expected:
            raise ReferenceCounterfactualError(
                f"{self.side} must be classified by the persisted "
                f"canonical_from_physical transform, not by camera identity hints."
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the exact JSON representation."""
        return {
            "side": self.side,
            "camera_id": self.camera_id,
            "local_index": self.local_index,
            "provenance": cast("dict[str, JsonValue]", self.provenance.to_dict()),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        location: str,
    ) -> ReferenceSideSelection:
        """Parse one exact persisted selection without inference."""
        mapping = _require_exact_fields(
            value,
            {"side", "camera_id", "local_index", "provenance"},
            location=location,
        )
        side = mapping["side"]
        if side not in _SIDES:
            raise ReferenceCounterfactualError(f"{location}.side is unknown: {side!r}.")
        try:
            provenance = CourtReferenceFrameProvenance.from_mapping(
                mapping["provenance"],
                location=f"{location}.provenance",
            )
        except ValueError as error:
            raise ReferenceCounterfactualError(str(error)) from error
        return cls(
            side=side,
            camera_id=_require_non_empty_string(
                mapping["camera_id"], location=f"{location}.camera_id"
            ),
            local_index=cast("int", mapping["local_index"]),
            provenance=provenance,
        )


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualScene:
    """Six-view scene identity and its deterministic transform-class pair."""

    scene_id: str
    view_camera_ids: tuple[str, ...]
    local_ordering: tuple[str, ...]
    same_side: ReferenceSideSelection
    opposite_side: ReferenceSideSelection

    def __post_init__(self) -> None:
        key = PairedReferenceKey(
            scene_id=self.scene_id,
            view_camera_ids=self.view_camera_ids,
            local_ordering=self.local_ordering,
        )
        if len(key.view_camera_ids) != REFERENCE_COUNTERFACTUAL_NUM_VIEWS:
            raise ReferenceCounterfactualError(
                "Counterfactual scenes must have exactly six valid camera IDs."
            )
        if tuple(sorted(key.view_camera_ids)) != key.view_camera_ids:
            raise ReferenceCounterfactualError(
                "view_camera_ids must be the lexicographically sorted stable ID set."
            )
        object.__setattr__(self, "scene_id", key.scene_id)
        object.__setattr__(self, "view_camera_ids", key.view_camera_ids)
        object.__setattr__(self, "local_ordering", key.local_ordering)
        if (
            self.same_side.side != "same_side"
            or self.opposite_side.side != "opposite_side"
        ):
            raise ReferenceCounterfactualError(
                "Scene side fields must contain their explicitly named side."
            )
        if self.same_side.camera_id == self.opposite_side.camera_id:
            raise ReferenceCounterfactualError(
                "Same-side and opposite-side references must be distinct cameras."
            )
        for selection in (self.same_side, self.opposite_side):
            if self.local_ordering[selection.local_index] != selection.camera_id:
                raise ReferenceCounterfactualError(
                    "Reference local index must resolve its stable ID in local order."
                )

    @property
    def key(self) -> PairedReferenceKey:
        """Return the existing shared pair identity primitive."""
        return PairedReferenceKey(
            scene_id=self.scene_id,
            view_camera_ids=self.view_camera_ids,
            local_ordering=self.local_ordering,
        )

    def selection(self, side: ReferenceCounterfactualSide) -> ReferenceSideSelection:
        """Return the explicit selection for one known side."""
        if side == "same_side":
            return self.same_side
        if side == "opposite_side":
            return self.opposite_side
        raise ReferenceCounterfactualError(f"Unknown reference side {side!r}.")

    def validate_camera_codes(
        self,
        view_camera_id_codes: Sequence[int],
        *,
        reference_camera_id: str,
        reference_camera_id_code: int,
    ) -> None:
        """Check complete-table ranks without treating local indices as IDs."""
        actual = tuple(view_camera_id_codes)
        if len(actual) != REFERENCE_COUNTERFACTUAL_NUM_VIEWS or any(
            type(code) is not int for code in actual
        ):
            raise ReferenceCounterfactualError(
                "Raw camera codes must contain six exact integer ranks."
            )
        rank_by_id = {
            camera_id: rank for rank, camera_id in enumerate(self.view_camera_ids)
        }
        expected = tuple(rank_by_id[camera_id] for camera_id in self.local_ordering)
        if actual != expected:
            raise ReferenceCounterfactualError(
                "Raw stable camera codes must be complete-scene lexicographic ranks, "
                "not local indices or selected-subset ranks."
            )
        try:
            expected_reference_code = rank_by_id[reference_camera_id]
        except KeyError as error:
            raise ReferenceCounterfactualError(
                "Raw reference camera identity is absent from the complete scene table."
            ) from error
        if (
            type(reference_camera_id_code) is not int
            or reference_camera_id_code != expected_reference_code
        ):
            raise ReferenceCounterfactualError(
                "Raw reference camera code does not match its canonical string rank."
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the exact JSON representation."""
        return {
            "scene_id": self.scene_id,
            "view_camera_ids": list(self.view_camera_ids),
            "local_ordering": list(self.local_ordering),
            "same_side": self.same_side.to_dict(),
            "opposite_side": self.opposite_side.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        location: str,
    ) -> ReferenceCounterfactualScene:
        """Parse one exact manifest scene."""
        mapping = _require_exact_fields(
            value,
            {
                "scene_id",
                "view_camera_ids",
                "local_ordering",
                "same_side",
                "opposite_side",
            },
            location=location,
        )
        raw_views = mapping["view_camera_ids"]
        raw_order = mapping["local_ordering"]
        if not isinstance(raw_views, list) or not isinstance(raw_order, list):
            raise ReferenceCounterfactualError(
                f"{location} camera ID fields must be JSON lists."
            )
        return cls(
            scene_id=_require_non_empty_string(
                mapping["scene_id"], location=f"{location}.scene_id"
            ),
            view_camera_ids=tuple(cast("list[str]", raw_views)),
            local_ordering=tuple(cast("list[str]", raw_order)),
            same_side=ReferenceSideSelection.from_dict(
                mapping["same_side"], location=f"{location}.same_side"
            ),
            opposite_side=ReferenceSideSelection.from_dict(
                mapping["opposite_side"], location=f"{location}.opposite_side"
            ),
        )


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualManifest:
    """Immutable deterministic side manifest for one dataset scene set."""

    schema_version: int
    dataset_schema_id: str
    scenes: tuple[ReferenceCounterfactualScene, ...]

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION:
            raise ReferenceCounterfactualError(
                "Reference manifest schema_version must be "
                f"{REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION}."
            )
        _require_non_empty_string(
            self.dataset_schema_id, location="manifest dataset_schema_id"
        )
        scenes = tuple(self.scenes)
        if not scenes:
            raise ReferenceCounterfactualError("Reference manifest cannot be empty.")
        scene_ids = tuple(scene.scene_id for scene in scenes)
        if len(set(scene_ids)) != len(scene_ids):
            raise ReferenceCounterfactualError(
                "Reference manifest cannot contain duplicate scene IDs."
            )
        if tuple(sorted(scene_ids)) != scene_ids:
            raise ReferenceCounterfactualError(
                "Reference manifest scenes must use deterministic lexicographic order."
            )
        object.__setattr__(self, "scenes", scenes)

    @property
    def digest(self) -> str:
        """Return the stable digest used by both counterfactual passes."""
        return canonical_json_sha256(self.to_dict())

    def scene(self, scene_id: str) -> ReferenceCounterfactualScene:
        """Resolve one scene exactly; missing identities are hard errors."""
        matches = tuple(scene for scene in self.scenes if scene.scene_id == scene_id)
        if len(matches) != 1:
            raise ReferenceCounterfactualError(
                f"Manifest must resolve exactly one scene {scene_id!r}; got {len(matches)}."
            )
        return matches[0]

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the exact JSON representation."""
        return {
            "schema_version": self.schema_version,
            "dataset_schema_id": self.dataset_schema_id,
            "scenes": [scene.to_dict() for scene in self.scenes],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        location: str = "manifest",
    ) -> ReferenceCounterfactualManifest:
        """Parse a stale-schema-intolerant manifest."""
        mapping = _require_exact_fields(
            value,
            {"schema_version", "dataset_schema_id", "scenes"},
            location=location,
        )
        raw_scenes = mapping["scenes"]
        if not isinstance(raw_scenes, list):
            raise ReferenceCounterfactualError(f"{location}.scenes must be a list.")
        return cls(
            schema_version=cast("int", mapping["schema_version"]),
            dataset_schema_id=_require_non_empty_string(
                mapping["dataset_schema_id"],
                location=f"{location}.dataset_schema_id",
            ),
            scenes=tuple(
                ReferenceCounterfactualScene.from_dict(
                    scene,
                    location=f"{location}.scenes[{index}]",
                )
                for index, scene in enumerate(raw_scenes)
            ),
        )


def _root_scene_summaries(
    root_metadata: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_summaries = root_metadata.get("scenes")
    if not isinstance(raw_summaries, list):
        raise ReferenceCounterfactualError(
            "Dataset root metadata must contain a scenes list for identity validation."
        )
    summaries: dict[str, Mapping[str, object]] = {}
    for index, raw_summary in enumerate(raw_summaries):
        if not isinstance(raw_summary, Mapping):
            raise ReferenceCounterfactualError(
                f"Dataset root scenes[{index}] must be a mapping."
            )
        summary = cast("Mapping[str, object]", raw_summary)
        scene_id = _require_non_empty_string(
            summary.get("scene_id"), location=f"root scenes[{index}].scene_id"
        )
        if scene_id in summaries:
            raise ReferenceCounterfactualError(
                f"Dataset root contains duplicate scene identity {scene_id!r}."
            )
        summaries[scene_id] = summary
    return summaries


def _selection_from_views(
    *,
    side: ReferenceCounterfactualSide,
    views: tuple[CourtViewRecord, ...],
) -> ReferenceSideSelection:
    expected_transform = (
        IDENTITY_ROTATION_3D if side == "same_side" else RZ_PI_ROTATION_3D
    )
    candidates = tuple(
        sorted(
            view.camera_id
            for view in views
            if view.canonical_from_physical == expected_transform
        )
    )
    if not candidates:
        raise ReferenceCounterfactualError(
            f"Scene has no persisted canonical_from_physical class for {side}."
        )
    camera_id = candidates[0]
    provenance = build_reference_frame_provenance(
        views,
        reference_camera_id=camera_id,
    )
    local_index = provenance.reference_camera_local_index
    if local_index is None:
        raise ReferenceCounterfactualError(
            "Persisted reference local index is missing."
        )
    return ReferenceSideSelection(
        side=side,
        camera_id=camera_id,
        local_index=local_index,
        provenance=provenance,
    )


def build_reference_counterfactual_manifest_from_documents(
    *,
    root_metadata: Mapping[str, object],
    scene_metadata: Mapping[str, Mapping[str, object]],
    expected_dataset_schema_id: str,
    dataset_location: str = "dataset",
) -> ReferenceCounterfactualManifest:
    """Build the two-class manifest only from validated persisted metadata."""
    if not scene_metadata:
        raise ReferenceCounterfactualError("Counterfactual scene metadata is empty.")
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    try:
        dataset = validate_dataset_court_keypoint_contract_documents(
            root_metadata=root_metadata,
            scene_metadata=scene_metadata,
            runtime_contract=contract,
            expected_dataset_schema_id=expected_dataset_schema_id,
            dataset_location=dataset_location,
        )
    except ValueError as error:
        raise ReferenceCounterfactualError(str(error)) from error
    summaries = _root_scene_summaries(root_metadata)
    records_by_scene = {
        record.scene_id: record.court_views for record in dataset.scenes
    }
    scenes: list[ReferenceCounterfactualScene] = []
    for scene_id in sorted(scene_metadata):
        document = scene_metadata[scene_id]
        if scene_id not in summaries:
            raise ReferenceCounterfactualError(
                f"Scene {scene_id!r} is absent from dataset root metadata."
            )
        summary = summaries[scene_id]
        if document.get("scene_id") != scene_id:
            raise ReferenceCounterfactualError(
                f"Scene directory identity {scene_id!r} disagrees with scene metadata."
            )
        if summary.get("file") != scene_id:
            raise ReferenceCounterfactualError(
                f"Root file identity for {scene_id!r} is inconsistent."
            )
        for location, value in (
            (f"root scene {scene_id!r}", summary.get("num_cameras")),
            (f"scene {scene_id!r}", document.get("num_cameras")),
        ):
            if type(value) is not int or value != REFERENCE_COUNTERFACTUAL_NUM_VIEWS:
                raise ReferenceCounterfactualError(
                    f"{location} must declare exactly six cameras; got {value!r}."
                )
        views = records_by_scene.get(scene_id)
        if views is None or len(views) != REFERENCE_COUNTERFACTUAL_NUM_VIEWS:
            raise ReferenceCounterfactualError(
                f"Scene {scene_id!r} must persist exactly six court-view records."
            )
        local_ordering = tuple(view.camera_id for view in views)
        view_camera_ids = tuple(sorted(local_ordering))
        scenes.append(
            ReferenceCounterfactualScene(
                scene_id=scene_id,
                view_camera_ids=view_camera_ids,
                local_ordering=local_ordering,
                same_side=_selection_from_views(side="same_side", views=views),
                opposite_side=_selection_from_views(side="opposite_side", views=views),
            )
        )
    return ReferenceCounterfactualManifest(
        schema_version=REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION,
        dataset_schema_id=expected_dataset_schema_id,
        scenes=tuple(scenes),
    )


def build_reference_counterfactual_manifest(
    dataset_root: str | Path,
    *,
    expected_dataset_schema_id: str,
    scene_ids: Sequence[str] | None = None,
) -> ReferenceCounterfactualManifest:
    """Load a dataset root and build its deterministic requested-scene manifest."""
    root = Path(dataset_root)
    root_metadata = _read_json_mapping(root / "meta.json")
    if scene_ids is None:
        scenes_dir = root / "scenes"
        if not scenes_dir.is_dir():
            raise ReferenceCounterfactualError(
                f"Dataset scenes directory is missing: {scenes_dir}."
            )
        requested = tuple(
            sorted(path.name for path in scenes_dir.iterdir() if path.is_dir())
        )
    else:
        requested = tuple(scene_ids)
        if not requested or any(
            type(scene_id) is not str or not scene_id.strip() for scene_id in requested
        ):
            raise ReferenceCounterfactualError(
                "scene_ids must contain non-empty canonical scene strings."
            )
        if len(set(requested)) != len(requested):
            raise ReferenceCounterfactualError("scene_ids cannot contain duplicates.")
        requested = tuple(sorted(requested))
    documents = {
        scene_id: _read_json_mapping(root / "scenes" / scene_id / "meta.json")
        for scene_id in requested
    }
    return build_reference_counterfactual_manifest_from_documents(
        root_metadata=root_metadata,
        scene_metadata=documents,
        expected_dataset_schema_id=expected_dataset_schema_id,
        dataset_location=str(root),
    )


def canonical_json_text(value: object) -> str:
    """Return the canonical finite JSON text persisted as the resolved config."""
    normalized = _normalize_json(value, location="resolved config")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualRunIdentity:
    """Run-wide identity that must remain exact across the two raw passes."""

    task: ReferenceCounterfactualTask
    seed: int
    selector_mode: ReferenceCounterfactualSelectorMode
    resolved_config_json: str
    resolved_config_digest: str
    checkpoint_sha256: str
    manifest_digest: str
    court_keypoint_contract: str
    target_frame_contract: str
    track_query_rope_contract: str

    def __post_init__(self) -> None:
        if self.task not in ("blcs", "plcs"):
            raise ReferenceCounterfactualError(f"Unknown task {self.task!r}.")
        if type(self.seed) is not int or self.seed < 0:
            raise ReferenceCounterfactualError(
                "Counterfactual seed must be a non-negative exact integer."
            )
        if self.selector_mode not in ("reference", "selector_zero"):
            raise ReferenceCounterfactualError(
                f"Unknown selector mode {self.selector_mode!r}."
            )
        for field_name in (
            "resolved_config_digest",
            "checkpoint_sha256",
            "manifest_digest",
        ):
            _require_digest(getattr(self, field_name), location=field_name)
        for field_name in (
            "court_keypoint_contract",
            "target_frame_contract",
            "track_query_rope_contract",
        ):
            _require_non_empty_string(getattr(self, field_name), location=field_name)
        config_text = _require_non_empty_string(
            self.resolved_config_json,
            location="resolved_config_json",
        )
        try:
            config: Any = json.loads(config_text)
        except json.JSONDecodeError as error:
            raise ReferenceCounterfactualError(
                f"resolved_config_json is invalid JSON: {error}."
            ) from error
        canonical = canonical_json_text(config)
        if canonical != config_text:
            raise ReferenceCounterfactualError(
                "resolved_config_json must use canonical sorted compact JSON."
            )
        if hashlib.sha256(config_text.encode("utf-8")).hexdigest() != (
            self.resolved_config_digest
        ):
            raise ReferenceCounterfactualError(
                "resolved_config_digest does not match resolved_config_json."
            )

    @classmethod
    def create(
        cls,
        *,
        task: ReferenceCounterfactualTask,
        seed: int,
        selector_mode: ReferenceCounterfactualSelectorMode,
        resolved_config: object,
        checkpoint_sha256: str,
        manifest_digest: str,
        court_keypoint_contract: str,
        target_frame_contract: str,
        track_query_rope_contract: str,
    ) -> ReferenceCounterfactualRunIdentity:
        """Create identity fields from one fully resolved config value."""
        config_json = canonical_json_text(resolved_config)
        return cls(
            task=task,
            seed=seed,
            selector_mode=selector_mode,
            resolved_config_json=config_json,
            resolved_config_digest=hashlib.sha256(
                config_json.encode("utf-8")
            ).hexdigest(),
            checkpoint_sha256=checkpoint_sha256,
            manifest_digest=manifest_digest,
            court_keypoint_contract=court_keypoint_contract,
            target_frame_contract=target_frame_contract,
            track_query_rope_contract=track_query_rope_contract,
        )

    @property
    def resolved_config(self) -> JsonValue:
        """Return a detached parsed resolved configuration."""
        return cast("JsonValue", json.loads(self.resolved_config_json))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the exact JSON representation."""
        return {
            "task": self.task,
            "seed": self.seed,
            "selector_mode": self.selector_mode,
            "resolved_config": self.resolved_config,
            "resolved_config_digest": self.resolved_config_digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_digest": self.manifest_digest,
            "court_keypoint_contract": self.court_keypoint_contract,
            "target_frame_contract": self.target_frame_contract,
            "track_query_rope_contract": self.track_query_rope_contract,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        location: str = "run_identity",
    ) -> ReferenceCounterfactualRunIdentity:
        """Parse an exact run identity and verify its config digest."""
        fields = {
            "task",
            "seed",
            "selector_mode",
            "resolved_config",
            "resolved_config_digest",
            "checkpoint_sha256",
            "manifest_digest",
            "court_keypoint_contract",
            "target_frame_contract",
            "track_query_rope_contract",
        }
        mapping = _require_exact_fields(value, fields, location=location)
        task = mapping["task"]
        selector_mode = mapping["selector_mode"]
        if task not in ("blcs", "plcs"):
            raise ReferenceCounterfactualError(f"{location}.task is unknown.")
        if selector_mode not in ("reference", "selector_zero"):
            raise ReferenceCounterfactualError(f"{location}.selector_mode is unknown.")
        return cls(
            task=task,
            seed=cast("int", mapping["seed"]),
            selector_mode=selector_mode,
            resolved_config_json=canonical_json_text(mapping["resolved_config"]),
            resolved_config_digest=_require_digest(
                mapping["resolved_config_digest"],
                location=f"{location}.resolved_config_digest",
            ),
            checkpoint_sha256=_require_digest(
                mapping["checkpoint_sha256"],
                location=f"{location}.checkpoint_sha256",
            ),
            manifest_digest=_require_digest(
                mapping["manifest_digest"],
                location=f"{location}.manifest_digest",
            ),
            court_keypoint_contract=_require_non_empty_string(
                mapping["court_keypoint_contract"],
                location=f"{location}.court_keypoint_contract",
            ),
            target_frame_contract=_require_non_empty_string(
                mapping["target_frame_contract"],
                location=f"{location}.target_frame_contract",
            ),
            track_query_rope_contract=_require_non_empty_string(
                mapping["track_query_rope_contract"],
                location=f"{location}.track_query_rope_contract",
            ),
        )


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualQuantitySchema:
    """Explicit quantity presence; task-inapplicable fields cannot be inferred."""

    position: bool
    vector: bool
    heading: bool
    world_joints: bool

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (self.position, self.vector, self.heading, self.world_joints)
        ):
            raise ReferenceCounterfactualError(
                "Quantity availability fields must be exact booleans."
            )
        if not self.position:
            raise ReferenceCounterfactualError(
                "Counterfactual evaluation always requires position arrays."
            )

    @classmethod
    def for_task(
        cls,
        task: ReferenceCounterfactualTask,
        *,
        vector: bool = False,
        world_joints: bool = False,
    ) -> ReferenceCounterfactualQuantitySchema:
        """Build an explicit task schema without substituting quantities."""
        if task == "blcs":
            if world_joints:
                raise ReferenceCounterfactualError(
                    "BLCS cannot declare PLCS world-joint quantities."
                )
            return cls(
                position=True,
                vector=vector,
                heading=False,
                world_joints=False,
            )
        if task == "plcs":
            if vector:
                raise ReferenceCounterfactualError(
                    "PLCS cannot declare BLCS vector quantities."
                )
            return cls(
                position=True,
                vector=False,
                heading=True,
                world_joints=world_joints,
            )
        raise ReferenceCounterfactualError(f"Unknown task {task!r}.")

    def validate_task(self, task: ReferenceCounterfactualTask) -> None:
        """Reject mixed task/quantity schemas."""
        if task == "blcs" and (self.heading or self.world_joints):
            raise ReferenceCounterfactualError(
                "BLCS quantity schema must explicitly omit heading and world joints."
            )
        if task == "plcs" and (self.vector or not self.heading):
            raise ReferenceCounterfactualError(
                "PLCS quantity schema must explicitly omit vector and include heading."
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the exact JSON representation."""
        return {
            "position": self.position,
            "vector": self.vector,
            "heading": self.heading,
            "world_joints": self.world_joints,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        location: str = "quantity_schema",
    ) -> ReferenceCounterfactualQuantitySchema:
        """Parse explicit availability booleans."""
        mapping = _require_exact_fields(
            value,
            {"position", "vector", "heading", "world_joints"},
            location=location,
        )
        return cls(
            position=cast("bool", mapping["position"]),
            vector=cast("bool", mapping["vector"]),
            heading=cast("bool", mapping["heading"]),
            world_joints=cast("bool", mapping["world_joints"]),
        )


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualPassRow:
    """Strict per-window identity and parity evidence for one raw side pass."""

    key: PairedReferenceKey
    window_start: int
    window_stop: int
    reference_camera_id: str
    reference_view_index: int
    provenance: CourtReferenceFrameProvenance
    frame_digest: str
    lifecycle_digest: str
    observation_digest: str
    target_digest: str

    def __post_init__(self) -> None:
        if len(self.key.view_camera_ids) != REFERENCE_COUNTERFACTUAL_NUM_VIEWS:
            raise ReferenceCounterfactualError(
                "Raw pass rows must contain exactly six valid camera IDs."
            )
        if type(self.window_start) is not int or type(self.window_stop) is not int:
            raise ReferenceCounterfactualError(
                "Window boundaries must be exact integers."
            )
        if self.window_start < 0 or self.window_stop <= self.window_start:
            raise ReferenceCounterfactualError(
                "Window boundaries must define a positive non-negative interval."
            )
        _require_non_empty_string(
            self.reference_camera_id, location="row reference_camera_id"
        )
        if (
            type(self.reference_view_index) is not int
            or self.reference_view_index < 0
            or self.reference_view_index >= REFERENCE_COUNTERFACTUAL_NUM_VIEWS
        ):
            raise ReferenceCounterfactualError(
                "reference_view_index must select one of the six non-padding views."
            )
        if (
            self.key.local_ordering[self.reference_view_index]
            != self.reference_camera_id
        ):
            raise ReferenceCounterfactualError(
                "Reference local index/identity disagrees with ordered camera IDs."
            )
        if self.provenance.reference_camera_id != self.reference_camera_id:
            raise ReferenceCounterfactualError(
                "Row reference identity disagrees with persisted provenance."
            )
        if self.provenance.reference_camera_local_index != self.reference_view_index:
            raise ReferenceCounterfactualError(
                "Row reference index disagrees with persisted provenance."
            )
        for field_name in _PARITY_FIELDS:
            _require_digest(getattr(self, field_name), location=field_name)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return exact row metadata, including all input parity digests."""
        return {
            "scene_id": self.key.scene_id,
            "view_camera_ids": list(self.key.view_camera_ids),
            "local_ordering": list(self.key.local_ordering),
            "window_start": self.window_start,
            "window_stop": self.window_stop,
            "reference_camera_id": self.reference_camera_id,
            "reference_view_index": self.reference_view_index,
            "provenance": cast("dict[str, JsonValue]", self.provenance.to_dict()),
            "frame_digest": self.frame_digest,
            "lifecycle_digest": self.lifecycle_digest,
            "observation_digest": self.observation_digest,
            "target_digest": self.target_digest,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        location: str,
    ) -> ReferenceCounterfactualPassRow:
        """Parse one exact raw-pass row."""
        fields = {
            "scene_id",
            "view_camera_ids",
            "local_ordering",
            "window_start",
            "window_stop",
            "reference_camera_id",
            "reference_view_index",
            "provenance",
            *_PARITY_FIELDS,
        }
        mapping = _require_exact_fields(value, fields, location=location)
        raw_views = mapping["view_camera_ids"]
        raw_order = mapping["local_ordering"]
        if not isinstance(raw_views, list) or not isinstance(raw_order, list):
            raise ReferenceCounterfactualError(
                f"{location} camera identities must be lists."
            )
        try:
            provenance = CourtReferenceFrameProvenance.from_mapping(
                mapping["provenance"],
                location=f"{location}.provenance",
            )
        except ValueError as error:
            raise ReferenceCounterfactualError(str(error)) from error
        return cls(
            key=PairedReferenceKey(
                scene_id=cast("str", mapping["scene_id"]),
                view_camera_ids=tuple(cast("list[str]", raw_views)),
                local_ordering=tuple(cast("list[str]", raw_order)),
            ),
            window_start=cast("int", mapping["window_start"]),
            window_stop=cast("int", mapping["window_stop"]),
            reference_camera_id=cast("str", mapping["reference_camera_id"]),
            reference_view_index=cast("int", mapping["reference_view_index"]),
            provenance=provenance,
            frame_digest=_require_digest(
                mapping["frame_digest"], location=f"{location}.frame_digest"
            ),
            lifecycle_digest=_require_digest(
                mapping["lifecycle_digest"],
                location=f"{location}.lifecycle_digest",
            ),
            observation_digest=_require_digest(
                mapping["observation_digest"],
                location=f"{location}.observation_digest",
            ),
            target_digest=_require_digest(
                mapping["target_digest"], location=f"{location}.target_digest"
            ),
        )


def _readonly_float_array(
    value: object,
    *,
    quantity: str,
    trailing_width: int,
) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{quantity} must be a numpy.ndarray.")
    if value.ndim < 2 or value.shape[-1] != trailing_width:
        raise ReferenceCounterfactualError(
            f"{quantity} must have trailing width {trailing_width}; got {value.shape}."
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise ReferenceCounterfactualError(f"{quantity} must use a floating dtype.")
    if not np.isfinite(value).all():
        raise ReferenceCounterfactualError(
            f"{quantity} must be finite, including unsupervised/padded cells."
        )
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return cast("FloatArray", result)


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualQuantityArrays:
    """Immutable prediction/target arrays for one explicitly named quantity."""

    prediction: FloatArray
    target: FloatArray
    quantity: ReferenceTransformQuantity

    def __post_init__(self) -> None:
        if self.quantity not in ("point", "vector", "heading", "world_joints"):
            raise ReferenceCounterfactualError(
                f"Unknown reference transform quantity {self.quantity!r}."
            )
        trailing_width = 2 if self.quantity == "heading" else 3
        prediction = _readonly_float_array(
            self.prediction,
            quantity=f"{self.quantity} prediction",
            trailing_width=trailing_width,
        )
        target = _readonly_float_array(
            self.target,
            quantity=f"{self.quantity} target",
            trailing_width=trailing_width,
        )
        if prediction.shape != target.shape:
            raise ReferenceCounterfactualError(
                f"{self.quantity} prediction and target shapes differ."
            )
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "target", target)


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualPass:
    """One immutable raw same/opposite inference pass."""

    schema_version: int
    side: ReferenceCounterfactualSide
    identity: ReferenceCounterfactualRunIdentity
    quantity_schema: ReferenceCounterfactualQuantitySchema
    rows: tuple[ReferenceCounterfactualPassRow, ...]
    valid_mask: BoolArray
    position: ReferenceCounterfactualQuantityArrays
    vector: ReferenceCounterfactualQuantityArrays | None = None
    heading: ReferenceCounterfactualQuantityArrays | None = None
    world_joints: ReferenceCounterfactualQuantityArrays | None = None

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION:
            raise ReferenceCounterfactualError(
                "Raw pass schema_version is stale or unknown."
            )
        if self.side not in _SIDES:
            raise ReferenceCounterfactualError(f"Unknown pass side {self.side!r}.")
        self.quantity_schema.validate_task(self.identity.task)
        rows = tuple(self.rows)
        if not rows:
            raise ReferenceCounterfactualError("Raw counterfactual pass is empty.")
        keys = tuple(row.key for row in rows)
        if len(set(keys)) != len(keys):
            raise ReferenceCounterfactualError(
                "Raw counterfactual pass contains duplicate pair keys."
            )
        object.__setattr__(self, "rows", rows)
        if (
            not isinstance(self.valid_mask, np.ndarray)
            or self.valid_mask.dtype != np.bool_
        ):
            raise ReferenceCounterfactualError("valid_mask must be a bool numpy array.")
        valid_mask = np.ascontiguousarray(self.valid_mask).copy()
        if valid_mask.ndim < 1 or valid_mask.shape[0] != len(rows):
            raise ReferenceCounterfactualError(
                "valid_mask must have one leading row for every pass row."
            )
        if not valid_mask.any():
            raise ReferenceCounterfactualError(
                "A raw pass needs at least one supervised observation across all rows."
            )
        valid_mask.setflags(write=False)
        object.__setattr__(self, "valid_mask", valid_mask)
        if self.position.quantity != "point":
            raise ReferenceCounterfactualError(
                "position arrays must use the point transform quantity."
            )
        if self.position.prediction.shape[:-1] != valid_mask.shape:
            raise ReferenceCounterfactualError(
                "valid_mask must match position leading axes exactly."
            )
        expected = {
            "vector": self.quantity_schema.vector,
            "heading": self.quantity_schema.heading,
            "world_joints": self.quantity_schema.world_joints,
        }
        quantities = {
            "vector": self.vector,
            "heading": self.heading,
            "world_joints": self.world_joints,
        }
        expected_kind: dict[str, ReferenceTransformQuantity] = {
            "vector": "vector",
            "heading": "heading",
            "world_joints": "world_joints",
        }
        for name, present in expected.items():
            value = quantities[name]
            if present != (value is not None):
                raise ReferenceCounterfactualError(
                    f"{name} arrays must exactly match explicit quantity availability."
                )
            if value is None:
                continue
            if value.quantity != expected_kind[name]:
                raise ReferenceCounterfactualError(
                    f"{name} arrays use the wrong transform quantity."
                )
            leading = value.prediction.shape[:-1]
            if name == "world_joints":
                if value.prediction.ndim != valid_mask.ndim + 2:
                    raise ReferenceCounterfactualError(
                        "world_joints must add exactly one joint axis to valid_mask."
                    )
                if value.prediction.shape[:-2] != valid_mask.shape:
                    raise ReferenceCounterfactualError(
                        "world_joints leading sample axes must match valid_mask."
                    )
            elif leading != valid_mask.shape:
                raise ReferenceCounterfactualError(
                    f"{name} leading axes must match valid_mask."
                )

    def metadata_dict(self) -> dict[str, JsonValue]:
        """Return pass metadata; numeric arrays live only in the paired NPZ."""
        return {
            "schema_version": self.schema_version,
            "side": self.side,
            "identity": self.identity.to_dict(),
            "quantity_schema": self.quantity_schema.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }


def _torch_array(value: np.ndarray[Any, Any]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value).copy())


def _quantity_valid_mask(
    valid_mask: BoolArray,
    quantity: ReferenceTransformQuantity,
    value: FloatArray,
) -> BoolArray:
    if quantity != "world_joints":
        return valid_mask
    expanded = np.broadcast_to(valid_mask[..., None], value.shape[:-1])
    return cast("BoolArray", np.ascontiguousarray(expanded))


def _mean_physical_consistency(
    first: ReferenceCounterfactualQuantityArrays,
    first_rows: tuple[ReferenceCounterfactualPassRow, ...],
    second: ReferenceCounterfactualQuantityArrays,
    second_rows: tuple[ReferenceCounterfactualPassRow, ...],
    *,
    valid_mask: BoolArray,
) -> float:
    if (
        first.quantity != second.quantity
        or first.prediction.shape != second.prediction.shape
    ):
        raise ReferenceCounterfactualError(
            "Counterfactual quantity kind/shape differs across reference passes."
        )
    mask = _quantity_valid_mask(valid_mask, first.quantity, first.prediction)
    weighted_sum = 0.0
    total_count = 0
    for index, (first_row, second_row) in enumerate(
        zip(first_rows, second_rows, strict=True)
    ):
        row_mask = mask[index : index + 1]
        count = int(row_mask.sum())
        if count <= 0:
            continue
        error = compute_reference_transform_consistency_error(
            _torch_array(first.prediction[index : index + 1]),
            first_row.provenance,
            _torch_array(second.prediction[index : index + 1]),
            second_row.provenance,
            quantity=first.quantity,
            valid_mask=_torch_array(row_mask),
        )
        weighted_sum += error * count
        total_count += count
    if total_count <= 0:
        raise ReferenceCounterfactualError(
            "Physical consistency needs at least one supervised observation."
        )
    result = weighted_sum / total_count
    if not math.isfinite(result):
        raise ReferenceCounterfactualError(
            "Physical consistency computation produced a non-finite value."
        )
    return result


def _mean_target_physical_consistency(
    first: ReferenceCounterfactualQuantityArrays,
    first_rows: tuple[ReferenceCounterfactualPassRow, ...],
    second: ReferenceCounterfactualQuantityArrays,
    second_rows: tuple[ReferenceCounterfactualPassRow, ...],
    *,
    valid_mask: BoolArray,
) -> float:
    first_targets = ReferenceCounterfactualQuantityArrays(
        prediction=first.target,
        target=first.target,
        quantity=first.quantity,
    )
    second_targets = ReferenceCounterfactualQuantityArrays(
        prediction=second.target,
        target=second.target,
        quantity=second.quantity,
    )
    return _mean_physical_consistency(
        first_targets,
        first_rows,
        second_targets,
        second_rows,
        valid_mask=valid_mask,
    )


@dataclass(frozen=True, slots=True)
class ReferenceTargetFrameMetrics:
    """Metrics computed only in one side's authoritative reference target frame."""

    position: PairedReferencePositionMetrics
    heading_error_deg: float | None

    def __post_init__(self) -> None:
        values = (
            self.position.y_sign_accuracy,
            self.position.axis_wise_position_error.x,
            self.position.axis_wise_position_error.y,
            self.position.axis_wise_position_error.z,
            *self.position.local_reference_index_error.values(),
        )
        if any(not math.isfinite(value) for value in values):
            raise ReferenceCounterfactualError(
                "Reference target-frame metrics must be finite."
            )
        if not 0.0 <= self.position.y_sign_accuracy <= 1.0:
            raise ReferenceCounterfactualError("Y-sign accuracy must be in [0, 1].")
        if self.heading_error_deg is not None and (
            not math.isfinite(self.heading_error_deg) or self.heading_error_deg < 0.0
        ):
            raise ReferenceCounterfactualError(
                "Heading error degrees must be finite and non-negative."
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return stable aggregate field names."""
        axis = self.position.axis_wise_position_error
        return {
            "y_sign_accuracy": self.position.y_sign_accuracy,
            "axis_wise_position_error_m": {
                "x": axis.x,
                "y": axis.y,
                "z": axis.z,
            },
            "reference_local_index_position_error_m": {
                str(index): value
                for index, value in sorted(
                    self.position.local_reference_index_error.items()
                )
            },
            "heading_error_deg": self.heading_error_deg,
        }


@dataclass(frozen=True, slots=True)
class ReferencePhysicalConsistencyMetrics:
    """Cross-side differences computed only after restoring both to physical."""

    position_error_m: float
    vector_error_m: float | None
    heading_error: float | None
    world_joints_error_m: float | None

    def __post_init__(self) -> None:
        for name, value in (
            ("position_error_m", self.position_error_m),
            ("vector_error_m", self.vector_error_m),
            ("heading_error", self.heading_error),
            ("world_joints_error_m", self.world_joints_error_m),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ReferenceCounterfactualError(
                    f"Physical consistency {name} must be finite and non-negative."
                )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return exact quantities; unavailable task quantities remain null."""
        return {
            "position_error_m": self.position_error_m,
            "vector_error_m": self.vector_error_m,
            "heading_error": self.heading_error,
            "world_joints_error_m": self.world_joints_error_m,
        }


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualMetrics:
    """Both target-frame side metrics and physical-restored consistency."""

    same_side: ReferenceTargetFrameMetrics
    opposite_side: ReferenceTargetFrameMetrics
    physical_consistency: ReferencePhysicalConsistencyMetrics

    def to_dict(self) -> dict[str, JsonValue]:
        """Return exact metric sections with frame semantics in their names."""
        return {
            "reference_target_frame": {
                "same_side": self.same_side.to_dict(),
                "opposite_side": self.opposite_side.to_dict(),
            },
            "physical_restored_consistency": self.physical_consistency.to_dict(),
        }

    def flat_dict(self) -> dict[str, float]:
        """Return queue-registerable metrics as stable flat finite numbers."""
        result: dict[str, float] = {}
        for side, metrics in (
            ("same_side", self.same_side),
            ("opposite_side", self.opposite_side),
        ):
            prefix = f"reference_target_{side}"
            axis = metrics.position.axis_wise_position_error
            result[f"{prefix}_y_sign_accuracy"] = metrics.position.y_sign_accuracy
            result[f"{prefix}_position_error_x_m"] = axis.x
            result[f"{prefix}_position_error_y_m"] = axis.y
            result[f"{prefix}_position_error_z_m"] = axis.z
            for index, value in sorted(
                metrics.position.local_reference_index_error.items()
            ):
                result[
                    f"{prefix}_reference_index_{index}_position_error_m"
                ] = value
            if metrics.heading_error_deg is not None:
                result[f"{prefix}_heading_error_deg"] = metrics.heading_error_deg

        physical = self.physical_consistency
        result["physical_restored_position_consistency_error_m"] = (
            physical.position_error_m
        )
        if physical.vector_error_m is not None:
            result["physical_restored_vector_consistency_error_m"] = (
                physical.vector_error_m
            )
        if physical.heading_error is not None:
            result["physical_restored_heading_consistency_l2"] = (
                physical.heading_error
            )
        if physical.world_joints_error_m is not None:
            result["physical_restored_world_joints_consistency_error_m"] = (
                physical.world_joints_error_m
            )
        if any(not math.isfinite(value) for value in result.values()):
            raise ReferenceCounterfactualError(
                "Flat counterfactual metrics must contain only finite numbers."
            )
        return result


def _target_frame_metrics(
    pass_value: ReferenceCounterfactualPass,
) -> ReferenceTargetFrameMetrics:
    reference_indices = np.asarray(
        [row.reference_view_index for row in pass_value.rows],
        dtype=np.int64,
    )
    position = compute_paired_reference_position_metrics(
        _torch_array(pass_value.position.prediction),
        _torch_array(pass_value.position.target),
        _torch_array(reference_indices),
        valid_mask=_torch_array(pass_value.valid_mask),
    )
    heading_error_deg: float | None = None
    if pass_value.heading is not None:
        heading_error_deg = math.degrees(
            compute_heading_error_radians(
                _torch_array(pass_value.heading.prediction),
                _torch_array(pass_value.heading.target),
                valid_mask=_torch_array(pass_value.valid_mask),
            )
        )
    return ReferenceTargetFrameMetrics(
        position=position,
        heading_error_deg=heading_error_deg,
    )


def _validate_pass_against_manifest(
    pass_value: ReferenceCounterfactualPass,
    manifest: ReferenceCounterfactualManifest,
) -> None:
    if pass_value.identity.manifest_digest != manifest.digest:
        raise ReferenceCounterfactualError(
            "Raw pass manifest digest is stale or belongs to another manifest."
        )
    row_scene_ids = tuple(row.key.scene_id for row in pass_value.rows)
    manifest_scene_ids = tuple(scene.scene_id for scene in manifest.scenes)
    if row_scene_ids != manifest_scene_ids:
        raise ReferenceCounterfactualError(
            "Raw pass scene count/identity/order must exactly cover the manifest: "
            f"expected {manifest_scene_ids!r}, got {row_scene_ids!r}."
        )
    for row in pass_value.rows:
        scene = manifest.scene(row.key.scene_id)
        if row.key != scene.key:
            raise ReferenceCounterfactualError(
                f"Raw row camera set/order differs from manifest scene {row.key.scene_id!r}."
            )
        expected = scene.selection(pass_value.side)
        if (
            row.reference_camera_id != expected.camera_id
            or row.reference_view_index != expected.local_index
            or row.provenance != expected.provenance
        ):
            raise ReferenceCounterfactualError(
                f"Raw {pass_value.side} row does not use its persisted-transform "
                "manifest selection."
            )
        if row.provenance.contract_id != pass_value.identity.court_keypoint_contract:
            raise ReferenceCounterfactualError(
                "Row provenance and run court contract differ."
            )
        if row.provenance.target_frame_id != pass_value.identity.target_frame_contract:
            raise ReferenceCounterfactualError(
                "Row provenance and run target-frame contract differ."
            )


def _pair_parity_value(
    identity: ReferenceCounterfactualRunIdentity,
    schema: ReferenceCounterfactualQuantitySchema,
    first_rows: tuple[ReferenceCounterfactualPassRow, ...],
    second_rows: tuple[ReferenceCounterfactualPassRow, ...],
) -> dict[str, JsonValue]:
    rows: list[JsonValue] = []
    for first, second in zip(first_rows, second_rows, strict=True):
        rows.append(
            {
                "scene_id": first.key.scene_id,
                "view_camera_ids": list(first.key.view_camera_ids),
                "local_ordering": list(first.key.local_ordering),
                "window_start": first.window_start,
                "window_stop": first.window_stop,
                "same_side_reference_camera_id": first.reference_camera_id,
                "same_side_reference_view_index": first.reference_view_index,
                "opposite_side_reference_camera_id": second.reference_camera_id,
                "opposite_side_reference_view_index": second.reference_view_index,
                **{field: getattr(first, field) for field in _PARITY_FIELDS},
            }
        )
    return {
        "identity": identity.to_dict(),
        "quantity_schema": schema.to_dict(),
        "rows": rows,
    }


def _validate_target_consistency(
    same_side: ReferenceCounterfactualPass,
    opposite_side: ReferenceCounterfactualPass,
) -> None:
    quantities: tuple[
        tuple[
            ReferenceCounterfactualQuantityArrays | None,
            ReferenceCounterfactualQuantityArrays | None,
        ],
        ...,
    ] = (
        (same_side.position, opposite_side.position),
        (same_side.vector, opposite_side.vector),
        (same_side.heading, opposite_side.heading),
        (same_side.world_joints, opposite_side.world_joints),
    )
    for first, second in quantities:
        if first is None or second is None:
            continue
        error = _mean_target_physical_consistency(
            first,
            same_side.rows,
            second,
            opposite_side.rows,
            valid_mask=same_side.valid_mask,
        )
        tolerance = (
            1e-9
            if np.dtype(first.target.dtype) == np.dtype(np.float64)
            and np.dtype(second.target.dtype) == np.dtype(np.float64)
            else 1e-5
        )
        if error > tolerance:
            raise ReferenceCounterfactualError(
                f"{first.quantity} targets do not restore to the same physical "
                f"quantity: mean error {error} > {tolerance}."
            )


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualReport:
    """Validated joined report retaining every array required to recompute it."""

    schema_version: int
    manifest: ReferenceCounterfactualManifest
    same_side_pass: ReferenceCounterfactualPass
    opposite_side_pass: ReferenceCounterfactualPass
    metrics: ReferenceCounterfactualMetrics
    parity_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION:
            raise ReferenceCounterfactualError("Joined report schema is stale.")
        _require_digest(self.parity_digest, location="parity_digest")

    @property
    def identity(self) -> ReferenceCounterfactualRunIdentity:
        """Return the exactly shared run identity."""
        return self.same_side_pass.identity

    @property
    def quantity_schema(self) -> ReferenceCounterfactualQuantitySchema:
        """Return the exactly shared quantity availability."""
        return self.same_side_pass.quantity_schema

    def npz_arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return numeric and fixed-width identity arrays for recomputation."""
        same_rows = self.same_side_pass.rows
        opposite_rows = self.opposite_side_pass.rows
        arrays: dict[str, np.ndarray[Any, Any]] = {
            "valid_mask": self.same_side_pass.valid_mask.copy(),
            "scene_ids": np.asarray([row.key.scene_id for row in same_rows]),
            "view_camera_ids": np.asarray(
                [row.key.view_camera_ids for row in same_rows]
            ),
            "local_ordering": np.asarray([row.key.local_ordering for row in same_rows]),
            "window_start": np.asarray(
                [row.window_start for row in same_rows], dtype=np.int64
            ),
            "window_stop": np.asarray(
                [row.window_stop for row in same_rows], dtype=np.int64
            ),
            "same_side_reference_camera_id": np.asarray(
                [row.reference_camera_id for row in same_rows]
            ),
            "opposite_side_reference_camera_id": np.asarray(
                [row.reference_camera_id for row in opposite_rows]
            ),
            "same_side_reference_view_index": np.asarray(
                [row.reference_view_index for row in same_rows], dtype=np.int64
            ),
            "opposite_side_reference_view_index": np.asarray(
                [row.reference_view_index for row in opposite_rows], dtype=np.int64
            ),
            "same_side_reference_from_physical": np.asarray(
                [row.provenance.reference_from_physical for row in same_rows],
                dtype=np.float64,
            ),
            "opposite_side_reference_from_physical": np.asarray(
                [row.provenance.reference_from_physical for row in opposite_rows],
                dtype=np.float64,
            ),
            "same_side_physical_from_reference": np.asarray(
                [row.provenance.physical_from_reference for row in same_rows],
                dtype=np.float64,
            ),
            "opposite_side_physical_from_reference": np.asarray(
                [row.provenance.physical_from_reference for row in opposite_rows],
                dtype=np.float64,
            ),
        }
        for field in _PARITY_FIELDS:
            arrays[field] = np.asarray([getattr(row, field) for row in same_rows])
        for side, pass_value in (
            ("same_side", self.same_side_pass),
            ("opposite_side", self.opposite_side_pass),
        ):
            for name, value in (
                ("position", pass_value.position),
                ("vector", pass_value.vector),
                ("heading", pass_value.heading),
                ("world_joints", pass_value.world_joints),
            ):
                if value is not None:
                    arrays[f"{side}_{name}_prediction"] = value.prediction.copy()
                    arrays[f"{side}_{name}_target"] = value.target.copy()
        return arrays

    @property
    def arrays_digest(self) -> str:
        """Return a stable digest independent of NPZ zip metadata."""
        return array_payload_sha256(self.npz_arrays())

    def core_document(self) -> dict[str, JsonValue]:
        """Return report JSON excluding only the self-authenticating digest."""
        return {
            "schema_version": self.schema_version,
            "artifact_type": "reference_counterfactual_paired_report",
            "manifest": self.manifest.to_dict(),
            "identity": self.identity.to_dict(),
            "quantity_schema": self.quantity_schema.to_dict(),
            "passes": {
                "same_side": self.same_side_pass.metadata_dict(),
                "opposite_side": self.opposite_side_pass.metadata_dict(),
            },
            "metrics": self.metrics.to_dict(),
            "parity_digest": self.parity_digest,
            "arrays_digest": self.arrays_digest,
        }

    @property
    def report_digest(self) -> str:
        """Return the stable digest over report metadata and array content digest."""
        return canonical_json_sha256(self.core_document())

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the complete JSON document."""
        result = self.core_document()
        result["report_digest"] = self.report_digest
        return result


def evaluate_reference_counterfactual(
    manifest: ReferenceCounterfactualManifest,
    same_side: ReferenceCounterfactualPass,
    opposite_side: ReferenceCounterfactualPass,
) -> ReferenceCounterfactualReport:
    """Strictly join two passes and compute frame-correct aggregate metrics."""
    if same_side.side != "same_side" or opposite_side.side != "opposite_side":
        raise ReferenceCounterfactualError(
            "Counterfactual join requires explicitly labelled same/opposite passes."
        )
    if same_side.identity != opposite_side.identity:
        raise ReferenceCounterfactualError(
            "Seed, resolved config, checkpoint, selector mode, contracts, or manifest "
            "differs across the two reference passes."
        )
    if same_side.quantity_schema != opposite_side.quantity_schema:
        raise ReferenceCounterfactualError(
            "Quantity availability differs across reference passes."
        )
    if len(same_side.rows) != len(opposite_side.rows):
        raise ReferenceCounterfactualError(
            "Same/opposite passes have unequal row counts or a missing side."
        )
    _validate_pass_against_manifest(same_side, manifest)
    _validate_pass_against_manifest(opposite_side, manifest)
    if not np.array_equal(same_side.valid_mask, opposite_side.valid_mask):
        raise ReferenceCounterfactualError(
            "Frame validity/padding masks differ across reference passes."
        )
    for index, (first, second) in enumerate(
        zip(same_side.rows, opposite_side.rows, strict=True)
    ):
        if first.key != second.key:
            raise ReferenceCounterfactualError(
                f"Pair key/order mismatch at joined row {index}."
            )
        if (first.window_start, first.window_stop) != (
            second.window_start,
            second.window_stop,
        ):
            raise ReferenceCounterfactualError(
                f"Window mismatch at joined row {index}."
            )
        for field in _PARITY_FIELDS:
            if getattr(first, field) != getattr(second, field):
                raise ReferenceCounterfactualError(
                    f"{field} mismatch at joined row {index}; only reference may change."
                )
    for first_quantity, second_quantity in (
        (same_side.position, opposite_side.position),
        (same_side.vector, opposite_side.vector),
        (same_side.heading, opposite_side.heading),
        (same_side.world_joints, opposite_side.world_joints),
    ):
        if (first_quantity is None) != (second_quantity is None):
            raise ReferenceCounterfactualError(
                "A quantity is missing from exactly one reference side."
            )
        if (
            first_quantity is not None
            and second_quantity is not None
            and (
                first_quantity.prediction.shape != second_quantity.prediction.shape
                or first_quantity.target.shape != second_quantity.target.shape
            )
        ):
            raise ReferenceCounterfactualError(
                "Quantity shapes/count/order differ across reference passes."
            )
    _validate_target_consistency(same_side, opposite_side)
    consistency = ReferencePhysicalConsistencyMetrics(
        position_error_m=_mean_physical_consistency(
            same_side.position,
            same_side.rows,
            opposite_side.position,
            opposite_side.rows,
            valid_mask=same_side.valid_mask,
        ),
        vector_error_m=(
            _mean_physical_consistency(
                same_side.vector,
                same_side.rows,
                opposite_side.vector,
                opposite_side.rows,
                valid_mask=same_side.valid_mask,
            )
            if same_side.vector is not None and opposite_side.vector is not None
            else None
        ),
        heading_error=(
            _mean_physical_consistency(
                same_side.heading,
                same_side.rows,
                opposite_side.heading,
                opposite_side.rows,
                valid_mask=same_side.valid_mask,
            )
            if same_side.heading is not None and opposite_side.heading is not None
            else None
        ),
        world_joints_error_m=(
            _mean_physical_consistency(
                same_side.world_joints,
                same_side.rows,
                opposite_side.world_joints,
                opposite_side.rows,
                valid_mask=same_side.valid_mask,
            )
            if same_side.world_joints is not None
            and opposite_side.world_joints is not None
            else None
        ),
    )
    parity_digest = canonical_json_sha256(
        _pair_parity_value(
            same_side.identity,
            same_side.quantity_schema,
            same_side.rows,
            opposite_side.rows,
        )
    )
    return ReferenceCounterfactualReport(
        schema_version=REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION,
        manifest=manifest,
        same_side_pass=same_side,
        opposite_side_pass=opposite_side,
        metrics=ReferenceCounterfactualMetrics(
            same_side=_target_frame_metrics(same_side),
            opposite_side=_target_frame_metrics(opposite_side),
            physical_consistency=consistency,
        ),
        parity_digest=parity_digest,
    )


def array_payload_sha256(
    arrays: Mapping[str, np.ndarray[Any, Any]],
) -> str:
    """Digest named arrays by canonical name, dtype, shape, and value bytes.

    Byte order is normalized to little-endian before hashing.  Object arrays
    and non-finite numeric values are rejected so parity cannot depend on
    pickle behavior or silently ignored invalid observations.
    """
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping of names to numpy arrays.")
    if not arrays:
        raise ReferenceCounterfactualError("Array payload mapping cannot be empty.")
    descriptors: list[JsonValue] = []
    for name in sorted(arrays):
        if type(name) is not str or not name.strip():
            raise ReferenceCounterfactualError(
                "Array payload names must be non-empty exact strings."
            )
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Array payload {name!r} must be a numpy.ndarray.")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ReferenceCounterfactualError(
                f"Array payload {name!r} cannot use an object dtype."
            )
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ReferenceCounterfactualError(
                f"Array payload {name!r} cannot contain non-finite values."
            )
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
        payload_digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
        descriptors.append(
            {
                "name": name,
                "dtype": canonical_dtype.str,
                "shape": list(array.shape),
                "payload_digest": payload_digest,
            }
        )
    return canonical_json_sha256(descriptors)


def validate_reference_counterfactual_raw_payload(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    *,
    task: ReferenceCounterfactualTask,
) -> int:
    """Validate shared raw-pass identity tensors before task adaptation."""
    if task not in ("blcs", "plcs"):
        raise ReferenceCounterfactualError(f"Unknown raw payload task {task!r}.")
    # Hashing is also the fail-closed whole-payload finite/object check. The
    # returned digest is intentionally not used as pair identity because Court
    # and reference-frame targets legitimately differ between the two sides.
    array_payload_sha256(arrays)
    required_shapes: dict[str, tuple[int | None, ...]] = {
        "scene_ids": (None,),
        "view_camera_id_strings": (None, REFERENCE_COUNTERFACTUAL_NUM_VIEWS),
        "reference_camera_id_string": (None,),
        "reference_view_index": (None,),
        "reference_camera_id": (None,),
        "view_camera_ids": (None, REFERENCE_COUNTERFACTUAL_NUM_VIEWS),
        "reference_from_physical": (None, 3, 3),
        "physical_from_reference": (None, 3, 3),
    }
    try:
        scene_ids = arrays["scene_ids"]
    except KeyError as error:
        raise ReferenceCounterfactualError(
            f"{task.upper()} raw prediction payload is missing 'scene_ids'."
        ) from error
    if not isinstance(scene_ids, np.ndarray) or scene_ids.ndim != 1:
        raise ReferenceCounterfactualError(
            f"{task.upper()} raw scene_ids must have exact shape (B,)."
        )
    batch_size = int(scene_ids.shape[0])
    if batch_size <= 0:
        raise ReferenceCounterfactualError(
            f"{task.upper()} raw prediction payload cannot be empty."
        )
    for name, shape_contract in required_shapes.items():
        try:
            value = arrays[name]
        except KeyError as error:
            raise ReferenceCounterfactualError(
                f"{task.upper()} raw prediction payload is missing {name!r}."
            ) from error
        expected_shape = tuple(
            batch_size if width is None else width for width in shape_contract
        )
        if not isinstance(value, np.ndarray) or value.shape != expected_shape:
            raise ReferenceCounterfactualError(
                f"{task.upper()} raw {name} must have exact shape {expected_shape}."
            )
    for name in ("scene_ids", "view_camera_id_strings", "reference_camera_id_string"):
        if not np.issubdtype(arrays[name].dtype, np.str_):
            raise ReferenceCounterfactualError(
                f"{task.upper()} raw {name} must use fixed-width unicode dtype."
            )
    for name in ("reference_view_index", "reference_camera_id", "view_camera_ids"):
        if arrays[name].dtype != np.int64:
            raise ReferenceCounterfactualError(
                f"{task.upper()} raw {name} must use exact int64 dtype."
            )
    for name in ("reference_from_physical", "physical_from_reference"):
        if not np.issubdtype(arrays[name].dtype, np.floating):
            raise ReferenceCounterfactualError(
                f"{task.upper()} raw {name} must use a floating dtype."
            )
    return batch_size


def masked_counterfactual_quantity_for_digest(
    value: np.ndarray[Any, Any],
    valid_mask: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Zero unsupervised cells and canonicalize signed zero for parity hashing."""
    if not isinstance(value, np.ndarray) or not np.issubdtype(
        value.dtype, np.number
    ):
        raise ReferenceCounterfactualError(
            "Counterfactual digest quantity must be a numeric numpy array."
        )
    if not isinstance(valid_mask, np.ndarray) or valid_mask.dtype != np.bool_:
        raise ReferenceCounterfactualError(
            "Counterfactual digest validity must be a bool numpy array."
        )
    if value.shape[: valid_mask.ndim] != valid_mask.shape:
        raise ReferenceCounterfactualError(
            "Counterfactual digest validity must match quantity leading axes."
        )
    broadcast_mask = valid_mask.reshape(
        valid_mask.shape + (1,) * (value.ndim - valid_mask.ndim)
    )
    result = np.where(broadcast_mask, value, np.zeros((), dtype=value.dtype))
    result[result == 0] = np.zeros((), dtype=result.dtype)
    return cast("np.ndarray[Any, Any]", np.ascontiguousarray(result))


@dataclass(frozen=True, slots=True)
class ReferenceCounterfactualReportPaths:
    """The standard queue files and full JSON forming one report bundle."""

    json_path: Path
    npz_path: Path
    metrics_path: Path


def _report_paths(
    output_dir: str | Path,
    *,
    stem: str,
) -> ReferenceCounterfactualReportPaths:
    if type(stem) is not str or re.fullmatch(r"[a-z][a-z0-9_-]*", stem) is None:
        raise ReferenceCounterfactualError(
            "Report stem must be a non-empty lowercase artifact name."
        )
    directory = Path(output_dir)
    return ReferenceCounterfactualReportPaths(
        json_path=directory / f"{stem}.json",
        npz_path=directory / REFERENCE_COUNTERFACTUAL_PREDICTIONS_FILENAME,
        metrics_path=directory / REFERENCE_COUNTERFACTUAL_METRICS_FILENAME,
    )


def _write_json_temp(path: Path, value: object) -> None:
    payload = json.dumps(
        _normalize_json(value, location="report"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_temp(
    path: Path,
    arrays: Mapping[str, np.ndarray[Any, Any]],
) -> None:
    with path.open("wb") as handle:
        save_npz = cast("Any", np.savez_compressed)
        save_npz(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def write_reference_counterfactual_report(
    report: ReferenceCounterfactualReport,
    output_dir: str | Path,
    *,
    stem: str = REFERENCE_COUNTERFACTUAL_REPORT_STEM,
) -> ReferenceCounterfactualReportPaths:
    """Publish standard NPZ/metrics plus full JSON, with no overwrite."""
    if not isinstance(report, ReferenceCounterfactualReport):
        raise TypeError("report must be ReferenceCounterfactualReport.")
    paths = _report_paths(output_dir, stem=stem)
    paths.json_path.parent.mkdir(parents=True, exist_ok=True)
    final_paths = (paths.npz_path, paths.json_path, paths.metrics_path)
    lock_path = paths.json_path.parent / f".{stem}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ReferenceCounterfactualError(
            f"Counterfactual report publication is already active or stale: {lock_path}."
        ) from error
    os.close(lock_fd)
    published: list[Path] = []
    temp_paths: list[Path] = []
    try:
        existing = tuple(path for path in final_paths if path.exists())
        if existing:
            raise ReferenceCounterfactualError(
                "Refusing to overwrite a stale, partial, or mixed counterfactual "
                f"report: {existing!r}."
            )
        json_fd, json_name = tempfile.mkstemp(
            prefix=f".{stem}.",
            suffix=".json.tmp",
            dir=paths.json_path.parent,
        )
        npz_fd, npz_name = tempfile.mkstemp(
            prefix=f".{stem}.",
            suffix=".npz.tmp",
            dir=paths.npz_path.parent,
        )
        metrics_fd, metrics_name = tempfile.mkstemp(
            prefix=f".{stem}.",
            suffix=".metrics.tmp",
            dir=paths.metrics_path.parent,
        )
        for descriptor in (json_fd, npz_fd, metrics_fd):
            os.close(descriptor)
        json_temp = Path(json_name)
        npz_temp = Path(npz_name)
        metrics_temp = Path(metrics_name)
        temp_paths.extend((json_temp, npz_temp, metrics_temp))

        arrays = report.npz_arrays()
        _write_npz_temp(npz_temp, arrays)
        _write_json_temp(json_temp, report.to_dict())
        _write_json_temp(metrics_temp, report.metrics.flat_dict())

        # Publish metrics last: queue discovery must never mistake a partial
        # pair for a registerable completed result. Hard links are an atomic
        # no-overwrite boundary on the output filesystem.
        for temporary, final in (
            (npz_temp, paths.npz_path),
            (json_temp, paths.json_path),
            (metrics_temp, paths.metrics_path),
        ):
            try:
                os.link(temporary, final)
            except FileExistsError as error:
                raise ReferenceCounterfactualError(
                    f"Refusing to overwrite counterfactual artifact {final}."
                ) from error
            published.append(final)
        directory_fd = os.open(paths.json_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)
    # A successful producer is also its first consumer: do not let a queue job
    # finish until all aggregates have been recomputed from the published NPZ
    # and checked against both JSON files.
    read_reference_counterfactual_report(output_dir, stem=stem)
    return paths


def _read_flat_metrics(path: Path) -> dict[str, float]:
    document = _read_json_mapping(path)
    if not document:
        raise ReferenceCounterfactualError(
            "Counterfactual metrics.json must contain flat numeric metrics."
    )
    result: dict[str, float] = {}
    for key, value in document.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReferenceCounterfactualError(
                "Counterfactual metrics.json values must be finite flat numbers; "
                f"{key!r} is invalid."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ReferenceCounterfactualError(
                "Counterfactual metrics.json values must be finite flat numbers; "
                f"{key!r} is invalid."
            )
        result[key] = numeric_value
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as error:
        raise ReferenceCounterfactualError(
            f"Invalid counterfactual NPZ report {path}: {error}."
        ) from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ReferenceCounterfactualError(
            "Counterfactual NPZ must not contain object arrays."
        )
    return arrays


def _required_array(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    name: str,
) -> np.ndarray[Any, Any]:
    try:
        return arrays[name]
    except KeyError as error:
        raise ReferenceCounterfactualError(
            f"Counterfactual NPZ is missing required array {name!r}."
        ) from error


def _quantity_from_npz(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    *,
    side: ReferenceCounterfactualSide,
    name: str,
    quantity: ReferenceTransformQuantity,
    present: bool,
) -> ReferenceCounterfactualQuantityArrays | None:
    prediction_key = f"{side}_{name}_prediction"
    target_key = f"{side}_{name}_target"
    has_prediction = prediction_key in arrays
    has_target = target_key in arrays
    if present != has_prediction or present != has_target:
        raise ReferenceCounterfactualError(
            f"NPZ {name} arrays do not match explicit quantity availability."
        )
    if not present:
        return None
    return ReferenceCounterfactualQuantityArrays(
        prediction=cast("FloatArray", arrays[prediction_key]),
        target=cast("FloatArray", arrays[target_key]),
        quantity=quantity,
    )


def _pass_from_report_document(
    value: object,
    arrays: Mapping[str, np.ndarray[Any, Any]],
    *,
    expected_side: ReferenceCounterfactualSide,
    location: str,
) -> ReferenceCounterfactualPass:
    mapping = _require_exact_fields(
        value,
        {"schema_version", "side", "identity", "quantity_schema", "rows"},
        location=location,
    )
    if mapping["side"] != expected_side:
        raise ReferenceCounterfactualError(
            f"{location}.side must be {expected_side!r}."
        )
    raw_rows = mapping["rows"]
    if not isinstance(raw_rows, list):
        raise ReferenceCounterfactualError(f"{location}.rows must be a list.")
    identity = ReferenceCounterfactualRunIdentity.from_dict(
        mapping["identity"], location=f"{location}.identity"
    )
    schema = ReferenceCounterfactualQuantitySchema.from_dict(
        mapping["quantity_schema"], location=f"{location}.quantity_schema"
    )
    position = _quantity_from_npz(
        arrays,
        side=expected_side,
        name="position",
        quantity="point",
        present=True,
    )
    if position is None:
        raise ReferenceCounterfactualError("Position arrays are always required.")
    return ReferenceCounterfactualPass(
        schema_version=cast("int", mapping["schema_version"]),
        side=expected_side,
        identity=identity,
        quantity_schema=schema,
        rows=tuple(
            ReferenceCounterfactualPassRow.from_dict(
                row,
                location=f"{location}.rows[{index}]",
            )
            for index, row in enumerate(raw_rows)
        ),
        valid_mask=cast("BoolArray", _required_array(arrays, "valid_mask")),
        position=position,
        vector=_quantity_from_npz(
            arrays,
            side=expected_side,
            name="vector",
            quantity="vector",
            present=schema.vector,
        ),
        heading=_quantity_from_npz(
            arrays,
            side=expected_side,
            name="heading",
            quantity="heading",
            present=schema.heading,
        ),
        world_joints=_quantity_from_npz(
            arrays,
            side=expected_side,
            name="world_joints",
            quantity="world_joints",
            present=schema.world_joints,
        ),
    )


def _assert_exact_npz_arrays(
    loaded: Mapping[str, np.ndarray[Any, Any]],
    expected: Mapping[str, np.ndarray[Any, Any]],
) -> None:
    if set(loaded) != set(expected):
        raise ReferenceCounterfactualError(
            "Counterfactual NPZ field set is stale or mixed: "
            f"expected {sorted(expected)!r}, got {sorted(loaded)!r}."
        )
    for name in sorted(expected):
        actual = loaded[name]
        wanted = expected[name]
        if (
            actual.dtype != wanted.dtype
            or actual.shape != wanted.shape
            or not np.array_equal(actual, wanted)
        ):
            raise ReferenceCounterfactualError(
                f"Counterfactual NPZ metadata/content mismatch in {name!r}."
            )


def read_reference_counterfactual_report(
    output_dir: str | Path,
    *,
    stem: str = REFERENCE_COUNTERFACTUAL_REPORT_STEM,
) -> ReferenceCounterfactualReport:
    """Read, validate, and recompute a paired report before returning it."""
    paths = _report_paths(output_dir, stem=stem)
    presence = (
        paths.json_path.is_file(),
        paths.npz_path.is_file(),
        paths.metrics_path.is_file(),
    )
    if presence != (True, True, True):
        raise ReferenceCounterfactualError(
            "Counterfactual report is missing one or more required files: "
            f"json={presence[0]}, npz={presence[1]}, metrics={presence[2]}."
        )
    document = _read_json_mapping(paths.json_path)
    expected_fields = {
        "schema_version",
        "artifact_type",
        "manifest",
        "identity",
        "quantity_schema",
        "passes",
        "metrics",
        "parity_digest",
        "arrays_digest",
        "report_digest",
    }
    mapping = _require_exact_fields(document, expected_fields, location="report")
    if mapping["schema_version"] != REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION:
        raise ReferenceCounterfactualError("Counterfactual report schema is stale.")
    if mapping["artifact_type"] != "reference_counterfactual_paired_report":
        raise ReferenceCounterfactualError(
            "Counterfactual report artifact type is unknown."
        )
    arrays = _load_npz(paths.npz_path)
    if array_payload_sha256(arrays) != _require_digest(
        mapping["arrays_digest"], location="report.arrays_digest"
    ):
        raise ReferenceCounterfactualError(
            "Counterfactual NPZ content digest does not match JSON metadata."
        )
    manifest = ReferenceCounterfactualManifest.from_dict(
        mapping["manifest"], location="report.manifest"
    )
    raw_passes = _require_exact_fields(
        mapping["passes"],
        {"same_side", "opposite_side"},
        location="report.passes",
    )
    same_side = _pass_from_report_document(
        raw_passes["same_side"],
        arrays,
        expected_side="same_side",
        location="report.passes.same_side",
    )
    opposite_side = _pass_from_report_document(
        raw_passes["opposite_side"],
        arrays,
        expected_side="opposite_side",
        location="report.passes.opposite_side",
    )
    report = evaluate_reference_counterfactual(manifest, same_side, opposite_side)
    _assert_exact_npz_arrays(arrays, report.npz_arrays())
    flat_metrics = _read_flat_metrics(paths.metrics_path)
    if flat_metrics != report.metrics.flat_dict():
        raise ReferenceCounterfactualError(
            "Saved flat metrics do not match recomputation from paired arrays."
        )
    if mapping["identity"] != report.identity.to_dict():
        raise ReferenceCounterfactualError("Outer and pass run identities differ.")
    if mapping["quantity_schema"] != report.quantity_schema.to_dict():
        raise ReferenceCounterfactualError("Outer and pass quantity schemas differ.")
    if mapping["metrics"] != report.metrics.to_dict():
        raise ReferenceCounterfactualError(
            "Saved metrics do not match recomputation from paired arrays."
        )
    if mapping["parity_digest"] != report.parity_digest:
        raise ReferenceCounterfactualError("Saved parity digest is stale.")
    stored_report_digest = _require_digest(
        mapping["report_digest"], location="report.report_digest"
    )
    unsigned = dict(document)
    del unsigned["report_digest"]
    if canonical_json_sha256(unsigned) != stored_report_digest:
        raise ReferenceCounterfactualError("Saved report digest does not match JSON.")
    if stored_report_digest != report.report_digest:
        raise ReferenceCounterfactualError(
            "Saved report digest does not match recomputed report content."
        )
    return report


__all__ = [
    "REFERENCE_COUNTERFACTUAL_METRICS_FILENAME",
    "REFERENCE_COUNTERFACTUAL_NUM_VIEWS",
    "REFERENCE_COUNTERFACTUAL_PREDICTIONS_FILENAME",
    "REFERENCE_COUNTERFACTUAL_REPORT_STEM",
    "REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION",
    "ReferenceCounterfactualError",
    "ReferenceCounterfactualManifest",
    "ReferenceCounterfactualMetrics",
    "ReferenceCounterfactualPass",
    "ReferenceCounterfactualPassRow",
    "ReferenceCounterfactualQuantityArrays",
    "ReferenceCounterfactualQuantitySchema",
    "ReferenceCounterfactualReport",
    "ReferenceCounterfactualReportPaths",
    "ReferenceCounterfactualRunIdentity",
    "ReferenceCounterfactualScene",
    "ReferenceCounterfactualSelectorMode",
    "ReferenceCounterfactualSide",
    "ReferenceCounterfactualTask",
    "ReferencePhysicalConsistencyMetrics",
    "ReferenceSideSelection",
    "ReferenceTargetFrameMetrics",
    "array_payload_sha256",
    "build_reference_counterfactual_manifest",
    "build_reference_counterfactual_manifest_from_documents",
    "canonical_json_sha256",
    "canonical_json_text",
    "evaluate_reference_counterfactual",
    "file_sha256",
    "masked_counterfactual_quantity_for_digest",
    "read_reference_counterfactual_report",
    "validate_reference_counterfactual_raw_payload",
    "write_reference_counterfactual_report",
]

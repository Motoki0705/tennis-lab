"""PLCS adapters for the shared CourtKP20 semantic and frame contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.data import (
    ReferenceViewSelection,
    StableCameraIdTable,
    resolve_evaluation_reference_camera_id,
    select_seeded_training_reference_camera_id,
)
from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    PHYSICAL_V1_SELECTOR,
    CameraCourtViewError,
    CourtKeypointArtifactMetadata,
    CourtKeypointContract,
    CourtKeypointContractMetadata,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    DatasetCourtKeypointContract,
    InvalidCourtKeypointMetadataError,
    align_court_keypoints_to_reference,
    build_court_view_record,
    court_headings_physical_to_target,
    court_headings_target_to_physical,
    court_points_physical_to_target,
    court_points_target_to_physical,
    resolve_court_keypoint_contract,
    validate_dataset_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    TRACK_QUERY_REFERENCE_METADATA_KEY,
    TrackQueryReferenceContract,
    TrackQueryReferenceContractMetadata,
)
from src.tasks.base.models import resolve_reference_selector_mode
from src.utils.configuration import (
    ConfigurationTypeError,
    UnknownConfigurationKeyError,
)
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
)

PLCS_GENERATED_DATASET_SCHEMA_ID: Final = "plcs_generated_dataset_v2"


@dataclass(frozen=True, slots=True)
class PLCSCourtKeypointRuntimeConfig:
    """Exact task runtime selection of one shared CourtKP20 contract."""

    contract: CourtKeypointContract

    @classmethod
    def from_config(
        cls,
        value: object,
    ) -> PLCSCourtKeypointRuntimeConfig:
        """Parse ``court_keypoints.selector`` without inferring from arrays."""
        root = as_config_mapping(value, path="configuration")
        if "court_keypoints" not in root:
            raise ConfigurationTypeError(
                "Required config key 'configuration.court_keypoints' was not composed."
            )
        section = require_config_mapping(root, "court_keypoints", path="configuration")
        unknown = sorted(set(section) - {"selector"})
        if unknown:
            raise UnknownConfigurationKeyError(
                "Unknown configuration key(s): "
                + ", ".join(f"court_keypoints.{key}" for key in unknown)
            )
        if set(section) != {"selector"} or type(section["selector"]) is not str:
            raise ConfigurationTypeError(
                "court_keypoints must contain exactly one string field: selector."
            )
        return cls(resolve_court_keypoint_contract(section["selector"]))


def court_keypoint_contract_document(
    contract: CourtKeypointContract,
) -> dict[str, object]:
    """Build the exact direct-input/checkpoint contract document."""
    return {
        "court_keypoints": CourtKeypointContractMetadata.from_contract(
            contract
        ).to_dict()
    }


def track_query_reference_contract_document(
    value: object,
    contract: CourtKeypointContract,
) -> dict[str, object] | None:
    """Build the exact v2 model-ready semantic marker from composed config.

    Non-reference model families return ``None`` and retain their existing I/O
    contracts. Reference track-query profiles must declare every independent
    target/RoPE/selector marker; values are never inferred from the CourtKP20
    array shape.
    """
    root = as_config_mapping(value, path="configuration")
    model = require_config_mapping(root, "model", path="configuration")
    model_name = model.get("name")
    if model_name not in {
        "plcs_track_query_reference",
        "plcs_track_query_reference_ablation",
    }:
        return None
    for key in (
        "target_frame_contract",
        "track_query_rope_contract",
        "reference_selector_mode",
    ):
        if type(model.get(key)) is not str:
            raise ConfigurationTypeError(
                f"model.{key} must be an explicit string for reference "
                "track-query data."
            )
    runtime = TrackQueryReferenceContract.reference_v2(
        resolve_reference_selector_mode(cast(str, model["reference_selector_mode"]))
    )
    actual = (
        contract.contract_id,
        cast(str, model["target_frame_contract"]),
        cast(str, model["track_query_rope_contract"]),
    )
    expected = (
        runtime.court_keypoint_contract,
        runtime.target_frame_contract,
        runtime.track_query_rope_contract.value,
    )
    if actual != expected:
        raise CourtKeypointContractMismatchError(
            "PLCS reference track-query data contract does not match composed "
            f"court/target/RoPE markers: expected {expected!r}, got {actual!r}."
        )
    return {
        TRACK_QUERY_REFERENCE_METADATA_KEY: (
            TrackQueryReferenceContractMetadata.from_contract(runtime).to_dict()
        )
    }


def plcs_artifact_metadata(
    contract: CourtKeypointContract,
) -> CourtKeypointArtifactMetadata:
    """Build the task-qualified generated-dataset metadata record."""
    return CourtKeypointArtifactMetadata.from_contract(
        contract,
        dataset_schema_id=PLCS_GENERATED_DATASET_SCHEMA_ID,
    )


def resolve_split_scene_paths(
    scene_dir: str | Path,
    split_file: str | Path,
) -> tuple[Path, ...]:
    """Resolve split entries before a dataset indexes any scene arrays."""
    root = Path(scene_dir)
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = root / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    scenes_root = root / "scenes"
    if not scenes_root.is_dir():
        scenes_root = root
    return tuple(
        scenes_root / name
        for raw in split_path.read_text(encoding="utf-8").splitlines()
        if (name := raw.strip())
    )


def validate_plcs_dataset_court_keypoints(
    scene_dir: str | Path,
    split_file: str | Path,
    contract: CourtKeypointContract,
) -> DatasetCourtKeypointContract:
    """Validate PLCS root/scene/camera metadata before payload access."""
    scene_paths = resolve_split_scene_paths(scene_dir, split_file)
    validation = validate_dataset_court_keypoint_contract(
        scene_dir,
        contract,
        expected_dataset_schema_id=PLCS_GENERATED_DATASET_SCHEMA_ID,
        scene_paths=scene_paths,
    )
    validate_plcs_court_keypoint_headers(validation, scene_paths)
    return validation


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InvalidCourtKeypointMetadataError(
            f"Required PLCS camera header does not exist: {path}."
        ) from error
    except json.JSONDecodeError as error:
        raise InvalidCourtKeypointMetadataError(
            f"{path}: invalid JSON camera header: {error}."
        ) from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InvalidCourtKeypointMetadataError(
            f"{path}: expected a JSON object with string keys."
        )
    return value


def validate_plcs_court_keypoint_headers(
    validation: DatasetCourtKeypointContract,
    scene_paths: Sequence[Path],
) -> None:
    """Cross-check Court records against PLCS scene/scalar camera headers."""
    if validation.legacy_metadata_free:
        return
    records_by_scene = {scene.scene_id: scene.court_views for scene in validation.scenes}
    for scene_path in scene_paths:
        views = records_by_scene[scene_path.name]
        meta_path = scene_path / "meta.json"
        scalars_path = scene_path / "scalars.json"
        meta = _load_json_object(meta_path)
        scalars = _load_json_object(scalars_path)
        for location, value in (
            (f"{meta_path}.num_cameras", meta.get("num_cameras")),
            (f"{scalars_path}.num_cameras", scalars.get("num_cameras")),
        ):
            if type(value) is not int or value != len(views):
                raise CourtKeypointContractMismatchError(
                    f"{location}: expected {len(views)} cameras from CourtKP20 "
                    f"metadata, got {value!r}."
                )
        expected_parameter_slots = {
            f"cam_{camera_index}_params" for camera_index in range(len(views))
        }
        actual_parameter_slots = {
            key
            for key in scalars
            if key.startswith("cam_") and key.endswith("_params")
        }
        if actual_parameter_slots != expected_parameter_slots:
            raise CourtKeypointContractMismatchError(
                f"{scalars_path}: camera parameter slots must exactly match "
                f"court_keypoint_views IDs; expected "
                f"{sorted(expected_parameter_slots)!r}, got "
                f"{sorted(actual_parameter_slots)!r}."
            )
        for camera_index, view in enumerate(views):
            expected_camera_id = f"camera_{camera_index}"
            if view.camera_id != expected_camera_id:
                raise CourtKeypointContractMismatchError(
                    f"{scalars_path}: camera slot {camera_index} requires stable ID "
                    f"{expected_camera_id!r}; got {view.camera_id!r}."
                )
            location = f"{scalars_path}.cam_{camera_index}_params"
            params = scalars.get(f"cam_{camera_index}_params")
            if not isinstance(params, Mapping):
                raise InvalidCourtKeypointMetadataError(
                    f"{location}: expected a camera parameter object."
                )
            try:
                parameter_view = build_court_view_record(
                    camera_id=expected_camera_id,
                    camera_center_court_m=params.get("C"),
                    contract=view.contract,
                )
            except CameraCourtViewError as error:
                raise InvalidCourtKeypointMetadataError(
                    f"{location}.C: expected three finite physical-metre values."
                ) from error
            center = parameter_view.camera_center_court_m
            persisted = view.camera_center_court_m
            if center != persisted:
                raise CourtKeypointContractMismatchError(
                    f"{location}.C: {list(center)!r} does not match CourtKP20 "
                    f"camera metadata {list(persisted)!r}."
                )


def scene_court_views(
    validation: DatasetCourtKeypointContract,
    scene_path: str | Path,
) -> tuple[CourtViewRecord, ...]:
    """Return validated ordered camera records for one scene path."""
    scene_id = Path(scene_path).name
    for scene in validation.scenes:
        if scene.scene_id == scene_id:
            views: tuple[CourtViewRecord, ...] = scene.court_views
            return views
    raise CourtKeypointContractMismatchError(
        f"Validated CourtKP20 metadata has no scene {scene_id!r}."
    )


def selected_court_views(
    validation: DatasetCourtKeypointContract,
    scene_path: str | Path,
    camera_indices: Sequence[int],
) -> tuple[CourtViewRecord, ...]:
    """Resolve selected records while preserving stable scene camera IDs."""
    views = scene_court_views(validation, scene_path)
    if validation.legacy_metadata_free:
        return ()
    selected: list[CourtViewRecord] = []
    for index in camera_indices:
        if index < 0 or index >= len(views):
            raise CourtKeypointContractMismatchError(
                f"Camera index {index} is outside validated CourtKP20 metadata "
                f"capacity {len(views)} for scene {Path(scene_path).name!r}."
            )
        selected.append(views[index])
    return tuple(selected)


def choose_reference_selection(
    contract: CourtKeypointContract,
    complete_scene_views: Sequence[CourtViewRecord],
    selected_views: Sequence[CourtViewRecord],
    *,
    rng: np.random.Generator | None,
    requested_camera_id: str | None,
) -> ReferenceViewSelection | None:
    """Compose the sole v2 selection after camera subset/order resolution.

    Training passes its seeded worker generator. Evaluation/inference passes an
    explicit canonical identity; only single-view evaluation may omit it.  The
    collision-free tensor codec is always built from the complete scene table,
    never from the selected subset.
    """
    if contract.selector == PHYSICAL_V1_SELECTOR:
        if requested_camera_id is not None:
            raise CourtKeypointContractMismatchError(
                "physical_v1 must not specify a reference camera ID."
            )
        return None
    if rng is not None and requested_camera_id is not None:
        raise CourtKeypointContractMismatchError(
            "Training reference sampling and an explicit evaluation reference "
            "camera are mutually exclusive."
        )
    complete_views = tuple(complete_scene_views)
    views = tuple(selected_views)
    if not complete_views or not views:
        raise CourtKeypointContractMismatchError(
            "camera_view_v2 requires persisted per-camera CourtKP20 metadata."
        )
    complete_ids = tuple(view.camera_id for view in complete_views)
    selected_ids = tuple(view.camera_id for view in views)
    stable_table = StableCameraIdTable.from_complete_scene_camera_ids(complete_ids)
    reference_id = (
        select_seeded_training_reference_camera_id(selected_ids, rng=rng)
        if rng is not None
        else resolve_evaluation_reference_camera_id(
            selected_ids,
            requested_camera_id=requested_camera_id,
        )
    )
    return ReferenceViewSelection.create(
        stable_camera_id_table=stable_table,
        selected_views=views,
        reference_camera_id=reference_id,
    )


def align_selected_court_array(
    value: np.ndarray,
    source_view: CourtViewRecord | None,
    reference_view: CourtViewRecord | None,
    *,
    keypoint_axis: int,
) -> np.ndarray:
    """Align v2 Court slots; physical-v1 arrays remain byte-for-byte ordered."""
    if reference_view is None and (
        source_view is None or source_view.contract.selector == PHYSICAL_V1_SELECTOR
    ):
        return value
    if source_view is None or reference_view is None:
        raise CourtKeypointContractMismatchError(
            "CourtKP20 source/reference metadata must either both exist or both be absent."
        )
    return cast(
        np.ndarray,
        align_court_keypoints_to_reference(
            value,
            source_view,
            reference_view,
            keypoint_axis=keypoint_axis,
        ),
    )


def normalized_points_physical_to_target(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor:
    """Rotate fixed-contract normalized points through physical metres."""
    physical = denormalize_court_position(value)
    if not isinstance(physical, Tensor):
        raise TypeError("PLCS normalization must preserve torch.Tensor values.")
    target = court_points_physical_to_target(physical, provenance)
    if not isinstance(target, Tensor):
        raise TypeError("PLCS Court transform must preserve torch.Tensor values.")
    normalized = normalize_court_position(target)
    if not isinstance(normalized, Tensor):
        raise TypeError("PLCS normalization must preserve torch.Tensor values.")
    return normalized


def normalized_points_target_to_physical(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor:
    """Return normalized target-frame points to physical court metres."""
    target_m = denormalize_court_position(value)
    if not isinstance(target_m, Tensor):
        raise TypeError("PLCS normalization must preserve torch.Tensor values.")
    physical = court_points_target_to_physical(target_m, provenance)
    if not isinstance(physical, Tensor):
        raise TypeError("PLCS Court transform must preserve torch.Tensor values.")
    return physical


def normalized_headings_physical_to_target(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor:
    """Rotate PLCS ``(cos(yaw), sin(yaw))`` supervision."""
    transformed = court_headings_physical_to_target(value, provenance)
    if not isinstance(transformed, Tensor):
        raise TypeError("PLCS Court transform must preserve torch.Tensor values.")
    return transformed


def headings_target_to_physical(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor:
    """Return target-frame heading vectors to physical court orientation."""
    transformed = court_headings_target_to_physical(value, provenance)
    if not isinstance(transformed, Tensor):
        raise TypeError("PLCS Court transform must preserve torch.Tensor values.")
    return transformed


def world_joints_physical_to_target(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor:
    """Rotate court-space joints without touching player-local canonical pose."""
    transformed = court_points_physical_to_target(value, provenance)
    if not isinstance(transformed, Tensor):
        raise TypeError("PLCS Court transform must preserve torch.Tensor values.")
    return transformed


def provenance_from_value(
    value: object,
    *,
    location: str,
) -> CourtReferenceFrameProvenance:
    """Parse one provenance object or exact serialized mapping."""
    if isinstance(value, CourtReferenceFrameProvenance):
        return value
    return CourtReferenceFrameProvenance.from_mapping(value, location=location)


def validate_provenance_contract(
    provenance: CourtReferenceFrameProvenance,
    contract: CourtKeypointContract,
    *,
    location: str,
) -> None:
    """Reject prediction/sample provenance from another semantic contract."""
    if provenance.contract != contract:
        raise CourtKeypointContractMismatchError(
            f"{location}: provenance contract {provenance.contract_id!r} does not "
            f"match runtime {contract.contract_id!r}."
        )


def contract_requires_provenance(contract: CourtKeypointContract) -> bool:
    """Return whether identity cannot be assumed at a model boundary."""
    return bool(contract.selector == CAMERA_VIEW_V2_SELECTOR)

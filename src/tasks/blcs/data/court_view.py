"""BLCS dataset adapters for the shared CourtKP20 frame contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.data import (
    CAMERA_ID_PADDING_VALUE,
    ReferenceViewSelection,
    StableCameraIdTable,
    resolve_evaluation_reference_camera_id,
    select_seeded_training_reference_camera_id,
    validate_reference_view_batch,
)
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    DatasetCourtKeypointContract,
    align_court_keypoints_to_reference,
    build_physical_court_provenance,
)
from src.tasks.base.model_io import (
    TRACK_QUERY_REFERENCE_METADATA_KEY,
    TrackQueryReferenceContract,
    TrackQueryReferenceContractMetadata,
)
from src.tasks.base.models import resolve_reference_selector_mode
from src.tasks.blcs.generate_dataset.io.dataset_io import (
    validate_blcs_dataset_court_keypoints as validate_blcs_artifact,
)


@dataclass(frozen=True, slots=True)
class BLCSSampleCourtFrame:
    """Validated selected views plus one reversible target-frame provenance."""

    selected_views: tuple[CourtViewRecord, ...]
    reference_view: CourtViewRecord | None
    provenance: CourtReferenceFrameProvenance
    reference_selection: ReferenceViewSelection | None


_REFERENCE_SAMPLE_FIELDS = frozenset(
    {
        "reference_view_selection",
        "stable_camera_id_table",
        "reference_camera_id_string",
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
    }
)


def blcs_track_query_reference_contract_document(
    value: object,
    contract: CourtKeypointContract,
) -> dict[str, object] | None:
    """Build exact v2 data markers from an explicitly composed model config."""
    root = as_config_mapping(value, path="configuration")
    model = require_config_mapping(root, "model", path="configuration")
    model_name = model.get("name")
    if model_name not in {
        "blcs_track_query_reference",
        "blcs_track_query_reference_ablation",
    }:
        return None
    required = (
        "target_frame_contract",
        "track_query_rope_contract",
        "reference_selector_mode",
    )
    invalid = [key for key in required if type(model.get(key)) is not str]
    if invalid:
        raise TypeError(
            "BLCS reference track-query model markers must be explicit strings; "
            f"invalid fields are {invalid!r}."
        )
    runtime = TrackQueryReferenceContract.reference_v2(
        resolve_reference_selector_mode(
            cast("str", model["reference_selector_mode"])
        )
    )
    actual = (
        contract.contract_id,
        cast("str", model["target_frame_contract"]),
        cast("str", model["track_query_rope_contract"]),
    )
    expected = (
        runtime.court_keypoint_contract,
        runtime.target_frame_contract,
        runtime.track_query_rope_contract.value,
    )
    if actual != expected:
        raise ValueError(
            "BLCS reference track-query data contract does not match composed "
            f"court/target/RoPE markers: expected {expected!r}, got {actual!r}."
        )
    return {
        TRACK_QUERY_REFERENCE_METADATA_KEY: (
            TrackQueryReferenceContractMetadata.from_contract(runtime).to_dict()
        )
    }


def blcs_reference_sample_fields(
    selection: ReferenceViewSelection,
    *,
    dtype: torch.dtype,
    track_query_reference_document: dict[str, object] | None,
) -> dict[str, object]:
    """Materialize one v2 selection without independently deriving any field."""
    fields: dict[str, object] = {
        "reference_view_selection": selection,
        "stable_camera_id_table": selection.stable_camera_id_table,
        "reference_camera_id_string": selection.reference_camera_id,
        **selection.to_tensor_fields(dtype=dtype),
        "physical_from_reference": torch.tensor(
            selection.provenance.physical_from_reference,
            dtype=dtype,
        ),
    }
    if track_query_reference_document is not None:
        fields.update(track_query_reference_document)
    return fields


def collate_blcs_reference_fields(
    batch: list[dict[str, Any]],
    *,
    max_views: int,
    model_tensor_key: str,
    transform_dtype_key: str,
) -> dict[str, object]:
    """Validate and collate optional v2 fields with trailing ``-1`` padding."""
    if not batch:
        raise ValueError("Cannot collate BLCS reference metadata from an empty batch.")
    field_sets = [set(sample) & _REFERENCE_SAMPLE_FIELDS for sample in batch]
    semantic_documents = [sample.get("track_query_reference") for sample in batch]
    if all(not fields for fields in field_sets):
        if any(document is not None for document in semantic_documents):
            raise ValueError(
                "BLCS v2 track-query marker requires complete reference metadata."
            )
        return {}
    for sample_index, fields in enumerate(field_sets):
        if fields != _REFERENCE_SAMPLE_FIELDS:
            raise ValueError(
                "BLCS batch has missing/mixed reference schema at sample "
                f"{sample_index}: expected {sorted(_REFERENCE_SAMPLE_FIELDS)!r}, "
                f"got {sorted(fields)!r}."
            )

    selections: list[ReferenceViewSelection] = []
    tables: list[StableCameraIdTable] = []
    for sample_index, sample in enumerate(batch):
        selection = sample["reference_view_selection"]
        table = sample["stable_camera_id_table"]
        if not isinstance(selection, ReferenceViewSelection):
            raise TypeError(
                f"BLCS sample {sample_index} reference_view_selection has invalid type."
            )
        if not isinstance(table, StableCameraIdTable):
            raise TypeError(
                f"BLCS sample {sample_index} stable_camera_id_table has invalid type."
            )
        if table != selection.stable_camera_id_table:
            raise ValueError(
                f"BLCS sample {sample_index} stable camera ID table does not "
                "match its typed reference selection."
            )
        transform_source = sample.get(transform_dtype_key)
        if not isinstance(transform_source, Tensor):
            raise TypeError(
                f"BLCS sample {sample_index} {transform_dtype_key!r} must be a tensor."
            )
        expected = selection.to_tensor_fields(dtype=transform_source.dtype)
        for key, expected_value in expected.items():
            stored = sample[key]
            if not isinstance(stored, Tensor) or not torch.equal(
                stored,
                expected_value,
            ):
                raise ValueError(
                    f"BLCS sample {sample_index} {key} does not match its typed "
                    "reference selection."
                )
        if sample["reference_camera_id_string"] != selection.reference_camera_id:
            raise ValueError(
                f"BLCS sample {sample_index} canonical reference ID does not "
                "match its typed reference selection."
            )
        expected_inverse = torch.tensor(
            selection.provenance.physical_from_reference,
            dtype=transform_source.dtype,
        )
        stored_inverse = sample["physical_from_reference"]
        if not isinstance(stored_inverse, Tensor) or not torch.equal(
            stored_inverse,
            expected_inverse,
        ):
            raise ValueError(
                f"BLCS sample {sample_index} inverse transform does not match "
                "its typed reference selection."
            )
        selections.append(selection)
        tables.append(table)

    view_rows: list[Tensor] = []
    for sample in batch:
        row = cast("Tensor", sample["view_camera_ids"])
        pad_views = max_views - int(row.shape[0])
        if pad_views < 0:
            raise ValueError("max_views is smaller than a BLCS sample view width.")
        if pad_views:
            row = torch.cat(
                (
                    row,
                    torch.full(
                        (pad_views,),
                        CAMERA_ID_PADDING_VALUE,
                        dtype=torch.int64,
                        device=row.device,
                    ),
                )
            )
        view_rows.append(row)
    result: dict[str, object] = {
        "reference_view_index": torch.stack(
            [cast("Tensor", sample["reference_view_index"]) for sample in batch]
        ),
        "view_camera_ids": torch.stack(view_rows),
        "reference_camera_id": torch.stack(
            [cast("Tensor", sample["reference_camera_id"]) for sample in batch]
        ),
        "reference_from_physical": torch.stack(
            [cast("Tensor", sample["reference_from_physical"]) for sample in batch]
        ),
        "physical_from_reference": torch.stack(
            [cast("Tensor", sample["physical_from_reference"]) for sample in batch]
        ),
        "reference_view_selection": tuple(selections),
        "stable_camera_id_table": tuple(tables),
        "reference_camera_id_string": tuple(
            selection.reference_camera_id for selection in selections
        ),
    }
    model_tensor = batch[0].get(model_tensor_key)
    if not isinstance(model_tensor, Tensor):
        raise TypeError(f"BLCS {model_tensor_key!r} must be a tensor.")
    validate_reference_view_batch(
        reference_view_index=cast("Tensor", result["reference_view_index"]),
        view_camera_ids=cast("Tensor", result["view_camera_ids"]),
        reference_camera_id=cast("Tensor", result["reference_camera_id"]),
        stable_camera_id_tables=tables,
        view_valid_mask=cast("Tensor", result["view_camera_ids"]).ge(0),
        reference_from_physical=cast("Tensor", result["reference_from_physical"]),
        physical_from_reference=cast("Tensor", result["physical_from_reference"]),
        expected_device=model_tensor.device,
    )
    if any(document is not None for document in semantic_documents):
        if any(document is None for document in semantic_documents):
            raise ValueError(
                "BLCS batch contains mixed track-query reference contract markers."
            )
        first_document = semantic_documents[0]
        if any(document != first_document for document in semantic_documents[1:]):
            raise ValueError(
                "BLCS batch contains non-identical track-query reference contracts."
            )
        result["track_query_reference"] = first_document
    return result


def _scene_paths(scene_dir: Path, split_file: Path) -> list[Path]:
    split_path = split_file if split_file.is_absolute() else scene_dir / split_file
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    scenes_base = scene_dir / "scenes"
    if not scenes_base.is_dir():
        scenes_base = scene_dir
    return [
        scenes_base / name
        for raw in split_path.read_text(encoding="utf-8").splitlines()
        if (name := raw.strip())
    ]


def validate_blcs_dataset_court_keypoints(
    *,
    scene_dir: str | Path,
    split_file: str | Path,
    contract: CourtKeypointContract,
) -> DatasetCourtKeypointContract:
    """Validate root/scene/camera headers before any task array is consumed."""
    root = Path(scene_dir)
    paths = _scene_paths(root, Path(split_file))
    return validate_blcs_artifact(
        root,
        contract,
        scene_paths=paths,
    )


def court_views_by_scene(
    result: DatasetCourtKeypointContract,
) -> dict[str, tuple[CourtViewRecord, ...]]:
    """Index validated ordered camera records by stable scene directory name."""
    return {record.scene_id: record.court_views for record in result.scenes}


def resolve_blcs_sample_court_frame(
    *,
    scene: Scene,
    selected_camera_indices: tuple[int, ...],
    court_views: tuple[CourtViewRecord, ...],
    contract: CourtKeypointContract,
    rng: np.random.Generator | None,
    training: bool,
    reference_camera_id: str | None = None,
) -> BLCSSampleCourtFrame:
    """Resolve one typed reference after subset/order selection."""
    if contract.selector == PHYSICAL_V1_SELECTOR:
        if reference_camera_id is not None:
            raise ValueError(
                "physical_v1 BLCS samples must not specify a reference camera ID."
            )
        selected = (
            tuple(court_views[index] for index in selected_camera_indices)
            if court_views
            else ()
        )
        return BLCSSampleCourtFrame(
            selected_views=selected,
            reference_view=None,
            provenance=build_physical_court_provenance(),
            reference_selection=None,
        )
    if len(court_views) != scene.num_cameras:
        raise ValueError(
            f"{scene.path}: camera-view-v2 metadata has {len(court_views)} records "
            f"for {scene.num_cameras} persisted cameras."
        )
    selected = tuple(court_views[index] for index in selected_camera_indices)
    stable_table = StableCameraIdTable.from_complete_scene_camera_ids(
        tuple(view.camera_id for view in court_views)
    )
    selected_ids = tuple(view.camera_id for view in selected)
    if training:
        if rng is None:
            raise TypeError("BLCS training reference selection requires a seeded RNG.")
        if reference_camera_id is not None:
            raise ValueError(
                "BLCS training reference sampling and an explicit evaluation "
                "reference camera are mutually exclusive."
            )
        reference_id = select_seeded_training_reference_camera_id(
            selected_ids,
            rng=rng,
        )
    else:
        reference_id = resolve_evaluation_reference_camera_id(
            selected_ids,
            requested_camera_id=reference_camera_id,
        )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=stable_table,
        selected_views=selected,
        reference_camera_id=reference_id,
    )
    return BLCSSampleCourtFrame(
        selected_views=selected,
        reference_view=selected[selection.reference_view_index],
        provenance=selection.provenance,
        reference_selection=selection,
    )


def align_blcs_court_array(
    value: np.ndarray,
    *,
    source_view: CourtViewRecord | None,
    frame: BLCSSampleCourtFrame,
    keypoint_axis: int,
) -> np.ndarray:
    """Align a persisted camera-local Court array before consumer truncation."""
    if frame.reference_view is None:
        return value
    if source_view is None:
        raise ValueError("camera-view-v2 Court alignment requires camera metadata.")
    aligned = align_court_keypoints_to_reference(
        value,
        source_view,
        frame.reference_view,
        keypoint_axis=keypoint_axis,
    )
    if not isinstance(aligned, np.ndarray):
        raise TypeError("BLCS numpy Court alignment returned a non-array.")
    return aligned


__all__ = [
    "BLCSSampleCourtFrame",
    "align_blcs_court_array",
    "blcs_reference_sample_fields",
    "blcs_track_query_reference_contract_document",
    "collate_blcs_reference_fields",
    "court_views_by_scene",
    "resolve_blcs_sample_court_frame",
    "validate_blcs_dataset_court_keypoints",
]

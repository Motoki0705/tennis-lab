"""Typed BLCS model calls, training targets, and decoded predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data import (
    ReferenceViewSelection,
    StableCameraIdTable,
    validate_reference_view_batch,
)
from src.tasks.base.generate_dataset import (
    CourtReferenceFrameProvenance,
    build_physical_court_provenance,
    court_points_target_to_physical,
    court_vectors_target_to_physical,
)
from src.tasks.base.model_io import (
    ModelCall,
    TrackQueryReferenceContract,
    extract_track_query_reference_contract_metadata,
    write_track_query_reference_contract,
)


@dataclass(frozen=True, slots=True)
class BLCSReferenceMetadata:
    """Validated camera identities and frame transforms carried past forward."""

    selections: tuple[ReferenceViewSelection, ...]
    stable_camera_id_tables: tuple[StableCameraIdTable, ...]
    reference_view_index: Tensor
    view_camera_ids: Tensor
    reference_camera_id: Tensor
    reference_from_physical: Tensor
    physical_from_reference: Tensor
    track_query_contract: TrackQueryReferenceContract | None = None

    def __post_init__(self) -> None:
        selections = tuple(self.selections)
        tables = tuple(self.stable_camera_id_tables)
        batch_size = len(selections)
        if batch_size == 0 or len(tables) != batch_size:
            raise ValueError(
                "BLCS reference metadata requires one selection/table per sample."
            )
        validate_reference_view_batch(
            reference_view_index=self.reference_view_index,
            view_camera_ids=self.view_camera_ids,
            reference_camera_id=self.reference_camera_id,
            stable_camera_id_tables=tables,
            view_valid_mask=self.view_camera_ids.ge(0),
            reference_from_physical=self.reference_from_physical,
            physical_from_reference=self.physical_from_reference,
            expected_device=self.reference_view_index.device,
        )
        for index, (selection, table) in enumerate(
            zip(selections, tables, strict=True)
        ):
            if table != selection.stable_camera_id_table:
                raise ValueError(
                    f"BLCS reference metadata sample {index} table/selection mismatch."
                )
            valid_views = len(selection.selected_views)
            actual_codes = tuple(
                int(value)
                for value in self.view_camera_ids[index, :valid_views].tolist()
            )
            if actual_codes != selection.view_camera_id_codes:
                raise ValueError(
                    f"BLCS reference metadata sample {index} canonical ID/code "
                    "mapping mismatch."
                )
            if bool((self.view_camera_ids[index, valid_views:] != -1).any().item()):
                raise ValueError(
                    f"BLCS reference metadata sample {index} must use -1 only "
                    "for trailing padded views."
                )
            if int(self.reference_view_index[index]) != selection.reference_view_index:
                raise ValueError(
                    f"BLCS reference metadata sample {index} local index mismatch."
                )
            if int(self.reference_camera_id[index]) != (
                selection.reference_camera_id_code
            ):
                raise ValueError(
                    f"BLCS reference metadata sample {index} stable reference "
                    "code mismatch."
                )
            expected_matrix = torch.tensor(
                selection.provenance.reference_from_physical,
                dtype=self.reference_from_physical.dtype,
                device=self.reference_from_physical.device,
            )
            if not torch.equal(self.reference_from_physical[index], expected_matrix):
                raise ValueError(
                    f"BLCS reference metadata sample {index} transform mismatch."
                )
            if self.track_query_contract is not None and (
                selection.provenance.contract_id
                != self.track_query_contract.court_keypoint_contract
                or selection.provenance.target_frame_id
                != self.track_query_contract.target_frame_contract
            ):
                raise ValueError(
                    f"BLCS reference metadata sample {index} Court/target-frame "
                    "contracts do not match its track-query marker."
                )
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "stable_camera_id_tables", tables)

    @property
    def reference_camera_ids(self) -> tuple[str, ...]:
        """Return canonical string reference IDs in batch order."""
        return tuple(selection.reference_camera_id for selection in self.selections)

    @property
    def selected_camera_ids(self) -> tuple[tuple[str, ...], ...]:
        """Return canonical local-order view IDs in batch order."""
        return tuple(selection.selected_camera_ids for selection in self.selections)

    @property
    def target_frame_contracts(self) -> tuple[str, ...]:
        """Return the authoritative #799 target-frame ID for every sample."""
        return tuple(
            selection.provenance.target_frame_id for selection in self.selections
        )

    def to_batch_fields(self) -> dict[str, object]:
        """Return strict model/persistence fields without deriving defaults."""
        result: dict[str, object] = {
            "reference_view_selection": self.selections,
            "stable_camera_id_table": self.stable_camera_id_tables,
            "reference_view_index": self.reference_view_index,
            "view_camera_ids": self.view_camera_ids,
            "reference_camera_id": self.reference_camera_id,
            "reference_from_physical": self.reference_from_physical,
            "physical_from_reference": self.physical_from_reference,
        }
        if self.track_query_contract is not None:
            write_track_query_reference_contract(
                result,
                self.track_query_contract,
                location="BLCS reference metadata",
            )
        return result

    def prediction_payload(self, *, max_views: int | None = None) -> dict[str, object]:
        """Return batch-first numeric and fixed-width canonical string fields."""
        present_views = int(self.view_camera_ids.shape[1])
        output_views = present_views if max_views is None else max_views
        if output_views < present_views:
            raise ValueError(
                f"prediction max_views={output_views} is smaller than the "
                f"collated width {present_views}."
            )
        view_camera_ids = self.view_camera_ids
        if output_views > present_views:
            view_camera_ids = torch.cat(
                (
                    view_camera_ids,
                    torch.full(
                        (len(self.selections), output_views - present_views),
                        -1,
                        dtype=torch.int64,
                        device=view_camera_ids.device,
                    ),
                ),
                dim=1,
            )
        max_id_width = max(
            len(value) for row in self.selected_camera_ids for value in row
        )
        view_id_strings: np.ndarray[Any, np.dtype[np.str_]] = np.full(
            (len(self.selections), output_views),
            "",
            dtype=f"<U{max_id_width}",
        )
        for row_index, camera_ids in enumerate(self.selected_camera_ids):
            view_id_strings[row_index, : len(camera_ids)] = camera_ids
        result: dict[str, object] = {
            "reference_view_index": self.reference_view_index,
            "view_camera_ids": view_camera_ids,
            "reference_camera_id": self.reference_camera_id,
            "reference_camera_id_string": np.asarray(self.reference_camera_ids),
            "view_camera_id_strings": view_id_strings,
            "reference_from_physical": self.reference_from_physical,
            "physical_from_reference": self.physical_from_reference,
            "target_frame_contract": np.asarray(self.target_frame_contracts),
        }
        contract = self.track_query_contract
        if contract is not None:
            selector = contract.reference_selector_mode
            if selector is None:
                raise ValueError(
                    "BLCS reference prediction metadata requires a v2 selector."
                )
            batch_size = len(self.selections)
            result.update(
                {
                    "court_keypoint_contract": np.full(
                        batch_size,
                        contract.court_keypoint_contract,
                    ),
                    "track_query_rope_contract": np.full(
                        batch_size,
                        contract.track_query_rope_contract.value,
                    ),
                    "reference_selector_mode": np.full(
                        batch_size,
                        selector.value,
                    ),
                }
            )
        return result

    def cpu(self) -> BLCSReferenceMetadata:
        """Return tensor metadata detached on CPU for serialization."""
        return BLCSReferenceMetadata(
            selections=self.selections,
            stable_camera_id_tables=self.stable_camera_id_tables,
            reference_view_index=self.reference_view_index.detach().cpu(),
            view_camera_ids=self.view_camera_ids.detach().cpu(),
            reference_camera_id=self.reference_camera_id.detach().cpu(),
            reference_from_physical=self.reference_from_physical.detach().cpu(),
            physical_from_reference=self.physical_from_reference.detach().cpu(),
            track_query_contract=self.track_query_contract,
        )


_REFERENCE_METADATA_FIELDS = frozenset(
    {
        "reference_view_selection",
        "stable_camera_id_table",
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
    }
)


def blcs_reference_metadata_from_batch(
    batch: Mapping[str, object],
) -> BLCSReferenceMetadata | None:
    """Parse strict BLCS v2 metadata without defaults or schema inference."""
    marker = extract_track_query_reference_contract_metadata(
        batch,
        location="BLCS prepared batch",
    )
    present = set(batch) & _REFERENCE_METADATA_FIELDS
    if not present:
        if marker is not None and marker.contract.reference_selector_mode is not None:
            raise ValueError(
                "BLCS v2 track-query marker requires complete reference metadata."
            )
        return None
    if present != _REFERENCE_METADATA_FIELDS:
        raise ValueError(
            "BLCS reference metadata is missing/mixed: expected "
            f"{sorted(_REFERENCE_METADATA_FIELDS)!r}, got {sorted(present)!r}."
        )
    raw_selections = batch["reference_view_selection"]
    raw_tables = batch["stable_camera_id_table"]
    if not isinstance(raw_selections, Sequence) or isinstance(
        raw_selections,
        (str, bytes, bytearray),
    ):
        raw_selections = (raw_selections,)
    if not isinstance(raw_tables, Sequence) or isinstance(
        raw_tables,
        (str, bytes, bytearray),
    ):
        raw_tables = (raw_tables,)
    selections = tuple(cast("Sequence[object]", raw_selections))
    tables = tuple(cast("Sequence[object]", raw_tables))
    if any(not isinstance(value, ReferenceViewSelection) for value in selections):
        raise TypeError("reference_view_selection must contain typed selections.")
    if any(not isinstance(value, StableCameraIdTable) for value in tables):
        raise TypeError("stable_camera_id_table must contain typed complete tables.")
    tensors: dict[str, Tensor] = {}
    for key in (
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
    ):
        value = batch[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"{key} must be a torch.Tensor.")
        tensors[key] = value
    if marker is not None and marker.contract.reference_selector_mode is None:
        raise ValueError("BLCS reference metadata cannot use the legacy v1 marker.")
    return BLCSReferenceMetadata(
        selections=cast("tuple[ReferenceViewSelection, ...]", selections),
        stable_camera_id_tables=cast("tuple[StableCameraIdTable, ...]", tables),
        reference_view_index=tensors["reference_view_index"],
        view_camera_ids=tensors["view_camera_ids"],
        reference_camera_id=tensors["reference_camera_id"],
        reference_from_physical=tensors["reference_from_physical"],
        physical_from_reference=tensors["physical_from_reference"],
        track_query_contract=marker.contract if marker is not None else None,
    )


@dataclass(frozen=True, slots=True)
class BLCSTrajectoryPrediction:
    """Decoded single-ball trajectory model output."""

    position: Tensor
    velocity: Tensor | None
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )
    coordinates_in_metres: bool = False
    reference_metadata: BLCSReferenceMetadata | None = None


@dataclass(frozen=True, slots=True)
class BLCSTrackQueryPrediction:
    """Decoded lifecycle-query output including configured presence semantics."""

    position: Tensor
    presence_logits: Tensor
    presence_probability: Tensor
    presence: Tensor
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )
    coordinates_in_metres: bool = False
    reference_metadata: BLCSReferenceMetadata | None = None


@dataclass(frozen=True, slots=True)
class BLCSTrajectoryTrainingBatch:
    """Validated standard-model call and all tensors consumed by training."""

    call: ModelCall
    position: Tensor
    velocity: Tensor
    loss_mask: Tensor
    target_uv: Tensor
    target_vis: Tensor
    camera_R: Tensor
    camera_C: Tensor
    camera_f: Tensor
    camera_cx: Tensor
    camera_cy: Tensor
    camera_w: Tensor
    camera_h: Tensor
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )
    reference_metadata: BLCSReferenceMetadata | None = None


@dataclass(frozen=True, slots=True)
class BLCSTrackQueryTrainingBatch:
    """Validated tracking-model call and lifecycle supervision tensors."""

    call: ModelCall
    target_position: Tensor
    target_velocity: Tensor
    target_presence: Tensor
    target_instance_id: Tensor
    target_slot_mask: Tensor
    frame_valid: Tensor
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )
    reference_metadata: BLCSReferenceMetadata | None = None


def _physical_batch(
    value: Tensor,
    provenance: tuple[CourtReferenceFrameProvenance, ...],
    *,
    vector: bool,
) -> Tensor:
    if value.ndim < 2 or len(provenance) != value.shape[0]:
        raise ValueError(
            "BLCS prediction provenance must contain exactly one record per batch item."
        )
    rows: list[Tensor] = []
    for batch_index, record in enumerate(provenance):
        transformed = (
            court_vectors_target_to_physical(value[batch_index], record)
            if vector
            else court_points_target_to_physical(value[batch_index], record)
        )
        if not isinstance(transformed, Tensor):
            raise TypeError("BLCS prediction frame conversion returned a non-tensor.")
        rows.append(transformed)
    return torch.stack(rows)


def blcs_trajectory_prediction_to_physical(
    prediction: BLCSTrajectoryPrediction,
) -> BLCSTrajectoryPrediction:
    """Restore a metre-valued standard prediction to physical court space."""
    if not prediction.coordinates_in_metres:
        raise ValueError(
            "BLCS predictions must be denormalized to metres before frame restoration."
        )
    position = _physical_batch(
        prediction.position,
        prediction.court_reference_provenance,
        vector=False,
    )
    velocity = (
        None
        if prediction.velocity is None
        else _physical_batch(
            prediction.velocity,
            prediction.court_reference_provenance,
            vector=True,
        )
    )
    identity = tuple(
        build_physical_court_provenance()
        for _ in prediction.court_reference_provenance
    )
    return BLCSTrajectoryPrediction(
        position=position,
        velocity=velocity,
        court_reference_provenance=identity,
        coordinates_in_metres=True,
        reference_metadata=prediction.reference_metadata,
    )


def blcs_track_query_prediction_to_physical(
    prediction: BLCSTrackQueryPrediction,
) -> BLCSTrackQueryPrediction:
    """Restore a metre-valued tracking prediction to physical court space."""
    if not prediction.coordinates_in_metres:
        raise ValueError(
            "BLCS predictions must be denormalized to metres before frame restoration."
        )
    position = _physical_batch(
        prediction.position,
        prediction.court_reference_provenance,
        vector=False,
    )
    identity = tuple(
        build_physical_court_provenance()
        for _ in prediction.court_reference_provenance
    )
    return BLCSTrackQueryPrediction(
        position=position,
        presence_logits=prediction.presence_logits,
        presence_probability=prediction.presence_probability,
        presence=prediction.presence,
        court_reference_provenance=identity,
        coordinates_in_metres=True,
        reference_metadata=prediction.reference_metadata,
    )


__all__ = [
    "BLCSReferenceMetadata",
    "BLCSTrackQueryPrediction",
    "BLCSTrackQueryTrainingBatch",
    "BLCSTrajectoryPrediction",
    "BLCSTrajectoryTrainingBatch",
    "blcs_reference_metadata_from_batch",
    "blcs_track_query_prediction_to_physical",
    "blcs_trajectory_prediction_to_physical",
]

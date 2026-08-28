"""Typed PLCS model input, output, and inference contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data import (
    ReferenceViewSelection,
    StableCameraIdTable,
    validate_reference_view_batch,
)
from src.tasks.base.generate_dataset import CourtReferenceFrameProvenance
from src.tasks.base.model_io import (
    ModelCall,
    TrackQueryReferenceContract,
    extract_track_query_reference_contract_metadata,
    write_track_query_reference_contract,
)


class PLCSInputProfile(StrEnum):
    """Resolved PLCS tensor layout selected at composition time."""

    FRAME = "frame"
    SEQUENCE = "sequence"
    MULTIVIEW = "multiview"
    TRACK_QUERY = "track_query"


@dataclass(frozen=True, slots=True)
class PLCSReprojectionTarget:
    """Clean 2D pose targets and fixed cameras for reprojection supervision."""

    target_uv: Tensor
    target_vis: Tensor
    padding_mask: Tensor
    camera_R: Tensor
    camera_C: Tensor
    camera_f: Tensor
    camera_cx: Tensor
    camera_cy: Tensor
    camera_w: Tensor
    camera_h: Tensor


@dataclass(frozen=True, slots=True)
class PLCSReferenceMetadata:
    """Validated sample identities and frame transforms carried past forward."""

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
                "PLCS reference metadata requires one selection/table per sample."
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
        if self.physical_from_reference.shape != (batch_size, 3, 3):
            raise ValueError(
                "physical_from_reference must have shape "
                f"({batch_size}, 3, 3)."
            )
        if (
            not self.physical_from_reference.is_floating_point()
            or self.physical_from_reference.dtype
            != self.reference_from_physical.dtype
            or self.physical_from_reference.device
            != self.reference_from_physical.device
        ):
            raise ValueError(
                "physical/reference matrices must share floating dtype and device."
            )
        tolerance = (
            1e-12 if self.reference_from_physical.dtype == torch.float64 else 1e-6
        )
        if not torch.allclose(
            self.physical_from_reference,
            self.reference_from_physical.transpose(-1, -2),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "physical_from_reference must equal reference_from_physical.T."
            )
        for index, (selection, table) in enumerate(
            zip(selections, tables, strict=True)
        ):
            if table != selection.stable_camera_id_table:
                raise ValueError(
                    f"PLCS reference metadata sample {index} table/selection mismatch."
                )
            valid_views = len(selection.selected_views)
            actual_codes = tuple(
                int(value)
                for value in self.view_camera_ids[index, :valid_views].tolist()
            )
            if actual_codes != selection.view_camera_id_codes:
                raise ValueError(
                    f"PLCS reference metadata sample {index} canonical ID/code "
                    "mapping mismatch."
                )
            if bool((self.view_camera_ids[index, valid_views:] != -1).any().item()):
                raise ValueError(
                    f"PLCS reference metadata sample {index} must use -1 only "
                    "for trailing padded views."
                )
            if int(self.reference_view_index[index]) != selection.reference_view_index:
                raise ValueError(
                    f"PLCS reference metadata sample {index} local index mismatch."
                )
            if int(self.reference_camera_id[index]) != (
                selection.reference_camera_id_code
            ):
                raise ValueError(
                    f"PLCS reference metadata sample {index} stable reference "
                    "code mismatch."
                )
            expected_matrix = torch.tensor(
                selection.provenance.reference_from_physical,
                dtype=self.reference_from_physical.dtype,
                device=self.reference_from_physical.device,
            )
            if not torch.equal(self.reference_from_physical[index], expected_matrix):
                raise ValueError(
                    f"PLCS reference metadata sample {index} transform mismatch."
                )
            if self.track_query_contract is not None and (
                selection.provenance.contract_id
                != self.track_query_contract.court_keypoint_contract
                or selection.provenance.target_frame_id
                != self.track_query_contract.target_frame_contract
            ):
                raise ValueError(
                    f"PLCS reference metadata sample {index} Court/target-frame "
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
        """Return the exact #799 target-frame marker for every sample."""
        return tuple(
            selection.provenance.target_frame_id for selection in self.selections
        )

    def to_batch_fields(self) -> dict[str, object]:
        """Return the strict model/persistence fields without deriving defaults."""
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
                location="PLCS reference metadata",
            )
        return result

    def prediction_payload(
        self,
        *,
        max_views: int | None = None,
    ) -> dict[str, object]:
        """Return batch-first tensor and fixed-width string serialization fields."""
        present_views = self.view_camera_ids.shape[1]
        output_views = present_views if max_views is None else max_views
        if output_views < present_views:
            raise ValueError(
                f"prediction max_views={output_views} is smaller than the "
                f"collated width {present_views}."
            )
        view_camera_ids = self.view_camera_ids
        if output_views > present_views:
            view_camera_ids = torch.cat(
                [
                    view_camera_ids,
                    torch.full(
                        (len(self.selections), output_views - present_views),
                        -1,
                        dtype=torch.int64,
                        device=view_camera_ids.device,
                    ),
                ],
                dim=1,
            )
        view_id_strings = np.full(
            (len(self.selections), output_views),
            "",
            dtype=f"<U{max(len(value) for row in self.selected_camera_ids for value in row)}",
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
                    "PLCS reference prediction metadata requires a v2 selector."
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

    def cpu(self) -> PLCSReferenceMetadata:
        """Return tensor metadata detached on CPU for prediction serialization."""
        return PLCSReferenceMetadata(
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


def plcs_reference_metadata_from_batch(
    batch: Mapping[str, object],
) -> PLCSReferenceMetadata | None:
    """Parse strict PLCS v2 metadata without defaults or schema inference."""
    marker = extract_track_query_reference_contract_metadata(
        batch,
        location="PLCS prepared batch",
    )
    present = set(batch) & _REFERENCE_METADATA_FIELDS
    if not present:
        if marker is not None and marker.contract.reference_selector_mode is not None:
            raise ValueError(
                "PLCS v2 track-query marker requires complete reference metadata."
            )
        return None
    if present != _REFERENCE_METADATA_FIELDS:
        raise ValueError(
            "PLCS reference metadata is missing/mixed: expected "
            f"{sorted(_REFERENCE_METADATA_FIELDS)!r}, got {sorted(present)!r}."
        )
    raw_selections = batch["reference_view_selection"]
    raw_tables = batch["stable_camera_id_table"]
    if not isinstance(raw_selections, Sequence) or isinstance(
        raw_selections, (str, bytes, bytearray)
    ):
        raw_selections = (raw_selections,)
    if not isinstance(raw_tables, Sequence) or isinstance(
        raw_tables, (str, bytes, bytearray)
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
        raise ValueError("PLCS reference metadata cannot use the legacy v1 marker.")
    return PLCSReferenceMetadata(
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
class PLCSPreparedBatch:
    """Validated model call plus the output layout required by its consumer."""

    call: ModelCall
    sequence_shape: tuple[int, int] | None = None
    target_position: Tensor | None = None
    target_rotation: Tensor | None = None
    target_human_kp_3d: Tensor | None = None
    target_padding_mask: Tensor | None = None
    reprojection_target: PLCSReprojectionTarget | None = None
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] | None = None
    reference_metadata: PLCSReferenceMetadata | None = None


@dataclass(frozen=True, slots=True)
class PLCSDecodedPrediction:
    """Canonical decoded PLCS model output."""

    position: Tensor
    rotation: Tensor
    canonical_pose: Tensor | None = None
    auxiliary_position: Tensor | None = None
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] | None = None
    reference_metadata: PLCSReferenceMetadata | None = None


@dataclass(frozen=True, slots=True)
class PLCSTrackingDecodedPrediction:
    """Canonical decoded output for the fixed track-query profile."""

    position: Tensor
    rotation: Tensor
    presence_logits: Tensor
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] | None = None
    reference_metadata: PLCSReferenceMetadata | None = None


Float32Array: TypeAlias = np.ndarray


@dataclass(frozen=True, slots=True)
class PLCSPhysicalPrediction:
    """CPU NumPy prediction used by integrated inference consumers."""

    position_meters: Float32Array
    yaw_radians: Float32Array
    canonical_pose: Float32Array | None = None
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] | None = None
    reference_metadata: PLCSReferenceMetadata | None = None


__all__ = [
    "Float32Array",
    "PLCSDecodedPrediction",
    "PLCSInputProfile",
    "PLCSPhysicalPrediction",
    "PLCSPreparedBatch",
    "PLCSReprojectionTarget",
    "PLCSReferenceMetadata",
    "PLCSTrackingDecodedPrediction",
    "plcs_reference_metadata_from_batch",
]

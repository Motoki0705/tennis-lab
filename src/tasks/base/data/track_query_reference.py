"""Stable camera identity and reference-view contracts for track-query v2.

Canonical string camera IDs remain the semantic source of truth.  Tensor IDs
are collision-free ranks in a complete, lexicographically ordered scene table;
they are never hashes, parsed suffixes, or ranks within a selected subset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_V2_SELECTOR,
    IDENTITY_ROTATION_3D,
    RZ_PI_ROTATION_3D,
    CourtKeypointContractError,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    build_reference_frame_provenance,
    validate_reference_frame_provenance,
)

STABLE_CAMERA_ID_TABLE_SCHEMA_VERSION: Final = 1
CAMERA_ID_PADDING_VALUE: Final = -1


class TrackQueryReferenceDataError(ValueError):
    """Raised when a stable-ID or reference-view data contract is violated."""


class StableCameraIdTableError(TrackQueryReferenceDataError):
    """Raised when the complete-scene camera ID table is invalid."""


class ReferenceViewSelectionError(TrackQueryReferenceDataError):
    """Raised when one sample does not identify exactly one valid reference."""


class ReferenceViewBatchError(TrackQueryReferenceDataError):
    """Raised when a collated reference-view tensor contract is invalid."""


def _validate_camera_ids(
    camera_ids: Sequence[str],
    *,
    location: str,
) -> tuple[str, ...]:
    ids = tuple(camera_ids)
    if not ids:
        raise StableCameraIdTableError(f"{location} must not be empty.")
    if any(type(camera_id) is not str or not camera_id.strip() for camera_id in ids):
        raise StableCameraIdTableError(
            f"{location} must contain non-empty canonical strings; got {ids!r}."
        )
    if len(set(ids)) != len(ids):
        raise StableCameraIdTableError(
            f"{location} must contain unique canonical strings; got {ids!r}."
        )
    return ids


@dataclass(frozen=True, slots=True)
class StableCameraIdTable:
    """Collision-free int64 codec built from every camera ID in one scene."""

    camera_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = _validate_camera_ids(
            self.camera_ids,
            location="StableCameraIdTable.camera_ids",
        )
        expected = tuple(sorted(ids))
        if ids != expected:
            raise StableCameraIdTableError(
                "StableCameraIdTable.camera_ids must be in canonical "
                f"lexicographic order; expected {expected!r}, got {ids!r}."
            )
        if len(ids) - 1 > torch.iinfo(torch.int64).max:
            raise StableCameraIdTableError("Camera ID ranks do not fit in int64.")
        object.__setattr__(self, "camera_ids", ids)

    @classmethod
    def from_complete_scene_camera_ids(
        cls,
        camera_ids: Sequence[str],
    ) -> StableCameraIdTable:
        """Build the deterministic table from the complete scene ID domain."""
        ids = _validate_camera_ids(
            camera_ids,
            location="complete scene camera IDs",
        )
        return cls(camera_ids=tuple(sorted(ids)))

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        location: str = "stable_camera_id_table",
    ) -> StableCameraIdTable:
        """Parse an exact persisted table without deriving IDs from payloads."""
        if not isinstance(value, Mapping):
            raise StableCameraIdTableError(f"{location} must be a mapping.")
        mapping = cast("Mapping[object, object]", value)
        expected_fields = {"schema_version", "camera_ids"}
        if set(mapping) != expected_fields:
            raise StableCameraIdTableError(
                f"{location} must have exactly fields {sorted(expected_fields)!r}; "
                f"got {sorted(str(key) for key in mapping)!r}."
            )
        if (
            type(mapping["schema_version"]) is not int
            or mapping["schema_version"] != STABLE_CAMERA_ID_TABLE_SCHEMA_VERSION
        ):
            raise StableCameraIdTableError(
                f"{location}.schema_version must be "
                f"{STABLE_CAMERA_ID_TABLE_SCHEMA_VERSION}."
            )
        raw_ids = mapping["camera_ids"]
        if not isinstance(raw_ids, Sequence) or isinstance(
            raw_ids, (str, bytes, bytearray)
        ):
            raise StableCameraIdTableError(f"{location}.camera_ids must be a list.")
        if any(type(camera_id) is not str for camera_id in raw_ids):
            raise StableCameraIdTableError(
                f"{location}.camera_ids must contain only strings."
            )
        return cls(camera_ids=tuple(cast("Sequence[str]", raw_ids)))

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-serializable table contract."""
        return {
            "schema_version": STABLE_CAMERA_ID_TABLE_SCHEMA_VERSION,
            "camera_ids": list(self.camera_ids),
        }

    def encode(self, camera_id: str) -> int:
        """Return one collision-free scene-table rank."""
        if type(camera_id) is not str or not camera_id.strip():
            raise StableCameraIdTableError(
                f"camera_id must be a non-empty canonical string, got {camera_id!r}."
            )
        try:
            return self.camera_ids.index(camera_id)
        except ValueError as error:
            raise StableCameraIdTableError(
                f"Camera ID {camera_id!r} is absent from the complete scene table "
                f"{self.camera_ids!r}."
            ) from error

    def encode_many(self, camera_ids: Sequence[str]) -> tuple[int, ...]:
        """Encode selected IDs against this complete table, never the subset."""
        return tuple(self.encode(camera_id) for camera_id in camera_ids)

    def decode(self, rank: int) -> str:
        """Recover the canonical string ID for a valid non-padding rank."""
        if type(rank) is not int or rank < 0 or rank >= len(self.camera_ids):
            raise StableCameraIdTableError(
                f"Camera rank must be in [0, {len(self.camera_ids)}), got {rank!r}; "
                f"{CAMERA_ID_PADDING_VALUE} is reserved for padding only."
            )
        return self.camera_ids[rank]


@dataclass(frozen=True, slots=True)
class ReferenceViewSelection:
    """One sample's typed reference, stable codec, views, and #799 provenance."""

    stable_camera_id_table: StableCameraIdTable
    selected_views: tuple[CourtViewRecord, ...]
    provenance: CourtReferenceFrameProvenance

    def __post_init__(self) -> None:
        views = tuple(self.selected_views)
        if not views:
            raise ReferenceViewSelectionError("selected_views must not be empty.")
        selected_ids = tuple(view.camera_id for view in views)
        try:
            self.stable_camera_id_table.encode_many(selected_ids)
            reference_view = validate_reference_frame_provenance(
                self.provenance,
                views,
            )
        except (StableCameraIdTableError, CourtKeypointContractError) as error:
            raise ReferenceViewSelectionError(str(error)) from error
        if self.provenance.contract.selector != CAMERA_VIEW_V2_SELECTOR:
            raise ReferenceViewSelectionError(
                "ReferenceViewSelection requires camera_view_v2 provenance."
            )
        if reference_view is None:
            raise ReferenceViewSelectionError(
                "ReferenceViewSelection must resolve exactly one reference view."
            )
        object.__setattr__(self, "selected_views", views)

    @classmethod
    def create(
        cls,
        *,
        stable_camera_id_table: StableCameraIdTable,
        selected_views: Sequence[CourtViewRecord],
        reference_camera_id: str,
    ) -> ReferenceViewSelection:
        """Create one selection through the authoritative #799 geometry API."""
        views = tuple(selected_views)
        try:
            provenance = build_reference_frame_provenance(
                views,
                reference_camera_id=reference_camera_id,
            )
        except CourtKeypointContractError as error:
            raise ReferenceViewSelectionError(str(error)) from error
        return cls(
            stable_camera_id_table=stable_camera_id_table,
            selected_views=views,
            provenance=provenance,
        )

    @property
    def selected_camera_ids(self) -> tuple[str, ...]:
        """Return selected canonical IDs in batch-local view order."""
        return tuple(view.camera_id for view in self.selected_views)

    @property
    def reference_camera_id(self) -> str:
        """Return the semantic source-of-truth reference identity."""
        value: str | None = self.provenance.reference_camera_id
        if value is None:  # guarded in __post_init__; keeps the type explicit
            raise ReferenceViewSelectionError("Reference camera ID is missing.")
        return value

    @property
    def reference_view_index(self) -> int:
        """Return the selected-order local index resolved from the stable ID."""
        value: int | None = self.provenance.reference_camera_local_index
        if value is None:  # guarded in __post_init__; keeps the type explicit
            raise ReferenceViewSelectionError("Reference local index is missing.")
        return value

    @property
    def view_camera_id_codes(self) -> tuple[int, ...]:
        """Return complete-table ranks in selected local order."""
        return self.stable_camera_id_table.encode_many(self.selected_camera_ids)

    @property
    def reference_camera_id_code(self) -> int:
        """Return the complete-table rank for the semantic reference ID."""
        return self.stable_camera_id_table.encode(self.reference_camera_id)

    def to_tensor_fields(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Build one unbatched model-ready tensor field mapping."""
        if not dtype.is_floating_point:
            raise ReferenceViewSelectionError(
                f"reference transform dtype must be floating, got {dtype}."
            )
        return {
            "reference_view_index": torch.tensor(
                self.reference_view_index,
                dtype=torch.int64,
                device=device,
            ),
            "view_camera_ids": torch.tensor(
                self.view_camera_id_codes,
                dtype=torch.int64,
                device=device,
            ),
            "reference_camera_id": torch.tensor(
                self.reference_camera_id_code,
                dtype=torch.int64,
                device=device,
            ),
            "reference_from_physical": torch.tensor(
                self.provenance.reference_from_physical,
                dtype=dtype,
                device=device,
            ),
        }


def select_seeded_training_reference_camera_id(
    valid_camera_ids: Sequence[str],
    *,
    rng: np.random.Generator,
) -> str:
    """Select a training reference using caller-owned seeded worker RNG.

    Candidate sorting makes the random draw independent of the later view
    permutation.  Repeated draws can still select different references.
    """
    ids = _validate_camera_ids(
        valid_camera_ids,
        location="valid training reference camera IDs",
    )
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator.")
    candidates = tuple(sorted(ids))
    index = int(rng.integers(0, len(candidates)))
    return candidates[index]


def resolve_evaluation_reference_camera_id(
    valid_camera_ids: Sequence[str],
    *,
    requested_camera_id: str | None,
) -> str:
    """Resolve the caller's explicit eval/inference identity.

    Only a single-view sample may omit the request because it has exactly one
    possible semantic reference.  Multi-view evaluation never defaults to the
    first local view.
    """
    ids = _validate_camera_ids(
        valid_camera_ids,
        location="valid evaluation reference camera IDs",
    )
    if requested_camera_id is None:
        if len(ids) == 1:
            return ids[0]
        raise ReferenceViewSelectionError(
            "Multi-view evaluation/inference requires an explicit canonical "
            "reference camera ID."
        )
    if type(requested_camera_id) is not str or not requested_camera_id.strip():
        raise ReferenceViewSelectionError(
            "requested_camera_id must be a non-empty canonical string."
        )
    if requested_camera_id not in ids:
        raise ReferenceViewSelectionError(
            f"Requested reference {requested_camera_id!r} is not in valid cameras "
            f"{ids!r}."
        )
    return requested_camera_id


def include_evaluation_reference_camera(
    complete_camera_ids: Sequence[str],
    selected_camera_indices: Sequence[int],
    *,
    requested_camera_id: str | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Keep an explicit evaluation reference in the selected camera subset.

    Camera selection remains responsible for subset width and local ordering.
    If its independently sampled subset omits the requested stable identity,
    this function replaces one uniformly sampled local slot.  Retained views
    keep their relative order, and the reference is not forced to local index
    zero.  Training callers must not use this evaluation-only adjustment.

    A missing request is left for :func:`resolve_evaluation_reference_camera_id`
    to validate after selection, preserving the single-view-only implicit
    reference rule.
    """
    complete_ids = _validate_camera_ids(
        complete_camera_ids,
        location="complete evaluation camera IDs",
    )
    indices = tuple(selected_camera_indices)
    if not indices:
        raise ReferenceViewSelectionError(
            "selected_camera_indices must contain at least one camera."
        )
    if any(type(index) is not int for index in indices):
        raise ReferenceViewSelectionError(
            "selected_camera_indices must contain only exact integer indices."
        )
    if len(set(indices)) != len(indices):
        raise ReferenceViewSelectionError(
            "selected_camera_indices must contain unique camera indices."
        )
    if any(index < 0 or index >= len(complete_ids) for index in indices):
        raise ReferenceViewSelectionError(
            "selected_camera_indices contain an index outside the complete "
            f"camera domain [0, {len(complete_ids)})."
        )
    if requested_camera_id is None:
        return indices
    if type(requested_camera_id) is not str or not requested_camera_id.strip():
        raise ReferenceViewSelectionError(
            "requested_camera_id must be a non-empty canonical string."
        )
    try:
        required_index = complete_ids.index(requested_camera_id)
    except ValueError as error:
        raise ReferenceViewSelectionError(
            f"Requested reference {requested_camera_id!r} is absent from the "
            f"complete camera domain {complete_ids!r}."
        ) from error
    if required_index in indices:
        return indices
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator.")
    replacement_index = int(rng.integers(0, len(indices)))
    adjusted = list(indices)
    adjusted[replacement_index] = required_index
    return tuple(adjusted)


def validate_reference_view_index(
    reference_view_index: Tensor,
    *,
    batch_size: int,
    num_views: int,
    device: torch.device | str,
) -> None:
    """Validate the model's exact ``int64[B]`` local-reference tensor."""
    if not isinstance(reference_view_index, Tensor):
        raise TypeError("reference_view_index must be a torch.Tensor.")
    if reference_view_index.dtype != torch.int64:
        raise ReferenceViewBatchError(
            "reference_view_index must have dtype torch.int64; "
            f"got {reference_view_index.dtype}."
        )
    if reference_view_index.shape != (batch_size,):
        raise ReferenceViewBatchError(
            f"reference_view_index must have shape ({batch_size},), got "
            f"{tuple(reference_view_index.shape)}."
        )
    expected_device = torch.device(device)
    if reference_view_index.device != expected_device:
        raise ReferenceViewBatchError(
            "reference_view_index must be on the model device "
            f"{expected_device}; got {reference_view_index.device}."
        )
    if num_views <= 0:
        raise ReferenceViewBatchError(f"num_views must be positive, got {num_views}.")
    if ((reference_view_index < 0) | (reference_view_index >= num_views)).any().item():
        raise ReferenceViewBatchError(
            f"reference_view_index values must be in [0, {num_views}); padding, "
            "negative, and out-of-range indices are invalid."
        )


def _validate_reference_transform_batch(
    reference_from_physical: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    physical_from_reference: Tensor | None,
) -> None:
    if reference_from_physical.shape != (batch_size, 3, 3):
        raise ReferenceViewBatchError(
            "reference_from_physical must have shape "
            f"({batch_size}, 3, 3), got {tuple(reference_from_physical.shape)}."
        )
    if not reference_from_physical.is_floating_point():
        raise ReferenceViewBatchError(
            "reference_from_physical must use a floating dtype."
        )
    if reference_from_physical.device != device:
        raise ReferenceViewBatchError(
            "reference_from_physical must share the reference tensor device."
        )
    if not torch.isfinite(reference_from_physical).all().item():
        raise ReferenceViewBatchError(
            "reference_from_physical must contain only finite values."
        )
    tolerance = 1e-12 if reference_from_physical.dtype == torch.float64 else 1e-6
    validation_dtype = (
        torch.float64
        if reference_from_physical.dtype == torch.float64
        else torch.float32
    )
    with torch.autocast(device_type=device.type, enabled=False):
        validation_transform = reference_from_physical.to(dtype=validation_dtype)
        identity = torch.eye(
            3,
            dtype=validation_dtype,
            device=device,
        ).expand(batch_size, -1, -1)
        orthogonal = torch.allclose(
            validation_transform.transpose(-1, -2) @ validation_transform,
            identity,
            rtol=0.0,
            atol=tolerance,
        )
        determinant = torch.linalg.det(validation_transform)
        proper = torch.allclose(
            determinant,
            torch.ones_like(determinant),
            rtol=0.0,
            atol=tolerance,
        )
    if not orthogonal or not proper:
        raise ReferenceViewBatchError(
            "reference_from_physical must contain finite proper rotations."
        )
    allowed = torch.tensor(
        (IDENTITY_ROTATION_3D, RZ_PI_ROTATION_3D),
        dtype=validation_dtype,
        device=device,
    )
    is_allowed = (
        torch.isclose(
            validation_transform[:, None],
            allowed[None],
            rtol=0.0,
            atol=tolerance,
        )
        .all(dim=-1)
        .all(dim=-1)
        .any(dim=1)
    )
    if not is_allowed.all().item():
        raise ReferenceViewBatchError(
            "reference_from_physical must match a #799 identity or Rz(pi) "
            "reference transform."
        )
    if physical_from_reference is not None:
        if not isinstance(physical_from_reference, Tensor):
            raise TypeError("physical_from_reference must be a torch.Tensor.")
        if (
            physical_from_reference.shape != (batch_size, 3, 3)
            or physical_from_reference.dtype != reference_from_physical.dtype
            or physical_from_reference.device != device
        ):
            raise ReferenceViewBatchError(
                "physical_from_reference must match reference_from_physical "
                "shape, floating dtype, and device."
            )
        if not torch.isfinite(physical_from_reference).all().item():
            raise ReferenceViewBatchError(
                "physical_from_reference must contain only finite values."
            )
        with torch.autocast(device_type=device.type, enabled=False):
            inverse_matches = torch.allclose(
                physical_from_reference.to(dtype=validation_dtype),
                validation_transform.transpose(-1, -2),
                rtol=0.0,
                atol=tolerance,
            )
        if not inverse_matches:
            raise ReferenceViewBatchError(
                "physical_from_reference must equal reference_from_physical.T."
            )


def validate_reference_view_batch(
    *,
    reference_view_index: Tensor,
    view_camera_ids: Tensor,
    reference_camera_id: Tensor,
    stable_camera_id_table: StableCameraIdTable | None = None,
    stable_camera_id_tables: Sequence[StableCameraIdTable] | None = None,
    view_valid_mask: Tensor | None = None,
    reference_from_physical: Tensor | None = None,
    physical_from_reference: Tensor | None = None,
    expected_device: torch.device | str | None = None,
) -> None:
    """Validate identity/index/padding/device agreement for a collated batch."""
    if not isinstance(view_camera_ids, Tensor) or not isinstance(
        reference_camera_id, Tensor
    ):
        raise TypeError("view_camera_ids and reference_camera_id must be tensors.")
    if view_camera_ids.ndim != 2:
        raise ReferenceViewBatchError(
            "view_camera_ids must have shape (B,V), got "
            f"{tuple(view_camera_ids.shape)}."
        )
    batch_size, num_views = view_camera_ids.shape
    device = torch.device(expected_device or view_camera_ids.device)
    validate_reference_view_index(
        reference_view_index,
        batch_size=batch_size,
        num_views=num_views,
        device=device,
    )
    for name, value, expected_shape in (
        ("view_camera_ids", view_camera_ids, (batch_size, num_views)),
        ("reference_camera_id", reference_camera_id, (batch_size,)),
    ):
        if value.dtype != torch.int64:
            raise ReferenceViewBatchError(
                f"{name} must have dtype torch.int64, got {value.dtype}."
            )
        if value.shape != expected_shape:
            raise ReferenceViewBatchError(
                f"{name} must have shape {expected_shape}, got {tuple(value.shape)}."
            )
        if value.device != device:
            raise ReferenceViewBatchError(
                f"{name} must be on device {device}, got {value.device}."
            )
    if (view_camera_ids < CAMERA_ID_PADDING_VALUE).any().item():
        raise ReferenceViewBatchError(
            f"view_camera_ids values below {CAMERA_ID_PADDING_VALUE} are invalid."
        )
    padding = view_camera_ids.eq(CAMERA_ID_PADDING_VALUE)
    valid_after_padding = view_camera_ids.ge(0) & padding.cumsum(dim=1).gt(0)
    if valid_after_padding.any().item():
        raise ReferenceViewBatchError(
            "view_camera_ids padding must be trailing within each sample."
        )
    for row_index, row in enumerate(view_camera_ids):
        valid_codes = row[row.ge(0)]
        if valid_codes.numel() == 0:
            raise ReferenceViewBatchError(
                f"sample {row_index} must contain at least one valid camera ID."
            )
        if torch.unique(valid_codes).numel() != valid_codes.numel():
            raise ReferenceViewBatchError(
                f"sample {row_index} contains duplicate stable camera ID codes."
            )
    if stable_camera_id_table is not None and stable_camera_id_tables is not None:
        raise ReferenceViewBatchError(
            "Supply either one shared stable_camera_id_table or per-sample "
            "stable_camera_id_tables, not both."
        )
    tables: tuple[StableCameraIdTable, ...] | None
    if stable_camera_id_tables is not None:
        tables = tuple(stable_camera_id_tables)
        if len(tables) != batch_size or any(
            not isinstance(table, StableCameraIdTable) for table in tables
        ):
            raise ReferenceViewBatchError(
                "stable_camera_id_tables must contain one StableCameraIdTable "
                f"per sample ({batch_size})."
            )
    elif stable_camera_id_table is not None:
        tables = (stable_camera_id_table,) * batch_size
    else:
        tables = None
    if tables is not None:
        for row_index, table in enumerate(tables):
            table_size = len(table.camera_ids)
            valid_codes = view_camera_ids[row_index][view_camera_ids[row_index].ge(0)]
            if (valid_codes >= table_size).any().item():
                raise ReferenceViewBatchError(
                    f"sample {row_index} view_camera_ids contain ranks outside "
                    f"complete table size {table_size}."
                )
            reference_code = int(reference_camera_id[row_index].item())
            if reference_code < 0 or reference_code >= table_size:
                raise ReferenceViewBatchError(
                    f"sample {row_index} reference_camera_id is padding or outside "
                    f"complete table size {table_size}."
                )
    elif (reference_camera_id < 0).any().item():
        raise ReferenceViewBatchError(
            "reference_camera_id must contain non-padding stable ID ranks."
        )
    gathered = view_camera_ids.gather(1, reference_view_index[:, None]).squeeze(1)
    if not torch.equal(gathered, reference_camera_id):
        raise ReferenceViewBatchError(
            "reference_camera_id must exactly equal view_camera_ids at each "
            "reference_view_index."
        )
    if view_valid_mask is not None:
        if not isinstance(view_valid_mask, Tensor):
            raise TypeError("view_valid_mask must be a torch.Tensor.")
        if view_valid_mask.dtype != torch.bool:
            raise ReferenceViewBatchError("view_valid_mask must have bool dtype.")
        if view_valid_mask.shape != (batch_size, num_views):
            raise ReferenceViewBatchError(
                "view_valid_mask must have shape matching view_camera_ids."
            )
        if view_valid_mask.device != device:
            raise ReferenceViewBatchError(
                "view_valid_mask must share the reference tensor device."
            )
        if not torch.equal(view_valid_mask, ~padding):
            raise ReferenceViewBatchError(
                "view_valid_mask must exactly mark non-padding camera IDs."
            )
    if reference_from_physical is not None:
        if not isinstance(reference_from_physical, Tensor):
            raise TypeError("reference_from_physical must be a torch.Tensor.")
        _validate_reference_transform_batch(
            reference_from_physical,
            batch_size=batch_size,
            device=device,
            physical_from_reference=physical_from_reference,
        )
    elif physical_from_reference is not None:
        raise ReferenceViewBatchError(
            "physical_from_reference cannot be validated without "
            "reference_from_physical."
        )


__all__ = [
    "CAMERA_ID_PADDING_VALUE",
    "STABLE_CAMERA_ID_TABLE_SCHEMA_VERSION",
    "ReferenceViewBatchError",
    "ReferenceViewSelection",
    "ReferenceViewSelectionError",
    "StableCameraIdTable",
    "StableCameraIdTableError",
    "TrackQueryReferenceDataError",
    "include_evaluation_reference_camera",
    "resolve_evaluation_reference_camera_id",
    "select_seeded_training_reference_camera_id",
    "validate_reference_view_batch",
    "validate_reference_view_index",
]

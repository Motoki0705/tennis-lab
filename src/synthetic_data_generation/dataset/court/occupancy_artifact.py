"""Immutable Court V4 residual-occupancy publication contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.schema import (
    COURT_DATASET_SCHEMA_V4,
    COURT_PLAN_SCHEMA_V4,
)

COURT_V4_SUPPORT_OCCUPANCY_SCHEMA = "court_v4_support_occupancy_v1"
COURT_V4_SUPPORT_OCCUPANCY_METADATA_FILE = "trajectory-support-occupancy.json"
COURT_V4_SUPPORT_OCCUPANCY_CELLS_FILE = "trajectory-support-occupancy.npy"
COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH = (
    f"diagnostics/{COURT_V4_SUPPORT_OCCUPANCY_METADATA_FILE}"
)
COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH = (
    f"diagnostics/{COURT_V4_SUPPORT_OCCUPANCY_CELLS_FILE}"
)
COURT_V4_SUPPORT_OCCUPANCY_COORDINATE_SPACE = "metric_scene_metres"
COURT_V4_SUPPORT_OCCUPANCY_DTYPE = "little_endian_int64"

_METADATA_KEYS = {
    "schema",
    "dataset_schema",
    "plan_schema",
    "scene_id",
    "profile",
    "policy_decision_id",
    "support_input_digest",
    "coordinate_space",
    "voxel_size_m",
    "cell_count",
    "cells_file",
    "cells_dtype",
    "cells_shape",
    "content_digest",
}
_IDENTITY_KEYS = {
    "schema",
    "policy_decision_id",
    "support_input_digest",
    "coordinate_space",
    "voxel_size_m",
    "cell_count",
    "cells_dtype",
    "cells_shape",
    "content_digest",
}


@dataclass(frozen=True, slots=True)
class CourtV4SupportOccupancyIdentity:
    """Cell-free persisted identity for one exact final occupancy snapshot."""

    coordinate_space: str
    voxel_size_m: float
    cell_count: int
    support_input_digest: str
    policy_decision_id: str
    content_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinate_space",
            _coordinate_space(self.coordinate_space),
        )
        object.__setattr__(
            self,
            "voxel_size_m",
            _positive_finite(self.voxel_size_m, name="voxel_size_m"),
        )
        object.__setattr__(
            self,
            "cell_count",
            _positive_integer(self.cell_count, name="cell_count"),
        )
        object.__setattr__(
            self,
            "support_input_digest",
            _sha256(self.support_input_digest, name="support_input_digest"),
        )
        object.__setattr__(
            self,
            "policy_decision_id",
            _text(self.policy_decision_id, name="policy_decision_id"),
        )
        object.__setattr__(
            self,
            "content_digest",
            _sha256(self.content_digest, name="content_digest"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> CourtV4SupportOccupancyIdentity:
        """Parse the exact persisted identity without accepting extra fields."""
        raw = _exact_mapping(value, keys=_IDENTITY_KEYS, name="identity")
        if raw["schema"] != COURT_V4_SUPPORT_OCCUPANCY_SCHEMA:
            raise ValueError("Unknown Court V4 occupancy identity schema.")
        if raw["cells_dtype"] != COURT_V4_SUPPORT_OCCUPANCY_DTYPE:
            raise ValueError("Court V4 occupancy identity dtype is invalid.")
        cell_count = _positive_integer(raw["cell_count"], name="cell_count")
        _validate_cells_shape(raw["cells_shape"], cell_count=cell_count)
        return cls(
            coordinate_space=_coordinate_space(raw["coordinate_space"]),
            voxel_size_m=_positive_finite(raw["voxel_size_m"], name="voxel_size_m"),
            cell_count=cell_count,
            support_input_digest=_sha256(
                raw["support_input_digest"],
                name="support_input_digest",
            ),
            policy_decision_id=_text(
                raw["policy_decision_id"],
                name="policy_decision_id",
            ),
            content_digest=_sha256(raw["content_digest"], name="content_digest"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact strict JSON identity without serializing cells."""
        return {
            "schema": COURT_V4_SUPPORT_OCCUPANCY_SCHEMA,
            "policy_decision_id": self.policy_decision_id,
            "support_input_digest": self.support_input_digest,
            "coordinate_space": self.coordinate_space,
            "voxel_size_m": self.voxel_size_m,
            "cell_count": self.cell_count,
            "cells_dtype": COURT_V4_SUPPORT_OCCUPANCY_DTYPE,
            "cells_shape": [self.cell_count, 3],
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True, eq=False)
class CourtV4SupportOccupancySnapshot:
    """Canonical immutable cells captured at the authoritative support build."""

    cells: NDArray[np.int64]
    coordinate_space: str
    voxel_size_m: float
    support_input_digest: str
    policy_decision_id: str
    content_digest: str

    def __post_init__(self) -> None:
        cells = _canonical_cells(self.cells)
        coordinate_space = _coordinate_space(self.coordinate_space)
        voxel_size_m = _positive_finite(
            self.voxel_size_m,
            name="voxel_size_m",
        )
        support_input_digest = _sha256(
            self.support_input_digest,
            name="support_input_digest",
        )
        policy_decision_id = _text(
            self.policy_decision_id,
            name="policy_decision_id",
        )
        content_digest = _sha256(self.content_digest, name="content_digest")
        computed_digest = occupancy_cells_content_digest(cells)
        if content_digest != computed_digest:
            raise ValueError(
                "Court V4 occupancy content_digest disagrees with the exact cells."
            )
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "coordinate_space", coordinate_space)
        object.__setattr__(self, "voxel_size_m", voxel_size_m)
        object.__setattr__(self, "support_input_digest", support_input_digest)
        object.__setattr__(self, "policy_decision_id", policy_decision_id)
        object.__setattr__(self, "content_digest", content_digest)

    @property
    def cell_count(self) -> int:
        """Return the exact final occupied-cell count."""
        return int(self.cells.shape[0])

    @property
    def identity(self) -> CourtV4SupportOccupancyIdentity:
        """Return the persisted cell-free identity for this exact snapshot."""
        return CourtV4SupportOccupancyIdentity(
            coordinate_space=self.coordinate_space,
            voxel_size_m=self.voxel_size_m,
            cell_count=self.cell_count,
            support_input_digest=self.support_input_digest,
            policy_decision_id=self.policy_decision_id,
            content_digest=self.content_digest,
        )


@dataclass(frozen=True, slots=True)
class PublishedCourtV4SupportOccupancy:
    """Validated dataset-bound Court V4 occupancy artifact."""

    snapshot: CourtV4SupportOccupancySnapshot
    scene_id: str
    profile: str

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, CourtV4SupportOccupancySnapshot):
            raise TypeError("snapshot must be CourtV4SupportOccupancySnapshot.")
        object.__setattr__(self, "scene_id", _text(self.scene_id, name="scene_id"))
        object.__setattr__(self, "profile", _text(self.profile, name="profile"))

    def metadata(self) -> dict[str, object]:
        """Return the exact strict JSON companion payload."""
        snapshot = self.snapshot
        return {
            **snapshot.identity.to_dict(),
            "dataset_schema": COURT_DATASET_SCHEMA_V4,
            "plan_schema": COURT_PLAN_SCHEMA_V4,
            "scene_id": self.scene_id,
            "profile": self.profile,
            "cells_file": COURT_V4_SUPPORT_OCCUPANCY_CELLS_FILE,
        }


def build_court_v4_support_occupancy_snapshot(
    cells: object,
    *,
    voxel_size_m: float,
    support_input_digest: str,
    policy_decision_id: str,
) -> CourtV4SupportOccupancySnapshot:
    """Freeze lexicographically sorted final cells without reconstructing them."""
    raw = np.asarray(cells)
    if raw.dtype.str != "<i8":
        raise TypeError(
            "Court V4 occupancy cells must use exact little-endian int64 dtype."
        )
    if raw.ndim != 2 or raw.shape[1:] != (3,) or len(raw) == 0:
        raise ValueError("Court V4 occupancy cells must have non-empty shape (N,3).")
    if _cells_are_strictly_sorted(raw):
        canonical = _canonical_cells(raw)
    else:
        order = np.lexsort((raw[:, 2], raw[:, 1], raw[:, 0]))
        canonical = _canonical_cells(raw[order])
    return CourtV4SupportOccupancySnapshot(
        cells=canonical,
        coordinate_space=COURT_V4_SUPPORT_OCCUPANCY_COORDINATE_SPACE,
        voxel_size_m=voxel_size_m,
        support_input_digest=support_input_digest,
        policy_decision_id=policy_decision_id,
        content_digest=occupancy_cells_content_digest(canonical),
    )


def occupancy_cells_content_digest(cells: object) -> str:
    """Hash the canonical little-endian C-order cell payload."""
    array = np.asarray(cells)
    if array.dtype.str != "<i8" or array.ndim != 2 or array.shape[1:] != (3,):
        raise TypeError("Occupancy digest input must have exact int64 shape (N,3).")
    canonical = np.ascontiguousarray(array, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"court_v4_support_occupancy_cells_v1\0")
    digest.update(np.asarray(canonical.shape, dtype="<u8").tobytes(order="C"))
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def write_court_v4_support_occupancy(
    diagnostics_root: Path,
    *,
    snapshot: CourtV4SupportOccupancySnapshot,
    scene_id: str,
    profile: str,
) -> PublishedCourtV4SupportOccupancy:
    """Atomically publish and immediately revalidate one fixed artifact pair."""
    if (
        not isinstance(diagnostics_root, Path)
        or not diagnostics_root.is_absolute()
        or diagnostics_root.is_symlink()
        or not diagnostics_root.is_dir()
    ):
        raise ValueError("diagnostics_root must be an absolute ordinary directory.")
    published = PublishedCourtV4SupportOccupancy(
        snapshot=snapshot,
        scene_id=scene_id,
        profile=profile,
    )
    cells_path = diagnostics_root / COURT_V4_SUPPORT_OCCUPANCY_CELLS_FILE
    metadata_path = diagnostics_root / COURT_V4_SUPPORT_OCCUPANCY_METADATA_FILE
    if any(path.exists() or path.is_symlink() for path in (cells_path, metadata_path)):
        raise FileExistsError("Court V4 occupancy artifact already exists.")
    cells_publication = _write_npy_atomic(cells_path, snapshot.cells)
    metadata_publication: _OwnedPublishedFile | None = None
    try:
        metadata_publication = _write_json_atomic(metadata_path, published.metadata())
        validated = load_court_v4_support_occupancy(
            diagnostics_root.parent,
            expected_scene_id=published.scene_id,
            expected_profile=published.profile,
            expected_policy_decision_id=snapshot.policy_decision_id,
            expected_support_input_digest=snapshot.support_input_digest,
            expected_voxel_size_m=snapshot.voxel_size_m,
            expected_cell_count=snapshot.cell_count,
            expected_content_digest=snapshot.content_digest,
        )
    except Exception:
        if metadata_publication is not None:
            metadata_publication.cleanup()
        cells_publication.cleanup()
        raise
    return validated


def load_court_v4_support_occupancy(
    dataset_root: Path,
    *,
    expected_scene_id: str | None = None,
    expected_profile: str | None = None,
    expected_policy_decision_id: str | None = None,
    expected_support_input_digest: str | None = None,
    expected_voxel_size_m: float | None = None,
    expected_cell_count: int | None = None,
    expected_content_digest: str | None = None,
    maximum_cells: int | None = None,
) -> PublishedCourtV4SupportOccupancy:
    """Load the fixed pair with exact schema, binding, dtype, and digest checks."""
    root = _ordinary_root(dataset_root)
    metadata_path = _contained_file(
        root,
        COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH,
    )
    metadata_value = json.loads(
        metadata_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    metadata = _exact_mapping(metadata_value, keys=_METADATA_KEYS, name="metadata")
    if (
        metadata["dataset_schema"] != COURT_DATASET_SCHEMA_V4
        or metadata["plan_schema"] != COURT_PLAN_SCHEMA_V4
    ):
        raise ValueError("Court V4 occupancy dataset/plan identity is invalid.")
    scene_id = _text(metadata["scene_id"], name="scene_id")
    profile = _text(metadata["profile"], name="profile")
    identity = CourtV4SupportOccupancyIdentity.from_mapping(
        {key: metadata[key] for key in _IDENTITY_KEYS}
    )
    if maximum_cells is not None:
        limit = _positive_integer(maximum_cells, name="maximum_cells")
        if identity.cell_count > limit:
            raise ValueError(
                "Court V4 occupancy cell_count exceeds configured maximum_cells."
            )
    if metadata["cells_file"] != COURT_V4_SUPPORT_OCCUPANCY_CELLS_FILE:
        raise ValueError("Court V4 occupancy numeric artifact contract is invalid.")
    _expect_equal(scene_id, expected_scene_id, name="scene_id")
    _expect_equal(profile, expected_profile, name="profile")
    _expect_equal(
        identity.policy_decision_id,
        expected_policy_decision_id,
        name="policy_decision_id",
    )
    _expect_equal(
        identity.support_input_digest,
        expected_support_input_digest,
        name="support_input_digest",
    )
    if expected_voxel_size_m is not None and not math.isclose(
        identity.voxel_size_m,
        _positive_finite(expected_voxel_size_m, name="expected_voxel_size_m"),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Court V4 occupancy voxel_size_m binding disagrees.")
    if expected_cell_count is not None and identity.cell_count != _positive_integer(
        expected_cell_count,
        name="expected_cell_count",
    ):
        raise ValueError("Court V4 occupancy cell_count binding disagrees.")
    _expect_equal(
        identity.content_digest,
        expected_content_digest,
        name="content_digest",
    )
    cells_path = _contained_file(root, COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH)
    cells_value = np.load(cells_path, allow_pickle=False, mmap_mode="r")
    if cells_value.dtype.str != "<i8" or cells_value.shape != (
        identity.cell_count,
        3,
    ):
        raise ValueError("Court V4 occupancy array dtype/shape is invalid.")
    snapshot = CourtV4SupportOccupancySnapshot(
        cells=cast(NDArray[np.int64], cells_value),
        coordinate_space=identity.coordinate_space,
        voxel_size_m=identity.voxel_size_m,
        support_input_digest=identity.support_input_digest,
        policy_decision_id=identity.policy_decision_id,
        content_digest=identity.content_digest,
    )
    return PublishedCourtV4SupportOccupancy(
        snapshot=snapshot,
        scene_id=scene_id,
        profile=profile,
    )


def _canonical_cells(value: object) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.dtype.str != "<i8":
        raise TypeError(
            "Court V4 occupancy cells must use exact little-endian int64 dtype."
        )
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) == 0:
        raise ValueError("Court V4 occupancy cells must have non-empty shape (N,3).")
    canonical = np.ascontiguousarray(array, dtype="<i8")
    if not _cells_are_strictly_sorted(canonical):
        raise ValueError(
            "Court V4 occupancy cells must be unique and lexicographically sorted."
        )
    if _is_bytes_backed(canonical):
        return cast(NDArray[np.int64], canonical)
    payload = canonical.tobytes(order="C")
    immutable = np.frombuffer(payload, dtype="<i8").reshape(canonical.shape)
    return cast(NDArray[np.int64], immutable)


def _is_bytes_backed(array: NDArray[np.int64]) -> bool:
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _cells_are_strictly_sorted(cells: NDArray[np.int64]) -> bool:
    previous = cells[:-1]
    current = cells[1:]
    if len(current) == 0:
        return True
    out_of_order = (
        (current[:, 0] < previous[:, 0])
        | (
            (current[:, 0] == previous[:, 0])
            & (
                (current[:, 1] < previous[:, 1])
                | (
                    (current[:, 1] == previous[:, 1])
                    & (current[:, 2] <= previous[:, 2])
                )
            )
        )
    )
    return not bool(np.any(out_of_order))


def _ordinary_root(value: Path) -> Path:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or value.is_symlink()
        or not value.is_dir()
    ):
        raise ValueError("dataset_root must be an absolute ordinary directory.")
    return value.resolve(strict=True)


def _contained_file(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*relative.split("/"))
    if candidate.is_symlink():
        raise ValueError("Court V4 occupancy paths must not be symbolic links.")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Court V4 occupancy path escapes the dataset root.")
    return resolved


@dataclass(frozen=True, slots=True)
class _OwnedPublishedFile:
    path: Path
    device: int
    inode: int

    def cleanup(self) -> None:
        """Remove only the final hard link still owned by this invocation."""
        try:
            stat = self.path.lstat()
        except FileNotFoundError:
            return
        if (stat.st_dev, stat.st_ino) == (self.device, self.inode):
            self.path.unlink()


def _write_npy_atomic(
    path: Path,
    cells: NDArray[np.int64],
) -> _OwnedPublishedFile:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, cells, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
            stat = os.fstat(handle.fileno())
        _publish_exclusively(temporary, path)
        return _OwnedPublishedFile(path=path, device=stat.st_dev, inode=stat.st_ino)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, object],
) -> _OwnedPublishedFile:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            stat = os.fstat(handle.fileno())
        _publish_exclusively(temporary, path)
        return _OwnedPublishedFile(path=path, device=stat.st_dev, inode=stat.st_ino)
    finally:
        temporary.unlink(missing_ok=True)


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"Court V4 occupancy {name} must be an object.")
    result = cast(Mapping[str, object], value)
    if set(result) != keys:
        raise ValueError(f"Court V4 occupancy {name} keys are invalid.")
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Court V4 occupancy metadata key: {key!r}.")
        result[key] = value
    return result


def _publish_exclusively(staged: Path, target: Path) -> None:
    try:
        os.link(staged, target, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(f"Artifact target already exists: {target}") from error


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"Court V4 occupancy {name} must be trimmed text.")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"Court V4 occupancy {name} must be lowercase SHA-256.")
    return result


def _coordinate_space(value: object) -> str:
    result = _text(value, name="coordinate_space")
    if result != COURT_V4_SUPPORT_OCCUPANCY_COORDINATE_SPACE:
        raise ValueError("Unknown Court V4 occupancy coordinate_space.")
    return result


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Court V4 occupancy {name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Court V4 occupancy {name} must be positive and finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"Court V4 occupancy {name} must be a positive integer.")
    return value


def _validate_cells_shape(value: object, *, cell_count: int) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value != [cell_count, 3]
    ):
        raise ValueError("Court V4 occupancy cells_shape is invalid.")


def _expect_equal(actual: str, expected: str | None, *, name: str) -> None:
    if expected is not None and actual != _text(expected, name=f"expected_{name}"):
        raise ValueError(f"Court V4 occupancy {name} binding disagrees.")


__all__ = [
    "COURT_V4_SUPPORT_OCCUPANCY_CELLS_FILE",
    "COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH",
    "COURT_V4_SUPPORT_OCCUPANCY_COORDINATE_SPACE",
    "COURT_V4_SUPPORT_OCCUPANCY_METADATA_FILE",
    "COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH",
    "COURT_V4_SUPPORT_OCCUPANCY_SCHEMA",
    "CourtV4SupportOccupancyIdentity",
    "CourtV4SupportOccupancySnapshot",
    "PublishedCourtV4SupportOccupancy",
    "build_court_v4_support_occupancy_snapshot",
    "load_court_v4_support_occupancy",
    "occupancy_cells_content_digest",
    "write_court_v4_support_occupancy",
]

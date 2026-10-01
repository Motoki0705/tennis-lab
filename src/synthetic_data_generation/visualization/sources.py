"""Strict, frame-streaming sources for canonical generated datasets."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.assembler import (
    CourtArrayValidationMode,
    validate_court_dataset,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
    court_schema_from_dataset_schema,
)
from src.utils.data.float32_store import read_float32


@dataclass(frozen=True, slots=True)
class CourtSourceFrame:
    """One manifest-ordered Court image and its semantic label payload."""

    rgb: NDArray[np.float32]
    sample_id: str
    view_id: str
    trajectory_frame_index: int
    projection: Mapping[str, object]
    schema_version: CourtDatasetSchemaVersion = CourtDatasetSchemaVersion.V1


class CourtVisualizationSource:
    """Validated Court trajectory reader preserving canonical manifest order."""

    def __init__(self, root: Path, *, trajectory_id: str) -> None:
        validate_court_dataset(
            root,
            array_validation=CourtArrayValidationMode.FULL,
        )
        manifest = _object(
            _load_json(_contained_file(root, "dataset.json")),
            name="Court dataset",
        )
        self.dataset_schema = _text(manifest.get("schema"), name="Court schema")
        self.schema_definition = court_schema_from_dataset_schema(self.dataset_schema)
        self.dataset_scene_id = _text(manifest.get("scene_id"), name="Court scene_id")
        groups = tuple(
            _object(value, name="Court trajectory group")
            for value in _array(
                manifest.get("trajectory_groups"), name="Court trajectory_groups"
            )
        )
        selected_group: Mapping[str, object] | None = None
        for group in groups:
            trajectory = _object(group.get("trajectory"), name="Court trajectory")
            if trajectory.get("trajectory_id") == trajectory_id:
                selected_group = group
                break
        if selected_group is None:
            raise KeyError(f"Unknown Court trajectory_id: {trajectory_id!r}.")
        views = tuple(
            _text(
                _object(value, name="Court view").get("view_id"),
                name="Court view_id",
            )
            for value in _array(selected_group.get("views"), name="Court views")
        )
        if not views or len(views) != len(set(views)):
            raise ValueError("Court trajectory view inventory is invalid.")
        sample_count = _positive_integer(
            selected_group.get("sample_count"), name="Court sample_count"
        )
        records = tuple(
            record
            for value in _array(manifest.get("samples"), name="Court samples")
            for record in (_object(value, name="Court sample"),)
            if record.get("trajectory_id") == trajectory_id
        )
        if not records:
            raise ValueError(
                f"Court trajectory {trajectory_id!r} has no accepted rendered frames."
            )
        expected_order = tuple(
            record
            for view_id in views
            for record in records
            if record.get("view_id") == view_id
        )
        if records != expected_order:
            raise ValueError(
                "Court trajectory frames do not follow canonical view/frame order."
            )
        for view_id in views:
            indices = tuple(
                _nonnegative_integer(
                    record.get("trajectory_frame_index"),
                    name="Court trajectory_frame_index",
                )
                for record in records
                if record.get("view_id") == view_id
            )
            if not indices or indices != tuple(sorted(set(indices))):
                raise ValueError(
                    f"Court view {view_id!r} source-frame ordering is inconsistent."
                )
            if indices[-1] >= sample_count:
                raise ValueError("Court frame index exceeds its trajectory inventory.")
        dimensions = {
            (
                _positive_integer(record.get("width"), name="Court width"),
                _positive_integer(record.get("height"), name="Court height"),
            )
            for record in records
        }
        if len(dimensions) != 1:
            raise ValueError("Court trajectory frame dimensions are inconsistent.")
        self.width, self.height = next(iter(dimensions))
        self.root = root
        self.trajectory_id = trajectory_id
        self._records = records
        self.frame_order = tuple(
            {
                "sample_id": _text(record.get("sample_id"), name="sample_id"),
                "view_id": _text(record.get("view_id"), name="view_id"),
                "trajectory_frame_index": _nonnegative_integer(
                    record.get("trajectory_frame_index"),
                    name="trajectory_frame_index",
                ),
            }
            for record in records
        )

    @property
    def frame_count(self) -> int:
        """Return the accepted rendered-frame count for the selected trajectory."""
        return len(self._records)

    def frames(self) -> Iterator[CourtSourceFrame]:
        """Stream each NHT RGB array and corresponding label in manifest order."""
        for record in self._records:
            rgb = _float32_rgb(
                _contained_file(
                    self.root,
                    _text(record.get("rgb"), name="Court rgb path"),
                ),
                width=self.width,
                height=self.height,
            )
            label = _object(
                _load_json(
                    _contained_file(
                        self.root,
                        _text(record.get("labels"), name="Court labels path"),
                    )
                ),
                name="Court labels",
            )
            label_schema = label.get("schema")
            if (
                not isinstance(label_schema, str)
                or label_schema != self.schema_definition.sample_schema
            ):
                raise ValueError("Court labels schema changed after validation.")
            for field in (
                "sample_id",
                "view_id",
                "trajectory_frame_index",
                "projection",
            ):
                if label.get(field) != record.get(field):
                    raise ValueError(
                        f"Court labels changed after validation at field {field!r}."
                    )
            yield CourtSourceFrame(
                rgb=rgb,
                sample_id=_text(label["sample_id"], name="Court sample_id"),
                view_id=_text(label["view_id"], name="Court view_id"),
                trajectory_frame_index=_nonnegative_integer(
                    label["trajectory_frame_index"],
                    name="Court trajectory_frame_index",
                ),
                projection=_object(label["projection"], name="Court projection"),
                schema_version=self.schema_definition.version,
            )


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Required visualization JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object.")
    return cast(Mapping[str, object], value)


def _exact_object(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    result = _object(value, name=name)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
        )
    return result


def _array(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _contained_file(root: Path, relative: str) -> Path:
    return _contained(root, relative, directory=False)


def _contained_directory(root: Path, relative: str) -> Path:
    return _contained(root, relative, directory=True)


def _contained(root: Path, relative: str, *, directory: bool) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("Dataset reference must be a contained relative POSIX path.")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise ValueError("Dataset references must not be symbolic links.")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("Dataset reference escapes its canonical root.")
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        raise ValueError("Dataset reference has the wrong file type.")
    return resolved


def _float32_rgb(path: Path, *, width: int, height: int) -> NDArray[np.float32]:
    value = read_float32(path)
    if value.dtype != np.float32 or value.shape != (height, width, 3):
        raise ValueError(f"NHT RGB frame has an invalid contract: {path}")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError(f"NHT RGB frame is non-finite or outside [0,1]: {path}")
    return cast(NDArray[np.float32], value)


__all__ = ["CourtSourceFrame", "CourtVisualizationSource"]

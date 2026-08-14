"""Attempt-local, trajectory-group-disjoint Court render shards.

The public NHT client performs one complete value scan while retaining only file
metadata. Assembly reopens one sample at a time; attempt-local recovery inspects
immutable file identity plus ``.npy`` headers. This keeps shard inventory strict
without retaining multi-gigabyte payloads across shards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.synthetic_data_generation.dataset.court.components.labels import (
    MultiCourtProjection,
)
from src.synthetic_data_generation.dataset.court.contracts import PlannedCourtSample
from src.synthetic_data_generation.rendering.nht import NHTRenderRecord
from src.utils.io import load_json, save_json_atomic

COURT_SHARD_SCHEMA = "court_render_shard_attempt_v1"


class StaleCourtShardError(ValueError):
    """Raised when a shard belongs to another stage attempt."""


@dataclass(frozen=True, slots=True)
class CourtRenderedSample:
    """Validated public NHT files bound to one planned sample."""

    sample: PlannedCourtSample
    rgb_path: Path
    rgb_preview_path: Path
    alpha_path: Path
    alpha_preview_path: Path
    depth_path: Path

    @property
    def source_directory(self) -> Path:
        """Return the single NHT camera-result directory."""
        parents = {
            self.rgb_path.parent,
            self.rgb_preview_path.parent,
            self.alpha_path.parent,
            self.alpha_preview_path.parent,
            self.depth_path.parent,
        }
        if len(parents) != 1:
            raise ValueError("NHT sample files do not share one camera directory.")
        return next(iter(parents))


@dataclass(frozen=True, slots=True)
class CourtShardTiming:
    """Measured wall time for one external NHT shard invocation."""

    shard_id: str
    camera_count: int
    wall_seconds: float

    def __post_init__(self) -> None:
        if not self.shard_id or self.shard_id != self.shard_id.strip():
            raise ValueError("shard_id must be a non-empty trimmed string.")
        if isinstance(self.camera_count, bool) or self.camera_count <= 0:
            raise ValueError("camera_count must be a positive integer.")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, (int, float))
            or not np.isfinite(float(self.wall_seconds))
            or self.wall_seconds < 0.0
        ):
            raise ValueError("wall_seconds must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class CourtRenderResult:
    """Complete Court render output plus measurable public-boundary evidence."""

    samples: tuple[CourtRenderedSample, ...]
    pre_render_projections: tuple[MultiCourtProjection, ...]
    pre_render_rejected_sample_ids: tuple[str, ...]
    resolved_shard_count: int
    nht_invocations: int
    request_path_count: int
    maximum_shard_sample_count: int
    generated_bytes: int
    nht_complete_array_scans: int
    scene_validation_count: int
    preview_validation_count: int
    loaded_array_bytes: int
    maximum_nht_live_array_bytes: int
    retained_nht_array_bytes: int
    shard_timings: tuple[CourtShardTiming, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        projections = tuple(self.pre_render_projections)
        rejected_ids = tuple(self.pre_render_rejected_sample_ids)
        timings = tuple(self.shard_timings)
        rendered_ids = tuple(item.sample.sample_id for item in samples)
        projection_ids = tuple(projection.camera_id for projection in projections)
        if not samples or not projections:
            raise ValueError("Court render samples/projections must be non-empty.")
        if len(projection_ids) != len(set(projection_ids)):
            raise ValueError("Court pre-render projections contain duplicate cameras.")
        if len(rejected_ids) != len(set(rejected_ids)):
            raise ValueError("Court pre-render rejections contain duplicate cameras.")
        if set(rendered_ids) & set(rejected_ids) or set(projection_ids) != (
            set(rendered_ids) | set(rejected_ids)
        ):
            raise ValueError(
                "Court rendered/rejected cameras do not partition pre-render projections."
            )
        for name in (
            "resolved_shard_count",
            "nht_invocations",
            "request_path_count",
            "maximum_shard_sample_count",
            "generated_bytes",
            "nht_complete_array_scans",
            "scene_validation_count",
            "preview_validation_count",
            "loaded_array_bytes",
            "maximum_nht_live_array_bytes",
            "retained_nht_array_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.resolved_shard_count <= 0:
            raise ValueError("resolved_shard_count must be positive.")
        if self.generated_bytes <= 0:
            raise ValueError("Court generated byte evidence must be positive.")
        if self.maximum_shard_sample_count <= 0:
            raise ValueError("Court maximum shard sample count must be positive.")
        if self.nht_complete_array_scans != sum(
            timing.camera_count for timing in timings
        ):
            raise ValueError("Court NHT array-scan evidence disagrees with shard timings.")
        if self.scene_validation_count not in {0, 1}:
            raise ValueError("Court NHT scene validation must occur at most once.")
        if self.preview_validation_count != 2 * self.nht_complete_array_scans:
            raise ValueError("Court NHT preview validation evidence is inconsistent.")
        if self.nht_complete_array_scans > 0 and self.loaded_array_bytes <= 0:
            raise ValueError("Court NHT array scan lacks loaded-byte evidence.")
        if self.nht_invocations == 0:
            if self.maximum_nht_live_array_bytes != 0:
                raise ValueError("Reused Court shards cannot report live NHT arrays.")
        elif not (
            0 < self.maximum_nht_live_array_bytes <= self.loaded_array_bytes
        ):
            raise ValueError("Court maximum live NHT array evidence is inconsistent.")
        if self.retained_nht_array_bytes != 0:
            raise ValueError("Court render results must not retain dense NHT arrays.")
        if self.nht_invocations > self.resolved_shard_count:
            raise ValueError("NHT invocation count exceeds resolved Court shards.")
        if self.request_path_count != self.nht_invocations:
            raise ValueError("Every Court NHT invocation must own exactly one request path.")
        if len(timings) != self.nht_invocations:
            raise ValueError("Court shard timings must cover every NHT invocation.")
        if len({timing.shard_id for timing in timings}) != len(timings):
            raise ValueError("Court shard timing identities must be unique.")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "pre_render_projections", projections)
        object.__setattr__(self, "pre_render_rejected_sample_ids", rejected_ids)
        object.__setattr__(self, "shard_timings", timings)

    @property
    def external_nht_boundary_wall_seconds(self) -> float:
        """Return measured external NHT cost, separate from Court CPU/IO."""
        return float(sum(timing.wall_seconds for timing in self.shard_timings))

    @property
    def projection_by_sample_id(self) -> Mapping[str, MultiCourtProjection]:
        """Return the immutable deterministic pre-render projection inventory."""
        return MappingProxyType(
            {
                projection.camera_id: projection
                for projection in self.pre_render_projections
            }
        )


def group_samples_by_shard(
    samples: Sequence[PlannedCourtSample],
) -> dict[str, tuple[PlannedCourtSample, ...]]:
    """Group samples while proving each trajectory group belongs to one shard."""
    result: dict[str, list[PlannedCourtSample]] = {}
    shard_by_group: dict[str, str] = {}
    for sample in samples:
        previous = shard_by_group.setdefault(sample.trajectory_group_id, sample.shard_id)
        if previous != sample.shard_id:
            raise ValueError("A trajectory group crosses Court render shards.")
        result.setdefault(sample.shard_id, []).append(sample)
    if not result:
        raise ValueError("Court render shards must not be empty.")
    return {
        shard_id: tuple(sorted(values, key=lambda item: item.sample_index))
        for shard_id, values in sorted(result.items())
    }


def rendered_from_nht_records(
    samples: Sequence[PlannedCourtSample],
    records: Sequence[NHTRenderRecord],
) -> tuple[CourtRenderedSample, ...]:
    """Bind exact ordered public renderer records to planned samples."""
    sample_tuple = tuple(samples)
    record_tuple = tuple(records)
    if len(sample_tuple) != len(record_tuple):
        raise ValueError("NHT render result count disagrees with the Court shard plan.")
    rendered: list[CourtRenderedSample] = []
    for sample, record in zip(sample_tuple, record_tuple, strict=True):
        if record.camera_id != sample.sample_id or record.request_source != "arbitrary":
            raise ValueError("NHT record identity/source disagrees with the Court sample.")
        if (record.width, record.height) != (sample.camera.width, sample.camera.height):
            raise ValueError("NHT record resolution disagrees with the Court camera.")
        expected_metadata = (
            (record.rgb_path, (record.height, record.width, 3)),
            (record.alpha_path, (record.height, record.width, 1)),
            (record.depth_path, (record.height, record.width, 1)),
        )
        if any(
            metadata.path != path
            or metadata.shape != shape
            or metadata.dtype != "float32"
            for metadata, (path, shape) in zip(
                record.array_metadata,
                expected_metadata,
                strict=True,
            )
        ):
            raise ValueError("NHT record metadata disagrees with the Court sample files.")
        item = CourtRenderedSample(
            sample=sample,
            rgb_path=record.rgb_path,
            rgb_preview_path=record.rgb_preview_path,
            alpha_path=record.alpha_path,
            alpha_preview_path=record.alpha_preview_path,
            depth_path=record.depth_path,
        )
        inspect_rendered_sample(item)
        rendered.append(item)
    return tuple(rendered)


def inspect_rendered_sample(value: CourtRenderedSample) -> None:
    """Validate file identity and array headers without a complete value scan."""
    height = value.sample.camera.height
    width = value.sample.camera.width
    specifications = (
        (value.rgb_path, "rgb.npy", (height, width, 3)),
        (value.alpha_path, "alpha.npy", (height, width, 1)),
        (value.depth_path, "depth.npy", (height, width, 1)),
    )
    source = value.source_directory
    if source.is_symlink() or not source.is_dir():
        raise NotADirectoryError("NHT sample source directory is unavailable.")
    if source.name != value.sample.sample_id:
        raise ValueError("NHT sample directory disagrees with the planned sample ID.")
    for path, filename, shape in specifications:
        if path.name != filename or path.parent != source:
            raise ValueError("NHT render array path disagrees with its fixed file contract.")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"NHT render array is missing: {path}")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.dtype != np.dtype(np.float32):
            raise TypeError(f"NHT render array must be float32: {path}")
        if array.shape != shape:
            raise ValueError(
                f"NHT render array shape mismatch at {path}: {array.shape} != {shape}."
            )
    for preview, filename in (
        (value.rgb_preview_path, "rgb.png"),
        (value.alpha_preview_path, "alpha.png"),
    ):
        if preview.name != filename or preview.parent != source:
            raise ValueError("NHT preview path disagrees with its fixed file contract.")
        if preview.is_symlink() or not preview.is_file():
            raise FileNotFoundError(f"NHT preview is missing: {preview}")


def validate_rendered_sample(value: CourtRenderedSample) -> None:
    """Perform an explicit independent complete semantic validation pass."""
    inspect_rendered_sample(value)
    specifications = (
        (value.rgb_path, True, False),
        (value.alpha_path, True, False),
        (value.depth_path, False, True),
    )
    for path, unit_range, nonnegative in specifications:
        array = np.load(path, allow_pickle=False)
        if not np.isfinite(array).all():
            raise ValueError(f"NHT render array contains non-finite values: {path}")
        if unit_range and (np.any(array < 0.0) or np.any(array > 1.0)):
            raise ValueError(f"NHT render array must stay in [0, 1]: {path}")
        if nonnegative and np.any(array < 0.0):
            raise ValueError(f"NHT depth must be non-negative: {path}")
    for preview in (value.rgb_preview_path, value.alpha_preview_path):
        try:
            with Image.open(preview) as image:
                size = image.size
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"NHT preview is not a readable image: {preview}") from error
        if size != (value.sample.camera.width, value.sample.camera.height):
            raise ValueError(f"NHT preview resolution mismatch: {preview}")


def write_attempt_shard_marker(
    shard_root: Path,
    *,
    attempt_token: str,
    shard_id: str,
    samples: Sequence[PlannedCourtSample],
) -> Path:
    """Record only attempt-local reuse semantics after public result validation."""
    if not attempt_token or not shard_id:
        raise ValueError("attempt_token and shard_id must be non-empty.")
    sample_tuple = tuple(samples)
    group_ids = sorted({sample.trajectory_group_id for sample in sample_tuple})
    if not sample_tuple or any(sample.shard_id != shard_id for sample in sample_tuple):
        raise ValueError("Shard marker samples disagree with shard_id.")
    payload = {
        "schema": COURT_SHARD_SCHEMA,
        "attempt_token": attempt_token,
        "shard_id": shard_id,
        "trajectory_group_ids": group_ids,
        "sample_ids": [sample.sample_id for sample in sample_tuple],
    }
    return Path(save_json_atomic(payload, shard_root / "court-shard.json"))


def load_attempt_local_shard(
    shard_root: Path,
    *,
    attempt_token: str,
    shard_id: str,
    samples: Sequence[PlannedCourtSample],
) -> tuple[CourtRenderedSample, ...] | None:
    """Reuse only a complete shard bearing the exact in-memory attempt token."""
    marker = shard_root / "court-shard.json"
    if not marker.exists():
        return None
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Court shard marker must be an ordinary file.")
    raw = load_json(marker)
    keys = {
        "schema",
        "attempt_token",
        "shard_id",
        "trajectory_group_ids",
        "sample_ids",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise ValueError("Court shard marker schema is invalid.")
    if raw["schema"] != COURT_SHARD_SCHEMA or raw["shard_id"] != shard_id:
        raise ValueError("Court shard marker schema/shard identity is invalid.")
    if raw["attempt_token"] != attempt_token:
        raise StaleCourtShardError(
            f"Court shard {shard_id} belongs to another stage attempt."
        )
    sample_tuple = tuple(samples)
    expected_ids = [sample.sample_id for sample in sample_tuple]
    expected_groups = sorted({sample.trajectory_group_id for sample in sample_tuple})
    if raw["sample_ids"] != expected_ids or raw["trajectory_group_ids"] != expected_groups:
        raise ValueError("Court shard marker disagrees with the resolved plan.")
    rendered: list[CourtRenderedSample] = []
    for sample in sample_tuple:
        camera_root = shard_root / sample.sample_id
        item = CourtRenderedSample(
            sample=sample,
            rgb_path=camera_root / "rgb.npy",
            rgb_preview_path=camera_root / "rgb.png",
            alpha_path=camera_root / "alpha.npy",
            alpha_preview_path=camera_root / "alpha.png",
            depth_path=camera_root / "depth.npy",
        )
        inspect_rendered_sample(item)
        rendered.append(item)
    return tuple(rendered)


__all__ = [
    "COURT_SHARD_SCHEMA",
    "CourtRenderResult",
    "CourtRenderedSample",
    "CourtShardTiming",
    "StaleCourtShardError",
    "group_samples_by_shard",
    "inspect_rendered_sample",
    "load_attempt_local_shard",
    "rendered_from_nht_records",
    "validate_rendered_sample",
    "write_attempt_shard_marker",
]

"""Strict measured performance evidence for the Court dataset stage."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from src.synthetic_data_generation.dataset.runtime import (
    DatasetPerformanceBudget,
    DatasetPerformanceMetrics,
)

COURT_PERFORMANCE_SCHEMA = "court_dataset_performance_v1"


@dataclass(frozen=True, slots=True)
class CourtPerformanceEvidence:
    """Measured Court budget, processing, frame, byte, and semantic evidence."""

    budget: DatasetPerformanceBudget
    metrics: DatasetPerformanceMetrics
    resolved_shard_count: int
    maximum_shard_sample_count: int
    request_path_count: int
    proposal_count: int
    accepted_frame_count: int
    rejected_frame_count: int
    pre_render_checked_sample_count: int
    pre_render_rejected_sample_count: int
    depth_conversion_count: int
    nht_boundary_complete_array_scans: int
    staged_complete_array_scans: int
    scene_validation_count: int
    preview_validation_count: int
    loaded_array_bytes: int
    external_nht_boundary_wall_seconds: float
    shard_wall_seconds: Mapping[str, float]
    visible_points_by_class: Mapping[str, int]
    schema: str = COURT_PERFORMANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COURT_PERFORMANCE_SCHEMA:
            raise ValueError("Unsupported Court performance evidence schema.")
        if not isinstance(self.budget, DatasetPerformanceBudget):
            raise TypeError("Court performance budget must be DatasetPerformanceBudget.")
        if self.metrics.domain != "court":
            raise ValueError("Court performance metrics must use domain='court'.")
        for name in (
            "resolved_shard_count",
            "maximum_shard_sample_count",
            "proposal_count",
            "accepted_frame_count",
            "pre_render_checked_sample_count",
            "depth_conversion_count",
        ):
            _integer(getattr(self, name), name=name, minimum=1)
        for name in (
            "request_path_count",
            "rejected_frame_count",
            "pre_render_rejected_sample_count",
            "nht_boundary_complete_array_scans",
            "staged_complete_array_scans",
            "scene_validation_count",
            "preview_validation_count",
            "loaded_array_bytes",
        ):
            _integer(getattr(self, name), name=name, minimum=0)
        if self.accepted_frame_count + self.rejected_frame_count != self.proposal_count:
            raise ValueError("Court performance frame inventory is inconsistent.")
        if self.maximum_shard_sample_count > self.proposal_count:
            raise ValueError("Court maximum shard sample count exceeds proposals.")
        if self.pre_render_checked_sample_count != self.proposal_count:
            raise ValueError("Court pre-render gate did not inspect every proposal.")
        if self.pre_render_rejected_sample_count > self.rejected_frame_count:
            raise ValueError(
                "Court pre-render rejections exceed the complete rejection inventory."
            )
        if self.depth_conversion_count != self.accepted_frame_count:
            raise ValueError("Every accepted Court depth must be converted exactly once.")
        if self.budget.maximum_complete_array_scans_per_sample > 2:
            raise ValueError("Court complete-array scan budget is not satisfied.")
        if (
            not self.budget.require_cuda
            or not self.budget.execution_device.startswith("cuda")
        ):
            raise ValueError("Court production performance requires configured CUDA.")
        if self.request_path_count != self.metrics.nht_invocations:
            raise ValueError("Court request paths and NHT invocations must be one-to-one.")
        if (
            self.budget.maximum_nht_invocations > self.resolved_shard_count
            or self.metrics.nht_invocations > self.resolved_shard_count
        ):
            raise ValueError("Court NHT invocations exceed the resolved shard count.")
        if self.maximum_shard_sample_count > self.budget.maximum_batch_frames:
            raise ValueError("Court shard batch exceeds its configured frame budget.")
        if self.metrics.background_cache_misses != 0:
            raise ValueError("Court camera-specific renders cannot report background caches.")
        if (
            self.metrics.frame_count != self.accepted_frame_count
            or self.metrics.camera_count != self.accepted_frame_count
            or self.metrics.sample_count != self.accepted_frame_count
        ):
            raise ValueError("Court performance metrics disagree with accepted samples.")
        if self.metrics.complete_array_scans != (
            self.nht_boundary_complete_array_scans
            + self.staged_complete_array_scans
        ):
            raise ValueError("Court aggregate array-scan evidence is inconsistent.")
        if self.metrics.complete_array_scans != (
            self.proposal_count - self.pre_render_rejected_sample_count
        ):
            raise ValueError(
                "Every renderable Court proposal must be scanned exactly once."
            )
        if self.scene_validation_count not in {0, 1}:
            raise ValueError("Court scene export may be validated at most once.")
        if self.preview_validation_count != 2 * self.nht_boundary_complete_array_scans:
            raise ValueError("Court preview validation evidence is inconsistent.")
        if self.nht_boundary_complete_array_scans > 0 and self.loaded_array_bytes <= 0:
            raise ValueError("Court NHT scan evidence lacks loaded array bytes.")
        if self.metrics.complete_array_scans > (
            self.budget.maximum_complete_array_scans_per_sample
            * self.accepted_frame_count
        ):
            raise ValueError("Court measured array scans exceed the accepted-frame budget.")
        if self.metrics.dense_reference_bytes != self.metrics.published_bytes:
            raise ValueError(
                "Court camera-specific publication must equal its dense reference."
            )
        boundary_seconds = _nonnegative_float(
            self.external_nht_boundary_wall_seconds,
            name="external_nht_boundary_wall_seconds",
        )
        shard_seconds = {
            _identifier(shard_id, name="shard_id"): _nonnegative_float(
                seconds,
                name="shard_wall_seconds",
            )
            for shard_id, seconds in self.shard_wall_seconds.items()
        }
        if len(shard_seconds) != self.metrics.nht_invocations:
            raise ValueError("Court shard timing evidence must cover every invocation.")
        if not math.isclose(
            boundary_seconds,
            sum(shard_seconds.values()),
            abs_tol=1.0e-9,
            rel_tol=1.0e-9,
        ):
            raise ValueError("Court external NHT boundary timing is inconsistent.")
        visible = {
            _identifier(name, name="semantic class"): _integer(
                count,
                name="visible semantic count",
                minimum=1,
            )
            for name, count in self.visible_points_by_class.items()
        }
        if len(visible) != 7:
            raise ValueError("Court performance evidence requires seven visible classes.")
        self.metrics.validate_budget(self.budget)
        object.__setattr__(self, "external_nht_boundary_wall_seconds", boundary_seconds)
        object.__setattr__(self, "shard_wall_seconds", dict(sorted(shard_seconds.items())))
        object.__setattr__(
            self,
            "visible_points_by_class",
            dict(sorted(visible.items())),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict machine-readable evidence payload."""
        return {
            "schema": self.schema,
            "budget": {
                "maximum_wall_seconds": self.budget.maximum_wall_seconds,
                "maximum_published_bytes": self.budget.maximum_published_bytes,
                "maximum_published_fraction_of_dense_reference": (
                    self.budget.maximum_published_fraction_of_dense_reference
                ),
                "maximum_nht_invocations": self.budget.maximum_nht_invocations,
                "maximum_background_cache_misses": (
                    self.budget.maximum_background_cache_misses
                ),
                "maximum_complete_array_scans_per_sample": (
                    self.budget.maximum_complete_array_scans_per_sample
                ),
                "maximum_batch_frames": self.budget.maximum_batch_frames,
                "execution_device": self.budget.execution_device,
                "require_cuda": self.budget.require_cuda,
            },
            "metrics": self.metrics.to_dict(),
            "processing": {
                "request_path_count": self.request_path_count,
                "pre_render_checked_sample_count": self.pre_render_checked_sample_count,
                "pre_render_rejected_sample_count": self.pre_render_rejected_sample_count,
                "nht_boundary_complete_array_scans": (
                    self.nht_boundary_complete_array_scans
                ),
                "staged_complete_array_scans": self.staged_complete_array_scans,
                "scene_validation_count": self.scene_validation_count,
                "preview_validation_count": self.preview_validation_count,
                "loaded_array_bytes": self.loaded_array_bytes,
                "resolved_shard_count": self.resolved_shard_count,
                "maximum_shard_sample_count": self.maximum_shard_sample_count,
                "depth_conversion_count": self.depth_conversion_count,
                "external_nht_boundary_wall_seconds": (
                    self.external_nht_boundary_wall_seconds
                ),
                "external_nht_boundary_wall_seconds_per_camera": (
                    self.external_nht_boundary_wall_seconds
                    / self.nht_boundary_complete_array_scans
                    if self.nht_boundary_complete_array_scans
                    else 0.0
                ),
                "shard_wall_seconds": dict(self.shard_wall_seconds),
            },
            "semantic": {
                "proposal_count": self.proposal_count,
                "accepted_frame_count": self.accepted_frame_count,
                "rejected_frame_count": self.rejected_frame_count,
                "visible_points_by_class": dict(self.visible_points_by_class),
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> CourtPerformanceEvidence:
        """Parse the exact current evidence schema and resolved budget."""
        raw = _mapping(
            value,
            name="Court performance evidence",
            keys={"schema", "budget", "metrics", "processing", "semantic"},
        )
        budget = _mapping(
            raw["budget"],
            name="Court performance budget",
            keys={
                "maximum_wall_seconds",
                "maximum_published_bytes",
                "maximum_published_fraction_of_dense_reference",
                "maximum_nht_invocations",
                "maximum_background_cache_misses",
                "maximum_complete_array_scans_per_sample",
                "maximum_batch_frames",
                "execution_device",
                "require_cuda",
            },
        )
        processing = _mapping(
            raw["processing"],
            name="Court processing evidence",
            keys={
                "request_path_count",
                "pre_render_checked_sample_count",
                "pre_render_rejected_sample_count",
                "nht_boundary_complete_array_scans",
                "staged_complete_array_scans",
                "scene_validation_count",
                "preview_validation_count",
                "loaded_array_bytes",
                "resolved_shard_count",
                "maximum_shard_sample_count",
                "depth_conversion_count",
                "external_nht_boundary_wall_seconds",
                "external_nht_boundary_wall_seconds_per_camera",
                "shard_wall_seconds",
            },
        )
        semantic = _mapping(
            raw["semantic"],
            name="Court semantic performance evidence",
            keys={
                "proposal_count",
                "accepted_frame_count",
                "rejected_frame_count",
                "visible_points_by_class",
            },
        )
        if raw["schema"] != COURT_PERFORMANCE_SCHEMA:
            raise ValueError("Unsupported Court performance evidence schema.")
        performance_budget = DatasetPerformanceBudget(
            maximum_wall_seconds=_positive_float(
                budget["maximum_wall_seconds"],
                name="maximum_wall_seconds",
            ),
            maximum_published_bytes=_integer(
                budget["maximum_published_bytes"],
                name="maximum_published_bytes",
                minimum=1,
            ),
            maximum_published_fraction_of_dense_reference=_positive_float(
                budget["maximum_published_fraction_of_dense_reference"],
                name="maximum_published_fraction_of_dense_reference",
            ),
            maximum_nht_invocations=_integer(
                budget["maximum_nht_invocations"],
                name="maximum_nht_invocations",
                minimum=1,
            ),
            maximum_background_cache_misses=_integer(
                budget["maximum_background_cache_misses"],
                name="maximum_background_cache_misses",
                minimum=1,
            ),
            maximum_complete_array_scans_per_sample=_integer(
                budget["maximum_complete_array_scans_per_sample"],
                name="maximum_complete_array_scans_per_sample",
                minimum=1,
            ),
            maximum_batch_frames=_integer(
                budget["maximum_batch_frames"],
                name="maximum_batch_frames",
                minimum=1,
            ),
            execution_device=_identifier(
                budget["execution_device"],
                name="execution_device",
            ),
            require_cuda=_boolean(budget["require_cuda"], name="require_cuda"),
        )
        proposal_count = _integer(
            semantic["proposal_count"],
            name="proposal_count",
            minimum=1,
        )
        resolved_shards = _integer(
            processing["resolved_shard_count"],
            name="resolved_shard_count",
            minimum=1,
        )
        maximum_shard_sample_count = _integer(
            processing["maximum_shard_sample_count"],
            name="maximum_shard_sample_count",
            minimum=1,
        )
        shard_wall = _mapping(
            processing["shard_wall_seconds"],
            name="shard_wall_seconds",
        )
        visible = _mapping(
            semantic["visible_points_by_class"],
            name="visible_points_by_class",
        )
        boundary_seconds = _nonnegative_float(
            processing["external_nht_boundary_wall_seconds"],
            name="external_nht_boundary_wall_seconds",
        )
        boundary_scans = _integer(
            processing["nht_boundary_complete_array_scans"],
            name="nht_boundary_complete_array_scans",
            minimum=0,
        )
        expected_per_camera = boundary_seconds / boundary_scans if boundary_scans else 0.0
        if not math.isclose(
            _nonnegative_float(
                processing["external_nht_boundary_wall_seconds_per_camera"],
                name="external_nht_boundary_wall_seconds_per_camera",
            ),
            expected_per_camera,
            abs_tol=1.0e-12,
            rel_tol=1.0e-12,
        ):
            raise ValueError("Court per-camera external boundary timing is inconsistent.")
        return cls(
            schema=_identifier(raw["schema"], name="schema"),
            budget=performance_budget,
            metrics=DatasetPerformanceMetrics.from_dict(raw["metrics"]),
            resolved_shard_count=resolved_shards,
            maximum_shard_sample_count=maximum_shard_sample_count,
            request_path_count=_integer(
                processing["request_path_count"],
                name="request_path_count",
                minimum=0,
            ),
            proposal_count=proposal_count,
            accepted_frame_count=_integer(
                semantic["accepted_frame_count"],
                name="accepted_frame_count",
                minimum=1,
            ),
            rejected_frame_count=_integer(
                semantic["rejected_frame_count"],
                name="rejected_frame_count",
                minimum=0,
            ),
            pre_render_checked_sample_count=_integer(
                processing["pre_render_checked_sample_count"],
                name="pre_render_checked_sample_count",
                minimum=1,
            ),
            pre_render_rejected_sample_count=_integer(
                processing["pre_render_rejected_sample_count"],
                name="pre_render_rejected_sample_count",
                minimum=0,
            ),
            depth_conversion_count=_integer(
                processing["depth_conversion_count"],
                name="depth_conversion_count",
                minimum=1,
            ),
            nht_boundary_complete_array_scans=boundary_scans,
            staged_complete_array_scans=_integer(
                processing["staged_complete_array_scans"],
                name="staged_complete_array_scans",
                minimum=0,
            ),
            scene_validation_count=_integer(
                processing["scene_validation_count"],
                name="scene_validation_count",
                minimum=0,
            ),
            preview_validation_count=_integer(
                processing["preview_validation_count"],
                name="preview_validation_count",
                minimum=0,
            ),
            loaded_array_bytes=_integer(
                processing["loaded_array_bytes"],
                name="loaded_array_bytes",
                minimum=0,
            ),
            external_nht_boundary_wall_seconds=boundary_seconds,
            shard_wall_seconds={
                _identifier(key, name="shard_id"): _nonnegative_float(
                    item,
                    name="shard wall seconds",
                )
                for key, item in shard_wall.items()
            },
            visible_points_by_class={
                _identifier(key, name="semantic class"): _integer(
                    item,
                    name="visible semantic count",
                    minimum=1,
                )
                for key, item in visible.items()
            },
        )


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _nonnegative_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative.")
    return float(value)


def _positive_float(value: object, *, name: str) -> float:
    result = _nonnegative_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def _mapping(
    value: object,
    *,
    name: str,
    keys: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    result = dict(value)
    if keys is not None and set(result) != keys:
        raise ValueError(f"{name} fields are missing or unexpected.")
    return result


__all__ = [
    "COURT_PERFORMANCE_SCHEMA",
    "CourtPerformanceEvidence",
]

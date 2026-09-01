"""Streaming MP4 publication for canonical synthetic-dataset views."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.occupancy_artifact import (
    COURT_V4_SUPPORT_OCCUPANCY_COORDINATE_SPACE,
    COURT_V4_SUPPORT_OCCUPANCY_SCHEMA,
    CourtV4SupportOccupancySnapshot,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_DATASET_SCHEMA_V4,
    COURT_PLAN_SCHEMA_V4,
)
from src.synthetic_data_generation.visualization.contracts import (
    VISUALIZATION_METADATA_SCHEMA,
    VISUALIZATION_METADATA_SCHEMA_V2,
    CourtAABBRenderStyle,
    CourtAABBTrajectoryFilterRadiusMode,
    CourtAABBTrajectoryFilterScope,
    CourtOverlayMode,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
    DatasetVisualizationResult,
)
from src.synthetic_data_generation.visualization.court_aabb import (
    COURT_AABB_TRAJECTORY_DISTANCE_METRIC,
    CourtAABBRenderConfig,
    CourtAABBRenderStats,
    CourtAABBTrajectoryFilterResult,
    PreparedCourtAABBGeometry,
    PreparedCourtAABBTrajectoryFilter,
    prepare_court_aabb_trajectory_filter,
    prepare_court_obstacle_aabbs,
    render_prepared_court_obstacle_aabbs,
)
from src.synthetic_data_generation.visualization.overlays import (
    new_ball_history,
    render_blcs_overlay,
    render_court_overlay,
    render_plcs_overlay,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSVisualizationSource,
    CourtVisualizationSource,
    PLCSVisualizationSource,
)
from src.utils.video.writer import VideoWriter


def visualize_dataset(
    request: DatasetVisualizationRequest,
) -> DatasetVisualizationResult:
    """Validate one exact selection, stream overlays, and publish MP4 + JSON."""
    if not isinstance(request, DatasetVisualizationRequest):
        raise TypeError("visualize_dataset requires DatasetVisualizationRequest.")
    frame_order: tuple[Mapping[str, object], ...]
    frame_iterator: Iterator[NDArray[np.uint8]]
    aabb_snapshot: CourtV4SupportOccupancySnapshot | None = None
    aabb_statistics: _CourtAABBAggregateStatistics | None = None
    aabb_filter: PreparedCourtAABBTrajectoryFilter | None = None
    if request.domain is DatasetVisualizationDomain.COURT:
        assert request.trajectory_id is not None
        if request.court_overlay.mode is CourtOverlayMode.SEMANTIC:
            source = CourtVisualizationSource(
                request.dataset_root,
                trajectory_id=request.trajectory_id,
            )
        else:
            source = CourtVisualizationSource(
                request.dataset_root,
                trajectory_id=request.trajectory_id,
                overlay_mode=request.court_overlay.mode,
                maximum_occupancy_cells=request.court_overlay.maximum_cells,
            )
        dataset_schema = source.dataset_schema
        dataset_scene_id = source.dataset_scene_id
        source_width, source_height = source.width, source.height

        if request.court_overlay.mode is CourtOverlayMode.SEMANTIC:
            frame_iterator = (
                render_court_overlay(frame, trajectory_id=request.trajectory_id)
                for frame in source.frames()
            )
        else:
            aabb_snapshot = source.support_occupancy
            if aabb_snapshot is None:
                raise RuntimeError("Court AABB source did not expose occupancy authority.")
            centers = source.trajectory_camera_centers_scene_m
            if centers is None:
                raise RuntimeError(
                    "Court AABB source did not expose selected trajectory geometry."
                )
            aabb_config = _court_aabb_config(
                request,
                voxel_size_m=aabb_snapshot.voxel_size_m,
            )
            aabb_filter = prepare_court_aabb_trajectory_filter(
                aabb_snapshot.cells,
                trajectory_centers_scene_m=centers,
                voxel_size_m=aabb_snapshot.voxel_size_m,
                scope=request.court_overlay.trajectory_filter_scope,
                radius_mode=request.court_overlay.trajectory_filter_radius_mode,
                resolved_radius_m=_resolved_trajectory_filter_radius(
                    source,
                    request=request,
                ),
                maximum_cells=request.court_overlay.maximum_cells,
            )
            aabb_geometry: PreparedCourtAABBGeometry | None = None
            static_filter_result: CourtAABBTrajectoryFilterResult | None = None
            if (
                aabb_filter.scope
                is not CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS
            ):
                static_filter_result = aabb_filter.filter()
                aabb_geometry = prepare_court_obstacle_aabbs(
                    static_filter_result.cells,
                    config=aabb_config,
                )
            aabb_statistics = _CourtAABBAggregateStatistics(
                variable_geometry=(
                    aabb_filter.scope
                    is CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS
                )
            )
            frame_iterator = _court_aabb_frames(
                source,
                geometry=aabb_geometry,
                trajectory_filter=aabb_filter,
                static_filter_result=static_filter_result,
                config=aabb_config,
                aggregate=aabb_statistics,
            )
        frame_order = source.frame_order
        selection: dict[str, object] = {
            "trajectory_id": request.trajectory_id,
            "logical_scene_id": None,
            "camera_id": None,
        }
        source_fps: float | None = None
    elif request.domain is DatasetVisualizationDomain.BLCS:
        assert request.logical_scene_id is not None
        assert request.camera_id is not None
        blcs_source = BLCSVisualizationSource(
            request.dataset_root,
            logical_scene_id=request.logical_scene_id,
            camera_id=request.camera_id,
        )
        dataset_schema = blcs_source.dataset_schema
        dataset_scene_id = blcs_source.dataset_scene_id
        source_width, source_height = blcs_source.width, blcs_source.height
        history = new_ball_history(
            blcs_source.object_ids,
            history_frames=request.history_frames,
        )

        frame_iterator = (
            render_blcs_overlay(
                frame,
                logical_scene_id=request.logical_scene_id,
                camera_id=request.camera_id,
                object_ids=blcs_source.object_ids,
                court_kp=blcs_source.court_kp,
                court_vis=blcs_source.court_vis,
                history=history,
                history_frames=request.history_frames,
            )
            for frame in blcs_source.frames()
        )
        frame_order = blcs_source.frame_order
        selection = {
            "trajectory_id": None,
            "logical_scene_id": request.logical_scene_id,
            "camera_id": request.camera_id,
        }
        source_fps = blcs_source.source_fps
    else:
        assert request.logical_scene_id is not None
        assert request.camera_id is not None
        plcs_source = PLCSVisualizationSource(
            request.dataset_root,
            logical_scene_id=request.logical_scene_id,
            camera_id=request.camera_id,
        )
        dataset_schema = plcs_source.dataset_schema
        dataset_scene_id = plcs_source.dataset_scene_id
        source_width, source_height = plcs_source.width, plcs_source.height

        frame_iterator = (
            render_plcs_overlay(
                frame,
                logical_scene_id=request.logical_scene_id,
                camera_id=request.camera_id,
                object_ids=plcs_source.object_ids,
            )
            for frame in plcs_source.frames()
        )
        frame_order = plcs_source.frame_order
        selection = {
            "trajectory_id": None,
            "logical_scene_id": request.logical_scene_id,
            "camera_id": request.camera_id,
        }
        source_fps = None
    width = source_width + source_width % 2
    height = source_height + source_height % 2
    right_padding = width - source_width
    bottom_padding = height - source_height
    if right_padding or bottom_padding:
        frame_iterator = (
            _pad_frame_for_yuv420(
                frame,
                source_width=source_width,
                source_height=source_height,
                output_width=width,
                output_height=height,
            )
            for frame in frame_iterator
        )
    if len(frame_order) == 0:
        raise ValueError("Visualization selection produced no frames.")
    output = request.output_video
    metadata_path = request.metadata_path
    output.parent.mkdir(parents=True, exist_ok=True)
    for published in (output, metadata_path):
        if published.exists() or published.is_symlink():
            raise FileExistsError(
                f"Visualization publication appeared after request validation: {published}"
            )
    temporary_video = _new_staging_file(output, suffix=".mp4")
    temporary_metadata: _OwnedStagingFile | None = None
    encoded_count = 0
    try:
        with VideoWriter(
            temporary_video.path,
            fps=request.fps,
            crf=request.crf,
        ) as writer:
            for value in frame_iterator:
                writer.write_frame(value)
                encoded_count += 1
        if encoded_count != len(frame_order):
            raise ValueError(
                "Encoded frame count differs from the canonical selected inventory."
            )
        metadata: dict[str, object] = {
            "schema": (
                VISUALIZATION_METADATA_SCHEMA_V2
                if aabb_snapshot is not None
                else VISUALIZATION_METADATA_SCHEMA
            ),
            "domain": request.domain.value,
            "dataset_schema": dataset_schema,
            "dataset_scene_id": dataset_scene_id,
            "selection": selection,
            "frame_count": encoded_count,
            "source_frame_order": list(frame_order),
            "source_width": source_width,
            "source_height": source_height,
            "width": width,
            "height": height,
            "padding": {
                "right": right_padding,
                "bottom": bottom_padding,
            },
            "output_fps": request.fps,
            "source_fps": source_fps,
            "history_frames": request.history_frames,
            "video": {
                "file_name": output.name,
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "crf": request.crf,
            },
        }
        if aabb_snapshot is not None:
            metadata["court_overlay"] = _court_overlay_metadata(
                request,
                snapshot=aabb_snapshot,
                aggregate=aabb_statistics,
                trajectory_filter=aabb_filter,
            )
        temporary_metadata = _new_staging_file(output, suffix=".json")
        temporary_metadata.path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish_exclusively(temporary_metadata.path, metadata_path)
        try:
            _publish_exclusively(temporary_video.path, output)
        except Exception:
            _unlink_if_same_file(metadata_path, temporary_metadata.path)
            raise
    finally:
        temporary_video.cleanup()
        if temporary_metadata is not None:
            temporary_metadata.cleanup()
    return DatasetVisualizationResult(
        video_path=output,
        metadata_path=metadata_path,
        frame_count=encoded_count,
        width=width,
        height=height,
    )


_AABB_GEOMETRY_STAT_FIELDS = (
    "cell_count",
    "surface_face_count",
    "source_triangle_count",
    "candidate_edge_segment_count",
    "edge_segment_count",
    "suppressed_seam_segment_count",
)
_AABB_LOCAL_FILTER_STAT_FIELDS = (
    "filter_segment_count",
    "retained_cell_count",
    "removed_cell_count",
    "surface_face_count",
    "candidate_edge_segment_count",
    "edge_segment_count",
    "suppressed_seam_segment_count",
)


@dataclass(slots=True)
class _CountSummary:
    """Streaming minimum, maximum, and total for one non-negative count."""

    minimum: int | None = None
    maximum: int | None = None
    total: int = 0

    def observe(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Court AABB count summaries require non-negative integers.")
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.total += value

    def to_dict(self) -> dict[str, int]:
        if self.minimum is None or self.maximum is None:
            raise ValueError("Court AABB count summary has no observations.")
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "total": self.total,
        }


@dataclass(slots=True)
class _CourtAABBAggregateStatistics:
    """Bounded aggregate counters retained while frames stream to the encoder."""

    variable_geometry: bool
    frame_count: int = 0
    cell_count: int | None = None
    surface_face_count: int | None = None
    source_triangle_count: int | None = None
    candidate_edge_segment_count: int | None = None
    edge_segment_count: int | None = None
    suppressed_seam_segment_count: int | None = None
    totals: dict[str, int] | None = None
    geometry_summaries: dict[str, _CountSummary] | None = None
    filter_summaries: dict[str, _CountSummary] | None = None
    static_filter: CourtAABBTrajectoryFilterResult | None = None
    count_sequence_digest: str | None = None

    def observe(
        self,
        stats: CourtAABBRenderStats,
        *,
        trajectory_filter: CourtAABBTrajectoryFilterResult,
    ) -> None:
        """Accumulate one complete frame and its exact display subset."""
        if not isinstance(stats, CourtAABBRenderStats):
            raise TypeError("Court AABB aggregate requires CourtAABBRenderStats.")
        if not isinstance(trajectory_filter, CourtAABBTrajectoryFilterResult):
            raise TypeError(
                "Court AABB aggregate requires CourtAABBTrajectoryFilterResult."
            )
        if stats.cell_count != trajectory_filter.retained_cell_count:
            raise ValueError("Court AABB render/filter cell counts disagree.")
        if self.totals is None:
            self.totals = {
                item.name: 0
                for item in fields(stats)
                if item.name not in _AABB_GEOMETRY_STAT_FIELDS
            }
            self.cell_count = stats.cell_count
            self.surface_face_count = stats.surface_face_count
            self.source_triangle_count = stats.source_triangle_count
            self.candidate_edge_segment_count = stats.candidate_edge_segment_count
            self.edge_segment_count = stats.edge_segment_count
            self.suppressed_seam_segment_count = stats.suppressed_seam_segment_count
            self.geometry_summaries = {
                name: _CountSummary() for name in _AABB_GEOMETRY_STAT_FIELDS
            }
            self.filter_summaries = {
                name: _CountSummary() for name in _AABB_LOCAL_FILTER_STAT_FIELDS
            }
            if not self.variable_geometry:
                self.static_filter = trajectory_filter
        elif not self.variable_geometry and (
            self.cell_count != stats.cell_count
            or self.surface_face_count != stats.surface_face_count
            or self.source_triangle_count != stats.source_triangle_count
            or self.candidate_edge_segment_count
            != stats.candidate_edge_segment_count
            or self.edge_segment_count != stats.edge_segment_count
            or self.suppressed_seam_segment_count
            != stats.suppressed_seam_segment_count
        ):
            raise ValueError("Court AABB geometry counts changed between frames.")
        elif not self.variable_geometry and self.static_filter != trajectory_filter:
            raise ValueError("Court AABB static filter changed between frames.")
        assert self.totals is not None
        assert self.geometry_summaries is not None
        assert self.filter_summaries is not None
        for name in self.totals:
            self.totals[name] += cast(int, getattr(stats, name))
        for name, summary in self.geometry_summaries.items():
            summary.observe(cast(int, getattr(stats, name)))
        local_counts = {
            "filter_segment_count": trajectory_filter.filter_segment_count,
            "retained_cell_count": trajectory_filter.retained_cell_count,
            "removed_cell_count": trajectory_filter.removed_cell_count,
            "surface_face_count": stats.surface_face_count,
            "candidate_edge_segment_count": stats.candidate_edge_segment_count,
            "edge_segment_count": stats.edge_segment_count,
            "suppressed_seam_segment_count": stats.suppressed_seam_segment_count,
        }
        for name, summary in self.filter_summaries.items():
            summary.observe(local_counts[name])
        if self.variable_geometry:
            payload = json.dumps(
                tuple(local_counts[name] for name in _AABB_LOCAL_FILTER_STAT_FIELDS),
                separators=(",", ":"),
            ).encode("ascii")
            previous = (
                b""
                if self.count_sequence_digest is None
                else bytes.fromhex(self.count_sequence_digest)
            )
            self.count_sequence_digest = hashlib.sha256(
                previous + b"\n" + payload
            ).hexdigest()
        self.frame_count += 1

    def to_dict(self) -> dict[str, object]:
        """Return exact per-output sums and invariant geometry counts."""
        if (
            self.frame_count <= 0
            or self.cell_count is None
            or self.surface_face_count is None
            or self.source_triangle_count is None
            or self.candidate_edge_segment_count is None
            or self.edge_segment_count is None
            or self.suppressed_seam_segment_count is None
            or self.totals is None
        ):
            raise ValueError("Court AABB rendering produced no aggregate statistics.")
        if self.variable_geometry:
            assert self.geometry_summaries is not None
            geometry: dict[str, object] = {
                name: summary.to_dict()
                for name, summary in self.geometry_summaries.items()
            }
        else:
            geometry = {
                "cell_count": self.cell_count,
                "surface_face_count": self.surface_face_count,
                "source_triangle_count": self.source_triangle_count,
                "candidate_edge_segment_count": self.candidate_edge_segment_count,
                "edge_segment_count": self.edge_segment_count,
                "suppressed_seam_segment_count": self.suppressed_seam_segment_count,
            }
        return {
            "frame_count": self.frame_count,
            "geometry": geometry,
            "totals": dict(self.totals),
        }

    def trajectory_filter_to_dict(
        self,
        prepared: PreparedCourtAABBTrajectoryFilter,
    ) -> dict[str, object]:
        """Return static-v3-compatible or frame-local filter evidence."""
        if self.frame_count <= 0:
            raise ValueError("Court AABB rendering produced no filter statistics.")
        if not self.variable_geometry:
            if self.static_filter is None:
                raise ValueError("Court AABB static filter authority is unavailable.")
            return self.static_filter.to_dict()
        if prepared.scope is not CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS:
            raise ValueError("Variable Court AABB geometry requires local scope.")
        if self.filter_summaries is None or self.count_sequence_digest is None:
            raise ValueError("Court AABB local filter statistics are unavailable.")
        return {
            "scope": prepared.scope.value,
            "radius_mode": (
                None if prepared.radius_mode is None else prepared.radius_mode.value
            ),
            "resolved_radius_m": prepared.resolved_radius_m,
            "distance_metric": COURT_AABB_TRAJECTORY_DISTANCE_METRIC,
            "original_cell_count": prepared.original_cell_count,
            "trajectory_center_count": prepared.trajectory_center_count,
            "trajectory_segment_count": prepared.trajectory_segment_count,
            "closed_trajectory": True,
            "affects_collision_authority": False,
            "frame_count": self.frame_count,
            "frame_local_counts": {
                name: summary.to_dict()
                for name, summary in self.filter_summaries.items()
            },
            "count_sequence_digest": self.count_sequence_digest,
            "count_sequence_digest_algorithm": "sha256-chain-v1",
        }


def _court_aabb_config(
    request: DatasetVisualizationRequest,
    *,
    voxel_size_m: float,
) -> CourtAABBRenderConfig:
    overlay = request.court_overlay
    return CourtAABBRenderConfig(
        voxel_size_m=voxel_size_m,
        render_style=overlay.render_style,
        wireframe_topology=overlay.wireframe_topology,
        near_plane_m=overlay.near_plane_m,
        depth_epsilon_m=overlay.depth_epsilon_m,
        surface_color_rgb=(
            overlay.color_rgb[0] / 255.0,
            overlay.color_rgb[1] / 255.0,
            overlay.color_rgb[2] / 255.0,
        ),
        surface_opacity=overlay.opacity,
        edge_opacity=overlay.edge_opacity,
        edge_width_px=overlay.edge_width_px,
        background_color_rgb=(
            overlay.background_color_rgb[0] / 255.0,
            overlay.background_color_rgb[1] / 255.0,
            overlay.background_color_rgb[2] / 255.0,
        ),
        maximum_cells=overlay.maximum_cells,
        maximum_surface_faces=overlay.maximum_surface_faces,
        maximum_edge_segments=overlay.maximum_edge_segments,
        maximum_projected_pixels=overlay.maximum_projected_pixels,
    )


def _resolved_trajectory_filter_radius(
    source: CourtVisualizationSource,
    *,
    request: DatasetVisualizationRequest,
) -> float | None:
    scope = request.court_overlay.trajectory_filter_scope
    mode = request.court_overlay.trajectory_filter_radius_mode
    configured_radius = request.court_overlay.trajectory_filter_radius_m
    if scope is CourtAABBTrajectoryFilterScope.ALL:
        if mode is not None or configured_radius is not None:
            raise ValueError(
                "all trajectory filtering requires radius mode and radius None."
            )
        return None
    if mode is CourtAABBTrajectoryFilterRadiusMode.SUPPORT_RADIUS:
        policy = source.support_policy
        if policy is None:
            raise RuntimeError(
                "support_radius filtering requires the selected V4 support policy."
            )
        if configured_radius is not None:
            raise ValueError(
                "support_radius filtering does not accept an explicit radius."
            )
        return float(policy.support_radius_m)
    if mode is CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS:
        if configured_radius is None:
            raise ValueError("explicit_radius filtering requires a radius.")
        return float(configured_radius)
    raise ValueError(f"Unknown Court AABB trajectory radius mode: {mode!r}.")


def _court_aabb_frames(
    source: CourtVisualizationSource,
    *,
    geometry: PreparedCourtAABBGeometry | None,
    trajectory_filter: PreparedCourtAABBTrajectoryFilter,
    static_filter_result: CourtAABBTrajectoryFilterResult | None,
    config: CourtAABBRenderConfig,
    aggregate: _CourtAABBAggregateStatistics,
) -> Iterator[NDArray[np.uint8]]:
    """Render exact metric occupancy while retaining only aggregate counters."""
    if geometry is not None and config.voxel_size_m != geometry.voxel_size_m:
        raise ValueError("Court AABB render config changed the geometry voxel size.")
    for frame in source.frames():
        if frame.alpha is None or frame.depth_metric_m is None or frame.camera is None:
            raise ValueError("Court AABB frame lacks alpha, metric depth, or camera.")
        if (
            trajectory_filter.scope
            is CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS
        ):
            frame_filter = trajectory_filter.filter(
                frame_index=frame.trajectory_frame_index
            )
            frame_geometry = prepare_court_obstacle_aabbs(
                frame_filter.cells,
                config=config,
            )
        else:
            if geometry is None or static_filter_result is None:
                raise RuntimeError("Court AABB static geometry was not prepared.")
            frame_filter = static_filter_result
            frame_geometry = geometry
        result = render_prepared_court_obstacle_aabbs(
            rgb=frame.rgb,
            alpha=frame.alpha,
            metric_depth=frame.depth_metric_m,
            camera=frame.camera,
            geometry=frame_geometry,
            config=config,
        )
        aggregate.observe(result.stats, trajectory_filter=frame_filter)
        yield result.rgb


def _court_overlay_metadata(
    request: DatasetVisualizationRequest,
    *,
    snapshot: CourtV4SupportOccupancySnapshot | None,
    aggregate: _CourtAABBAggregateStatistics | None,
    trajectory_filter: PreparedCourtAABBTrajectoryFilter | None,
) -> Mapping[str, object]:
    if (
        request.domain is not DatasetVisualizationDomain.COURT
        or request.court_overlay.mode is not CourtOverlayMode.TRAJECTORY_SUPPORT_AABB
    ):
        raise ValueError("Court AABB metadata requires the explicit AABB mode.")
    if snapshot is None or aggregate is None or trajectory_filter is None:
        raise RuntimeError("Court AABB metadata authority is unavailable.")
    if snapshot.coordinate_space != COURT_V4_SUPPORT_OCCUPANCY_COORDINATE_SPACE:
        raise ValueError("Court AABB occupancy coordinate space changed during render.")
    return {
        "mode": CourtOverlayMode.TRAJECTORY_SUPPORT_AABB.value,
        "wireframe_topology": (
            request.court_overlay.wireframe_topology.value
            if request.court_overlay.render_style is CourtAABBRenderStyle.WIREFRAME
            else None
        ),
        "artifact": {
            "schema": COURT_V4_SUPPORT_OCCUPANCY_SCHEMA,
            "dataset_schema": COURT_DATASET_SCHEMA_V4,
            "plan_schema": COURT_PLAN_SCHEMA_V4,
            "coordinate_space": snapshot.coordinate_space,
            "voxel_size_m": snapshot.voxel_size_m,
            "policy_decision_id": snapshot.policy_decision_id,
            "support_input_digest": snapshot.support_input_digest,
            "cell_count": snapshot.cell_count,
            "content_digest": snapshot.content_digest,
        },
        "trajectory_filter": aggregate.trajectory_filter_to_dict(trajectory_filter),
        "config": request.court_overlay.to_dict(),
        "drawing_statistics": aggregate.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class _OwnedStagingFile:
    """Invocation-private staging path plus its immutable inode identity."""

    path: Path
    device: int
    inode: int

    def cleanup(self) -> None:
        """Remove only the staging inode created by this invocation."""
        try:
            stat = self.path.lstat()
        except FileNotFoundError:
            return
        if (stat.st_dev, stat.st_ino) == (self.device, self.inode):
            self.path.unlink()


def _new_staging_file(output: Path, *, suffix: str) -> _OwnedStagingFile:
    """Create an exclusive invocation-unique sibling for atomic publication."""
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=f".staging{suffix}",
        dir=output.parent,
    )
    try:
        stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return _OwnedStagingFile(
        path=Path(raw_path),
        device=stat.st_dev,
        inode=stat.st_ino,
    )


def _publish_exclusively(staged: Path, target: Path) -> None:
    """Atomically link one staged inode to an absent final name."""
    try:
        os.link(staged, target, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Visualization publication already exists: {target}"
        ) from error


def _unlink_if_same_file(target: Path, staged: Path) -> None:
    """Roll back only a final link still owned by the current staging inode."""
    try:
        target_stat = target.lstat()
        staged_stat = staged.lstat()
    except FileNotFoundError:
        return
    if (target_stat.st_dev, target_stat.st_ino) == (
        staged_stat.st_dev,
        staged_stat.st_ino,
    ):
        target.unlink()


def _pad_frame_for_yuv420(
    frame: NDArray[np.uint8],
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> NDArray[np.uint8]:
    """Pad only the right/bottom edge when canonical RGB dimensions are odd."""
    expected_shape = (source_height, source_width, 3)
    if frame.shape != expected_shape or frame.dtype != np.uint8:
        raise ValueError(
            "Rendered visualization frame differs from its declared source shape: "
            f"expected {expected_shape} uint8, got {frame.shape} {frame.dtype}."
        )
    padded: NDArray[np.uint8] = np.zeros(
        (output_height, output_width, 3), dtype=np.uint8
    )
    padded[:source_height, :source_width] = frame
    return padded


__all__ = ["visualize_dataset"]

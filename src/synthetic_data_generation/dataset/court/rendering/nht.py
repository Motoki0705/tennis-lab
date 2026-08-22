"""Court renderer using only the public ``nht-render`` file boundary."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    validate_camera_looks_at_resolved_court,
    validate_resolved_target_court,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    AmbiguousCameraRelativeNearFarError,
    MultiCourtProjectionAny,
    project_court_semantics_for_version,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    CourtDatasetPlanAny,
    CourtDatasetPlanV2,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.court.shards import (
    CourtRenderedSample,
    CourtRenderResult,
    CourtShardTiming,
    StaleCourtShardError,
    group_samples_by_shard,
    load_attempt_local_shard,
    rendered_from_nht_records,
    write_attempt_shard_marker,
)
from src.synthetic_data_generation.dataset.runtime import directory_size_bytes
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTRenderCamera,
    NHTRenderClient,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True, slots=True)
class CourtPreRenderEvaluation:
    """Deterministic geometry gate over every proposal before NHT execution."""

    projections: tuple[MultiCourtProjectionAny, ...]
    rejected_sample_ids: tuple[str, ...]
    rejection_reasons: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class CourtNHTRenderer:
    """Render deterministic trajectory-group shards through ``NHTRenderClient``."""

    executable: str | Path
    client: NHTRenderClient
    environment: Mapping[str, str]
    timeout_seconds: float

    def preflight(self, scene_path: Path) -> StandardSceneExport:
        """Validate the executable and prime the exact-session scene cache."""
        if scene_path.name != "scene.json" or not scene_path.is_file():
            raise FileNotFoundError(f"Court scene export is unavailable: {scene_path}")
        if isinstance(self.executable, str):
            if self.executable != "nht-render":
                raise ValueError("String Court render executable must be nht-render.")
            if shutil.which(self.executable) is None:
                raise FileNotFoundError("nht-render is unavailable on PATH.")
        elif (
            not isinstance(self.executable, Path)
            or not self.executable.is_absolute()
            or self.executable.name != "nht-render"
            or not self.executable.is_file()
            or not os.access(self.executable, os.X_OK)
        ):
            raise FileNotFoundError(
                f"Court nht-render executable is unavailable: {self.executable}"
            )
        return self.client.validate_scene(scene_path)

    def render(
        self,
        *,
        plan: CourtDatasetPlanAny,
        scene: StandardSceneExport,
        attempt_root: Path,
        attempt_token: str,
        alignment: AlignmentResult,
    ) -> CourtRenderResult:
        """Render or reuse only exact valid shards from this in-memory attempt."""
        if not attempt_token:
            raise ValueError("attempt_token must be non-empty.")
        if scene.scene_id != plan.scene_id:
            raise ValueError("Court plan scene_id disagrees with NHT scene export.")
        _validate_plan_alignment(plan, alignment)
        pre_render = validate_pre_render_plan(plan, alignment=alignment)
        rejected_ids = set(pre_render.rejected_sample_ids)
        renderable_samples = tuple(
            sample for sample in plan.samples if sample.sample_id not in rejected_ids
        )
        if not renderable_samples:
            raise ValueError("Court pre-render gate rejected every planned camera.")
        if not attempt_root.is_absolute() or attempt_root.is_symlink():
            raise ValueError("Court attempt_root must be an absolute ordinary path.")
        attempt_root.mkdir(parents=True, exist_ok=True)
        shards = group_samples_by_shard(renderable_samples)
        if len(shards) > plan.policy.shard_count:
            raise ValueError(
                "Resolved Court shard inventory exceeds the sampling policy."
            )
        rendered: list[CourtRenderedSample] = []
        timings: list[CourtShardTiming] = []
        request_paths: set[Path] = set()
        nht_complete_array_scans = 0
        scene_validation_count = 0
        preview_validation_count = 0
        loaded_array_bytes = 0
        maximum_nht_live_array_bytes = 0
        retained_nht_array_bytes = 0
        for shard_id, samples in shards.items():
            output_directory = attempt_root / "renders" / shard_id
            try:
                reusable = load_attempt_local_shard(
                    output_directory,
                    attempt_token=attempt_token,
                    shard_id=shard_id,
                    samples=samples,
                    schema_version=plan.schema_version,
                )
            except StaleCourtShardError:
                _discard_stale_shard(output_directory, attempt_root=attempt_root)
                reusable = None
            if reusable is not None:
                rendered.extend(reusable)
                continue
            if output_directory.exists():
                if output_directory.is_symlink() or not output_directory.is_dir():
                    raise ValueError(
                        "Partial Court shard output is not an ordinary directory."
                    )
                shutil.rmtree(output_directory)
            request = NHTRenderRequest(
                cameras=tuple(
                    _nht_camera_from_metric_plan(
                        sample.camera,
                        alignment=alignment,
                    )
                    for sample in samples
                )
            )
            request_path = attempt_root / "requests" / f"{shard_id}.json"
            if request_path in request_paths:
                raise ValueError("A Court shard reused an NHT request path.")
            request_paths.add(request_path)
            command = NHTRenderCommandRequest(
                scene_path=scene.scene_path,
                output_directory=output_directory,
                arbitrary_cameras=request,
                arbitrary_request_path=request_path,
                executable=self.executable,
            )
            result = self.client.render(
                command,
                environment=self.environment,
                timeout_seconds=self.timeout_seconds,
            )
            timings.append(
                CourtShardTiming(
                    shard_id=shard_id,
                    camera_count=result.evidence.camera_count,
                    wall_seconds=result.evidence.subprocess_wall_seconds,
                )
            )
            nht_complete_array_scans += result.evidence.complete_payload_scan_count
            scene_validation_count += result.evidence.scene_validation_count
            preview_validation_count += result.evidence.preview_validation_count
            loaded_array_bytes += result.evidence.loaded_array_bytes
            maximum_nht_live_array_bytes = max(
                maximum_nht_live_array_bytes,
                result.evidence.maximum_live_array_bytes,
            )
            retained_nht_array_bytes += result.evidence.retained_array_bytes
            if result.scene_id != plan.scene_id:
                raise ValueError("NHT result scene_id disagrees with the Court plan.")
            shard_rendered = rendered_from_nht_records(samples, result.records)
            write_attempt_shard_marker(
                output_directory,
                attempt_token=attempt_token,
                shard_id=shard_id,
                samples=samples,
                schema_version=plan.schema_version,
            )
            rendered.extend(shard_rendered)
            del result
        rendered.sort(key=lambda item: item.sample.sample_index)
        if [item.sample.sample_id for item in rendered] != [
            sample.sample_id for sample in renderable_samples
        ]:
            raise ValueError(
                "Rendered shard assembly changed the planned sample order."
            )
        return CourtRenderResult(
            samples=tuple(rendered),
            pre_render_projections=pre_render.projections,
            pre_render_rejected_sample_ids=pre_render.rejected_sample_ids,
            resolved_shard_count=plan.policy.shard_count,
            nht_invocations=len(timings),
            request_path_count=len(request_paths),
            maximum_shard_sample_count=max(len(samples) for samples in shards.values()),
            generated_bytes=directory_size_bytes(attempt_root),
            nht_complete_array_scans=nht_complete_array_scans,
            scene_validation_count=scene_validation_count,
            preview_validation_count=preview_validation_count,
            loaded_array_bytes=loaded_array_bytes,
            maximum_nht_live_array_bytes=maximum_nht_live_array_bytes,
            retained_nht_array_bytes=retained_nht_array_bytes,
            shard_timings=tuple(timings),
            pre_render_rejection_reasons=pre_render.rejection_reasons,
        )


def validate_pre_render_plan(
    plan: CourtDatasetPlanAny,
    *,
    alignment: AlignmentResult,
) -> CourtPreRenderEvaluation:
    """Classify invalid cameras before NHT while preserving proposal accounting."""
    _validate_plan_alignment(plan, alignment)
    if plan.proposal_count > plan.policy.proposal_budget:
        raise ValueError("Court plan exceeds its resolved pre-render proposal budget.")
    projections: list[MultiCourtProjectionAny] = []
    rejected: list[str] = []
    rejection_reasons: list[tuple[str, tuple[str, ...]]] = []
    schema_version = (
        plan.schema_version
        if isinstance(plan, CourtDatasetPlan | CourtDatasetPlanV2)
        else CourtDatasetSchemaVersion.V1
    )
    for sample in plan.samples:
        try:
            projection = project_court_semantics_for_version(
                sample.camera,
                alignment.layout,
                schema_version=schema_version,
            )
        except AmbiguousCameraRelativeNearFarError as error:
            rejected.append(sample.sample_id)
            rejection_reasons.append((sample.sample_id, (error.reason,)))
            continue
        in_frame_points = sum(court.in_frame_point_count for court in projection.courts)
        if in_frame_points < 4:
            rejected.append(sample.sample_id)
            rejection_reasons.append(
                (sample.sample_id, ("insufficient_pre_render_semantic_coverage",))
            )
        projections.append(projection)
    return CourtPreRenderEvaluation(
        projections=tuple(projections),
        rejected_sample_ids=tuple(rejected),
        rejection_reasons=tuple(rejection_reasons),
    )


def _discard_stale_shard(output_directory: Path, *, attempt_root: Path) -> None:
    """Remove only a stale shard contained by the current attempt directory."""
    resolved = output_directory.resolve(strict=False)
    root = attempt_root.resolve(strict=True)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Refusing to discard a Court shard outside the attempt root.")
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("Stale Court shard output is not an ordinary directory.")
    shutil.rmtree(output_directory)


def _nht_camera_from_metric_plan(
    camera: SceneCamera,
    *,
    alignment: AlignmentResult,
) -> NHTRenderCamera:
    """Adapt one planned metric pose exactly at the public request boundary."""
    if not isinstance(alignment, AlignmentResult):
        raise TypeError("Court rendering requires a complete AlignmentResult.")
    return NHTRenderCamera(
        camera_id=camera.camera_id,
        width=camera.width,
        height=camera.height,
        intrinsics=camera.intrinsics,
        camera_to_scene=alignment.metric_adapter.nht_from_metric_camera(
            camera.camera_to_scene
        ),
    )


def _validate_plan_alignment(
    plan: CourtDatasetPlanAny,
    alignment: AlignmentResult,
) -> None:
    """Require every planned court binding to come from this alignment result."""
    if not isinstance(alignment, AlignmentResult):
        raise TypeError("Court rendering requires a complete AlignmentResult.")
    if isinstance(plan, CourtDatasetPlanV2):
        groups = {group.trajectory_group_id: group for group in plan.groups}
        for sample in plan.samples:
            group_v2 = groups[sample.trajectory_group_id]
            validate_resolved_target_court(
                policy=group_v2.target_court_policy,
                camera_center_scene_m=sample.camera_center_scene_m,
                target_court=sample.target_court,
                layout=alignment.layout,
            )
            view = next(
                value for value in group_v2.views if value.view_id == sample.view_id
            )
            validate_camera_looks_at_resolved_court(
                camera=sample.camera,
                target_court=sample.target_court,
                layout=alignment.layout,
                look_at_height_m=view.look_at_height_m,
            )
        return
    for group_v1 in plan.groups:
        try:
            court = alignment.layout.court(group_v1.target_court.court_instance_id)
        except KeyError as error:
            raise ValueError(
                "Court plan references a court outside the alignment inventory."
            ) from error
        if (
            group_v1.target_court.candidate_id != court.candidate_id
            or group_v1.target_court.scene_from_court != court.scene_from_court
        ):
            raise ValueError(
                "Court plan binding disagrees with the complete alignment inventory."
            )


__all__ = [
    "CourtNHTRenderer",
    "CourtPreRenderEvaluation",
    "validate_pre_render_plan",
]

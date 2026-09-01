"""Court renderer using only the public ``nht-render`` file boundary."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.dataset.court.components.camera_sampling.anchored_paths import (
    validate_anchored_trajectory_provenance,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    TrajectorySupportModel,
    build_trajectory_support_model,
    evaluate_trajectory_safety,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    validate_camera_looks_at_resolved_court,
    validate_resolved_target_court,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    MultiCourtProjectionAny,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    CourtDatasetPlanAny,
    CourtDatasetPlanV2,
    CourtDatasetPlanV4,
    PathConstructorV4,
    TrajectorySafetyEvaluation,
    TrajectorySemanticPhaseEvaluation,
    build_selected_trajectory_coverage,
    required_coverage_shortfall,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.court.semantic_pre_render import (
    CourtSemanticFrameDisposition,
    court_semantic_phase_disposition_digest,
    evaluate_court_semantic_pre_render,
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
    trajectory_safety_evaluations: tuple[TrajectorySafetyEvaluation, ...]
    trajectory_semantic_phase_evaluations: tuple[TrajectorySemanticPhaseEvaluation, ...]


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
        support_model = _v4_support_model(
            plan,
            scene=scene,
            alignment=alignment,
        )
        pre_render = validate_pre_render_plan(
            plan,
            alignment=alignment,
            support_model=support_model,
        )
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
    support_model: TrajectorySupportModel | None = None,
) -> CourtPreRenderEvaluation:
    """Classify invalid cameras before NHT while preserving proposal accounting."""
    _validate_plan_alignment(plan, alignment)
    if plan.proposal_count > plan.policy.proposal_budget:
        raise ValueError("Court plan exceeds its resolved pre-render proposal budget.")
    projections: list[MultiCourtProjectionAny] = []
    rejected: list[str] = []
    rejection_reasons: list[tuple[str, tuple[str, ...]]] = []
    semantic_dispositions_by_group: dict[str, list[CourtSemanticFrameDisposition]] = {}
    safety_evaluations: tuple[TrajectorySafetyEvaluation, ...] = ()
    if isinstance(plan, CourtDatasetPlanV4):
        if support_model is None:
            raise ValueError(
                "missing_support_capability: V4 pre-render requires its public support model"
            )
        if (
            support_model.policy != plan.support_policy
            or support_model.summary != plan.support_summary
            or support_model.occupancy_snapshot.content_digest
            != plan.support_occupancy_snapshot.content_digest
        ):
            raise ValueError("V4 pre-render support authority disagrees with the plan.")
        selected_coverage = build_selected_trajectory_coverage(
            plan.groups,
            required_raised_lift_m=plan.required_coverage.required_raised_lift_m,
        )
        required_shortfall = required_coverage_shortfall(
            plan.required_coverage,
            selected_coverage,
        )
        if (
            selected_coverage != plan.selected_coverage
            or required_shortfall != plan.required_coverage_shortfall
            or required_shortfall
        ):
            raise ValueError(
                "V4 pre-render required coverage authority disagrees or is incomplete."
            )
        recomputed: list[TrajectorySafetyEvaluation] = []
        for group in plan.groups:
            if (
                group.trajectory.constructor
                is PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
            ):
                validate_anchored_trajectory_provenance(
                    group.trajectory,
                    center=group.center,
                    support_model=support_model,
                )
            path = sample_uniform_arc_length(
                group.trajectory, group.center, plan.policy
            )
            evaluation = evaluate_trajectory_safety(
                trajectory_id=group.trajectory.trajectory_id,
                trajectory_group_id=group.trajectory_group_id,
                path=path,
                support_model=support_model,
            )
            if evaluation != group.safety_evaluation or not evaluation.safe:
                raise ValueError(
                    "V4 pre-render safety evaluation disagrees with selected plan authority."
                )
            group_samples = tuple(
                group_sample
                for group_sample in plan.samples
                if group_sample.trajectory_group_id == group.trajectory_group_id
            )
            expected_frame_indices = tuple(range(len(path.points_scene_m)))
            if (
                len(group_samples) != len(path.points_scene_m)
                or tuple(
                    group_sample.trajectory_frame_index
                    for group_sample in group_samples
                )
                != expected_frame_indices
            ):
                raise ValueError(
                    "V4 pre-render safety/path binding requires the exact canonical "
                    "frame range for every selected group."
                )
            for group_sample in group_samples:
                expected_center = path.points_scene_m[
                    group_sample.trajectory_frame_index
                ]
                if not np.allclose(
                    group_sample.camera_center_scene_m,
                    expected_center,
                    atol=1.0e-9,
                    rtol=0.0,
                ) or not np.allclose(
                    group_sample.camera.camera_to_scene.matrix()[:3, 3],
                    expected_center,
                    atol=1.0e-9,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "V4 pre-render safety/path binding disagrees with a consumed "
                        "sample camera centre."
                    )
            recomputed.append(evaluation)
        safety_evaluations = tuple(recomputed)
    schema_version = (
        plan.schema_version
        if isinstance(plan, CourtDatasetPlan | CourtDatasetPlanV2)
        else CourtDatasetSchemaVersion.V1
    )
    for sample in plan.samples:
        decision = evaluate_court_semantic_pre_render(
            sample.camera,
            alignment.layout,
            schema_version=schema_version,
        )
        if not decision.accepted:
            rejected.append(sample.sample_id)
            rejection_reasons.append((sample.sample_id, decision.rejection_reasons))
        if decision.projection is not None:
            projections.append(decision.projection)
        if isinstance(plan, CourtDatasetPlanV4):
            semantic_dispositions_by_group.setdefault(
                sample.trajectory_group_id, []
            ).append(
                CourtSemanticFrameDisposition(
                    trajectory_frame_index=sample.trajectory_frame_index,
                    camera=sample.camera,
                    decision=decision,
                )
            )
    semantic_phase_evaluations: list[TrajectorySemanticPhaseEvaluation] = []
    if isinstance(plan, CourtDatasetPlanV4):
        for group in plan.groups:
            expected = group.semantic_phase_evaluation
            dispositions = semantic_dispositions_by_group.get(
                group.trajectory_group_id, []
            )
            rejection_counts: Counter[str] = Counter()
            valid_count = 0
            for disposition in dispositions:
                if disposition.decision.accepted:
                    valid_count += 1
                else:
                    if len(disposition.decision.rejection_reasons) != 1:
                        raise ValueError(
                            "V4 semantic phase requires one rejection reason per frame."
                        )
                    rejection_counts.update(disposition.decision.rejection_reasons)
            observed = TrajectorySemanticPhaseEvaluation(
                trajectory_id=group.trajectory.trajectory_id,
                trajectory_group_id=group.trajectory_group_id,
                phase_index=expected.phase_index,
                phase_count=expected.phase_count,
                view=group.views[0],
                expected_frame_count=len(dispositions),
                expected_valid_frame_count=valid_count,
                semantically_viable=valid_count > 0,
                rejection_counts=tuple(sorted(rejection_counts.items())),
                disposition_digest=court_semantic_phase_disposition_digest(
                    dispositions,
                    schema_version=CourtDatasetSchemaVersion.V4,
                    trajectory_group_id=group.trajectory_group_id,
                    phase_index=expected.phase_index,
                    phase_count=expected.phase_count,
                ),
            )
            if observed != expected:
                raise ValueError(
                    "V4 pre-render semantic phase disagrees with selection authority."
                )
            semantic_phase_evaluations.append(observed)
    return CourtPreRenderEvaluation(
        projections=tuple(projections),
        rejected_sample_ids=tuple(rejected),
        rejection_reasons=tuple(rejection_reasons),
        trajectory_safety_evaluations=safety_evaluations,
        trajectory_semantic_phase_evaluations=tuple(semantic_phase_evaluations),
    )


def _v4_support_model(
    plan: CourtDatasetPlanAny,
    *,
    scene: StandardSceneExport,
    alignment: AlignmentResult,
) -> TrajectorySupportModel | None:
    if not isinstance(plan, CourtDatasetPlanV4):
        return None
    metric_cameras = tuple(
        SceneCamera(
            camera_id=camera.camera_id,
            source_frame_index=camera.source_frame_index,
            width=camera.width,
            height=camera.height,
            intrinsics=camera.intrinsics,
            camera_to_scene=alignment.metric_adapter.metric_from_nht_camera(
                camera.camera_to_scene
            ),
            image_path=camera.image_path,
        )
        for camera in scene.cameras
    )
    metric_points = alignment.metric_adapter.metric_from_nht_points(
        scene.points_scene[:, :3]
    )
    return build_trajectory_support_model(
        cameras=metric_cameras,
        points_scene_m=metric_points,
        policy=plan.support_policy,
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

"""Canonical BLCS stage handler for one mutable :class:`SceneWorkspace`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from src.synthetic_data_generation.alignment import AlignmentResult
from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
)
from src.synthetic_data_generation.configuration import BLCSDatasetConfiguration
from src.synthetic_data_generation.dataset.blcs.assembler import (
    BLCSAssemblyResult,
    assemble_blcs_dataset,
    validate_blcs_dataset,
    validate_blcs_dataset_envelope,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSCompositionAssets,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.blcs.rendering import BLCSNHTRenderer
from src.synthetic_data_generation.dataset.blcs.source import BLCSTrajectoryProvider
from src.synthetic_data_generation.dataset.blcs.timeline import build_blcs_plans
from src.synthetic_data_generation.dataset.runtime import PerformanceTimer
from src.synthetic_data_generation.pipeline.contracts import (
    StageExecutionContext,
    StageExecutionSummary,
    StageName,
)
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace
from src.tasks.base.generate_dataset.camera_profiles import CameraProfileConfig

_METADATA_FIELDS = {
    "source_trajectory",
    "source_frame",
    "target_court",
    "candidate_id",
    "transform",
    "camera_profile",
    "camera_parameters",
    "seed",
}


@dataclass(frozen=True, slots=True)
class BLCSDatasetStageHandler:
    """Plan, chunk-render, assemble, and validate every BLCS source frame."""

    workspace: SceneWorkspace
    configuration: BLCSDatasetConfiguration
    camera_configuration: CameraProfileConfig
    seed: int
    assets: BLCSCompositionAssets
    trajectory_provider: BLCSTrajectoryProvider
    renderer: BLCSNHTRenderer
    _attempt_result: BLCSAssemblyResult | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _attempt_output: Path | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("BLCS selection seed must be a non-negative integer.")

    def preflight(self, context: StageExecutionContext) -> None:
        """Validate upstream, config, source, and renderer boundaries before invalidation."""
        self._validate_context(context, require_staging=False)
        if set(self.configuration.metadata_fields) != _METADATA_FIELDS:
            raise ValueError(
                "BLCS metadata_fields must exactly match the emitted canonical provenance."
            )
        alignment = self._load_alignment()
        if not alignment.layout.courts:
            raise ValueError("BLCS requires at least one fit/holdout-accepted court.")
        scene = self.renderer.client.validate_scene(self._scene_path)
        if scene.scene_id != context.request.scene_id:
            raise ValueError("NHT scene_id disagrees with the BLCS stage request.")
        self.trajectory_provider.preflight(
            scene_id=context.request.scene_id, seed=self.seed
        )
        renderer_device = (
            self.renderer.test_cpu_oracle.execution_device
            if self.renderer.test_cpu_oracle is not None
            else self.renderer.execution_device
        )
        if renderer_device != self.configuration.performance.execution_device:
            raise ValueError(
                "BLCS renderer device differs from config-owned performance authority."
            )
        if (
            self.renderer.maximum_batch_frames
            != self.configuration.performance.maximum_batch_frames
        ):
            raise ValueError(
                "BLCS renderer batch bound differs from config-owned performance authority."
            )

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Generate only current-attempt staging and assemble exact fixed-path outputs."""
        self._validate_context(context, require_staging=True)
        performance_timer = PerformanceTimer()
        alignment = self._load_alignment()
        raw_trajectories = tuple(
            self.trajectory_provider.load(
                scene_id=context.request.scene_id,
                seed=self.seed,
            )
        )
        if any(not isinstance(value, BLCSTrajectory) for value in raw_trajectories):
            raise TypeError("BLCS trajectory provider returned an unsupported value.")
        if len(raw_trajectories) < len(alignment.layout.courts):
            raise ValueError(
                "BLCS source scene count cannot cover every accepted court inventory item."
            )
        plans = build_blcs_plans(
            raw_trajectories,
            dataset_scene_id=context.request.scene_id,
            layout=alignment.layout,
            camera_config=self.camera_configuration,
            assets=self.assets,
            seed=self.seed,
            chunk_size_frames=self.configuration.timeline.chunk_size_frames,
        )
        render_attempt = self.renderer.render(
            plans=plans,
            scene_path=self._scene_path,
            samples_directory=context.staging_path / "samples",
            metric_adapter=alignment.metric_adapter,
            attempt_token=(f"{context.request.scene_id}-blcs-{uuid4().hex}"),
        )
        result = assemble_blcs_dataset(
            context.staging_path,
            plans=plans,
            metric_adapter=alignment.metric_adapter,
            render_attempt=render_attempt,
            performance_timer=performance_timer,
            performance_budget=self.configuration.performance,
        )
        object.__setattr__(self, "_attempt_result", result)
        object.__setattr__(self, "_attempt_output", context.staging_path)
        self._validate_alignment_authority(result, alignment=alignment)
        return StageExecutionSummary(
            values={
                "trajectory_count": len(plans),
                "source_frame_count": result.manifest.frame_inventory.source_count,
                "planned_frame_count": len(
                    result.manifest.frame_inventory.planned_indices
                ),
                "rendered_frame_count": len(
                    result.manifest.frame_inventory.rendered_indices
                ),
                "labelled_frame_count": len(
                    result.manifest.frame_inventory.labelled_indices
                ),
                "sample_count": len(result.sample_records),
                "chunk_count": result.continuity.chunk_count,
                "camera_profile": self.camera_configuration.profile,
                "camera_count_per_trajectory": self.camera_configuration.expected_camera_count,
                "target_court_count": len(result.manifest.target_courts),
                "wall_seconds": result.performance.wall_seconds,
                "cpu_seconds": result.performance.cpu_seconds,
                "peak_rss_bytes": result.performance.peak_rss_bytes,
                "execution_device": result.performance.execution_device,
                "cuda_peak_bytes": result.performance.cuda_peak_bytes,
                "nht_invocations": result.performance.nht_invocations,
                "background_cache_misses": (result.performance.background_cache_misses),
                "generated_bytes": result.performance.generated_bytes,
                "published_bytes": result.performance.published_bytes,
            }
        )

    def validate(self, context: StageExecutionContext) -> None:
        """Reload the whole staging dataset before atomic fixed-path publication."""
        self._validate_context(context, require_staging=True)
        alignment = self._load_alignment()
        if (
            self._attempt_result is not None
            and self._attempt_output == context.staging_path
        ):
            validate_blcs_dataset_envelope(context.staging_path)
            result = self._attempt_result
        else:
            result = validate_blcs_dataset(context.staging_path)
        if result.manifest.scene_id != context.request.scene_id:
            raise ValueError("BLCS staged dataset belongs to a different scene.")
        self._validate_alignment_authority(result, alignment=alignment)
        result.performance.validate_budget(self.configuration.performance)

    def _load_alignment(self) -> AlignmentResult:
        """Load the complete accepted inventory and its metric/NHT adapter together."""
        return validate_alignment_outputs(self.workspace.root / "alignment")

    @staticmethod
    def _validate_alignment_authority(
        result: BLCSAssemblyResult,
        *,
        alignment: AlignmentResult,
    ) -> None:
        if result.metric_adapter != alignment.metric_adapter:
            raise ValueError(
                "BLCS dataset metric adapter differs from the accepted alignment."
            )
        accepted = {court.court_instance_id: court for court in alignment.layout.courts}
        bindings = {
            binding.court_instance_id: binding
            for binding in result.manifest.target_courts
        }
        if set(bindings) != set(accepted):
            raise ValueError(
                "BLCS target courts differ from the accepted alignment inventory."
            )
        for court_id, binding in bindings.items():
            court = accepted[court_id]
            if (
                binding.candidate_id != court.candidate_id
                or binding.scene_from_court != court.scene_from_court
            ):
                raise ValueError(
                    "BLCS target-court binding differs from accepted alignment geometry."
                )

    @property
    def _scene_path(self) -> Path:
        return self.workspace.root / "reconstruction" / "export" / "scene.json"

    def _validate_context(
        self,
        context: StageExecutionContext,
        *,
        require_staging: bool,
    ) -> None:
        if context.stage.name is not StageName.BLCS_DATASET:
            raise ValueError("BLCSDatasetStageHandler received a non-BLCS stage.")
        if context.request.scene_id != self.workspace.scene_id:
            raise ValueError("BLCS request and SceneWorkspace scene_id values differ.")
        expected_owner = self.workspace.root / "datasets" / "blcs"
        if context.owner_path != expected_owner:
            raise ValueError("BLCS stage owner is not the canonical workspace path.")
        expected_staging = expected_owner / "staging"
        if context.staging_path != expected_staging:
            raise ValueError(
                "BLCS stage must use only its fixed attempt-local staging path."
            )
        if require_staging and (
            not context.staging_path.is_dir() or context.staging_path.is_symlink()
        ):
            raise ValueError("BLCS staging must be an ordinary existing directory.")


__all__ = [
    "BLCSDatasetStageHandler",
]

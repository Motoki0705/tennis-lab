"""One-command mutable orchestration across NHT, alignment, and datasets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alignment import align_standard_scene
from .config import ScenePipelineConfig
from .datasets import generate_domain_dataset
from .process import run_process
from .scene import StandardScene
from .stages import (
    BY_STAGE,
    ORDER,
    TARGET_STAGE,
    Stage,
    Target,
    execution_order,
    required_dependencies,
)
from .state import SceneRunState
from .workspace import SceneWorkspace, WorkspaceLock, capture_log, remove_path


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """A complete, explicit scene pipeline request."""

    scene_id: str
    config_path: Path
    repository_root: Path
    input_video: Path | None
    from_stage: Stage
    targets: tuple[Target, ...]
    nht_from_stage: str = "frames"

    def __post_init__(self) -> None:
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be unique and non-empty")
        if self.nht_from_stage not in {
            "frames",
            "preprocess",
            "sfm",
            "sfm_selection",
            "nht_training",
            "scene_export",
            "reconstruction_report",
        }:
            raise ValueError("Unsupported nht_from_stage")
        if self.input_video is not None and self.from_stage is not Stage.INGEST:
            raise ValueError("input_video is accepted only when from_stage=ingest")


def _write_snapshot(path: Path, config: ScenePipelineConfig) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config.snapshot(), indent=2) + "\n")
    temporary.replace(path)


def _earliest_config_stage(
    previous: dict[str, Any] | None, current: dict[str, object]
) -> Stage | None:
    if previous is None:
        return Stage.INGEST
    if previous["nht"] != current["nht"]:
        return Stage.RECONSTRUCTION
    if (
        previous["alignment"] != current["alignment"]
        or previous["seed"] != current["seed"]
    ):
        return Stage.ALIGNMENT
    if previous["datasets"] != current["datasets"]:
        # Dataset configuration is shared by all three parallel dataset stages;
        # alignment is their nearest common ancestor in the typed DAG.
        return Stage.ALIGNMENT
    return None


def _earlier(left: Stage, right: Stage | None) -> Stage:
    if right is None:
        return left
    return ORDER[min(ORDER.index(left), ORDER.index(right))]


def _validate_outputs(workspace: SceneWorkspace, stage: Stage, root: Path) -> None:
    missing = [
        str(path)
        for path in BY_STAGE[stage].fixed_outputs
        if not (root / path).exists()
    ]
    if missing:
        raise RuntimeError(f"Stage {stage.value} did not produce outputs: {missing}")


def _ingest(
    workspace: SceneWorkspace, staging: Path, input_video: Path
) -> dict[str, Any]:
    if not input_video.is_absolute() or not input_video.is_file():
        raise ValueError("input_video must be an existing absolute file")
    source = staging / "source"
    source.mkdir(parents=True)
    destination = source / "video.mp4"
    shutil.copy2(input_video, destination)
    metadata = {
        "schema": "tennis_scene_source_v1",
        "scene_id": workspace.scene_id,
        "source_path": str(input_video),
        "video": "video.mp4",
    }
    (source / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"video": "source/video.mp4", "source_path": str(input_video)}


def _original_source(workspace: SceneWorkspace) -> str | None:
    metadata = workspace.path("source/metadata.json")
    if not metadata.is_file():
        return None
    value = json.loads(metadata.read_text()).get("source_path")
    return value if isinstance(value, str) else None


def _reconstruct(
    workspace: SceneWorkspace,
    config: ScenePipelineConfig,
    nht_from_stage: str,
) -> dict[str, Any]:
    reconstruction = workspace.path("reconstruction")
    reconstruction.mkdir(parents=True, exist_ok=True)
    command = [
        *config.nht.reconstruct_command,
        "--scene-id",
        workspace.scene_id,
        "--input-video",
        str(workspace.path("source/video.mp4")),
        "--workspace",
        str(reconstruction),
        "--from-stage",
        nht_from_stage,
    ]
    if config.nht.config is not None:
        command.extend(["--config", str(config.nht.config)])
    run_process(
        command,
        working_directory=config.nht.working_directory,
        environment=config.nht.environment,
    )
    nht_run = json.loads((reconstruction / "run.json").read_text())
    if nht_run.get("status") != "completed":
        raise RuntimeError(
            f"NHT reconstruction did not complete: {nht_run.get('status')}"
        )
    scene = StandardScene.load(reconstruction / "export/scene.json")
    if scene.payload.get("scene_id") != workspace.scene_id:
        raise ValueError("NHT scene export belongs to a different scene_id")
    return {
        "backend": "neural-harmonic-textures-subprocess",
        "nht_run": "reconstruction/run.json",
        "scene_export": "reconstruction/export/scene.json",
        "nht_status": nht_run["status"],
        "nht_stages": {
            name: record["status"] for name, record in nht_run["stages"].items()
        },
        "camera_count": len(scene.cameras),
        "point_count": len(scene.points),
    }


def _alignment(
    workspace: SceneWorkspace,
    staging: Path,
    config: ScenePipelineConfig,
) -> dict[str, Any]:
    scene = StandardScene.load(workspace.path("reconstruction/export/scene.json"))
    output = staging / "alignment/alignment.json"
    result = align_standard_scene(scene, output, config.alignment, config.seed)
    return {
        "alignment": "alignment/alignment.json",
        "status": result["status"],
        "ground_support_fraction": result["support"]["fraction"],
        "holdout_ground_residual_rms_scene": result["holdout_metrics"][
            "ground_residual_rms_scene"
        ],
    }


def _dataset(
    workspace: SceneWorkspace,
    staging: Path,
    config: ScenePipelineConfig,
    target: Target,
) -> dict[str, Any]:
    scene = StandardScene.load(workspace.path("reconstruction/export/scene.json"))
    root = staging / f"datasets/{target.value}"
    result = generate_domain_dataset(
        target,
        scene,
        workspace.path("alignment/alignment.json"),
        root,
        sample_count=config.datasets.samples_per_domain,
        seed=config.seed,
        render_command=config.nht.render_command,
        working_directory=config.nht.working_directory,
        environment=config.nht.environment,
    )
    return {
        "dataset": f"datasets/{target.value}/dataset.json",
        "domain": target.value,
        "sample_count": result["sample_count"],
    }


def _report(
    workspace: SceneWorkspace,
    staging: Path,
    targets: tuple[Target, ...],
) -> dict[str, Any]:
    manifests = {
        target.value: json.loads(
            workspace.path(f"datasets/{target.value}/dataset.json").read_text()
        )
        for target in targets
    }
    alignment = json.loads(workspace.path("alignment/alignment.json").read_text())
    rows = "\n".join(
        f"<tr><td>{domain}</td><td>{manifest['sample_count']}</td>"
        f"<td>{manifest['status']}</td></tr>"
        for domain, manifest in manifests.items()
    )
    output = staging / "report/index.html"
    output.parent.mkdir(parents=True)
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{workspace.scene_id} reconstruction report</title></head><body>"
        f"<h1>{workspace.scene_id}</h1>"
        f"<p>Alignment: {alignment['status']}</p>"
        "<table><thead><tr><th>Domain</th><th>Samples</th><th>Status</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></body></html>\n"
    )
    return {
        "report": "report/index.html",
        "alignment_status": alignment["status"],
        "datasets": {
            domain: manifest["sample_count"] for domain, manifest in manifests.items()
        },
    }


def _target_for_stage(stage: Stage) -> Target:
    return next(
        target for target, target_stage in TARGET_STAGE.items() if target_stage is stage
    )


def _require_dependencies(
    state: SceneRunState, stage: Stage, targets: tuple[Target, ...]
) -> None:
    missing = [
        dependency.value
        for dependency in required_dependencies(stage, targets)
        if state.payload["stages"][dependency.value]["status"] != "completed"
    ]
    if missing:
        raise RuntimeError(f"Stage {stage.value} requires completed stages: {missing}")


def run_scene_pipeline(request: PipelineRequest) -> Path:
    """Execute the request and return the canonical top-level run manifest."""
    repository_root = request.repository_root.resolve()
    config = ScenePipelineConfig.load(request.config_path, repository_root)
    workspace = SceneWorkspace.resolve(config.resolver, request.scene_id)
    with WorkspaceLock(workspace) as lock:
        state = SceneRunState.create_or_load(workspace)
        recovered = state.recover_interrupted()
        if lock.recovered or recovered:
            workspace.cleanup_staging()
        snapshot_path = workspace.path("resolved-config.yaml")
        previous = (
            json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
        )
        effective = _earlier(
            request.from_stage,
            _earliest_config_stage(previous, config.snapshot()),
        )
        if request.input_video is not None:
            previous_source = _original_source(workspace)
            if previous_source is not None and previous_source != str(
                request.input_video.resolve()
            ):
                effective = Stage.INGEST
        if effective is Stage.INGEST and request.input_video is None:
            raise ValueError("An ingest run requires input_video")
        state.request(effective, request.targets, config.seed, request.nht_from_stage)
        workspace.invalidate(effective)
        workspace.cleanup_staging()
        _write_snapshot(snapshot_path, config)
        executed = execution_order(effective, request.targets)
        for stage in executed:
            _require_dependencies(state, stage, request.targets)
            state.running(stage)
            attempt = state.payload["stages"][stage.value]["attempts"]
            log_path = workspace.path(
                Path("logs") / stage.value / f"attempt-{attempt}.log"
            )
            staging: Path | None = None
            try:
                with capture_log(log_path):
                    if stage is Stage.INGEST:
                        assert request.input_video is not None
                        staging = workspace.staging(stage)
                        summary = _ingest(
                            workspace, staging, request.input_video.resolve()
                        )
                    elif stage is Stage.RECONSTRUCTION:
                        summary = _reconstruct(
                            workspace,
                            config,
                            "frames"
                            if effective is Stage.INGEST
                            else request.nht_from_stage,
                        )
                    elif stage is Stage.ALIGNMENT:
                        staging = workspace.staging(stage)
                        summary = _alignment(workspace, staging, config)
                    elif stage in TARGET_STAGE.values():
                        staging = workspace.staging(stage)
                        summary = _dataset(
                            workspace, staging, config, _target_for_stage(stage)
                        )
                    elif stage is Stage.REPORT:
                        staging = workspace.staging(stage)
                        summary = _report(workspace, staging, request.targets)
                    else:  # pragma: no cover - exhaustive StrEnum dispatch
                        raise AssertionError(f"Unhandled stage: {stage}")
                validation_root = workspace.root if staging is None else staging
                _validate_outputs(workspace, stage, validation_root)
                if staging is not None:
                    workspace.publish(stage, staging)
                if stage is Stage.INGEST:
                    if request.input_video is None:
                        raise AssertionError(
                            "Validated ingest request lost input_video"
                        )
                    state.payload["source_video"] = "source/video.mp4"
                summary["log"] = str(log_path.relative_to(workspace.root))
                state.completed(stage, summary)
            except BaseException as error:
                if staging is not None:
                    remove_path(staging)
                workspace.cleanup_staging()
                state.failed(stage, error)
                raise
        state.finish(executed)
    return workspace.path("run.json")

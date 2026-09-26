"""Application boundary for the declared component runner and clip store."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    file_identity,
    standard_definition,
)
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tennis_scene.configuration import PipelineRuntimeConfig


class TennisSceneOrchestrator:
    def __init__(self, config: PipelineRuntimeConfig, *, components: dict[str, Any] | None = None) -> None:
        self.config, self.components = config, components
        self.last_receipt: dict[str, Any] = {}
        self.last_runner: ComponentRunner | None = None
        self.last_store: ClipStore | None = None
        source = PROJECT_ROOT / "src"
        self.code_identity = document_digest({str(path.relative_to(source)): dual_sha256(path) for path in sorted(source.rglob("*.py"))})

    @classmethod
    def from_runtime_config(cls, cfg: PipelineRuntimeConfig) -> TennisSceneOrchestrator:
        return cls(cfg)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TennisSceneOrchestrator:
        from src.tennis_scene.configuration import PipelineRuntimeConfig
        return cls.from_runtime_config(PipelineRuntimeConfig.from_config(cfg))

    def publication_identity(self) -> dict[str, Any]:
        cfg = self.config
        return {"schema": "declared_component_pipeline_v1", "code_sha256": self.code_identity,
            "settings": json_value(cfg.processing_settings), "execution": dict(cfg.component_sources),
            "checkpoints": {key: file_identity(path) for key, path in enabled_model_assets(cfg).items()}}

    def run(self, video_paths: Sequence[Path], *, video_role: PathRole, camera_ids: Sequence[str],
            max_frames: int | None = None, clip_id: str | None = None, store_root: Path | None = None) -> SceneResult:
        paths = tuple(self.config.resolver.validate(video_role, Path(p)) for p in video_paths)
        # Structured clips retain one store next to their source manifest.
        clip_directory = paths[0].parent.parent if paths else None
        if clip_directory is not None and (clip_directory / "clip.json").is_file():
            manifest = json.loads((clip_directory / "clip.json").read_text())
            clip_id = manifest["clip_id"] if clip_id is None else clip_id
            store_root = clip_directory / "annotations/tennis_scene" if store_root is None else store_root
        source = build_clip_source(paths, camera_ids, max_frames=max_frames, clip_id=clip_id)
        if self.config.camera_geometry.reference_camera is not None and self.config.camera_geometry.reference_camera not in source.camera_ids:
            raise ValueError("Reference camera is not a source camera")
        source_document = json_value(source)
        root = store_root or self.config.cache_directory / document_digest(source_document)[:20]
        store = ClipStore(root, source_document)
        nodes = standard_definition(self.config, source, code_identity=self.code_identity, overrides=self.components)
        runner = ComponentRunner(nodes, store, overwrite=self.config.cache_overwrite)
        self.last_runner, self.last_store = runner, store
        self.last_receipt = {"schema": "tennis_scene_run_v3", "status": "running", "clip_id": source.clip_id,
            "source": source_document, "scene_index": str(store.index_path)}
        try:
            runner.run()
            scene: SceneResult = runner.output("scene_assembly")
            self._export(scene, runner, store)
            self.last_receipt.update(status=scene.metadata["status"], validity=scene.metadata["validity_statistics"])
            return scene
        except Exception as exc:
            self.last_receipt.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            self.last_receipt.update(active_stage=runner.active_node, stage_status=runner.statuses,
                stage_seconds=runner.seconds, artifacts=json_value(runner.references))
            write_json_atomic(store.root / "run.json", self.last_receipt)

    @staticmethod
    def _export(scene: SceneResult, runner: ComponentRunner, store: ClipStore) -> None:
        reference = runner.references["scene_assembly"]
        exports = store.root / "exports"
        exports.mkdir(exist_ok=True)
        destination = exports / reference.artifact_id
        if not destination.exists():
            temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=exports))
            try:
                save_scene_result(scene, temporary / "scene.npz")
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        store.record_export("scene", {"scene": destination / "scene.npz", "metadata": destination / "scene.metadata.json"},
            {"scene_assembly": reference})

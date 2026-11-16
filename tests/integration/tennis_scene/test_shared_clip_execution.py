"""Both existing entries share plans, immutable exports and dataset publication."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.generate_dataset.manifest import register_exported_clip
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    generate_pseudo_annotations,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.contracts import AssemblyContext, ComponentIO
from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ComponentNode,
    ExecutionPlan,
)
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathRole
from src.utils.io import save_json_atomic
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import setup_pipeline


class EmptyInput:
    version = 1

    def assemble(self, context: AssemblyContext, artifacts: Any) -> dict[str, Any]:
        return {}


class FailingScene:
    def process(self, inputs: dict[str, Any]) -> SceneResult:
        raise ValueError("clip b failure")


def test_existing_entries_and_multi_clip_plan_publish_the_same_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    clip_id = "video_000/clip_000"
    dataset = tmp_path / "dataset"
    clip = dataset / "videos/video_000/clips/clip_000"
    clip.mkdir(parents=True)
    # The fake predictors use real per-clip stores; only media probing and model
    # predictions are replaced by the existing deterministic fixture.
    for path in paths:
        (clip / path.name).write_bytes(path.read_bytes())
    paths = tuple(clip / path.name for path in paths)
    save_json_atomic(dict(version=2, dataset_id="shared-plan", clip_id=clip_id, video_id="video_000", clip_name="clip_000",
        fps=30., num_frames=24, width=1280, height=720, global_start_sec=0., global_end_sec=.8,
        camera_ids=["cam0", "cam1", "cam2"], video_paths=[p.name for p in paths], cameras=[],
        sync_source="clip_studio", exported_at="2026-11-16T00:00:00Z"), clip / "clip.json")
    register_exported_clip(dataset, clip / "clip.json")
    root = clip / "annotations/tennis_scene"
    expected = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=root, clip_id=clip_id)
    assert pipeline.last_runner is not None and pipeline.last_store is not None
    before = json.loads((root / "scene.json").read_text())
    spec = pipeline.last_runner.plan.clips[clip_id]
    source_b = replace(spec.nodes[0].context.source, clip_id="b")
    failing = FailingScene()
    fail_node = ComponentNode("scene_assembly", failing, ComponentIO("scene_assembly", dict, SceneResult, {}, "scene_result", 2),
                              EmptyInput(), {}, AssemblyContext(source_b), {}, "test")
    failed_store = ClipStore(tmp_path / "failed", json_value(source_b))
    plan = ExecutionPlan([spec, ClipExecution("b", [fail_node], failed_store)])
    executor = PlanExecutor(plan)
    completed: list[str] = []
    def publish_completed(ready_clip: str) -> None:
        assert ready_clip == clip_id
        assert not plan.clip_complete("b")
        executor.finalize_clip(ready_clip)
        completed.append(ready_clip)
    with pytest.raises(ValueError, match="clip b failure"):
        executor.run(on_clip_complete=publish_completed)
    assert completed == [clip_id]
    actual, archive = executor.finalize_clip(clip_id)
    assert archive.is_file()
    finalized = (root / "scene.json").read_bytes()
    executor.finalize_clip(clip_id)
    assert (root / "scene.json").read_bytes() == finalized
    assert not list((failed_store.root / "exports").glob("*/scene.npz"))
    with pytest.raises(ValueError, match="incomplete"):
        executor.finalize_clip("b")
    for field in fields(SceneResult):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    assert plan.statuses[("b", "scene_assembly")] == "failed"
    assert {status for (clip_key, _), status in plan.statuses.items() if clip_key == clip_id} == {"cached"}
    after = json.loads((root / "scene.json").read_text())
    assert after["artifacts"] == before["artifacts"] and after["exports"] == before["exports"]

    calls: list[int] = []
    def run(videos: Sequence[Path], cameras: Sequence[str], directory: Path) -> SceneResult:
        calls.append(1)
        return pipeline.run(videos, video_role=PathRole.DATA, camera_ids=cameras,
                            store_root=directory / "annotations/tennis_scene", clip_id=clip_id)
    # publication_identity normally hashes enabled assets; all component
    # overrides are fake here, so this integration uses their fixed identity.
    first = generate_pseudo_annotations(dataset, run, pipeline_config_yaml="device: cpu\n", publication_identity={"fixture": "fixed"})
    assert first[0].status == "generated" and first[0].annotation_path is not None
    marker = json.loads(first[0].annotation_path.read_text())
    assert marker["scene_result"] == str(archive.relative_to(root)) and marker["scene_index"] == "scene.json"
    assert marker["arrays"]["ball_3d"] == {"shape": [24, 3], "dtype": "float32"}
    assert json.loads((root / "scene.json").read_text())["artifacts"] == before["artifacts"]
    assert all(stage.calls == 1 for stage in stages.values())
    second = generate_pseudo_annotations(dataset, run, pipeline_config_yaml="device: cpu\n", publication_identity={"fixture": "fixed"})
    assert second[0].status == "skipped" and calls == [1]


def test_standalone_input_change_prevents_scene_finalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    stage = stages["ball_detection/cam0"]
    process = stage.process
    def changed(inputs: Any) -> Any:
        result = process(inputs)
        paths[0].write_bytes(b"changed during inference")
        return result
    monkeypatch.setattr(stage, "process", changed)
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="inputs changed during reconstruction"):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=root)
    assert json.loads((root / "scene.json").read_text())["exports"] == {}
    assert pipeline.last_receipt["status"] == "failed"

"""Run real stage adapters through a multi-clip plan and artifact store on CPU."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.submodules.models import PersonDetectionRequest, PersonDetectionResult
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.batching import BatchConfig, BatchMemoryLimit, BatchSizes
from src.tennis_scene.pipeline.components import person_detection
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ComponentNode,
    ExecutionPlan,
)
from src.tennis_scene.pipeline.inference_runtime import ModelSpec, model_identities
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    PersonDetectionInputAssembler,
)
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config


def test_dino_batches_only_compatible_sources_and_preserves_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = {}
    packets = {}
    for clip, frames, shape, values in (
        ("a", 3, (20, 30), ((11, 0, 13), (21, 22, 0))),
        ("b", 2, (40, 60), ((31, 0), (41, 42))),
        ("c", 1, (16, 32), ((51,), (61,))),
    ):
        height, width = shape
        videos[clip] = tuple(SourceVideo(f"cam{camera}", tmp_path / f"{clip}-{camera}.mp4",
                                       f"{clip}-{camera}", frames, 30., width, height)
                             for camera in range(2))
        for video, codes in zip(videos[clip], values, strict=True):
            packets[video.path] = [SimpleNamespace(index=i, frame=np.full((*shape, 3), code, np.uint8))
                                   for i, code in enumerate(codes)]
    monkeypatch.setattr(person_detection, "OpenCVVideoFrameReader", lambda path, **kw: iter(packets[path]))
    batches: list[list[int]] = []

    class Detector:
        def predict_batch(self, requests: Sequence[PersonDetectionRequest], *, batch_size: int) -> list[PersonDetectionResult]:
            assert len(requests) == batch_size
            codes = [int(item.frame_bgr[0, 0, 0]) for item in requests]
            batches.append(codes)
            return [PersonDetectionResult(
                np.array([[code, code + 1, code + 2, code + 3]], np.float32) if code else np.empty((0, 4), np.float32),
                np.array([.8], np.float32) if code else np.empty(0, np.float32)) for code in codes]

    class Component(PersonDetectionModule):
        def model_specs(self) -> dict[str, ModelSpec[Detector]]:
            return {"detector": ModelSpec("dino", {}, "fake", "cpu", "float32", {}, Detector)}

    component = Component(make_people_config(tmp_path))
    settings = {"models": model_identities(component.model_specs(), lambda path: {})}

    def run(batch_size: int, *, max_bytes: int = 1024**3) -> dict:
        executions = []
        for clip_id, cameras in videos.items():
            source = ClipSource(clip_id, cameras)
            assembler = PersonDetectionInputAssembler()
            nodes = tuple(ComponentNode(f"person_detection/{video.camera_id}", component, component.io,
                       assembler, {}, AssemblyContext(source, video.camera_id), settings, "test") for video in cameras)
            executions.append(ClipExecution(clip_id, nodes, ClipStore(tmp_path / f"{batch_size}-{max_bytes}" / clip_id, json_value(source))))
        config = BatchConfig(enabled=True, sizes=replace(BatchSizes(), dino=batch_size),
                             max_sources=6, max_working_bytes=max_bytes)
        executor = PlanExecutor(ExecutionPlan(executions), batching=config)
        try:
            executor.run()
            return {key: json_value(executor.output(key)) for key in executor.plan.required}
        finally:
            assert executor.working_memory.used == 0
            assert executor.working_memory.peak <= max_bytes

    reference = run(1)
    assert all(len(batch) == 1 for batch in batches)
    batches.clear()
    assert run(3) == reference
    assert any(11 in batch and 31 in batch for batch in batches)
    assert any(11 in batch and 21 in batch for batch in batches)
    assert {len(batch) for batch in batches} == {1, 2, 3}
    assert all(not (set(batch) & {51, 61}) or set(batch) <= {51, 61} for batch in batches)
    assert reference[("a", "person_detection/cam0")]["frame_offsets"] == [0, 1, 1, 2]
    assert reference[("a", "person_detection/cam0")]["source_rows"] == [0, 1]
    assert reference[("b", "person_detection/cam0")]["boxes_xyxy"] == [[31., 32., 33., 34.]]
    with pytest.raises(BatchMemoryLimit, match="max_working_bytes"):
        run(3, max_bytes=32)

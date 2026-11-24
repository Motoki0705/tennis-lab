"""Exercise ECPose recipe selection, artifact alignment and cache reuse."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models.ecpose.runtime import ECPoseConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components import ecpose, person_features
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.dataset_publication import annotation_root
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.video.types import FramePacket
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)

OPTIONS = (
    "recipe=ecpose",
    "+ecpose={repository:EdgeCrafter,checkpoint:ecpose/l.pth,model:ecpose-l-o365,"
    "backend:pt32,batch_size:2,confidence:0.3,box_margin:0.1}",
)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("recipe=ecpose",), "requires ecpose configuration"),
        (OPTIONS[1:], "require recipe=ecpose"),
        (
            (*OPTIONS, "ecpose.model=ecdet-l-coco"),
            "requires ecpose-l-o365 or ecpose-x-o365",
        ),
        ((*OPTIONS, "ecpose.backend=pt16"), "requires CUDA"),
    ],
)
def test_recipe_rejects_inactive_or_inconsistent_options(
    tmp_path: Path,
    options: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        runtime(tmp_path, overrides=options)


def test_one_joint_pass_supplies_both_compatible_feature_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = runtime(tmp_path, overrides=OPTIONS)
    assert cfg.ecpose is not None
    assert cfg.output_path.parent == tmp_path / "run/recipes/ecpose"
    upstream = cfg.ecpose.repository / "model.py"
    assets = enabled_model_assets(cfg)
    materialize_assets((*assets.values(), upstream), tmp_path, court=cfg.court_kp)
    assert (
        not cfg.people.dino_checkpoint.exists()
        and not cfg.people.vitpose_checkpoint.exists()
    )
    monkeypatch.setattr(ECPoseConfig, "source_assets", {"source/model.py": upstream})

    frames = [
        FramePacket(index, np.full((10, 20, 3), index, np.uint8), (20, 10))
        for index in range(3)
    ]
    monkeypatch.setattr(
        person_features, "OpenCVVideoFrameReader", lambda *a, **kw: frames
    )
    loads: list[str] = []
    batches: list[int] = []

    def infer(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(pixels)
        batches.append(count)
        logits = torch.full((count, 60, 2), -20.0)
        coordinates = torch.zeros((count, 60, 17, 2))
        logits[0, 11, 1] = 4
        coordinates[0, 11, :2] = torch.tensor([[0.2, 0.2], [0.7, 0.7]])
        coordinates[0, 11, 2] = torch.tensor([-1.0, -1.0])
        coordinates[0, 11, 3:] = torch.tensor([0.4, 0.4])
        if count == 2:
            logits[0, 4, 1] = 3
            coordinates[0, 4, :2] = torch.tensor([[0.6, 0.3], [0.9, 0.8]])
            coordinates[0, 4, 2:] = torch.tensor([0.8, 0.5])
        return logits, coordinates

    def load(*args: Any) -> Any:
        loads.append(args[2])
        return infer

    monkeypatch.setattr(ecpose, "load_backend", load)
    cameras = ("cam0", "cam1", "cam2")
    source = ClipSource(
        "video_000/clip_003",
        tuple(
            SourceVideo(camera, tmp_path / f"{camera}.mp4", "1" * 64, 3, 60, 20, 10)
            for camera in cameras
        ),
    )
    orchestrator = TennisSceneOrchestrator(cfg)
    identity = orchestrator.publication_identity()
    root = annotation_root(tmp_path, identity)
    assert root == tmp_path / "annotations/tennis_scene/recipes/ecpose"
    assert (
        annotation_root(tmp_path, {**identity, "recipe": "standard"})
        == tmp_path / "annotations/tennis_scene"
    )
    assert set(identity["models"]["ecpose"]) == {"ecpose"}
    assert (
        identity["models"]["person_detection"]
        == identity["models"]["person_pose_features"]
        == {}
    )
    assert not {"detector", "vitpose"} & identity["checkpoints"].keys()
    nodes = standard_definition(cfg, source, code_identity=orchestrator.code_identity)
    by_name = {node.name: node for node in nodes}
    detections, poses = "person_detection/cam0", "person_pose_features/cam0"
    assert by_name[detections].bindings == {"joint": "ecpose/cam0"}
    assert by_name[poses].bindings == {"joint": "ecpose/cam0", "detections": detections}
    assert by_name["person_tracking/cam0"].bindings["poses"] == poses
    assert (
        by_name["person_tracking/cam0"].settings["expected_features"]["models"]["pose"][
            "kind"
        ]
        == "ecpose"
    )
    assert by_name["body_placement"].bindings["recovered"] == "gvhmr"
    targets = (poses, "person_pose_features/cam1")
    runner = ComponentRunner(nodes, ClipStore(root, json_value(source)))
    runner.run(targets=targets)
    boxes, features = runner.output(detections), runner.output(poses)
    np.testing.assert_array_equal(boxes.frame_offsets, [0, 2, 2, 3])
    np.testing.assert_array_equal(boxes.source_rows, [0, 1, 2])
    np.testing.assert_array_equal(features.frame_offsets, [0, 2, 2, 3])
    np.testing.assert_array_equal(features.source_rows, [0, 1, 2])
    np.testing.assert_allclose(
        boxes.boxes_xyxy, [[3, 1.5, 15, 7.5], [11.4, 2.5, 18.6, 8.5], [3, 1.5, 15, 7.5]]
    )
    np.testing.assert_allclose(
        features.keypoints[:, 1], [[14, 7, 1], [18, 8, 1], [14, 7, 1]]
    )
    np.testing.assert_array_equal(features.keypoints[0, 2], [0, 0, 0])

    cache = ComponentRunner(nodes, ClipStore(root, json_value(source)))
    cache.run(targets=targets)
    assert set(cache.statuses.values()) == {"cached"}
    assert loads == ["ecpose-l-o365"]
    assert batches == [2, 1, 2, 1]
    changed = replace(
        cfg, ecpose=replace(cfg.ecpose, box_margin=0.15), cache_source="load"
    )
    other_nodes = standard_definition(
        changed, source, code_identity=orchestrator.code_identity
    )
    with pytest.raises(ValueError, match="Loaded component .*ecpose/cam0"):
        ComponentRunner(other_nodes, ClipStore(root, json_value(source))).run(
            targets=(poses,)
        )

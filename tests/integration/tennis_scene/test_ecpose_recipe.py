"""Exercise ECPose recipe selection, artifact alignment and cache reuse."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models.ecpose.releases import RELEASES
from src.submodules.models.ecpose.runtime import ECPoseConfig
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.clip_inputs import (
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.components import ecpose, person_features
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.dataset_publication import (
    annotation_root,
    completed_annotation,
    publish_annotation,
)
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.ecpose_artifacts import require_current_ecpose_artifacts
from src.tennis_scene.pipeline.imports.publish import (
    BoundImport,
    bind_import,
    publish_import,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.video.types import FramePacket
from tests.support.tennis_scene.video_dataset import add_clip
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
    setup_pipeline,
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


@pytest.fixture
def ecpose_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> PipelineRuntimeConfig:
    cfg: PipelineRuntimeConfig = runtime(tmp_path, overrides=OPTIONS)
    assert cfg.ecpose is not None
    upstream = cfg.ecpose.repository / "model.py"
    assets = enabled_model_assets(cfg)
    materialize_assets((*assets.values(), upstream), tmp_path, court=cfg.court_kp)
    monkeypatch.setattr(ECPoseConfig, "source_assets", {"source/model.py": upstream})
    checkpoint = cfg.ecpose.checkpoint
    monkeypatch.setitem(
        RELEASES,
        cfg.ecpose.model_id,
        replace(
            cfg.ecpose.release,
            size=checkpoint.stat().st_size,
            sha256=dual_sha256(checkpoint),
        ),
    )
    return cfg


@pytest.fixture
def clip_source(tmp_path: Path) -> ClipSource:
    return ClipSource(
        "video_000/clip_003",
        tuple(
            SourceVideo(camera, tmp_path / f"{camera}.mp4", "1" * 64, 3, 60, 20, 10)
            for camera in ("cam0", "cam1", "cam2")
        ),
    )


@pytest.fixture
def model_calls(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[int]]:
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
    return loads, batches


def test_one_joint_pass_supplies_both_compatible_feature_artifacts(
    tmp_path: Path,
    ecpose_config: PipelineRuntimeConfig,
    clip_source: ClipSource,
    model_calls: tuple[list[str], list[int]],
) -> None:
    cfg, source = ecpose_config, clip_source
    assert cfg.ecpose is not None
    assert cfg.output_path.parent == tmp_path / "run/recipes/ecpose"
    assert (
        not cfg.people.dino_checkpoint.exists()
        and not cfg.people.vitpose_checkpoint.exists()
    )
    loads, batches = model_calls
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
    load_config = replace(cfg, cache_source="load")
    load_nodes = standard_definition(
        load_config, source, code_identity=orchestrator.code_identity
    )
    loaded = ComponentRunner(load_nodes, ClipStore(root, json_value(source)))
    loaded.run(targets=targets)
    assert set(loaded.statuses.values()) == {"loaded"}
    assert loads == ["ecpose-l-o365"]
    assert batches == [2, 1, 2, 1]
    for changed in (
        replace(cfg.ecpose, box_margin=0.15),
        replace(cfg.ecpose, backend="pt16", device="cuda:1"),
    ):
        other_nodes = standard_definition(
            replace(load_config, ecpose=changed),
            source,
            code_identity=orchestrator.code_identity,
        )
        with pytest.raises(
            ValueError, match="Loaded component identity changed: ecpose/cam0"
        ):
            ComponentRunner(other_nodes, ClipStore(root, json_value(source))).run(
                targets=(poses,)
            )


@pytest.mark.parametrize(
    "stage", ["ecpose", "person_detection", "person_pose_features"]
)
def test_recipe_rejects_imported_people_artifacts(
    tmp_path: Path,
    ecpose_config: PipelineRuntimeConfig,
    clip_source: ClipSource,
    model_calls: tuple[list[str], list[int]],
    stage: str,
) -> None:
    name = f"{stage}/cam0"
    store = ClipStore(tmp_path / "artifacts", json_value(clip_source))
    nodes = standard_definition(ecpose_config, clip_source, code_identity="test")
    runner = ComponentRunner(nodes, store)
    runner.run(targets=(name,))
    value = runner.output(name)
    sources = {**ecpose_config.component_sources, stage: "load"}
    load_nodes = standard_definition(
        replace(ecpose_config, component_sources=sources),
        clip_source,
        code_identity="test",
    )
    node = next(node for node in load_nodes if node.name == name)
    dependencies = {
        port: runner.references[parent] for port, parent in node.bindings.items()
    }
    message = f"{name} requires component origin"
    with pytest.raises(ValueError, match=message):
        bind_import(load_nodes, name, store)
    with pytest.raises(ValueError, match=message):
        publish_import(
            BoundImport(node, dependencies, {}),
            store,
            value,
            importer="test-import",
            version=1,
            identity={"backend": "pt32"},
            provenance={},
        )

    # Previously imported artifacts can have the same schemas and dependencies.
    imported = store.publish(
        name,
        value,
        ArtifactCodec(node.io.output_type),
        schema=node.io.output_schema,
        version=node.io.version,
        dependencies=dependencies,
        identity={"importer": "test-import", "backend": "pt32"},
        provenance={"origin": "import"},
    )
    load = ComponentRunner(load_nodes, store)
    with pytest.raises(ValueError, match=message):
        load.run(targets=(name,))
    assert name not in load.references
    assert store.active(name) == imported
    assert model_calls[0] == ["ecpose-l-o365"]


@pytest.mark.parametrize("entry", ["plan", "publication"])
def test_recipe_rejects_changed_checkpoint_before_inference(
    ecpose_config: PipelineRuntimeConfig,
    clip_source: ClipSource,
    entry: str,
) -> None:
    assert ecpose_config.ecpose is not None
    orchestrator = TennisSceneOrchestrator(ecpose_config)
    orchestrator.publication_identity()  # Populate the shared asset cache with the valid digest.
    checkpoint = ecpose_config.ecpose.checkpoint
    checkpoint.write_bytes(checkpoint.read_bytes()[::-1])

    with pytest.raises(
        ValueError, match="EdgeCrafter weight does not match ecpose_l_o3652coco.pth"
    ):
        if entry == "plan":
            standard_definition(
                ecpose_config, clip_source, code_identity=orchestrator.code_identity
            )
        else:
            orchestrator.publication_identity()


def test_dataset_publication_binds_ecpose_to_the_scene_backend_and_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ecpose_config: PipelineRuntimeConfig,
    model_calls: tuple[list[str], list[int]],
) -> None:
    cfg = ecpose_config
    assert cfg.ecpose is not None
    _, _, stages = setup_pipeline(tmp_path, monkeypatch)
    overrides = {
        name: component
        for name, component in stages.items()
        if not name.startswith(("person_detection/", "person_pose_features/"))
    }
    dataset = tmp_path / "dataset"
    clip = add_clip(dataset, "video_000/clip_000")
    record = select_dataset_clips(dataset, ("video_000/clip_000",))[0]
    inputs = resolve_dataset_clip(dataset, record)
    image = np.zeros((record.height, record.width, 3), np.uint8)
    frames = [
        FramePacket(i, image, (record.width, record.height))
        for i in range(record.num_frames)
    ]
    monkeypatch.setattr(
        person_features, "OpenCVVideoFrameReader", lambda *_a, **_kw: frames
    )
    orchestrator = TennisSceneOrchestrator(cfg, components=overrides)
    identity = orchestrator.publication_identity()
    scene = orchestrator.run(
        inputs.video_paths,
        video_role=PathRole.DATA,
        camera_ids=inputs.camera_ids,
        store_root=annotation_root(clip, {}),
        clip_id=record.clip_id,
    )
    config_yaml = "recipe: ecpose\n"
    marker = publish_annotation(
        inputs, scene, pipeline_config_yaml=config_yaml, publication_identity=identity
    )
    assert completed_annotation(inputs, identity) == marker
    marker_bytes = marker.read_bytes()

    x_path = cfg.ecpose.checkpoint.with_name("x.pth")
    x_path.write_bytes(b"x checkpoint")
    x_model = "ecpose-x-o365"
    monkeypatch.setitem(
        RELEASES,
        x_model,
        replace(
            RELEASES[x_model], size=x_path.stat().st_size, sha256=dual_sha256(x_path)
        ),
    )
    for different in (
        replace(cfg.ecpose, backend="pt16", device="cuda:1"),
        replace(cfg.ecpose, model_id=x_model, checkpoint=x_path),
        replace(cfg.ecpose, box_margin=0.2),
    ):
        wrong = TennisSceneOrchestrator(
            replace(cfg, ecpose=different), components=overrides
        ).publication_identity()
        message = (
            "Published ecpose/cam0 differs from the declared ECPose execution settings"
        )
        with pytest.raises(ValueError, match=message):
            publish_annotation(
                inputs,
                scene,
                pipeline_config_yaml=config_yaml,
                publication_identity=wrong,
            )
        assert marker.read_bytes() == marker_bytes

        # Simulate a marker written before the publication boundary validated the actual scene.
        old_marker = {
            **json.loads(marker_bytes),
            "publication_identity_sha256": document_digest(wrong),
        }
        marker.write_text(json.dumps(old_marker))
        with pytest.raises(ValueError, match=message):
            completed_annotation(inputs, wrong)
        marker.write_bytes(marker_bytes)
    assert completed_annotation(inputs, identity) == marker
    assert model_calls[0] == ["ecpose-l-o365"]

    store, runner = orchestrator.last_store, orchestrator.last_runner
    assert store is not None and runner is not None
    nodes = store.references()
    node = "person_pose_features/cam0"
    pose_ref = nodes[node]
    descriptor = store.descriptor(pose_ref)
    output: person_features.PersonPoseFeaturesOutput = runner.output(node)
    mixed = {
        "joint": nodes["ecpose/cam1"],
        "detections": nodes["person_detection/cam0"],
    }
    store.publish(
        node,
        output,
        ArtifactCodec(type(output)),
        schema=pose_ref.schema,
        version=pose_ref.version,
        identity={**descriptor["identity"], "dependencies": json_value(mixed)},
        dependencies=mixed,
        provenance={"origin": "component"},
    )
    with pytest.raises(
        ValueError,
        match="Published person_pose_features/cam0 has different ECPose dependencies",
    ):
        require_current_ecpose_artifacts(
            store.root,
            json.loads(store.index_path.read_bytes()),
            inputs.camera_ids,
            identity,
        )

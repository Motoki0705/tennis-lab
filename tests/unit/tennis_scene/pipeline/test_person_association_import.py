"""Historical GVHMR axes must match present tracks before confirmed IDs load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.plcs.model_io.person_association import PersonReIDResult
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.person_association import (
    PlayerReIDModule,
    PlayerReIDOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.imports.person_association import (
    import_confirmed_person_association,
)
from src.tennis_scene.pipeline.input_assembly.observations import (
    PersonAssociationInputAssembler,
)
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def _fixture(tmp_path: Path) -> tuple[ClipStore, ClipSource, ComponentNode, Path, Path]:
    cameras = ("cam0", "cam1", "cam2")
    source = ClipSource("confirmed-clip", tuple(SourceVideo(camera, tmp_path / f"{camera}.mp4", camera,
        40, 30., 100, 80) for camera in cameras))
    store = ClipStore(tmp_path / "store", json_value(source))
    dependencies = {"calibration": store.publish("court_calibration", {"value": 1}, ArtifactCodec(dict),
        schema="local_court_calibration", version=1, identity={"calibration": 1}, dependencies={}, provenance={"origin": "test"})}
    for camera in cameras:
        pose_name = f"pose_estimation/{camera}"
        dependencies[f"pose_{camera}"] = store.publish(pose_name, {"camera": camera}, ArtifactCodec(dict),
            schema="person_poses", version=1, identity={"camera": camera}, dependencies={}, provenance={"origin": "test"})
        boxes: np.ndarray = np.zeros((2, 40, 4), np.float32)
        boxes[0] = [10, 10, 20, 30]
        boxes[1] = [70, 10, 80, 30]
        tracks = PersonTrackingOutput(camera, np.array([1, 2], np.int64), boxes,
            np.ones((2, 40), bool), ((1,), (2,)), ())
        store.publish(f"person_tracking/{camera}", tracks, ArtifactCodec(PersonTrackingOutput),
            schema="person_tracks", version=3, identity={"camera": camera}, dependencies={}, provenance={"origin": "test"})
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    old_boxes: np.ndarray = np.zeros((2, 40, 3), np.float64)
    old_boxes[0] = [15, 20, 20]
    old_boxes[1] = [75, 20, 20]
    for camera in cameras:
        (legacy / f"gvhmr_result_{camera}.json").write_text(json.dumps({"track_ids": [1, 2], "bbx_xys": old_boxes.tolist()}))
    association = tmp_path / "player_association_result.json"
    association.write_text(json.dumps({"camera_ids": list(cameras), "canonical_player_ids": [0, 1],
        "segments": [{"start_frame": 0, "end_frame": 40,
                      "assignments": [[0, 0, 1], [1, 1, 0]]}], "reference_camera": "cam0"}))
    component = PlayerReIDModule(tmp_path / "unused.ckpt", camera_ids=cameras)
    node = ComponentNode("person_reid", component, component.io,
        PersonAssociationInputAssembler(.5), {"calibration": "court_calibration",
        **{f"pose_{camera}": f"pose_estimation/{camera}" for camera in cameras}},
        AssemblyContext(source), {}, "test-implementation", "load")
    model = PersonReIDResult(torch.tensor([[0, 1], [2, 0], [2, 0]], dtype=torch.int64),
        torch.tensor([[0, 1], [2, 0], [2, 0]], dtype=torch.int64),
        torch.tensor([[1, 2]] * 3, dtype=torch.int64),
        torch.nn.functional.normalize(torch.arange(1, 25, dtype=torch.float32).reshape(3, 2, 4), dim=-1),
        torch.ones((3, 2), dtype=torch.bool), .775)
    store.publish("person_reid", PlayerReIDOutput(cameras, model), ArtifactCodec(PlayerReIDOutput),
        schema="person_identities", version=1, identity={"model": 1},
        dependencies=dependencies, provenance={"origin": "component"})
    return store, source, node, association, legacy


def test_imports_confirmed_axes_without_changing_model_cosines(tmp_path: Path) -> None:
    store, source, node, association, legacy = _fixture(tmp_path)
    model_ref = store.active("person_reid")
    assert model_ref is not None
    model = store.load(model_ref, ArtifactCodec(PlayerReIDOutput))
    assert model.result is not None

    reference, confirmation = import_confirmed_person_association(node, store, source,
        model_reference=model_ref, historical_association=association, legacy_gvhmr_directory=legacy)

    assert store.active("person_reid") == reference
    imported = store.load(reference, ArtifactCodec(PlayerReIDOutput))
    assert imported.result is not None
    assert imported.result.raw_track_ids.tolist() == [[0, 1], [0, 1], [1, 0]]
    assert torch.equal(imported.result.track_embedding, model.result.track_embedding)
    assert torch.equal(imported.result.track_valid, model.result.track_valid)
    assert imported.result.cosine_threshold == model.result.cosine_threshold
    assert confirmation["axis_to_current_track"] == {camera: [1, 2] for camera in source.camera_ids}
    assert store.descriptor(reference)["provenance"]["model_artifact_id"] == model_ref.artifact_id
    assert store.load(model_ref, ArtifactCodec(PlayerReIDOutput)).result is not None


def test_rejects_legacy_axis_without_unique_track_match(tmp_path: Path) -> None:
    store, source, node, association, legacy = _fixture(tmp_path)
    model_ref = store.active("person_reid")
    assert model_ref is not None
    path = legacy / "gvhmr_result_cam2.json"
    document: dict[str, Any] = json.loads(path.read_text())
    document["bbx_xys"][1] = document["bbx_xys"][0]
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="same current track"):
        import_confirmed_person_association(node, store, source, model_reference=model_ref,
            historical_association=association, legacy_gvhmr_directory=legacy)
    assert store.active("person_reid") == model_ref

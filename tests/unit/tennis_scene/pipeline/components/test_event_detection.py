"""Exercise the sparse event component on a three-frame assembly."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner_3d.model_io.contracts import RefinerPrediction
from src.tennis_scene.ball_events import detect_scene_events
from src.tennis_scene.ball_events.annotation import ANNOTATION_SCHEMA_FILENAME
from src.tennis_scene.pipeline.components.event_detection import (
    EventDetectionConfig,
    EventDetectionInput,
    EventDetectionModule,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.schema import SceneResult

COMMIT = "0" * 40


@pytest.fixture
def assembly(v2_scene: SceneResult, tmp_path: Path) -> EventDetectionInput:
    clip = "video_000/clip_012"
    v2_scene.metadata["dataset_clip_id"] = clip
    source = ClipSource(clip, tuple(
        SourceVideo(camera, tmp_path / f"{camera}.mp4", "b" * 64, 3, 30.0, 640, 360)
        for camera in ("cam1", "cam0")
    ))
    return EventDetectionInput(source, v2_scene)


def _shot(identifier: str, frame: int, hint: str) -> dict[str, Any]:
    return {
        "kind": "shot", "id": identifier, "frame": frame, "player_hint": hint,
        "frame_uncertainty": 2, "status": "probable", "in_point": True, "subtype": "serve",
        "cameras": ["cam1"], "evidence": "Racket changes the ball direction.",
    }


@pytest.fixture
def annotation(tmp_path: Path, assembly: EventDetectionInput) -> tuple[Path, dict[str, Any]]:
    document = {
        "schema": "meiji_event_labels.v1", "benchmark": "multiview-v2",
        "clip_id": assembly.source.clip_id, "fps": 30.0, "frames": 3,
        "notes": "player_hint uses cam0.", "labeler": "ChatGPT (visual multiview review)",
        "role": "scoring", "unlabelable": [{"start": 1, "end": 2, "reason": "occlusion"}],
        "events": [
            _shot("second", 2, "far"),
            {**_shot("bounce", 1, "far"), "kind": "bounce", "subtype": None, "in_point": False},
            _shot("first", 0, "near"),
        ],
    }
    # The published schema is an external input. Use a small schema with the
    # same status constraint to exercise the schema-reading boundary in CI.
    schema = {
        "type": "object", "properties": {
            "events": {"type": "array", "items": {"properties": {
                "status": {"enum": ["certain", "probable", "uncertain"]},
            }}},
        },
    }
    (tmp_path / ANNOTATION_SCHEMA_FILENAME).write_text(json.dumps(schema))
    path = tmp_path / "clip_012_ball_events.json"
    path.write_text(json.dumps(document))
    return path, document


def _camera_center(scene: SceneResult, y: float) -> None:
    scene.metadata["court_reference"] = {
        "reference_camera": "cam1",
        "camera_ids": ["cam1", "cam0"],
        "camera_fits": [
            {"K": np.eye(3).tolist(), "R": np.eye(3).tolist(), "t": [0.0, y, 2.0]},
            {"K": np.eye(3).tolist(), "R": np.eye(3).tolist(), "t": [0.0, -y, 2.0]},
        ],
    }


@pytest.mark.parametrize(("center_y", "sides"), [(-18., [-1, 0, 1]), (18., [1, 0, -1])])
def test_import_preserves_labels_evidence_and_coverage(
    assembly: EventDetectionInput, annotation: tuple[Path, dict[str, Any]],
    center_y: float, sides: list[int],
) -> None:
    path, document = annotation
    _camera_center(assembly.scene, center_y)
    module = EventDetectionModule(
        EventDetectionConfig("annotation", annotation_directory=path.parent), assembly.source.clip_id, code_commit=COMMIT,
    )
    result = module.process(assembly)
    assert result.schema_version == 3 and assembly.scene.schema_version == 2
    assert result.ball_3d is assembly.scene.ball_3d
    np.testing.assert_array_equal(result.ball_event_side, sides)
    np.testing.assert_array_equal(result.ball_event_frame, [0, 1, 2])
    np.testing.assert_array_equal(result.ball_event_kind, [0, 1, 0])
    np.testing.assert_array_equal(result.ball_event_coverage, [True, False, False])
    np.testing.assert_array_equal(result.ball_event_in_point, [True, False, True])
    np.testing.assert_array_equal(result.ball_event_subtype, [0, 7, 0])
    assert result.ball_event_score is None
    assert result.event_evidence["events"][0] == document["events"][2]
    assert "role" not in result.event_evidence
    assert result.metadata["ball_events"]["provenance"] == {
        "annotation_sha256": sha256(path.read_bytes()).hexdigest(),
        "annotation_schema_sha256": sha256((path.parent / ANNOTATION_SCHEMA_FILENAME).read_bytes()).hexdigest(),
        "labeler": "chatgpt", "labeler_original": document["labeler"], "code_commit": COMMIT,
    }


@pytest.mark.parametrize(("field", "value"), [("clip_id", "video_000/clip_002"), ("fps", 30.01), ("frames", 4)])
def test_annotation_rejects_wrong_clip_or_timeline(
    assembly: EventDetectionInput, annotation: tuple[Path, dict[str, Any]], field: str, value: object,
) -> None:
    path, document = annotation
    path.write_text(json.dumps({**document, field: value}))
    module = EventDetectionModule(
        EventDetectionConfig("annotation", annotation_directory=path.parent), assembly.source.clip_id, code_commit=COMMIT,
    )
    with pytest.raises(ValueError, match="clip_id/fps/frames"):
        module.process(assembly)


@pytest.mark.parametrize(("problem", "message"), [
    ("status", "Invalid annotation"),
    ("duplicate_id", "IDs must be unique"),
    ("uncertainty_overflow", "frame_uncertainty"),
    ("interval", "unlabelable interval"),
    ("wrong_view", "other than cam0"),
    ("wrong_media", "video SHA-256"),
    ("net_plane", "center lies on the net"),
    ("changed_asset", "asset changed"),
])
def test_annotation_rejects_invalid_labels_or_reference(
    assembly: EventDetectionInput, annotation: tuple[Path, dict[str, Any]], problem: str, message: str,
) -> None:
    path, document = annotation
    _camera_center(assembly.scene, 0.0 if problem == "net_plane" else -18.0)
    if problem == "status":
        document["events"][0]["status"] = "unknown"
    elif problem == "duplicate_id":
        document["events"][0]["id"] = "first"
    elif problem == "uncertainty_overflow":
        document["events"][0]["frame_uncertainty"] = 256
    elif problem == "interval":
        document["unlabelable"][0]["end"] = 3
    elif problem == "wrong_view":
        document["notes"] = "player_hint uses cam1."
    elif problem == "wrong_media":
        document["notes"] = f"Source SHA-256: clip_012_cam0.mp4={'a' * 64}"
    path.write_text(json.dumps(document))
    module = EventDetectionModule(
        EventDetectionConfig("annotation", annotation_directory=path.parent), assembly.source.clip_id, code_commit=COMMIT,
    )
    if problem == "changed_asset":
        path.write_text(json.dumps({**document, "labeler": "changed"}))
    with pytest.raises(ValueError, match=message):
        module.process(assembly)


def test_rule_uses_original_motion_and_reports_coverage(assembly: EventDetectionInput) -> None:
    assembly.scene.ball_3d = np.array([[0., 0., .4], [0., 0., .1], [0., 0., .2]], np.float32)
    assembly.scene.ball_3d_valid = np.ones(3, dtype=bool)
    result = EventDetectionModule(EventDetectionConfig("rule"), assembly.source.clip_id, code_commit=COMMIT).process(assembly)
    np.testing.assert_array_equal(result.ball_event_frame, [1])
    np.testing.assert_array_equal(result.ball_event_kind, [1])
    np.testing.assert_array_equal(result.ball_event_coverage, [False, True, False])
    assert result.ball_event_status is None
    assert result.ball_event_score is not None and result.ball_event_score[0] > 0
    assert [(event.frame, event.kind) for event in detect_scene_events(result)] == [(1, "bounce")]


def test_refiner_verifies_pin_and_uses_real_frame_grid(
    assembly: EventDetectionInput, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "refiner.ckpt"
    checkpoint.write_bytes(b"fixture weights")
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    invalid = EventDetectionConfig("refiner", checkpoint=checkpoint, checkpoint_sha256="a" * 64, fps_tolerance=.06)
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        EventDetectionModule(invalid, assembly.source.clip_id, code_commit=COMMIT)

    class Predictor:
        def predict(self, coordinates: torch.Tensor, missing: torch.Tensor, **settings: Any) -> RefinerPrediction:
            assert settings == {"fps": 30.0, "seed": 42, "fps_tolerance": .06}
            assert missing.tolist() == [[False, True, True]]
            return RefinerPrediction(coordinates, torch.tensor([[[.1, .2, .3], [.7, .4, .6], [.2, .8, .1]]]))

    monkeypatch.setattr(
        "src.tasks.ball_refiner_3d.inference.predictor.RefinerPredictor.from_checkpoint",
        lambda path, *, device: Predictor() if path == checkpoint and device == "cpu" else pytest.fail("Incorrect predictor asset"),
    )
    config = EventDetectionConfig("refiner", checkpoint=checkpoint, checkpoint_sha256=digest, fps_tolerance=.06)
    result = EventDetectionModule(config, assembly.source.clip_id, code_commit=COMMIT).process(assembly)
    np.testing.assert_array_equal(result.ball_event_frame, [1, 1, 2])
    np.testing.assert_array_equal(result.ball_event_kind, [2, 0, 1])
    np.testing.assert_allclose(result.ball_event_score, [.6, .7, .8])
    np.testing.assert_array_equal(result.ball_event_coverage, [True, True, True])
    assert result.metadata["ball_events"]["provenance"]["checkpoint_sha256"] == digest

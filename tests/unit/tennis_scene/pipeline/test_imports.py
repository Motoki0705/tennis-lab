"""Confirmed-data imports publish through load-only nodes and drive the declared pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.imports.ball_annotations import (
    EXPECTED_VISIBILITY,
    convert_ball_annotation,
    import_ball_annotations,
)
from src.tennis_scene.pipeline.imports.publish import bind_import
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    TrackIdentities,
    camera_stages,
    inputs,
    materialize_assets,
    patch_video_probe,
    runtime,
)

CAMERAS = ("cam0", "cam1", "cam2")


def write_ball_annotation(path: Path, video: SourceVideo, uv: NDArray[np.float32], statuses: list[str]) -> None:
    rows = []
    for frame, status in enumerate(statuses):
        point = None if status == "unresolved" else {"x": float(uv[frame, 0]), "y": float(uv[frame, 1])}
        rows.append({"frame_index": frame, "status": status, "label": "tennis_ball", "track_id": 1,
            "visibility": EXPECTED_VISIBILITY[status], "center_px": point,
            "center_normalized": None if point is None else {"x": point["x"] / video.width, "y": point["y"] / video.height}})
    path.write_text(json.dumps({"schema_version": "video_ball_annotation.v2",
        "source": {"sha256": video.sha256, "width": video.width, "height": video.height, "frame_count": video.num_frames,
                   "fps_numerator": 30, "fps_denominator": 1},
        "coordinate_system": {"origin": "top_left", "frame_index": "zero_based", "normalization": {"x": "x_px / width", "y": "y_px / height"}},
        "target": {"track_id": 1}, "review": {"status": "approved"}, "frames": rows}))


def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, frames: int = 40, ball_load: bool = True, overrides: tuple[str, ...] = ()
          ) -> tuple[Any, ClipSource, ClipStore, tuple[ComponentNode, ...], tuple[BallDetectionOutput, ...]]:
    cfg = runtime(tmp_path, overrides=(*(("execution.ball_detection=load",) if ball_load else ()), *overrides))
    materialize_assets([cfg.ball_detection.checkpoint], tmp_path)
    court, people, balls = inputs(frames=frames)
    stages: dict[str, Any] = {name: stage for name, stage in camera_stages(court, people, balls).items() if not name.startswith("ball_detection/")}
    stages["player_association"] = TrackIdentities(CAMERAS)
    source = build_clip_source(patch_video_probe(tmp_path, monkeypatch, frames), CAMERAS)
    store = ClipStore(tmp_path / "store", json_value(source))
    return cfg, source, store, standard_definition(cfg, source, code_identity="test", overrides=stages), balls


def test_imports_drive_the_declared_pipeline_and_stay_bound_to_their_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, source, store, nodes, balls = build(tmp_path, monkeypatch)
    annotations = tmp_path / "outsource"
    annotations.mkdir()
    statuses = ["observed"] * source.num_frames
    statuses[3], statuses[5] = "interpolated", "unresolved"
    for video, ball in zip(source.videos, balls, strict=True):
        write_ball_annotation(annotations / f"{video.camera_id}_annotations.json", video, ball.uv_px, statuses)
    imported = import_ball_annotations(nodes, store, source, annotations)
    ball = store.load(imported["ball_detection/cam0"], ArtifactCodec(BallDetectionOutput))
    assert ball.point_kind[[3, 5]].tolist() == [2, 0] and not ball.observed[[3, 5]].any()
    assert ball.confidence.tolist() == ball.observed.astype(np.float32).tolist()
    np.testing.assert_allclose(ball.uv_px[3], balls[0].uv_px[3])

    provenance = store.descriptor(imported["ball_detection/cam0"])["provenance"]
    assert provenance["origin"] == "import" and provenance.get("model_inference", False) is False

    runner = ComponentRunner(nodes, store)
    runner.run()
    assert {runner.statuses[n] for n in imported} == {"loaded"}
    # The association runs as a component, after the side it depends on.
    assert runner.statuses["player_association"] == "executed"
    # The side is decided by the component from the imported ball, not imported.
    assert runner.statuses["court_side"] == "executed"
    side = runner.output("court_side")
    # The fixture's cam2 court is half-turned; all four hypotheses were scored.
    assert side.view_half_turns == (False, False, True) and len(side.hypotheses) == 4
    scene = runner.output("scene_assembly")
    assert scene.player_track_ids.tolist() == [0] and scene.player_kp_3d_vis.any()
    assert scene.metadata["court_reference"]["view_half_turns"] == [False, False, True]

    # Replacing an imported ball re-decides the side from the new artifact.
    import_ball_annotations(nodes, store, source, annotations)  # same bytes: same artifacts
    assert ComponentRunner(nodes, store).run() and store.active("court_side") == runner.references["court_side"]
    write_ball_annotation(annotations / "cam1_annotations.json", source.video("cam1"), balls[1].uv_px, ["observed"] * source.num_frames)
    import_ball_annotations(nodes, store, source, annotations)
    rerun = ComponentRunner(nodes, store)
    rerun.run()
    assert rerun.statuses["court_side"] == "executed" and rerun.references["court_side"] != runner.references["court_side"]


def test_imports_require_a_load_only_node_and_adopted_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, source, store, nodes, _ = build(tmp_path, monkeypatch, ball_load=False)
    with pytest.raises(ValueError, match="load-only"):
        bind_import(nodes, "ball_detection/cam0", store)
    with pytest.raises(ValueError, match="load-only"):
        bind_import(nodes, "player_association", store)
    with pytest.raises(ValueError, match="load-only"):
        bind_import(nodes, "court_side", store)
    (tmp_path / "loaded").mkdir()
    _, _, store, nodes, _ = build(tmp_path / "loaded", monkeypatch, ball_load=False, overrides=("execution.court_side=load",))
    with pytest.raises(FileNotFoundError, match="court_calibration"):
        bind_import(nodes, "court_side", store)
    with pytest.raises(ValueError, match="Unknown import target"):
        bind_import(nodes, "person_reid", store)


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda d: d["source"].update(sha256="0" * 64), "does not identify"),
    (lambda d: d["frames"][2].update(visibility="occluded"), "disagrees"),
    (lambda d: d["frames"][2]["center_normalized"].update(x=.9), "disagree"),
    (lambda d: d["frames"][2]["center_px"].update(x=100.01), "disagree"),
    (lambda d: d["frames"].pop(), "every frame once"),
    (lambda d: d["frames"][1].update(center_px={"x": 5000., "y": 1.}), "outside"),
])
def test_ball_annotation_contract_violations_stop(tmp_path: Path, mutation: Any, message: str) -> None:
    video = SourceVideo("cam0", tmp_path / "cam0.mp4", "a" * 64, 6, 30., 1280, 720)
    path = tmp_path / "cam0_annotations.json"
    write_ball_annotation(path, video, np.full((6, 2), 100, np.float32), ["observed"] * 6)
    document = json.loads(path.read_text())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        convert_ball_annotation(path, video)


def test_ball_annotation_accepts_pixels_rounded_after_normalization(tmp_path: Path) -> None:
    # Meiji video_001/clip_000 rounds center_px to 3 decimals after computing center_normalized.
    video = SourceVideo("cam0", tmp_path / "cam0.mp4", "a" * 64, 6, 30., 1920, 1080)
    path = tmp_path / "cam0_annotations.json"
    write_ball_annotation(path, video, np.full((6, 2), 354.3155, np.float32), ["observed"] * 6)
    document = json.loads(path.read_text())
    for row in document["frames"]:
        row["center_normalized"] = {"x": round(354.3155 / 1920, 9), "y": round(354.3155 / 1080, 9)}
        row["center_px"] = {"x": 354.315, "y": 354.315}
    path.write_text(json.dumps(document))
    ball, _ = convert_ball_annotation(path, video)
    np.testing.assert_allclose(ball.uv_px, 354.315, rtol=0, atol=1e-4)

"""Confirmed-data imports publish through load-only nodes and drive the declared pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.plcs.data.manual_association import (
    PlayerAssociationResult,
    PlayerAssociationSegment,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.imports.ball_annotations import (
    EXPECTED_VISIBILITY,
    convert_ball_annotation,
    import_ball_annotations,
)
from src.tennis_scene.pipeline.imports.court_side import import_ball_confirmed_sides
from src.tennis_scene.pipeline.imports.person_association import (
    confirmed_identities,
    import_confirmed_person_association,
    match_legacy_axes,
)
from src.tennis_scene.pipeline.imports.publish import bind_import
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
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


def write_legacy_gvhmr(path: Path, track_ids: list[int], boxes: NDArray[np.float64]) -> None:
    # Field order and surrounding payload mimic the large legacy files the reader scans.
    path.write_text(json.dumps({"smpl_params": {"body_pose": [[0.0] * 6]}, "track_ids": track_ids, "bbx_xys": boxes.tolist()}))


def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, frames: int = 40, ball_load: bool = True
          ) -> tuple[Any, ClipSource, ClipStore, tuple[ComponentNode, ...], tuple[BallDetectionOutput, ...]]:
    cfg = runtime(tmp_path, overrides=("execution.ball_detection=load",) if ball_load else ())
    materialize_assets([cfg.ball_detection.checkpoint], tmp_path)
    court, people, balls = inputs(frames=frames)
    stages = {name: stage for name, stage in camera_stages(court, people, balls).items() if not name.startswith("ball_detection/")}
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

    upstream = ("court_calibration", *(f"pose_estimation/{c}" for c in CAMERAS))
    ComponentRunner(nodes, store).run(targets=upstream)
    side, confirmation = import_ball_confirmed_sides(nodes, store, source, ball_threshold=cfg.ball_detection.score_threshold,
        ball_reprojection_px=cfg.ball_reprojection_px, max_frames=cfg.sampling_max_frames, config=cfg.camera_geometry)
    # The fixture's cam2 court is half-turned; all four hypotheses were scored.
    assert confirmation["view_half_turns"] == [False, False, True] and len(confirmation["candidates"]) == 4

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for camera in CAMERAS:
        write_legacy_gvhmr(legacy / f"gvhmr_result_{camera}.json", [11], np.tile([0., 0., 100.], (1, source.num_frames, 1)))
    history = tmp_path / "player_association_result.json"
    PlayerAssociationResult(list(CAMERAS), np.array([5], np.int32),
        [PlayerAssociationSegment(0, source.num_frames, np.zeros((1, 3), np.int32))], "cam0").save(history)
    person, document = import_confirmed_person_association(nodes, store, source,
        historical_association=history, legacy_gvhmr_directory=legacy)
    assert document["player_id_matrix"] == [[5], [5], [5]]
    for reference in (imported["ball_detection/cam0"], side, person):
        provenance = store.descriptor(reference)["provenance"]
        assert provenance["origin"] == "import" and provenance.get("model_inference", False) is False

    runner = ComponentRunner(nodes, store)
    runner.run()
    assert {runner.statuses[n] for n in (*imported, "court_side", "player_association")} == {"loaded"}
    scene = runner.output("scene_assembly")
    assert scene.player_track_ids.tolist() == [5] and scene.player_kp_3d_vis.any()
    assert scene.metadata["court_reference"]["view_half_turns"] == [False, False, True]

    # An import records its bound inputs: replacing one invalidates it explicitly.
    import_ball_annotations(nodes, store, source, annotations)  # same bytes: same artifacts
    write_ball_annotation(annotations / "cam1_annotations.json", source.video("cam1"), balls[1].uv_px, ["observed"] * source.num_frames)
    import_ball_annotations(nodes, store, source, annotations)
    with pytest.raises(ValueError, match="dependencies changed: court_side"):
        ComponentRunner(nodes, store).run()


def test_imports_require_a_load_only_node_and_adopted_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, source, store, nodes, _ = build(tmp_path, monkeypatch, ball_load=False)
    with pytest.raises(ValueError, match="load-only"):
        bind_import(nodes, "ball_detection/cam0", store)
    with pytest.raises(FileNotFoundError, match="court_calibration"):
        bind_import(nodes, "court_side", store)
    with pytest.raises(ValueError, match="Unknown import target"):
        bind_import(nodes, "person_reid", store)


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda d: d["source"].update(sha256="0" * 64), "does not identify"),
    (lambda d: d["frames"][2].update(visibility="occluded"), "disagrees"),
    (lambda d: d["frames"][2]["center_normalized"].update(x=.9), "disagree"),
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


def _poses(camera: str, centres: list[list[float]], observed: list[int], track_ids: list[int], frames: int = 40) -> ObjectObservations:
    carriers = len(track_ids)
    boxes: NDArray[np.float32] = np.zeros((1, frames, carriers, 3), np.float32)
    boxes[0, :, :, :2] = np.asarray(centres, np.float32)[None]
    boxes[..., 2] = 100
    mask: NDArray[np.bool_] = np.zeros((1, frames, carriers), bool)
    for carrier, count in enumerate(observed):
        mask[0, :count, carrier] = True
    uv: NDArray[np.float32] = np.zeros((1, frames, carriers, 17, 2), np.float32)
    return ObjectObservations((camera,), (1280, 720), 30., uv, np.ones(uv.shape[:-1], np.float32), mask,
                              np.asarray([track_ids], np.int64), boxes)


def _legacy(centres: list[list[float]], frames: int = 40) -> NDArray[np.float64]:
    boxes = np.zeros((len(centres), frames, 3))
    boxes[:, :, :2] = np.asarray(centres)[:, None]
    boxes[..., 2] = 100
    return boxes


def test_identities_follow_legacy_axes_and_record_every_exclusion() -> None:
    source = ClipSource("clip", tuple(SourceVideo(c, Path(f"{c}.mp4"), c * 20, 40, 30., 1280, 720) for c in CAMERAS))
    # cam1 is uncalibrated: it is left out of the identity rows.
    calibration = cast(CourtCalibrationOutput, SimpleNamespace(calibration=SimpleNamespace(views=(
        SimpleNamespace(source_index=0), SimpleNamespace(source_index=2)))))
    poses = {
        # Carriers: player 0, player 1, a short fragment near player 0, an unobserved track,
        # a legacy-tracked person the record leaves unassigned, and an unrelated person.
        "pose_cam0": _poses("cam0", [[100, 100], [900, 500], [110, 100], [0, 0], [600, 600], [1200, 50]],
                            [40, 40, 10, 0, 40, 40], [1, 2, 3, 4, 5, 6]),
        "pose_cam1": _poses("cam1", [[100, 100]], [40], [7]),
        # Carrier order differs from the legacy axis order in cam2.
        "pose_cam2": _poses("cam2", [[600, 300], [100, 100]], [40, 40], [4, 9]),
    }
    legacy = {"cam0": _legacy([[100, 100], [900, 500], [600, 600]]), "cam1": _legacy([[100, 100], [300, 300]]),
              "cam2": _legacy([[100, 100], [600, 300]])}
    association = PlayerAssociationResult(list(CAMERAS), np.array([0, 1], np.int32),
        [PlayerAssociationSegment(0, 40, np.array([[0, 0, 0], [1, 1, 1]], np.int32))], "cam0")
    identities, document = confirmed_identities(source, calibration, poses, association, legacy)
    assert identities.camera_ids == ("cam0", "cam2")
    assert identities.local_track_ids.tolist() == [[1, 2, 3, 4, 5, 6], [4, 9, -1, -1, -1, -1]]
    assert identities.player_ids.tolist() == [[0, 1, -1, -1, -1, -1], [1, 0, -1, -1, -1, -1]]
    assert document["excluded_cameras"] == ["cam1"]
    assert [x["disposition"] for x in document["unassigned_tracks"]["cam0"]] == [
        "excluded_insufficient_observations", "excluded_unobserved", "excluded_unassigned_legacy_axis", "excluded_non_target"]


def test_identity_matching_stops_on_missing_or_ambiguous_tracks() -> None:
    legacy = _legacy([[100, 100]])
    with pytest.raises(ValueError, match="no close current track"):
        match_legacy_axes(legacy, _poses("cam0", [[400, 400]], [40], [1]))
    with pytest.raises(ValueError, match="no close current track"):  # too few observations to match
        match_legacy_axes(legacy, _poses("cam0", [[100, 100]], [29], [1]))
    with pytest.raises(ValueError, match="multiple current tracks"):
        match_legacy_axes(legacy, _poses("cam0", [[100, 100], [130, 100]], [40, 40], [1, 2]))
    with pytest.raises(ValueError, match="same current track"):
        match_legacy_axes(_legacy([[100, 100], [110, 100]]), _poses("cam0", [[100, 100]], [40], [1]))

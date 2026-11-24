"""Exercise the fixed-input import at its real component, archive and marker boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.generate_dataset.manifest import (
    DatasetClipRecord,
    DatasetManifest,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.assembly import assemble_automatic_scene
from src.tennis_scene.pipeline.clip_inputs import resolve_dataset_clip
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.camera_geometry import (
    describe_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    publish_annotation,
)
from src.tennis_scene.pipeline.imports import external_ball_publication
from src.tennis_scene.pipeline.imports.external_ball_geometry import (
    SAVED_GEOMETRY,
    SHARED_GEOMETRY,
    PinnedBallCourt,
)
from src.tennis_scene.pipeline.imports.external_ball_publication import (
    PublicationKind,
    ScenePointers,
    current_publication,
    publish_scene_revision,
    restore_publication,
)
from src.tennis_scene.pipeline.imports.external_ball_update import update_clip
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.scripts.import_external_ball import main
from src.utils.configuration import PathContractError
from src.utils.io import load_json
from src.utils.video import VideoInfo
from tests.integration.tennis_scene.test_video_court_side import build, prefix_stores


def _saved_clip(tmp_path, monkeypatch, mode):
    pipeline, _, (source, _), court, pins = prefix_stores(
        tmp_path, monkeypatch, "agreement"
    )
    side = build(pipeline, court, pins, tmp_path)
    pinned = PinnedBallCourt.load(tmp_path / "side", side.digest, court.root, None)
    cal = pins[source.clip_id].assemble().calibration

    record = DatasetClipRecord(
        source.clip_id,
        "video_000",
        "clip_a",
        f"videos/{source.clip_id.rsplit('/', 1)[0]}/clips/clip_a",
        len(source.videos),
        source.num_frames,
        source.fps,
        *source.size,
    )
    dataset = tmp_path / "dataset"
    clip = dataset / record.path
    (clip / "media").mkdir(parents=True)
    videos = []
    for video in source.videos:
        path = clip / "media" / f"{video.camera_id}.mp4"
        path.hardlink_to(video.path)
        videos.append(replace(video, path=path))
    source = replace(source, videos=tuple(videos))
    write_json_atomic(
        clip / "clip.json",
        {
            "version": 2,
            "dataset_id": "fixture",
            "clip_id": record.clip_id,
            "video_id": record.video_id,
            "clip_name": record.clip_name,
            "fps": record.fps,
            "num_frames": record.num_frames,
            "width": record.width,
            "height": record.height,
            "global_start_sec": 0,
            "global_end_sec": record.num_frames / record.fps,
            "camera_ids": source.camera_ids,
            "video_paths": [str(p.relative_to(clip)) for p in source.paths],
            "cameras": [],
            "sync_source": "fixture",
            "exported_at": "2026-11-24T00:00:00Z",
        },
    )
    store = ClipStore(clip / ANNOTATION_RELATIVE_DIR, json_value(source))
    cal_ref = store.publish(
        "court_calibration",
        cal,
        ArtifactCodec(CourtCalibrationOutput),
        schema="local_court_calibration",
        version=1,
        identity={"test": "calibration"},
        dependencies={},
        provenance={"origin": "component"},
    )
    decision = side.output()
    cameras = tuple(
        view.camera.half_turned(turn)
        for view, turn in zip(
            cal.calibration.views, decision.view_half_turns, strict=True
        )
    )
    geometry = describe_camera_geometry(
        cal.calibration,
        decision.reference_camera,
        decision.view_half_turns,
        cameras,
        {},
    )
    truth = np.array([0.35, 0.95, 1.3])
    streams = {}
    for i, camera in enumerate(cameras):
        pixel, front = camera.project(truth)
        assert front
        key = f"{record.clip_name}_{camera.camera_id}"
        frames = [
            {
                "frame": t,
                "x": float(pixel[0]) if i == 0 or t < record.num_frames - 1 else -1,
                "y": float(pixel[1]),
                "status": ("observed", "estimated", "candidate_supported")[i],
                "visible_in_frame": False,
            }
            for t in range(record.num_frames)
        ]
        streams[key] = {
            "input_file": f"{key}.mp4",
            "frame_count": record.num_frames,
            "fps": record.fps,
            "width": record.width,
            "height": record.height,
            "frames": frames,
        }
    positions = tmp_path / "clip_a_ball_positions.json"
    write_json_atomic(
        positions,
        {
            "cameras": streams,
            "metadata": {
                "fps": record.fps,
                "total_cameras": record.num_cameras,
                "frames_per_camera": record.num_frames,
            },
        },
    )
    if mode == "fallback":
        write_json_atomic(
            clip / "annotations/tennis_scene.failure.json",
            {
                "clip_id": record.clip_id,
                "error": "ReconstructionUnavailable: side evidence insufficient",
            },
        )
        return pinned, record, positions, store, None, truth

    if mode == "conflict":
        geometry = replace(geometry, view_half_turns=(False, False, False))
    al_ref = store.publish(
        "camera_alignment",
        CameraAlignmentOutput(geometry),
        ArtifactCodec(CameraAlignmentOutput),
        schema="aligned_cameras",
        version=1,
        identity={"test": "alignment"},
        dependencies={"calibration": cal_ref},
        provenance={"origin": "component"},
    )
    shape = (1, record.num_cameras, record.num_frames, 17)
    people = GroupedObservations(
        np.array([42], np.int64),
        np.full((*shape, 2), 12.5, np.float32),
        np.ones(shape, np.float32),
        np.ones(shape, bool),
        np.zeros(shape[:3], np.int64),
    )
    scene = assemble_automatic_scene(
        video_paths=source.paths,
        camera_ids=source.camera_ids,
        info=VideoInfo(source.fps, *source.size, source.num_frames),
        court=cal.court,
        active_indices=(0, 1, 2),
        geometry=geometry,
        grouped_people=people,
        skeleton=None,
        players=None,
        ball=None,
        metadata={"status": "ok", "pipeline_contract": "declared_components_v1"},
    )
    ref = store.publish(
        "scene_assembly",
        scene,
        ArtifactCodec(SceneResult),
        schema="scene_result",
        version=2,
        identity={"test": "old scene"},
        dependencies={"calibration": cal_ref, "alignment": al_ref},
        provenance={"origin": "component"},
    )
    export_scene(store, scene, ref)
    publish_annotation(
        resolve_dataset_clip(dataset, record),
        scene,
        pipeline_config_yaml="ball: disabled",
        publication_identity={"schema": "fixture"},
    )
    return pinned, record, positions, store, scene, truth


@pytest.mark.parametrize("mode", ["saved", "fallback", "conflict"])
def test_external_ball_swaps_both_pointers_and_reuses_the_saved_scene(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    court, record, positions, store, player_scene, truth = _saved_clip(
        tmp_path, monkeypatch, mode
    )
    before = ScenePointers.read(store.root)
    old_refs = store.references()
    (tmp_path / "hardlinked-scene.json").hardlink_to(store.index_path)
    if before.annotation:
        (tmp_path / "hardlinked-annotation.json").hardlink_to(
            store.root / "annotation.json"
        )
    dataset = tmp_path / "dataset"

    def publish():
        return update_clip(dataset, record, positions, court, "fixture-code")

    if mode == "conflict":
        with pytest.raises(ValueError, match="camera_alignment conflicts"):
            publish()
        assert ScenePointers.read(store.root) == before
        assert not (store.root / "components/external_ball_positions").exists()
        return

    if mode == "saved":
        replace_text = external_ball_publication._replace_text

        def interrupt_annotation(path, content):
            if path.name == "annotation.json":
                raise OSError("simulated stop after index rename")
            replace_text(path, content)

        with monkeypatch.context() as patch:
            patch.setattr(
                external_ball_publication, "_replace_text", interrupt_annotation
            )
            with pytest.raises(OSError, match="simulated stop"):
                publish()
        interrupted = ScenePointers.read(store.root)
        assert (
            interrupted.scene != before.scene
            and interrupted.annotation == before.annotation
        )

    report = publish()
    assert report["reused_publication"] == (mode == "saved")
    after = ScenePointers.read(store.root)
    marker = after.marker()
    assert marker is not None
    path = annotation_scene_path(store.root, marker)
    scene = load_scene_result(path)
    assert path.parent.name == store.active("scene_assembly").artifact_id
    assert report["valid_3d"] == record.num_frames - 1
    np.testing.assert_allclose(scene.ball_3d[0], truth, atol=2e-5)
    assert scene.ball_rejection_code[-1] == 1
    assert not scene.ball_3d_valid[-1]
    assert scene.metadata["external_ball"]["status_codes"][1][0] == 3

    for node, ref in old_refs.items():
        if node != "scene_assembly":
            assert store.active(node) == ref
    geometry = scene.metadata["external_ball"]["geometry_provenance"]
    if player_scene is None:
        assert geometry["origin"] == SHARED_GEOMETRY
        assert geometry["player_validation"] == "not_run"
        assert not scene.player_valid.any() and scene.player_position.shape[0] == 0
        assert (
            scene.metadata["player_reconstruction"]["reason_code"]
            == "court_side_failed"
        )
    else:
        assert geometry["origin"] == SAVED_GEOMETRY
        np.testing.assert_array_equal(scene.human_kp_2d, player_scene.human_kp_2d)
        np.testing.assert_array_equal(scene.player_track_ids, [42])
        assert (store.root / old_refs["scene_assembly"].path).is_file()
        assert (
            tmp_path / "hardlinked-annotation.json"
        ).read_text() == before.annotation
    assert (tmp_path / "hardlinked-scene.json").read_text() == before.scene
    assert restore_publication(store.root)
    assert ScenePointers.read(store.root) == before
    assert path.is_file()
    assert not restore_publication(store.root)


def test_second_scene_revision_recovers_and_restores_the_external_ball_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    court, record, positions, store, _, _ = _saved_clip(tmp_path, monkeypatch, "saved")
    dataset = tmp_path / "dataset"
    original = ScenePointers.read(store.root)
    update_clip(dataset, record, positions, court, "fixture-code")
    baseline = ScenePointers.read(store.root)
    baseline_path = baseline.verify(store.root)
    assert baseline_path is not None
    ball_scene = load_scene_result(baseline_path)
    assert ball_scene.ball_3d_valid is not None and ball_scene.ball_3d_valid.any()
    scene = replace(
        ball_scene, metadata={**ball_scene.metadata, "doubles_selection": True}
    )
    old_refs = store.references()
    dependencies = {
        port: ArtifactRef(**ref)
        for port, ref in store.descriptor(old_refs["scene_assembly"])[
            "dependencies"
        ].items()
    }
    revision = PublicationKind("doubles_players", "test_reconstruction")
    identity = {
        "test": "doubles reconstruction",
        "previous": old_refs["scene_assembly"].artifact_id,
    }
    reference = store.publish(
        "scene_assembly",
        scene,
        ArtifactCodec(SceneResult),
        schema="scene_result",
        version=2,
        identity=identity,
        dependencies=dependencies,
        provenance={"origin": "component"},
        adopt=False,
    )
    inputs = resolve_dataset_clip(dataset, record)

    def publish(current: ScenePointers) -> None:
        publish_scene_revision(
            store.root,
            current,
            baseline,
            {"scene_assembly": reference},
            scene,
            inputs,
            inputs.snapshot,
            identity,
            {},
            kind=revision,
        )

    replace_text = external_ball_publication._replace_text

    def interrupt_annotation(path: Path, content: str | None) -> None:
        if path.name == "annotation.json":
            raise OSError("interrupted second revision")
        replace_text(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(external_ball_publication, "_replace_text", interrupt_annotation)
        with pytest.raises(OSError, match="interrupted second revision"):
            publish(baseline)
    interrupted, receipt = current_publication(store.root, revision)
    assert receipt is not None and receipt.before == baseline
    assert (
        interrupted.scene == receipt.after.scene
        and interrupted.annotation == baseline.annotation
    )
    publish(interrupted)

    current, receipt = current_publication(store.root, revision)
    assert receipt is not None and current == receipt.after
    revised_path = current.verify(store.root)
    assert (
        revised_path is not None and revised_path.parent.name == reference.artifact_id
    )
    revised = load_scene_result(revised_path)
    assert revised.metadata["doubles_selection"] is True
    np.testing.assert_array_equal(revised.ball_3d, ball_scene.ball_3d)
    assert store.references()["camera_alignment"] == old_refs["camera_alignment"]
    assert store.references()["ball_triangulation"] == old_refs["ball_triangulation"]
    with pytest.raises(ValueError, match="pointers changed independently"):
        restore_publication(store.root)
    assert restore_publication(store.root, revision)
    assert ScenePointers.read(store.root) == baseline
    assert restore_publication(store.root)
    assert ScenePointers.read(store.root) == original
    assert revised_path.is_file()


def test_external_ball_cli_validates_paths_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    court, record, _, store, _, _ = _saved_clip(tmp_path, monkeypatch, "fallback")
    DatasetManifest("fixture", {record.clip_id: record}).save(tmp_path / "dataset")
    before = ScenePointers.read(store.root)
    monkeypatch.chdir(tmp_path)
    selection = [
        "--dataset",
        "dataset",
        "--video",
        record.video_id,
        "--clip",
        record.clip_name,
    ]
    publish = [
        "publish",
        *selection,
        "--positions",
        ".",
        "--side-store",
        "side",
        "--side-digest",
        court.side.digest,
        "--court-store",
        str(court.court.root),
    ]
    with pytest.raises(PathContractError, match="report output must be a file"):
        main([*publish, "--report", "dataset"])
    assert ScenePointers.read(store.root) == before
    assert not (store.root / "components/external_ball_positions").exists()

    assert main([*publish, "--report", "report.json"]) == 0
    report = load_json(tmp_path / "report.json")
    assert (
        report["clips"][0]["scene_assembly"]["artifact_id"]
        == ScenePointers.read(store.root).revision()
    )
    assert report["clips"][0]["valid_3d"] == record.num_frames - 1
    assert main(["restore", *selection]) == 0
    assert ScenePointers.read(store.root) == before

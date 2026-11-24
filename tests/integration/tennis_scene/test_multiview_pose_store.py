from dataclasses import replace
from pathlib import Path
from sys import argv

import numpy as np
import pytest

from src.tennis_scene.multiview_pose.__main__ import main
from src.tennis_scene.multiview_pose.inputs import PinnedCourtSide, read_saved_inputs
from src.tennis_scene.multiview_pose.reconstruct import DEFAULT_CONFIG, reconstruct
from src.tennis_scene.multiview_pose.storage import publish_candidates, read_candidates
from src.tennis_scene.multiview_tracking.__main__ import main as tracking_main
from src.tennis_scene.multiview_tracking.schema import (
    SampleEvent,
    TrackEnd,
    TrackingConfig,
)
from src.tennis_scene.multiview_tracking.storage import read_tracks, track_store
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
)
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, unpack_value
from src.tennis_scene.pipeline.video_artifact import publish_document
from src.utils.configuration import PathContractError
from tests.integration.tennis_scene.test_video_court_side import build, prefix_stores
from tests.unit.tennis_scene.multiview_pose.test_reconstruct import person


@pytest.fixture
def saved_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pipeline, _, sources, court, ball_inputs = prefix_stores(
        tmp_path, monkeypatch, "agreement"
    )
    decision = build(pipeline, court, ball_inputs, tmp_path)
    source = sources[0]
    ball_store = ClipStore(ball_inputs[source.clip_id].directory, json_value(source))
    calibration_ref = ball_store.references()["court_calibration"]
    calibration = ball_store.load(
        calibration_ref, ArtifactCodec(CourtCalibrationOutput)
    )
    xyz = person() + [0.0, -3.0, 0.0]
    for view, turn in zip(
        calibration.calibration.views, decision.output().view_half_turns, strict=True
    ):
        camera = view.camera.half_turned(turn)
        uv, valid = camera.project(xyz)
        assert valid.all()
        stage = pipeline.components[f"person_pose_features/{camera.camera_id}"]
        keypoints = np.tile(
            np.c_[uv, np.ones(17)].astype(np.float32), (source.num_frames, 1, 1)
        )
        stage.result = replace(stage.result, keypoints=keypoints)

    store = ClipStore(tmp_path / "observations", json_value(source))
    nodes = standard_definition(
        pipeline.config,
        source,
        code_identity=pipeline.code_identity,
        video_court=court,
        overrides=pipeline.components,
    )
    # The inference stub supplies DINO's rows. Give the saved descriptor the
    # production detector setting while the real runner binds the pose port.
    detector_settings = {
        "detector": "dino",
        "scope": "full_frame",
        "merge_duplicates": False,
    }
    nodes = [
        replace(node, settings={**node.settings, **detector_settings})
        if node.name.startswith("person_detection/")
        else node
        for node in nodes
    ]
    ComponentRunner(nodes, store).run(
        [
            "court_calibration",
            *(f"person_pose_features/{camera}" for camera in source.camera_ids),
        ]
    )
    assert not any(node.startswith("ball_") for node in store.references())
    pinned = PinnedCourtSide.load(decision.directory, decision.digest, court.root)
    return store, source, calibration, decision, court, pinned


def test_cli_writes_candidates_from_only_raw_ports_and_pinned_side(
    saved_inputs, tmp_path, monkeypatch
):
    store, source, _, decision, court, _ = saved_inputs
    store.publish(
        "camera_alignment",
        {},
        ArtifactCodec(dict),
        schema="aligned_cameras",
        version=1,
        identity={"fixture": "unused legacy artifact"},
        dependencies={},
        provenance={},
    )
    before = store.index_path.read_bytes()
    output = tmp_path / "pose-result"
    monkeypatch.setattr(
        "sys.argv",
        [
            argv[0],
            "--clip-store",
            str(store.root),
            "--output",
            str(output),
            "--side-store",
            str(decision.directory),
            "--side-digest",
            decision.digest,
            "--court-store",
            str(court.root),
        ],
    )
    main()
    assert store.index_path.read_bytes() == before
    result = read_candidates(output)
    np.testing.assert_array_equal(
        np.diff(result.frame_offsets), np.ones(source.num_frames, np.int64)
    )
    np.testing.assert_array_equal(result.source_rows[15], [15, 15, 15])
    np.testing.assert_allclose(
        result.positions_m[15], person() + [0.0, -3.0, 0.0], atol=1e-5
    )
    result_store = ClipStore(output, json_value(source))
    ref = result_store.references()["multiview_pose"]
    assert (ref.schema, ref.version) == ("frame_pose_candidates", 1)
    evidence = result_store.descriptor(ref)["provenance"]
    assert evidence["court_side"]["digest"] == decision.digest
    assert "camera_alignment" not in evidence["observations"]


def test_tracking_cli_pins_candidates_and_preserves_their_source_timeline(
    saved_inputs, tmp_path, monkeypatch
):
    store, source, _, _, _, side = saved_inputs
    inputs = read_saved_inputs(store.root, side)
    candidates = reconstruct(inputs.observations)
    upstream, output = tmp_path / "pose", tmp_path / "tracking"
    pose_ref = publish_candidates(
        upstream, store.root, inputs, candidates, DEFAULT_CONFIG
    )
    input_index = (upstream / "scene.json").read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            argv[0],
            "--candidates",
            str(upstream),
            "--candidate-artifact",
            pose_ref.artifact_id,
            "--output",
            str(output),
            "--players-per-side",
            "1",
        ],
    )
    tracking_main()

    result = read_tracks(output)
    tracking = ClipStore(output, json_value(source))
    ref = tracking.references()["multiview_tracking"]
    assert (ref.schema, ref.version) == ("multiview_pose_tracks", 1)
    assert tracking.descriptor(ref)["provenance"][
        "frame_pose_candidates"
    ] == json_value(pose_ref)
    assert result.camera_ids == source.camera_ids
    np.testing.assert_array_equal(result.track_ids, np.zeros(source.num_frames))
    np.testing.assert_array_equal(
        result.frame_offsets, np.arange(source.num_frames + 1)
    )
    np.testing.assert_array_equal(result.source_rows[15], [15, 15, 15])
    np.testing.assert_allclose(
        result.positions_m[15], person() + [0.0, -3.0, 0.0], atol=1e-5
    )
    assert result.event[0] == SampleEvent.BORN
    assert result.event[-1] == SampleEvent.OBSERVED
    assert result.end_reasons.tolist() == [TrackEnd.CLIP_BOUNDARY]
    assert result.selection.selected_track_ids == (0,)
    assert result.selection.missing_per_side == (0, 1)

    assert (
        track_store(
            upstream,
            output,
            TrackingConfig(1),
            expected_artifact_id=pose_ref.artifact_id,
        )
        == ref
    )
    before = tracking.index_path.read_bytes()
    with pytest.raises(ValueError, match="pinned artifact"):
        track_store(upstream, output, TrackingConfig(1), expected_artifact_id="0" * 64)
    assert tracking.index_path.read_bytes() == before
    assert (upstream / "scene.json").read_bytes() == input_index


@pytest.mark.parametrize(
    ("invalid_field", "error"),
    [
        ("--drift-store", r"drift_store must be an existing directory"),
        ("--output", r"Pose output must be a directory"),
    ],
)
def test_cli_rejects_non_directory_before_loading_side(
    invalid_field: str, error: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pose-result"
    file = tmp_path / "not-a-store"
    file.write_text("keep this file")
    paths = {
        "--clip-store": tmp_path,
        "--side-store": tmp_path,
        "--court-store": tmp_path,
        "--output": output,
        invalid_field: file,
    }
    monkeypatch.setattr(
        "sys.argv",
        [
            argv[0],
            "--side-digest",
            "0" * 64,
            *(arg for flag, path in paths.items() for arg in (flag, str(path))),
        ],
    )

    with pytest.raises(PathContractError, match=error):
        main()

    assert not output.exists()
    assert file.read_text() == "keep this file"


def test_output_publication_rejects_an_existing_store_with_other_components(
    saved_inputs, tmp_path
):
    store, source, _, _, _, side = saved_inputs
    occupied = ClipStore(tmp_path / "occupied", json_value(source))
    occupied.publish(
        "camera_alignment",
        {},
        ArtifactCodec(dict),
        schema="aligned_cameras",
        version=1,
        identity={"fixture": "other recipe"},
        dependencies={},
        provenance={},
    )
    before = occupied.index_path.read_bytes()
    inputs = read_saved_inputs(store.root, side)
    with pytest.raises(ValueError, match="another recipe"):
        publish_candidates(
            occupied.root,
            store.root,
            inputs,
            reconstruct(inputs.observations),
            DEFAULT_CONFIG,
        )
    assert occupied.index_path.read_bytes() == before


@pytest.mark.parametrize(
    ("node", "schema", "port"),
    [
        ("person_tracking/cam0", "person_tracks", "person_pose_features/cam0"),
        ("player_selection/cam0", "selected_player_tracks", "court_calibration"),
        ("player_association", "person_identities", "person_detection/cam0"),
        ("camera_alignment", "aligned_cameras", "court_detection/cam0"),
    ],
)
def test_legacy_dependencies_are_rejected_at_each_input_level(
    saved_inputs, node, schema, port
):
    store, _, _, _, _, side = saved_inputs
    forbidden = store.publish(
        node,
        {},
        ArtifactCodec(dict),
        schema=schema,
        version=1,
        identity={"fixture": node},
        dependencies={},
        provenance={},
    )
    original = store.references()[port]
    descriptor = store.descriptor(original)
    value = unpack_value(
        descriptor["payload"], (store.root / original.path).parent, descriptor["arrays"]
    )
    dependencies = {
        key: ArtifactRef(**dependency)
        for key, dependency in descriptor["dependencies"].items()
    }
    store.publish(
        port,
        value,
        ArtifactCodec(dict),
        schema=original.schema,
        version=original.version,
        identity=descriptor["identity"],
        dependencies={**dependencies, "forbidden": forbidden},
        provenance={},
    )
    with pytest.raises(ValueError, match="Unexpected .*dependencies"):
        read_saved_inputs(store.root, side)


def test_original_detection_rows_must_match_saved_pose(saved_inputs):
    store, _, _, _, _, side = saved_inputs
    node = "person_pose_features/cam0"
    ref = store.references()[node]
    codec = ArtifactCodec(PersonPoseFeaturesOutput)
    original = store.load(ref, codec)
    descriptor = store.descriptor(ref)
    store.publish(
        node,
        replace(original, source_rows=original.source_rows + 1),
        codec,
        schema=ref.schema,
        version=ref.version,
        identity=descriptor["identity"],
        dependencies={"detections": store.references()["person_detection/cam0"]},
        provenance={},
    )
    with pytest.raises(ValueError, match="source row mismatch"):
        read_saved_inputs(store.root, side)


def test_side_rejects_missing_video_wrong_digest_player_evidence_and_changed_camera_or_media(
    saved_inputs,
):
    _, source, calibration, decision, court, side = saved_inputs
    with pytest.raises(FileNotFoundError):
        PinnedCourtSide.load(decision.directory, "a" * 64, court.root)
    player_digest = publish_document(
        decision.directory, {**decision.decision.document(), "evidence": "player"}
    )
    with pytest.raises(ValueError, match="ball-based"):
        PinnedCourtSide.load(decision.directory, player_digest, court.root)
    with pytest.raises(ValueError, match="differs from the pinned video"):
        side.bind(replace(source, clip_id="video_999/clip_000"), calibration)
    first, *remaining = calibration.calibration.views
    changed = replace(
        first,
        camera=replace(
            first.camera, translation=first.camera.translation + [0.01, 0, 0]
        ),
    )
    with pytest.raises(ValueError, match="Camera matrices differ"):
        side.bind(
            source,
            replace(
                calibration,
                calibration=replace(
                    calibration.calibration, views=(changed, *remaining)
                ),
            ),
        )
    source.videos[0].path.write_bytes(b"Changed media")
    with pytest.raises(ValueError, match="Source video SHA-256 mismatch"):
        side.bind(source, calibration)

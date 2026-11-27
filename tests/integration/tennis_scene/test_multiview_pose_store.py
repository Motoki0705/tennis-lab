import json
from dataclasses import replace
from pathlib import Path
from sys import argv
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.__main__ import main
from src.tennis_scene.multiview_pose.inputs import (
    PinnedCourtSide,
    SavedPoseInputs,
    read_saved_inputs,
)
from src.tennis_scene.multiview_pose.reconstruct import DEFAULT_CONFIG, reconstruct
from src.tennis_scene.multiview_pose.storage import publish_candidates, read_candidates
from src.tennis_scene.multiview_tracking.__main__ import main as tracking_main
from src.tennis_scene.multiview_tracking.schema import (
    SampleEvent,
    TrackEnd,
    TrackingConfig,
)
from src.tennis_scene.multiview_tracking.storage import (
    read_candidate_store,
    read_tracks,
    track_store,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_detection import (
    PersonDetectionModule,
    PersonDetectionOutput,
)
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesModule,
    PersonPoseFeaturesOutput,
)
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.people_artifacts import require_current_people_artifacts
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, unpack_value
from src.tennis_scene.pipeline.video_artifact import publish_document
from src.utils.configuration import PathContractError
from tests.integration.tennis_scene.test_video_court_side import build, prefix_stores
from tests.unit.tennis_scene.multiview_pose.test_reconstruct import (
    cameras,
    observed_inputs,
    person,
)


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
    # The inference stubs supply rows and poses while the real runner binds
    # the corresponding ports with the production observation settings.
    observation_settings: dict[str, dict[str, bool | str]] = {
        "person_detection": {
            "detector": "dino",
            "scope": "full_frame",
            "merge_duplicates": False,
            "enabled": True,
        },
        "person_pose_features": {"enabled": True},
    }
    nodes = tuple(
        replace(
            node,
            settings=observation_settings.get(node.name.split("/", 1)[0], node.settings),
        )
        for node in nodes
    )
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
    provenance = tracking.descriptor(ref)["provenance"]
    assert provenance["frame_pose_candidates"] == json_value(pose_ref)
    assert provenance["fragment_integration"] == {
        "first_stage_track_count": 1,
        "slots": [
            {
                "side": -1,
                "core_fragment": 0,
                "track_id": 0,
                "fragments": [0],
                "joins": [],
            }
        ],
    }
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


def test_republishing_legacy_candidates_preserves_the_pinned_identity(tmp_path: Path) -> None:
    # Outputs of publish_candidates at the pre-integration main commit 329dc6880f94.
    artifact_id = "f38f15795f89961aad6413e0c5cb787f3ec684e6b8a9cec7e435ba07c149e10a"
    execution_key = "1a23770335514e82b3aa10c02cc64ac29d4c3aa89953c84c901e3f3772d12dc3"
    observations = observed_inputs(cameras(), [[np.empty((0, 17, 3))]] * 3)
    refs = {"person_detection/cam0": ArtifactRef(
        "1" * 64, "components/person_detection/cam0/fixture.json", "2" * 64, "person_detections", 2, "3" * 64
    )}
    inputs = SavedPoseInputs(observations, refs, {"digest": "a" * 64, "evidence": "ball"}, ("dino", "vitpose-h"))
    destination, origin = tmp_path / "candidates", tmp_path / "observations"
    candidates = reconstruct(observations)
    for _ in range(2):
        ref = publish_candidates(destination, origin, inputs, candidates, DEFAULT_CONFIG)
        assert (ref.artifact_id, ref.execution_key) == (artifact_id, execution_key)
    source, recorded, result = read_candidate_store(destination, expected_artifact_id=artifact_id)
    assert (source, recorded) == (observations.source, ref)
    np.testing.assert_array_equal(result.frame_offsets, [0, 0])


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


@pytest.mark.parametrize(
    ("dino_enabled", "vitpose_enabled", "disabled"),
    [(False, True, "DINO"), (True, False, "ViTPose"), (True, True, None)],
    ids=["dino-disabled", "vitpose-disabled", "enabled-empty"],
)
def test_empty_observations_require_enabled_person_inputs(
    saved_inputs, dino_enabled, vitpose_enabled, disabled
):
    store, source, _, _, _, side = saved_inputs
    offsets = np.zeros(source.num_frames + 1, np.int64)
    rows: NDArray[np.int64] = np.empty(0, np.int64)
    det_codec = ArtifactCodec(PersonDetectionOutput)
    pose_codec = ArtifactCodec(PersonPoseFeaturesOutput)
    refs = store.references()
    for camera in source.camera_ids:
        d_node, p_node = f"person_detection/{camera}", f"person_pose_features/{camera}"
        d_ref, p_ref = refs[d_node], refs[p_node]
        d_identity = store.descriptor(d_ref)["identity"]
        p_identity = store.descriptor(p_ref)["identity"]
        pose = store.load(p_ref, pose_codec)
        detection = PersonDetectionOutput(
            camera, offsets, np.empty((0, 4), np.float32), np.empty(0, np.float32), rows
        )
        d_ref = store.publish(
            d_node,
            detection,
            det_codec,
            schema=d_ref.schema,
            version=d_ref.version,
            identity={
                **d_identity,
                "settings": {**d_identity["settings"], "enabled": dino_enabled},
            },
            dependencies={},
            provenance={},
        )
        store.publish(
            p_node,
            replace(
                pose,
                frame_offsets=offsets,
                source_rows=rows,
                keypoints=pose.keypoints[:0],
            ),
            pose_codec,
            schema=p_ref.schema,
            version=p_ref.version,
            identity={
                **p_identity,
                "settings": {**p_identity["settings"], "enabled": vitpose_enabled},
            },
            dependencies={"detections": d_ref},
            provenance={},
        )

    if disabled is not None:
        with pytest.raises(ValueError, match=rf"{disabled}.*enabled=true.*cam0"):
            read_saved_inputs(store.root, side)
        return

    result = reconstruct(read_saved_inputs(store.root, side).observations)
    np.testing.assert_array_equal(result.frame_offsets, offsets)
    assert result.positions_m.shape == (0, 17, 3)


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


def _people_observation_settings(detector: str, pose: str) -> tuple[dict[str, Any], dict[str, Any]]:
    detection = {
        "detector": detector, "scope": "full_frame", "enabled": True, "merge_duplicates": False,
        "runtime": {"confidence": .30},
        "models": {"detector": {
            "architecture": "dino_4scale_swin_l" if detector == "dino" else detector,
            "assets": {"checkpoint": {"sha256": "d" * 64}},
            "settings": {"backend": "tensorrt", "engines": [{"plan_sha256": "e" * 64}], "runtime": {"confidence": .30}},
        }},
    }
    feature_model: dict[str, Any] = {"architecture": "vitpose_huge_coco17" if pose == "vitpose-h" else pose}
    features = {"enabled": True, "models": {"pose": feature_model}}
    if pose == "rtmpose-x":
        features.update(pose=pose, joint_confidence="raw_simcc_peak")
        feature_model["settings"] = {"joint_confidence": "raw_simcc_peak"}
    return detection, features


def _publish_people_observations(store: ClipStore, camera: str, detection: dict[str, Any], pose: dict[str, Any]) -> None:
    refs = store.references()
    dependencies: dict[str, ArtifactRef] = {}
    for component, output, settings in (
        (PersonDetectionModule, PersonDetectionOutput, detection),
        (PersonPoseFeaturesModule, PersonPoseFeaturesOutput, pose),
    ):
        node = f"{component.io.name}/{camera}"
        reference = refs[node]
        codec = ArtifactCodec(output)
        identity = store.descriptor(reference)["identity"]
        published = store.publish(
            node, store.load(reference, codec), codec, schema=reference.schema, version=reference.version,
            identity={**identity, "component_class": f"{component.__module__}.{component.__qualname__}", "settings": settings},
            dependencies=dependencies, provenance={"origin": "component"},
        )
        dependencies = {"detections": published}


@pytest.mark.parametrize(("detector", "pose"), [
    ("rfdetr-l", "rtmpose-x"), ("rfdetr-l", "vitpose-h"), ("dino", "rtmpose-x"),
])
def test_alternative_people_artifacts_preserve_the_3d_input_and_publication_contract(saved_inputs, detector, pose):
    store, source, _, _, _, side = saved_inputs
    code = store.descriptor(store.references()["person_detection/cam0"])["identity"]["implementation"]
    detection, features = _people_observation_settings(detector, pose)
    for camera in source.camera_ids:
        _publish_people_observations(store, camera, detection, features)
    adopted = read_saved_inputs(store.root, side)
    assert len(adopted.observations.poses) == len(source.camera_ids)
    result = reconstruct(adopted.observations)
    assert result.joint_valid.any()
    document = json.loads(store.index_path.read_text())
    publication = {"code_sha256": code, "people_models": {"detector": detector, "pose": pose},
                   "settings": {"people_models": {"merge_duplicate_person_boxes": False}},
                   "models": {"person_detection": detection["models"], "person_pose_features": features["models"]},
                   "people_component_settings": {
                       stage: {key: value for key, value in settings.items() if key != "models"}
                       for stage, settings in (("person_detection", detection), ("person_pose_features", features))
                   }}
    required = require_current_people_artifacts(store.root, document, source.camera_ids, publication)
    assert len(required) == 2 * len(source.camera_ids)
    different = {**publication, "people_models": {"detector": "dino" if detector == "rfdetr-l" else "rfdetr-l", "pose": pose}}
    with pytest.raises(ValueError, match="differ from the configured selection"):
        require_current_people_artifacts(store.root, document, source.camera_ids, different)

    _publish_people_observations(store, "cam0", {**detection, "merge_duplicates": True}, features)
    document = json.loads(store.index_path.read_text())
    with pytest.raises(ValueError, match="differs from its declared people settings"):
        require_current_people_artifacts(store.root, document, source.camera_ids, publication)


@pytest.mark.parametrize("conflict", [
    "detector-selector", "pose-selector", "missing-detector-selector", "missing-pose-selector",
    "legacy-raw-score", "model-score",
])
def test_saved_inputs_reject_contradictory_people_declarations(saved_inputs, conflict):
    store, source, _, _, _, side = saved_inputs
    detection, pose = _people_observation_settings("rfdetr-l", "rtmpose-x")
    if conflict == "detector-selector":
        detection["detector"] = "dino"
    elif conflict == "pose-selector":
        pose["pose"] = "vitpose-h"
    elif conflict == "missing-detector-selector":
        del detection["detector"]
    elif conflict == "missing-pose-selector":
        del pose["pose"]
    elif conflict == "legacy-raw-score":
        pose = {**_people_observation_settings("dino", "vitpose-h")[1], "joint_confidence": "raw_simcc_peak"}
    else:
        pose["models"]["pose"]["settings"]["joint_confidence"] = "heatmap_peak"
    for camera in source.camera_ids:
        _publish_people_observations(store, camera, detection, pose)
    with pytest.raises(ValueError, match=r"model declaration|raw SimCC"):
        read_saved_inputs(store.root, side)


@pytest.mark.parametrize("changed", ["threshold", "backend", "checkpoint", "engine"])
def test_saved_inputs_reject_mixed_people_settings_across_cameras(saved_inputs, changed):
    store, source, _, _, _, side = saved_inputs
    for camera in source.camera_ids:
        detection, pose = _people_observation_settings("rfdetr-l", "rtmpose-x")
        if camera == "cam1":
            model = detection["models"]["detector"]
            if changed == "threshold":
                detection["runtime"]["confidence"] = .25
                model["settings"]["runtime"]["confidence"] = .25
            elif changed == "backend":
                model["settings"]["backend"] = "pytorch"
                del model["settings"]["engines"]
            elif changed == "checkpoint":
                model["assets"]["checkpoint"]["sha256"] = "c" * 64
            else:
                model["settings"]["engines"][0]["plan_sha256"] = "f" * 64
        _publish_people_observations(store, camera, detection, pose)
    with pytest.raises(ValueError, match="People observation settings differ between cameras: cam1"):
        read_saved_inputs(store.root, side)


def test_saved_inputs_accept_legacy_model_declarations_without_selectors(saved_inputs):
    store, source, _, _, _, side = saved_inputs
    detection, features = _people_observation_settings("dino", "vitpose-h")
    del detection["detector"]
    for camera in source.camera_ids:
        _publish_people_observations(store, camera, detection, features)
    assert len(read_saved_inputs(store.root, side).observations.poses) == len(source.camera_ids)


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

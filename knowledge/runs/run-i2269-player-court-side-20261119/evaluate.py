"""CPU qualification of player court side using the pinned #1752 observations."""

from __future__ import annotations

import argparse
import platform
import subprocess
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from hydra import compose, initialize_config_dir

from src.tasks.court_side.player import _informative_frames, _matched_distances
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationInput,
    CourtCalibrationModule,
)
from src.tennis_scene.pipeline.components.court_side import (
    PlayerCourtSideInput,
    PlayerCourtSideModule,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, unpack_value
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.tennis_scene.pipeline.video_court.component import (
    SharedCourtAssembler,
    SharedCourtModule,
)
from src.tennis_scene.pipeline.video_court.diagnostics import (
    DriftReference,
    load_video_drift,
    publish_video_drift,
)
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.tennis_scene.pipeline.video_court_side import PinnedSideInputs
from src.tennis_scene.pipeline.video_court_side.component import SharedCourtSideModule
from src.tennis_scene.pipeline.video_court_side.prepare import (
    calibration_digest,
    measure_clip_side,
)
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT

CLIPS = ("video_000/clip_000", "video_000/clip_027", "video_000/clip_037")
CAMERAS = ("cam0", "cam1", "cam2", "cam3")
EXPECTED = (False, False, True, True)
STORES = Path("video-batch-cli/tennis_scene/evaluate/video-batch-shared-fp16-20261118/datasets/batch/videos")
COURT = Path("court-video-shared/tennis_scene/evaluate/court-video-shared-acceptance-20261117/store")
COURT_DIGEST = "4ac8e386763687fcb6c81f46abf0774e38fc118af4f79b31cdb6e3c2283e3f75"
DRIFT = Path("court-video-shared/tennis_scene/evaluate/court-video-drift-lines-20261117/drift-store")
DRIFT_DIGEST = "7a4be8bf20397552a7b287131c3cb0e5fa57ba0dedc16e4a05fa23287c81a7f6"


def runtime(output, overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        cfg = compose(config_name="pipeline", overrides=[
            "device=cpu", "court_side.evidence=player", "ball_detection.enabled=false", "ball_reconstruction.enabled=false",
            "player_association.players_per_side=2", f"paths.artifact_root={output}", *overrides,
        ])
    return PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)


def scoped_court(full, output):
    documents = {}
    for camera, original in full.cameras.items():
        sources = {clip: original["sources"][clip] for clip in CLIPS}
        documents[camera] = {
            **original, "sources": sources, "clips": {clip: original["clips"][clip] for clip in CLIPS},
            "identity": {**original["identity"], "sources": sources,
                         "qualification_subset_of": full.manifest["cameras"][camera]},
        }
    store = VideoCourtStore(output / "court")
    with store.lock():
        court = store.publish(documents)
    first_protocol = full.drift_cameras[CAMERAS[0]]["protocol"]
    assert all(row["protocol"] == first_protocol for row in full.drift_cameras.values())
    reports = {camera: {clip: row["clips"][clip] for clip in CLIPS} for camera, row in full.drift_cameras.items()}
    return publish_video_drift(court, output / "drift",
                               protocol={**first_protocol, "qualification_subset_of": full.drift.digest}, reports=reports)


def replay_calibration_and_copy_observations(pin, inputs, full, scoped, cfg, output):
    source = pin.source
    context = AssemblyContext(source)
    store = ClipStore(output / "clips" / source.clip_id, json_value(source))
    provenance = {"origin": "qualification_replay", "source_inputs": pin.digest}
    shared = SharedCourtModule(cfg.video_court_unobservable_policy)
    assemble = SharedCourtAssembler(scoped)
    calibration = CourtCalibrationModule(source.camera_ids, cfg.camera_geometry,
                                         roi_margins=cfg.person_roi_margins, court_schema_version=3)
    observations = tuple(shared.process(assemble.assemble(replace(context, camera_id=camera), {})) for camera in source.camera_ids)
    scoped_calibration = calibration.process(CourtCalibrationInput(source, observations))
    geometry = calibration_digest(inputs, full)
    assert calibration_digest(replace(inputs, calibration=scoped_calibration), scoped) == geometry
    np.testing.assert_array_equal(scoped_calibration.court.keypoints, inputs.calibration.court.keypoints)
    np.testing.assert_array_equal(scoped_calibration.court.visibility, inputs.calibration.court.visibility)
    assert scoped_calibration.footpoint_polygons == inputs.calibration.footpoint_polygons
    parents = {
        camera: store.publish(f"court_detection/{camera}", observation, ArtifactCodec(shared.io.output_type),
                              schema=shared.io.output_schema, version=shared.io.version, dependencies={},
                              identity={"court_camera": scoped.manifest["cameras"][camera], "drift": scoped.drift.digest},
                              provenance=provenance)
        for camera, observation in zip(source.camera_ids, observations, strict=True)
    }
    calibration_ref = store.publish("court_calibration", scoped_calibration, ArtifactCodec(calibration.io.output_type),
                                    schema=calibration.io.output_schema, version=calibration.io.version, dependencies=parents,
                                    identity={"calibration": geometry, "court": scoped.digest}, provenance=provenance)
    for node, ref in pin.references.items():
        if node == "court_calibration":
            continue
        descriptor = read_component_descriptor(pin.directory, ref, node=node, source_sha256=pin.source_digest)
        raw = unpack_value(descriptor["payload"], (pin.directory / ref["path"]).parent, descriptor["arrays"])
        store.publish(node, raw, ArtifactCodec(dict), schema=ref["schema"], version=ref["version"],
                      identity={"source_selection": ref, "calibration": document_digest(calibration_ref)},
                      dependencies={"calibration": calibration_ref}, provenance=provenance)
    return PinnedSideInputs.from_store(store.root, evidence="player")


def ground_error(inputs, cfg):
    _, sampled = inputs.sample(cfg.sampling_max_frames, cfg.court_side)
    pairs = {}
    frame_maxima = []
    for _, points in _informative_frames(sampled, cfg.court_side):
        pair_maxima = []
        for a, b in combinations(points, 2):
            distances = _matched_distances(points[a], -points[b] if EXPECTED[a] != EXPECTED[b] else points[b])
            pairs.setdefault(f"{CAMERAS[a]}-{CAMERAS[b]}", []).extend(distances.tolist())
            pair_maxima.append(float(distances.max()))
        frame_maxima.append(max(pair_maxima))
    return {
        "pair_p90_m": {pair: float(np.quantile(distances, .9)) for pair, distances in pairs.items()},
        "frame_max_p20_p50_p90_m": np.quantile(frame_maxima, [.2, .5, .9]).tolist(),
    }


def evaluate(common, output):
    started = perf_counter()
    config = runtime(output)
    base = VideoCourtStore(common / COURT).load(COURT_DIGEST)
    full = load_video_drift(base, DriftReference(common / DRIFT, DRIFT_DIGEST))
    scoped = scoped_court(full, output)
    pins = {}
    clips = {}
    for clip in CLIPS:
        video_id, clip_id = clip.split("/")
        directory = common / STORES / video_id / "clips" / clip_id / "annotations/tennis_scene"
        pin = PinnedSideInputs.from_store(directory, evidence="player")
        pin.assert_current()
        inputs = pin.assemble()
        assert isinstance(inputs, PlayerCourtSideInput)
        assert inputs.calibration.calibration.camera_ids == CAMERAS and inputs.calibration.reference_camera == CAMERAS[0]
        stage = PlayerCourtSideModule(CAMERAS, config.court_side, max_frames=config.sampling_max_frames)
        try:
            standalone = {"view_half_turns": stage.process(inputs).view_half_turns, "reason": None}
        except ReconstructionUnavailable as error:
            standalone = {"view_half_turns": None, "reason": error.reason}
        diagnostics = {}
        for distance in (2.5, 3.0, 3.5):
            thresholds = replace(config.court_side, match_distance_m=distance)
            measurement, _ = measure_clip_side(inputs, thresholds, config.sampling_max_frames)
            diagnostics[str(distance)] = asdict(measurement)
        pins[clip] = replay_calibration_and_copy_observations(pin, inputs, full, scoped, config, output)
        pin.assert_current()
        clips[clip] = {**standalone, "source_num_frames": inputs.source.num_frames, "inputs_digest": pin.digest,
                       "inputs": pin.references, "distance_sweep": diagnostics, "geometry_errors": ground_error(inputs, config)}
        print(clip, standalone, flush=True)

    request = output / "side-inputs.json"
    write_json_atomic(request, {"schema": "video_court_side_inputs_v1", "evidence_clip_ids": None, "clips": {
        clip: {"role": "artifact", "directory": str(pin.directory.relative_to(output)), "inputs_digest": pin.digest}
        for clip, pin in pins.items()
    }})
    scopes = [
        "court_kp.mode=video_shared", "court_kp.video_shared.action=load", "court_kp.video_shared.dataset_id=meiji_4cam",
        "court_kp.video_shared.video_id=video_000", "court_kp.video_shared.directory=court",
        f"court_kp.video_shared.artifact_digest={scoped.digest}", "court_kp.video_shared.drift_directory=drift",
        f"court_kp.video_shared.drift_digest={scoped.drift.digest}",
        "court_side.mode=video_shared", "court_side.video_shared.directory=side",
        "court_side.video_shared.unobservable_policy=fail",
    ]
    build = TennisSceneOrchestrator(runtime(output, [
        *scopes, "court_side.video_shared.action=build", "court_side.video_shared.request=side-inputs.json",
        f"court_side.video_shared.request_sha256={dual_sha256(request)}",
    ]))
    artifact = build.prepare_video_court_side(build.prepare_video_court())
    assert artifact is not None
    load = TennisSceneOrchestrator(runtime(output, [
        *scopes, "court_side.video_shared.action=load", f"court_side.video_shared.artifact_digest={artifact.digest}",
    ]))
    loaded_court = load.prepare_video_court()
    loaded_side = load.prepare_video_court_side(loaded_court)
    assert loaded_court is not None and loaded_side is not None
    assert loaded_side.digest == artifact.digest and loaded_side.decision == artifact.decision
    assert loaded_side.decision.evidence == "player" and loaded_side.decision.evidence_clip_ids == CLIPS
    module = SharedCourtSideModule(CAMERAS, loaded_side, loaded_court, policy="fail")
    outcomes = {}
    for clip, pin in pins.items():
        measurement = loaded_side.clips[clip]
        assert asdict(measurement) == clips[clip]["distance_sweep"][str(config.court_side.match_distance_m)]
        try:
            result = module.process(pin.assemble())
            outcomes[clip] = {"status": "ok", "view_half_turns": result.view_half_turns, "reason": None}
        except ReconstructionUnavailable as error:
            outcomes[clip] = {"status": "failed", "view_half_turns": None, "reason": error.reason}
    report = {
        "dataset": "meiji_4cam", "camera_ids": CAMERAS, "reference_camera": CAMERAS[0], "expected": EXPECTED,
        "thresholds": json_value(config.court_side), "max_frames": config.sampling_max_frames,
        "source_court_digest": full.digest, "source_drift_digest": full.drift.digest,
        "scoped_court_digest": scoped.digest, "scoped_drift_digest": scoped.drift.digest,
        "scope_calibration_equal": True, "clips": clips, "video_decision": artifact.decision.document(),
        "video_digest": artifact.digest, "video_clip_outcomes": outcomes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "wall_seconds": perf_counter() - started,
    }
    write_json_atomic(output / "report.json", report)
    print("video", outcomes, "margin", loaded_side.decision.shared.margin, flush=True)
    if any(clip["view_half_turns"] != EXPECTED for clip in clips.values()) or any(
        result["view_half_turns"] != EXPECTED for result in outcomes.values()
    ):
        raise SystemExit("Player evidence did not pass qualification. See report.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-outputs", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluate(args.common_outputs.resolve(), output)

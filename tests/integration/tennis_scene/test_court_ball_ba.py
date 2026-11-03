"""Public config, production assembly/codec and fixed-input LOVO boundaries."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.court_ball_ba import CourtBallBAMethod
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementOutput
from src.tennis_scene.pipeline.components.scene_assembly import SceneAssemblyInput
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationModule,
    PlayerTriangulationOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.support.court_ball_calibration import (
    calibration_method_inputs,
    calibration_pipeline_artifacts,
    main_regression_inputs,
)


def runtime(directory, calibration):
    config_dir = Path(__file__).resolve().parents[3] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="pipeline", overrides=[
            "device=cpu", "gvhmr.enabled=false",
            f"paths.checkpoint_root={directory / 'ckpt'}",
            f"paths.external_asset_root={directory / 'external'}",
            f"ball_reconstruction.calibration={calibration}",
        ])
    result = PipelineRuntimeConfig.from_config(cfg)
    for path in enabled_model_assets(result).values():
        assert path.is_relative_to(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"definition-only synthetic asset")
    return result


def test_public_config_assembly_optional_stage_and_real_codec(tmp_path):
    inputs = calibration_method_inputs(frames=20)
    source, artifacts = calibration_pipeline_artifacts(inputs, tmp_path)
    cfg = runtime(tmp_path, "court_ball_ba")
    nodes = standard_definition(cfg, source, code_identity="contract")
    # Constructing the actual graph catches mismatched ball v2 consumer ports.
    ComponentRunner(nodes, ClipStore(tmp_path / "store", {}))
    node = next(n for n in nodes if n.name == "ball_triangulation")
    assert node.io.version == 2
    assembled = node.assembler.assemble(AssemblyContext(source), artifacts)
    actual = node.component.process(assembled)
    assert actual.calibration is not None
    assert actual.calibration.diagnostics["reference_camera"] == "a_near"
    assert actual.calibration.time_offsets_seconds[1] > 0
    codec = ArtifactCodec(node.io.output_type)
    payload, arrays = codec.dump(actual, tmp_path)
    restored = codec.load(payload, tmp_path, arrays)
    for a, b in zip(restored.calibration.cameras, actual.calibration.cameras, strict=True):
        np.testing.assert_array_equal(a.matrix, b.matrix)
    np.testing.assert_array_equal(restored.calibration.time_offsets_seconds, actual.calibration.time_offsets_seconds)
    np.testing.assert_array_equal(restored.ball.trajectory.valid, actual.ball.trajectory.valid)
    empty_people = GroupedObservations(
        np.empty(0, np.int64), np.empty((0, 3, 20, 17, 2), np.float32),
        np.empty((0, 3, 20, 17), np.float32), np.empty((0, 3, 20, 17), bool),
        np.empty((0, 3, 20), np.int64))
    scene_node = next(n for n in nodes if n.name == "scene_assembly")
    scene = scene_node.component.process(SceneAssemblyInput(
        source, artifacts["calibration"], artifacts["alignment"], empty_people,
        PlayerTriangulationOutput(None), actual, BodyPlacementOutput(None)))
    np.testing.assert_array_equal(scene.ball_3d, actual.ball.trajectory.positions)
    np.testing.assert_array_equal(scene.metadata["ball_calibration"]["time_offsets_seconds"], actual.calibration.time_offsets_seconds)


def test_fixed_default_matches_captured_main_arrays(tmp_path):
    inputs = main_regression_inputs()
    source, artifacts = calibration_pipeline_artifacts(inputs, tmp_path)
    cfg = runtime(tmp_path, "fixed")
    assert cfg.ball_bundle_adjustment is None
    node = next(n for n in standard_definition(cfg, source, code_identity="contract") if n.name == "ball_triangulation")
    assert isinstance(node.component, BallTriangulationModule)
    assert node.io.version == 1
    assembled = node.assembler.assemble(AssemblyContext(source), artifacts)
    actual = node.component.process(assembled).ball
    frozen = json.loads((Path(__file__).parents[2] / "support/fixtures/court_ball_main.json").read_text())
    for field, expected in frozen["ball"].items():
        np.testing.assert_array_equal(getattr(actual, field), np.array(expected, dtype=getattr(actual, field).dtype))
    for field, expected in frozen["trajectory"].items():
        np.testing.assert_array_equal(getattr(actual.trajectory, field), np.array(expected, dtype=getattr(actual.trajectory, field).dtype))


def test_lovo_recomputes_selected_cameras_only_and_saves_typed_refinement(tmp_path):
    inputs = calibration_method_inputs(frames=20)
    method = CourtBallBAMethod(min_frames=1)
    runs = run_method(method, inputs, bundle_sha256="fixture", code_commit="contract", code_fingerprint="contract")
    assert len(runs) == 4
    for run in runs:
        assert run.result.cameras is not None
        assert tuple(c.camera_id for c in run.result.cameras) == run.result.camera_ids
        assert run.held_out_camera not in run.result.camera_ids
        directory = tmp_path / run.key
        digest = save_run(directory, run)
        restored = load_run(directory, expected_sha256=digest)
        np.testing.assert_array_equal(restored.result.time_offsets_seconds, run.result.time_offsets_seconds)
    # Alter only the excluded view. Recompute that exact fold via the same adapter.
    views = list(inputs.views)
    views[2] = replace(views[2], court=replace(views[2].court, keypoints_px=views[2].court.keypoints_px + [10., 7.]))
    altered = replace(inputs, views=tuple(views))
    result = method(altered.select_cameras(inputs.camera_ids[:2]))
    np.testing.assert_array_equal(result.positions, runs[3].result.positions)
    np.testing.assert_array_equal(result.time_offsets_seconds, runs[3].result.time_offsets_seconds)


def test_unknown_calibration_choice_is_rejected_before_loading_assets(tmp_path):
    with pytest.raises(ValueError, match="fixed/court_ball_ba"):
        runtime(tmp_path, "guess")

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.candidate_gmm import CandidateGMM
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.pipeline.components.ball_gmm import (
    CandidateGMMModule,
    CandidateGMMOutput,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationModule
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.support.ball3d_evaluation import make_inputs, make_scoring_bundle, make_store
from tests.support.ball3d_gmm import make_gmm_inputs
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)


def node_for(tmp_path, inputs, method):
    cfg = runtime(tmp_path, overrides=() if method is None else (f"ball_reconstruction.method={method}",))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    source = make_scoring_bundle(inputs).source
    node = next(n for n in standard_definition(cfg, source, code_identity="candidate-gmm-test") if n.name == "ball_triangulation")
    return cfg, node, source


def test_default_pipeline_matches_frozen_main_arrays(tmp_path):
    inputs = make_inputs(frames=12)
    cfg, node, source = node_for(tmp_path, inputs, None)
    assert cfg.ball_method == "triangulation"
    assert isinstance(node.component, BallTriangulationModule)
    assert node.bindings["ball_right"] == "ball_points/right"
    store = make_store(tmp_path / "store", inputs)
    artifacts = {"alignment": store.load(store.active("camera_alignment"), ArtifactCodec(CameraAlignmentOutput)),
                 "calibration": store.load(store.active("court_calibration"), ArtifactCodec(CourtCalibrationOutput)),
                 **{f"ball_{v.camera.camera_id}": v.points for v in inputs.views}}
    output = node.component.process(node.assembler.assemble(AssemblyContext(source), artifacts))
    reference = json.loads((Path(__file__).parents[2]/"fixtures/tennis_scene/ball3d_default_main.json").read_text())
    for name in ("positions", "valid"):
        actual = getattr(output.ball.trajectory, name)
        np.testing.assert_array_equal(actual, np.asarray(reference[name], dtype=actual.dtype))
    np.testing.assert_array_equal(output.ball.trajectory.inliers, reference["used_views"])
    np.testing.assert_array_equal(output.ball.uv_px, np.asarray(reference["accepted_uv_px"], np.float32))
    np.testing.assert_array_equal(output.ball.trajectory.reasons, reference["rejection_codes"])
    np.testing.assert_array_equal(output.ball.trajectory.reprojection_px, np.asarray(reference["reprojection_px"], np.float32))


def test_public_config_component_codec_and_lovo_share_production_algorithm(tmp_path):
    inputs, _ = make_gmm_inputs()
    cfg, node, source = node_for(tmp_path, inputs, "candidate_gmm")
    assert isinstance(node.component, CandidateGMMModule)
    assert node.bindings["detections_right"] == "ball_detection/right"
    assert node.io.inputs["detections_right"].version == 2
    store = make_store(tmp_path/"store", inputs)
    artifacts = {"alignment": store.load(store.active("camera_alignment"), ArtifactCodec(CameraAlignmentOutput)),
                 **{f"detections_{v.camera.camera_id}": v.detections for v in inputs.views}}
    output = node.component.process(node.assembler.assemble(AssemblyContext(source), artifacts))
    (tmp_path/"codec").mkdir()
    codec = ArtifactCodec(CandidateGMMOutput)
    payload, arrays = codec.dump(output, tmp_path/"codec")
    restored = codec.load(payload, tmp_path/"codec", arrays)
    runs = run_method(CandidateGMM(cfg.ball_candidate_gmm), inputs,
        bundle_sha256="synthetic", code_commit="synthetic", code_fingerprint="synthetic")
    np.testing.assert_array_equal(runs[0].result.positions, restored.ball.trajectory.positions)
    np.testing.assert_array_equal(runs[0].result.covariance, restored.covariance)
    for run in runs:
        save_run(tmp_path/run.key, run)
        loaded = load_run(tmp_path/run.key)
        np.testing.assert_array_equal(loaded.result.covariance, run.result.covariance)
        np.testing.assert_array_equal(loaded.result.valid, [True, True, False, True, True])
    # Alter excluded evidence, but its LOVO result remains identical.
    changed = replace(inputs, views=(replace(inputs.views[0], detections=replace(inputs.views[0].detections,
        evidence=inputs.views[1].detections.evidence)), *inputs.views[1:]))
    two_view = CandidateGMM(cfg.ball_candidate_gmm)(changed.select_cameras(inputs.camera_ids[1:]))
    np.testing.assert_array_equal(two_view.positions, runs[1].result.positions)
    artifacts["detections_right"] = replace(inputs.views[0].detections, evidence=None,
        score_semantics="annotation_acceptance_not_probability")
    with pytest.raises(ValueError, match="evidence"):
        node.assembler.assemble(AssemblyContext(source), artifacts)

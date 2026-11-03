import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.baseline import PipelineBaseline
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.evaluation.ball3d.topk import TopKMultiHypothesis
from src.tennis_scene.pipeline.components.ball_topk import (
    TopKBallOutput,
    TopKBallTriangulationModule,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationModule
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.support.ball3d_evaluation import make_inputs, make_scoring_bundle, make_store
from tests.support.ball3d_topk import make_topk_inputs
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)


def node_for(tmp_path, inputs, method):
    from src.tennis_scene.pipeline.definition import enabled_model_assets

    cfg = runtime(tmp_path, overrides=() if method is None else (f"ball_reconstruction.method={method}",))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    source = make_scoring_bundle(inputs).source
    node = next(n for n in standard_definition(cfg, source, code_identity="topk-test") if n.name == "ball_triangulation")
    return cfg, node, source


def test_default_pipeline_matches_main_arrays(tmp_path):
    inputs = make_inputs(frames=12)
    cfg, node, source = node_for(tmp_path, inputs, None)
    assert cfg.ball_method == "triangulation"
    assert isinstance(node.component, BallTriangulationModule)
    assert node.bindings["ball_right"] == "ball_points/right"
    store = make_store(tmp_path / "reference-store", inputs)
    artifacts = {"alignment": store.load(store.active("camera_alignment"), ArtifactCodec(CameraAlignmentOutput)),
                 "calibration": store.load(store.active("court_calibration"), ArtifactCodec(CourtCalibrationOutput)),
                 **{f"ball_{v.camera.camera_id}": v.points for v in inputs.views}}
    output = node.component.process(node.assembler.assemble(AssemblyContext(source), artifacts))
    reference = json.loads((Path(__file__).parents[2] / "fixtures/tennis_scene/ball3d_default_main.json").read_text())
    for name in ("positions", "valid"):
        np.testing.assert_array_equal(getattr(output.ball.trajectory, name), np.asarray(reference[name], dtype=getattr(output.ball.trajectory, name).dtype))
    np.testing.assert_array_equal(output.ball.trajectory.inliers, reference["used_views"])
    np.testing.assert_array_equal(output.ball.uv_px, np.asarray(reference["accepted_uv_px"], np.float32))
    np.testing.assert_array_equal(output.ball.trajectory.reasons, reference["rejection_codes"])
    # The unchanged solver computes residuals from float64 fit.x, before the
    # separate float32 position cast. Switching the OpenBLAS kernel reproduces
    # one residual ULP of drift with every saved position still identical.
    # Four float32 epsilons cover the observed CI relative drift <=1.9e-7;
    # even at 20px this is <1e-5px. atol=0 keeps exact zeros exact. All other
    # arrays above deliberately retain bitwise equality against main.
    np.testing.assert_allclose(output.ball.trajectory.reprojection_px,
        np.asarray(reference["reprojection_px"], np.float32),
        rtol=4 * np.finfo(np.float32).eps, atol=0)
    # The baseline adapter and the default public component share the same output.
    baseline = PipelineBaseline()(inputs)
    np.testing.assert_array_equal(baseline.positions, output.ball.trajectory.positions)


def test_config_definition_assembly_codec_and_lovo_use_the_same_candidate_algorithm(tmp_path):
    inputs, xyz, indices = make_topk_inputs()
    cfg, node, source = node_for(tmp_path, inputs, "topk_multihypothesis")
    assert isinstance(node.component, TopKBallTriangulationModule)
    assert node.bindings["detections_right"] == "ball_detection/right"
    assert node.io.inputs["detections_right"].version == 2
    store = make_store(tmp_path / "reference-store", inputs)
    artifacts = {"alignment": store.load(store.active("camera_alignment"), ArtifactCodec(CameraAlignmentOutput)),
                 **{f"detections_{v.camera.camera_id}": v.detections for v in inputs.views}}
    data = node.assembler.assemble(AssemblyContext(source), artifacts)
    output = node.component.process(data)
    assert output.association is not None
    np.testing.assert_allclose(output.ball.trajectory.positions, xyz, atol=2e-6)
    np.testing.assert_array_equal(output.association.selected_indices, indices)
    codec = ArtifactCodec(TopKBallOutput)
    (tmp_path / "codec").mkdir()
    payload, arrays = codec.dump(output, tmp_path / "codec")
    loaded = codec.load(payload, tmp_path / "codec", arrays)
    assert loaded.association is not None
    np.testing.assert_array_equal(loaded.association.selected_uv_px, output.association.selected_uv_px)
    runs = run_method(TopKMultiHypothesis(cfg.ball_topk), inputs,
        bundle_sha256="synthetic", code_commit="synthetic", code_fingerprint="synthetic")
    for run in runs:
        save_run(tmp_path / run.key, run)
        restored = load_run(tmp_path / run.key)
        diag = restored.result.diagnostics
        assert diag["selected_candidate_indices"].shape == (len(restored.result.camera_ids), len(xyz))
    np.testing.assert_array_equal(runs[0].result.positions, loaded.ball.trajectory.positions)
    np.testing.assert_array_equal(runs[0].result.used_views, loaded.ball.trajectory.inliers)
    # Changing the excluded camera cannot change its two-view fold.
    modified = replace(inputs, views=(replace(inputs.views[0], detections=replace(inputs.views[0].detections,
        evidence=inputs.views[1].detections.evidence)), *inputs.views[1:]))
    result = TopKMultiHypothesis()(modified.select_cameras(inputs.camera_ids[1:]))
    np.testing.assert_array_equal(result.positions, runs[1].result.positions)
    artifacts["detections_right"] = replace(inputs.views[0].detections, evidence=None,
                                              score_semantics="annotation_acceptance_not_probability")
    with pytest.raises(ValueError, match="evidence"):
        node.assembler.assemble(AssemblyContext(source), artifacts)

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.bundle import create_bundle
from src.tennis_scene.evaluation.ball3d.dense_likelihood import DenseHeatmapMethod
from src.tennis_scene.evaluation.ball3d.runner import run_method
from src.tennis_scene.evaluation.ball3d.storage import load_typed, save_typed
from src.tennis_scene.pipeline.components.ball_likelihood import DenseBallOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.support.ball3d_dense import dense_inputs
from tests.support.ball3d_evaluation import make_inputs, make_store
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)


def test_default_pipeline_matches_main_fixed_arrays(tmp_path):
    inputs = make_inputs(12)
    store = make_store(tmp_path / "source", inputs)
    bundle = create_bundle(store.root, tmp_path / "bundle", clip_id=inputs.clip_id,
                           source_code_commit="a" * 40, source_code_evidence="synthetic main regression fixture")
    cfg = runtime(tmp_path)
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    node = next(n for n in standard_definition(cfg, bundle.source, code_identity="test") if n.name == "ball_triangulation")
    artifacts = {f"ball_{v.camera.camera_id}": v.points for v in inputs.views}
    index = json.loads((store.root / "scene.json").read_text())
    for port, key, typ in (("alignment", "camera_alignment", CameraAlignmentOutput),):
        ref = index["artifacts"][key]
        doc = json.loads((store.root / ref["path"]).read_text())
        artifacts[port] = ArtifactCodec(typ).load(doc["payload"], (store.root / ref["path"]).parent, doc["arrays"])
    # The standard assembler also requires calibration; read it through its codec.
    from src.tennis_scene.pipeline.components.court_calibration import (
        CourtCalibrationOutput,
    )
    ref = index["artifacts"]["court_calibration"]
    doc = json.loads((store.root / ref["path"]).read_text())
    artifacts["calibration"] = ArtifactCodec(CourtCalibrationOutput).load(doc["payload"], (store.root / ref["path"]).parent, doc["arrays"])
    result = node.component.process(node.assembler.assemble(AssemblyContext(bundle.source), artifacts)).ball
    expected = json.loads((Path(__file__).parents[2] / "fixtures/tennis_scene/ball3d_dense_default_main.json").read_text())
    np.testing.assert_array_equal(result.trajectory.positions, np.array(expected["positions"], np.float32))
    np.testing.assert_array_equal(result.trajectory.valid, expected["valid"])
    np.testing.assert_array_equal(result.trajectory.inliers, expected["used_views"])
    np.testing.assert_array_equal(result.uv_px, np.array(expected["uv_px"], np.float32))
    np.testing.assert_array_equal(result.visibility, expected["visibility"])
    np.testing.assert_array_equal(result.trajectory.reasons, expected["reasons"])
    np.testing.assert_array_equal(result.trajectory.reprojection_px, np.array(expected["reprojection_px"], np.float32))
    assert node.bindings["ball_right"] == "ball_points/right"


def test_config_definition_component_codec_and_lovo_share_dense_algorithm(tmp_path):
    inputs, xyz, _ = dense_inputs()
    store = make_store(tmp_path / "source", inputs)
    bundle = create_bundle(store.root, tmp_path / "bundle", clip_id=inputs.clip_id,
                           source_code_commit="a" * 40, source_code_evidence="synthetic dense fixture")
    cfg = runtime(tmp_path, overrides=("ball_reconstruction.method=dense_likelihood", "ball_reconstruction.dense_likelihood.interpolation=quadratic_logit"))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    nodes = {n.name: n for n in standard_definition(cfg, bundle.source, code_identity="test")}
    node = nodes["ball_triangulation"]
    assert nodes["court_side"].bindings["ball_right"] == "ball_points/right"
    assert node.bindings["ball_right"] == "ball_detection/right"
    assert node.io.inputs["ball_right"].version == 2
    # Use the real stored geometry in explicit camera order.
    index = json.loads((store.root / "scene.json").read_text())
    ref = index["artifacts"]["camera_alignment"]
    doc = json.loads((store.root / ref["path"]).read_text())
    alignment = ArtifactCodec(CameraAlignmentOutput).load(doc["payload"], (store.root / ref["path"]).parent, doc["arrays"])
    artifacts = {"alignment": alignment, **{f"ball_{v.camera.camera_id}": v.detections for v in inputs.views}}
    assembled = node.assembler.assemble(AssemblyContext(bundle.source), artifacts)
    result = node.component.process(assembled)
    assert result.dense.valid.all()
    np.testing.assert_allclose(result.ball.trajectory.positions, xyz, atol=.01)
    folder = tmp_path / "saved"
    digest = save_typed(folder, result, DenseBallOutput, schema="dense_test")
    restored = load_typed(folder, DenseBallOutput, schema="dense_test", expected_sha256=digest)
    np.testing.assert_array_equal(restored.dense.positions, result.dense.positions)
    adapter = DenseHeatmapMethod(cfg.dense_likelihood)
    runs = run_method(adapter, bundle.inputs, bundle_sha256=bundle.manifest_sha256, code_commit="b" * 40, code_fingerprint="c" * 64)
    np.testing.assert_array_equal(runs[0].result.positions, result.ball.trajectory.positions)
    assert all(r.result.valid.all() for r in runs)
    assert [r.result.camera_ids for r in runs[1:]] == [("near", "far"), ("right", "far"), ("right", "near")]
    artifacts["ball_right"] = replace(inputs.views[0].detections, camera_id="wrong")
    with pytest.raises(ValueError, match="camera-bound"):
        node.assembler.assemble(AssemblyContext(bundle.source), artifacts)


@pytest.mark.parametrize("override", ["ball_reconstruction.method=unknown", "ball_reconstruction.dense_likelihood.temperature=-1"])
def test_dense_config_fails_before_pipeline_execution(tmp_path, override):
    with pytest.raises(ValueError):
        runtime(tmp_path, overrides=(override,))

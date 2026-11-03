import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.baseline import PipelineBaseline
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.evaluation.ball3d.segment_physics import SegmentPhysicsMethod
from src.tennis_scene.pipeline.components.ball_segment_physics import (
    SegmentPhysicsModule,
    SegmentPhysicsOutput,
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
from tests.support.ball3d_segment_physics import synthetic_method_inputs
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)


def node_and_data(tmp_path, inputs, method):
    cfg = runtime(
        tmp_path,
        overrides=() if method is None else (f"ball_reconstruction.method={method}",),
    )
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    source = make_scoring_bundle(inputs).source
    node = next(
        n
        for n in standard_definition(cfg, source, code_identity="physics-test")
        if n.name == "ball_triangulation"
    )
    store = make_store(tmp_path / "source-store", inputs)
    artifacts = {
        "alignment": store.load(
            store.active("camera_alignment"), ArtifactCodec(CameraAlignmentOutput)
        ),
        "calibration": store.load(
            store.active("court_calibration"), ArtifactCodec(CourtCalibrationOutput)
        ),
        **{f"ball_{v.camera.camera_id}": v.points for v in inputs.views},
        **{f"detections_{v.camera.camera_id}": v.detections for v in inputs.views},
    }
    return cfg, node, source, artifacts


def test_default_pipeline_matches_fixed_main_arrays(tmp_path):
    inputs = make_inputs(frames=12)
    cfg, node, source, artifacts = node_and_data(tmp_path, inputs, None)
    assert cfg.ball_method == "triangulation"
    assert isinstance(node.component, BallTriangulationModule)
    assert node.bindings["ball_right"] == "ball_points/right"
    assert not any(k.startswith("detections_") for k in node.bindings)
    output = node.component.process(
        node.assembler.assemble(AssemblyContext(source), artifacts)
    )
    reference = json.loads(
        (
            Path(__file__).parents[2]
            / "fixtures/tennis_scene/ball3d_segment_default_main.json"
        ).read_text()
    )
    for field in ("positions", "valid"):
        actual = getattr(output.ball.trajectory, field)
        np.testing.assert_array_equal(
            actual, np.asarray(reference[field], dtype=actual.dtype)
        )
    np.testing.assert_array_equal(
        output.ball.trajectory.inliers, reference["used_views"]
    )
    np.testing.assert_array_equal(
        output.ball.trajectory.reasons, reference["rejection_codes"]
    )
    np.testing.assert_array_equal(
        output.ball.uv_px, np.asarray(reference["accepted_uv_px"], np.float32)
    )


def test_opt_in_pipeline_codec_and_lovo_use_same_solver_without_heldout_evidence(
    tmp_path,
):
    inputs, *_ = synthetic_method_inputs()
    cfg, node, source, artifacts = node_and_data(tmp_path, inputs, "segment_physics")
    assert isinstance(node.component, SegmentPhysicsModule)
    assert node.bindings["detections_right"] == "ball_detection/right"
    assert node.io.inputs["detections_right"].version == 2
    data = node.assembler.assemble(AssemblyContext(source), artifacts)
    output = node.component.process(data)
    assert output.fit is not None and output.fit.diagnostics["status"] == "ok"
    directory = tmp_path / "codec"
    directory.mkdir()
    codec = ArtifactCodec(SegmentPhysicsOutput)
    payload, arrays = codec.dump(output, directory)
    restored = codec.load(payload, directory, arrays)
    assert restored.fit is not None
    np.testing.assert_array_equal(
        restored.fit.time_offsets_seconds, output.fit.time_offsets_seconds
    )
    np.testing.assert_array_equal(
        restored.fit.diagnostics["offset_covariance_seconds2"],
        output.fit.diagnostics["offset_covariance_seconds2"],
    )
    method = SegmentPhysicsMethod(cfg.ball_segment_physics, PipelineBaseline())
    runs = run_method(
        method,
        inputs,
        bundle_sha256="synthetic",
        code_commit="synthetic",
        code_fingerprint="synthetic",
    )
    np.testing.assert_array_equal(
        runs[0].result.positions, output.ball.trajectory.positions
    )
    np.testing.assert_array_equal(
        runs[0].result.used_views, output.ball.trajectory.inliers
    )
    for run in runs:
        save_run(tmp_path / run.key, run)
        loaded = load_run(tmp_path / run.key)
        np.testing.assert_array_equal(
            loaded.result.time_offsets_seconds, run.result.time_offsets_seconds
        )
        assert loaded.result.diagnostics["camera_ids"] == list(run.result.camera_ids)
    # Corrupt only the excluded camera's raw evidence. The held-out fold remains
    # byte-identical and re-estimates its own anchor among retained cameras.
    excluded = inputs.views[0]
    altered_view = replace(
        excluded,
        detections=replace(
            excluded.detections, evidence=inputs.views[1].detections.evidence
        ),
    )
    altered = replace(inputs, views=(altered_view, *inputs.views[1:]))
    selected = method(altered.select_cameras(inputs.camera_ids[1:]))
    np.testing.assert_array_equal(selected.positions, runs[1].result.positions)
    assert selected.diagnostics["anchor_camera"] == "far"
    bad = {
        **artifacts,
        "detections_right": replace(
            excluded.detections,
            evidence=None,
            score_semantics="annotation_acceptance_not_probability",
        ),
    }
    with pytest.raises(ValueError, match="evidence"):
        node.assembler.assemble(AssemblyContext(source), bad)
    with pytest.raises(ValueError, match="camera"):
        node.assembler.assemble(
            AssemblyContext(source),
            {**artifacts, "detections_right": inputs.views[1].detections},
        )

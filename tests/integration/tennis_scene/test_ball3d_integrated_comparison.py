"""Experiment compositions retain fold boundaries and portable artifact links."""

import importlib.util
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.tennis_scene.evaluation.ball3d.contracts import Ball3DResult
from src.tennis_scene.evaluation.ball3d.runner import run_method
from src.tennis_scene.evaluation.ball3d.scoring import project_result
from src.tennis_scene.visualization.ball3d.report import broken_html_references
from tests.support.ball3d_evaluation import PipelineBaseline
from tests.support.ball3d_topk import make_topk_inputs, projected_pixels
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import runtime

RUN = (
    Path(__file__).resolve().parents[3]
    / "knowledge/runs/run-ball3d-integrated-comparison-20261104"
)


def load_script(monkeypatch, name):
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    # Avoid retaining common/methods modules from another experiment's run directory.
    for key in ("common", "methods"):
        monkeypatch.delitem(sys.modules, key, raising=False)
    spec = importlib.util.spec_from_file_location(
        "integrated_test_" + name, RUN / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ba_camera_composition_uses_each_fold_and_never_applies_clock_or_heldout_camera(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, "methods")
    inputs, xyz, _ = make_topk_inputs()
    cfg = runtime(tmp_path)
    displacement = np.array([0.24, -0.17, 0.08])
    calls = []

    def controlled_ba(selected):
        calls.append(selected.camera_ids)
        cameras = tuple(
            replace(
                v.camera,
                translation=v.camera.translation - v.camera.rotation @ displacement,
            )
            for v in selected.views
        )
        return Ball3DResult(
            selected.camera_ids,
            selected.frame_indices,
            np.zeros_like(xyz, dtype=np.float32),
            np.zeros(len(xyz), bool),
            np.zeros((len(cameras), len(xyz)), bool),
            cameras=cameras,
            time_offsets_seconds=np.arange(len(cameras), dtype=np.float64) * 0.003,
            diagnostics={"selected": selected.camera_ids},
        )

    monkeypatch.setattr(module, "ba_method", lambda config: controlled_ba)
    method = module.Composition(
        "ba_cameras_topk",
        {
            "composition": "ba_cameras",
            "downstream": "topk",
            "ba": {},
            "topk": asdict(cfg.ball_topk),
        },
        PipelineBaseline(),
    )
    values = run_method(
        method,
        inputs,
        bundle_sha256="test",
        code_commit="test",
        code_fingerprint="test",
    )
    assert calls == [
        inputs.camera_ids,
        *(
            tuple(c for c in inputs.camera_ids if c != held)
            for held in inputs.camera_ids
        ),
    ]
    for run in values:
        np.testing.assert_allclose(
            run.result.positions, xyz + displacement, atol=3e-6, rtol=0
        )
        assert run.result.time_offsets_seconds is None
        assert tuple(c.camera_id for c in run.result.cameras) == run.result.camera_ids
        if run.held_out_camera is not None:
            uv, valid = project_result(inputs, run.result, run.held_out_camera)
            camera = inputs.views[inputs.camera_ids.index(run.held_out_camera)].camera
            np.testing.assert_allclose(
                uv, projected_pixels(camera, xyz + displacement), atol=2e-5, rtol=0
            )
            assert valid.all()
    # A held-out payload can change without affecting its fold's intermediate or output.
    altered = replace(
        inputs,
        views=(
            replace(
                inputs.views[0],
                detections=replace(
                    inputs.views[0].detections,
                    evidence=inputs.views[1].detections.evidence,
                ),
            ),
            *inputs.views[1:],
        ),
    )
    fold = method(altered.select_cameras(inputs.camera_ids[1:]))
    np.testing.assert_array_equal(fold.positions, values[1].result.positions)


def test_portable_report_uses_relative_video_files_after_copy(monkeypatch, tmp_path):
    module = load_script(monkeypatch, "report")
    output = tmp_path / "original"
    output.mkdir()
    for directory in ("all-methods", "shortlist"):
        (output / directory).mkdir()
        (output / directory / "comparison.mp4").write_bytes(b"test link only")
    for filename in (
        "metrics.csv",
        "cohorts.csv",
        "execution-plan.json",
        "prior-comparison.json",
        "scores.json",
        "cohort-samples.csv",
        "verification.json",
    ):
        (output / filename).write_text("{}")
    for segment in module.SEGMENTS:
        (output / f"metrics-{segment}.png").write_bytes(b"test link only")
    rows = [
        {
            "method": "pipeline_baseline",
            "segment": segment,
            "held_out_camera": camera,
            "count": 0,
            "eligible": 5,
            "coverage": 0.0,
            "median_px": None,
            "p90_px": None,
            "above20_rate": None,
            "all_view_valid_rate": 0.0,
        }
        for segment in module.SEGMENTS
        for camera in ("pooled", "cam0", "cam1", "cam2", "all_views")
    ]
    metric = {"count": 0, "median": None, "p90": None, "above_threshold_rate": None}
    cohorts = {
        "pipeline_baseline": {
            "strata": [
                {
                    "segment": segment,
                    "held_out_camera": "all",
                    "reference_kind": "all",
                    "common": {"method": metric, "baseline": metric},
                    "method_only": {"method": metric},
                    "baseline_only": {"baseline": metric},
                }
                for segment in module.SEGMENTS
            ]
        }
    }
    module.write_page(
        output,
        {"methods": [{"name": "pipeline_baseline"}], "diagnostics": []},
        rows,
        cohorts,
        {},
        [],
        [],
        ["pipeline_baseline"],
    )
    copied = tmp_path / "portable"
    shutil.copytree(output, copied)
    assert not broken_html_references(copied / "index.html")
    page = (copied / "index.html").read_text()
    assert (
        'src="all-methods/comparison.mp4"' in page
        and 'src="shortlist/comparison.mp4"' in page
    )
    assert "data:video" not in page and str(tmp_path) not in page

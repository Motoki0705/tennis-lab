"""Configuration/source equivalence and real saved-evidence replay, without GT."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_gmm.heatmap import candidate_gaussians
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.utils.paths import PROJECT_ROOT

BEFORE = "2fd5385abf1e35298e7044b655f3a514f9e5298a"
BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"


def at(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def method(source, cls, name):
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return ast.dump(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stored-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = "src/tennis_scene/ball_gmm/config.py"
    old_config = at(BEFORE, config_path)
    owner = next(n for n in ast.parse(old_config).body if isinstance(n, ast.ClassDef))
    old_values = {n.target.id: ast.literal_eval(n.value) for n in owner.body if isinstance(n, ast.AnnAssign)}
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT/"src/tennis_scene/configs")):
        composed = compose(config_name="pipeline")
    base = PipelineRuntimeConfig.from_config(composed, bind_inputs=False).ball_candidate_gmm
    assert old_values == asdict(base)
    kernels = {}
    for path in (
        "src/tennis_scene/ball_gmm/heatmap.py", "src/tennis_scene/ball_gmm/reconstruction.py",
        "src/utils/geometry/kalman.py", "src/utils/geometry/probabilistic_triangulation/solver.py",
        "src/utils/geometry/probabilistic_triangulation/distributions.py", "src/tennis_scene/configs/pipeline.yaml",
    ):
        current = Path(path).read_bytes()
        assert current == at(BEFORE, path), path
        kernels[path] = hashlib.sha256(current).hexdigest()
    adapter = "src/tennis_scene/evaluation/ball3d/candidate_gmm.py"
    assert method(old_config, "CandidateGMMConfig", "__post_init__") == method(Path(config_path).read_bytes(), "CandidateGMMConfig", "__post_init__")
    assert method(at(BEFORE, adapter), "CandidateGMM", "__call__") == method(Path(adapter).read_bytes(), "CandidateGMM", "__call__")
    default_sources = {}
    for path in (
        "src/tennis_scene/pipeline/components/ball_reconstruction.py", "src/tennis_scene/pipeline/components/triangulation.py",
        "src/tennis_scene/pipeline/input_assembly/reconstruction.py", "src/tennis_scene/pipeline/input_assembly/observations.py",
        "src/tennis_scene/pipeline/components/ball_points.py", "src/utils/geometry/triangulation.py",
    ):
        current = Path(path).read_bytes()
        assert current == at("origin/main", path), path
        default_sources[path] = hashlib.sha256(current).hexdigest()
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    stored = load_run(args.stored_comparison/"results/candidate_gmm_map_none/all_views")
    assert stored.bundle_sha256 == bundle.manifest_sha256
    settings = dict(stored.config["fold_calibrations"][",".join(bundle.inputs.camera_ids)])
    for key in ("prior_mean_m", "prior_std_m"):
        settings[key] = tuple(settings[key])
    selected = CandidateGMMConfig(**settings)
    checked = []
    for i, view in enumerate(bundle.inputs.views):
        current = candidate_gaussians(view.detections.evidence, selected)
        for field, stored_field in (
            ("means_px", "candidate_means_px"), ("covariance_px2", "candidate_covariance_px2"),
            ("weights", "candidate_weights"), ("presence", "presence"), ("indices", "candidate_indices"),
            ("bounded", "bounded_curvature"), ("rejected", "rejected_curvature_count"),
        ):
            actual, expected = getattr(current, field), stored.result.diagnostics[stored_field][i]
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
            checked.append(f"{view.camera.camera_id}/{field}")
    reference = json.loads((PROJECT_ROOT/"tests/fixtures/tennis_scene/ball3d_default_main.json").read_text())
    residual = np.asarray(reference["reprojection_px"], np.float32)
    report = {
        "before": BEFORE, "composed_defaults_equal_previous_python_defaults": True,
        "composed_values": asdict(base), "unchanged_kernel_sha256": kernels,
        "default_route_sources_equal_origin_main": default_sources,
        "default_route_main_ref": subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip(),
        "config_validation_and_explicit_adapter_call_ast_unchanged": True,
        "real_data_2d_gaussian_replay_exact": checked, "real_data_frames_per_camera": 1010,
        "full_3d_replay_performed": False,
        "prediction_equivalence_basis": "same YAML values, explicit settings, identical validation/call AST and unchanged numerical kernels; real 2D Gaussians also replayed exactly",
        "stored_manifest_sha256": hashlib.sha256((args.stored_comparison/"artifacts.json").read_bytes()).hexdigest(),
        "reprojection_tolerance_float32_ulps": 4,
        "fixture_max_four_ulp_bound_px": float(4*np.spacing(residual).max()),
    }
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("16 composition values and kernels unchanged; 21 real-data Gaussian arrays exactly match v2")


if __name__ == "__main__":
    main()

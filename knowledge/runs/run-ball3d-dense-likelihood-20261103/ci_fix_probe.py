"""CPU-only rounding investigation and pinned-input prediction invariance."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import MISSING, asdict, fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
from hydra import compose, initialize_config_dir

from src.tennis_scene.ball_likelihood.config import DenseLikelihoodConfig
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.baseline import PipelineBaseline
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.dense_likelihood import DenseHeatmapMethod
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.pipeline.components import ball_reconstruction
from src.utils.geometry import triangulation
from src.utils.paths import PROJECT_ROOT
from tests.support.ball3d_evaluation import make_inputs

OLD_HEAD = "4a4f48a616f81c85dd7039187d1a88a5e4ec7709"
BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"


def blob(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value))


def ulp_difference(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return int(np.max(np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))))


def math_probe():
    source = make_inputs(12)
    fitted = []
    observations = {}
    original = triangulation.least_squares
    original_triangulate = ball_reconstruction.triangulate_points

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        fitted.append(result.x.copy())
        return result

    def capture_triangulation(uv, visible, *args, **kwargs):
        observations["uv"] = uv.copy()
        observations["visible"] = visible.copy()
        observations["result"] = original_triangulate(uv, visible, *args, **kwargs)
        return observations["result"]

    with patch.object(triangulation, "least_squares", capture), patch.object(ball_reconstruction, "triangulate_points", capture_triangulation):
        result = PipelineBaseline(min_frames=1)(source)
    expected = json.loads(Path("tests/fixtures/tennis_scene/ball3d_dense_default_main.json").read_text())
    ref = np.array(expected["reprojection_px"], np.float32)
    np.testing.assert_array_equal(result.positions, np.array(expected["positions"], np.float32))
    np.testing.assert_array_equal(result.valid, expected["valid"])
    np.testing.assert_array_equal(result.used_views, expected["used_views"])
    assert result.diagnostics is not None
    uv = observations["uv"].astype(np.float64)
    visible = observations["visible"]
    np.testing.assert_array_max_ulp(observations["result"].reprojection_px, ref, maxulp=4)
    points = np.stack(fitted)
    residual_einsum = np.zeros(ref.shape)
    residual_scalar = np.zeros(ref.shape)
    # Two mathematically equivalent float64 projection reductions. Python fsum
    # has a different summation path from NumPy's SIMD/einsum kernels.
    for frame, point in enumerate(points):
        for index, view in enumerate(source.views):
            if not visible[index, frame]:
                continue
            camera = view.camera
            for destination, scalar in ((residual_einsum, False), (residual_scalar, True)):
                if scalar:
                    xyz = np.array([math.fsum(float(a) * float(b) for a, b in zip(row, point, strict=True)) + t
                                    for row, t in zip(camera.rotation, camera.translation, strict=True)])
                    q = np.array([math.fsum(float(a) * float(b) for a, b in zip(row, xyz, strict=True)) for row in camera.intrinsic])
                else:
                    xyz = np.einsum("ij,j->i", camera.rotation, point) + camera.translation
                    q = np.einsum("ij,j->i", camera.intrinsic, xyz)
                destination[index, frame] = np.linalg.norm(q[:2] / q[2] - uv[index, frame])
    simd = np.__config__.show(mode="dicts")
    # Controlled numerical demonstration, not a change to the implementation:
    # Eight float64 ULPs are invisible after storing float32 XYZ, but can change the
    # cancellation-sensitive pixel residual computed before that conversion.
    bumped = points.copy()
    for _ in range(8):
        bumped = np.nextafter(bumped, np.inf)
    np.testing.assert_array_equal(points.astype(np.float32), bumped.astype(np.float32))
    projections = []
    for values in (points, bumped):
        residuals = np.zeros_like(ref)
        for frame, point in enumerate(values):
            rotations = np.stack([v.camera.rotation for v in source.views])
            translations = np.stack([v.camera.translation for v in source.views])
            intrinsics = np.stack([v.camera.intrinsic for v in source.views])
            xyz = np.einsum("vij,j->vi", rotations, point) + translations
            q = np.einsum("vij,vj->vi", intrinsics, xyz)
            residuals[:, frame] = np.where(visible[:, frame],
                                           np.linalg.norm(q[:, :2] / q[:, 2, None] - uv[:, frame], axis=-1), 0)
        projections.append(residuals)
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "openblas_coretype_override": os.environ.get("OPENBLAS_CORETYPE"),
            "numpy_build": simd, "fitted_dtype": str(points.dtype), "stored_positions_dtype": str(result.positions.dtype),
            "stored_residual_dtype": str(ref.dtype), "positions_exact": True,
            "max_float64_projection_difference_px": float(np.max(np.abs(residual_einsum - residual_scalar))),
            "max_float32_projection_difference_ulp": ulp_difference(residual_einsum, residual_scalar),
            "max_float32_projection_difference_px": float(np.max(np.abs(residual_einsum.astype(np.float32) - residual_scalar.astype(np.float32)))),
            "reference_max_residual_px": float(ref.max()), "four_ulp_max_px": float((4 * np.spacing(ref)).max()),
            "controlled_float64_nextafter_steps": 8,
            "controlled_float64_nextafter_positions_exact_in_float32": True,
            "controlled_float64_nextafter_max_residual_difference_px": float(np.max(np.abs(projections[0] - projections[1]))),
            "controlled_float64_nextafter_max_residual_difference_ulp": ulp_difference(*projections),
            "actual_fit_residual_vs_reference_max_ulp": ulp_difference(residual_einsum, ref),
            "residuals_einsum_float32": residual_einsum.astype(np.float32).tolist(),
            "residuals_scalar_float32": residual_scalar.astype(np.float32).tolist()}


def invariance(bundle_path, comparison):
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        composed = compose(config_name="pipeline")
    defaults = PipelineRuntimeConfig.from_config(composed, bind_inputs=False).dense_likelihood
    config_path = "src/tennis_scene/ball_likelihood/config.py"
    tree = ast.parse(blob(OLD_HEAD, config_path))
    old_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DenseLikelihoodConfig")
    old_defaults = {n.target.id: ast.literal_eval(n.value) for n in old_class.body if isinstance(n, ast.AnnAssign)}
    assert canonical(asdict(defaults)) == canonical(old_defaults)
    assert all(f.default is MISSING and f.default_factory is MISSING for f in fields(DenseLikelihoodConfig))
    new_tree = ast.parse(Path(config_path).read_text())
    new_class = next(n for n in new_tree.body if isinstance(n, ast.ClassDef) and n.name == "DenseLikelihoodConfig")
    old_methods = [ast.dump(n) for n in old_class.body if isinstance(n, ast.FunctionDef)]
    new_methods = [ast.dump(n) for n in new_class.body if isinstance(n, ast.FunctionDef)]
    assert old_methods == new_methods
    paths = ["src/tennis_scene/ball_likelihood/likelihood.py", "src/tennis_scene/ball_likelihood/reconstruction.py",
             "src/tennis_scene/configs/pipeline.yaml", "src/tennis_scene/configuration.py",
             "src/tennis_scene/pipeline/definition.py"]
    unchanged = {}
    for path in paths:
        before, after = blob(OLD_HEAD, path), Path(path).read_bytes()
        assert before == after, path
        unchanged[path] = sha(after)
    main_paths = ["src/utils/geometry/triangulation.py", "src/utils/geometry/temporal_smoothing.py",
                  "src/tennis_scene/pipeline/components/ball_reconstruction.py", "src/tennis_scene/pipeline/components/triangulation.py",
                  "src/tennis_scene/pipeline/components/ball_points.py", "src/tennis_scene/pipeline/input_assembly/reconstruction.py"]
    main_identical = {}
    for path in main_paths:
        current = Path(path).read_bytes()
        assert blob("origin/main", path) == current, path
        main_identical[path] = sha(current)
    names = {f.name for f in fields(DenseLikelihoodConfig)}
    checked = []
    for folder in sorted((comparison / "results").glob("*/*")):
        run = load_run(folder)
        if run.method.startswith("dense_"):
            values = {name: run.config[name] for name in names}
            assert canonical(asdict(DenseLikelihoodConfig(**values))) == canonical(values)
        checked.append(str(folder.relative_to(comparison)))
    assert len(checked) == 28
    # Rerun the selected method only, using its explicit, frozen parameters.
    # No scoring labels or retuning are involved.
    saved = load_run(comparison / "results/dense_selected/all_views")
    parameters = DenseLikelihoodConfig(**{name: saved.config[name] for name in names})
    bundle = load_bundle(bundle_path, expected_sha256=BUNDLE_SHA)
    started = time.perf_counter()
    actual = DenseHeatmapMethod(parameters, saved.method)(bundle.inputs)
    matched = []
    for name in ("positions", "valid", "used_views"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(saved.result, name))
        matched.append(name)
    assert actual.diagnostics is not None and saved.result.diagnostics is not None
    for name in actual.diagnostics:
        np.testing.assert_array_equal(actual.diagnostics[name], saved.result.diagnostics[name])
        matched.append("diagnostics." + name)
    elapsed = time.perf_counter() - started
    inventory = json.loads((Path(__file__).parent / "checksums.json").read_text())["files"]
    unchanged_outputs = {}
    for name in ("artifacts.json", "scores.json", "selection.json", "index.html"):
        current_sha = sha((comparison / name).read_bytes())
        assert current_sha == inventory[name]["sha256"], name
        unchanged_outputs[name] = current_sha
    return {"defaults_exactly_match_old_python_values": True, "all_fields_required": True,
            "config_validation_and_helpers_unchanged": True,
            "pre_fix_head": OLD_HEAD, "bundle_manifest_sha256": bundle.manifest_sha256,
            "main_reference_commit": subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip(),
            "configuration_fields": sorted(names), "all_28_saved_configs_accepted": checked,
            "prediction_implementation_and_composition_unchanged": unchanged,
            "default_pipeline_computation_identical_to_main": main_identical,
            "selected_all_views_replay_frames": bundle.inputs.num_frames, "selected_all_views_replay_seconds": elapsed,
            "unchanged_saved_output_checksums": unchanged_outputs,
            "exact_replay_fields": matched, "retuning": False, "predictions_changed": False,
            "scope": "all 28 conditions checked by code/config identity; selected all-view 1010 frames also rerun bitwise"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("math", "invariance"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = math_probe() if args.action == "math" else invariance(args.bundle, args.comparison)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print({"output": str(args.output), "action": args.action}, flush=True)


if __name__ == "__main__":
    main()

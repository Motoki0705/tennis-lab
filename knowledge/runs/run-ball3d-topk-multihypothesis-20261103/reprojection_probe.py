"""Small CPU-only probe of float64 fit/residual and independent float32 casts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def child(output: Path) -> None:
    import numpy as np
    import scipy

    from src.tennis_scene.pipeline.components.camera_alignment import (
        CameraAlignmentOutput,
    )
    from src.tennis_scene.pipeline.components.court_calibration import (
        CourtCalibrationOutput,
    )
    from src.tennis_scene.pipeline.contracts import AssemblyContext
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
    from tests.integration.tennis_scene.test_ball3d_topk import node_for
    from tests.support.ball3d_evaluation import make_inputs, make_store

    geometry = importlib.import_module("src.utils.geometry.triangulation")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs = make_inputs(frames=12)
        _, node, source = node_for(root, inputs, None)
        store = make_store(root / "store", inputs)
        artifacts = {
            "alignment": store.load(store.active("camera_alignment"), ArtifactCodec(CameraAlignmentOutput)),
            "calibration": store.load(store.active("court_calibration"), ArtifactCodec(CourtCalibrationOutput)),
            **{f"ball_{view.camera.camera_id}": view.points for view in inputs.views},
        }
        data = node.assembler.assemble(AssemblyContext(source), artifacts)
        fits = []
        least_squares = geometry.least_squares

        def capture(*args, **kwargs):
            result = least_squares(*args, **kwargs)
            fits.append(result.x.copy())
            return result

        with patch.object(geometry, "least_squares", capture):
            ball = node.component.process(data).ball
        fit64 = np.array(fits)
        assert fit64.dtype == np.float64
        np.testing.assert_array_equal(fit64.astype(np.float32), ball.trajectory.positions)
        # Same float64 projection formula as production, with the saved float32
        # position substituted only in this diagnostic (never in production).
        cameras = data.alignment.geometry.cameras
        rotations = np.stack([camera.rotation for camera in cameras])
        translations = np.stack([camera.translation for camera in cameras])
        intrinsics = np.stack([camera.intrinsic for camera in cameras])

        def residual(points):
            xyz = np.einsum("vij,tj->vti", rotations, points) + translations[:, None]
            uvw = np.einsum("vij,vtj->vti", intrinsics, xyz)
            return np.where(ball.visibility, np.linalg.norm(uvw[..., :2] / uvw[..., 2, None] - ball.uv_px, axis=-1), 0)

        residual64 = residual(fit64)
        residual_from_saved = residual(ball.trajectory.positions.astype(np.float64))
        reference = json.loads(Path("tests/fixtures/tennis_scene/ball3d_default_main.json").read_text())
        identity = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == checksum
                    for name, checksum in reference["source_files"].items()}
        assert all(identity.values())
        output.write_text(json.dumps({
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "npy_disable_cpu_features": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
            "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"),
            "default_solver_files_equal_recorded_main": identity,
            "fit64": fit64.tolist(), "positions": ball.trajectory.positions.tolist(),
            "valid": ball.trajectory.valid.tolist(), "inliers": ball.trajectory.inliers.tolist(),
            "uv_px": ball.uv_px.tolist(), "rejection_codes": ball.trajectory.reasons.tolist(),
            "reprojection_px": ball.trajectory.reprojection_px.tolist(),
            "residual64": residual64.tolist(),
            "max_residual_change_if_recomputed_from_saved_float32_position_px":
                float(np.max(np.abs(residual_from_saved - residual64))),
        }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.output)
        return
    variants = {
        "native": {},
        "reduced_simd": {"NPY_DISABLE_CPU_FEATURES": "AVX2,FMA3,AVX512F"},
        "core2_blas": {"OPENBLAS_CORETYPE": "CORE2"},
    }
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, overrides in variants.items():
            output = Path(directory) / f"{name}.json"
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1", **overrides}
            subprocess.run([sys.executable, __file__, "--child", "--output", str(output)], env=env, check=True)
            results[name] = json.loads(output.read_text())
    import numpy as np

    comparisons = {}
    for name, result in results.items():
        expected = np.array(results["native"]["reprojection_px"], np.float32)
        actual = np.array(result["reprojection_px"], np.float32)
        np.testing.assert_allclose(actual, expected, rtol=4 * np.finfo(np.float32).eps, atol=0)
        comparisons[name] = {
            "exact_arrays": {key: result[key] == results["native"][key]
                             for key in ("positions", "valid", "inliers", "uv_px", "rejection_codes")},
            "different_reprojection_values": int(np.sum(actual != expected)),
            "max_absolute_reprojection_difference_px": float(np.max(np.abs(actual - expected))),
            "max_reprojection_float32_ulps": int(np.max(np.abs(actual.view(np.int32).astype(np.int64) - expected.view(np.int32).astype(np.int64)))),
        }
    args.output.write_text(json.dumps({"variants": results, "comparisons": comparisons,
                                      "residual_rtol": 4 * float(np.finfo(np.float32).eps), "residual_atol": 0}, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()

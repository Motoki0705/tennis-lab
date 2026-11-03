"""Reproduce row-order sensitivity at a pinned revision and at the repaired code."""

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np

from tests.support.ball3d_segment_physics import (
    segment_physics_config,
    synthetic_flights,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new evidence file")
    source = subprocess.check_output(
        ["git", "show", f"{args.revision}:src/tennis_scene/ball_trajectory/fit.py"],
        text=True,
    )
    name = "src.tennis_scene.ball_trajectory._ci_order_probe"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"{args.revision}:fit.py", "exec"), module.__dict__)
    real = module.least_squares
    snapshots, records = [], []

    def traced(function, parameters, **kwargs):
        residual = function(parameters)
        snapshot = (
            parameters.copy(),
            residual.copy(),
            kwargs["jac_sparsity"].toarray(),
        )
        snapshots.append(snapshot)
        result = real(function, parameters, **kwargs)
        cameras = inspect.getclosurevars(function).nonlocals["cameras"]
        records.append(
            {
                "residual_camera_order": [c.camera_id for c in cameras],
                "initial_parameters_sha256": hashlib.sha256(
                    parameters.tobytes()
                ).hexdigest(),
                "initial_residual_sha256": hashlib.sha256(
                    residual.tobytes()
                ).hexdigest(),
                "nfev": int(result.nfev),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "termination": str(result.message),
            }
        )
        return result

    module.least_squares = traced
    cameras, evidence, initial, fps, *_ = synthetic_flights()
    config = segment_physics_config()
    first = module.fit_trajectory(cameras, evidence, initial, fps=fps, config=config)
    order = [2, 0, 1]
    second = module.fit_trajectory(
        tuple(cameras[i] for i in order),
        tuple(evidence[i] for i in order),
        replace(
            initial,
            inliers=initial.inliers[order],
            reprojection_px=initial.reprojection_px[order],
        ),
        fps=fps,
        config=config,
    )
    result = {
        "revision": args.revision,
        "permutation": order,
        "records": records,
        "initial_parameters_equal": np.array_equal(snapshots[0][0], snapshots[1][0]),
        "solver_residual_rows_equal": np.array_equal(snapshots[0][1], snapshots[1][1]),
        "solver_sparsity_equal": np.array_equal(snapshots[0][2], snapshots[1][2]),
        "max_offset_difference_seconds": float(
            np.max(
                np.abs(first.time_offsets_seconds[order] - second.time_offsets_seconds)
            )
        ),
        "positions_bitwise_equal": np.array_equal(first.positions, second.positions),
        "offsets_bitwise_equal": np.array_equal(
            first.time_offsets_seconds[order], second.time_offsets_seconds
        ),
        "covariance_bitwise_equal": np.array_equal(
            first.diagnostics["offset_covariance_seconds2"][np.ix_(order, order)],
            second.diagnostics["offset_covariance_seconds2"],
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

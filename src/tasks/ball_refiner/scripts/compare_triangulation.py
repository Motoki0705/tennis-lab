"""Reproducible CPU-only A/B/C comparison using a frozen camera-only fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy
import torch

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.comparison import (
    VoxelConfig,
    triangulate_samples,
    triangulate_volume,
)
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianPrior3D,
    LaplaceConfig,
    triangulate_gmm,
)
from src.utils.geometry.probabilistic_triangulation.distributions import FloatArray
from src.utils.geometry.triangulation import PinholeCamera

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner.compare_triangulation",
    fields=(
        BoundaryPathField("fixture", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


class Density(Protocol):
    def log_prob(self, points: FloatArray) -> FloatArray: ...
    def sample(self, count: int, rng: np.random.Generator) -> FloatArray: ...


def synthetic_observations(
    cameras: tuple[PinholeCamera, ...],
    sizes: FloatArray,
    truth: FloatArray,
    scenario: str,
    rng: np.random.Generator,
) -> CameraGMM:
    alternatives = (
        [truth, truth + np.array([1.2, 2.4, 0.4])]
        if scenario == "ambiguous"
        else [truth]
    )
    means = np.stack([c.project(np.stack(alternatives))[0] for c in cameras])
    covariance = np.tile(
        np.array([[36.0, 12.6], [12.6, 64.0]]), (len(cameras), len(alternatives), 1, 1)
    )
    means += np.einsum(
        "vkij,vkj->vki",
        np.linalg.cholesky(covariance),
        rng.standard_normal(means.shape),
    )
    scale = sizes - 1
    normalized = means / scale[:, None, :]
    if ((normalized < 0) | (normalized > 1)).any():
        raise ValueError("Synthetic means left the #935 source-grid contract")
    chol = np.linalg.cholesky(covariance) / scale[:, None, :, None]
    presence = (
        np.array([0.85, 0.6, 0.2])
        if scenario == "low_presence"
        else np.ones(len(cameras))
    )
    if scenario == "prior_only":
        presence[:] = 0
    # Finite logits can export exact saturated 0/1 without violating #935.
    presence_logits = np.empty_like(presence)
    presence_logits[presence == 0] = -1000
    presence_logits[presence == 1] = 1000
    interior = (presence > 0) & (presence < 1)
    presence_logits[interior] = np.log(presence[interior] / (1 - presence[interior]))
    logits = (
        np.array([np.log(0.55), np.log(0.45)])
        if len(alternatives) == 2
        else np.zeros(1)
    )
    distribution = BallGMM2D(
        torch.from_numpy(normalized[:, None]),
        torch.from_numpy(chol[:, None]),
        torch.from_numpy(
            np.broadcast_to(logits, (len(cameras), 1, len(alternatives))).copy()
        ),
        torch.from_numpy(presence_logits[:, None]),
    )
    return frame_observations(distribution, torch.from_numpy(sizes), frame=0)


def run_comparison(
    fixture: Path,
    output: Path,
    *,
    trials: int,
    seed: int,
    particles: int,
    hdr_samples: int,
    voxel: VoxelConfig,
) -> None:
    if trials < 1 or hdr_samples < 100 or output.exists():
        raise ValueError(
            "Need positive trials, >=100 HDR samples and a new output path"
        )
    torch.set_num_threads(1)
    raw = json.loads(fixture.read_text())
    cameras = tuple(
        PinholeCamera(
            c["camera_id"], np.array(c["K"]), np.array(c["R"]), np.array(c["t"])
        )
        for c in raw["cameras"]
    )
    sizes = np.asarray([[c["w"], c["h"]] for c in raw["cameras"]], dtype=np.float64)
    prior = GaussianPrior3D(np.array([0.0, 0.0, 2.0]), np.diag([0.36, 1.44, 0.25]))
    laplace = LaplaceConfig(4096, 100)
    records: list[dict[str, object]] = []
    for scenario_id, scenario in enumerate(
        ("unimodal", "ambiguous", "low_presence", "prior_only")
    ):
        for trial in range(trials):
            data_rng = np.random.default_rng(
                np.random.SeedSequence([seed, scenario_id, trial, 0])
            )
            truth = data_rng.multivariate_normal(prior.mean, prior.covariance)
            obs = synthetic_observations(cameras, sizes, truth, scenario, data_rng)
            for method_id, method in enumerate(("A", "B", "C")):
                sample_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, scenario_id, trial, 1])
                )
                start = time.perf_counter()
                density: Density
                if method == "A":
                    density = triangulate_gmm(
                        obs, cameras, prior=prior, config=laplace
                    ).distribution
                elif method == "B":
                    density = triangulate_volume(
                        obs, cameras, prior=prior, config=voxel
                    )
                else:
                    density = triangulate_samples(
                        obs,
                        cameras,
                        prior=prior,
                        samples=particles,
                        rng=sample_rng,
                        max_nfev=100,
                    )
                elapsed = time.perf_counter() - start
                hdr_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, scenario_id, trial, 2, method_id])
                )
                samples = density.sample(hdr_samples, hdr_rng)
                log_density = density.log_prob(samples)
                log_truth = float(density.log_prob(truth))
                records.append(
                    {
                        "scenario": scenario,
                        "trial": trial,
                        "method": method,
                        "nll_m3": -log_truth,
                        "coverage90": bool(log_truth >= np.quantile(log_density, 0.10)),
                        "coverage95": bool(log_truth >= np.quantile(log_density, 0.05)),
                        "seconds": elapsed,
                    }
                )
        print(f"completed {scenario}: {trials} trials", flush=True)
    summary = []
    for scenario in ("unimodal", "ambiguous", "low_presence", "prior_only"):
        for method in ("A", "B", "C"):
            selected = [
                r
                for r in records
                if r["scenario"] == scenario and r["method"] == method
            ]
            # Arrays keep all trials, including poor estimates; no accepted-only subset.
            nll = np.asarray([r["nll_m3"] for r in selected], dtype=np.float64)
            seconds = np.asarray([r["seconds"] for r in selected], dtype=np.float64)
            summary.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "trials": len(selected),
                    "mean_nll_m3": float(nll.mean()),
                    "nll_standard_error": float(nll.std() / np.sqrt(len(nll))),
                    "coverage90": float(np.mean([r["coverage90"] for r in selected])),
                    "coverage95": float(np.mean([r["coverage95"] for r in selected])),
                    "median_ms": float(np.median(seconds) * 1000),
                    "p95_ms": float(np.quantile(seconds, 0.95) * 1000),
                }
            )
    result = {
        "schema": 1,
        "seed": seed,
        "trials_per_scenario": trials,
        "particles": particles,
        "hdr_samples": hdr_samples,
        "voxel": asdict(voxel),
        "laplace": asdict(laplace),
        "prior": {"mean": prior.mean.tolist(), "covariance": prior.covariance.tolist()},
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "clip_id": raw["clip_id"],
        "cameras": [c.camera_id for c in cameras],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "summary": summary,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in summary:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--seed", type=int, default=936)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--hdr-samples", type=int, default=1024)
    parser.add_argument("--initial-cells", type=int, default=16)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--refine-cells", type=int, default=512)
    args = parser.parse_args()
    if not args.fixture.is_absolute() or not args.output.is_absolute():
        parser.error("--fixture and --output must be absolute")
    roots = RuntimePathRoots(
        project_root=args.output.parent, data_root=args.fixture.parent,
        artifact_root=args.fixture.parent, output_root=args.output.parent,
        checkpoint_root=args.output.parent, cache_root=args.output.parent,
        external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(
        {"fixture": args.fixture, "output": args.output},
        resolver=PathResolver(roots),
    )
    run_comparison(
        paths.declared("fixture").path,
        paths.declared("output").path,
        trials=args.trials,
        seed=args.seed,
        particles=args.particles,
        hdr_samples=args.hdr_samples,
        voxel=VoxelConfig(args.initial_cells, args.levels, args.refine_cells, 4.0),
    )


if __name__ == "__main__":
    main()

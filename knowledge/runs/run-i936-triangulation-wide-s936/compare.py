"""A/B/C plus explicit hybrid on fixed generator inputs; every failure is counted."""
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.tasks.ball_refiner.refiner_3d.comparison import triangulate_samples
from src.utils.geometry.probabilistic_triangulation import CameraGMM, GaussianPrior3D, LaplaceConfig, triangulate_gmm
from src.utils.geometry.probabilistic_triangulation.solver import HybridConfig, triangulate_hybrid
from src.utils.geometry.probabilistic_triangulation.volume import VoxelConfig, triangulate_volume
from src.utils.geometry.triangulation import PinholeCamera
from experimental_ray_volume import RayVolumeConfig
from experimental_hybrid import HybridConfig as RayHybridConfig, triangulate_hybrid as triangulate_ray_hybrid

root = Path(__file__).parent
config = json.loads((root / "cases.json").read_text())
cases = np.load(root / "cases.npz", allow_pickle=False)
d = config["plan"]["degradation"]
prior = GaussianPrior3D(np.array(d["prior_mean_m"], float), np.diag(d["prior_covariance_diagonal_m2"]).astype(float))
voxel = VoxelConfig(16, 7, 512, 4.)
laplace = LaplaceConfig(64, 100)
hybrid = HybridConfig(laplace, voxel)
records = []
out = root / sys.argv[1]
if out.exists():
    raise FileExistsError(out)
methods = sys.argv[2:] or ["A", "B", "C", "H"]
for n, row in enumerate(config["records"]):
    if methods in (["B_hi", "H_hi"], ["R_hi"], ["R_top"]) and not (row["frame"] == 0 or row["cohort"] in ("shared_gap", "run2_failure")):
        continue
    key = row["key"]
    cameras = tuple(PinholeCamera(str(i), cases[f"{key}_K"][i], cases[f"{key}_R"][i], cases[f"{key}_t"][i]) for i in range(3))
    obs = CameraGMM(*(cases[f"{key}_{name}"] for name in ("means","covariance","weights","presence")))
    truth = cases[f"{key}_truth"]
    for method in methods:
        record = dict(**row, method=method)
        started = time.perf_counter()
        try:
            if method == "A0":
                from legacy_solver import triangulate_gmm as legacy_triangulate
                density = legacy_triangulate(obs, cameras, prior=prior, config=laplace).distribution
            elif method == "A":
                result = triangulate_gmm(obs, cameras, prior=prior, config=laplace)
                density = result.distribution
            elif method in ("B", "B_hi"):
                density = triangulate_volume(obs, cameras, prior=prior, config=VoxelConfig(24, 8, 1024, 4.) if method == "B_hi" else voxel)
            elif method == "C":
                density = triangulate_samples(obs, cameras, prior=prior, samples=128, rng=np.random.default_rng([936,n,1]), max_nfev=100)
            elif method in ("R", "R_hi", "R_top"):
                budgets = {"R": (12,48), "R_hi": (24,96), "R_top": (32,128)}
                result = triangulate_ray_hybrid(obs, cameras, prior=prior, config=RayHybridConfig(laplace, RayVolumeConfig(*budgets[method])))
                density = result.distribution
                record["component_methods"] = dict(Counter(result.component_methods))
                record["components"] = len(density.weights)
                record["ray_config"] = budgets[method]
            elif method in ("H", "H_hi"):
                result = triangulate_hybrid(obs, cameras, prior=prior, config=HybridConfig(laplace, VoxelConfig(24, 8, 1024, 4.)) if method == "H_hi" else hybrid)
                density = result.distribution
                record["component_methods"] = dict(Counter(result.component_methods))
                record["components"] = len(density.weights)
            else:
                raise ValueError(method)
            record["seconds"] = time.perf_counter()-started
            samples = density.sample(1024, np.random.default_rng([936,n,2]))
            log_density = density.log_prob(samples)
            log_truth = float(density.log_prob(truth))
            if not np.isfinite(log_truth):
                raise ValueError("nonfinite_truth_density")
            record.update(status="success", nll_m3=-log_truth, coverage90=bool(log_truth>=np.quantile(log_density,.1)), coverage95=bool(log_truth>=np.quantile(log_density,.05)))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            record.update(status="failed", error=str(exc), seconds=time.perf_counter()-started, nll_m3=None, coverage90=False, coverage95=False)
        records.append(record)
    out.write_text(json.dumps(dict(voxel=asdict(VoxelConfig(24, 8, 1024, 4.) if methods == ["B_hi","H_hi"] else voxel),laplace=asdict(laplace),particles=128,hdr_samples=1024,records=records),indent=2,allow_nan=False)+"\n")
    print(n, row["cohort"], [(r["method"],r["status"],round(r["seconds"],3)) for r in records[-len(methods):]], flush=True)

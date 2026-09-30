"""Deterministic per-rally CPU generation; failures remain explicit artifacts."""

from __future__ import annotations

import json
import multiprocessing
import platform
import resource
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import (
    GenerationPlan,
    sha256,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.observations import (
    load_cameras,
    make_distribution,
    perturb_cameras,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.simulation import accepted_rally
from src.tasks.ball_refiner.refiner_3d.synthetic.timebase import (
    event_masks,
    resample,
    retained_events,
)
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.geometry.probabilistic_triangulation import (
    GaussianPrior3D,
    LaplaceConfig,
)
from src.utils.geometry.probabilistic_triangulation.convergence import (
    convergence_config,
    triangulate_converged,
)
from src.utils.geometry.probabilistic_triangulation.solver import (
    COMPONENT_METHODS,
)
from src.utils.schema.court import X_MAX, Y_MAX


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def generate_rally(plan: GenerationPlan, split_index: int, index: int, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    seed = int(np.random.SeedSequence([plan.values["seed"], split_index, index]).generate_state(1)[0])
    source = plan.values["geometry"]["sources"][split_index]
    split = source["split"]
    rally_id = f"{split}-{index:05d}"
    result, physics, rng, proposals = accepted_rally(plan, seed=seed, index=index)
    sample = plan.values["sampling"]
    timestamps, positions = resample(
        result.trajectory_sim.numpy(), native_hz=result.sim_fps,
        numerator=sample["fps_numerator"], denominator=sample["fps_denominator"],
        max_frames=sample["max_frames_per_rally"],
    )
    events = retained_events(result, timestamps)
    labels, event_region, free_flight = event_masks(len(timestamps), events, radius=sample["physics_event_mask_radius_frames"])
    # Simulator has no fence timestamps. Exclude its near-fence region instead.
    near_fence = (np.abs(positions[:, 0]) > X_MAX - 0.5) | (np.abs(positions[:, 1]) > Y_MAX - 0.5)
    for frame in np.flatnonzero(near_fence):
        free_flight[max(0, frame - 5):frame + 6] = False
    simulation_seconds = time.perf_counter() - started
    base_cameras, sizes = load_cameras(plan.camera_paths[split_index], source["camera_keys"])
    cameras = perturb_cameras(base_cameras, plan.values["geometry"]["perturbation_per_scene"], rng)
    degradation = plan.values["degradation"]
    distribution, masks, observation_metadata = make_distribution(positions, cameras, sizes, degradation, rng, rally_index=index, calibration=plan.calibration)
    prior = GaussianPrior3D(np.asarray(degradation["prior_mean_m"], dtype=float), np.diag(degradation["prior_covariance_diagonal_m2"]).astype(float))
    laplace = LaplaceConfig(degradation["max_components"], degradation["max_nfev"])
    convergence = convergence_config(degradation["boundary_convergence"])
    tri_started = time.perf_counter()
    posteriors = []
    checks = []
    for frame in range(len(timestamps)):
        try:
            observations = frame_observations(distribution, torch.from_numpy(sizes), frame=frame)
            checked = triangulate_converged(observations, cameras, prior=prior, laplace=laplace, config=convergence)
            posterior = checked.posterior
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"{rally_id} frame={frame} seed={seed}: {exc}") from exc
        if len(posterior.distribution.weights) != degradation["max_components"]:
            raise ValueError("The recipe must preserve every calibrated component product")
        posteriors.append(posterior)
        checks.append(checked)
        if frame % 16 == 0 or frame == len(timestamps) - 1:
            write_json(output / f"{rally_id}.progress.json", {"rally_id": rally_id, "completed_frames": frame + 1, "frames": len(timestamps), "nonconverged_frames": sum(not item.converged for item in checks), "elapsed_seconds": time.perf_counter() - started})
    triangulation_seconds = time.perf_counter() - tri_started
    arrays = {
        "timestamps_seconds": timestamps,
        "native_positions_3d_m": result.trajectory_sim.numpy().astype(np.float32),
        "positions_3d_m": positions,
        "event_labels": labels,
        "event_region_mask": event_region,
        "free_flight_mask": free_flight,
        **masks,
        "source_size_wh": sizes.astype(np.int64),
        "gmm2d_means_uv": distribution.means.numpy(),
        "gmm2d_scale_tril_uv": distribution.scale_tril.numpy(),
        "gmm2d_mixture_logits": distribution.mixture_logits.numpy(),
        "gmm2d_presence_logits": distribution.presence_logits.numpy(),
        "gmm3d_means_m": np.stack([p.distribution.means for p in posteriors]).astype(np.float32),
        "gmm3d_covariance_m2": np.stack([p.distribution.covariance for p in posteriors]).astype(np.float32),
        "gmm3d_weights": np.stack([p.distribution.weights for p in posteriors]).astype(np.float32),
        "gmm3d_camera_subsets": np.stack([p.camera_subsets for p in posteriors]),
        "gmm3d_method_codes": np.asarray([[COMPONENT_METHODS.index(method) for method in p.component_methods] for p in posteriors], dtype=np.uint8),
        "integration_converged": np.asarray([item.converged for item in checks], dtype=bool),
        "integration_rounds": np.asarray([item.rounds for item in checks], dtype=np.uint8),
        "integration_component_converged": np.stack([item.component_converged for item in checks]),
        "integration_component_changes": np.stack([item.component_changes for item in checks]),
        "integration_nll_delta_nat": np.asarray([item.nll_delta_nat for item in checks]),
        "integration_component_embedded_error": np.asarray([[d.get("embedded_relative_error", 0.) for d in p.component_integration_diagnostics] for p in posteriors]),
        "integration_component_metric_codes": np.asarray([[0 if not d else (1 if d["metric_is_local_hessian"] else 2) for d in p.component_integration_diagnostics] for p in posteriors], dtype=np.uint8),
        "prior_only_probability": np.asarray([p.prior_only_probability for p in posteriors], dtype=np.float32),
    }
    for label, selected in (("base", base_cameras), ("true", cameras), ("estimated", cameras)):
        arrays[f"camera_{label}_K"] = np.stack([c.intrinsic for c in selected])
        arrays[f"camera_{label}_R"] = np.stack([c.rotation for c in selected])
        arrays[f"camera_{label}_t"] = np.stack([c.translation for c in selected])
    # Verify the exported precision too, not only the float64 solver state.
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite exported array in {rally_id}")
    np.linalg.cholesky(arrays["gmm3d_covariance_m2"])
    destination = output / f"{rally_id}.npz"
    np.savez_compressed(destination, **arrays)
    metadata = {
        "rally_id": rally_id, "split": split, "seed": seed,
        "physics_proposals": proposals,
        "geometry_clip": source["clip_id"], "geometry_sha256": source["sha256"],
        "frames": len(timestamps), "native_frames": len(result.trajectory_sim),
        "fps_numerator": sample["fps_numerator"], "fps_denominator": sample["fps_denominator"],
        "native_hz": result.sim_fps, "events": [asdict(event) for event in events],
        "end_reason": result.end_reason.value, "physics": asdict(physics),
        "simulation_seconds": simulation_seconds, "triangulation_seconds": triangulation_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "npz_bytes": destination.stat().st_size, "npz_sha256": sha256(destination),
        "components_per_frame": degradation["max_components"],
        "integration": {"converged_frames": sum(item.converged for item in checks), "nonconverged_frames": sum(not item.converged for item in checks), "nonconverged_rate": sum(not item.converged for item in checks) / len(checks), "rule": degradation["boundary_convergence"], "history": [item.history for item in checks]},
        "component_method_labels": COMPONENT_METHODS,
        "component_method_counts": dict(Counter(method for p in posteriors for method in p.component_methods)),
        "float32_zero_weight_components": int((arrays["gmm3d_weights"] == 0).sum()),
        "all_camera_occluded_frames": int(masks["occlusion_mask"].all(0).sum()),
        "out_of_frame_camera_frames": int(masks["out_of_frame_mask"].sum()),
        "events_hit": int(labels[:, 0].sum()), "events_bounce": int(labels[:, 1].sum()),
        "event_region_frames": int(event_region.sum()), "free_flight_frames": int(free_flight.sum()),
        **observation_metadata,
    }
    write_json(output / f"{rally_id}.json", metadata)
    return metadata


def _job(arguments: tuple[GenerationPlan, int, int, Path]) -> dict[str, Any]:
    plan, split, index, output = arguments
    try:
        return generate_rally(plan, split, index, output)
    except Exception as exc:
        write_json(output / f"failed-{split}-{index:05d}.json", {"error": str(exc), "type": type(exc).__name__, "split_index": split, "rally_index": index})
        raise


def generate_dataset(plan: GenerationPlan, output: Path, *, mode: str) -> dict[str, Any]:
    if mode not in ("smoke", "dev", "pilot"):
        raise ValueError("Mode must be smoke, dev or pilot")
    if output.exists():
        raise FileExistsError(output)
    plan.verify_inputs()
    counts = {split: plan.values["counts"]["smoke_rallies_per_split"] for split in ("train", "val", "test")} if mode == "smoke" else plan.values["counts"][f"{mode}_rallies"]
    workers = plan.values["simulation"]["workers"]
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": "ball_refiner_3d.synthetic.v2", "status": "running", "mode": mode,
        "plan": plan.values, "resolved_simulator": {"physics": plan.physics, "rally": plan.rally, "targeted": plan.targeted},
        "input_hashes": plan.input_hashes, "counts": counts, "workers": workers,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "rallies": [],
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    jobs = [(plan, split_index, index, output) for split_index, split in enumerate(("train", "val", "test")) for index in range(counts[split])]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_job, job): job[1:3] for job in jobs}
            failures = []
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:
                    split_index, index = futures[future]
                    failures.append({"split_index": split_index, "rally_index": index, "error": str(exc)})
                else:
                    manifest["rallies"].append(record)
                    print(json.dumps({"completed": record["rally_id"], "frames": record["frames"], "seconds": record["elapsed_seconds"]}), flush=True)
                manifest["failures"] = failures
                write_json(output / "manifest.json", manifest)
            if failures:
                raise RuntimeError(f"{len(failures)} of {len(jobs)} rallies failed; all outcomes recorded")
        manifest["rallies"].sort(key=lambda record: record["rally_id"])
        plan.verify_inputs()
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise
    records = manifest["rallies"]
    manifest.update(
        status="complete", elapsed_seconds=time.perf_counter() - started,
        total_frames=sum(r["frames"] for r in records),
        npz_bytes=sum(r["npz_bytes"] for r in records),
        nonconverged_frames=sum(r["integration"]["nonconverged_frames"] for r in records),
        sum_rally_seconds=sum(r["elapsed_seconds"] for r in records),
    )
    pilot_counts = plan.values["counts"]["pilot_rallies"]
    projection = sum(
        np.mean([r["elapsed_seconds"] for r in records if r["split"] == split]) * pilot_counts[split]
        for split in pilot_counts
    )
    manifest["pilot_projection"] = {
        "rallies": sum(pilot_counts.values()),
        "serial_seconds": float(projection), "ideal_4_worker_seconds": float(projection / 4),
        "bytes": int(sum(np.mean([r["npz_bytes"] for r in records if r["split"] == split]) * pilot_counts[split] for split in pilot_counts)),
        "note": "linear extrapolation by split; parallel overhead and seed-dependent rejection/solver failures not predicted",
    }
    write_json(output / "manifest.json", manifest)
    return manifest

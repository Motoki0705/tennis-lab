"""Verified, variable-length rally storage shared by smoke and training."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.synthetic.calibration import load_calibration
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.timebase import resample
from src.utils.geometry.probabilistic_triangulation.solver import COMPONENT_METHODS
from src.utils.paths import PROJECT_ROOT


class SyntheticDataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["schema"] not in ("ball_refiner_3d.synthetic.v1", "ball_refiner_3d.synthetic.v2") or self.manifest["status"] != "complete":
            raise ValueError("Require a complete synthetic.v1/v2 dataset")
        self.records = self.manifest["rallies"]
        ids = [record["rally_id"] for record in self.records]
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r"(train|val|test)-[0-9]{5}", name) for name in ids):
            raise ValueError("Invalid or duplicate rally identity")
        for split, count in self.manifest["counts"].items():
            selected = [r for r in self.records if r["split"] == split]
            if len(selected) != count or any(not r["rally_id"].startswith(split + "-") for r in selected):
                raise ValueError("Rally split/count mismatch")

    def load(self, record: dict[str, Any]) -> dict[str, NDArray[Any]]:
        path = self.directory / (record["rally_id"] + ".npz")
        if path.stat().st_size != record["npz_bytes"] or sha256(path) != record["npz_sha256"]:
            raise ValueError(f"Corrupt rally: {path.name}")
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        validate_rally(arrays, record, self.manifest["plan"])
        return arrays


def validate_rally(arrays: dict[str, NDArray[Any]], record: dict[str, Any], plan: dict[str, Any]) -> None:
    t = record["frames"]
    if t < 2 or any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Empty or nonfinite rally")
    sample = plan["sampling"]
    times, truth = resample(
        arrays["native_positions_3d_m"], native_hz=record["native_hz"],
        numerator=sample["fps_numerator"], denominator=sample["fps_denominator"],
        max_frames=sample["max_frames_per_rally"],
    )
    if len(times) != t or not np.array_equal(arrays["timestamps_seconds"], times) or not np.array_equal(arrays["positions_3d_m"], truth):
        raise ValueError("Stored trajectory violates native resampling")
    distribution = BallGMM2D(
        torch.from_numpy(arrays["gmm2d_means_uv"]),
        torch.from_numpy(arrays["gmm2d_scale_tril_uv"]),
        torch.from_numpy(arrays["gmm2d_mixture_logits"]),
        torch.from_numpy(arrays["gmm2d_presence_logits"]),
    )
    k = plan["degradation"]["components_per_camera"]
    components = (k + 1) ** 3
    if distribution.means.shape != (3, t, k, 2):
        raise ValueError("Wrong camera/time/component axes")
    covariance, weights = arrays["gmm3d_covariance_m2"], arrays["gmm3d_weights"]
    if arrays["gmm3d_means_m"].shape != (t, components, 3) or covariance.shape != (t, components, 3, 3) or weights.shape != (t, components):
        raise ValueError("Missing full component posterior")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-7) or (weights < 0).any() or not np.allclose(weights.sum(-1), 1, atol=1e-6):
        raise ValueError("Invalid posterior covariance/weights")
    np.linalg.cholesky(covariance)
    subsets = arrays["gmm3d_camera_subsets"]
    if subsets.shape != (t, components, 3) or subsets.dtype != np.bool_ or not np.array_equal(subsets, np.broadcast_to(subsets[0], subsets.shape)):
        raise ValueError("Camera subset topology changed within a rally")
    codes = arrays["gmm3d_method_codes"]
    if codes.shape != (t, components) or codes.dtype != np.uint8 or (codes >= len(COMPONENT_METHODS)).any() or tuple(record["component_method_labels"]) != COMPONENT_METHODS:
        raise ValueError("Invalid component integration diagnostics")
    counts = dict(Counter(COMPONENT_METHODS[int(code)] for code in codes.ravel()))
    if counts != record["component_method_counts"]:
        raise ValueError("Component integration counts mismatch")
    points = arrays["gmm3d_means_m"].astype(np.float64)
    for camera in range(3):
        depths = points @ arrays["camera_estimated_R"][camera, 2] + arrays["camera_estimated_t"][camera, 2]
        if (depths[subsets[:, :, camera]] <= 0).any():
            raise ValueError("Posterior component mean behind an active camera")
    presence = distribution.presence_probability.numpy().T.astype(np.float64)
    for mask in np.unique(subsets[0], axis=0):
        selected = (subsets[0] == mask).all(-1)
        if selected.sum() != k ** int(mask.sum()):
            raise ValueError("Camera subset is missing component combinations")
        mass = np.prod(np.where(mask, presence, 1 - presence), axis=-1)
        if not np.allclose(weights[:, selected].sum(-1), mass, atol=1e-6):
            raise ValueError("Presence marginalization mass changed")
    if not np.allclose(arrays["prior_only_probability"], np.prod(1 - presence, axis=-1), atol=1e-6):
        raise ValueError("Prior-only probability mismatch")
    expected_labels = np.zeros((t, 2), dtype=bool)
    expected_region = np.zeros(t, dtype=bool)
    for event in record["events"]:
        native_frame, frame = event["native_frame"], event["frame"]
        seconds = native_frame / record["native_hz"]
        if event["seconds"] != seconds or frame != int(np.abs(times - seconds).argmin()):
            raise ValueError("Event seconds/frame mismatch")
        lo, hi = max(0, frame - 5), min(t, frame + 6)
        if arrays["free_flight_mask"][lo:hi].any():
            raise ValueError("Physics mask crosses an event")
        if event["kind"] in ("hit", "bounce"):
            expected_labels[frame, int(event["kind"] == "bounce")] = True
            expected_region[lo:hi] = True
    if not np.array_equal(arrays["event_labels"], expected_labels) or not np.array_equal(arrays["event_region_mask"], expected_region):
        raise ValueError("Event labels or +/-5 mask mismatch")
    occlusion = np.zeros((3, t), dtype=bool)
    for gap in record["gap_intervals"]:
        occlusion[gap["cameras"], gap["start"]:gap["stop"]] = True
    if not np.array_equal(occlusion, arrays["occlusion_mask"]):
        raise ValueError("Gap interval/mask mismatch")
    if plan["schema_version"] == 1:
        expected_logits = np.where(arrays["out_of_frame_mask"], -plan["degradation"]["presence_logit_magnitude"], plan["degradation"]["presence_logit_magnitude"])
        if not np.array_equal(arrays["gmm2d_presence_logits"], expected_logits):
            raise ValueError("Occlusion must not change amodal presence")
    elif plan["schema_version"] == 2:
        _validate_v2(arrays, record, plan, components)
    else:
        raise ValueError("Unknown synthetic plan schema")
    for field in ("K", "R", "t"):
        if not np.array_equal(arrays[f"camera_true_{field}"], arrays[f"camera_estimated_{field}"]):
            raise ValueError("v1 clean geometry requires matched true/estimated cameras")


def _validate_v2(arrays: dict[str, NDArray[Any]], record: dict[str, Any], plan: dict[str, Any], components: int) -> None:
    t = record["frames"]
    config = plan["degradation"]["boundary_convergence"]
    changes, flags = arrays["integration_component_changes"], arrays["integration_component_converged"]
    nll, converged, rounds = arrays["integration_nll_delta_nat"], arrays["integration_converged"], arrays["integration_rounds"]
    if changes.shape != (t, components, 3) or flags.shape != (t, components) or flags.dtype != np.bool_ or (changes < 0).any():
        raise ValueError("Invalid per-component convergence diagnostics")
    if any(value.shape != (t,) for value in (nll, converged, rounds)) or converged.dtype != np.bool_ or (nll < 0).any():
        raise ValueError("Invalid per-frame convergence diagnostics")
    tolerances = np.asarray([config["log_evidence_tolerance_nat"], config["mean_tolerance"], config["covariance_relative_tolerance"]])
    if not np.array_equal(flags, (changes <= tolerances).all(-1)) or not np.array_equal(converged, flags.all(-1) & (nll <= config["nll_tolerance_nat"])):
        raise ValueError("Convergence flags disagree with achieved tolerance")
    cap = len(config["initial_cells"])
    if not np.issubdtype(rounds.dtype, np.integer) or (rounds < 2).any() or (rounds > cap).any() or (rounds[~converged] != cap).any():
        raise ValueError("Nonconverged frames must exhaust the explicit cap")
    summary = record["integration"]
    if summary["rule"] != config or summary["converged_frames"] != int(converged.sum()) or summary["nonconverged_frames"] != int((~converged).sum()):
        raise ValueError("Convergence summary mismatch")
    settings = plan["degradation"]["calibration"]
    bank = load_calibration(PROJECT_ROOT / settings["bank"], settings["bank_sha256"]).arrays
    rows = arrays["calibration_rows"]
    if rows.shape != (3, t) or not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any() or (rows >= len(bank["error_uv"])).any():
        raise ValueError("Invalid calibration row provenance")
    if not np.array_equal(bank["camera_index"][rows], np.broadcast_to(np.arange(3)[:, None], (3, t))) or not np.array_equal(bank["condition_index"][rows], arrays["occlusion_mask"].astype(int)):
        raise ValueError("Calibration camera/condition mismatch")
    for field, bank_field in (("gmm2d_scale_tril_uv", "scale_tril_uv"), ("gmm2d_mixture_logits", "mixture_logits")):
        if not np.array_equal(arrays[field], bank[bank_field][rows]):
            raise ValueError("Calibrated GMM components were modified")
    expected_presence = np.where(arrays["out_of_frame_mask"], settings["out_of_frame_presence_logit"], bank["presence_logits"][rows])
    if not np.array_equal(arrays["gmm2d_presence_logits"], expected_presence):
        raise ValueError("Calibrated amodal presence mismatch")

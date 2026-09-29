"""Verified, variable-length rally storage shared by smoke and training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.timebase import resample


class SyntheticDataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["schema"] != "ball_refiner_3d.synthetic.v1" or self.manifest["status"] != "complete":
            raise ValueError("Require a complete synthetic.v1 dataset")
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
    if distribution.means.shape != (3, t, 3, 2):
        raise ValueError("Wrong camera/time/component axes")
    covariance, weights = arrays["gmm3d_covariance_m2"], arrays["gmm3d_weights"]
    if arrays["gmm3d_means_m"].shape != (t, 64, 3) or covariance.shape != (t, 64, 3, 3) or weights.shape != (t, 64):
        raise ValueError("Missing full 64-component posterior")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-7) or (weights < 0).any() or not np.allclose(weights.sum(-1), 1, atol=1e-6):
        raise ValueError("Invalid posterior covariance/weights")
    np.linalg.cholesky(covariance)
    subsets = arrays["gmm3d_camera_subsets"]
    if subsets.shape != (t, 64, 3) or subsets.dtype != np.bool_ or not np.array_equal(subsets, np.broadcast_to(subsets[0], subsets.shape)):
        raise ValueError("Camera subset topology changed within a rally")
    presence = distribution.presence_probability.numpy().T.astype(np.float64)
    for mask in np.unique(subsets[0], axis=0):
        selected = (subsets[0] == mask).all(-1)
        if selected.sum() != 3 ** int(mask.sum()):
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
    expected_logits = np.where(arrays["out_of_frame_mask"], -plan["degradation"]["presence_logit_magnitude"], plan["degradation"]["presence_logit_magnitude"])
    if not np.array_equal(arrays["gmm2d_presence_logits"], expected_logits):
        raise ValueError("Occlusion must not change amodal presence")
    for field in ("K", "R", "t"):
        if not np.array_equal(arrays[f"camera_true_{field}"], arrays[f"camera_estimated_{field}"]):
            raise ValueError("v1 clean geometry requires matched true/estimated cameras")

"""Validate and expose manifest-registered synthetic v1/v2 rallies for review."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SCHEMAS = {"ball_refiner_3d.synthetic.v1": 1, "ball_refiner_3d.synthetic.v2": 2}
SPLITS = {"train", "val", "test"}
Arrays = dict[str, NDArray[Any]]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def softmax(values: NDArray[Any]) -> NDArray[np.float64]:
    shifted = np.asarray(values, dtype=np.float64) - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def presence(values: NDArray[Any]) -> NDArray[np.float64]:
    return np.exp(-np.logaddexp(0.0, -values.astype(np.float64)))


def project(
    points: NDArray[Any], k: NDArray[Any], r: NDArray[Any], t: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Project with stored calibration; undefined or behind-camera points stay invalid."""
    camera = np.asarray(points, dtype=np.float64) @ r.T + t
    homogeneous = camera @ k.T
    valid = (camera[..., 2] > 0) & (np.abs(homogeneous[..., 2]) > 1e-12)
    pixels = np.zeros((*camera.shape[:-1], 2), dtype=np.float64)
    np.divide(
        homogeneous[..., :2],
        homogeneous[..., 2, None],
        out=pixels,
        where=valid[..., None],
    )
    return pixels, valid


def validate(arrays: Arrays, record: dict[str, Any], plan: dict[str, Any]) -> None:
    """Check saved review semantics; do not regenerate a trajectory or triangulation."""
    frames = record["frames"]
    components = record["components_per_frame"]
    views = len(plan["geometry"]["sources"][0]["camera_keys"])
    modes = plan["degradation"]["components_per_camera"]
    shapes = {
        "timestamps_seconds": (frames,),
        "positions_3d_m": (frames, 3),
        "source_size_wh": (views, 2),
        "occlusion_mask": (views, frames),
        "out_of_frame_mask": (views, frames),
        "event_labels": (frames, 2),
        "event_region_mask": (frames,),
        "free_flight_mask": (frames,),
        "gmm2d_means_uv": (views, frames, modes, 2),
        "gmm2d_scale_tril_uv": (views, frames, modes, 2, 2),
        "gmm2d_mixture_logits": (views, frames, modes),
        "gmm2d_presence_logits": (views, frames),
        "gmm3d_means_m": (frames, components, 3),
        "gmm3d_covariance_m2": (frames, components, 3, 3),
        "gmm3d_weights": (frames, components),
        "gmm3d_camera_subsets": (frames, components, views),
        "gmm3d_method_codes": (frames, components),
        "prior_only_probability": (frames,),
    }
    for prefix in ("true", "estimated"):
        for field, shape in (
            ("K", (views, 3, 3)),
            ("R", (views, 3, 3)),
            ("t", (views, 3)),
        ):
            shapes[f"camera_{prefix}_{field}"] = shape
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"Missing/wrong axes: {name}; expected {shape}")
    if frames < 2 or components != (modes + 1) ** views:
        raise ValueError("Invalid frame/component count")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite saved array")
    for name in (
        "occlusion_mask",
        "out_of_frame_mask",
        "event_labels",
        "event_region_mask",
        "free_flight_mask",
        "gmm3d_camera_subsets",
    ):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"Expected boolean mask: {name}")
    sizes = arrays["source_size_wh"]
    if not np.issubdtype(sizes.dtype, np.integer) or (sizes <= 1).any():
        raise ValueError("Invalid source image size")
    sampling = plan["sampling"]
    if (record["fps_numerator"], record["fps_denominator"]) != (
        sampling["fps_numerator"],
        sampling["fps_denominator"],
    ):
        raise ValueError("Record/plan FPS mismatch")
    expected = (
        np.arange(frames, dtype=np.float64)
        * sampling["fps_denominator"]
        / sampling["fps_numerator"]
    )
    if not np.allclose(arrays["timestamps_seconds"], expected, atol=1e-12, rtol=0):
        raise ValueError("Timestamp grid does not match rational FPS")
    for view in range(views):
        pixels, front = project(
            arrays["positions_3d_m"],
            arrays["camera_true_K"][view],
            arrays["camera_true_R"][view],
            arrays["camera_true_t"][view],
        )
        outside = ~front | (pixels < 0).any(-1) | (pixels > sizes[view] - 1).any(-1)
        if not np.array_equal(outside, arrays["out_of_frame_mask"][view]):
            raise ValueError("Out-of-frame mask disagrees with saved true camera")
    expected_events = np.zeros((frames, 2), dtype=bool)
    expected_region = np.zeros(frames, dtype=bool)
    radius = sampling["physics_event_mask_radius_frames"]
    for event in record["events"]:
        nearest = int(np.abs(expected - event["seconds"]).argmin())
        if event["frame"] != nearest:
            raise ValueError("Event seconds/frame mismatch")
        if event["kind"] in {"hit", "bounce"}:
            expected_events[nearest, int(event["kind"] == "bounce")] = True
            expected_region[
                max(0, nearest - radius) : min(frames, nearest + radius + 1)
            ] = True
    if not np.array_equal(
        expected_events, arrays["event_labels"]
    ) or not np.array_equal(expected_region, arrays["event_region_mask"]):
        raise ValueError("Event labels/region mismatch")
    means, scale = arrays["gmm2d_means_uv"], arrays["gmm2d_scale_tril_uv"]
    if (
        ((means < 0) | (means > 1)).any()
        or (np.diagonal(scale, axis1=-2, axis2=-1) <= 0).any()
        or (np.triu(scale, 1) != 0).any()
    ):
        raise ValueError("Invalid normalized 2D GMM")
    covariance, weights = arrays["gmm3d_covariance_m2"], arrays["gmm3d_weights"]
    if (weights < 0).any() or not np.allclose(weights.sum(-1), 1, atol=1e-6):
        raise ValueError("Invalid full-mixture weights")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-7):
        raise ValueError("Asymmetric covariance")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Non-positive-definite covariance") from exc
    subsets = arrays["gmm3d_camera_subsets"]
    if not np.array_equal(subsets, np.broadcast_to(subsets[0], subsets.shape)):
        raise ValueError("Camera subset topology changed")
    probabilities = presence(arrays["gmm2d_presence_logits"]).T
    for mask in np.unique(subsets[0], axis=0):
        selected = (subsets[0] == mask).all(-1)
        expected_mass = np.prod(
            np.where(mask, probabilities, 1 - probabilities), axis=-1
        )
        if selected.sum() != modes ** int(mask.sum()) or not np.allclose(
            weights[:, selected].sum(-1), expected_mass, atol=1e-6
        ):
            raise ValueError("Camera subset/presence mass mismatch")
    prior = ~subsets.any(-1)
    if (
        (arrays["prior_only_probability"] < 0) | (arrays["prior_only_probability"] > 1)
    ).any():
        raise ValueError("Invalid prior-only probability")
    if not (prior.sum(-1) == 1).all() or not np.allclose(
        (weights * prior).sum(-1), arrays["prior_only_probability"], atol=1e-6
    ):
        raise ValueError("Prior-only component/mass mismatch")
    codes = arrays["gmm3d_method_codes"]
    if (
        not np.issubdtype(codes.dtype, np.integer)
        or (codes < 0).any()
        or (codes >= len(record["component_method_labels"])).any()
    ):
        raise ValueError("Invalid method code")
    convergence = plan["degradation"].get("boundary_convergence")
    if convergence is not None:
        for name in ("integration_converged", "integration_rounds"):
            if name not in arrays or arrays[name].shape != (frames,):
                raise ValueError(f"Missing convergence diagnostic: {name}")
        if arrays["integration_converged"].dtype != np.bool_:
            raise ValueError("Invalid convergence mask")
        assessed = arrays.get("integration_convergence_assessed")
        if assessed is not None and (
            assessed.shape != (frames,) or assessed.dtype != np.bool_
        ):
            raise ValueError("Invalid convergence-assessed mask")
        if convergence.get("method") == "fixed_hybrid" and (
            assessed is None or assessed.any() or arrays["integration_converged"].any()
        ):
            raise ValueError("Fixed budget must declare convergence unassessed")


class DatasetReview:
    """A small read-only catalog and bounded rally cache; only registered NPZs load."""

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise ValueError(f"Dataset root is not a directory: {root}")
        self.root = root.resolve()
        self.datasets: dict[str, dict[str, Any]] = {}
        self.cache: OrderedDict[tuple[str, str], Arrays] = OrderedDict()
        self.lock = threading.RLock()
        for path in sorted(self.root.glob("*/manifest.json")):
            manifest = json.loads(path.read_text())
            if manifest.get("schema") not in SCHEMAS:
                continue
            if manifest["plan"]["schema_version"] != SCHEMAS[manifest["schema"]]:
                raise ValueError(f"Dataset/plan schema mismatch: {path.parent.name}")
            records = manifest["rallies"]
            ids = [row["rally_id"] for row in records]
            if len(set(ids)) != len(ids) or any(
                not re.fullmatch(r"(train|val|test)-[0-9]{5}", name) for name in ids
            ):
                raise ValueError(f"Invalid/duplicate rally id: {path.parent.name}")
            if any(
                row["split"] not in SPLITS
                or not row["rally_id"].startswith(row["split"] + "-")
                for row in records
            ):
                raise ValueError(f"Invalid split: {path.parent.name}")
            counts = {
                split: sum(row["split"] == split for row in records)
                for split in sorted(SPLITS)
            }
            if manifest["status"] == "complete" and counts != manifest["counts"]:
                raise ValueError(f"Complete dataset count mismatch: {path.parent.name}")
            self.datasets[path.parent.name] = {
                "path": path,
                "hash": sha256(path),
                "manifest": manifest,
                "records": {row["rally_id"]: row for row in records},
                "counts": counts,
            }
        if not self.datasets:
            raise ValueError("No saved synthetic v1/v2 manifests found")

    def _dataset(self, dataset: str) -> dict[str, Any]:
        if dataset not in self.datasets:
            raise ValueError("Unknown dataset")
        item = self.datasets[dataset]
        if sha256(item["path"]) != item["hash"]:
            raise ValueError("Manifest changed; restart review with the new manifest")
        return item

    def catalog(self) -> list[dict[str, Any]]:
        rows = []
        for name in self.datasets:
            item = self._dataset(name)
            manifest = item["manifest"]
            records = item["records"]
            folder = item["path"].parent
            missing = [key for key in records if not (folder / f"{key}.npz").is_file()]
            orphan = [p.name for p in folder.glob("*.npz") if p.stem not in records]
            rows.append(
                {
                    "id": name,
                    "schema": manifest["schema"],
                    "status": manifest["status"],
                    "mode": manifest["mode"],
                    "counts": item["counts"],
                    "planned_counts": manifest["counts"],
                    "frames": sum(r["frames"] for r in records.values()),
                    "missing": missing,
                    "orphan_npz": sorted(orphan),
                    "available": bool(records) and manifest["status"] != "failed",
                    "degradation": manifest["plan"]["degradation"]["status"],
                    "integration": manifest["plan"]["degradation"].get(
                        "boundary_convergence"
                    ),
                    "manifest_sha256": item["hash"],
                    "manifest_path": str(item["path"]),
                    "source": "3D教師は合成。cameraはMeiji校正、2D劣化は保存モデル分布または仮定。RGB/実測3D教師なし。main未統合の実験artifact。",
                }
            )
        return rows

    def rallies(self, dataset: str, split: str) -> list[dict[str, Any]]:
        if split not in SPLITS:
            raise ValueError("Unknown split")
        item = self._dataset(dataset)
        return [
            {
                "id": r["rally_id"],
                "frames": r["frames"],
                "seed": r["seed"],
                "geometry_clip": r["geometry_clip"],
                "gap_frames": r["all_camera_occluded_frames"],
            }
            for r in item["records"].values()
            if r["split"] == split
        ]

    def load(
        self, dataset: str, rally: str
    ) -> tuple[Arrays, dict[str, Any], dict[str, Any]]:
        with self.lock:
            return self._load(dataset, rally)

    def _load(
        self, dataset: str, rally: str
    ) -> tuple[Arrays, dict[str, Any], dict[str, Any]]:
        item = self._dataset(dataset)
        if item["manifest"]["status"] == "failed" or rally not in item["records"]:
            raise ValueError("Rally is not registered for review")
        record = item["records"][rally]
        path = item["path"].parent / f"{rally}.npz"
        key = (dataset, rally)
        if (
            path.stat().st_size != record["npz_bytes"]
            or sha256(path) != record["npz_sha256"]
        ):
            self.cache.pop(key, None)
            raise ValueError("NPZ checksum/size mismatch")
        if key not in self.cache:
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in stored.files}
            validate(arrays, record, item["manifest"]["plan"])
            self.cache[key] = arrays
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key], record, item

    def sequence(self, dataset: str, rally: str) -> dict[str, Any]:
        arrays, record, item = self.load(dataset, rally)
        mean = np.einsum("tk,tkd->td", arrays["gmm3d_weights"], arrays["gmm3d_means_m"])
        return {
            "dataset": dataset,
            "rally": rally,
            "record": {
                k: record[k]
                for k in (
                    "split",
                    "seed",
                    "frames",
                    "geometry_clip",
                    "npz_sha256",
                    "events",
                )
            },
            "manifest_sha256": item["hash"],
            "manifest_path": str(item["path"]),
            "timestamps_seconds": arrays["timestamps_seconds"].tolist(),
            "truth_m": arrays["positions_3d_m"].tolist(),
            "mixture_mean_m": mean.tolist(),
            "occlusion": arrays["occlusion_mask"].tolist(),
            "out_of_frame": arrays["out_of_frame_mask"].tolist(),
            "presence": presence(arrays["gmm2d_presence_logits"]).tolist(),
            "prior_only_probability": arrays["prior_only_probability"].tolist(),
            "event_region": arrays["event_region_mask"].tolist(),
            "free_flight": arrays["free_flight_mask"].tolist(),
        }

    def frame(self, dataset: str, rally: str, index: int) -> dict[str, Any]:
        arrays, record, _ = self.load(dataset, rally)
        if not 0 <= index < record["frames"]:
            raise ValueError("Frame outside saved sequence")
        means, cov, weights = (
            arrays[name][index]
            for name in ("gmm3d_means_m", "gmm3d_covariance_m2", "gmm3d_weights")
        )
        subsets = arrays["gmm3d_camera_subsets"][index]
        truth = arrays["positions_3d_m"][index]
        probability = presence(arrays["gmm2d_presence_logits"][:, index])
        cameras = []
        # Stored true cameras project synthetic GT; estimated cameras supply the input context.
        court = np.array(
            [
                [-5.485, -11.885, 0],
                [5.485, -11.885, 0],
                [5.485, 11.885, 0],
                [-5.485, 11.885, 0],
                [-5.485, -11.885, 0],
            ],
            dtype=np.float64,
        )
        for view, size in enumerate(arrays["source_size_wh"]):
            projected, valid = project(
                truth[None],
                arrays["camera_true_K"][view],
                arrays["camera_true_R"][view],
                arrays["camera_true_t"][view],
            )
            court_px, court_valid = project(
                court,
                arrays["camera_true_K"][view],
                arrays["camera_true_R"][view],
                arrays["camera_true_t"][view],
            )
            scale_px = (
                arrays["gmm2d_scale_tril_uv"][view, index].astype(np.float64)
                * (size - 1)[None, :, None]
            )
            covariance_px = scale_px @ scale_px.swapaxes(-1, -2)
            cameras.append(
                {
                    "id": view,
                    "size_wh": size.tolist(),
                    "presence": float(probability[view]),
                    "occluded": bool(arrays["occlusion_mask"][view, index]),
                    "out_of_frame": bool(arrays["out_of_frame_mask"][view, index]),
                    "truth_px": projected[0].tolist() if valid[0] else None,
                    "truth_in_front": bool(valid[0]),
                    "means_px": (
                        arrays["gmm2d_means_uv"][view, index] * (size - 1)
                    ).tolist(),
                    "covariance_px2": covariance_px.tolist(),
                    "weights": softmax(
                        arrays["gmm2d_mixture_logits"][view, index]
                    ).tolist(),
                    "court_px": [
                        point.tolist() if ok else None
                        for point, ok in zip(court_px, court_valid, strict=True)
                    ],
                    "center_m": (
                        -arrays["camera_estimated_R"][view].T
                        @ arrays["camera_estimated_t"][view]
                    ).tolist(),
                }
            )
        convergence = "not_saved"
        if "integration_converged" in arrays:
            assessed = arrays.get("integration_convergence_assessed")
            convergence = (
                "unassessed"
                if assessed is not None and not assessed[index]
                else (
                    "converged"
                    if arrays["integration_converged"][index]
                    else "nonconverged"
                )
            )
        subset_mass = [
            {
                "cameras": mask.tolist(),
                "mass": float(weights[(subsets == mask).all(-1)].sum()),
            }
            for mask in np.unique(subsets, axis=0)
        ]
        return {
            "dataset": dataset,
            "rally": rally,
            "frame": index,
            "seconds": float(arrays["timestamps_seconds"][index]),
            "truth_m": truth.tolist(),
            "cameras": cameras,
            "means_m": means.tolist(),
            "covariance_m2": cov.tolist(),
            "scale_tril_m": np.linalg.cholesky(cov).tolist(),
            "weights": weights.tolist(),
            "subsets": subsets.tolist(),
            "methods": [
                record["component_method_labels"][int(code)]
                for code in arrays["gmm3d_method_codes"][index]
            ],
            "prior_only_probability": float(arrays["prior_only_probability"][index]),
            "subset_mass": subset_mass,
            "convergence": convergence,
            "event_region": bool(arrays["event_region_mask"][index]),
            "free_flight": bool(arrays["free_flight_mask"][index]),
        }

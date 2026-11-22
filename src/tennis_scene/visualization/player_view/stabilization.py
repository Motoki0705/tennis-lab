"""Finite-window rotational stabilization of saved player-view cameras."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.utils.checksum import dual_sha256

from .camera_bundle import CAMERA_SCHEMA, load_head_cameras

ALGORITHM = "gaussian_markley_quaternion_v1"
RELATIVE_EIGENGAP = 1e-12


@dataclass(frozen=True, slots=True)
class StabilizationConfig:
    """Seconds, with finite support; strength is an explicit caller decision."""

    sigma_seconds: float
    truncate_sigma: float = 3.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.sigma_seconds, bool)
            or not np.isfinite(self.sigma_seconds)
            or self.sigma_seconds < 0
        ):
            raise ValueError("sigma_seconds must be finite and nonnegative")
        if (
            isinstance(self.truncate_sigma, bool)
            or not np.isfinite(self.truncate_sigma)
            or self.truncate_sigma <= 0
        ):
            raise ValueError("truncate_sigma must be finite and positive")

    def radius(self, fps: float) -> int:
        samples = self.truncate_sigma * self.sigma_seconds * fps
        if not np.isfinite(samples):
            raise ValueError("sigma and FPS produce a nonfinite window radius")
        return math.ceil(samples)


def valid_camera_runs(valid: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return half-open intervals; missing frames never belong to a window."""
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def smooth_camera_rotations(
    rotations: NDArray,
    valid: NDArray,
    fps: float,
    config: StabilizationConfig,
) -> NDArray[np.float64]:
    """Markley means of Gaussian-weighted quaternions, without gap filling.

    Rotations use camera-to-court column vectors. At each run edge the window
    is truncated and renormalized; one sample returns that orientation. The
    quaternion outer-product mean is invariant to q/-q. No horizon lock or
    replacement of an ambiguous mean with a neighbouring/raw pose is allowed.
    """
    if (
        rotations.ndim != 3
        or rotations.shape[1:] != (3, 3)
        or valid.shape != (len(rotations),)
        or valid.dtype != np.bool_
        or len(rotations) == 0
        or isinstance(fps, bool)
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(
            "Stabilization requires rotations, bool validity and positive FPS"
        )
    active = np.asarray(rotations[valid], dtype=np.float64)
    if (
        not np.isnan(rotations[~valid]).all()
        or not np.isfinite(active).all()
        or not np.allclose(
            active @ active.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-6
        )
        or not np.allclose(np.linalg.det(active), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("Rotations must be proper when valid and all NaN when missing")
    output = np.array(rotations, dtype=np.float64, copy=True)
    radius = config.radius(fps)
    if config.sigma_seconds == 0:
        return output
    for start, stop in valid_camera_runs(valid):
        quaternions = Rotation.from_matrix(rotations[start:stop]).as_quat()
        for local in range(stop - start):
            indices = np.arange(
                max(0, local - radius), min(stop - start, local + radius + 1)
            )
            weights = np.exp(
                -0.5 * ((indices - local) / (fps * config.sigma_seconds)) ** 2
            )
            weights /= weights.sum()
            selected = quaternions[indices]
            matrix = np.einsum("i,ij,ik->jk", weights, selected, selected)
            eigenvalues, vectors = np.linalg.eigh(matrix)
            if eigenvalues[-1] - eigenvalues[-2] <= RELATIVE_EIGENGAP:
                raise ValueError(f"Frame {start + local}: nonunique quaternion mean")
            output[start + local] = Rotation.from_quat(vectors[:, -1]).as_matrix()
    return output


def export_stabilized_cameras(
    *,
    cameras_dir: Path,
    output_dir: Path,
    config: StabilizationConfig,
) -> Path:
    """Derive cameras from raw C2, retaining C1 provenance and all translations."""
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Camera output already exists: {output_dir}")
    if output_dir.resolve().is_relative_to(cameras_dir.resolve()):
        raise ValueError("Stabilization output must be outside the read-only cameras")
    source_manifest = cameras_dir / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    if (
        manifest["schema"] != CAMERA_SCHEMA
        or manifest["head"]["smoothing"] != "none"
        or "stabilization" in manifest
    ):
        raise ValueError(
            "Stabilization requires raw head cameras, not an already filtered bundle"
        )
    if not manifest["players"] or [p["index"] for p in manifest["players"]] != list(
        range(len(manifest["players"]))
    ):
        raise ValueError("Camera bundle requires nonempty ordered player indices")
    source_digests = {"manifest.json": dual_sha256(source_manifest)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        for record in manifest["players"]:
            saved = load_head_cameras(cameras_dir, player_id=record["player_id"])
            raw_path = (
                cameras_dir / record["path"]
            )  # load_head_cameras validates containment.
            source_digests[record["path"]] = dual_sha256(raw_path)
            with np.load(raw_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            transform = saved.series.camera_to_court.copy()
            transform[:, :3, :3] = smooth_camera_rotations(
                transform[:, :3, :3], saved.series.camera_valid, saved.fps, config
            )
            arrays["camera_to_court"] = transform
            support: NDArray[np.int64] = np.zeros((len(transform), 2), dtype=np.int64)
            radius = config.radius(saved.fps)
            runs = valid_camera_runs(saved.series.camera_valid)
            for start, stop in runs:
                indices = np.arange(start, stop)
                support[start:stop, 0] = np.minimum(indices - start, radius)
                support[start:stop, 1] = np.minimum(stop - 1 - indices, radius)
            arrays["stabilization_support_frames"] = support
            record["path"] = f"player_{record['index']}.npz"
            np.savez(staging / record["path"], **arrays)
            delta = (
                Rotation.from_matrix(
                    transform[saved.series.camera_valid, :3, :3]
                    @ saved.series.camera_to_court[
                        saved.series.camera_valid, :3, :3
                    ].transpose(0, 2, 1)
                ).magnitude()
                if saved.series.camera_valid.any()
                else np.array([])
            )
            record["stabilization"] = {
                "valid_runs": runs,
                "correction_degrees": {
                    "rms": float(np.rad2deg(np.sqrt(np.mean(delta**2))))
                    if len(delta)
                    else None,
                    "p95": float(np.rad2deg(np.percentile(delta, 95)))
                    if len(delta)
                    else None,
                    "max": float(np.rad2deg(delta.max())) if len(delta) else None,
                },
            }
        # The referenced scene may be elsewhere or offline; do not re-read/recompute it.
        scene_path = cameras_dir / manifest["source_bundle"]["path"]
        manifest["source_bundle"]["path"] = os.path.relpath(
            scene_path.resolve(), output_dir.resolve()
        )
        manifest["source_cameras"] = {
            "path": os.path.relpath(cameras_dir.resolve(), output_dir.resolve()),
            "files": source_digests,
        }
        manifest["stabilization"] = {
            "algorithm": ALGORITHM,
            "config": asdict(config),
            "radius_frames": config.radius(manifest["fps"]),
            "kernel": "exp(-0.5 * ((t_j-t_i)/sigma_seconds)^2)",
            "edge_policy": "truncate to the same valid run and normalize weights",
            "missing_policy": "split valid runs; retain NaN, masks, reasons and times",
            "position": "unchanged input camera center; face offset is not recomputed",
            "horizon_lock": False,
            "quaternion_order": "xyzw",
            "arithmetic": "float64",
            "relative_eigengap_min": RELATIVE_EIGENGAP,
            "implementation_sha256": dual_sha256(Path(__file__)),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        }
        manifest["head"]["raw_rotation"] = manifest["head"]["rotation"]
        manifest["head"]["rotation"] = "stabilization of input camera-to-court rotation"
        manifest["head"]["smoothing"] = ALGORITHM
        manifest["head"]["center"] = (
            "unchanged raw input center, including its raw head-rotated face offset"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False)
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Camera output already exists: {output_dir}")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output_dir / "manifest.json"


def validate_stabilization_record(
    manifest: dict[str, Any], arrays: dict[str, NDArray]
) -> None:
    """Validate a portable derived bundle without requiring its raw input online."""
    record = manifest.get("stabilization")
    if not isinstance(record, dict) or record.get("algorithm") != ALGORITHM:
        raise ValueError("Unknown or missing stabilization algorithm")
    parameters = record.get("config")
    if not isinstance(parameters, dict) or set(parameters) != {
        "sigma_seconds",
        "truncate_sigma",
    }:
        raise ValueError(
            "Stabilization config requires sigma_seconds and truncate_sigma"
        )
    config = StabilizationConfig(**parameters)
    radius = config.radius(manifest["fps"])
    if (
        manifest["head"]["smoothing"] != ALGORITHM
        or record.get("radius_frames") != radius
        or record.get("horizon_lock") is not False
        or record.get("relative_eigengap_min") != RELATIVE_EIGENGAP
        or record.get("position")
        != "unchanged input camera center; face offset is not recomputed"
    ):
        raise ValueError(
            "Stabilization metadata disagrees with the supported rotation-only algorithm"
        )
    source = manifest.get("source_cameras")
    files = source.get("files") if isinstance(source, dict) else None
    if not isinstance(files, dict) or "manifest.json" not in files or len(files) < 2:
        raise ValueError(
            "Stabilization provenance requires raw manifest and camera digests"
        )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for value in files.values()
    ):
        raise ValueError("Stabilization provenance contains an invalid input digest")
    valid = arrays["camera_valid"]
    support = arrays.get("stabilization_support_frames")
    expected: NDArray[np.int64] = np.zeros((len(valid), 2), np.int64)
    for start, stop in valid_camera_runs(valid):
        indices = np.arange(start, stop)
        expected[start:stop, 0] = np.minimum(indices - start, radius)
        expected[start:stop, 1] = np.minimum(stop - 1 - indices, radius)
    if (
        support is None
        or support.dtype != np.int64
        or not np.array_equal(support, expected)
    ):
        raise ValueError(
            "Stabilization support crosses a gap or disagrees with the configured window"
        )

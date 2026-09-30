"""Provisional, camera-specific bootstrap of saved #935 validation GMM errors."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, softmax

Array: TypeAlias = NDArray[Any]
CONDITIONS = ("observed", "evidence_gap")
BANK_FIELDS = {
    "error_uv", "scale_tril_uv", "mixture_logits", "presence_logits",
    "camera_index", "condition_index", "source_artifact", "source_frame", "continues",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CalibrationBank:
    arrays: dict[str, Array]

    def __post_init__(self) -> None:
        if set(self.arrays) != BANK_FIELDS:
            raise ValueError("Unknown/incomplete calibration bank")
        a = self.arrays
        n, k, xy = a["error_uv"].shape
        if n < 1 or k < 1 or xy != 2 or a["scale_tril_uv"].shape != (n, k, 2, 2) or a["mixture_logits"].shape != (n, k):
            raise ValueError("Invalid calibration GMM axes")
        if any(a[key].shape != (n,) for key in BANK_FIELDS - {"error_uv", "scale_tril_uv", "mixture_logits"}):
            raise ValueError("Invalid calibration row axes")
        if any(not np.isfinite(value).all() for value in a.values()):
            raise ValueError("Nonfinite calibration bank")
        chol = a["scale_tril_uv"]
        if (chol[..., 0, 1] != 0).any() or (chol.diagonal(axis1=-2, axis2=-1) <= 0).any():
            raise ValueError("Calibration covariance must be SPD Cholesky")
        if a["continues"].dtype != np.bool_ or a["continues"][-1]:
            raise ValueError("Invalid calibration continuation")
        for key in ("camera_index", "condition_index", "source_artifact", "source_frame"):
            if not np.issubdtype(a[key].dtype, np.integer) or (a[key] < 0).any():
                raise ValueError("Invalid calibration identity")
        if set(a["camera_index"].tolist()) != {0, 1, 2} or set(a["condition_index"].tolist()) != {0, 1}:
            raise ValueError("Need three cameras and both observed/gap strata")
        continuation = a["continues"][:-1]
        for key in ("camera_index", "condition_index", "source_artifact"):
            if (a[key][1:][continuation] != a[key][:-1][continuation]).any():
                raise ValueError("Calibration block crosses a source boundary")
        if (np.diff(a["source_frame"])[continuation] != 1).any():
            raise ValueError("Calibration block skips source frames")
        for camera in range(3):
            for condition in range(2):
                if not ((a["camera_index"] == camera) & (a["condition_index"] == condition)).any():
                    raise ValueError("Empty calibration stratum")

    @property
    def components(self) -> int:
        return int(self.arrays["error_uv"].shape[1])

    def draw_rows(self, camera: int, gap: NDArray[np.bool_], rng: np.random.Generator, *, block_frames: int) -> NDArray[np.int64]:
        """Replay up to block_frames adjacent scored frames; no component selection."""
        if type(block_frames) is not int or block_frames < 1 or gap.ndim != 1:
            raise ValueError("Need a positive block length and one temporal gap axis")
        a = self.arrays
        pools = [np.flatnonzero((a["camera_index"] == camera) & (a["condition_index"] == c)) for c in range(2)]
        if any(len(pool) == 0 for pool in pools):
            raise ValueError("Requested camera has no calibration stratum")
        result: NDArray[np.int64] = np.empty(len(gap), dtype=np.int64)
        frame = 0
        while frame < len(gap):
            condition = int(gap[frame])
            row = int(rng.choice(pools[condition]))
            for _ in range(block_frames):
                result[frame] = row
                frame += 1
                if frame == len(gap) or int(gap[frame]) != condition or not a["continues"][row]:
                    break
                row += 1
        return result


def load_calibration(path: Path, expected_sha256: str) -> CalibrationBank:
    if _sha(path) != expected_sha256:
        raise ValueError("Calibration bank SHA mismatch")
    with np.load(path, allow_pickle=False) as stored:
        return CalibrationBank({key: stored[key] for key in stored.files})


def with_calibration_report(settings: dict[str, Any], report_path: Path) -> dict[str, Any]:
    """Replace one calibration bundle, pinning report and sibling bank identities.

    The caller declares the project input boundary. Temporal bootstrap length and
    the explicit out-of-frame hypothesis remain parameters of the generation plan.
    """
    report = json.loads(report_path.read_text())
    if report.get("schema") != "ball_refiner_3d.degradation_calibration.v1":
        raise ValueError("Unknown calibration report schema")
    bank_path = report_path.parent / "bank.npz"
    bank = load_calibration(bank_path, report["bank_sha256"])
    if bank.components != report["components"] or not isinstance(report["status"], str) or not report["status"]:
        raise ValueError("Calibration report identity mismatch")
    result = deepcopy(settings)
    result.update(status=report["status"], components_per_camera=bank.components, max_components=(bank.components + 1) ** 3)
    result["calibration"].update(bank=str(bank_path), bank_sha256=report["bank_sha256"], report=str(report_path), report_sha256=_sha(report_path))
    return result


def _statistics(parts: list[dict[str, Array]]) -> dict[str, Any]:
    data = {key: np.concatenate([part[key] for part in parts]).astype(np.float64) for key in parts[0]}
    weights = softmax(data["mixture_logits"], axis=-1)
    error = data["error_px"]
    chol = data["scale_tril_px"]
    mean = np.einsum("nk,nki->ni", weights, error)
    delta = error - mean[:, None]
    covariance = np.einsum("nk,nkij->nij", weights, chol @ chol.swapaxes(-1, -2) + delta[..., :, None] * delta[..., None, :])
    ratio = np.einsum("ni,nij,nj->n", mean, np.linalg.inv(covariance), mean) / 2
    presence = expit(data["presence_logits"])
    bins = []
    for low, high in zip((0., .5, .9, .95, .99), (.5, .9, .95, .99, 1.), strict=True):
        selected = (presence >= low) & (presence <= high if high == 1 else presence < high)
        bins.append({"lower": low, "upper": high, "count": int(selected.sum()), "predicted_mean": float(presence[selected].mean()) if selected.any() else None, "positive_frequency": 1. if selected.any() else None})
    return {
        "frames": len(mean), "error_to_mixture_covariance_ratio": {"definition": "(mixture_mean-target)^T mixture_covariance^-1 (mixture_mean-target) / 2", "mean": float(ratio.mean()), "p50": float(np.median(ratio)), "p95": float(np.quantile(ratio, .95))},
        "mixture_rms_sigma_px_p50": float(np.median(np.sqrt(np.trace(covariance, axis1=-2, axis2=-1) / 2))),
        "component_weights_mean_original_order": weights.mean(0).tolist(),
        "component_weights_mean_descending": np.sort(weights, axis=1)[:, ::-1].mean(0).tolist(),
        "presence": {"positives": len(presence), "negatives": 0, "mean": float(presence.mean()), "bce_positive": float(np.logaddexp(0, -data["presence_logits"]).mean()), "brier_positive": float(np.mean((1 - presence) ** 2)), "bins": bins, "negative_calibration": "unavailable"},
    }


def build_calibration(source: Path, output: Path, *, status: str = "provisional_saved_validation") -> dict[str, Any]:
    """Extract all scored positive rows from a frozen calibration partition."""
    if not status.strip():
        raise ValueError("Calibration status must be explicit and nonempty")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "ball_refiner_validation_diagnostics.v1" or manifest["partition"] != "calibration":
        raise ValueError("Require #935 saved calibration-partition diagnostics")
    identities = [(a["clip_id"], a["condition"]) for a in manifest["artifacts"]]
    if len(set(identities)) != len(identities) or set(manifest["clip_ids"]) != {key[0] for key in identities} or any((clip, condition) not in identities for clip in manifest["clip_ids"] for condition in CONDITIONS):
        raise ValueError("Duplicate or incomplete calibration predictions")
    if output.exists():
        raise FileExistsError(output)
    rows: dict[str, list[Array]] = {key: [] for key in BANK_FIELDS}
    stats: dict[str, list[dict[str, Array]]] = {}
    inputs = []
    for artifact_index, artifact in enumerate(manifest["artifacts"]):
        path = source / artifact["path"]
        digest = _sha(path)
        if digest != artifact["sha256"]:
            raise ValueError(f"Saved prediction SHA mismatch: {path}")
        clip = artifact["clip_id"]
        if not clip.startswith("meiji/video_000/") or clip.rsplit("/", 1)[-1] not in ("cam0", "cam1", "cam2"):
            raise ValueError("Calibration must use the saved validation cameras only")
        camera = int(clip[-1])
        condition = CONDITIONS.index(artifact["condition"])
        with np.load(path, allow_pickle=False) as z:
            # Source indices are identities, not assumed array offsets.
            source_frames = z["frame_index"]
            indices: NDArray[np.int64] = np.asarray(np.searchsorted(source_frames, z["scored_frame_index"]), dtype=np.int64)
            if (np.diff(source_frames) <= 0).any() or (indices >= len(source_frames)).any() or not np.array_equal(source_frames[indices], z["scored_frame_index"]):
                raise ValueError("Invalid scored-frame identities")
            if len(indices) != artifact["scored_frames"] or not z["position_valid"][indices].all():
                raise ValueError("Need all saved scored, position-known positive frames")
            if condition and not z["gap_mask"][indices].all():
                raise ValueError("Gap calibration must use gap-scored frames")
            error = z["means"][indices] - z["target_uv"][indices, None]
            chol, logits, presence = z["scale_tril"][indices], z["mixture_logits"][indices], z["presence_logits"][indices]
            count = len(indices)
            records = {"error_uv": error, "scale_tril_uv": chol, "mixture_logits": logits, "presence_logits": presence,
                       "camera_index": np.full(count, camera, np.int8), "condition_index": np.full(count, condition, np.int8),
                       "source_artifact": np.full(count, artifact_index, np.int32), "source_frame": source_frames[indices],
                       "continues": np.r_[np.diff(source_frames[indices]) == 1, False]}
            for key, value in records.items():
                rows[key].append(value)
            scale = np.asarray(artifact["source_size_wh"], np.float64) - 1
            stats.setdefault(f"cam{camera}/{CONDITIONS[condition]}", []).append({"error_px": error * scale, "scale_tril_px": chol * scale[:, None], "mixture_logits": logits, "presence_logits": presence})
        inputs.append({**artifact, "sha256": digest})
    bank = CalibrationBank({key: np.concatenate(value) for key, value in rows.items()})
    output.mkdir(parents=True, exist_ok=False)
    bank_path = output / "bank.npz"
    np.savez_compressed(bank_path, **bank.arrays)
    report = {
        "schema": "ball_refiner_3d.degradation_calibration.v1", "status": status,
        "source_directory": str(source), "source_manifest_sha256": _sha(manifest_path),
        "source_checkpoint": manifest["checkpoint"], "source_inputs": manifest["input_sha256"], "source_predictions": inputs,
        "components": bank.components, "rows": len(bank.arrays["error_uv"]), "bank_sha256": _sha(bank_path),
        "statistics": {key: _statistics(parts) for key, parts in stats.items()},
        "limits": ["positive-only presence; out-of-frame logit is an explicit hypothesis", "new detector and person context are not calibrated", "camera errors are bootstrapped independently", "up to 16-frame contiguous blocks; 32/64-frame gaps extrapolate the 1/4/8/16 evidence-gap bank", "translation to synthetic UV and explicit head clipping can change residual statistics; no real trajectory evaluation"],
    }
    (output / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report

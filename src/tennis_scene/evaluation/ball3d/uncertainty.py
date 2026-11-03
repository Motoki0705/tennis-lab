"""Projected posterior covariance, evaluated on the foundation scoring cohort."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from .annotations import Ball2DAnnotations
from .bundle import FixedInputBundle
from .contracts import Ball3DResult, MethodInput
from .runner import MethodRun
from .scoring import error_samples, project_result


def project_covariance(
    inputs: MethodInput, result: Ball3DResult, camera_id: str,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """First-order pinhole propagation; no calibration/reference noise is added.

    Nonzero time offsets need a joint temporal posterior and are rejected.
    A per-frame covariance does not determine cross-time interpolation covariance.
    """
    if result.covariance is None:
        raise ValueError("Projected uncertainty requires 3D covariance")
    if result.time_offsets_seconds is not None and np.any(result.time_offsets_seconds != 0):
        raise ValueError("Projected covariance does not support nonzero time offsets")
    camera = inputs.views[inputs.camera_ids.index(camera_id)].camera
    if result.cameras is not None and camera_id in result.camera_ids:
        camera = result.cameras[result.camera_ids.index(camera_id)]
    uv, valid = project_result(inputs, result, camera_id)
    q = result.positions.astype(float) @ camera.matrix[:, :3].T + camera.matrix[:, 3]
    jac = np.zeros((inputs.num_frames, 2, 3))
    jac[valid] = (camera.matrix[:2, :3] - uv[valid, :, None]*camera.matrix[2, :3]) / q[valid, 2, None, None]
    projected = jac @ result.covariance @ jac.swapaxes(-1, -2)
    return (projected + projected.swapaxes(-1, -2))/2, valid


def ellipse_vertices(
    mean: NDArray[np.floating], covariance: NDArray[np.floating], *, probability: float = .95,
) -> NDArray[np.float64]:
    if (mean.shape != (2,) or covariance.shape != (2, 2) or not 0 < probability < 1
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T)):
        raise ValueError("Ellipse requires a finite 2D mean and symmetric covariance")
    eigen, basis = np.linalg.eigh(covariance)
    if eigen[0] < -1e-10:
        raise ValueError("Ellipse covariance must be PSD")
    angle = np.linspace(0, 2*np.pi, 65)
    circle = np.stack((np.cos(angle), np.sin(angle)), -1)
    return mean + (circle * np.sqrt(np.maximum(eigen, 0)*chi2.ppf(probability, 2))) @ basis.T


def score_uncertainty(
    runs: Mapping[str, Sequence[MethodRun]], bundle: FixedInputBundle,
    annotations: Ball2DAnnotations, *, split_frame: int = 505,
) -> dict[str, Any]:
    """Counts/ellipse coverage by fold, split and reference kind, including LOVO.

    Uses error_samples for sample identities and all its provenance checks.
    Coverage here is conditional on a scored prediction, separately from the
    foundation's prediction/label availability coverage.
    """
    report: dict[str, Any] = {"probabilities": [.5, .9, .95], "degrees_of_freedom": 2,
        "approximation": "first-order Gaussian ellipse; posterior only, no reference/calibration noise",
        "methods": {}}
    for name, values in runs.items():
        if not all(r.result.covariance is not None for r in values):
            continue
        folds, pooled = {}, []
        for run in values:
            samples = error_samples(run, bundle, annotations)
            frames, cameras = samples["frame_index"], samples["camera_index"]
            d2 = np.zeros(len(frames))
            for camera_i in np.unique(cameras):
                camera_id = bundle.inputs.camera_ids[camera_i]
                cov, valid = project_covariance(bundle.inputs, run.result, camera_id)
                uv, _ = project_result(bundle.inputs, run.result, camera_id)
                selected = np.flatnonzero(cameras == camera_i)
                at = frames[selected]
                if not valid[at].all():
                    raise ValueError("Covariance and scored point validity disagree")
                residual = uv[at] - annotations.uv_px[camera_i, at]
                d2[selected] = np.einsum("ni,ni->n", residual, np.linalg.solve(cov[at], residual[..., None])[..., 0])
            record = (frames, samples["reference_kind"], d2)
            folds[run.key] = _summarize(record, split_frame)
            if run.held_out_camera is not None:
                pooled.append(record)
        pool = tuple(np.concatenate([r[i] for r in pooled]) for i in range(3))
        report["methods"][name] = {"folds": folds, "lovo": _summarize(pool, split_frame)}
    return report


def _summarize(record: tuple[NDArray[Any], ...], split: int) -> dict[str, Any]:
    frames, kinds, d2 = record
    result: dict[str, Any] = {}
    for segment, mask in (("tuning", frames < split), ("evaluation", frames >= split), ("all", np.ones(len(frames), bool))):
        result[segment] = {}
        for kind in ("all", "observed", "estimated"):
            selected = mask & ((kinds == kind) if kind != "all" else True)
            values = d2[selected]
            result[segment][kind] = {"count": int(len(values)),
                "median_squared_mahalanobis": float(np.median(values)) if len(values) else None,
                "ellipse_coverage": {str(p): float(np.mean(values <= chi2.ppf(p, 2))) if len(values) else None for p in (.5, .9, .95)}}
    return result

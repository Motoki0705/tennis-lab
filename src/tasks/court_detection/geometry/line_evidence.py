"""Image-space LINE evidence for court registration (no camera/GT assumptions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.spatial import cKDTree

FloatArray: TypeAlias = NDArray[np.float64]


class LineDiagnostics(TypedDict):
    forward_support: float
    reverse_support: float
    segment_support: list[float]
    visible_segments: float
    supported_segments: float
    forward_rms_px: float
    reverse_rms_px: float


@dataclass(frozen=True)
class LineEvidence:
    """Ridge distances and unoriented tangents, expressed in original pixels.

    Probability >= threshold defines support. Local maxima of the interior
    distance transform reduce thick predictions to ridges; probabilities remain
    attached to the observations. No RGB edge detector or neural forward runs.
    """

    width: int
    height: int
    diagonal: float
    distance: FloatArray
    points: FloatArray
    weights: FloatArray
    tangents: FloatArray
    coherence: FloatArray
    tree: cKDTree

    @classmethod
    def from_probability(
        cls,
        probability: NDArray[np.floating],
        image_size_hw: tuple[int, int],
        *,
        threshold: float = 0.5,
        max_observations: int = 768,
    ) -> LineEvidence:
        values = np.asarray(probability, dtype=np.float64)
        height, width = image_size_hw
        if min(height, width) < 2 or values.ndim != 2 or min(values.shape) < 2:
            raise ValueError("LINE and image must be two-dimensional, at least 2x2")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("LINE probabilities must be finite and in [0,1]")
        if not 0 < threshold < 1 or max_observations < 16:
            raise ValueError("Invalid LINE threshold or observation budget")
        resized = cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (resized >= threshold).astype(np.uint8)
        interior = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ridge = (interior > 0) & (interior >= maximum_filter(interior, size=3))
        ys, xs = np.nonzero(ridge)
        if len(xs) < 16 or mask.mean() > 0.25:
            raise ValueError("insufficient_or_diffuse_line_evidence")
        all_points = np.column_stack((xs, ys)).astype(np.float64)
        all_tree = cKDTree(all_points)
        _, neighbors = all_tree.query(all_points, k=min(13, len(all_points)))
        local = all_points[neighbors]
        local -= local.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", local, local)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangents = eigenvectors[:, :, -1]
        coherence = (eigenvalues[:, -1] - eigenvalues[:, 0]) / np.maximum(
            eigenvalues.sum(axis=1), 1e-12
        )
        # Equally spaced in deterministic raster order; never random subsampling.
        chosen = np.linspace(0, len(xs) - 1, min(len(xs), max_observations)).astype(int)
        distance = cv2.distanceTransform(
            (~ridge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        ).astype(np.float64)
        # All ridge points define the distance/tangent field. The reverse term
        # uses the bounded subset only, with its original sigmoid probabilities.
        return cls(
            width,
            height,
            float(np.hypot(width, height)),
            distance,
            all_points[chosen],
            resized[ys[chosen], xs[chosen]],
            tangents,
            coherence,
            all_tree,
        )

    def distances(self, xy: FloatArray, *, smoothing_px: float = 0) -> FloatArray:
        field = (
            gaussian_filter(self.distance, smoothing_px)
            if smoothing_px > 0
            else self.distance
        )
        return np.asarray(
            map_coordinates(
                field, xy.T[::-1], order=1, mode="constant", cval=self.diagonal
            ),
            dtype=np.float64,
        )


def clip_segments(
    segments: FloatArray, width: int, height: int
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Liang--Barsky clipping; observations outside the image are not negatives."""
    a, b = segments[:, 0], segments[:, 1]
    delta = b - a
    lo, hi = np.zeros(len(a)), np.ones(len(a))
    visible: NDArray[np.bool_] = np.ones(len(a), dtype=bool)
    for axis, limit in ((0, width - 1), (1, height - 1)):
        parallel = np.abs(delta[:, axis]) < 1e-12
        visible &= ~(parallel & ((a[:, axis] < 0) | (a[:, axis] > limit)))
        denominator = np.where(parallel, 1.0, delta[:, axis])
        first = -a[:, axis] / denominator
        second = (limit - a[:, axis]) / denominator
        lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(first, second)))
        hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(first, second)))
    visible &= hi > lo
    clipped = np.stack((a + lo[:, None] * delta, a + hi[:, None] * delta), axis=1)
    return clipped, visible


def segment_distances(points: FloatArray, segments: FloatArray) -> FloatArray:
    """All point-to-finite-segment distances, shape [observation, segment]."""
    delta = segments[:, 1] - segments[:, 0]
    offset = points[:, None] - segments[None, :, 0]
    position = np.einsum("nsi,si->ns", offset, delta) / np.maximum(
        (delta**2).sum(axis=1), 1e-12
    )
    foot = segments[None, :, 0] + np.clip(position, 0, 1)[..., None] * delta
    return np.asarray(np.linalg.norm(points[:, None] - foot, axis=2), dtype=np.float64)


def line_residuals(
    projected: FloatArray,
    edges: NDArray[np.int64],
    evidence: LineEvidence,
    tolerance_px: float,
    *,
    samples_per_line: int = 40,
    distance_field: FloatArray | None = None,
) -> tuple[FloatArray, LineDiagnostics]:
    """Balanced bidirectional distances plus local tangent disagreement.

    Forward residuals average each visible segment equally. Reverse residuals
    cover the *fixed* set of observed LINE ridges, preventing image cropping or
    court collapse from cheaply hiding evidence. Distances are truncated at
    4*tolerance; missing line pieces cannot dominate the complete court.
    """
    segments, visible = clip_segments(projected[edges], evidence.width, evidence.height)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    visible &= lengths >= 0.02 * evidence.diagonal
    n = samples_per_line
    samples = segments[:, :1] + np.linspace(0, 1, n)[None, :, None] * (
        segments[:, 1:] - segments[:, :1]
    )
    flattened = samples.reshape(-1, 2)
    raw_distance = np.asarray(
        map_coordinates(
            evidence.distance if distance_field is None else distance_field,
            flattened.T[::-1],
            order=1,
            mode="constant",
            cval=evidence.diagonal,
        )
    ).reshape(len(edges), n)
    count = max(1, int(visible.sum()))
    forward = np.minimum(raw_distance / tolerance_px, 4) * visible[:, None]
    forward /= np.sqrt(count * n)

    distances = segment_distances(evidence.points, segments)
    distances[:, ~visible] = evidence.diagonal
    reverse_px = distances.min(axis=1)
    reverse = np.minimum(reverse_px / tolerance_px, 4) * np.sqrt(
        evidence.weights / evidence.weights.sum()
    )
    _, nearest = evidence.tree.query(flattened)
    tangent = evidence.tangents[nearest].reshape(len(edges), n, 2)
    unit = (segments[:, 1] - segments[:, 0]) / np.maximum(lengths[:, None], 1e-12)
    cross = unit[:, 0, None] * tangent[:, :, 1] - unit[:, 1, None] * tangent[:, :, 0]
    orientation = cross * np.sqrt(evidence.coherence[nearest].reshape(len(edges), n))
    orientation *= visible[:, None] * np.exp(-0.5 * (raw_distance / tolerance_px) ** 2)
    orientation /= np.sqrt(count * n)
    supports = (raw_distance <= tolerance_px).mean(axis=1) * visible
    diagnostics: LineDiagnostics = {
        "forward_support": float(supports.sum() / count),
        "reverse_support": float(
            np.average(reverse_px <= tolerance_px, weights=evidence.weights)
        ),
        "segment_support": supports.tolist(),
        "visible_segments": float(visible.sum()),
        "supported_segments": float(np.sum(supports >= 0.5)),
        "forward_rms_px": float(np.sqrt(np.mean(raw_distance[visible] ** 2)))
        if visible.any()
        else evidence.diagonal,
        "reverse_rms_px": float(
            np.sqrt(np.average(reverse_px**2, weights=evidence.weights))
        ),
    }
    return np.concatenate(
        (forward.ravel(), reverse, np.sqrt(0.1) * orientation.ravel())
    ), diagnostics

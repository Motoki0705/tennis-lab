"""Single-frame paint centres from paired RGB stripe edges, in source pixels.

No learned heads, camera identity, reference coordinates or annotation masks are
used here. Multiple stripe hypotheses survive extraction for ambiguity checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .homography import COURT_HOMOGRAPHY_EDGES, court_template_xy
from .line_evidence import FloatArray, clip_segments

PAINT_NAMES = ("far_baseline", "near_baseline", "left_doubles", "right_doubles",
               "left_singles", "right_singles", "far_service", "near_service", "centre_service")


@dataclass(frozen=True, slots=True)
class PaintConfig:
    """Pixel tolerances describe the paint measurement, not GT correctness."""

    version: str = "semantic_consensus_v2"
    radius_px: float = 48.
    fine_radius_px: float = 6.
    sample_spacing_px: float = 8.
    profile_step_px: float = .5
    profile_sigma_px: float = .6
    min_gradient: float = 2.
    min_contrast: float = 6.
    min_width_px: float = 1.
    max_width_px: float = 18.
    line_angle_limit_degrees: float = 8.
    line_inlier_distance_px: float = 1.5
    ransac_trials: int = 500
    min_line_locations: int = 8
    min_coverage: float = .35
    min_court_lines: int = 5
    max_line_residual_px: float = 3.
    ambiguity_support_ratio: float = .8
    ambiguity_separation_px: float = 3.
    max_projection_std_px: float = 5.
    max_leave_line_out_px: float = 10.
    max_normalized_condition: float = 10000.
    max_nfev: int = 200

    def __post_init__(self) -> None:
        from dataclasses import fields

        if self.version not in {"paired_edges_v1", "semantic_consensus_v2"}:
            raise ValueError("Unknown paint extraction version")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != "version" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"paint.{field.name} must be positive and finite")
        if self.min_width_px >= self.max_width_px or self.fine_radius_px > self.radius_px:
            raise ValueError("Invalid paint search/width interval")
        if not 0 < self.min_coverage <= 1 or not 0 < self.ambiguity_support_ratio <= 1:
            raise ValueError("Invalid paint support fraction")
        if not 4 <= self.min_court_lines <= 9 or self.min_line_locations < 4:
            raise ValueError("Insufficient paint constraints")


@dataclass(frozen=True)
class PaintLine:
    index: int
    status: str
    points: FloatArray
    line: FloatArray | None
    locations: int
    coverage: float
    ambiguous: bool = False
    alternatives: tuple[FloatArray, ...] = ()


def stripe_pairs(profile: FloatArray, offsets: FloatArray, config: PaintConfig) -> list[tuple[float, float]]:
    smooth = gaussian_filter1d(profile, config.profile_sigma_px / config.profile_step_px)
    derivative = np.gradient(smooth, config.profile_step_px)
    positive, _ = find_peaks(derivative, height=config.min_gradient, distance=2)
    negative, _ = find_peaks(-derivative, height=config.min_gradient, distance=2)

    def subpixel(index: int) -> float:
        d = derivative if derivative[index] > 0 else -derivative
        denominator = d[index - 1] - 2 * d[index] + d[index + 1]
        delta = .5 * (d[index - 1] - d[index + 1]) / denominator if abs(denominator) > 1e-8 else 0.
        return float(offsets[index] + np.clip(delta, -.5, .5) * config.profile_step_px)

    pairs = []
    for a in positive:
        for b in negative:
            if not config.min_width_px <= offsets[b] - offsets[a] <= config.max_width_px:
                continue
            outside = max(float(np.median(smooth[max(0, a - 8):max(1, a - 3)])),
                          float(np.median(smooth[min(len(smooth) - 1, b + 3):min(len(smooth), b + 8)])))
            contrast = float(np.mean(smooth[a:b + 1])) - outside
            if contrast < config.min_contrast or np.any((positive > a) & (positive < b) & (derivative[positive] > .7 * derivative[a])):
                continue
            pairs.append(((subpixel(a) + subpixel(b)) / 2, contrast))
    return sorted(pairs, key=lambda p: (-p[1], abs(p[0]), p[0]))[:5]


def _fit_line(index: int, samples: FloatArray, locations: int, normal: FloatArray,
              length: float, config: PaintConfig) -> PaintLine:
    empty: FloatArray = np.empty((0, 2), dtype=float)
    if len(samples) < config.min_line_locations:
        return PaintLine(index, "insufficient_edge_pairs", empty, None, locations, 0.)
    rng = np.random.default_rng(2190 + index)
    models: list[tuple[int, float, FloatArray, NDArray[np.bool_]]] = []
    for _ in range(config.ransac_trials):
        a, b = rng.integers(0, len(samples), 2)
        delta = samples[b, :2] - samples[a, :2]
        if np.linalg.norm(delta) < .35 * length:
            continue
        n = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        if abs(n @ normal) < np.cos(np.deg2rad(config.line_angle_limit_degrees)):
            continue
        line = np.r_[n, -n @ samples[a, :2]]
        mask = np.abs(samples[:, :2] @ n + line[2]) <= config.line_inlier_distance_px
        models.append((len(np.unique(samples[mask, 2])), float(np.mean(np.minimum(samples[mask, 3], 40))), line, mask))
    if not models:
        return PaintLine(index, "no_line_consensus", empty, None, locations, 0.)
    models.sort(key=lambda m: (-m[0], -m[1]))
    count, _, line, mask = models[0]
    selected: NDArray[np.int64] = np.empty(0, dtype=int)
    for _ in range(4):
        candidates = np.flatnonzero(mask)
        ids = []
        for location in np.unique(samples[candidates, 2]):
            group = candidates[samples[candidates, 2] == location]
            ids.append(group[np.argmin(np.abs(samples[group, :2] @ line[:2] + line[2]))])
        selected = np.array(ids)
        points = samples[selected, :2]
        middle = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - middle, full_matrices=False)
        line = np.r_[vh[-1], -vh[-1] @ middle]
        mask = np.abs(samples[:, :2] @ line[:2] + line[2]) <= config.line_inlier_distance_px
    coverage = float(np.ptp(samples[selected, 2]) / max(locations - 1, 1))
    points = samples[selected, :2]
    # A second extended parallel stripe is an unresolved semantic alternative;
    # brightness alone cannot decide that it is paint rather than net tape.
    alternatives: list[FloatArray] = []
    alternative_lines: list[FloatArray] = []
    for n, _, alternative, alternative_mask in models[1:]:
        if n < config.ambiguity_support_ratio * count:
            break
        if float(np.median(np.abs(points @ alternative[:2] + alternative[2]))) < config.ambiguity_separation_px:
            continue
        candidate_points = samples[alternative_mask, :2]
        if any(float(np.median(np.abs(candidate_points @ old[:2] + old[2]))) < config.ambiguity_separation_px for old in alternative_lines):
            continue
        alternatives.append(candidate_points)
        alternative_lines.append(alternative)
        if len(alternatives) == 3:
            break
    status = "ok" if len(points) >= config.min_line_locations and coverage >= config.min_coverage else "insufficient_coverage"
    return PaintLine(index, status, points, line, locations, coverage, bool(alternatives), tuple(alternatives))


def extract_paint_lines(brightness: FloatArray, projected: FloatArray, radius: float,
                        config: PaintConfig) -> tuple[PaintLine, ...]:
    height, width = brightness.shape
    segments, visible = clip_segments(projected[np.asarray(COURT_HOMOGRAPHY_EDGES)], width, height)
    offsets: FloatArray = np.arange(-radius, radius + .01, config.profile_step_px, dtype=np.float64)
    lines = []
    for index, (segment, is_visible) in enumerate(zip(segments, visible, strict=True)):
        length = float(np.linalg.norm(segment[1] - segment[0]))
        if not is_visible or length < config.sample_spacing_px * config.min_line_locations:
            lines.append(PaintLine(index, "outside_or_short", np.empty((0, 2)), None, 0, 0.))
            continue
        direction = (segment[1] - segment[0]) / length
        normal = np.array([-direction[1], direction[0]])
        positions = np.linspace(.04, .96, max(10, int(length / config.sample_spacing_px)))
        centres = segment[0] + positions[:, None] * (segment[1] - segment[0])
        xy = centres[:, None] + offsets[None, :, None] * normal
        inside = (xy[..., 0] >= 1) & (xy[..., 0] < width - 2) & (xy[..., 1] >= 1) & (xy[..., 1] < height - 2)
        profiles = cv2.remap(brightness.astype(np.float32), xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        samples = []
        for location, profile in enumerate(profiles):
            if not inside[location].all():
                continue
            for offset, contrast in stripe_pairs(profile, offsets, config):
                point = centres[location] + offset * normal
                samples.append([*point, location, contrast])
        lines.append(_fit_line(index, np.asarray(samples, dtype=float).reshape(-1, 4), len(positions), normal, length, config))
    return tuple(lines)


def image_lines(matrix: FloatArray) -> FloatArray:
    world = court_template_xy()
    edges = np.asarray(COURT_HOMOGRAPHY_EDGES)
    lines = np.cross(np.c_[world[edges[:, 0]], np.ones(len(edges))], np.c_[world[edges[:, 1]], np.ones(len(edges))]) @ np.linalg.inv(matrix)
    return np.asarray(lines / np.linalg.norm(lines[:, :2], axis=1)[:, None], dtype=np.float64)

"""Measured KP/LINE evidence and the actual seeds of a hybrid court fit."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from numpy.typing import NDArray

from src.tasks.base.visualization.presentation.style import BACKGROUND, NEW
from src.tasks.base.visualization.presentation.timeline import Segment, StepClock
from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.geometry.hybrid_homography import HybridHypothesis
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
)

KP_ONLY_COLOR = "#E63946"
LINE_COLOR = "#4CC9F0"
SEED_COLOR = "#FFD166"
HYBRID_COLOR = NEW
STAGES = (
    ("input", 1.0),
    ("kp", 2.0),
    ("kp_only", 2.0),
    ("line", 2.0),
    ("seeds", 3.0),
    ("hybrid", 2.0),
    ("hold", 2.0),
)


def story_clock(fps: float = 30.0) -> StepClock:
    """The agreed fourteen seconds contain only presentation segments."""
    return StepClock([Segment(name, seconds, 0, 0) for name, seconds in STAGES], fps)


def story_stills(clock: StepClock) -> dict[str, int]:
    """Use settled stage endpoints; the final still contains only the green fit."""
    endpoints = {clock.at(frame)[0].name: frame for frame in range(clock.frame_count())}
    return {
        "01_input": endpoints["input"],
        "02_kp": endpoints["kp"],
        "03_kp_only": endpoints["kp_only"],
        "04_line": endpoints["line"],
        "05_seeds": endpoints["seeds"],
        "06_hybrid": endpoints["hold"],
    }


@dataclass(frozen=True)
class HybridFitStory:
    image_rgb: NDArray[np.uint8]
    kp_heatmap: NDArray[np.float32]
    kp_xy: NDArray[np.float64]
    kp_score: NDArray[np.float64]
    line_probability: NDArray[np.float32]
    kp_only_projected: NDArray[np.float64] | None
    kp_only_fit_inliers: NDArray[np.bool_]
    seeds: tuple[HybridHypothesis, ...]
    best: HybridHypothesis
    status: str

    def __post_init__(self) -> None:
        if self.status != "ok":
            raise ValueError(
                f"Hybrid fit story requires status='ok', got {self.status!r}"
            )
        if not self.seeds:
            raise ValueError("Hybrid fit story requires nonempty seeds")
        if (
            self.image_rgb.dtype != np.uint8
            or self.image_rgb.ndim != 3
            or self.image_rgb.shape[2] != 3
            or min(self.image_rgb.shape[:2]) < 2
        ):
            raise ValueError("image_rgb must be uint8 RGB")
        for probability in (self.kp_heatmap, self.line_probability):
            if (
                probability.shape != self.image_rgb.shape[:2]
                or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))
            ):
                raise ValueError(
                    "Display probability maps must match image size in [0,1]"
                )
        if (
            self.kp_xy.shape != (14, 2)
            or np.isinf(self.kp_xy).any()
            or self.kp_score.shape != (14,)
            or not np.isfinite(self.kp_score).all()
            or np.any((self.kp_score < 0) | (self.kp_score > 1))
            or self.kp_only_fit_inliers.shape != (14,)
            or self.kp_only_fit_inliers.dtype != np.bool_
        ):
            raise ValueError("Story requires KP14 coordinates, scores and fit inliers")
        if self.kp_only_projected is None:
            raise ValueError("Story requires an existing KP-only homography")
        for projected in (
            self.kp_only_projected,
            self.best.projected,
            *(seed.projected for seed in self.seeds),
        ):
            if projected.shape != (14, 2) or not np.isfinite(projected).all():
                raise ValueError("Story requires finite projected KP14")
        if self.best.selected.shape != (14,) or self.best.selected.dtype != np.bool_:
            raise ValueError("Story requires a KP14 selection mask")
        if any(not np.isfinite(seed.score) or seed.score < 0 for seed in self.seeds):
            raise ValueError("Seed scores must be finite and nonnegative")


def build_story(
    prediction: CourtPrediction, image_rgb: NDArray[np.uint8]
) -> HybridFitStory:
    """Resize native evidence for display only; never rerun or modify geometry."""
    result = prediction.homography
    if result is None or result.status != "ok" or result.best is None:
        raise ValueError("Hybrid fit story requires an accepted status='ok' prediction")
    if image_rgb.shape[:2] != prediction.original_size_hw:
        raise ValueError("Image size differs from the prediction's original size")
    kp, line = prediction.raw_heads.get("kp"), prediction.raw_heads.get("line")
    if not isinstance(kp, CourtKeypointPrediction) or not isinstance(
        line, CourtLinePrediction
    ):
        raise ValueError("Hybrid fit story requires raw KP and LINE heads")
    if kp.keypoints.shape != (14, 1, 2) or kp.heatmaps.shape[0] != 14:
        raise ValueError("Hybrid fit story requires one peak for each of KP14")
    xy = kp.keypoints[:, 0].detach().cpu().numpy().astype(np.float64)
    xy[~kp.valid[:, 0].detach().cpu().numpy()] = np.nan
    heatmap = kp.heatmaps.detach().cpu().numpy().max(axis=0)
    probability = line.probability.detach().cpu().numpy()
    height, width = prediction.original_size_hw
    return HybridFitStory(
        image_rgb=image_rgb.copy(),
        kp_heatmap=np.asarray(cv2.resize(heatmap, (width, height)), dtype=np.float32),
        kp_xy=xy,
        kp_score=kp.scores[:, 0].detach().cpu().numpy().astype(np.float64),
        line_probability=np.asarray(
            cv2.resize(probability, (width, height)), dtype=np.float32
        ),
        kp_only_projected=result.kp_only.projected
        if result.kp_only.matrix is not None
        else None,
        kp_only_fit_inliers=result.kp_only.fit_inliers.copy(),
        seeds=result.seeds,
        best=result.best,
        status=result.status,
    )


def _rgb(color: str) -> tuple[int, int, int]:
    red, green, blue = to_rgb(color)
    return round(red * 255), round(green * 255), round(blue * 255)


def _overlay(
    image: NDArray[np.uint8],
    colors: NDArray[np.floating] | tuple[int, int, int],
    alpha: NDArray[np.floating],
) -> NDArray[np.uint8]:
    weight = alpha[..., None]
    return np.rint(image * (1 - weight) + np.asarray(colors) * weight).astype(np.uint8)


def draw_court(
    image: NDArray[np.uint8], projected: NDArray[np.float64], color: str
) -> None:
    """Nine continuous ground markings, clipped before integer rasterization."""
    height, width = image.shape[:2]
    thickness = max(1, round(np.hypot(width, height) * 0.002))
    for a, b in COURT_HOMOGRAPHY_EDGES:
        points = np.rint(projected[[a, b]]).astype(np.int64)
        visible, start, stop = cv2.clipLine(
            (0, 0, width, height), tuple(points[0]), tuple(points[1])
        )
        if visible:
            cv2.line(image, start, stop, _rgb(color), thickness, cv2.LINE_AA)


def _draw_kp(image: NDArray[np.uint8], story: HybridFitStory, *, mode: str) -> None:
    radius = max(3, round(np.hypot(*image.shape[:2]) * 0.004))
    colors = np.rint(colormaps["viridis"](story.kp_score)[:, :3] * 255).astype(int)
    height, width = image.shape[:2]
    for index, xy in enumerate(story.kp_xy):
        if not np.isfinite(xy).all() or not (
            0 <= xy[0] < width and 0 <= xy[1] < height
        ):
            continue
        point = (round(float(xy[0])), round(float(xy[1])))
        if mode == "hybrid":
            if story.best.selected[index]:
                cv2.circle(image, point, radius, _rgb(HYBRID_COLOR), -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(
                    image,
                    point,
                    _rgb(KP_ONLY_COLOR),
                    cv2.MARKER_TILTED_CROSS,
                    radius * 2 + 1,
                    max(1, radius // 2),
                    cv2.LINE_AA,
                )
        else:
            filled = mode != "kp_only" or story.kp_only_fit_inliers[index]
            color = tuple(int(value) for value in colors[index])
            cv2.circle(
                image,
                point,
                radius,
                color,
                -1 if filled else max(1, radius // 3),
                cv2.LINE_AA,
            )


def render_story_frame(
    story: HybridFitStory, frame: int, clock: StepClock
) -> NDArray[np.uint8]:
    """Draw a text-free stage in original-image pixels."""
    segment, progress, _ = clock.at(frame)
    image = story.image_rgb.copy()
    name = segment.name
    if name == "input":
        return image
    if name == "kp":
        heat_colors = colormaps["inferno"](story.kp_heatmap)[..., :3] * 255
        image = _overlay(image, heat_colors, 0.55 * progress * story.kp_heatmap)
        _draw_kp(image, story, mode="kp")
    elif name == "kp_only":
        assert story.kp_only_projected is not None
        draw_court(image, story.kp_only_projected, KP_ONLY_COLOR)
        _draw_kp(image, story, mode="kp_only")
    elif name in {"line", "seeds", "hybrid", "hold"}:
        line_alpha = 0.6 * (1 - progress if name == "hold" else 1)
        image = _overlay(image, _rgb(LINE_COLOR), line_alpha * story.line_probability)
        if name == "seeds":
            index = min(int(progress * len(story.seeds)), len(story.seeds) - 1)
            seed = story.seeds[index]
            draw_court(image, seed.projected, SEED_COLOR)
            height, width = image.shape[:2]
            bar_width = max(1, round(width * 0.18))
            right, bottom = width - max(4, width // 40), height - max(4, height // 30)
            bar_height = max(3, height // 90)
            cv2.rectangle(
                image,
                (right - bar_width, bottom - bar_height),
                (right, bottom),
                _rgb(BACKGROUND),
                -1,
            )
            maximum = max(seed.score for seed in story.seeds)
            length = round(bar_width * seed.score / maximum) if maximum else 0
            if length:
                cv2.rectangle(
                    image,
                    (right - bar_width, bottom - bar_height),
                    (right - bar_width + length, bottom),
                    _rgb(SEED_COLOR),
                    -1,
                )
        elif name in {"hybrid", "hold"}:
            draw_court(image, story.best.projected, HYBRID_COLOR)
            if name == "hybrid":
                _draw_kp(image, story, mode="hybrid")
    else:
        raise ValueError(f"Unknown hybrid fit stage: {name!r}")
    return image


def fit_frame(
    image: NDArray[np.uint8], *, width: int, height: int
) -> NDArray[np.uint8]:
    """Letterbox without stretching the measured court geometry."""
    scale = min(width / image.shape[1], height / image.shape[0])
    new_width, new_height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    resized = cv2.resize(image, (new_width, new_height))
    canvas: NDArray[np.uint8] = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = _rgb(BACKGROUND)
    left, top = (width - new_width) // 2, (height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas

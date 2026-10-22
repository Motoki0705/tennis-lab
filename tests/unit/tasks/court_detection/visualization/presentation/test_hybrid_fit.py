"""Check measured evidence, stage colors, selection masks and final hold."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from numpy.typing import NDArray

from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
)
from src.tasks.court_detection.visualization.presentation.hybrid_fit import (
    HYBRID_COLOR,
    KP_ONLY_COLOR,
    LINE_COLOR,
    SEED_COLOR,
    build_story,
    fit_frame,
    render_story_frame,
    story_clock,
    story_stills,
)


def _rgb(color: str) -> NDArray[np.int64]:
    return np.rint(np.array(to_rgb(color)) * 255).astype(np.int64)


def _midpoint(
    projected: NDArray[np.float64], edge: tuple[int, int] = (0, 1)
) -> tuple[int, int]:
    x, y = np.rint(projected[list(edge)].mean(axis=0)).astype(int)
    return int(y), int(x)


def test_story_resizes_only_display_maps_and_preserves_evidence(
    hybrid_prediction,
) -> None:
    prediction, image = hybrid_prediction
    kp = prediction.raw_heads["kp"]
    line = prediction.raw_heads["line"]
    assert isinstance(kp, CourtKeypointPrediction)
    assert isinstance(line, CourtLinePrediction)
    assert kp.heatmaps.min() < 0 and kp.heatmaps.max() > 1
    original_heatmap, original_line = kp.heatmaps.clone(), line.probability.clone()
    story = build_story(prediction, image)
    assert np.isnan(story.kp_xy[-1]).all()
    np.testing.assert_array_equal(
        story.kp_only_fit_inliers, prediction.homography.kp_only.fit_inliers
    )
    np.testing.assert_allclose(
        story.kp_heatmap,
        cv2.resize(kp.heatmaps.sigmoid().numpy().max(axis=0), (320, 240)),
    )
    np.testing.assert_allclose(
        story.line_probability, cv2.resize(line.probability.numpy(), (320, 240))
    )
    np.testing.assert_array_equal(kp.heatmaps.numpy(), original_heatmap.numpy())
    np.testing.assert_array_equal(line.probability.numpy(), original_line.numpy())
    assert story.seeds is prediction.homography.seeds


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_kp_logits_cannot_be_hidden_by_display_sigmoid(
    hybrid_prediction, value
) -> None:
    prediction, image = hybrid_prediction
    kp = prediction.raw_heads["kp"]
    assert isinstance(kp, CourtKeypointPrediction)
    logits = kp.heatmaps.clone()
    logits[0, 0, 0] = value
    prediction = replace(
        prediction,
        raw_heads={**prediction.raw_heads, "kp": replace(kp, heatmaps=logits)},
    )
    with pytest.raises(ValueError, match="KP heatmap logits must be finite"):
        build_story(prediction, image)


@pytest.mark.parametrize(
    "status", ["joint_optimization_failed", "ambiguous_candidates"]
)
def test_rejected_geometry_cannot_be_presented_as_success(
    hybrid_prediction, status
) -> None:
    prediction, image = hybrid_prediction
    with pytest.raises(ValueError, match="status='ok'"):
        build_story(
            replace(
                prediction, homography=replace(prediction.homography, status=status)
            ),
            image,
        )


def test_missing_seeds_baseline_or_heads_fail_explicitly(hybrid_prediction) -> None:
    prediction, image = hybrid_prediction
    geometry = prediction.homography
    with pytest.raises(ValueError, match="seeds"):
        build_story(replace(prediction, homography=replace(geometry, seeds=())), image)
    with pytest.raises(ValueError, match="KP-only"):
        build_story(
            replace(
                prediction,
                homography=replace(
                    geometry, kp_only=replace(geometry.kp_only, matrix=None)
                ),
            ),
            image,
        )
    with pytest.raises(ValueError, match="heads"):
        build_story(
            replace(prediction, raw_heads={"kp": prediction.raw_heads["kp"]}), image
        )
    with pytest.raises(ValueError, match="size"):
        build_story(prediction, image[:100])


def test_fixed_timeline_stills_and_input(hybrid_prediction) -> None:
    prediction, image = hybrid_prediction
    story, clock = build_story(prediction, image), story_clock()
    assert clock.frame_count() == 420
    assert [clock.at(frame)[0].name for frame in [0, 30, 90, 150, 210, 300, 360]] == [
        "input",
        "kp",
        "kp_only",
        "line",
        "seeds",
        "hybrid",
        "hold",
    ]
    assert story_stills(clock) == {
        "01_input": 29,
        "02_kp": 89,
        "03_kp_only": 149,
        "04_line": 209,
        "05_seeds": 299,
        "06_hybrid": 419,
    }
    np.testing.assert_array_equal(render_story_frame(story, 0, clock), image)


@pytest.mark.parametrize(
    "stage_frame,color,projection",
    [
        (120, KP_ONLY_COLOR, "kp_only_projected"),
        (210, SEED_COLOR, "seed"),
        (330, HYBRID_COLOR, "best"),
    ],
)
def test_stage_color_lies_on_the_actual_projected_segments(
    hybrid_prediction, stage_frame, color, projection
) -> None:
    prediction, image = hybrid_prediction
    story = build_story(prediction, image)
    points = (
        story.seeds[0].projected
        if projection == "seed"
        else (story.best.projected if projection == "best" else story.kp_only_projected)
    )
    frame = render_story_frame(story, stage_frame, story_clock())
    # Check every continuous marking, away from point markers at its endpoints.
    for edge in COURT_HOMOGRAPHY_EDGES:
        y, x = _midpoint(points, edge)
        patch = frame[y - 2 : y + 3, x - 2 : x + 3].astype(float)
        # A one-pixel AA line blends its center with the underlying image.
        assert np.min(np.linalg.norm(patch - _rgb(color), axis=2)) < 35


def test_heatmap_fades_in_and_kp_color_is_confidence_viridis(hybrid_prediction) -> None:
    prediction, image = hybrid_prediction
    story, clock = build_story(prediction, image), story_clock()
    first, last = (render_story_frame(story, frame, clock) for frame in (30, 89))
    assert np.linalg.norm(last[22, 22].astype(float) - image[22, 22]) > np.linalg.norm(
        first[22, 22].astype(float) - image[22, 22]
    )
    x, y = np.rint(story.kp_xy[0]).astype(int)
    np.testing.assert_allclose(
        last[y, x],
        np.rint(np.array(colormaps["viridis"](story.kp_score[0])[:3]) * 255),
        atol=1,
    )


def test_kp_only_inliers_are_filled_and_other_points_hollow(hybrid_prediction) -> None:
    prediction, image = hybrid_prediction
    story = build_story(prediction, image)
    frame = render_story_frame(story, 120, story_clock())
    x, y = np.rint(story.kp_xy[0]).astype(int)
    assert not np.array_equal(frame[y, x], image[y, x])
    x, y = np.rint(story.kp_xy[8]).astype(int)
    np.testing.assert_array_equal(frame[y, x], image[y, x])


def test_line_overlay_and_final_hold_remove_evidence_and_markers(
    hybrid_prediction,
) -> None:
    prediction, image = hybrid_prediction
    story, clock = build_story(prediction, image), story_clock()
    line = render_story_frame(story, 180, clock)
    expected = np.rint(
        image[41, 20] * (1 - 0.6 * story.line_probability[41, 20])
        + _rgb(LINE_COLOR) * 0.6 * story.line_probability[41, 20]
    )
    np.testing.assert_array_equal(line[41, 20], expected)
    final = render_story_frame(story, 419, clock)
    np.testing.assert_array_equal(final[41, 20], image[41, 20])
    # Put one rejected peak away from all fitted segments to distinguish its x.
    peaks = story.kp_xy.copy()
    peaks[9] = [30, 180]
    story = replace(story, kp_xy=peaks)
    joint = render_story_frame(story, 330, clock)
    np.testing.assert_array_equal(joint[180, 30], _rgb(KP_ONLY_COLOR))
    np.testing.assert_array_equal(
        render_story_frame(story, 419, clock)[180, 30], image[180, 30]
    )


def test_all_seeds_get_equal_time_and_bar_uses_common_score_scale(
    hybrid_prediction,
) -> None:
    prediction, image = hybrid_prediction
    story, clock = build_story(prediction, image), story_clock()
    frames = [render_story_frame(story, frame, clock) for frame in (210, 254, 255, 299)]
    np.testing.assert_array_equal(frames[0], frames[1])
    np.testing.assert_array_equal(frames[2], frames[3])
    for index in (0, 2):
        y, x = _midpoint(story.seeds[index // 2].projected)
        assert np.linalg.norm(frames[index][y, x].astype(float) - _rgb(SEED_COLOR)) < 35
    yellow_counts: list[int] = [
        int(np.sum(np.all(frame[230:233, 250:313] == _rgb(SEED_COLOR), axis=2)))
        for frame in frames
    ]
    assert yellow_counts[0] < yellow_counts[2]


def test_frame_letterboxing_preserves_aspect_ratio(hybrid_prediction) -> None:
    _, image = hybrid_prediction
    frame = fit_frame(image, width=960, height=540)
    assert frame.shape == (540, 960, 3)
    np.testing.assert_array_equal(frame[:, 0], np.full((540, 3), 18))
    np.testing.assert_array_equal(frame[:, 120:840], np.full((540, 720, 3), 32))

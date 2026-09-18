"""Static court fitting rejects insufficient evidence and resists outliers."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tennis_scene.dataset_pipeline.court import (
    StaticCourtSettings,
    fit_static_court,
)
from src.utils.schema.court import CourtConfig, court_keypoints_3d


def test_static_fit_rejects_an_outlier_without_moving_the_court() -> None:
    world = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    h = np.array([[35, 0, 960], [0, 20, 540], [0, 0.01, 1]], np.float64)
    expected = cv2.perspectiveTransform(world[None], h)[0]
    raw = np.repeat(expected[None], 5, axis=0)
    raw[:, 3] += [300, -100]
    scores: np.ndarray = np.ones((5, 14), np.float32)
    settings = StaticCourtSettings(Path("model.ckpt"), 5, 0.3, 6, 10, 5)
    fitted, _, receipt = fit_static_court(
        raw, scores, size=(1920, 1080), settings=settings
    )
    assert 3 not in receipt["inlier_channels"]
    np.testing.assert_allclose(fitted * [1920, 1080], expected, atol=1e-3)
    assert receipt["is_measured_calibration"] is False


def test_static_fit_refuses_unobserved_court_channels() -> None:
    settings = StaticCourtSettings(Path("model.ckpt"), 5, 0.3, 6, 10, 5)
    with pytest.raises(ValueError, match="temporal support"):
        fit_static_court(
            np.zeros((5, 14, 2)),
            np.zeros((5, 14)),
            size=(1920, 1080),
            settings=settings,
        )


@pytest.mark.parametrize(
    ("base", "padding", "expected"),
    [
        ((20, 30, 70, 80), 2.5, (7, 18, 93, 98)),
        ((0, 0, 100, 100), 20.0, (0, 0, 100, 100)),
    ],
)
def test_court_extent_union_rounds_outward_and_clamps(
    base: tuple[int, int, int, int], padding: float, expected: tuple[int, int, int, int]
) -> None:
    from src.tennis_scene.dataset_pipeline.court import court_extent_roi

    points = np.tile([[10.2, 20.8], [90.1, 95.2]], (7, 1)).astype(np.float32)
    assert court_extent_roi(base, points, (100, 100), padding) == expected


@pytest.mark.parametrize("invalid", ["nan", "degenerate", "shape", "roi", "padding"])
def test_court_extent_rejects_invalid_inputs(invalid: str) -> None:
    from src.tennis_scene.dataset_pipeline.court import court_extent_roi

    points = np.tile([[10.0, 20.0], [90.0, 95.0]], (7, 1))
    base = (20, 30, 70, 80)
    padding = 20.0
    if invalid == "nan":
        points[0, 0] = np.nan
    elif invalid == "degenerate":
        points[:] = 1
    elif invalid == "shape":
        points = points[:13]
    elif invalid == "roi":
        base = (20, 30, 20, 80)
    else:
        padding = float("inf")
    with pytest.raises(ValueError):
        court_extent_roi(base, points, (100, 100), padding)


@pytest.mark.parametrize("mode", ["refined", "disabled", "full", "rejected"])
def test_observe_court_passes_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from src.tennis_scene.dataset_pipeline import court
    from src.utils.configuration import PathResolver, RuntimePathRoots

    world = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    expected = cv2.perspectiveTransform(
        world[None], np.array([[3.0, 0, 50], [0, 2.0, 50], [0, 0, 1]])
    )[0]
    frame: np.ndarray = np.zeros((100, 100, 3), np.uint8)
    frame[:, :, 0] = np.arange(100)[None, :]
    frame[:, :, 1] = np.arange(100)[:, None]
    capture = Mock()
    capture.read.return_value = (True, frame)
    monkeypatch.setattr(court.cv2, "VideoCapture", lambda path: capture)
    base = (0, 0, 100, 100) if mode == "full" else (35, 35, 65, 65)
    monkeypatch.setattr(court, "ball_guided_roi", lambda *args: base)
    seen = []

    def predict(rgb: np.ndarray) -> SimpleNamespace:
        seen.append(rgb.shape)
        second_pass = len(seen) > 3
        points = expected + ([1.0, 2.0] if second_pass else [0.0, 0.0])
        points = points - [rgb[0, 0, 2], rgb[0, 0, 1]]
        scores = torch.ones((14, 1))
        if mode == "rejected" and second_pass:
            scores.zero_()
        return SimpleNamespace(keypoints=torch.tensor(points[:, None]), scores=scores)

    load = Mock(return_value=SimpleNamespace(predict=predict))
    monkeypatch.setattr(court.CourtKeypointPredictor, "load_from_checkpoint", load)
    clip = SimpleNamespace(
        camera_ids=["cam0"],
        num_frames=3,
        width=100,
        height=100,
        media_path=lambda camera: tmp_path / "fake.mp4",
    )
    roots = RuntimePathRoots(*([tmp_path] * 7))
    settings = StaticCourtSettings(
        Path("model.ckpt"),
        3,
        0.3,
        6,
        10,
        5,
        {"cam0": None if mode == "full" else 0.25},
        None if mode == "disabled" else 2.0,
    )
    if mode == "rejected":
        with pytest.raises(ValueError, match="temporal support"):
            court.observe_static_court(
                clip,
                tmp_path,
                resolver=PathResolver(roots),
                settings=settings,
                device="cpu",
            )
        assert (
            json.loads((tmp_path / "cam0_court_refined.json").read_text())["status"]
            == "rejected"
        )
    else:
        observations, _, diagnostics = court.observe_static_court(
            clip,
            tmp_path,
            resolver=PathResolver(roots),
            settings=settings,
            device="cpu",
        )
        target = expected + ([1.0, 2.0] if mode == "refined" else [0.0, 0.0])
        np.testing.assert_allclose(observations[0, 0] * 100, target, atol=1e-4)
        assert diagnostics[0]["source_roi_xyxy"] == (
            court.court_extent_roi(base, expected, (100, 100), 2.0)
            if mode == "refined"
            else base
        )
        with np.load(tmp_path / "cam0_court_samples.npz") as saved:
            np.testing.assert_allclose(saved["keypoints_px"][0], target, atol=1e-4)
    if mode in ("refined", "rejected"):
        assert len(seen) == 6 and seen[0] != seen[3]
        initial = json.loads((tmp_path / "cam0_court_initial.json").read_text())
        refined = json.loads((tmp_path / "cam0_court_refined.json").read_text())
        assert initial["homography"] == refined["initial_homography"]
        with np.load(tmp_path / "cam0_court_initial_samples.npz") as saved:
            np.testing.assert_allclose(saved["keypoints_px"][0], expected, atol=1e-4)
    else:
        assert len(seen) == 3
        assert not (tmp_path / "cam0_court_initial_samples.npz").exists()
    assert capture.read.call_count == 3
    capture.release.assert_called_once()
    load.assert_called_once()

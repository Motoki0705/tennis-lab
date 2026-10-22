"""Visible-GT errors, threshold boundaries and strict CLI modes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.court_detection.visualization.presentation.hybrid_fit_config import (
    HybridFitConfig,
)
from src.tasks.court_detection.visualization.presentation.hybrid_fit_search import (
    FitErrors,
    HybridFitSearch,
    measure_errors,
)
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize(
    "errors,accepted",
    [
        (FitErrors(10, 3, "ok"), True),
        (FitErrors(10.01, 2.99, "ok"), True),
        (FitErrors(9.99, 3, "ok"), False),
        (FitErrors(10, 3.01, "ok"), False),
        (FitErrors(None, 0, "ok"), False),
        (FitErrors(12, None, "joint_optimization_failed"), False),
        (FitErrors(12, 1, "ambiguous_candidates"), False),
        (FitErrors(np.inf, 1, "ok"), False),
        (FitErrors(12, np.nan, "ok"), False),
    ],
)
def test_search_criteria(errors, accepted) -> None:
    assert HybridFitSearch(10, 3, 300, 20).accepts(errors) == accepted


def test_measure_errors_uses_identical_visible_original_pixel_points(
    hybrid_prediction,
) -> None:
    prediction, _ = hybrid_prediction
    geometry = prediction.homography
    truth = geometry.best.projected.copy()
    visible = np.arange(14) < 10
    truth[~visible] = np.nan
    errors = measure_errors(prediction, truth, visible)
    assert errors.kp_only_error == pytest.approx(np.hypot(25, 8))
    assert errors.hybrid_error == 0
    truth[visible] += [3, 4]
    assert measure_errors(prediction, truth, visible).hybrid_error == 5
    with pytest.raises(ValueError, match="visible GT"):
        measure_errors(prediction, truth, np.zeros(14, dtype=bool))
    failed = replace(
        prediction, homography=replace(geometry, status="ambiguous_candidates")
    )
    errors = measure_errors(failed, truth, visible)
    assert errors.hybrid_error is None
    assert not HybridFitSearch(10, 3, 300, 20).accepts(errors)
    no_baseline = replace(
        prediction,
        homography=replace(geometry, kp_only=replace(geometry.kp_only, matrix=None)),
    )
    assert measure_errors(no_baseline, truth, visible).kp_only_error is None


def _config(overrides: list[str] | None = None):
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks/court_detection/configs"),
    ):
        return compose(config_name="visualize_hybrid_fit", overrides=overrides or [])


def test_config_defaults() -> None:
    config = _config()
    runtime = HybridFitConfig.from_config(config)
    assert runtime.mode == "search"
    assert runtime.search == HybridFitSearch(10, 3, 300, 20)
    assert (runtime.width, runtime.height, runtime.fps) == (1920, 1080, 30)
    assert (
        runtime.output_dir
        == PROJECT_ROOT / "outputs" / config.run.output_dir / "presentation"
    )
    assert (
        runtime.checkpoint
        == PROJECT_ROOT / "ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt"
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        (["mode=render"], "sample_id"),
        (["mode=unknown"], "mode"),
        (["device=cpu"], "search requires cuda"),
        (["input.kind=image", "input.path=frame.png"], "mode=render"),
        (["input.path=frame.png"], "must be null"),
        (["sample_id=sample"], "only valid"),
        (["mode=render", "sample_id=../escape"], "filename stem"),
        (["search.max_samples=0"], "positive"),
        (["search.top_k=0"], "positive"),
        (["search.max_hybrid_error_px=-1"], "nonnegative"),
        (["output.width=961"], "even"),
        (["output.fps=0"], "positive"),
    ],
)
def test_config_rejects_inconsistent_modes_and_budgets(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        HybridFitConfig.from_config(_config(overrides))


def test_config_render_modes_and_no_unknown_keys() -> None:
    runtime = HybridFitConfig.from_config(
        _config(["mode=render", "sample_id=example", "device=cpu"])
    )
    assert runtime.sample_id == "example"
    runtime = HybridFitConfig.from_config(
        _config(["mode=render", "input.kind=image", "input.path=frame.png"])
    )
    assert runtime.image_path == PROJECT_ROOT / "data/frame.png"
    config = OmegaConf.create(OmegaConf.to_container(_config(), resolve=True))
    config.search.unused = True
    with pytest.raises(ValueError, match="exactly"):
        HybridFitConfig.from_config(config)

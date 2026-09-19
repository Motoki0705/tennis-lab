"""Train-only fitting, production label isolation and unweighted comparisons."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tasks.slcs.configuration import SLCSDataRuntimeConfig
from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.dataset import SLCSDataConfig
from src.tasks.slcs.evaluation.ball_baseline import (
    TrainBallMean,
    _fit_mean,
    _label_windows,
)
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _labels() -> dict[str, np.ndarray]:
    # Two overlapping windows and a separate camera observation. Last frame pads.
    target = np.array(
        [[[0, 0, 0], [2, 4, 6]], [[2, 4, 6], [0, 0, 0]], [[4, 8, 12], [0, 0, 0]]],
        dtype=np.float32,
    )
    return {
        "scene_ids": np.array(["train/c@a@0", "train/c@a@1", "train/c@b@0"]),
        "video_ids": np.array(["train", "train", "train"]),
        "clip_ids": np.array(["train/c"] * 3),
        "camera_ids": np.array(["a", "a", "b"]),
        "window_start": np.array([0, 1, 0]),
        "window_length": np.array([2, 1, 1]),
        "frame_idx": np.array([[0, 1], [1, -1], [0, -1]]),
        "padding_mask": np.array([[False, False], [False, True], [False, True]]),
        "target_ball_position": target,
        "ball_mask": np.array([[True, True], [True, False], [True, False]]),
        "ball_weight": np.array([[0.25, 0.75], [0.75, 0], [1.0, 0]], dtype=np.float32),
    }


def test_weighted_mean_deduplicates_overlaps_but_keeps_cameras() -> None:
    report = _fit_mean(_labels(), {"train"})
    np.testing.assert_allclose(report["constant_normalized"], [2.75, 5.5, 8.25])
    assert report["positive_weight_count"] == 3
    assert report["unique_nonpadding_count"] == 3
    assert report["weight_sum"] == 2
    np.testing.assert_allclose(
        report["constant_m"], np.array([2.75, 5.5, 8.25]) * COURT_COORD_SCALE_XYZ
    )


@pytest.mark.parametrize("key", ["target_ball_position", "ball_mask", "ball_weight"])
def test_conflicting_duplicate_labels_are_rejected(key: str) -> None:
    labels = _labels()
    if key == "ball_mask":
        labels[key][1, 0] = False
        labels["ball_weight"][1, 0] = 0
    else:
        labels[key][1, 0] *= 0.5
    with pytest.raises(ValueError, match="Inconsistent duplicate"):
        _fit_mean(labels, {"train"})


def test_invalid_and_zero_weight_labels_do_not_fit_but_duplicates_are_checked() -> None:
    labels = _labels()
    labels["ball_mask"][0, 0] = False
    labels["ball_weight"][0, 0] = 0
    labels["ball_weight"][[0, 1], [1, 0]] = 0
    result = _fit_mean(labels, {"train"})
    np.testing.assert_array_equal(result["constant_normalized"], [4, 8, 12])
    assert result["unique_valid_count"] == 2
    labels["target_ball_position"][1, 0] += 1
    with pytest.raises(ValueError, match="Inconsistent duplicate"):
        _fit_mean(labels, {"train"})


@pytest.mark.parametrize(
    "failure", ["nan", "negative", "above_one", "empty", "padding", "leak"]
)
def test_invalid_fit_inputs_fail_loudly(failure: str) -> None:
    labels = _labels()
    if failure == "nan":
        labels["target_ball_position"][2, 1, 0] = np.nan
    elif failure == "negative":
        labels["ball_weight"][0, 0] = -1
    elif failure == "above_one":
        labels["ball_weight"][0, 0] = 1.1
    elif failure == "empty":
        labels["ball_weight"][:] = 0
    elif failure == "padding":
        labels["ball_mask"][2, 1] = True
    else:
        labels["video_ids"][0] = "val"
    with pytest.raises(ValueError):
        _fit_mean(labels, {"train"})


@pytest.fixture
def runtime(
    synthetic_dataset: SLCSDataIndex,
    synthetic_split_file: Path,
    data_config: SLCSDataConfig,
) -> SLCSDataRuntimeConfig:
    return SLCSDataRuntimeConfig(
        dataset_root=synthetic_dataset.root,
        split_file=synthetic_split_file,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        overfit=False,
        pipeline=data_config,
    )


def test_fit_reads_only_train_without_dino_and_retains_quality(
    runtime: SLCSDataRuntimeConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.slcs.evaluation import ball_baseline

    def no_dino(*args: object, **kwargs: object) -> None:
        pytest.fail("Label-only fitting must not read DINO caches")

    monkeypatch.setattr("src.tasks.slcs.data.dataset.load_dino_tokens", no_dino)
    calls = []

    def labels(config: SLCSDataRuntimeConfig, split: str) -> dict[str, np.ndarray]:
        calls.append(split)
        assert config.pipeline.quality == runtime.pipeline.quality
        assert split == "train", "Fitting must never inspect val/test labels"
        return _label_windows(config, split)

    monkeypatch.setattr(ball_baseline, "_label_windows", labels)
    baseline = TrainBallMean.fit(runtime)
    assert calls == ["train"]
    assert baseline.fit_report["positive_weight_count"] > 0
    assert baseline.runtime.pipeline.require_dino
    assert baseline.fit_report["label_view_overrides"] == {
        "require_dino": False,
        "augment": False,
    }


@pytest.mark.parametrize("overfit,skip", [(True, False), (False, True)])
def test_unreliable_split_modes_rejected(
    runtime: SLCSDataRuntimeConfig, overfit: bool, skip: bool
) -> None:
    runtime = replace(
        runtime,
        overfit=overfit,
        pipeline=replace(runtime.pipeline, on_incomplete="skip" if skip else "error"),
    )
    with pytest.raises(ValueError, match="overfit|on_incomplete"):
        TrainBallMean.fit(runtime)


def test_comparison_uses_physical_unweighted_occurrences_and_flags_train(
    runtime: SLCSDataRuntimeConfig,
) -> None:
    labels = _labels()
    fit = _fit_mean(labels, {"train"})
    baseline = TrainBallMean(runtime, {"train": "train"}, fit)
    # Zero confidence remains included in the headline score, unlike fitting.
    labels["ball_weight"][0, 0] = 0
    expected = {key: value.copy() for key, value in labels.items()}
    labels["pred_ball_position"] = labels["target_ball_position"].copy()
    labels["pred_ball_position"][0, 0] += [1, 2, 3]
    distance = float(np.linalg.norm(np.array([1, 2, 3]) * COURT_COORD_SCALE_XYZ))
    result = baseline.compare(
        labels,
        expected=expected,
        split="train",
        domains={"train": "domain"},
        headline_error_m=distance / 4,
    )
    assert result["in_sample"] is True
    assert {row["group_type"] for row in result["rows"]} == {"all", "domain", "video"}
    row = result["rows"][0]
    assert row["valid_window_occurrences"] == 4
    assert row["model_error_m"] == pytest.approx(distance / 4)
    target = labels["target_ball_position"][labels["ball_mask"]]
    error = np.linalg.norm(
        (target - fit["constant_normalized"]) * COURT_COORD_SCALE_XYZ, axis=-1
    ).mean()
    assert row["train_mean_error_m"] == pytest.approx(error)
    with pytest.raises(ValueError, match="leakage"):
        baseline.compare(
            labels,
            expected=expected,
            split="val",
            domains={"train": "domain"},
            headline_error_m=distance / 4,
        )
    labels["ball_weight"][0, 0] = 0.1
    with pytest.raises(ValueError, match="production split labels"):
        baseline.compare(
            labels,
            expected=expected,
            split="train",
            domains={"train": "domain"},
            headline_error_m=distance / 4,
        )


def test_changed_validation_labels_never_change_train_constant(
    runtime: SLCSDataRuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.slcs.evaluation import ball_baseline

    before = TrainBallMean.fit(runtime)
    original_loader = ball_baseline._label_windows

    def changed(config: SLCSDataRuntimeConfig, split: str) -> dict[str, np.ndarray]:
        result = original_loader(config, split)
        if split == "val":
            result["target_ball_position"] += 100
        return result

    monkeypatch.setattr(ball_baseline, "_label_windows", changed)
    after = TrainBallMean.fit(runtime)
    assert (
        before.fit_report["constant_normalized"]
        == after.fit_report["constant_normalized"]
    )
    expected = after.expected_labels("val")
    arrays = {key: value.copy() for key, value in expected.items()}
    arrays["pred_ball_position"] = arrays["target_ball_position"].copy()
    result = after.compare(
        arrays,
        expected=expected,
        split="val",
        domains=dict.fromkeys(arrays["video_ids"].tolist(), "held-out"),
        headline_error_m=0,
    )
    assert not result["in_sample"]
    assert result["rows"][0]["train_mean_error_m"] > 100

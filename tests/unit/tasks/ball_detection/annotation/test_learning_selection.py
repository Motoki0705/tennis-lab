import copy

import pytest
import torch

from src.tasks.ball_detection.evaluation.coordinates import (
    CoordinateMetrics,
    selection_score,
)
from src.tasks.ball_detection.training.annotation.contracts import (
    BalancedSelectionPolicy,
    BallSelectionPolicy,
    LearningRuntime,
    SourceRegressionLimit,
)
from src.tasks.ball_detection.training.annotation.selection import rejection_reasons


def comparison(*, before: float = 2., after: float = 1., meiji_after: float = 1.) -> dict:
    def report(common: float, other: float) -> dict:
        metrics = CoordinateMetrics((1, 2))
        for step in (1, 2):
            for source, error in (("tracknet", common), ("meiji", other)):
                metrics.add(dict(clip_id=[source], frame_step=[step], start=[0], common_evaluation=[source == "tracknet"],
                    source=[source], frame_indices=torch.arange(32)[None], position_valid=torch.ones(1, 32, dtype=torch.bool)),
                    torch.full((1, 32), error))
        result: dict = metrics.report()
        return result
    return dict(baseline=dict(evaluation=report(before, before)), candidate=dict(evaluation=report(after, meiji_after)))


def policy(**changes) -> BallSelectionPolicy:
    return BallSelectionPolicy(**dict(minimum_common_frames_per_fps=32, maximum_common_error_px=1.,
        minimum_improvement_px=1., maximum_full_regression_px=0., source_regression_limits=(
            SourceRegressionLimit(source="meiji", minimum_frames_per_fps=32, maximum_regression_px=0.),), **changes))


def test_exact_policy_boundaries_pass_and_source_guard_is_independent() -> None:
    assert rejection_reasons(comparison(), policy()) == ()
    assert rejection_reasons(comparison(meiji_after=2.5), policy()) == ("meiji regression exceeds the agreed maximum",)


@pytest.mark.parametrize("change,expected", [
    ({"minimum_common_frames_per_fps": 33}, "support"),
    ({"maximum_common_error_px": .99}, "exceeds"),
    ({"minimum_improvement_px": 1.01}, "improvement"),
    ({"source_regression_limits": (SourceRegressionLimit(source="chat_annotation", minimum_frames_per_fps=1, maximum_regression_px=0.),)}, "chat_annotation support"),
])
def test_missing_support_and_failed_conditions_reject(change: dict, expected: str) -> None:
    agreed = policy().model_copy(update=change)
    assert any(expected in reason for reason in rejection_reasons(comparison(), agreed))


def test_full_regression_is_checked_even_when_common_improves() -> None:
    reasons = rejection_reasons(comparison(meiji_after=4.), policy())
    assert "Full coordinate regression exceeds the agreed maximum" in reasons
    assert "meiji regression exceeds the agreed maximum" in reasons


def test_missing_one_source_fps_cannot_use_the_other_fps_mean() -> None:
    raw = copy.deepcopy(comparison())
    group = raw["candidate"]["evaluation"]["scopes"]["full"]["by_source"]["meiji"]
    group["by_frame_step"]["2"]["observed_frames"] = 0
    group["macro_mean_error_px"] = None
    assert rejection_reasons(raw, policy()) == ("meiji support is below the agreed minimum",)


def test_production_policy_has_no_numerical_defaults() -> None:
    with pytest.raises(ValueError):
        BallSelectionPolicy()
    raw = policy().model_dump(mode="json")
    raw["minimum_improvement_px"] = float("nan")
    with pytest.raises(ValueError):
        BallSelectionPolicy.model_validate(raw)


@pytest.mark.parametrize("changes", [{"precision": "bf16"}, {"jpeg_decoder": "nvjpeg"}, {"pin_memory": True}, {"compile_mode": "default"}])
def test_cpu_fixture_runtime_cannot_silently_select_cuda_features(changes: dict) -> None:
    values = dict(device="cpu", precision="fp32", num_workers=0, pin_memory=False, prefetch_factor=1,
        cpu_threads=1, compile_mode="off", jpeg_decoder="opencv", image_prefetch=False)
    with pytest.raises(ValueError):
        LearningRuntime(**(values | changes))


def cohort_report(existing: float, new: float) -> dict:
    metrics = CoordinateMetrics((1, 2), {"old": "existing", "new": "new"})
    for step in (1, 2):
        for name, error, count in (("old", existing, 32), ("new", new, 8)):
            metrics.add(dict(clip_id=[name], frame_step=[step], start=[0], common_evaluation=[False],
                source=["chat_annotation"], frame_indices=torch.arange(count)[None],
                position_valid=torch.ones(1, count, dtype=torch.bool)), torch.full((1, count), error))
    return metrics.report()


def test_cohorts_have_equal_weight_even_when_sizes_differ() -> None:
    report = cohort_report(2., 4.)
    assert selection_score(report, "balanced_existing_new") == 3.
    assert selection_score(report, "full") == pytest.approx(2.4)
    assert report["scopes"]["common"]["macro_mean_error_px"] is None


@pytest.mark.parametrize("old,new,accepted", [(2., 1., True), (1., 2.5, True), (2., 2., False), (2.1, 0., False)])
def test_balanced_adoption_requires_strict_improvement_and_no_old_regression(old: float, new: float, accepted: bool) -> None:
    raw = dict(split="test", test_usage="final_comparison", baseline=dict(evaluation=cohort_report(2., 2.)),
               candidate=dict(evaluation=cohort_report(old, new)))
    agreed = BalancedSelectionPolicy(minimum_frames_per_fps=1)
    assert (not rejection_reasons(raw, agreed)) is accepted
    raw["split"] = "val"
    with pytest.raises(ValueError, match="final test comparison"):
        rejection_reasons(raw, agreed)

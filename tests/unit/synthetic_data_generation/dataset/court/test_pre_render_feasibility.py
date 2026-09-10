"""Impossible geometric release gates fail before invoking the GPU renderer."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.dataset.court.contracts import CourtDatasetPlan
from src.synthetic_data_generation.dataset.court.rendering import nht
from src.synthetic_data_generation.dataset.court.rendering.nht import (
    CourtNHTRenderer,
    CourtPreRenderEvaluation,
    validate_pre_render_feasibility,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderClient


def _plan(*, minimum_frames: int = 3, minimum_fraction: float = 0.75) -> CourtDatasetPlan:
    return cast(CourtDatasetPlan, SimpleNamespace(
        scene_id="B02",
        proposal_count=4,
        policy=SimpleNamespace(minimum_accepted_frames=minimum_frames, minimum_accepted_fraction=minimum_fraction),
        groups=tuple(SimpleNamespace(trajectory_group_id=f"group-{i}") for i in range(2)),
        samples=tuple(SimpleNamespace(sample_id=f"sample-{i}", trajectory_group_id=f"group-{i // 2}", camera=SimpleNamespace(camera_id=f"camera-{i}")) for i in range(4)),
    ))


def _evaluation(*, rejected: tuple[int, ...] = (), modes: tuple[str, ...] = ("full", "near_full", "partial", "full"), ambiguous: tuple[int, ...] = ()) -> CourtPreRenderEvaluation:
    return cast(CourtPreRenderEvaluation, SimpleNamespace(
        rejected_sample_ids=tuple(f"sample-{i}" for i in rejected),
        projections=tuple(SimpleNamespace(camera_id=f"camera-{i}", courts=(SimpleNamespace(coverage_mode=mode, renderer_visible=None),)) for i, mode in enumerate(modes) if i not in ambiguous),
    ))


@pytest.mark.parametrize(("plan", "evaluation", "message"), [
    (_plan(minimum_frames=5), _evaluation(), "candidate_count=4 < required_frames=5"),
    (_plan(minimum_fraction=0.9), _evaluation(rejected=(3,)), r"maximum_accepted_fraction=3/4.*required_fraction=0.9"),
    (_plan(minimum_frames=1, minimum_fraction=0.1), _evaluation(rejected=(2, 3)), "groups_with_zero_candidates=.*group-1"),
    (_plan(), _evaluation(rejected=(0,), modes=("full", "near_full", "partial", "partial")), "missing_geometric_coverage=.*full"),
    (_plan(), _evaluation(rejected=(0,), modes=("full", "near_full", "partial", "partial"), ambiguous=(0,)), "missing_geometric_coverage=.*full"),
])
def test_impossible_plan_fails_before_render_or_attempt_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    plan: CourtDatasetPlan, evaluation: CourtPreRenderEvaluation, message: str,
) -> None:
    monkeypatch.setattr(nht, "_validate_plan_alignment", lambda *_args: None)
    monkeypatch.setattr(nht, "validate_pre_render_plan", lambda *_args, **_kwargs: evaluation)
    client = Mock(spec=NHTRenderClient)
    renderer = CourtNHTRenderer(executable="nht-render", client=client, environment={}, timeout_seconds=1)
    attempt = tmp_path / "attempt"
    with pytest.raises(ValueError, match=message):
        renderer.render(plan=plan, scene=cast(StandardSceneExport, SimpleNamespace(scene_id="B02")), attempt_root=attempt, attempt_token="attempt", alignment=cast(AlignmentResult, object()))
    client.render.assert_not_called()
    assert not attempt.exists()


def test_feasible_geometry_accepts_exact_threshold_without_claiming_visibility() -> None:
    # Rejected camera has no projection (near/far ambiguity); use camera IDs,
    # not positional zip, to count surviving groups and geometric coverage.
    validate_pre_render_feasibility(_plan(), _evaluation(rejected=(3,), ambiguous=(3,)))


def test_all_impossible_gates_are_reported_together() -> None:
    with pytest.raises(ValueError) as error:
        validate_pre_render_feasibility(_plan(), _evaluation(rejected=(0, 1, 2, 3)))
    message = str(error.value)
    assert "candidate_count=0 < required_frames=3" in message
    assert "required_fraction=0.75" in message
    assert "group-0" in message and "group-1" in message
    assert "missing_geometric_coverage=['full', 'near_full', 'partial']" in message

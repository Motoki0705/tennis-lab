"""Chronological count propagation and explicit discontinuities."""

from typing import Any

import pytest

from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject, MatchSettings
from src.tennis_scene.clip_studio.score_chain import (
    score_views,
    set_clip_score,
    set_match,
)
from src.utils.tennis_scoring import ScoreCount, SoftTennisRules, TennisRules


def score_edit(
    before: dict[str, list[int]] | None = None,
    *,
    winner: str | None = None,
    status: str | None = None,
    server: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "status": status or ("confirmed" if winner else "auto"),
        "before_source": "chain" if before is None else "anchor",
        "server": server,
        "server_source": "auto" if server is None else "manual",
        "winner": winner,
        "note": note,
    }
    if status != "not_point":
        data["before"] = before
    return data


@pytest.mark.parametrize("stop", ["unknown", "multiple_points"])
def test_recompute_chronological_chain_up_to_stop(
    two_camera_project: ClipStudioProject, stop: str
) -> None:
    project = two_camera_project
    project.match = MatchSettings("A", "B", SoftTennisRules(7), "A")
    first, gap, second, third, blocked, next_clip = [
        Clip(name, index, index + 0.5)
        for index, name in enumerate(
            ("first", "gap", "second", "third", "blocked", "next")
        )
    ]
    project.clips = [next_clip, third, second, first, blocked, gap]
    set_clip_score(
        project, gap.name, score_edit(status="not_point", server="B", note="中断")
    )
    set_clip_score(
        project, first.name, score_edit({"games": [0, 0], "points": [2, 1]}, winner="A")
    )
    assert gap.score is not None and gap.score.before is None
    assert second.score is not None and second.score.status == "auto"
    assert second.score.before == ScoreCount(points=(3, 1))
    set_clip_score(project, second.name, score_edit(winner="A", note="映像"))
    set_clip_score(
        project, third.name, score_edit(winner="B", server="B", note="映像2")
    )
    set_clip_score(project, blocked.name, score_edit(status=stop, note="判定保留"))
    unchanged = blocked.score

    set_clip_score(
        project, first.name, score_edit({"games": [0, 0], "points": [2, 1]}, winner="B")
    )

    assert second.score.before == ScoreCount(points=(2, 2))
    assert (second.score.status, second.score.winner, second.score.note) == (
        "confirmed",
        "A",
        "映像",
    )
    assert third.score is not None and third.score.before == ScoreCount(points=(3, 2))
    assert (third.score.server, third.score.winner, third.score.note) == (
        "B",
        "B",
        "映像2",
    )
    assert blocked.score == unchanged
    assert score_views(project)["blocked"]["discontinuous"]
    assert next_clip.score is None
    with pytest.raises(ValueError, match="clip 'next'.*手入力"):
        set_clip_score(project, next_clip.name, score_edit(winner="A"))
    set_clip_score(
        project,
        next_clip.name,
        score_edit({"games": [1, 1], "points": [0, 0]}, winner="A"),
    )


def test_anchor_discontinuity_and_clearing_leave_saved_counts_in_place(
    two_camera_project: ClipStudioProject,
) -> None:
    project = two_camera_project
    project.match = MatchSettings("A", "B", SoftTennisRules(7), "A")
    first = project.clips[0]
    second, anchor = Clip("second", 4, 5), Clip("anchor", 5, 6)
    project.clips.extend([second, anchor])
    opening = {"games": [0, 0], "points": [0, 0]}
    set_clip_score(project, first.name, score_edit(opening, winner="A"))
    set_clip_score(project, second.name, score_edit(winner="B"))
    set_clip_score(
        project,
        anchor.name,
        score_edit({"games": [1, 0], "points": [0, 1]}, winner="A"),
    )
    saved_anchor = anchor.score
    assert score_views(project)["anchor"]["discontinuous"]
    set_clip_score(project, first.name, score_edit(opening, winner="B"))
    assert second.score is not None and second.score.before == ScoreCount(points=(0, 1))
    assert anchor.score == saved_anchor
    saved_second = second.score
    set_clip_score(project, first.name, None)
    assert first.score is None and second.score == saved_second
    assert score_views(project)["second"]["discontinuous"]
    with pytest.raises(ValueError, match="手入力"):
        set_clip_score(project, second.name, score_edit(winner="B"))


def test_hard_next_set_uses_tiebreak_opening_side_and_preserves_manual_override(
    two_camera_project: ClipStudioProject,
) -> None:
    project = two_camera_project
    project.match = MatchSettings("A", "B", TennisRules(3, False, "tie_break"), "A")
    first = project.clips[0]
    second = Clip("second", 4, 5)
    project.clips.append(second)
    set_clip_score(
        project,
        first.name,
        score_edit(
            {"sets": [0, 0], "games": [6, 6], "points": [6, 5]}, winner="A", server="B"
        ),
    )
    assert first.score is not None and first.score.server == "B"
    assert second.score is not None and second.score.before == ScoreCount(sets=(1, 0))
    assert second.score.server == "B"
    set_match(project, MatchSettings("A", "B", TennisRules(3, False, "tie_break"), "B"))
    assert first.score.server == "B"
    assert score_views(project)["second"]["chain_available"]
    updated = project.clips[1].score
    assert updated is not None and updated.server == "A"
    set_clip_score(
        project,
        second.name,
        score_edit({"sets": [1, 0], "games": [0, 0], "points": [0, 0]}),
    )
    project.clips.remove(first)
    set_clip_score(
        project,
        second.name,
        score_edit({"sets": [1, 0], "games": [1, 0], "points": [2, 0]}),
    )
    assert project.clips[0].score is not None and project.clips[0].score.server == "B"


def test_hard_disconnected_anchor_needs_observed_service_order(
    two_camera_project: ClipStudioProject,
) -> None:
    project = two_camera_project
    project.match = MatchSettings("A", "B", TennisRules(3, False, "tie_break"), "A")
    first = project.clips[0]
    second = Clip("second", 4, 5)
    project.clips.append(second)
    count = {"sets": [1, 0], "games": [1, 0], "points": [1, 0]}
    with pytest.raises(ValueError, match="サーブ側を計算できません"):
        set_clip_score(project, first.name, score_edit(count, winner="A"))
    set_clip_score(project, first.name, score_edit(count, winner="A", server="A"))
    assert second.score is not None and second.score.server == "A"

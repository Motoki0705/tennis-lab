"""Clip-name scoped, all-or-nothing CSV interchange."""

import pytest

from src.tennis_scene.clip_studio.project import (
    Clip,
    ClipScore,
    ClipStudioProject,
    MatchSettings,
)
from src.tennis_scene.clip_studio.score_csv import export_scores, import_scores
from src.utils.tennis_scoring import ScoreCount, SoftTennisRules, TennisRules

HEADER = "clip_name,status,sets_a,sets_b,games_a,games_b,points_a,points_b,server,winner,note\n"


def test_csv_round_trip_preserves_values_and_defines_import_sources(
    two_camera_project: ClipStudioProject,
) -> None:
    project = two_camera_project
    project.match = MatchSettings("青", "赤", SoftTennisRules(7), "A")
    clip = project.clips[0]
    clip.score = ClipScore(
        "confirmed",
        ScoreCount(games=(1, 0), points=(3, 1)),
        "chain",
        "B",
        "auto",
        "A",
        '映像,「確認」\n"note"',
    )
    project.clips.extend(
        [
            Clip(
                "gap",
                4,
                5,
                ClipScore("not_point", None, "chain", "A", "manual", None, "中断"),
            ),
            Clip("blank", 5, 6),
        ]
    )
    text = export_scores(project)
    assert text.splitlines()[0] + "\n" == HEADER
    assert "clip_000,confirmed,,,1,0,3,1,B,A," in text
    assert "gap,not_point,,,,,,,A,,中断" in text
    assert "blank" not in text
    import_scores(project, "\ufeff" + text)
    assert export_scores(project) == text
    assert clip.score.note == '映像,「確認」\n"note"'
    assert (clip.score.before_source, clip.score.server_source) == ("anchor", "manual")
    assert project.clips[1].score is not None and project.clips[1].score.before is None
    assert project.clips[2].score is None


@pytest.mark.parametrize(
    "second_row,message",
    [
        ("missing,auto,,,0,0,1,0,A,,", "見つかりません"),
        ("clip_000,auto,,,0,0,1,0,A,,", "重複"),
        ("second,auto,0,0,0,0,1,0,A,,", "空欄"),
        ("second,confirmed,,,0,0,4,1,A,B,", "ゲーム終了後"),
        ("second,auto,,,0,0,1,0,A,", "11 列"),
    ],
)
def test_invalid_late_row_never_applies_earlier_row(
    two_camera_project: ClipStudioProject, second_row: str, message: str
) -> None:
    project = two_camera_project
    project.match = MatchSettings("A", "B", SoftTennisRules(7), "A")
    project.clips.append(Clip("second", 4, 5))
    with pytest.raises(ValueError, match=f"CSV 3 行目.*{message}"):
        import_scores(
            project, HEADER + "clip_000,auto,,,0,0,1,0,A,,\n" + second_row + "\n"
        )
    assert all(clip.score is None for clip in project.clips)


def test_hard_csv_requires_set_pair_and_leaves_omitted_clips_untouched(
    two_camera_project: ClipStudioProject,
) -> None:
    project = two_camera_project
    project.match = MatchSettings("A", "B", TennisRules(3, False, "tie_break"), "A")
    omitted = ClipScore(
        "unknown", ScoreCount(sets=(0, 0)), "anchor", "A", "manual", None, "見えず"
    )
    project.clips.append(Clip("omitted", 4, 5, omitted))
    with pytest.raises(ValueError, match="ヘッダー"):
        import_scores(project, HEADER.replace("sets_a,sets_b", "sets_b,sets_a"))
    with pytest.raises(ValueError, match="sets_a / sets_b"):
        import_scores(project, HEADER + "clip_000,confirmed,,,5,3,3,2,A,A,\n")
    import_scores(project, HEADER + "clip_000,confirmed,1,0,5,3,3,2,B,A,\n")
    assert project.clips[0].score is not None
    assert project.clips[0].score.before == ScoreCount(
        sets=(1, 0), games=(5, 3), points=(3, 2)
    )
    assert project.clips[1].score == omitted

"""Point transitions, canonical counts and service order for both sports."""

import pytest

from src.utils.tennis_scoring import (
    ScoreCount,
    ScoringRules,
    Side,
    SoftTennisRules,
    TennisRules,
    advance_point,
    first_server_after,
    format_points,
    match_over,
    server_at,
    validate_before,
)

SOFT = SoftTennisRules(7)
HARD = TennisRules(3, False, "tie_break")


def test_soft_deuce_and_next_game() -> None:
    before = ScoreCount(games=(0, 1), points=(3, 3))
    assert format_points(SOFT, before) == "デュース"
    assert server_at(SOFT, before, "A") == "B"
    advantage = advance_point(SOFT, before, "B")
    assert advantage.points == (3, 4)
    assert format_points(SOFT, advantage) == "アドバンテージ B"
    assert advance_point(SOFT, advantage, "A") == before
    after = advance_point(SOFT, advantage, "B")
    assert after == ScoreCount(games=(0, 2))
    assert server_at(SOFT, after, "A") == "A"


def test_soft_final_keeps_serve_phase_through_repeated_deuce() -> None:
    before = ScoreCount(games=(3, 3), points=(6, 6))
    assert server_at(SOFT, before, "B") == "B"
    advantage = advance_point(SOFT, before, "A")
    assert server_at(SOFT, advantage, "B") == "B"
    second_deuce = advance_point(SOFT, advantage, "B")
    assert second_deuce.points == (7, 7)
    assert format_points(SOFT, second_deuce) == "デュース"
    assert server_at(SOFT, second_deuce, "B") == "A"
    second_advantage = advance_point(SOFT, second_deuce, "B")
    assert advance_point(SOFT, second_advantage, "A") == before
    won = advance_point(SOFT, second_advantage, "B")
    assert won == ScoreCount(games=(3, 4))
    assert match_over(SOFT, won)
    assert format_points(SOFT, won) == "試合終了"


def test_hard_standard_and_no_ad_end_differently_at_40_all() -> None:
    assert format_points(HARD, ScoreCount(points=(2, 1), sets=(0, 0))) == "30–15"
    deuce = ScoreCount(games=(2, 0), points=(3, 3), sets=(0, 0))
    advantage = advance_point(HARD, deuce, "B")
    assert advantage.points == (3, 4)
    assert format_points(HARD, advantage) == "AD B"
    assert advance_point(HARD, advantage, "A") == deuce
    no_ad = TennisRules(3, True, "tie_break")
    assert format_points(no_ad, deuce) == "40–40"
    assert advance_point(no_ad, deuce, "B") == ScoreCount(games=(2, 1), sets=(0, 0))


def test_hard_tiebreak_serves_one_then_two_and_carries_to_next_set() -> None:
    count = ScoreCount(games=(6, 6), sets=(0, 0))
    sequence: list[tuple[Side, Side]] = [("A", "A"), ("B", "B"), ("A", "B"), ("B", "A")]
    for winner, server in sequence:
        assert server_at(HARD, count, "A") == server
        count = advance_point(HARD, count, winner)

    deuce = ScoreCount(games=(6, 6), points=(6, 6), sets=(0, 0))
    advantage = advance_point(HARD, deuce, "B")
    second_deuce = advance_point(HARD, advantage, "A")
    assert second_deuce.points == (7, 7)
    assert format_points(HARD, second_deuce) == "7–7"
    assert server_at(HARD, second_deuce, "A") == "B"
    assert advance_point(HARD, advance_point(HARD, second_deuce, "B"), "A") == deuce
    next_set = advance_point(HARD, advantage, "B")
    assert next_set == ScoreCount(sets=(0, 1))
    assert first_server_after(advantage, next_set, "A") == "B"


def test_hard_match_tiebreak_replaces_deciding_set_and_needs_two_point_margin() -> None:
    rules = TennisRules(5, True, "match_tie_break")
    # At 1–1 this game is still part of a regular set in best-of-five.
    assert advance_point(
        rules, ScoreCount(points=(3, 3), sets=(1, 1)), "A"
    ) == ScoreCount(games=(1, 0), sets=(1, 1))
    deuce = ScoreCount(sets=(2, 2), points=(9, 9))
    advantage = advance_point(rules, deuce, "A")
    assert advantage.points == (10, 9)
    assert not match_over(rules, advantage)
    after = advance_point(rules, advantage, "A")
    assert after == ScoreCount(sets=(3, 2))
    assert match_over(rules, after)


def test_hard_advantage_deciding_set_continues_at_six_all() -> None:
    rules = TennisRules(3, False, "advantage")
    before = ScoreCount(sets=(1, 1), games=(6, 6), points=(3, 0))
    after = advance_point(rules, before, "A")
    assert after == ScoreCount(sets=(1, 1), games=(7, 6))
    won = advance_point(
        rules, ScoreCount(sets=(1, 1), games=(7, 6), points=(3, 2)), "A"
    )
    assert won == ScoreCount(sets=(2, 1))


@pytest.mark.parametrize(
    "rules,before,message",
    [
        (SOFT, ScoreCount(games=(4, 1)), "試合終了後"),
        (SOFT, ScoreCount(points=(4, 2)), "ゲーム終了後"),
        (SOFT, ScoreCount(points=(4, 4)), "正規化"),
        (HARD, ScoreCount(sets=(0, 2)), "試合終了後"),
        (HARD, ScoreCount(sets=(0, 0), games=(6, 4)), "セット終了後"),
        (HARD, ScoreCount(sets=(0, 0), games=(6, 6), points=(8, 8)), "正規化"),
        (
            TennisRules(3, False, "match_tie_break"),
            ScoreCount(sets=(1, 1), games=(0, 1)),
            "0–0",
        ),
    ],
)
def test_impossible_counts_cannot_start_a_point(
    rules: ScoringRules, before: ScoreCount, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_before(rules, before)

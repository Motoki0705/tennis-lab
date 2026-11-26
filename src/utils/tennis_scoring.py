"""Pure point, game, set and service-order rules for two fixed sides."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Side = Literal["A", "B"]
CountPair = tuple[int, int]
FinalSet = Literal["tie_break", "match_tie_break", "advantage"]
ZERO = (0, 0)


def require_side(value: object, name: str) -> Side:
    if value == "A":
        return "A"
    if value == "B":
        return "B"
    raise ValueError(f"{name} は A または B を指定してください。")


def other_side(side: Side) -> Side:
    return "B" if side == "A" else "A"


@dataclass(frozen=True)
class SoftTennisRules:
    match_games: int

    def __post_init__(self) -> None:
        if (
            type(self.match_games) is not int
            or self.match_games < 1
            or not self.match_games % 2
        ):
            raise ValueError("マッチのゲーム数は正の奇数を指定してください。")


@dataclass(frozen=True)
class TennisRules:
    best_of: int
    no_ad: bool
    final_set: FinalSet

    def __post_init__(self) -> None:
        if type(self.best_of) is not int or self.best_of not in (3, 5):
            raise ValueError("硬式のセット数は 3 または 5 を指定してください。")
        if type(self.no_ad) is not bool:
            raise ValueError("no_ad は真偽値を指定してください。")
        if self.final_set not in ("tie_break", "match_tie_break", "advantage"):
            raise ValueError(f"未対応の最終セット形式です: {self.final_set!r}")

    @property
    def sets_to_win(self) -> int:
        return (self.best_of + 1) // 2


ScoringRules = SoftTennisRules | TennisRules


@dataclass(frozen=True)
class ScoreCount:
    games: CountPair = ZERO
    points: CountPair = ZERO
    sets: CountPair | None = None

    def __post_init__(self) -> None:
        for name, pair in (
            ("games", self.games),
            ("points", self.points),
            ("sets", self.sets),
        ):
            if name == "sets" and pair is None:
                continue
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(type(value) is not int or value < 0 for value in pair)
            ):
                raise ValueError(f"{name} は A / B の非負整数を指定してください。")


def match_over(rules: ScoringRules, count: ScoreCount) -> bool:
    if isinstance(rules, SoftTennisRules):
        return max(count.games) >= (rules.match_games + 1) // 2
    return count.sets is not None and max(count.sets) >= rules.sets_to_win


def _deciding_set(rules: TennisRules, count: ScoreCount) -> bool:
    return count.sets == (rules.sets_to_win - 1, rules.sets_to_win - 1)


def _match_tie_break(rules: TennisRules, count: ScoreCount) -> bool:
    return _deciding_set(rules, count) and rules.final_set == "match_tie_break"


def _advantage_set(rules: TennisRules, count: ScoreCount) -> bool:
    return _deciding_set(rules, count) and rules.final_set == "advantage"


def _point_target(rules: ScoringRules, count: ScoreCount) -> int:
    if isinstance(rules, SoftTennisRules):
        last_game = rules.match_games // 2
        return 7 if count.games == (last_game, last_game) else 4
    if _match_tie_break(rules, count):
        return 10
    return 7 if count.games == (6, 6) and not _advantage_set(rules, count) else 4


def _margin(rules: ScoringRules, target: int) -> int:
    return 1 if isinstance(rules, TennisRules) and rules.no_ad and target == 4 else 2


def _won(pair: CountPair, target: int, margin: int = 2) -> bool:
    return max(pair) >= target and abs(pair[0] - pair[1]) >= margin


def _normalise(points: CountPair, target: int) -> CountPair:
    deuce = target - 1
    # In a tie-break, removing four total points preserves the two-point service blocks.
    excess = max(0, min(points) - deuce)
    removed = excess - (excess % 2) if target > 4 else excess
    return points[0] - removed, points[1] - removed


def validate_before(rules: ScoringRules, count: ScoreCount) -> None:
    """Reject states that cannot start a point, including noncanonical deuce counts."""
    if isinstance(rules, SoftTennisRules) != (count.sets is None):
        raise ValueError("セット数は硬式でのみ A / B ともに指定してください。")
    if match_over(rules, count):
        raise ValueError("試合終了後のカウントです。")
    if isinstance(rules, TennisRules):
        if _match_tie_break(rules, count) and count.games != ZERO:
            raise ValueError("マッチタイブレークのゲーム数は 0–0 を指定してください。")
        if _won(count.games, 6) or (
            max(count.games) > 6 and not _advantage_set(rules, count)
        ):
            raise ValueError("セット終了後のゲームカウントです。")
    target = _point_target(rules, count)
    if _won(count.points, target, _margin(rules, target)):
        raise ValueError("ゲーム終了後のポイントカウントです。")
    if count.points != _normalise(count.points, target):
        raise ValueError("デュース以降は正規化したポイントカウントを指定してください。")


def _increment(pair: CountPair, side: Side) -> CountPair:
    return (pair[0] + 1, pair[1]) if side == "A" else (pair[0], pair[1] + 1)


def advance_point(rules: ScoringRules, before: ScoreCount, winner: Side) -> ScoreCount:
    """Return the count after one point, including the terminal count of a match."""
    validate_before(rules, before)
    winner = require_side(winner, "得点側")
    target = _point_target(rules, before)
    points = _increment(before.points, winner)
    if not _won(points, target, _margin(rules, target)):
        return replace(before, points=_normalise(points, target))
    games = _increment(before.games, winner)
    if before.sets is not None and (target in (7, 10) or _won(games, 6)):
        return ScoreCount(sets=_increment(before.sets, winner))
    return replace(before, games=games, points=ZERO)


def server_at(rules: ScoringRules, before: ScoreCount, first_server: Side) -> Side:
    """Resolve the point's server from the first server of its set (or soft-tennis match).

    The same transformation in reverse recovers the first server from the point's server.
    """
    validate_before(rules, before)
    require_side(first_server, "最初のサーブ側")
    swaps = sum(before.games)
    if _point_target(rules, before) > 4:
        points_played = sum(before.points)
        # Soft tennis switches after the first two points. Hard tennis switches after one.
        swaps += (points_played + int(isinstance(rules, TennisRules))) // 2
    return other_side(first_server) if swaps % 2 else first_server


def first_server_after(
    before: ScoreCount, after: ScoreCount, first_server: Side
) -> Side:
    """Carry the opening server to the next set after a point transition."""
    if after.sets != before.sets and (sum(before.games) + 1) % 2:
        return other_side(first_server)
    return first_server


def format_points(rules: ScoringRules, count: ScoreCount) -> str:
    if match_over(rules, count):
        return "試合終了"
    validate_before(rules, count)
    a, b = count.points
    target = _point_target(rules, count)
    hard_game = isinstance(rules, TennisRules) and target == 4
    if isinstance(rules, TennisRules) and not hard_game:
        return f"{a}–{b}"
    if min(a, b) >= target - 1 and _margin(rules, target) == 2:
        if a == b:
            return "デュース"
        side = "A" if a > b else "B"
        return f"AD {side}" if hard_game else f"アドバンテージ {side}"
    if hard_game:
        labels = ("0", "15", "30", "40")
        return f"{labels[a]}–{labels[b]}"
    return f"{a}–{b}"


__all__ = [
    "CountPair",
    "FinalSet",
    "ScoreCount",
    "ScoringRules",
    "Side",
    "SoftTennisRules",
    "TennisRules",
    "advance_point",
    "first_server_after",
    "format_points",
    "match_over",
    "other_side",
    "require_side",
    "server_at",
    "validate_before",
]

"""Resolve clip-local scores in timeline order and recompute their dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from src.tennis_scene.clip_studio.project import (
    Clip,
    ClipScore,
    ClipStudioProject,
    MatchSettings,
    count_from_dict,
    count_to_dict,
)
from src.utils.tennis_scoring import (
    ScoreCount,
    Side,
    advance_point,
    first_server_after,
    format_points,
    match_over,
    other_side,
    require_side,
    server_at,
    validate_before,
)


@dataclass(frozen=True)
class Position:
    count: ScoreCount
    first_server: Side  # The opening server of this set, or of the soft-tennis match.

    def server(self, match: MatchSettings) -> Side:
        return server_at(match.rules, self.count, self.first_server)


@dataclass(frozen=True)
class ChainEntry:
    clip: Clip
    incoming: Position | None
    position: Position | None
    linked: bool
    after: Position | None


def _position(
    match: MatchSettings,
    count: ScoreCount,
    incoming: Position | None,
    *,
    observed_server: Side | None = None,
    fallback_first: Side | None = None,
) -> Position | None:
    if count.sets in (None, (0, 0)):
        return Position(count, match.first_server)
    if incoming is not None and count.sets == incoming.count.sets:
        return Position(count, incoming.first_server)
    # The number of games in previous sets is absent from the count. A recorded
    # server can establish the service order when starting a disconnected segment.
    if observed_server is not None:
        return Position(count, server_at(match.rules, count, observed_server))
    return Position(count, fallback_first) if fallback_first is not None else None


def _after(
    match: MatchSettings, position: Position, winner: Side | None
) -> Position | None:
    if winner is None:
        return None
    after = advance_point(match.rules, position.count, winner)
    return Position(
        after, first_server_after(position.count, after, position.first_server)
    )


def _timeline(project: ClipStudioProject) -> list[Clip]:
    return sorted(project.clips, key=lambda clip: clip.start_sec)


def _entries(clips: Iterable[Clip], match: MatchSettings) -> Iterator[ChainEntry]:
    incoming: Position | None = None
    for clip in clips:
        score = clip.score
        position = None
        linked = False
        after = None
        outgoing = None
        if score is not None:
            if score.before is None:  # not_point
                outgoing = incoming
            else:
                linked = incoming is not None and incoming.count == score.before
                position = _position(
                    match, score.before, incoming, observed_server=score.server
                )
                if position is not None:
                    after = _after(match, position, score.winner)
                if linked or score.before_source == "anchor":
                    outgoing = after
        yield ChainEntry(clip, incoming, position, linked, after)
        incoming = outgoing


def score_views(project: ClipStudioProject) -> dict[str, dict[str, Any]]:
    """Presentation data also used by the read-only viewer."""
    match = project.match
    if match is None:
        return {}
    views = {}
    for entry in _entries(_timeline(project), match):
        score = entry.clip.score
        before = entry.position.count if entry.position is not None else None
        after = entry.after
        views[entry.clip.name] = {
            "chain_available": entry.incoming is not None
            and not match_over(match.rules, entry.incoming.count),
            "discontinuous": before is not None
            and score is not None
            and not entry.linked
            and (entry.incoming is not None or score.before_source == "chain"),
            "before_points": format_points(match.rules, before)
            if before is not None
            else None,
            "after": (
                {
                    **count_to_dict(after.count),
                    "points_label": format_points(match.rules, after.count),
                }
                if after is not None
                else None
            ),
        }
    return views


@contextmanager
def _clip_error(clip: Clip) -> Iterator[None]:
    try:
        yield
    except ValueError as error:
        raise ValueError(f"clip '{clip.name}': {error}") from error


def _parse_edit(
    entry: ChainEntry, match: MatchSettings, raw: dict[str, Any]
) -> tuple[ClipScore, Position | None]:
    data = dict(raw)
    incoming = entry.incoming
    position = incoming
    if data.get("status") != "not_point":
        if data.get("before_source") == "chain":
            if "before" not in data or data["before"] is not None:
                raise ValueError(
                    "連鎖で入力する場合は before に null を指定してください。"
                )
            if incoming is None:
                raise ValueError(
                    "直前の終了後カウントがありません。開始前を手入力してください。"
                )
            validate_before(match.rules, incoming.count)
            data["before"] = count_to_dict(incoming.count)
        else:
            count = count_from_dict(data.get("before"))
            validate_before(match.rules, count)
            old = entry.clip.score
            position = _position(
                match,
                count,
                incoming,
                observed_server=(
                    require_side(data.get("server"), "サーブ側")
                    if data.get("server_source") == "manual"
                    else None
                ),
                fallback_first=(
                    entry.position.first_server
                    if entry.position
                    and old
                    and old.before is not None
                    and old.before.sets == count.sets
                    and old.server_source == "auto"
                    else None
                ),
            )
    if data.get("server_source") == "auto":
        if "server" not in data or data["server"] is not None:
            raise ValueError("自動サーブの場合は server に null を指定してください。")
        if position is None or match_over(match.rules, position.count):
            raise ValueError(
                "サーブ側を計算できません。A / B を手動で指定してください。"
            )
        data["server"] = position.server(match)
    return ClipScore.from_dict(data), position


def _recalculate_following(
    clips: Iterator[Clip], match: MatchSettings, position: Position | None
) -> None:
    for clip in clips:
        if position is None:
            return
        score = clip.score
        with _clip_error(clip):
            finished = match_over(match.rules, position.count)
            if score is None:
                if not finished:
                    clip.score = ClipScore(
                        "auto",
                        position.count,
                        "chain",
                        position.server(match),
                        "auto",
                        None,
                        "",
                    )
                return
            if score.status == "not_point":
                if score.server_source == "auto" and not finished:
                    clip.score = replace(score, server=position.server(match))
                continue
            if score.before_source == "anchor" or score.status in (
                "unknown",
                "multiple_points",
            ):
                return
            validate_before(match.rules, position.count)
            clip.score = replace(
                score,
                before=position.count,
                server=position.server(match)
                if score.server_source == "auto"
                else score.server,
            )
            position = _after(match, position, score.winner)


def set_clip_score(
    project: ClipStudioProject, name: str, data: dict[str, Any] | None
) -> None:
    """Apply one edit to a candidate project, including its contiguous score chain."""
    clip = project.clips[project.clip_index_by_name(name)]
    if data is None:
        clip.score = None
        return
    match = project.match
    if match is None:
        raise ValueError("カウントの入力前に試合設定を保存してください。")
    # Evaluate the saved prefix before changing any clip in the candidate.
    timeline = iter(_timeline(project))
    entry = next(item for item in _entries(timeline, match) if item.clip is clip)
    with _clip_error(clip):
        score, position = _parse_edit(entry, match, data)
        clip.score = score
        if score.before is not None and position is not None:
            position = _after(match, position, score.winner)
        elif score.before is not None:
            position = None
    if score.status in ("confirmed", "not_point"):
        _recalculate_following(timeline, match, position)


def set_match(project: ClipStudioProject, match: MatchSettings | None) -> None:
    """Re-evaluate established chains under the new settings, without filling empty clips."""
    old_match = project.match
    project.match = match
    if old_match is None or match is None:
        return
    if (old_match.rules, old_match.first_server) == (match.rules, match.first_server):
        return

    incoming: Position | None = None
    for entry in _entries(_timeline(project), old_match):
        clip = entry.clip
        score = clip.score
        with _clip_error(clip):
            if score is None:
                incoming = None
                continue
            if score.before is None:
                if (
                    score.server_source == "auto"
                    and incoming
                    and not match_over(match.rules, incoming.count)
                ):
                    clip.score = replace(score, server=incoming.server(match))
                continue
            linked = score.before_source == "chain" and entry.linked
            if linked and incoming is None:
                raise ValueError("新しい設定で直前のカウントから連鎖できません。")
            count = incoming.count if linked and incoming is not None else score.before
            validate_before(match.rules, count)
            first = entry.position.first_server if entry.position is not None else None
            if first is not None and old_match.first_server != match.first_server:
                first = other_side(first)
            position = _position(
                match,
                count,
                incoming,
                observed_server=score.server
                if score.server_source == "manual"
                else None,
                fallback_first=first,
            )
            if position is None:
                raise ValueError("新しい設定でサーブ側を計算できません。")
            clip.score = replace(
                score,
                before=count,
                server=position.server(match)
                if score.server_source == "auto"
                else score.server,
            )
            incoming = (
                _after(match, position, score.winner)
                if linked or score.before_source == "anchor"
                else None
            )

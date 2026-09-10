"""Container-level recording timestamps, independent of frame decoding."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import av


class CreationTimeError(ValueError):
    """A container has no usable, timezone-aware creation_time."""


def read_creation_time(video_path: Path) -> datetime:
    """Read only the container creation_time and normalize it to UTC.

    Stream tags, filesystem times and timestamps without a timezone are not
    substitutes for a container recording timestamp.
    """
    try:
        with av.open(str(video_path)) as container:
            raw = container.metadata.get("creation_time")
    except (av.FFmpegError, OSError) as error:
        raise CreationTimeError(f"動画メタデータを取得できません: {error}") from error
    if not raw:
        raise CreationTimeError("コンテナの creation_time がありません")
    try:
        timestamp = datetime.fromisoformat(raw)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("タイムゾーンがありません")
        return timestamp.astimezone(UTC)
    except (ValueError, OverflowError) as error:
        raise CreationTimeError(
            f"creation_time が不正です ({raw!r}): {error}"
        ) from error

"""Container-only timestamp parsing and explicit failure reasons."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.video.metadata import CreationTimeError, read_creation_time


@pytest.mark.parametrize(
    "raw", ["2026-07-09T16:08:33.123Z", "2026-07-10T01:08:33.123+09:00"]
)
def test_container_timestamp_normalized(raw):
    with patch("src.utils.video.metadata.av.open") as open_container:
        container = open_container.return_value.__enter__.return_value
        container.metadata = {"creation_time": raw}
        assert read_creation_time(Path("camera.mp4")) == datetime(
            2026, 7, 9, 16, 8, 33, 123000, tzinfo=UTC
        )
        container.decode.assert_not_called()
        open_container.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "raw",
    [None, "", "invalid", "2026-07-09T16:08:33", "2026-07-09", "2026-02-30T00:00:00Z"],
)
def test_missing_or_invalid_timestamp_is_explicit(raw):
    with patch("src.utils.video.metadata.av.open") as open_container:
        container = open_container.return_value.__enter__.return_value
        container.metadata = {} if raw is None else {"creation_time": raw}
        container.streams = [
            MagicMock(metadata={"creation_time": "2026-07-09T16:08:33Z"})
        ]
        with pytest.raises(CreationTimeError, match="creation_time"):
            read_creation_time(Path("camera.mp4"))


def test_unreadable_container(tmp_path):
    with pytest.raises(CreationTimeError, match="取得できません"):
        read_creation_time(tmp_path / "missing.mp4")

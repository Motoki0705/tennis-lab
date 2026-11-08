"""Selection configuration rejects ambiguous frame and identity requests."""

from __future__ import annotations

import pytest

from src.tennis_scene.visualization.player_view import PlayerViewBundleConfig


@pytest.mark.parametrize(
    "frames", [(0, 0), (-1, 2), (4, 3), (False, 2), (0, 1.5), (0, 1, 2), [0, 2]]
)
def test_rejects_invalid_frame_intervals(frames: object) -> None:
    with pytest.raises(ValueError, match="frames"):
        PlayerViewBundleConfig(frames=frames)  # type: ignore[arg-type]


@pytest.mark.parametrize("identities", [(), (1, 1), (-1,), (True,), (1.5,), [1, 2]])
def test_rejects_ambiguous_id_selection(identities: object) -> None:
    with pytest.raises(ValueError, match="player_ids"):
        PlayerViewBundleConfig(player_ids=identities)  # type: ignore[arg-type]

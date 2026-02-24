"""Resolve dataset splits for labeled, unlabeled, and pseudo pools."""

from __future__ import annotations

from src.tasks.ball_detection.data.type import ClipLayout


def resolve_game_split(layouts: list[ClipLayout], *, train_games: set[str], val_games: set[str], test_games: set[str]) -> dict[str, list[ClipLayout]]:
    """Partition clip layouts by game names."""
    out = {"train": [], "val": [], "test": []}
    for layout in layouts:
        if layout.game_name in train_games:
            out["train"].append(layout)
        elif layout.game_name in val_games:
            out["val"].append(layout)
        elif layout.game_name in test_games:
            out["test"].append(layout)
    return out

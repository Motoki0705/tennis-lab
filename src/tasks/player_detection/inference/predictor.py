"""Player-specific entry point for the shared DINO person detector."""

from __future__ import annotations

from src.submodules.models.dino.person_detector import DinoPersonDetector
from src.tasks.player_detection.inference.checkpoint import (
    PlayerCheckpointInfo,
    inspect_player_checkpoint,
)


class DinoPlayerDetector(DinoPersonDetector):
    """Detect court players from an exported player checkpoint.

    Inherits the deployed BGR preprocessing, strict DINO loading, confidence
    filtering, and original-pixel ``xyxy`` output from ``DinoPersonDetector``.
    A COCO person checkpoint is rejected rather than silently used as a player
    model.
    """

    checkpoint_info: PlayerCheckpointInfo | None = None

    def _load_impl(self) -> None:
        info = inspect_player_checkpoint(self.checkpoint)
        super()._load_impl()
        self.checkpoint_info = info

    def _unload_impl(self) -> None:
        super()._unload_impl()
        self.checkpoint_info = None

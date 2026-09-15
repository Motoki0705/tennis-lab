"""Read-only BLCS generated-dataset scene service for the review UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.base.visualization.review.service import (
    DatasetSceneReviewService,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import (
    validate_blcs_dataset_court_keypoints,
)

# One colour per slot; deliberately spread across hues for multi-object scenes.
BALL_COLORS: tuple[str, ...] = (
    "#e0611f",
    "#2f7fd1",
    "#1f9e6b",
    "#c0392b",
    "#8e5cc4",
    "#d4a017",
    "#0f8b8d",
    "#c85a8e",
    "#5b6b7f",
    "#7a8b2f",
)


class BLCSDatasetReviewService(DatasetSceneReviewService):
    """Catalog and payload assembly over ``data/blcs/<form>/scenes``."""

    task = "blcs"
    entity_kind = "ball"

    def _validate_scene_contract(
        self, scene_path: Path, contract: CourtKeypointContract
    ) -> None:
        validate_blcs_dataset_court_keypoints(
            scene_path.parent.parent,
            contract,
            scene_paths=[scene_path],
        )

    def _entity_joint_count(self) -> int:
        return 1

    def _entity_joint_names(self) -> list[str] | None:
        return None

    def _entity_skeleton(self) -> list[list[int]] | None:
        return None

    def _entity_colors(self) -> tuple[str, ...]:
        return BALL_COLORS

    def _entity_frames_file(self) -> str:
        return "ball_pos_world.npy"

    def _presence_file(self) -> str:
        return "ball_present.npy"

    def _orientation_file(self) -> str:
        return "rotation.npy"

    def _court_net_post_offset(self, meta: dict[str, Any]) -> float | None:
        # BLCS scenes always record their per-scene net post offset.
        if "court_config" not in meta:
            raise ValueError("BLCS scene meta.json is missing court_config.")
        court_config = meta["court_config"]
        if not isinstance(court_config, dict) or "net_post_offset_x" not in court_config:
            raise ValueError(
                "BLCS court_config must contain net_post_offset_x."
            )
        return float(court_config["net_post_offset_x"])

    def _fps(self, meta: dict[str, Any], scene_path: Path) -> float:
        if "fps_out" not in meta:
            raise ValueError(
                f"{scene_path / 'meta.json'}: required key 'fps_out' is missing."
            )
        value = float(meta["fps_out"])
        if not value > 0.0:
            raise ValueError(f"{scene_path / 'meta.json'}: fps_out must be positive.")
        return value


__all__ = ["BALL_COLORS", "BLCSDatasetReviewService"]

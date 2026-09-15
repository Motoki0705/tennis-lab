"""Read-only PLCS generated-dataset scene service for the review UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    validate_dataset_court_keypoint_contract,
)
from src.tasks.base.visualization.review.service import (
    DatasetSceneReviewService,
)
from src.tasks.plcs.court_keypoint_contract import (
    PLCS_GENERATED_DATASET_SCHEMA_ID,
    validate_plcs_court_keypoint_headers,
)
from src.utils.schema.player import (
    COCO17_SKELETON,
    COCO_KP_NAMES,
    NUM_HUMAN_KP,
)

# One colour per slot; deliberately spread across hues for multi-object scenes.
PLAYER_COLORS: tuple[str, ...] = (
    "#1f8a70",
    "#d1623f",
    "#3a6fb0",
    "#c79a12",
    "#8e5cc4",
    "#c0392b",
    "#0f8b8d",
    "#b3548a",
    "#5b6b7f",
    "#7a8b2f",
)


class PLCSDatasetReviewService(DatasetSceneReviewService):
    """Catalog and payload assembly over ``data/plcs/<form>/scenes``."""

    task = "plcs"
    entity_kind = "player"

    def _validate_scene_contract(
        self, scene_path: Path, contract: CourtKeypointContract
    ) -> None:
        validation = validate_dataset_court_keypoint_contract(
            scene_path.parent.parent,
            contract,
            expected_dataset_schema_id=PLCS_GENERATED_DATASET_SCHEMA_ID,
            scene_paths=(scene_path,),
        )
        validate_plcs_court_keypoint_headers(validation, (scene_path,))

    def _entity_joint_count(self) -> int:
        return int(NUM_HUMAN_KP)

    def _entity_joint_names(self) -> list[str] | None:
        return list(COCO_KP_NAMES)

    def _entity_skeleton(self) -> list[list[int]] | None:
        return [[int(a), int(b)] for a, b in COCO17_SKELETON]

    def _entity_colors(self) -> tuple[str, ...]:
        return PLAYER_COLORS

    def _entity_has_orientation(self) -> bool:
        return True

    def _entity_frames_file(self) -> str:
        return "human_kp_3d.npy"

    def _presence_file(self) -> str:
        return "person_present.npy"

    def _orientation_file(self) -> str:
        return "rotation.npy"

    def _court_net_post_offset(self, meta: dict[str, Any]) -> float | None:
        # PLCS scenes omit ``court_config`` and use the standard net post offset.
        if "court_config" not in meta:
            return None
        court_config = meta["court_config"]
        if not isinstance(court_config, dict) or "net_post_offset_x" not in court_config:
            raise ValueError(
                "PLCS court_config must contain net_post_offset_x when present."
            )
        return float(court_config["net_post_offset_x"])

    def _fps(self, meta: dict[str, Any], scene_path: Path) -> float:
        if "fps" not in meta:
            raise ValueError(f"{scene_path / 'meta.json'}: required key 'fps' is missing.")
        value = float(meta["fps"])
        if not value > 0.0:
            raise ValueError(f"{scene_path / 'meta.json'}: fps must be positive.")
        return value


__all__ = ["PLAYER_COLORS", "PLCSDatasetReviewService"]

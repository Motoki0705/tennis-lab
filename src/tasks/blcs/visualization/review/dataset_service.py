"""Read-only BLCS generated-dataset scene service for the review UI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.base.visualization.review.camera import parse_cameras
from src.tasks.base.visualization.review.service import (
    DEFAULT_CAMERA_DEPTH_M,
    DatasetSceneReviewService,
    load_json_object,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import (
    validate_blcs_dataset_court_keypoints,
)
from src.tasks.blcs.visualization.review.inspection import load_inspection

BALL_COLORS: tuple[str, ...] = ("#e0611f",)


class BLCSDatasetReviewService(DatasetSceneReviewService):
    """Catalog and payload assembly over ``data/blcs/<form>/scenes``."""

    task = "blcs"
    entity_kind = "ball"

    def __init__(
        self,
        data_root: str | Path,
        *,
        forms: Sequence[str] | None = None,
        camera_depth: float = DEFAULT_CAMERA_DEPTH_M,
    ) -> None:
        requested = tuple(forms) if forms is not None else ("single_object",)
        if requested != ("single_object",):
            raise ValueError(
                "BLCS review supports only the current single_object dataset."
            )
        super().__init__(data_root, forms=requested, camera_depth=camera_depth)

    def _resolve(self, form: str, scene_id: str) -> tuple[Path, CourtKeypointContract]:
        directory, contract = super()._resolve(form, scene_id)
        if contract.selector != "physical_v1":
            raise ValueError("BLCS review requires physical_v1 CourtKP20 semantics.")
        return directory, contract

    def _validate_scene_contract(
        self, scene_path: Path, contract: CourtKeypointContract
    ) -> None:
        validate_blcs_dataset_court_keypoints(
            scene_path.parent.parent,
            contract,
            scene_paths=[scene_path],
        )
        scalars = load_json_object(scene_path / "scalars.json")
        if type(scalars.get("num_balls")) is not int or scalars["num_balls"] != 1:
            raise ValueError("BLCS single_object requires scalars.num_balls = 1.")

    def _entity_shape(self, scene_path: Path) -> tuple[int, int]:
        array = np.load(
            scene_path / "ball_pos_world.npy", mmap_mode="r", allow_pickle=False
        )
        shape = array.shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError(
                f"BLCS single_object ball_pos_world must have shape (T, 3), got {shape}."
            )
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(
                "BLCS ball_pos_world must contain finite floating metre-valued coordinates."
            )
        return 1, 1

    def inspection(self, form: str, scene_id: str, revision: str) -> dict[str, Any]:
        document = self.scene(form, scene_id, revision)
        directory, _ = self._resolve(form, scene_id)
        meta = load_json_object(directory / "meta.json")
        scalars = load_json_object(directory / "scalars.json")
        evidence = load_inspection(directory, document, meta, parse_cameras(scalars))
        splits: dict[str, int | None] = {}
        membership = []
        for split in ("train", "val", "test"):
            path = directory.parent.parent / f"{split}.txt"
            names = (
                path.read_text(encoding="utf-8").splitlines()
                if path.is_file()
                else None
            )
            splits[split] = len(names) if names is not None else None
            if names is not None and scene_id in names:
                membership.append(split)
        evidence["dataset"] = {
            "scene_count": self._catalog.form(form).scene_count,
            "split_counts": splits,
            "scene_splits": membership,
        }
        if self._catalog.revision(form, scene_id) != document["revision"]:
            raise RuntimeError("Scene changed on disk. Reload the scene.")
        return evidence

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
        if (
            not isinstance(court_config, dict)
            or "net_post_offset_x" not in court_config
        ):
            raise ValueError("BLCS court_config must contain net_post_offset_x.")
        return float(court_config["net_post_offset_x"])

    def _fps(self, meta: dict[str, Any], scene_path: Path) -> float:
        if "fps_out" not in meta:
            raise ValueError(
                f"{scene_path / 'meta.json'}: required key 'fps_out' is missing."
            )
        value = float(meta["fps_out"])
        if not np.isfinite(value) or not value > 0.0:
            raise ValueError(
                f"{scene_path / 'meta.json'}: fps_out must be positive and finite."
            )
        return value


__all__ = ["BALL_COLORS", "BLCSDatasetReviewService"]

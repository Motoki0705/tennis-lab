"""Dataset I/O utilities for PLCS dataset generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.tasks.base.data.dataset_writer import BaseDatasetWriter
from src.tasks.base.generate_dataset import (
    COURT_VIEW_METADATA_KEY,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtViewRecord,
    InvalidCourtKeypointMetadataError,
    extract_court_keypoint_artifact_metadata,
    inject_court_keypoint_artifact_metadata,
    inject_scene_court_keypoint_metadata,
)
from src.tasks.plcs.court_keypoint_contract import plcs_artifact_metadata
from src.tasks.plcs.data.types import PLCSSceneMeta
from src.tasks.plcs.generate_dataset.scene_generator import SceneData
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SceneMetadataDocument:
    value: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.value)


class PLCSDatasetWriter(BaseDatasetWriter):
    """Writes PLCS scene data to disk as npy + json directories."""

    scenes_dir: Path

    def __init__(
        self,
        output_dir: str | Path,
        *,
        court_keypoint_contract: CourtKeypointContract | None = None,
        legacy_metadata_free_v1: bool = False,
    ) -> None:
        if court_keypoint_contract is None and not legacy_metadata_free_v1:
            raise CourtKeypointContractMismatchError(
                "PLCS dataset writing requires an explicit CourtKP20 contract. "
                "Use legacy_metadata_free_v1=True only for deliberate legacy-v1 "
                "metadata-free output."
            )
        if court_keypoint_contract is not None and legacy_metadata_free_v1:
            raise CourtKeypointContractMismatchError(
                "An explicit PLCS CourtKP20 contract and "
                "legacy_metadata_free_v1=True are mutually exclusive."
            )
        self.court_keypoint_contract = court_keypoint_contract
        self.legacy_metadata_free_v1 = legacy_metadata_free_v1
        self.court_keypoint_artifact_metadata = (
            plcs_artifact_metadata(court_keypoint_contract)
            if court_keypoint_contract is not None
            else None
        )
        output_path = Path(output_dir)
        root_metadata = self._validate_existing_root_contract(output_path)
        self._reject_nonempty_existing_dataset(output_path, root_metadata)
        super().__init__(output_dir)

    def _validate_existing_root_contract(
        self,
        output_dir: Path | None = None,
    ) -> dict[str, object] | None:
        """Reject an existing root whose CourtKP20 mode differs from this writer."""
        root = self.output_dir if output_dir is None else output_dir
        path = root / "meta.json"
        if not path.exists():
            return None
        if not path.is_file():
            raise InvalidCourtKeypointMetadataError(
                f"{path}: root metadata must be a JSON file."
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise InvalidCourtKeypointMetadataError(
                f"{path}: invalid JSON root metadata: {error}."
            ) from error
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise InvalidCourtKeypointMetadataError(
                f"{path}: root metadata must be a JSON object with string keys."
            )

        existing = extract_court_keypoint_artifact_metadata(
            value,
            location=str(path),
        )
        if self.legacy_metadata_free_v1:
            if existing is not None or COURT_VIEW_METADATA_KEY in value:
                raise CourtKeypointContractMismatchError(
                    "Legacy metadata-free PLCS output cannot append to a dataset "
                    "with CourtKP20 metadata."
                )
            return cast("dict[str, object]", value)

        expected = self.court_keypoint_artifact_metadata
        if expected is None:
            raise RuntimeError("PLCS writer lost its validated CourtKP20 contract.")
        if existing is None:
            raise CourtKeypointContractMismatchError(
                "Explicit PLCS CourtKP20 output cannot append to an existing "
                "metadata-free dataset root."
            )
        inject_court_keypoint_artifact_metadata(
            value,
            expected,
            location=str(path),
        )
        return cast("dict[str, object]", value)

    @staticmethod
    def _reject_nonempty_existing_dataset(
        output_dir: Path,
        root_metadata: dict[str, object] | None,
    ) -> None:
        """Reject reopen attempts because writer state is never hydrated from disk."""

        def reject() -> None:
            raise FileExistsError(
                "PLCSDatasetWriter does not support reopening a non-empty dataset: "
                f"{output_dir}."
            )

        scenes_dir = output_dir / "scenes"
        if scenes_dir.exists():
            if not scenes_dir.is_dir():
                raise InvalidCourtKeypointMetadataError(
                    f"{scenes_dir}: expected a scene directory."
                )
            if any(scenes_dir.iterdir()):
                reject()

        if root_metadata is None:
            return
        root_path = output_dir / "meta.json"
        scenes = root_metadata.get("scenes", [])
        if not isinstance(scenes, list):
            raise InvalidCourtKeypointMetadataError(
                f"{root_path}.scenes: expected a list."
            )
        if scenes:
            reject()

        stats = root_metadata.get("stats", {})
        if not isinstance(stats, dict) or any(
            not isinstance(key, str) for key in stats
        ):
            raise InvalidCourtKeypointMetadataError(
                f"{root_path}.stats: expected an object with string keys."
            )
        for key, value in stats.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise InvalidCourtKeypointMetadataError(
                    f"{root_path}.stats.{key}: expected a finite non-negative "
                    "number."
                )
            if value != 0:
                reject()

    def _validate_scene_court_keypoints(
        self,
        scene: SceneData,
        *,
        scene_path: Path,
    ) -> tuple[CourtViewRecord, ...]:
        """Validate the complete scene CourtKP20 contract before any scene write."""
        if self.legacy_metadata_free_v1:
            if scene.court_keypoint_contract is not None:
                raise CourtKeypointContractMismatchError(
                    "Legacy metadata-free PLCS output requires the scene CourtKP20 "
                    "contract to be absent."
                )
            for camera_index, camera in enumerate(scene.cameras):
                if camera.court_view is not None:
                    raise CourtKeypointContractMismatchError(
                        "Legacy metadata-free PLCS output requires every camera "
                        f"CourtKP20 record to be absent; camera {camera_index} has one."
                    )
            return ()

        contract = self.court_keypoint_contract
        artifact_metadata = self.court_keypoint_artifact_metadata
        if contract is None or artifact_metadata is None:
            raise RuntimeError("PLCS writer lost its validated CourtKP20 contract.")
        if scene.court_keypoint_contract != contract:
            raise CourtKeypointContractMismatchError(
                f"PLCS writer contract {contract.contract_id!r} does not match "
                "the generated scene contract."
            )

        court_views: list[CourtViewRecord] = []
        for camera_index, camera in enumerate(scene.cameras):
            view = camera.court_view
            if not isinstance(view, CourtViewRecord):
                raise CourtKeypointContractMismatchError(
                    f"PLCS camera {camera_index} is missing a valid CourtKP20 record."
                )
            expected_camera_id = f"camera_{camera_index}"
            if view.camera_id != expected_camera_id:
                raise CourtKeypointContractMismatchError(
                    f"PLCS camera {camera_index} CourtKP20 ID must be "
                    f"{expected_camera_id!r}, got {view.camera_id!r}."
                )
            center = np.asarray(camera.camera_params.get("C"), dtype=np.float64)
            view_center = np.asarray(
                view.camera_center_court_m,
                dtype=np.float64,
            )
            if (
                center.shape != (3,)
                or not np.isfinite(center).all()
                or not np.array_equal(center, view_center)
            ):
                raise CourtKeypointContractMismatchError(
                    f"PLCS camera {camera_index} params.C does not match its "
                    "CourtKP20 camera center."
                )
            court_views.append(view)

        # Validate contract IDs and record uniqueness before the scene directory
        # exists. The returned document is intentionally discarded here; the
        # final scene metadata is injected after PLCS metadata construction.
        inject_scene_court_keypoint_metadata(
            {},
            artifact_metadata,
            court_views,
            location=str(scene_path / "meta.json"),
        )
        return tuple(court_views)

    def save_scene(self, scene: SceneData) -> Path:
        """Save a single scene as a directory with npy + json files.

        Args:
            scene: Scene data to save.

        Returns:
            Path: Path to saved scene directory.
        """
        dirname = str(scene.meta["scene_id"])
        scene_path: Path = self.scenes_dir / dirname
        root_metadata = self._validate_existing_root_contract()
        if not self.legacy_metadata_free_v1 and root_metadata is None:
            raise CourtKeypointContractMismatchError(
                "Explicit PLCS CourtKP20 output requires compatible root "
                "metadata to be published before saving a scene."
            )
        court_views = self._validate_scene_court_keypoints(
            scene,
            scene_path=scene_path,
        )

        # Create metadata using dataclass (with optional Pydantic validation)
        meta_dict = {
            "scene_id": scene.meta["scene_id"],
            "motion_source": scene.meta["motion_source"],
            "motion_category": scene.meta["motion_category"],
            "gender": scene.meta["gender"],
            "fps": scene.meta["fps"],
            "num_frames": scene.meta["num_frames"],
            "initial_position": scene.meta["initial_position"],
            "initial_yaw": scene.meta["initial_yaw"],
            "num_cameras_sampled": scene.meta["num_cameras_sampled"],
            "num_cameras": len(scene.cameras),
            "court_coordinate_normalization": (
                court_coordinate_normalization_metadata()
            ),
            "track_instances": scene.track_instances,
        }

        meta = PLCSSceneMeta(**meta_dict)
        meta_document: dict[str, object] = meta.to_dict()
        if self.court_keypoint_artifact_metadata is not None:
            meta_document = inject_scene_court_keypoint_metadata(
                meta_document,
                self.court_keypoint_artifact_metadata,
                court_views,
                location=str(scene_path / "meta.json"),
            )

        arrays: dict[str, np.ndarray] = {
            "position": np.asarray(scene.position),
            "rotation": np.asarray(scene.rotation),
            "canonical_pose_3d": np.asarray(scene.canonical_pose_3d),
        }

        scalars: dict[str, Any] = {
            "num_cameras": len(scene.cameras),
            "num_persons": scene.num_persons,
        }

        if scene.person_present is not None:
            arrays["person_present"] = np.asarray(scene.person_present, dtype=bool)

        # Store pre-computed COCO17 world joints when available
        if scene.human_kp_3d is not None:
            arrays["human_kp_3d"] = np.asarray(scene.human_kp_3d).astype(np.float32)

        camera_metas = []
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"
            arrays[f"{prefix}human_kp_uv"] = cam.human_kp_uv.astype(np.float32)
            arrays[f"{prefix}human_kp_vis"] = cam.human_kp_vis.astype(bool)
            arrays[f"{prefix}human_visibility_ratio"] = np.array(
                cam.human_visibility_ratio, dtype=np.float32
            )
            scalars[f"{prefix}params"] = cam.camera_params
            arrays[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
            arrays[f"{prefix}court_kp_vis"] = cam.court_kp_vis.astype(bool)
            arrays[f"{prefix}court_visibility_count"] = np.array(
                cam.court_visibility_count,
                dtype=np.float32,
            )

            camera_metas.append(
                {
                    "human_visibility_ratio": float(cam.human_visibility_ratio),
                    "court_visibility_count": float(cam.court_visibility_count),
                }
            )

        scene_path.mkdir(parents=True, exist_ok=True)
        self._write_scene_files(
            scene_path,
            _SceneMetadataDocument(meta_document),
            scalars,
            arrays,
        )

        self.scene_records.append(
            {
                "file": dirname,
                "scene_id": scene.meta["scene_id"],
                "motion_category": scene.meta["motion_category"],
                "num_frames": int(scene.meta["num_frames"]),
                "num_cameras_sampled": scene.meta["num_cameras_sampled"],
                "num_cameras": len(scene.cameras),
                "cameras": camera_metas,
            }
        )
        self.scene_counter += 1

        return scene_path

    def save_meta_json(self, config: dict | None = None) -> None:
        """Save root metadata with the exact task-qualified CourtKP20 marker."""
        self._validate_existing_root_contract()
        total_cameras = sum(record["num_cameras"] for record in self.scene_records)
        avg_cameras = (
            total_cameras / len(self.scene_records) if self.scene_records else 0
        )
        document: dict[str, object] = {
            "generated_at": datetime.now().isoformat(),
            "config": config or {},
            "stats": {
                "total_scenes": len(self.scene_records),
                "total_cameras": total_cameras,
                "avg_cameras_per_scene": avg_cameras,
            },
            "scenes": self.scene_records,
        }
        path = self.output_dir / "meta.json"
        if self.court_keypoint_artifact_metadata is not None:
            document = inject_court_keypoint_artifact_metadata(
                document,
                self.court_keypoint_artifact_metadata,
                location=str(path),
            )

        # Never replace a compatible published root with a metadata-free or
        # partially written document while versioned scenes already exist.
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(document, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        logger.info(
            "meta.json saved: %s scenes, %s cameras",
            len(self.scene_records),
            total_cameras,
        )

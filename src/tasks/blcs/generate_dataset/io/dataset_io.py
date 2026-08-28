"""Dataset writer for BLCS dataset generation.

Saves scene data as a directory per scene with structure:
- meta.json: scene metadata
- scalars.json: scalar values (num_cameras, rally_length, end_reason,
  camera parameters)
- {key}.npy: array data files (ball_pos_world, ball_pos_norm, etc.)

Each camera produces per-camera npy files (cam_{i}_ball_uv.npy, etc.)
and its parameters are stored in scalars.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.data.dataset_writer import BaseDatasetWriter
from src.tasks.base.generate_dataset import (
    CameraCourtViewError,
    CourtKeypointArtifactMetadata,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtViewRecord,
    DatasetCourtKeypointContract,
    build_court_view_record,
    inject_court_keypoint_artifact_metadata,
    inject_scene_court_keypoint_metadata,
    resolve_court_keypoint_contract,
)
from src.tasks.base.generate_dataset import (
    validate_dataset_court_keypoint_contract as validate_dataset_court_keypoints,
)
from src.tasks.blcs.data.types import (
    BLCSSceneMeta,
)
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
    validate_court_coordinate_normalization,
)

logger = logging.getLogger(__name__)

BLCS_DATASET_SCHEMA_ID = "blcs_generated_dataset_v2"


@dataclass(frozen=True, slots=True)
class _SceneMetadataDocument:
    document: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.document)


def validate_blcs_dataset_court_keypoints(
    dataset_root: str | Path,
    contract: CourtKeypointContract,
    *,
    scene_paths: list[Path],
) -> DatasetCourtKeypointContract:
    """Validate BLCS root/scene/camera metadata including scalar camera slots."""
    result = validate_dataset_court_keypoints(
        dataset_root,
        contract,
        expected_dataset_schema_id=BLCS_DATASET_SCHEMA_ID,
        scene_paths=scene_paths,
    )
    if result.legacy_metadata_free:
        return result

    records_by_scene = {scene.scene_id: scene.court_views for scene in result.scenes}
    for scene_path in scene_paths:
        records = records_by_scene[scene_path.name]
        scene_metadata_path = scene_path / "meta.json"
        scene_metadata = json.loads(scene_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(scene_metadata, dict):
            raise CourtKeypointContractMismatchError(
                f"{scene_metadata_path}: expected a JSON object."
            )
        scene_num_cameras = scene_metadata.get("num_cameras")
        if type(scene_num_cameras) is not int:
            raise CourtKeypointContractMismatchError(
                f"{scene_metadata_path}: num_cameras must be an int; got "
                f"{scene_num_cameras!r}."
            )

        scalars_path = scene_path / "scalars.json"
        raw = json.loads(scalars_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CourtKeypointContractMismatchError(
                f"{scalars_path}: expected a JSON object."
            )
        num_cameras = raw.get("num_cameras")
        if type(num_cameras) is not int:
            raise CourtKeypointContractMismatchError(
                f"{scalars_path}: num_cameras must be an int; got "
                f"{num_cameras!r}."
            )
        if scene_num_cameras != len(records):
            raise CourtKeypointContractMismatchError(
                f"{scene_metadata_path}: num_cameras must exactly match "
                f"court_keypoint_views ({scene_num_cameras!r} != "
                f"{len(records)})."
            )
        if num_cameras != len(records):
            raise CourtKeypointContractMismatchError(
                f"{scalars_path}: num_cameras must exactly match "
                f"court_keypoint_views ({num_cameras!r} != {len(records)})."
            )
        expected_parameter_slots = {
            f"cam_{index}_params" for index in range(len(records))
        }
        actual_parameter_slots = {
            key
            for key in raw
            if key.startswith("cam_") and key.endswith("_params")
        }
        if actual_parameter_slots != expected_parameter_slots:
            raise CourtKeypointContractMismatchError(
                f"{scalars_path}: camera parameter slots must exactly match "
                f"court_keypoint_views IDs; expected "
                f"{sorted(expected_parameter_slots)!r}, got "
                f"{sorted(actual_parameter_slots)!r}."
            )
        for index, record in enumerate(records):
            expected_id = f"cam_{index}"
            if record.camera_id != expected_id:
                raise CourtKeypointContractMismatchError(
                    f"{scalars_path}: camera slot {index} requires stable ID "
                    f"{expected_id!r}; got {record.camera_id!r}."
            )
            params = raw.get(f"cam_{index}_params")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError as error:
                    raise CourtKeypointContractMismatchError(
                        f"{scalars_path}: {expected_id}_params must contain a "
                        "JSON object with a numeric C vector."
                    ) from error
            if not isinstance(params, dict) or "C" not in params:
                raise CourtKeypointContractMismatchError(
                    f"{scalars_path}: {expected_id}_params.C is required."
                )
            try:
                parameter_record = build_court_view_record(
                    camera_id=expected_id,
                    camera_center_court_m=params["C"],
                    contract=contract,
                )
            except CameraCourtViewError as error:
                raise CourtKeypointContractMismatchError(
                    f"{scalars_path}: invalid {expected_id}_params.C: {error}"
                ) from error
            if parameter_record.camera_center_court_m != (
                record.camera_center_court_m
            ):
                raise CourtKeypointContractMismatchError(
                    f"{scalars_path}: {expected_id} camera center does not exactly "
                    "match CourtKP metadata."
                )
    return result


class BLCSDatasetWriter(BaseDatasetWriter):
    """Writes BLCS scene data to disk as npy + json directories."""

    scenes_dir: Path

    def __init__(
        self,
        output_dir: str | Path,
        *,
        court_keypoint_contract: CourtKeypointContract | str = "physical_v1",
    ) -> None:
        output_path = Path(output_dir)
        if output_path.exists() and any(output_path.iterdir()):
            raise FileExistsError(
                "Refusing to write a BLCS dataset into a non-empty directory: "
                f"{output_path}."
            )
        self.court_keypoint_contract = (
            court_keypoint_contract
            if isinstance(court_keypoint_contract, CourtKeypointContract)
            else resolve_court_keypoint_contract(court_keypoint_contract)
        )
        self.court_keypoint_artifact_metadata = (
            CourtKeypointArtifactMetadata.from_contract(
                self.court_keypoint_contract,
                dataset_schema_id=BLCS_DATASET_SCHEMA_ID,
            )
        )
        super().__init__(output_dir)
        # Publish the root contract immediately so even incrementally written
        # datasets never expose versioned scenes under a metadata-free root.
        self.save_meta_json(config={})

    def _court_views(self, scene: BLCSSceneData) -> tuple[CourtViewRecord, ...]:
        records: list[CourtViewRecord] = []
        for index, camera in enumerate(scene.cameras):
            expected_id = f"cam_{index}"
            record = camera.court_view
            if record is None:
                if self.court_keypoint_contract.camera_view_semantics:
                    raise CourtKeypointContractMismatchError(
                        f"BLCS scene {scene.scene_id!r} camera {expected_id!r} is "
                        "missing its required camera-view CourtKP20 record."
                    )
                record = build_court_view_record(
                    camera_id=expected_id,
                    camera_center_court_m=camera.camera_params["C"],
                    contract=self.court_keypoint_contract,
                )
            if record.camera_id != expected_id:
                raise CourtKeypointContractMismatchError(
                    f"BLCS scene {scene.scene_id!r} camera slot {index} requires "
                    f"stable ID {expected_id!r}; got {record.camera_id!r}."
                )
            if record.contract_id != self.court_keypoint_contract.contract_id:
                raise CourtKeypointContractMismatchError(
                    f"BLCS scene {scene.scene_id!r} camera {expected_id!r} uses "
                    f"{record.contract_id!r}, expected "
                    f"{self.court_keypoint_contract.contract_id!r}."
                )
            recorded_center = np.asarray(record.camera_center_court_m, dtype=np.float64)
            parameter_center = np.asarray(camera.camera_params["C"], dtype=np.float64)
            if parameter_center.shape != (3,) or not np.array_equal(
                recorded_center, parameter_center
            ):
                raise CourtKeypointContractMismatchError(
                    f"BLCS scene {scene.scene_id!r} camera {expected_id!r} "
                    "metadata center does not exactly match camera params C."
                )
            records.append(record)
        return tuple(records)

    def _build_scene_meta(self, scene: BLCSSceneData) -> BLCSSceneMeta:
        scene_meta_dict: dict[str, Any] = {
            "scene_id": scene.scene_id,
            "initial_from_cell": scene.initial_from_cell,
            "initial_from_side": scene.initial_from_side,
            "rally_length": scene.rally_length,
            "end_reason": scene.end_reason,
            "winner_side": scene.winner_side,
            "shots": scene.shots,
            "fps_out": scene.fps_out,
            "sim_fps": scene.sim_fps,
            "num_frames": int(scene.ball_pos_world.shape[0]),
            "num_cameras_sampled": scene.num_cameras_sampled,
            "num_cameras": len(scene.cameras),
            "court_coordinate_normalization": (
                court_coordinate_normalization_metadata()
            ),
            "physics_config": scene.physics_config_dict,
            "court_config": scene.court_config_dict,
            "track_instances": scene.track_instances,
        }

        return BLCSSceneMeta(**scene_meta_dict)

    def _append_camera_arrays(
        self,
        arrays: dict[str, np.ndarray],
        scalars: dict[str, Any],
        scene: BLCSSceneData,
    ) -> list[dict[str, float]]:
        camera_records: list[dict[str, float]] = []
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"
            scalars[f"{prefix}params"] = cam.camera_params
            arrays[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
            arrays[f"{prefix}court_kp_vis"] = cam.court_kp_vis.astype(bool)
            arrays[f"{prefix}court_visibility_count"] = np.array(
                cam.court_visibility_count,
                dtype=np.float32,
            )
            # Ball-specific per-camera arrays (not shared with PLCS).
            arrays[f"{prefix}ball_uv"] = cam.ball_uv.astype(np.float32)
            arrays[f"{prefix}ball_vis"] = cam.ball_vis.astype(bool)
            arrays[f"{prefix}ball_visibility_ratio"] = np.array(
                cam.ball_visibility_ratio,
                dtype=np.float32,
            )
            camera_records.append(
                {
                    "ball_visibility_ratio": float(cam.ball_visibility_ratio),
                    "court_visibility_count": float(cam.court_visibility_count),
                }
            )
        return camera_records

    def save_scene(self, scene: BLCSSceneData) -> Path:
        """Save a BLCS scene (rally) as a directory with npy + json files."""
        dirname = scene.scene_id
        scene_path: Path = self.scenes_dir / dirname
        court_views = self._court_views(scene)
        scene_meta = self._build_scene_meta(scene)
        scene_meta_document = inject_scene_court_keypoint_metadata(
            scene_meta.to_dict(),
            self.court_keypoint_artifact_metadata,
            court_views,
            location=str(scene_path / "meta.json"),
        )

        arrays: dict[str, np.ndarray] = {
            "ball_pos_world": scene.ball_pos_world.numpy(),
            "ball_pos_norm": scene.ball_pos_norm.numpy(),
            "ball_vel_world": scene.ball_vel_world.numpy(),
            "ball_vel_norm": scene.ball_vel_norm.numpy(),
        }
        if scene.ball_present is not None:
            arrays["ball_present"] = scene.ball_present.cpu().numpy()
        scalars: dict[str, Any] = {
            "num_cameras": len(scene.cameras),
            "num_balls": scene.num_balls,
            "rally_length": scene.rally_length,
            "end_reason": scene.end_reason,
        }
        camera_records = self._append_camera_arrays(arrays, scalars, scene)

        scene_path.mkdir(parents=True, exist_ok=True)
        self._write_scene_files(
            scene_path,
            _SceneMetadataDocument(scene_meta_document),
            scalars,
            arrays,
        )

        self.scene_records.append(
            {
                "file": dirname,
                "scene_id": scene.scene_id,
                "rally_length": scene.rally_length,
                "end_reason": scene.end_reason,
                "winner_side": scene.winner_side,
                "num_frames": int(scene.ball_pos_world.shape[0]),
                "num_cameras_sampled": scene.num_cameras_sampled,
                "num_cameras": len(scene.cameras),
                "cameras": camera_records,
            }
        )
        self.scene_counter += 1
        return scene_path

    def save_meta_json(self, config: dict | None = None) -> None:
        """Write the root document with the exact task-qualified CourtKP marker."""
        super().save_meta_json(config=config)
        path = self.output_dir / "meta.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("BLCS root meta.json must contain an object.")
        document = inject_court_keypoint_artifact_metadata(
            raw,
            self.court_keypoint_artifact_metadata,
            location=str(path),
        )
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_scene(
    filepath: str | Path,
    *,
    court_keypoint_contract: CourtKeypointContract | str = "physical_v1",
) -> dict:
    """Load a scene from a npy + json scene directory.

    Args:
        filepath: Path to the scene directory.

    Returns:
        dict: Scene data with:
            - meta: parsed metadata
            - ball_pos_world, ball_pos_norm, ball_vel_world: 3D data
            - num_cameras: number of cameras
            - cameras: list of camera data dicts
    """
    scene_dir = Path(filepath)
    keypoint_contract = (
        court_keypoint_contract
        if isinstance(court_keypoint_contract, CourtKeypointContract)
        else resolve_court_keypoint_contract(court_keypoint_contract)
    )
    if scene_dir.parent.name != "scenes":
        raise ValueError(
            "BLCS scene contract validation requires "
            "<dataset>/scenes/<scene>."
        )
    court_keypoint_result = validate_blcs_dataset_court_keypoints(
        scene_dir.parent.parent,
        keypoint_contract,
        scene_paths=[scene_dir],
    )

    with open(scene_dir / "meta.json") as f:
        scene_meta = json.load(f)
    validate_court_coordinate_normalization(scene_meta, artifact=f"Scene {scene_dir}")
    with open(scene_dir / "scalars.json") as f:
        scalars = json.load(f)

    num_cameras = int(scalars["num_cameras"])

    scene_views = court_keypoint_result.scenes[0].court_views
    if scene_views and len(scene_views) != num_cameras:
        raise ValueError(
            "BLCS scene court_keypoint_views count must equal scalars.num_cameras."
        )
    cameras = []
    for i in range(num_cameras):
        prefix = f"cam_{i}_"
        params = scalars[f"{prefix}params"]
        if isinstance(params, str):
            params = json.loads(params)
        cam_data = {
            "camera_id": scene_views[i].camera_id if scene_views else f"cam_{i}",
            "court_view": scene_views[i] if scene_views else None,
            "params": params,
            "ball_uv": np.load(scene_dir / f"{prefix}ball_uv.npy"),
            "ball_vis": np.load(scene_dir / f"{prefix}ball_vis.npy"),
            "ball_visibility_ratio": float(
                np.load(scene_dir / f"{prefix}ball_visibility_ratio.npy")
            ),
            "court_kp_uv": np.load(scene_dir / f"{prefix}court_kp_uv.npy"),
            "court_kp_vis": np.load(scene_dir / f"{prefix}court_kp_vis.npy"),
            "court_visibility_count": float(
                np.load(scene_dir / f"{prefix}court_visibility_count.npy")
            ),
        }
        cameras.append(cam_data)

    result = {
        "meta": scene_meta,
        "court_keypoint_contract": court_keypoint_result.contract,
        "court_keypoint_legacy_metadata_free": (
            court_keypoint_result.legacy_metadata_free
        ),
        "ball_pos_world": np.load(scene_dir / "ball_pos_world.npy"),
        "ball_pos_norm": np.load(scene_dir / "ball_pos_norm.npy"),
        "ball_vel_world": np.load(scene_dir / "ball_vel_world.npy"),
        "ball_vel_norm": np.load(scene_dir / "ball_vel_norm.npy"),
        "num_cameras": num_cameras,
        "cameras": cameras,
    }
    ball_present_path = scene_dir / "ball_present.npy"
    if ball_present_path.exists():
        result["ball_present"] = np.load(ball_present_path)
    if "num_balls" not in scalars:
        raise ValueError(
            "BLCS scene is incompatible: required scalar num_balls is missing."
        )
    result["num_balls"] = int(scalars["num_balls"])
    return result

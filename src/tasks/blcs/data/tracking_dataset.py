"""Config-aware canonical-scene adapter for lifecycle BLCS tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data import include_evaluation_reference_camera
from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment
from src.tasks.base.data.observation_tracking import track_multiview_observations
from src.tasks.base.data.scene_dataset import CameraSelection, Scene
from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    PHYSICAL_V1_SELECTOR,
    CourtReferenceFrameError,
    CourtReferenceFrameProvenance,
    court_points_physical_to_target,
    court_vectors_physical_to_target,
)
from src.tasks.blcs.configuration import parse_court_keypoint_contract
from src.tasks.blcs.data.court_view import (
    align_blcs_court_array,
    blcs_reference_sample_fields,
    blcs_track_query_reference_contract_document,
    collate_blcs_reference_fields,
    court_views_by_scene,
    resolve_blcs_sample_court_frame,
    validate_blcs_dataset_court_keypoints,
)
from src.tasks.blcs.data.observation_candidates import (
    PhysicalObservationCandidates,
    align_clean_observations_after_tracking,
    build_physical_observation_candidates,
)
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingDetectionAugmentation,
)

BLCS_TRACKING_KEYS = (
    "scene_format_version",
    "ball_uv",
    "ball_vis",
    "court_kp",
    "court_vis",
    "padding_mask",
    "target_position",
    "target_velocity",
    "target_presence",
    "target_instance_id",
    "target_slot_mask",
    "clean_ball_uv",
    "clean_ball_vis",
    "candidate_gt_index",
    "court_reference_provenance",
)


class BLCSTrackingDataset(CanonicalTrackingDataset):
    """Corrupt physical detections, track per camera, and pack targets separately."""

    def __init__(
        self,
        *,
        reference_camera_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        config = kwargs.get("config")
        self.reference_camera_id = reference_camera_id
        self.court_keypoint_contract = parse_court_keypoint_contract(config)
        self.track_query_reference_document = (
            blcs_track_query_reference_contract_document(
                config,
                self.court_keypoint_contract,
            )
        )
        scene_dir = Path(kwargs["scene_dir"])
        split_file = Path(kwargs["split_file"])
        court_dataset = validate_blcs_dataset_court_keypoints(
            scene_dir=scene_dir,
            split_file=split_file,
            contract=self.court_keypoint_contract,
        )
        self._court_views_by_scene = court_views_by_scene(court_dataset)
        super().__init__(**kwargs)
        if self.num_queries is None:
            raise ValueError("BLCS tracking requires model.num_queries.")
        if not self.pack_to_query_slots:
            raise ValueError(
                "BLCS tracking requires data.lifecycle.pack_to_query_slots=true."
            )
        if self.observation_tracking_config.min_common_keypoints != 1:
            raise ValueError(
                "BLCS point tracking requires data.association.min_common_keypoints=1."
            )
        if self.observation_tracking_config.cost_reduction != "mean":
            raise ValueError(
                "BLCS point tracking requires data.association.cost_reduction='mean'."
            )
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self.tracking_augmentation = BLCSTrackingDetectionAugmentation(
            data_cfg["augmentation"],
            num_slots=self.num_queries,
        )

    def build_sample(self, scene: Scene) -> dict[str, Any]:
        if self.num_queries is None:
            raise RuntimeError("BLCS tracking dataset lost its fixed query width.")
        num_queries = self.num_queries
        position = torch.from_numpy(scene.get_array("ball_pos_norm")).float()
        velocity = torch.from_numpy(scene.get_array("ball_vel_norm")).float()
        if position.ndim != 3 or velocity.shape != position.shape:
            raise ValueError(
                "Tracking scenes require explicit (T,P,3) position/velocity arrays."
            )
        num_frames, num_physical = position.shape[:2]
        if not scene.has_key("ball_present"):
            raise ValueError(
                "Tracking scene is incompatible: required ball_present is missing."
            )
        physical_presence = torch.from_numpy(scene.get_array("ball_present")).bool()
        if physical_presence.shape != (num_frames, num_physical):
            raise ValueError(
                "ball_present must match the explicit (T,P) physical object axes."
            )
        window = self.select_window(scene, full_len=num_frames)
        cameras = self.select_cameras(scene)
        court_views = self._court_views_by_scene.get(scene.path.name, ())
        if (
            not self.augment
            and self.court_keypoint_contract.selector == CAMERA_VIEW_V2_SELECTOR
        ):
            cameras = CameraSelection(
                indices=include_evaluation_reference_camera(
                    tuple(view.camera_id for view in court_views),
                    cameras.indices,
                    requested_camera_id=self.reference_camera_id,
                    rng=self.rng,
                )
            )
        frame = resolve_blcs_sample_court_frame(
            scene=scene,
            selected_camera_indices=cameras.indices,
            court_views=court_views,
            contract=self.court_keypoint_contract,
            rng=self.rng,
            training=self.augment,
            reference_camera_id=(
                None if self.augment else self.reference_camera_id
            ),
        )
        position = position[window.sl]
        velocity = velocity[window.sl]
        if self.court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
            position = court_points_physical_to_target(position, frame.provenance)
            velocity = court_vectors_physical_to_target(velocity, frame.provenance)
        physical_presence = physical_presence[window.sl]
        target_packing = build_fixed_lifecycle_assignment(
            physical_presence,
            num_slots=num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
        )

        uv_rows: list[Tensor] = []
        vis_rows: list[Tensor] = []
        court_rows: list[Tensor] = []
        court_vis_rows: list[Tensor] = []
        for selected_index, camera_index in enumerate(cameras.indices):
            uv = torch.from_numpy(
                scene.get_camera_array(camera_index, "ball_uv", window=window)
            ).float()
            ball_vis = torch.from_numpy(
                scene.get_camera_array(camera_index, "ball_vis", window=window)
            ).bool()
            if uv.ndim == 2:
                uv = uv[:, None]
                ball_vis = ball_vis[:, None]
            ball_vis &= physical_presence
            uv[~physical_presence] = 0.0
            uv_rows.append(uv)
            vis_rows.append(ball_vis)

            source_view = (
                frame.selected_views[selected_index]
                if frame.selected_views
                else None
            )
            raw_court = scene.get_camera_array(camera_index, "court_kp_uv")
            court_np = align_blcs_court_array(
                raw_court,
                source_view=source_view,
                frame=frame,
                keypoint_axis=(0 if raw_court.ndim == 2 else 1),
            )
            raw_court_vis = scene.get_camera_array(camera_index, "court_kp_vis")
            court_vis_np = align_blcs_court_array(
                raw_court_vis,
                source_view=source_view,
                frame=frame,
                keypoint_axis=(0 if raw_court_vis.ndim == 1 else 1),
            )
            if court_np.ndim == 2:
                court = (
                    torch.from_numpy(court_np[:14])
                    .float()[None]
                    .expand(window.seq_len, -1, -1)
                )
                court_vis = (
                    torch.from_numpy(court_vis_np[:14])
                    .bool()[None]
                    .expand(window.seq_len, -1)
                )
            else:
                court = torch.from_numpy(court_np[window.sl, :14]).float()
                court_vis = torch.from_numpy(court_vis_np[window.sl, :14]).bool()
            court_rows.append(court)
            court_vis_rows.append(court_vis)

        physical_uv = torch.stack(uv_rows)
        physical_vis = torch.stack(vis_rows)
        observations = build_physical_observation_candidates(
            ball_uv=physical_uv,
            ball_vis=physical_vis,
            physical_presence=physical_presence,
        )
        sample = {
            "scene_format_version": torch.tensor(4),
            "_physical_ball_uv": observations.uv,
            "_physical_ball_vis": observations.vis,
            "_physical_gt_index": observations.gt_index,
            "_selected_camera_indices": tuple(int(index) for index in cameras.indices),
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "padding_mask": torch.zeros(
                len(cameras.indices), window.seq_len, dtype=torch.bool
            ),
            "target_position": target_packing.pack_tensor(position, physical_presence),
            "target_velocity": target_packing.pack_tensor(velocity, physical_presence),
            "target_presence": target_packing.target_presence,
            "target_instance_id": target_packing.target_instance_id,
            "target_slot_mask": target_packing.target_presence.any(0),
            "court_reference_provenance": frame.provenance,
            "selected_camera_ids": tuple(
                view.camera_id for view in frame.selected_views
            )
            or tuple(f"camera_{index}" for index in cameras.indices),
        }
        if frame.reference_selection is not None:
            sample.update(
                blcs_reference_sample_fields(
                    frame.reference_selection,
                    dtype=position.dtype,
                    track_query_reference_document=(
                        self.track_query_reference_document
                    ),
                )
            )
        return sample

    def augment_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply optional corruption before deterministic camera-local tracking."""
        if self.num_queries is None:
            raise RuntimeError("BLCS tracking dataset lost its fixed query width.")
        court_vis = sample["court_vis"]
        court_kp = sample["court_kp"].masked_fill(
            ~court_vis.unsqueeze(-1),
            0.0,
        )
        physical = PhysicalObservationCandidates(
            uv=sample["_physical_ball_uv"],
            vis=sample["_physical_ball_vis"],
            gt_index=sample["_physical_gt_index"],
        )
        detections = (
            self.tracking_augmentation(
                physical,
                court_kp=court_kp,
                court_vis=court_vis,
            )
            if self.augment
            else physical
        )
        camera_indices = sample["_selected_camera_indices"]
        if not isinstance(camera_indices, tuple):
            raise TypeError("Selected camera indices must be an immutable tuple.")
        tracked = track_multiview_observations(
            detections.uv.unsqueeze(-2),
            detections.vis.unsqueeze(-1),
            num_slots=self.num_queries,
            config=self.observation_tracking_config,
            camera_indices=camera_indices,
            debug_provenance=detections.gt_index,
        )
        if tracked.debug_provenance is None:
            raise RuntimeError("BLCS observation tracking lost debug provenance.")
        aligned_debug = align_clean_observations_after_tracking(
            clean=physical,
            detection_indices=tracked.detection_indices,
            candidate_gt_index=tracked.debug_provenance,
        )

        output = dict(sample)
        for internal_key in (
            "_physical_ball_uv",
            "_physical_ball_vis",
            "_physical_gt_index",
            "_selected_camera_indices",
        ):
            del output[internal_key]
        output.update(
            {
                "ball_uv": tracked.values[..., 0, :],
                "ball_vis": tracked.visibility[..., 0],
                "court_kp": court_kp,
                "clean_ball_uv": aligned_debug.clean_uv,
                "clean_ball_vis": aligned_debug.clean_vis,
                "candidate_gt_index": aligned_debug.gt_index,
            }
        )
        return output


def collate_blcs_tracking_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pad only camera/time dimensions and stack exact-width BLCS scenes."""
    provenance_rows: list[CourtReferenceFrameProvenance] = []
    contract_id: str | None = None
    for sample_index, sample in enumerate(batch):
        provenance = sample.get("court_reference_provenance")
        if not isinstance(provenance, CourtReferenceFrameProvenance):
            raise CourtReferenceFrameError(
                "Every BLCS tracking sample must provide a validated "
                "CourtReferenceFrameProvenance; "
                f"sample {sample_index} has {type(provenance).__name__}."
            )
        if contract_id is None:
            contract_id = provenance.contract_id
        elif provenance.contract_id != contract_id:
            raise CourtReferenceFrameError(
                "BLCS tracking batches cannot mix CourtKP20 contracts; "
                f"sample 0 uses {contract_id!r} and sample {sample_index} uses "
                f"{provenance.contract_id!r}."
            )
        provenance_rows.append(provenance)

    candidate_width = int(batch[0]["ball_uv"].shape[2]) if batch else 0
    for sample in batch:
        if (
            sample["ball_uv"].shape[2] != candidate_width
            or sample["target_position"].shape[1] != candidate_width
            or sample["target_slot_mask"].shape[0] != candidate_width
        ):
            raise ValueError(
                "Every BLCS tracking sample must already have the same exact "
                "candidate width before collation."
            )
    provenances = tuple(provenance_rows)
    reference_tensor_keys = {
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
    }
    tensor_batch = [
        {
            key: value
            for key, value in sample.items()
            if isinstance(value, Tensor) and key not in reference_tensor_keys
        }
        for sample in batch
    ]
    collated: dict[str, Any] = pad_and_stack_tracking_batch(
        tensor_batch,
        padding_dimensions={
            "ball_uv": (0, 1),
            "ball_vis": (0, 1),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "padding_mask": (0, 1),
            "target_position": (0,),
            "target_velocity": (0,),
            "target_presence": (0,),
            "target_instance_id": (0,),
            "clean_ball_uv": (0, 1),
            "clean_ball_vis": (0, 1),
            "candidate_gt_index": (0, 1),
        },
        pad_values={
            "target_instance_id": -1,
            "candidate_gt_index": -1,
            "padding_mask": True,
        },
    )
    collated["court_reference_provenance"] = provenances
    for sample_index, sample in enumerate(batch):
        camera_ids = sample.get("selected_camera_ids")
        if not isinstance(camera_ids, tuple) or any(
            type(camera_id) is not str or not camera_id
            for camera_id in camera_ids
        ):
            raise ValueError(
                "Every BLCS tracking sample must provide canonical "
                f"selected_camera_ids; sample {sample_index} is invalid."
            )
    collated["selected_camera_ids"] = tuple(
        sample["selected_camera_ids"] for sample in batch
    )
    collated.update(
        collate_blcs_reference_fields(
            batch,
            max_views=int(collated["ball_uv"].shape[1]),
            model_tensor_key="ball_uv",
            transform_dtype_key="target_position",
        )
    )
    return collated


__all__ = [
    "BLCS_TRACKING_KEYS",
    "BLCSTrackingDataset",
    "collate_blcs_tracking_batch",
]

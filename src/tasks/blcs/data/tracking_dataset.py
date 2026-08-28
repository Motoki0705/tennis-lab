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
    pack_observation_candidates,
)
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
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
    """Load ID-ordered objects, pack lifecycle slots, and corrupt observations."""

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
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self.tracking_augmentation = BLCSTrackingCandidateAugmentation(
            data_cfg["augmentation"]
        )
        if self.num_queries is None:
            raise ValueError("BLCS tracking requires model.num_queries.")
        if not self.pack_to_query_slots:
            raise ValueError(
                "BLCS tracking requires data.lifecycle.pack_to_query_slots=true."
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
        randomize_slots = self.augment and self.randomize_slots_train
        target_packing = build_fixed_lifecycle_assignment(
            physical_presence,
            num_slots=num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=randomize_slots,
            generator=None,
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
        observations = pack_observation_candidates(
            ball_uv=physical_uv,
            ball_vis=physical_vis,
            physical_presence=physical_presence,
            num_slots=num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=randomize_slots,
        )
        sample = {
            "scene_format_version": torch.tensor(4),
            "ball_uv": observations.uv,
            "ball_vis": observations.vis,
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
            "clean_ball_uv": observations.uv.clone(),
            "clean_ball_vis": observations.vis.clone(),
            "candidate_gt_index": observations.gt_index,
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
        if not self.augment:
            return sample
        metadata = {
            key: value for key, value in sample.items() if not isinstance(value, Tensor)
        }
        tensor_sample = {
            key: value for key, value in sample.items() if isinstance(value, Tensor)
        }
        augmented: dict[str, Tensor] = self.tracking_augmentation(tensor_sample)
        return {**augmented, **metadata}


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

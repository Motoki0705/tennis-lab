"""Config-aware canonical-scene adapter for lifecycle PLCS tracking."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data import (
    CAMERA_ID_PADDING_VALUE,
    ReferenceViewSelection,
    StableCameraIdTable,
    include_evaluation_reference_camera,
    validate_reference_view_batch,
)
from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment
from src.tasks.base.data.scene_dataset import CameraSelection, Scene
from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    build_physical_court_provenance,
    camera_extrinsics_physical_to_target,
)
from src.tasks.plcs.court_keypoint_contract import (
    PLCSCourtKeypointRuntimeConfig,
    align_selected_court_array,
    choose_reference_selection,
    court_keypoint_contract_document,
    normalized_headings_physical_to_target,
    normalized_points_physical_to_target,
    scene_court_views,
    selected_court_views,
    track_query_reference_contract_document,
    validate_plcs_dataset_court_keypoints,
    world_joints_physical_to_target,
)
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)
from src.utils.projection.camera_projector import camera_from_mapping

PLCS_TRACKING_KEYS = (
    "scene_format_version",
    "human_kp",
    "human_vis",
    "court_kp",
    "court_vis",
    "padding_mask",
    "target_position",
    "target_rotation",
    "target_canonical_pose_3d",
    "target_human_kp_3d",
    "target_presence",
    "target_instance_id",
    "target_slot_mask",
    "clean_human_kp",
    "clean_human_vis",
    "human_kp_target",
    "human_vis_target",
    "detection_gt_index",
    "camera_C",
    "camera_R",
    "camera_f",
    "camera_cx",
    "camera_cy",
    "camera_w",
    "camera_h",
)


class PLCSTrackingDataset(CanonicalTrackingDataset):
    """Load ID-ordered objects, pack lifecycle slots, and corrupt observations."""

    def __init__(
        self,
        *,
        reference_camera_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        config = kwargs.get("config")
        self.reference_camera_id = reference_camera_id
        self.court_keypoint_contract = PLCSCourtKeypointRuntimeConfig.from_config(
            config,
        ).contract
        self.track_query_reference_document = track_query_reference_contract_document(
            config,
            self.court_keypoint_contract,
        )
        self.court_keypoint_validation = validate_plcs_dataset_court_keypoints(
            kwargs["scene_dir"],
            kwargs["split_file"],
            self.court_keypoint_contract,
        )
        super().__init__(**kwargs)
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self.tracking_augmentation = PLCSTrackingDetectionAugmentation(
            data_cfg["augmentation"]
        )

    def build_sample(self, scene: Scene) -> dict[str, Any]:
        position = torch.from_numpy(scene.get_array("position")).float()
        rotation = torch.from_numpy(scene.get_array("rotation")).float()
        canonical_pose = torch.from_numpy(scene.get_array("canonical_pose_3d")).float()
        world_joints = torch.from_numpy(scene.get_array("human_kp_3d")).float()
        if position.ndim == 2:
            position = position[:, None]
            rotation = rotation[:, None]
            canonical_pose = canonical_pose[:, None]
            world_joints = world_joints[:, None]
        num_frames, num_physical = position.shape[:2]
        if scene.has_key("person_present"):
            physical_presence = torch.from_numpy(
                scene.get_array("person_present")
            ).bool()
        else:
            physical_presence = torch.ones((num_frames, num_physical), dtype=torch.bool)
        window = self.select_window(scene, full_len=num_frames)
        cameras = self.select_cameras(scene)
        complete_views = scene_court_views(
            self.court_keypoint_validation,
            scene.path,
        )
        if (
            not self.augment
            and self.court_keypoint_contract.selector == CAMERA_VIEW_V2_SELECTOR
        ):
            cameras = CameraSelection(
                indices=include_evaluation_reference_camera(
                    tuple(view.camera_id for view in complete_views),
                    cameras.indices,
                    requested_camera_id=self.reference_camera_id,
                    rng=self.rng,
                )
            )
        position = position[window.sl]
        rotation = rotation[window.sl]
        canonical_pose = canonical_pose[window.sl]
        world_joints = world_joints[window.sl]
        physical_presence = physical_presence[window.sl]
        views = selected_court_views(
            self.court_keypoint_validation,
            scene.path,
            cameras.indices,
        )
        selection = choose_reference_selection(
            self.court_keypoint_contract,
            complete_views,
            views,
            rng=self.rng if self.augment else None,
            requested_camera_id=(
                None if self.augment else self.reference_camera_id
            ),
        )
        provenance = (
            build_physical_court_provenance()
            if selection is None
            else selection.provenance
        )
        reference_view = (
            None
            if selection is None
            else selection.selected_views[selection.reference_view_index]
        )
        position = normalized_points_physical_to_target(
            position,
            provenance,
        )
        rotation = normalized_headings_physical_to_target(rotation, provenance)
        world_joints = world_joints_physical_to_target(world_joints, provenance)
        if self.num_queries is None:
            raise ValueError("PLCS tracking requires model.num_queries.")
        packing = build_fixed_lifecycle_assignment(
            physical_presence,
            num_slots=self.num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=self.augment and self.randomize_slots_train,
            generator=None,
        )

        kp_rows: list[Tensor] = []
        visible_rows: list[Tensor] = []
        index_rows: list[Tensor] = []
        clean_kp_rows: list[Tensor] = []
        clean_visible_rows: list[Tensor] = []
        court_rows: list[Tensor] = []
        court_vis_rows: list[Tensor] = []
        camera_center_rows: list[Tensor] = []
        camera_rotation_rows: list[Tensor] = []
        camera_f_rows: list[Tensor] = []
        camera_cx_rows: list[Tensor] = []
        camera_cy_rows: list[Tensor] = []
        camera_w_rows: list[Tensor] = []
        camera_h_rows: list[Tensor] = []
        for local_index, camera_index in enumerate(cameras.indices):
            keypoints = torch.from_numpy(
                scene.get_camera_array(camera_index, "human_kp_uv", window=window)
            ).float()
            visible = torch.from_numpy(
                scene.get_camera_array(camera_index, "human_kp_vis", window=window)
            ).bool()
            if keypoints.ndim == 3:
                keypoints = keypoints[:, None]
                visible = visible[:, None]
            visible &= physical_presence[..., None]
            keypoints[~visible] = 0.0
            packed_keypoints = packing.pack_tensor(keypoints, physical_presence)
            packed_visible = packing.pack_tensor(visible, physical_presence)
            clean_kp_rows.append(packed_keypoints.clone())
            clean_visible_rows.append(packed_visible.clone())
            detection_index = torch.where(
                packed_visible.any(-1), packing.target_instance_id, -1
            )
            kp_rows.append(packed_keypoints)
            visible_rows.append(packed_visible)
            index_rows.append(detection_index)
            source_view = views[local_index] if views else None
            court_rows.append(
                torch.from_numpy(
                    align_selected_court_array(
                        scene.get_camera_array(
                            camera_index, "court_kp_uv", window=window
                        ),
                        source_view,
                        reference_view,
                        keypoint_axis=-2,
                    )[:, :14]
                ).float()
            )
            court_vis_rows.append(
                torch.from_numpy(
                    align_selected_court_array(
                        scene.get_camera_array(
                            camera_index, "court_kp_vis", window=window
                        ),
                        source_view,
                        reference_view,
                        keypoint_axis=-1,
                    )[:, :14]
                ).bool()
            )
            params = scene.data.get(f"cam_{camera_index}_params")
            if not isinstance(params, Mapping):
                raise ValueError(
                    f"Scene {scene.path} camera {camera_index} has invalid params metadata."
                )
            camera = camera_from_mapping(dict(params))
            center, camera_rotation = camera_extrinsics_physical_to_target(
                camera.C,
                camera.R,
                provenance,
            )
            camera_center_rows.append(center)
            camera_rotation_rows.append(camera_rotation)
            camera_f_rows.append(torch.tensor(camera.f, dtype=torch.float32))
            camera_cx_rows.append(torch.tensor(camera.cx, dtype=torch.float32))
            camera_cy_rows.append(torch.tensor(camera.cy, dtype=torch.float32))
            camera_w_rows.append(torch.tensor(float(camera.w), dtype=torch.float32))
            camera_h_rows.append(torch.tensor(float(camera.h), dtype=torch.float32))

        human_kp = torch.stack(kp_rows)
        human_vis = torch.stack(visible_rows)
        clean_human_kp = torch.stack(clean_kp_rows)
        clean_human_vis = torch.stack(clean_visible_rows)
        rotation_fill = torch.tensor([1.0, 0.0], dtype=rotation.dtype)
        sample: dict[str, Any] = {
            "scene_format_version": torch.tensor(4),
            "human_kp": human_kp,
            "human_vis": human_vis,
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "padding_mask": torch.zeros(
                len(cameras.indices), window.seq_len, dtype=torch.bool
            ),
            "target_position": packing.pack_tensor(position, physical_presence),
            "target_rotation": packing.pack_tensor(
                rotation,
                physical_presence,
                fill_value=rotation_fill,
            ),
            "target_canonical_pose_3d": packing.pack_tensor(
                canonical_pose, physical_presence
            ),
            "target_human_kp_3d": packing.pack_tensor(world_joints, physical_presence),
            "target_presence": packing.target_presence,
            "target_instance_id": packing.target_instance_id,
            "target_slot_mask": packing.target_presence.any(0),
            "clean_human_kp": clean_human_kp,
            "clean_human_vis": clean_human_vis,
            "human_kp_target": clean_human_kp.clone(),
            "human_vis_target": clean_human_vis.clone(),
            "detection_gt_index": torch.stack(index_rows),
            "camera_C": torch.stack(camera_center_rows),
            "camera_R": torch.stack(camera_rotation_rows),
            "camera_f": torch.stack(camera_f_rows),
            "camera_cx": torch.stack(camera_cx_rows),
            "camera_cy": torch.stack(camera_cy_rows),
            "camera_w": torch.stack(camera_w_rows),
            "camera_h": torch.stack(camera_h_rows),
        }
        sample["court_keypoint_metadata"] = court_keypoint_contract_document(
            self.court_keypoint_contract
        )
        sample["court_reference_provenance"] = provenance
        sample["selected_camera_ids"] = tuple(
            view.camera_id for view in views
        ) or tuple(f"camera_{index}" for index in cameras.indices)
        if selection is not None:
            sample["reference_view_selection"] = selection
            sample["stable_camera_id_table"] = selection.stable_camera_id_table
            sample["reference_camera_id_string"] = selection.reference_camera_id
            sample.update(selection.to_tensor_fields(dtype=position.dtype))
            sample["physical_from_reference"] = torch.tensor(
                selection.provenance.physical_from_reference,
                dtype=position.dtype,
            )
            if self.track_query_reference_document is not None:
                sample.update(self.track_query_reference_document)
        return sample

    def augment_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.augment:
            return sample
        metadata = deepcopy(
            {
                key: value
                for key, value in sample.items()
                if not isinstance(value, Tensor)
            }
        )
        tensor_sample: dict[str, Tensor] = {
            key: value for key, value in sample.items() if isinstance(value, Tensor)
        }
        augmented = self.tracking_augmentation(tensor_sample)
        return {**augmented, **metadata}


def collate_plcs_tracking_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pad variable camera/time/detection dimensions and stack PLCS scenes."""
    reference_metadata_keys = {
        "reference_view_selection",
        "stable_camera_id_table",
        "reference_camera_id_string",
    }
    reference_tensor_keys = {
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
    }
    reference_fields = reference_metadata_keys | reference_tensor_keys
    sample_reference_fields = [set(sample) & reference_fields for sample in batch]
    has_reference = any(sample_reference_fields)
    if has_reference:
        for sample_index, fields in enumerate(sample_reference_fields):
            sample = batch[sample_index]
            if fields != reference_fields:
                raise ValueError(
                    "PLCS tracking batch has missing/mixed reference schema at "
                    f"sample {sample_index}: expected {sorted(reference_fields)!r}, "
                    f"got {sorted(fields)!r}."
                )
            selection = sample["reference_view_selection"]
            table = sample["stable_camera_id_table"]
            if not isinstance(selection, ReferenceViewSelection):
                raise TypeError(
                    f"PLCS tracking sample {sample_index} has invalid reference "
                    "selection type."
                )
            if not isinstance(table, StableCameraIdTable) or (
                table != selection.stable_camera_id_table
            ):
                raise ValueError(
                    f"PLCS tracking sample {sample_index} stable camera ID table "
                    "does not match its reference selection."
                )
            expected = selection.to_tensor_fields(
                dtype=sample["target_position"].dtype,
            )
            for key, expected_value in expected.items():
                stored = sample[key]
                if not isinstance(stored, Tensor) or not torch.equal(
                    stored,
                    expected_value,
                ):
                    raise ValueError(
                        f"PLCS tracking sample {sample_index} {key} does not "
                        "match its typed reference selection."
                    )
            if sample["reference_camera_id_string"] != selection.reference_camera_id:
                raise ValueError(
                    f"PLCS tracking sample {sample_index} canonical reference ID "
                    "does not match its typed reference selection."
                )
    metadata_keys = {
        "court_keypoint_metadata",
        "court_reference_provenance",
        "selected_camera_ids",
    }
    if has_reference:
        metadata_keys.update(reference_metadata_keys)
    semantic_documents = [sample.get("track_query_reference") for sample in batch]
    if any(document is not None for document in semantic_documents):
        if any(document is None for document in semantic_documents):
            raise ValueError(
                "PLCS tracking batch contains mixed track-query reference "
                "contract markers."
            )
        first_document = semantic_documents[0]
        if any(document != first_document for document in semantic_documents[1:]):
            raise ValueError(
                "PLCS tracking batch contains non-identical track-query reference "
                "contracts."
            )
        metadata_keys.add("track_query_reference")
    for key in metadata_keys:
        missing = [index for index, sample in enumerate(batch) if key not in sample]
        if missing:
            raise ValueError(
                f"PLCS tracking batch is missing required {key!r} for samples "
                f"{missing!r}."
            )
    tensor_batch = [
        {key: value for key, value in sample.items() if key not in metadata_keys}
        for sample in batch
    ]
    result: dict[str, Any] = pad_and_stack_tracking_batch(
        tensor_batch,
        padding_dimensions={
            "human_kp": (0, 1),
            "human_vis": (0, 1),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "padding_mask": (0, 1),
            "target_position": (0,),
            "target_rotation": (0,),
            "target_canonical_pose_3d": (0,),
            "target_human_kp_3d": (0,),
            "target_presence": (0,),
            "target_instance_id": (0,),
            "clean_human_kp": (0, 1),
            "clean_human_vis": (0, 1),
            "human_kp_target": (0, 1),
            "human_vis_target": (0, 1),
            "detection_gt_index": (0, 1),
            "camera_C": (0,),
            "camera_R": (0,),
            "camera_f": (0,),
            "camera_cx": (0,),
            "camera_cy": (0,),
            "camera_w": (0,),
            "camera_h": (0,),
            "view_camera_ids": (0,),
        },
        pad_values={
            "padding_mask": True,
            "target_instance_id": -1,
            "detection_gt_index": -1,
            "camera_w": 1.0,
            "camera_h": 1.0,
            "view_camera_ids": CAMERA_ID_PADDING_VALUE,
        },
    )
    for key in metadata_keys:
        if key == "track_query_reference":
            result[key] = batch[0][key]
        else:
            result[key] = tuple(sample[key] for sample in batch)
    if has_reference:
        result["reference_camera_id_string"] = tuple(
            selection.reference_camera_id
            for selection in result["reference_view_selection"]
        )
        validate_reference_view_batch(
            reference_view_index=result["reference_view_index"],
            view_camera_ids=result["view_camera_ids"],
            reference_camera_id=result["reference_camera_id"],
            view_valid_mask=result["view_camera_ids"].ge(0),
            reference_from_physical=result["reference_from_physical"],
            expected_device=result["human_kp"].device,
        )
        if not torch.allclose(
            result["physical_from_reference"],
            result["reference_from_physical"].transpose(-1, -2),
            rtol=0.0,
            atol=(
                1e-12
                if result["reference_from_physical"].dtype == torch.float64
                else 1e-6
            ),
        ):
            raise ValueError(
                "PLCS tracking physical_from_reference must equal "
                "reference_from_physical.T."
            )
    return result


__all__ = [
    "PLCS_TRACKING_KEYS",
    "PLCSTrackingDataset",
    "collate_plcs_tracking_batch",
]

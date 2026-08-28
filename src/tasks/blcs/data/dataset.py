"""Dataset and collate/adaptation utilities for BLCS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from src.tasks.base.data import include_evaluation_reference_camera
from src.tasks.base.data.rng import require_run_seed
from src.tasks.base.data.scene_dataset import (
    CameraSelection,
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
)
from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    PHYSICAL_V1_SELECTOR,
    camera_extrinsics_physical_to_target,
    court_points_physical_to_target,
    court_vectors_physical_to_target,
)
from src.tasks.blcs.configuration import parse_court_keypoint_contract
from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.blcs.data.court_view import (
    align_blcs_court_array,
    blcs_reference_sample_fields,
    blcs_track_query_reference_contract_document,
    collate_blcs_reference_fields,
    court_views_by_scene,
    resolve_blcs_sample_court_frame,
    validate_blcs_dataset_court_keypoints,
)
from src.tasks.blcs.data.types import BLCSMultiViewBatch, BLCSMultiViewSample
from src.utils.schema.court_normalization import (
    validate_court_coordinate_normalization,
)


class BallTrajectoryDataset(SceneDatasetBase[BLCSMultiViewSample]):
    """Unified BLCS dataset that always returns canonical multiview samples.

    The canonical sample format keeps camera and temporal dimensions:
    - ball_uv: (N, T, 2)
    - ball_vis: (N, T)
    - padding_mask: (N, T), ``True`` marks padding
    - court_kp: (N, T, 20, 2)
    - court_vis: (N, T, 20)
    - position_3d: (T, 3)
    - velocity_3d: (T, 3)
    """

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: object,
        seed: int | None = None,
        augment: bool = True,
        reference_camera_id: str | None = None,
    ) -> None:
        self.hydra_cfg = config
        self.augment = augment
        self.reference_camera_id = reference_camera_id
        self.court_keypoint_contract = parse_court_keypoint_contract(config)
        self.track_query_reference_document = (
            blcs_track_query_reference_contract_document(
                config,
                self.court_keypoint_contract,
            )
        )
        court_dataset = validate_blcs_dataset_court_keypoints(
            scene_dir=scene_dir,
            split_file=split_file,
            contract=self.court_keypoint_contract,
        )
        self._court_views_by_scene = court_views_by_scene(court_dataset)
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self._configure_task(data_cfg)
        super().__init__(
            config=self._build_scene_dataset_config(
                scene_dir=scene_dir,
                split_file=split_file,
                data_cfg=data_cfg,
            ),
            seed=require_run_seed(config) if seed is None else seed,
            sample_local_rng=not augment,
        )

    # -- Composed-method hooks ------------------------------------------

    def _configure_task(self, data_cfg: dict) -> None:
        # Multiview ranges
        self.seq_len_range = self._parse_int_range(data_cfg, "seq_len_range")
        self.num_views_range = self._parse_int_range(data_cfg, "num_views_range")
        self.camera_mode = self._parse_camera_mode(data_cfg)

        # Number of court keypoints to use (first N from the canonical order)
        self.num_court_kp = int(data_cfg["num_court_kp"])

        # Augmentation pipeline
        aug_cfg = data_cfg["augmentation"]
        self.augmentation_pipeline = BLCSBallObservationAugmentation(aug_cfg)

    def _build_scene_dataset_config(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        data_cfg: dict,
    ) -> SceneDatasetConfig:
        return SceneDatasetConfig(
            scene_dir=Path(scene_dir),
            split_file=Path(split_file),
            seq_len_range=self.seq_len_range,
            num_views_range=self.num_views_range,
            camera_mode=self.camera_mode,
            crop_mode=("random" if self.augment else "center"),
            min_num_frames=self.seq_len_range[0],
            min_num_cameras=self.num_views_range[0],
        )

    def _validate_scene_metadata(self, meta: dict, *, path: Path) -> None:
        validate_court_coordinate_normalization(meta, artifact=f"Scene {path}")

    def build_sample(self, scene: Scene) -> BLCSMultiViewSample:
        cams = self.select_cameras(
            scene, num_views_range=self.num_views_range, camera_mode=self.camera_mode
        )
        court_views = self._court_views_by_scene.get(scene.path.name, ())
        if (
            not self.augment
            and self.court_keypoint_contract.selector == CAMERA_VIEW_V2_SELECTOR
        ):
            cams = CameraSelection(
                indices=include_evaluation_reference_camera(
                    tuple(view.camera_id for view in court_views),
                    cams.indices,
                    requested_camera_id=self.reference_camera_id,
                    rng=self.rng,
                )
            )
        frame = resolve_blcs_sample_court_frame(
            scene=scene,
            selected_camera_indices=cams.indices,
            court_views=court_views,
            contract=self.court_keypoint_contract,
            rng=self.rng,
            training=self.augment,
            reference_camera_id=(
                None if self.augment else self.reference_camera_id
            ),
        )
        # Use camera trajectory length to guard against metadata drift.
        primary_len = int(scene.get_camera_array(cams.primary, "ball_uv").shape[0])
        pos_len = int(scene.data["ball_pos_norm"].shape[0])
        vel_len = int(scene.data["ball_vel_norm"].shape[0])
        full_len = scene.effective_num_frames(primary_len, pos_len, vel_len)
        window = self.select_window(scene, full_len=full_len)
        ball_uv_list: list[Tensor] = []
        ball_vis_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []
        cam_R_list: list[Tensor] = []
        cam_C_list: list[Tensor] = []
        cam_f_list: list[Tensor] = []
        cam_cx_list: list[Tensor] = []
        cam_cy_list: list[Tensor] = []
        cam_w_list: list[Tensor] = []
        cam_h_list: list[Tensor] = []

        for selected_index, cam_idx in enumerate(cams.indices):
            ball_uv = torch.from_numpy(
                scene.get_camera_array(cam_idx, "ball_uv", window=window)
            ).float()
            ball_vis = torch.from_numpy(
                scene.get_camera_array(cam_idx, "ball_vis", window=window)
            ).float()
            source_view = (
                frame.selected_views[selected_index]
                if frame.selected_views
                else None
            )
            court_kp = torch.from_numpy(
                align_blcs_court_array(
                    scene.get_camera_array(cam_idx, "court_kp_uv"),
                    source_view=source_view,
                    frame=frame,
                    keypoint_axis=0,
                )
            ).float()
            court_vis = torch.from_numpy(
                align_blcs_court_array(
                    scene.get_camera_array(cam_idx, "court_kp_vis"),
                    source_view=source_view,
                    frame=frame,
                    keypoint_axis=0,
                )
            ).float()
            court_kp = court_kp[: self.num_court_kp]
            court_vis = court_vis[: self.num_court_kp]

            court_kp_expanded = court_kp.unsqueeze(0).expand(window.seq_len, -1, -1)
            court_vis_expanded = court_vis.unsqueeze(0).expand(window.seq_len, -1)

            ball_uv_list.append(ball_uv)
            ball_vis_list.append(ball_vis)
            court_kp_list.append(court_kp_expanded)
            court_vis_list.append(court_vis_expanded)

            # Load camera parameters from scene payload
            params_key = f"cam_{cam_idx}_params"
            raw = scene.data[params_key]
            cam_params = raw if isinstance(raw, dict) else json.loads(str(raw))
            # Normalise key: generators may store centre as "C" or "center"
            physical_R = torch.tensor(cam_params["R"], dtype=torch.float32)
            physical_C = torch.tensor(cam_params["C"], dtype=torch.float32)
            if self.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR:
                transformed_C, transformed_R = physical_C, physical_R
            else:
                transformed_C, transformed_R = camera_extrinsics_physical_to_target(
                    physical_C,
                    physical_R,
                    frame.provenance,
                )
            cam_R_list.append(transformed_R)
            cam_C_list.append(transformed_C)
            cam_f_list.append(torch.tensor(cam_params["f"], dtype=torch.float32))
            cam_cx_list.append(torch.tensor(cam_params["cx"], dtype=torch.float32))
            cam_cy_list.append(torch.tensor(cam_params["cy"], dtype=torch.float32))
            cam_w_list.append(torch.tensor(float(cam_params["w"]), dtype=torch.float32))
            cam_h_list.append(torch.tensor(float(cam_params["h"]), dtype=torch.float32))

        position_3d = torch.from_numpy(
            scene.get_array("ball_pos_norm", window=window)
        ).float()
        velocity_3d = torch.from_numpy(
            scene.get_array("ball_vel_norm", window=window)
        ).float()
        if self.court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
            position_3d = court_points_physical_to_target(
                position_3d,
                frame.provenance,
            )
            velocity_3d = court_vectors_physical_to_target(
                velocity_3d,
                frame.provenance,
            )

        sample: BLCSMultiViewSample = {
            "ball_uv": torch.stack(ball_uv_list, dim=0),
            "ball_vis": torch.stack(ball_vis_list, dim=0),
            "padding_mask": torch.zeros(
                len(cams.indices), window.seq_len, dtype=torch.bool
            ),
            "court_kp": torch.stack(court_kp_list, dim=0),
            "court_vis": torch.stack(court_vis_list, dim=0),
            "position_3d": position_3d,
            "velocity_3d": velocity_3d,
            "seq_len": torch.tensor(window.seq_len, dtype=torch.long),
            "camera_R": torch.stack(cam_R_list, dim=0),
            "camera_C": torch.stack(cam_C_list, dim=0),
            "camera_f": torch.stack(cam_f_list, dim=0),
            "camera_cx": torch.stack(cam_cx_list, dim=0),
            "camera_cy": torch.stack(cam_cy_list, dim=0),
            "camera_w": torch.stack(cam_w_list, dim=0),
            "camera_h": torch.stack(cam_h_list, dim=0),
            "court_reference_provenance": frame.provenance,
            "selected_camera_ids": tuple(
                view.camera_id for view in frame.selected_views
            )
            or tuple(f"camera_{index}" for index in cams.indices),
        }
        if frame.reference_selection is not None:
            sample.update(
                cast(
                    "BLCSMultiViewSample",
                    blcs_reference_sample_fields(
                        frame.reference_selection,
                        dtype=position_3d.dtype,
                        track_query_reference_document=(
                            self.track_query_reference_document
                        ),
                    ),
                )
            )
        return sample

    def _apply_augmentation_multiview(
        self, sample: BLCSMultiViewSample
    ) -> BLCSMultiViewSample:
        augmented: BLCSMultiViewSample = self.augmentation_pipeline.forward(sample)
        return augmented

    def augment_sample(self, sample: BLCSMultiViewSample) -> BLCSMultiViewSample:
        if self.augment:
            return self._apply_augmentation_multiview(sample)
        return sample


def collate_multiview_trajectories(
    batch: list[BLCSMultiViewSample],
) -> BLCSMultiViewBatch:
    """Collate canonical BLCS samples into padded canonical batch tensors."""
    max_views = max(int(sample["ball_uv"].shape[0]) for sample in batch)
    max_seq_len = max(int(sample["seq_len"].item()) for sample in batch)
    has_clean_targets = any(
        "ball_uv_target" in sample and "ball_vis_target" in sample for sample in batch
    )

    ball_uv_batch = []
    ball_vis_batch = []
    ball_uv_target_batch = []
    ball_vis_target_batch = []
    padding_mask_batch = []
    court_kp_batch = []
    court_vis_batch = []
    position_3d_batch = []
    velocity_3d_batch = []
    seq_len_batch = []
    cam_R_batch: list[Tensor] = []
    cam_C_batch: list[Tensor] = []
    cam_f_batch: list[Tensor] = []
    cam_cx_batch: list[Tensor] = []
    cam_cy_batch: list[Tensor] = []
    cam_w_batch: list[Tensor] = []
    cam_h_batch: list[Tensor] = []
    for sample in batch:
        n_views = int(sample["ball_uv"].shape[0])
        seq_len = int(sample["seq_len"].item())
        pad_views = max_views - n_views
        pad_seq = max_seq_len - seq_len

        ball_uv = sample["ball_uv"]
        ball_vis = sample["ball_vis"]
        ball_uv_target = (
            sample.get("ball_uv_target", ball_uv) if has_clean_targets else None
        )
        ball_vis_target = (
            sample.get("ball_vis_target", ball_vis) if has_clean_targets else None
        )
        padding_mask = sample["padding_mask"]
        court_kp = sample["court_kp"]
        court_vis = sample["court_vis"]
        position_3d = sample["position_3d"]
        velocity_3d = sample["velocity_3d"]
        n_kp = court_kp.shape[-2]

        cam_R = sample["camera_R"]
        cam_C = sample["camera_C"]
        cam_f = sample["camera_f"]
        cam_cx = sample["camera_cx"]
        cam_cy = sample["camera_cy"]
        cam_w = sample["camera_w"]
        cam_h = sample["camera_h"]

        if pad_seq > 0:
            ball_uv = torch.cat([ball_uv, torch.zeros(n_views, pad_seq, 2)], dim=1)
            ball_vis = torch.cat([ball_vis, torch.zeros(n_views, pad_seq)], dim=1)
            if has_clean_targets:
                assert ball_uv_target is not None
                assert ball_vis_target is not None
                ball_uv_target = torch.cat(
                    [ball_uv_target, torch.zeros(n_views, pad_seq, 2)],
                    dim=1,
                )
                ball_vis_target = torch.cat(
                    [ball_vis_target, torch.zeros(n_views, pad_seq)],
                    dim=1,
                )
            padding_mask = torch.cat(
                [padding_mask, torch.ones(n_views, pad_seq, dtype=torch.bool)], dim=1
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(n_views, pad_seq, n_kp, 2)], dim=1
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(n_views, pad_seq, n_kp)], dim=1
            )
            position_3d = torch.cat([position_3d, torch.zeros(pad_seq, 3)], dim=0)
            velocity_3d = torch.cat([velocity_3d, torch.zeros(pad_seq, 3)], dim=0)

        if pad_views > 0:
            ball_uv = torch.cat(
                [ball_uv, torch.zeros(pad_views, max_seq_len, 2)], dim=0
            )
            ball_vis = torch.cat([ball_vis, torch.zeros(pad_views, max_seq_len)], dim=0)
            if has_clean_targets:
                assert ball_uv_target is not None
                assert ball_vis_target is not None
                ball_uv_target = torch.cat(
                    [ball_uv_target, torch.zeros(pad_views, max_seq_len, 2)],
                    dim=0,
                )
                ball_vis_target = torch.cat(
                    [ball_vis_target, torch.zeros(pad_views, max_seq_len)],
                    dim=0,
                )
            padding_mask = torch.cat(
                [
                    padding_mask,
                    torch.ones(pad_views, max_seq_len, dtype=torch.bool),
                ],
                dim=0,
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(pad_views, max_seq_len, n_kp, 2)], dim=0
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(pad_views, max_seq_len, n_kp)], dim=0
            )
            # Pad camera parameters with zeros for extra views
            cam_R = torch.cat([cam_R, torch.zeros(pad_views, 3, 3)], dim=0)
            cam_C = torch.cat([cam_C, torch.zeros(pad_views, 3)], dim=0)
            cam_f = torch.cat([cam_f, torch.zeros(pad_views)], dim=0)
            cam_cx = torch.cat([cam_cx, torch.zeros(pad_views)], dim=0)
            cam_cy = torch.cat([cam_cy, torch.zeros(pad_views)], dim=0)
            cam_w = torch.cat(
                [cam_w, torch.ones(pad_views)], dim=0
            )  # ones to avoid div-by-zero
            cam_h = torch.cat([cam_h, torch.ones(pad_views)], dim=0)

        ball_uv_batch.append(ball_uv)
        ball_vis_batch.append(ball_vis)
        if has_clean_targets:
            assert ball_uv_target is not None
            assert ball_vis_target is not None
            ball_uv_target_batch.append(ball_uv_target)
            ball_vis_target_batch.append(ball_vis_target)
        padding_mask_batch.append(padding_mask)
        court_kp_batch.append(court_kp)
        court_vis_batch.append(court_vis)
        position_3d_batch.append(position_3d)
        velocity_3d_batch.append(velocity_3d)
        seq_len_batch.append(sample["seq_len"])
        cam_R_batch.append(cam_R)
        cam_C_batch.append(cam_C)
        cam_f_batch.append(cam_f)
        cam_cx_batch.append(cam_cx)
        cam_cy_batch.append(cam_cy)
        cam_w_batch.append(cam_w)
        cam_h_batch.append(cam_h)
    collated: dict[str, object] = {
        "ball_uv": torch.stack(ball_uv_batch, dim=0),
        "ball_vis": torch.stack(ball_vis_batch, dim=0),
        "padding_mask": torch.stack(padding_mask_batch, dim=0),
        "court_kp": torch.stack(court_kp_batch, dim=0),
        "court_vis": torch.stack(court_vis_batch, dim=0),
        "position_3d": torch.stack(position_3d_batch, dim=0),
        "velocity_3d": torch.stack(velocity_3d_batch, dim=0),
        "seq_len": torch.stack(seq_len_batch, dim=0),
        "camera_R": torch.stack(cam_R_batch, dim=0),
        "camera_C": torch.stack(cam_C_batch, dim=0),
        "camera_f": torch.stack(cam_f_batch, dim=0),
        "camera_cx": torch.stack(cam_cx_batch, dim=0),
        "camera_cy": torch.stack(cam_cy_batch, dim=0),
        "camera_w": torch.stack(cam_w_batch, dim=0),
        "camera_h": torch.stack(cam_h_batch, dim=0),
        "court_reference_provenance": tuple(
            sample["court_reference_provenance"] for sample in batch
        ),
        "selected_camera_ids": tuple(
            sample["selected_camera_ids"] for sample in batch
        ),
    }
    collated.update(
        collate_blcs_reference_fields(
            cast("list[dict[str, object]]", batch),
            max_views=max_views,
            model_tensor_key="ball_uv",
            transform_dtype_key="position_3d",
        )
    )
    if has_clean_targets:
        collated["ball_uv_target"] = torch.stack(ball_uv_target_batch, dim=0)
        collated["ball_vis_target"] = torch.stack(ball_vis_target_batch, dim=0)
    return cast("BLCSMultiViewBatch", collated)

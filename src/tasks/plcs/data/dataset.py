"""Unified dataset for PLCS frame/sequence/single/multiview modes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from src.tasks.base.data.scene_dataset import (
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
)
from src.tasks.base.generate_dataset import (
    CourtReferenceFrameProvenance,
    camera_extrinsics_physical_to_target,
)
from src.tasks.plcs.court_keypoint_contract import (
    PLCSCourtKeypointRuntimeConfig,
    align_selected_court_array,
    choose_reference_provenance,
    court_keypoint_contract_document,
    normalized_headings_physical_to_target,
    normalized_points_physical_to_target,
    selected_court_views,
    validate_plcs_dataset_court_keypoints,
    world_joints_physical_to_target,
)
from src.tasks.plcs.data.augmentation import PLCSObservationAugmentation
from src.tasks.plcs.data.targets import build_coco17_world_targets
from src.tasks.plcs.data.types import PLCSBatch
from src.utils.projection.camera_projector import camera_from_mapping
from src.utils.schema.court_normalization import (
    validate_court_coordinate_normalization,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneDataset(SceneDatasetBase[dict[str, Tensor]]):
    """Unified PLCS dataset – scene-level indexing.

    Returns per-sample tensors with camera-time ordering:
    - human_kp: (N, T, 17, 2)
    - court_kp: (N, T, 20, 2)
    - human_vis: (N, T, 17)
    - court_vis: (N, T, 20)
    - padding_mask: (N, T), True for padding
    - position: (T, 3)
    - rotation: (T, 2)
    """

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: DictConfig,
        augment: bool = True,
    ) -> None:
        self.hydra_cfg = config
        self.augment = augment
        self.court_keypoint_contract = (
            PLCSCourtKeypointRuntimeConfig.from_config(config).contract
        )
        self.court_keypoint_validation = validate_plcs_dataset_court_keypoints(
            scene_dir,
            split_file,
            self.court_keypoint_contract,
        )
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self._configure_task(data_cfg)
        super().__init__(
            config=self._build_scene_dataset_config(
                scene_dir=scene_dir,
                split_file=split_file,
                data_cfg=data_cfg,
            )
        )

    # -- Composed-method hooks ------------------------------------------

    def _configure_task(self, data_cfg: dict) -> None:
        from omegaconf import DictConfig

        self.camera_mode_plcs = str(data_cfg["camera_mode"])
        self.is_multiview = str(data_cfg["mode"]) in {
            "multiview",
            "multiview_sequence",
        }

        r = data_cfg["num_views_range"]
        self._plcs_num_views_range: tuple[int, int] = (int(r[0]), int(r[1]))
        r = data_cfg["seq_len_range"]
        self._plcs_seq_len_range: tuple[int, int] = (int(r[0]), int(r[1]))

        augmentation_cfg = data_cfg["augmentation"]
        if not isinstance(augmentation_cfg, (dict, DictConfig)):
            raise ValueError("data.augmentation must be a mapping-like config.")
        self.augmentation = PLCSObservationAugmentation(augmentation_cfg)
        # Number of court keypoints to use (first N from the canonical order)
        self.num_court_kp = int(data_cfg["num_court_kp"])

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
            seq_len_range=self._plcs_seq_len_range,
            num_views_range=self._plcs_num_views_range,
            camera_mode=self.camera_mode_plcs,
            crop_mode=("random" if self.augment else "center"),
            min_num_frames=self._plcs_seq_len_range[0],
            min_num_cameras=self._plcs_num_views_range[0],
        )

    def _validate_scene_metadata(self, meta: dict, *, path: Path) -> None:
        validate_court_coordinate_normalization(meta, artifact=f"Scene {path}")

    def build_sample(self, scene: Scene) -> dict[str, Any]:
        cams = self.select_cameras(
            scene,
            num_views_range=self._plcs_num_views_range,
            camera_mode=self.camera_mode_plcs,
        )
        # Resolve effective frame count from arrays
        pos_len = int(scene.data["position"].shape[0])
        rot_len = int(scene.data["rotation"].shape[0])
        primary_len = int(scene.get_camera_array(cams.primary, "human_kp_uv").shape[0])
        full_len = scene.effective_num_frames(primary_len, pos_len, rot_len)
        window = self.select_window(scene, full_len=full_len)

        human_kp_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        human_vis_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []
        human_kp_target_list: list[Tensor] = []
        human_vis_target_list: list[Tensor] = []
        camera_R_list: list[Tensor] = []
        camera_C_list: list[Tensor] = []
        camera_f_list: list[Tensor] = []
        camera_cx_list: list[Tensor] = []
        camera_cy_list: list[Tensor] = []
        camera_w_list: list[Tensor] = []
        camera_h_list: list[Tensor] = []

        views = selected_court_views(
            self.court_keypoint_validation,
            scene.path,
            cams.indices,
        )
        provenance, reference_view = choose_reference_provenance(
            self.court_keypoint_contract,
            views,
            rng=self.rng if self.augment else None,
        )

        for local_index, cam_idx in enumerate(cams.indices):
            human_kp = torch.from_numpy(
                scene.get_camera_array(cam_idx, "human_kp_uv", window=window)
            ).float()
            court_kp_array = scene.get_camera_array(
                cam_idx, "court_kp_uv", window=window
            )
            human_vis = torch.from_numpy(
                scene.get_camera_array(cam_idx, "human_kp_vis", window=window)
            ).float()
            court_vis_array = scene.get_camera_array(
                cam_idx, "court_kp_vis", window=window
            )
            source_view = views[local_index] if views else None
            court_kp = torch.from_numpy(
                align_selected_court_array(
                    court_kp_array,
                    source_view,
                    reference_view,
                    keypoint_axis=-2,
                )
            ).float()
            court_vis = torch.from_numpy(
                align_selected_court_array(
                    court_vis_array,
                    source_view,
                    reference_view,
                    keypoint_axis=-1,
                )
            ).float()
            court_kp = court_kp[..., : self.num_court_kp, :]
            court_vis = court_vis[..., : self.num_court_kp]

            human_kp = human_kp * human_vis.unsqueeze(-1)
            court_kp = court_kp * court_vis.unsqueeze(-1)

            human_kp_list.append(human_kp)
            court_kp_list.append(court_kp)
            human_vis_list.append(human_vis)
            court_vis_list.append(court_vis)
            human_kp_target_list.append(human_kp.clone())
            human_vis_target_list.append(human_vis.clone())

            params_key = f"cam_{cam_idx}_params"
            raw_camera = scene.data.get(params_key)
            if not isinstance(raw_camera, Mapping):
                raise TypeError(
                    f"{scene.path}: {params_key} must be a camera-parameter mapping."
                )
            camera = camera_from_mapping(dict(raw_camera))
            transformed_C, transformed_R = camera_extrinsics_physical_to_target(
                camera.C,
                camera.R,
                provenance,
            )
            camera_R_list.append(transformed_R)
            camera_C_list.append(transformed_C)
            camera_f_list.append(torch.tensor(camera.f, dtype=torch.float32))
            camera_cx_list.append(torch.tensor(camera.cx, dtype=torch.float32))
            camera_cy_list.append(torch.tensor(camera.cy, dtype=torch.float32))
            camera_w_list.append(torch.tensor(float(camera.w), dtype=torch.float32))
            camera_h_list.append(torch.tensor(float(camera.h), dtype=torch.float32))

        position = torch.from_numpy(scene.get_array("position", window=window)).float()
        rotation = torch.from_numpy(scene.get_array("rotation", window=window)).float()
        position = normalized_points_physical_to_target(
            position,
            provenance,
        )
        rotation = normalized_headings_physical_to_target(rotation, provenance)

        sample: dict[str, Any] = {
            "human_kp": torch.stack(human_kp_list, dim=0),
            "court_kp": torch.stack(court_kp_list, dim=0),
            "human_vis": torch.stack(human_vis_list, dim=0),
            "court_vis": torch.stack(court_vis_list, dim=0),
            "padding_mask": torch.zeros(
                len(cams.indices),
                window.seq_len,
                dtype=torch.bool,
            ),
            "position": position,
            "rotation": rotation,
            "human_kp_target": torch.stack(human_kp_target_list, dim=0),
            "human_vis_target": torch.stack(human_vis_target_list, dim=0),
            "camera_R": torch.stack(camera_R_list, dim=0),
            "camera_C": torch.stack(camera_C_list, dim=0),
            "camera_f": torch.stack(camera_f_list, dim=0),
            "camera_cx": torch.stack(camera_cx_list, dim=0),
            "camera_cy": torch.stack(camera_cy_list, dim=0),
            "camera_w": torch.stack(camera_w_list, dim=0),
            "camera_h": torch.stack(camera_h_list, dim=0),
        }

        # Build COCO17 world targets from raw NPZ data
        # scene.data["meta"] is a raw numpy scalar (JSON string); use scene.meta
        # (already parsed dict) so build_coco17_world_targets gets a plain dict.
        payload_for_targets = {**scene.data, "meta": scene.meta}
        human_kp_3d = build_coco17_world_targets(payload_for_targets)
        sample["human_kp_3d"] = world_joints_physical_to_target(
            torch.from_numpy(human_kp_3d[window.sl].copy()).float(),
            provenance,
        )
        sample["court_keypoint_metadata"] = court_keypoint_contract_document(
            self.court_keypoint_contract
        )
        sample["court_reference_provenance"] = provenance
        sample["selected_camera_ids"] = tuple(
            view.camera_id for view in views
        ) or tuple(f"camera_{index}" for index in cams.indices)

        return sample

    def augment_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.augment:
            return sample
        augmented: dict[str, Any] = self.augmentation.forward(sample)
        return augmented


def collate_plcs_batch(batch: list[dict[str, Any]]) -> PLCSBatch | dict[str, Any]:
    """Collate variable-view/variable-length PLCS samples into a padded batch."""
    reprojection_keys = (
        "human_kp_target",
        "human_vis_target",
        "camera_f",
        "camera_cx",
        "camera_cy",
        "camera_w",
        "camera_h",
    )
    reprojection_presence = [
        tuple(key in sample for key in reprojection_keys) for sample in batch
    ]
    if any(any(presence) and not all(presence) for presence in reprojection_presence):
        raise ValueError(
            "PLCS reprojection sample fields must be provided as one complete group."
        )
    complete_reprojection = [all(presence) for presence in reprojection_presence]
    if any(complete_reprojection) and not all(complete_reprojection):
        raise ValueError(
            "PLCS batch cannot mix samples with and without reprojection targets."
        )
    has_reprojection = all(complete_reprojection)

    max_views = max(int(sample["human_kp"].shape[0]) for sample in batch)
    max_seq_len = max(int(sample["human_kp"].shape[1]) for sample in batch)

    human_kp_batch = []
    court_kp_batch = []
    human_vis_batch = []
    court_vis_batch = []
    padding_mask_batch = []
    position_batch = []
    rotation_batch = []
    human_kp_3d_batch = []
    human_kp_target_batch: list[Tensor] = []
    human_vis_target_batch: list[Tensor] = []
    camera_R_batch: list[Tensor] = []
    camera_C_batch: list[Tensor] = []
    camera_f_batch: list[Tensor] = []
    camera_cx_batch: list[Tensor] = []
    camera_cy_batch: list[Tensor] = []
    camera_w_batch: list[Tensor] = []
    camera_h_batch: list[Tensor] = []

    for sample in batch:
        n_views = int(sample["human_kp"].shape[0])
        seq_len = int(sample["human_kp"].shape[1])
        pad_views = max_views - n_views
        pad_seq = max_seq_len - seq_len

        human_kp = sample["human_kp"]
        court_kp = sample["court_kp"]
        n_kp = int(court_kp.shape[2])
        human_vis = sample["human_vis"]
        court_vis = sample["court_vis"]
        padding_mask = sample["padding_mask"]
        position = sample["position"]
        rotation = sample["rotation"]
        human_kp_3d = sample.get("human_kp_3d")
        camera_R = cast(Tensor, sample["camera_R"])
        camera_C = cast(Tensor, sample["camera_C"])
        if has_reprojection:
            human_kp_target = sample["human_kp_target"]
            human_vis_target = sample["human_vis_target"]
            camera_f = sample["camera_f"]
            camera_cx = sample["camera_cx"]
            camera_cy = sample["camera_cy"]
            camera_w = sample["camera_w"]
            camera_h = sample["camera_h"]

        if pad_seq > 0:
            human_kp = torch.cat(
                [human_kp, torch.zeros(n_views, pad_seq, 17, 2)], dim=1
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(n_views, pad_seq, n_kp, 2)], dim=1
            )
            human_vis = torch.cat([human_vis, torch.zeros(n_views, pad_seq, 17)], dim=1)
            court_vis = torch.cat(
                [court_vis, torch.zeros(n_views, pad_seq, n_kp)], dim=1
            )
            padding_mask = torch.cat(
                [padding_mask, torch.ones(n_views, pad_seq, dtype=torch.bool)], dim=1
            )
            position = torch.cat([position, torch.zeros(pad_seq, 3)], dim=0)
            rotation = torch.cat([rotation, torch.zeros(pad_seq, 2)], dim=0)
            if human_kp_3d is not None:
                human_kp_3d = torch.cat(
                    [human_kp_3d, torch.zeros(pad_seq, 17, 3)], dim=0
                )
            if has_reprojection:
                human_kp_target = torch.cat(
                    [human_kp_target, torch.zeros(n_views, pad_seq, 17, 2)],
                    dim=1,
                )
                human_vis_target = torch.cat(
                    [human_vis_target, torch.zeros(n_views, pad_seq, 17)],
                    dim=1,
                )

        if pad_views > 0:
            human_kp = torch.cat(
                [human_kp, torch.zeros(pad_views, max_seq_len, 17, 2)], dim=0
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(pad_views, max_seq_len, n_kp, 2)], dim=0
            )
            human_vis = torch.cat(
                [human_vis, torch.zeros(pad_views, max_seq_len, 17)], dim=0
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(pad_views, max_seq_len, n_kp)], dim=0
            )
            padding_mask = torch.cat(
                [
                    padding_mask,
                    torch.ones(pad_views, max_seq_len, dtype=torch.bool),
                ],
                dim=0,
            )
            camera_R = torch.cat([camera_R, torch.zeros(pad_views, 3, 3)], dim=0)
            camera_C = torch.cat([camera_C, torch.zeros(pad_views, 3)], dim=0)
            if has_reprojection:
                human_kp_target = torch.cat(
                    [
                        human_kp_target,
                        torch.zeros(pad_views, max_seq_len, 17, 2),
                    ],
                    dim=0,
                )
                human_vis_target = torch.cat(
                    [human_vis_target, torch.zeros(pad_views, max_seq_len, 17)],
                    dim=0,
                )
                camera_f = torch.cat([camera_f, torch.zeros(pad_views)], dim=0)
                camera_cx = torch.cat([camera_cx, torch.zeros(pad_views)], dim=0)
                camera_cy = torch.cat([camera_cy, torch.zeros(pad_views)], dim=0)
                camera_w = torch.cat([camera_w, torch.ones(pad_views)], dim=0)
                camera_h = torch.cat([camera_h, torch.ones(pad_views)], dim=0)

        human_kp_batch.append(human_kp)
        court_kp_batch.append(court_kp)
        human_vis_batch.append(human_vis)
        court_vis_batch.append(court_vis)
        padding_mask_batch.append(padding_mask)
        position_batch.append(position)
        rotation_batch.append(rotation)
        if human_kp_3d is not None:
            human_kp_3d_batch.append(human_kp_3d)
        camera_R_batch.append(camera_R)
        camera_C_batch.append(camera_C)
        if has_reprojection:
            human_kp_target_batch.append(human_kp_target)
            human_vis_target_batch.append(human_vis_target)
            camera_f_batch.append(camera_f)
            camera_cx_batch.append(camera_cx)
            camera_cy_batch.append(camera_cy)
            camera_w_batch.append(camera_w)
            camera_h_batch.append(camera_h)

    collated: dict[str, Any] = {
        "human_kp": torch.stack(human_kp_batch, dim=0),
        "court_kp": torch.stack(court_kp_batch, dim=0),
        "human_vis": torch.stack(human_vis_batch, dim=0),
        "court_vis": torch.stack(court_vis_batch, dim=0),
        "padding_mask": torch.stack(padding_mask_batch, dim=0),
        "position": torch.stack(position_batch, dim=0),
        "rotation": torch.stack(rotation_batch, dim=0),
        "camera_C": torch.stack(camera_C_batch, dim=0),
        "camera_R": torch.stack(camera_R_batch, dim=0),
    }
    if human_kp_3d_batch:
        collated["human_kp_3d"] = torch.stack(human_kp_3d_batch, dim=0)
    if has_reprojection:
        collated.update(
            {
                "human_kp_target": torch.stack(human_kp_target_batch, dim=0),
                "human_vis_target": torch.stack(human_vis_target_batch, dim=0),
                "camera_f": torch.stack(camera_f_batch, dim=0),
                "camera_cx": torch.stack(camera_cx_batch, dim=0),
                "camera_cy": torch.stack(camera_cy_batch, dim=0),
                "camera_w": torch.stack(camera_w_batch, dim=0),
                "camera_h": torch.stack(camera_h_batch, dim=0),
            }
        )

    metadata_keys = (
        "court_keypoint_metadata",
        "court_reference_provenance",
        "selected_camera_ids",
    )
    for key in metadata_keys:
        missing = [index for index, sample in enumerate(batch) if key not in sample]
        if missing:
            raise ValueError(
                f"PLCS batch is missing required {key!r} for samples {missing!r}."
            )
    collated["court_keypoint_metadata"] = tuple(
        sample["court_keypoint_metadata"] for sample in batch
    )
    collated["court_reference_provenance"] = tuple(
        cast(CourtReferenceFrameProvenance, sample["court_reference_provenance"])
        for sample in batch
    )
    collated["selected_camera_ids"] = tuple(
        cast(tuple[str, ...], sample["selected_camera_ids"]) for sample in batch
    )

    return collated

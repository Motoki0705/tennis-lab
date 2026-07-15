"""Dataset and collate/adaptation utilities for BLCS."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data.court_lines import CourtLineMapBuilder, CourtLineMapConfig
from src.tasks.base.data.scene_dataset import (
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
)
from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.blcs.data.types import BLCSBatch, BLCSMultiViewBatch, BLCSMultiViewSample

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallTrajectoryDataset(SceneDatasetBase[BLCSMultiViewSample]):
    """Unified BLCS dataset that always returns canonical multiview samples.

    The canonical sample format keeps camera and temporal dimensions:
    - ball_uv: (N, T, 2)
    - ball_vis: (N, T)
    - ball_mask: (N, T)
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
        config: DictConfig | None = None,
        augment: bool = True,
    ) -> None:
        self.hydra_cfg = config or {}
        self.augment = augment
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
        # Multiview ranges
        self.seq_len_range = self._parse_int_range(data_cfg, "seq_len_range")
        self.num_views_range = self._parse_int_range(data_cfg, "num_views_range")
        self.camera_mode = self._parse_camera_mode(data_cfg)

        # Number of court keypoints to use (first N from the canonical order)
        self.num_court_kp = int(data_cfg.get("num_court_kp", 20))
        self.court_input_type = str(data_cfg.get("court_input_type", "kp"))
        if self.court_input_type not in {"kp", "line"}:
            raise ValueError(
                "data.court_input_type must be 'kp' or 'line', got "
                f"{self.court_input_type!r}."
            )
        self.court_line_map_builder: CourtLineMapBuilder | None = None
        if self.court_input_type == "line":
            line_cfg = data_cfg.get("court_line")
            if not isinstance(line_cfg, Mapping):
                raise ValueError("data.court_line must be a mapping in line mode.")
            self.court_line_map_builder = CourtLineMapBuilder(
                CourtLineMapConfig.from_mapping(line_cfg)
            )

        # Augmentation pipeline
        aug_cfg = data_cfg.get("augmentation", {})
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
        )

    def build_sample(self, scene: Scene) -> BLCSMultiViewSample:
        cams = self.select_cameras(
            scene, num_views_range=self.num_views_range, camera_mode=self.camera_mode
        )
        # Use camera trajectory length to guard against metadata drift.
        primary_len = int(scene.get_camera_array(cams.primary, "ball_uv").shape[0])
        pos_len = int(scene.data["ball_pos_norm"].shape[0])
        vel_len = int(scene.data["ball_vel_world"].shape[0])
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

        for cam_idx in cams.indices:
            ball_uv = torch.from_numpy(
                scene.get_camera_array(cam_idx, "ball_uv", window=window)
            ).float()
            ball_vis = torch.from_numpy(
                scene.get_camera_array(cam_idx, "ball_visible", window=window)
            ).float()
            court_kp = torch.from_numpy(
                scene.get_camera_array(cam_idx, "court_kp_uv")
            ).float()
            court_vis = torch.from_numpy(
                scene.get_camera_array(cam_idx, "court_kp_visible")
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
            cam_R_list.append(torch.tensor(cam_params["R"], dtype=torch.float32))
            cam_C_list.append(torch.tensor(cam_params["C"], dtype=torch.float32))
            cam_f_list.append(torch.tensor(cam_params["f"], dtype=torch.float32))
            cam_cx_list.append(torch.tensor(cam_params["cx"], dtype=torch.float32))
            cam_cy_list.append(torch.tensor(cam_params["cy"], dtype=torch.float32))
            cam_w_list.append(torch.tensor(float(cam_params["w"]), dtype=torch.float32))
            cam_h_list.append(torch.tensor(float(cam_params["h"]), dtype=torch.float32))

        sample: BLCSMultiViewSample = {
            "ball_uv": torch.stack(ball_uv_list, dim=0),
            "ball_vis": torch.stack(ball_vis_list, dim=0),
            "ball_mask": torch.ones(
                len(cams.indices), window.seq_len, dtype=torch.float32
            ),
            "court_kp": torch.stack(court_kp_list, dim=0),
            "court_vis": torch.stack(court_vis_list, dim=0),
            "position_3d": torch.from_numpy(
                scene.get_array("ball_pos_norm", window=window)
            ).float(),
            "velocity_3d": torch.from_numpy(
                scene.get_array("ball_vel_world", window=window)
            ).float(),
            "seq_len": torch.tensor(window.seq_len, dtype=torch.long),
            "camera_R": torch.stack(cam_R_list, dim=0),
            "camera_C": torch.stack(cam_C_list, dim=0),
            "camera_f": torch.stack(cam_f_list, dim=0),
            "camera_cx": torch.stack(cam_cx_list, dim=0),
            "camera_cy": torch.stack(cam_cy_list, dim=0),
            "camera_w": torch.stack(cam_w_list, dim=0),
            "camera_h": torch.stack(cam_h_list, dim=0),
        }
        return sample

    def _apply_augmentation_multiview(
        self, sample: BLCSMultiViewSample
    ) -> BLCSMultiViewSample:
        return cast(
            BLCSMultiViewSample, self.augmentation_pipeline.forward(sample)
        )

    def augment_sample(self, sample: BLCSMultiViewSample) -> BLCSMultiViewSample:
        out = self._apply_augmentation_multiview(sample) if self.augment else sample
        if self.court_input_type == "kp":
            return out
        if self.augment:
            out["court_kp"] = sample["court_kp"]
            out["court_vis"] = sample["court_vis"]
        if self.court_line_map_builder is None:
            raise RuntimeError("court_line_map_builder is not initialized in line mode.")
        seed = int(torch.randint(0, 2**31 - 1, ()).item()) if self.augment else 0
        court_kp = out.get("court_kp")
        if court_kp is None:
            raise KeyError("court_kp is required to synthesize court_line_map.")
        result = dict(out)
        result["court_line_map"] = self.court_line_map_builder.build(
            court_kp,
            augment=self.augment,
            rng=np.random.default_rng(seed),
        )
        result.pop("court_kp", None)
        result.pop("court_vis", None)
        return result  # type: ignore[return-value]


def collate_multiview_trajectories(
    batch: list[BLCSMultiViewSample],
) -> BLCSMultiViewBatch:
    """Collate canonical BLCS samples into padded canonical batch tensors."""
    max_views = max(int(sample["ball_uv"].shape[0]) for sample in batch)
    max_seq_len = max(int(sample["seq_len"].item()) for sample in batch)
    has_clean_targets = any(
        "ball_uv_target" in sample and "ball_vis_target" in sample for sample in batch
    )
    has_court_line_map = "court_line_map" in batch[0]
    if any(
        ("court_line_map" in sample) != has_court_line_map for sample in batch
    ):
        raise ValueError("A batch cannot mix KP and line court-input samples.")

    ball_uv_batch = []
    ball_vis_batch = []
    ball_uv_target_batch = []
    ball_vis_target_batch = []
    ball_mask_batch = []
    court_kp_batch = []
    court_vis_batch = []
    court_line_map_batch = []
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
        ball_mask = sample["ball_mask"]
        court_kp = sample.get("court_kp")
        court_vis = sample.get("court_vis")
        court_line_map = sample.get("court_line_map")
        position_3d = sample["position_3d"]
        velocity_3d = sample["velocity_3d"]
        if has_court_line_map:
            if court_line_map is None or court_kp is not None or court_vis is not None:
                raise ValueError(
                    "Line samples must contain only court_line_map court input."
                )
            if court_line_map.ndim != 5 or court_line_map.shape[2] != 1:
                raise ValueError(
                    "court_line_map must have shape (N,T,1,H,W)."
                )
            map_height, map_width = court_line_map.shape[-2:]
        else:
            if court_kp is None or court_vis is None:
                raise ValueError("KP samples require court_kp and court_vis.")
            n_kp = int(court_kp.shape[-2])

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
            ball_mask = torch.cat([ball_mask, torch.zeros(n_views, pad_seq)], dim=1)
            if has_court_line_map:
                assert court_line_map is not None
                court_line_map = torch.cat(
                    [
                        court_line_map,
                        torch.zeros(n_views, pad_seq, 1, map_height, map_width),
                    ],
                    dim=1,
                )
            else:
                assert court_kp is not None and court_vis is not None
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
            ball_mask = torch.cat(
                [ball_mask, torch.zeros(pad_views, max_seq_len)], dim=0
            )
            if has_court_line_map:
                assert court_line_map is not None
                court_line_map = torch.cat(
                    [
                        court_line_map,
                        torch.zeros(
                            pad_views, max_seq_len, 1, map_height, map_width
                        ),
                    ],
                    dim=0,
                )
            else:
                assert court_kp is not None and court_vis is not None
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
        ball_mask_batch.append(ball_mask)
        if has_court_line_map:
            assert court_line_map is not None
            court_line_map_batch.append(court_line_map)
        else:
            assert court_kp is not None and court_vis is not None
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
    collated = {
        "ball_uv": torch.stack(ball_uv_batch, dim=0),
        "ball_vis": torch.stack(ball_vis_batch, dim=0),
        "ball_mask": torch.stack(ball_mask_batch, dim=0),
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
    }
    if has_court_line_map:
        collated["court_line_map"] = torch.stack(court_line_map_batch, dim=0)
    else:
        collated["court_kp"] = torch.stack(court_kp_batch, dim=0)
        collated["court_vis"] = torch.stack(court_vis_batch, dim=0)
    if has_clean_targets:
        collated["ball_uv_target"] = torch.stack(ball_uv_target_batch, dim=0)
        collated["ball_vis_target"] = torch.stack(ball_vis_target_batch, dim=0)
    return cast(BLCSMultiViewBatch, collated)


def adapt_batch_for_model_profile(
    batch: BLCSMultiViewBatch,
    *,
    input_profile: str,
) -> BLCSBatch | BLCSMultiViewBatch:
    """Adapt canonical BLCS batch ``(B,N,T,...)`` to model input profile."""
    _, n, _ = batch["ball_uv"].shape[:3]
    if n <= 0:
        raise ValueError("Expected at least one camera view in batch.")

    if input_profile == "multiview":
        return batch
    if input_profile == "single":
        adapted = {
            "ball_uv": batch["ball_uv"][:, 0],
            "ball_vis": batch["ball_vis"][:, 0],
            "ball_mask": batch["ball_mask"][:, 0],
            "position_3d": batch["position_3d"],
            "velocity_3d": batch["velocity_3d"],
            "seq_len": batch["seq_len"],
            "camera_R": batch["camera_R"][:, :1],
            "camera_C": batch["camera_C"][:, :1],
            "camera_f": batch["camera_f"][:, :1],
            "camera_cx": batch["camera_cx"][:, :1],
            "camera_cy": batch["camera_cy"][:, :1],
            "camera_w": batch["camera_w"][:, :1],
            "camera_h": batch["camera_h"][:, :1],
        }
        if "court_line_map" in batch:
            adapted["court_line_map"] = batch["court_line_map"][:, 0]
        else:
            adapted["court_kp"] = batch["court_kp"][:, 0, 0]
            adapted["court_vis"] = batch["court_vis"][:, 0, 0]
        if "ball_uv_target" in batch and "ball_vis_target" in batch:
            adapted["ball_uv_target"] = batch["ball_uv_target"][:, :1]
            adapted["ball_vis_target"] = batch["ball_vis_target"][:, :1]
        return cast(BLCSBatch | BLCSMultiViewBatch, adapted)
    raise ValueError(
        "Unknown model input profile: "
        f"{input_profile}. Supported: ['single', 'multiview']"
    )


def collate_and_adapt_blcs_batch(
    batch: list[BLCSMultiViewSample],
    *,
    input_profile: str,
) -> BLCSBatch | BLCSMultiViewBatch:
    """Collate canonical BLCS samples and adapt to model input profile."""
    collated = collate_multiview_trajectories(batch)
    return adapt_batch_for_model_profile(collated, input_profile=input_profile)

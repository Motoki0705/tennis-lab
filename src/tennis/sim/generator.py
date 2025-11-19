"""Synthetic scene generator for tennis pose (P1 minimal implementation).

Generates a simple scene with:
- Court 3D keypoints
- One or more cameras sampled on the fence
- A static, plausible 3D human skeleton + racket points
- 2D projections with Gaussian noise and random visibility drops

Note: This is a starter generator for P1. Future phases will replace the
static skeleton with retargeted motions and richer camera models.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch
from torch import Tensor

from src.tennis.geometry.court import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    court_keypoints_3d,
    make_look_at_camera,
    project_points,
    sample_camera_position_on_fence,
)
from src.tennis.geometry.skeleton import VITPOSE_17_NAMES, RACKET_3_NAMES


@dataclass(slots=True)
class GenConfig:
    fps: int = 60
    duration_sec: float = 3.0
    num_cameras: int = 4
    image_size: tuple[int, int] = (1280, 720)
    # noise ratios
    person_sigma_ratio_torso: float = 0.02
    person_sigma_ratio_extremity: float = 0.03
    person_sigma_ratio_head: float = 0.015
    racket_sigma_ratio: float = 0.04
    court_sigma_px_ratio: float = 0.003  # of image height
    # missing probabilities
    p_missing_extremity: float = 0.10
    p_missing_torso: float = 0.02
    p_missing_racket: float = 0.15
    p_missing_court: float = 0.01
    p_camera_drop: float = 0.05
    seed: int | None = 1234


def _static_person_racket_3d() -> tuple[Tensor, Tensor]:
    """Return a simple static 3D human+raket configuration in court coords.

    The pose is a rough, standing configuration with ~1.7m height.
    """
    # Pelvis/world anchor
    pelvis = torch.tensor([0.0, 0.0, 1.0])
    # Hips and shoulders
    left_hip = pelvis + torch.tensor([-0.12, 0.0, 0.0])
    right_hip = pelvis + torch.tensor([+0.12, 0.0, 0.0])
    left_knee = left_hip + torch.tensor([0.0, 0.0, -0.42])
    right_knee = right_hip + torch.tensor([0.0, 0.0, -0.42])
    left_ankle = left_knee + torch.tensor([0.0, 0.0, -0.42])
    right_ankle = right_knee + torch.tensor([0.0, 0.0, -0.42])
    left_shoulder = pelvis + torch.tensor([-0.14, 0.0, 0.35])
    right_shoulder = pelvis + torch.tensor([+0.14, 0.0, 0.35])
    left_elbow = left_shoulder + torch.tensor([-0.18, 0.0, -0.08])
    right_elbow = right_shoulder + torch.tensor([+0.18, 0.0, -0.08])
    left_wrist = left_elbow + torch.tensor([-0.18, 0.0, 0.0])
    right_wrist = right_elbow + torch.tensor([+0.18, 0.0, 0.0])
    nose = pelvis + torch.tensor([0.0, 0.0, 0.55])
    left_eye = nose + torch.tensor([-0.03, 0.0, 0.02])
    right_eye = nose + torch.tensor([+0.03, 0.0, 0.02])
    left_ear = nose + torch.tensor([-0.06, 0.0, 0.02])
    right_ear = nose + torch.tensor([+0.06, 0.0, 0.02])

    person_list = [
        nose,
        left_eye,
        right_eye,
        left_ear,
        right_ear,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
    ]
    person = torch.stack(person_list, dim=0).float()

    # Racket as rigid relative to right wrist pointing upward
    racket_handle = right_wrist + torch.tensor([0.05, 0.0, 0.0])
    racket_throat = racket_handle + torch.tensor([0.20, 0.0, 0.05])
    racket_head_top = racket_throat + torch.tensor([0.20, 0.0, 0.20])
    racket = torch.stack([racket_handle, racket_throat, racket_head_top], dim=0).float()
    return person, racket


def _noise_2d_points(uv: Tensor, mask: Tensor, sigma_px: Tensor) -> Tensor:
    noise = torch.randn_like(uv) * sigma_px.view(-1, 1)
    noisy = uv + noise
    return torch.where(mask.view(-1, 1), noisy, uv)


def _drop_visibility(vis: Tensor, p: float) -> Tensor:
    if p <= 0.0:
        return vis
    drops = torch.rand_like(vis, dtype=torch.float32) < float(p)
    return torch.where(drops, torch.zeros_like(vis), vis)


def _person_sigma_per_joint(px_height: float, num_joints: int) -> Tensor:
    # Approximate categories: torso/head/extremities
    # indices based on COCO17 layout
    sigma = torch.full((num_joints,), 0.02 * px_height, dtype=torch.float32)
    # extremities
    for idx in [9, 10, 15, 16]:
        sigma[idx] = 0.03 * px_height
    # head
    for idx in [0, 1, 2, 3, 4]:
        sigma[idx] = 0.015 * px_height
    return sigma


class TennisPoseSceneGenerator:
    def __init__(self, cfg: GenConfig | None = None) -> None:
        self.cfg = cfg or GenConfig()
        if self.cfg.seed is not None:
            random.seed(int(self.cfg.seed))
            torch.manual_seed(int(self.cfg.seed))

    def _sample_camera(self) -> Mapping[str, Any]:
        side = random.choice(["near", "far", "left", "right"])  # noqa: S311
        t = random.random()  # noqa: S311
        x, y, z = sample_camera_position_on_fence(t, side)
        cam = make_look_at_camera((x, y, z), image_size=self.cfg.image_size)
        return {
            "id": f"{side}-{t:.2f}",
            "image_size": [cam.w, cam.h],
            "_cam_internal": cam,  # kept for projection, not serialized later
        }

    def _build_cameras(self, n: int) -> List[Mapping[str, Any]]:
        return [self._sample_camera() for _ in range(n)]

    def generate_scene(self, scene_id: str | int) -> Mapping[str, Any]:
        fps = int(self.cfg.fps)
        frames_total = max(1, int(round(self.cfg.duration_sec * fps)))
        cameras = self._build_cameras(int(self.cfg.num_cameras))

        court3d = court_keypoints_3d()
        person3d, racket3d = _static_person_racket_3d()

        frames: List[Dict[str, Any]] = []
        for t_idx in range(frames_total):
            frame_payload: Dict[str, Any] = {}
            for cam_idx, cam_entry in enumerate(cameras):
                cam = cam_entry["_cam_internal"]
                # Project 3D
                uv_court, mask_court = project_points(cam, court3d)
                uv_person, mask_person = project_points(cam, person3d)
                uv_racket, mask_racket = project_points(cam, racket3d)

                # Image size
                w, h = cam.w, cam.h

                # Person noise based on pixel height
                if uv_person.numel() > 0:
                    v_vals = uv_person[:, 1]
                    H_person = (v_vals.max() - v_vals.min()).abs().item()
                else:
                    H_person = float(h)
                sigma_person = _person_sigma_per_joint(H_person, uv_person.shape[0])
                sigma_racket = torch.full((uv_racket.shape[0],), 0.04 * H_person)
                sigma_court = torch.full((uv_court.shape[0],), 0.003 * float(h))

                uv_court = _noise_2d_points(uv_court, mask_court, sigma_court)
                uv_person = _noise_2d_points(uv_person, mask_person, sigma_person)
                uv_racket = _noise_2d_points(uv_racket, mask_racket, sigma_racket)

                vis_court = mask_court.to(torch.uint8)
                vis_person = mask_person.to(torch.uint8)
                vis_racket = mask_racket.to(torch.uint8)

                # Random visibility drops
                if random.random() < self.cfg.p_camera_drop:  # noqa: S311
                    vis_court[:] = 0
                    vis_person[:] = 0
                    vis_racket[:] = 0
                else:
                    vis_court = _drop_visibility(vis_court, self.cfg.p_missing_court)
                    # torso drops fewer than extremities
                    vis_person = _drop_visibility(vis_person, self.cfg.p_missing_torso)
                    for j in [9, 10, 15, 16]:
                        if random.random() < self.cfg.p_missing_extremity:  # noqa: S311
                            vis_person[j] = 0
                    vis_racket = _drop_visibility(vis_racket, self.cfg.p_missing_racket)

                cam_key = f"cam_{cam_idx}"
                frame_payload[cam_key] = {
                    "court_keypoints_2d": {
                        "points": uv_court.tolist(),
                        "visibility": vis_court.tolist(),
                    },
                    "player_keypoints_2d": {
                        "joints": uv_person.tolist(),
                        "visibility": vis_person.tolist(),
                    },
                    "racket_keypoints_2d": {
                        "points": uv_racket.tolist(),
                        "visibility": vis_racket.tolist(),
                    },
                }
            # Include GT 3D for synthetic data
            frame_payload["player_joints_3d"] = person3d.tolist()
            frame_payload["racket_points_3d"] = racket3d.tolist()
            frames.append(frame_payload)

        # Serialize cameras without internal objects
        cameras_pub = [
            {"id": c["id"], "image_size": c["image_size"]} for c in cameras
        ]
        return {
            "scene_id": str(scene_id),
            "fps": fps,
            "num_cameras": len(cameras_pub),
            "cameras": cameras_pub,
            "frames": frames,
        }


def write_scene_json(path: Path | str, scene: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)


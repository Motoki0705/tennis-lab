"""Synthetic scene generator that instantiates 3DTennisDS clips on a court."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from src.tennis.sim.assets import AssetSample, TennisAssetLibrary


@dataclass(slots=True)
class GenConfig:
    """Configuration values consumed by :class:`TennisPoseSceneGenerator`."""

    fps: int = 60
    duration_sec: float = 3.0
    num_cameras: int = 4
    image_size: tuple[int, int] = (1280, 720)
    asset_root: str = "data/raw/3dtennisds"
    asset_min_frames: int = 30
    asset_max_files: int | None = None
    min_players: int = 1
    max_players: int = 20
    player_min_separation: float = 1.5
    spawn_margin_x: float = 0.5
    spawn_margin_y: float = 0.5
    max_anchor_attempts: int = 2000
    seed: int | None = 1234


@dataclass(slots=True)
class _PlayerSequence:
    """Container for a single player's projected trajectory."""

    joints_3d: Tensor  # (T, 17, 3)
    racket_3d: Tensor  # (T, 3, 3)


class TennisPoseSceneGenerator:
    """Generate clean multi-player tennis pose scenes."""

    def __init__(
        self,
        cfg: GenConfig | None = None,
        asset_library: TennisAssetLibrary | None = None,
    ) -> None:
        self.cfg = cfg or GenConfig()
        seed = self.cfg.seed
        self._rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(int(seed))
        self.asset_lib = asset_library or TennisAssetLibrary(
            self.cfg.asset_root,
            min_frames=self.cfg.asset_min_frames,
            max_files=self.cfg.asset_max_files,
        )

    def reseed(self, seed: int | None) -> None:
        """Reseed internal RNGs for deterministic multi-process generation.

        This does not reload assets but resets ``random.Random`` and
        ``torch.manual_seed`` so that each call to :meth:`generate_scene` can
        use a scene-specific seed.
        """
        self.cfg.seed = seed
        self._rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(int(seed))

    def _sample_camera(self) -> Mapping[str, Any]:
        side = self._rng.choice(["near", "far", "left", "right"])
        t = self._rng.random()
        x, y, z = sample_camera_position_on_fence(t, side)
        cam = make_look_at_camera((x, y, z), image_size=self.cfg.image_size)
        return {
            "id": f"{side}-{t:.2f}",
            "image_size": [cam.w, cam.h],
            # Camera extrinsics and intrinsics are kept both for projection
            # and for downstream consumers (JSON/memmap/datasets).
            "camera_C": cam.C.tolist(),
            "camera_R": cam.R.tolist(),
            "camera_intr": [cam.f, cam.cx, cam.cy],
            "_cam_internal": cam,  # kept for projection, not serialized later
        }

    def _build_cameras(self, n: int) -> list[Mapping[str, Any]]:
        return [self._sample_camera() for _ in range(n)]

    def _build_players(self, frames_total: int) -> list[_PlayerSequence]:
        min_players = max(1, int(self.cfg.min_players))
        max_players = max(min_players, int(self.cfg.max_players))
        num_players = self._rng.randint(min_players, max_players)
        samples = [
            self.asset_lib.sample_sequence(frames_total, self.cfg.fps, self._rng)
            for _ in range(num_players)
        ]
        anchors = self._sample_player_origins(num_players)
        sequences: list[_PlayerSequence] = []
        for sample, anchor in zip(samples, anchors, strict=True):
            yaw = self._rng.uniform(-math.pi, math.pi)
            sequences.append(self._place_sample(sample, anchor, yaw))
        return sequences

    def _sample_player_origins(self, count: int) -> list[Tensor]:
        anchors: list[Tensor] = []
        attempts = 0
        margin_x = max(0.0, float(self.cfg.spawn_margin_x))
        margin_y = max(0.0, float(self.cfg.spawn_margin_y))
        min_sep = max(0.1, float(self.cfg.player_min_separation))
        x_min = X_MIN + margin_x
        x_max = X_MAX - margin_x
        y_min = Y_MIN + margin_y
        y_max = Y_MAX - margin_y
        max_attempts = max(count * 10, int(self.cfg.max_anchor_attempts))
        while len(anchors) < count and attempts < max_attempts:
            attempts += 1
            x = self._rng.uniform(x_min, x_max)
            y = self._rng.uniform(y_min, y_max)
            candidate = torch.tensor([x, y, 0.0], dtype=torch.float32)
            if all(
                torch.linalg.norm(candidate[:2] - other[:2]) >= min_sep
                for other in anchors
            ):
                anchors.append(candidate)
        if len(anchors) < count:
            raise RuntimeError("failed to place all players without collision")
        return anchors

    def _place_sample(
        self,
        sample: AssetSample,
        anchor: Tensor,
        yaw: float,
    ) -> _PlayerSequence:
        joints = torch.from_numpy(sample.joints).float()
        racket = torch.from_numpy(sample.racket).float()
        pelvis = torch.from_numpy(sample.pelvis).float()
        rot = _rotation_matrix_z(yaw)
        joints_rot = torch.matmul(joints, rot.t())
        racket_rot = torch.matmul(racket, rot.t())
        pelvis_rot = torch.matmul(pelvis, rot.t()) + anchor.view(1, 3)
        joints_world = joints_rot + pelvis_rot.unsqueeze(1)
        racket_world = racket_rot + pelvis_rot.unsqueeze(1)
        return _PlayerSequence(joints_world, racket_world)

    def generate_scene(self, scene_id: str | int) -> Mapping[str, Any]:
        """Produce a clean multi-player scene dictionary.

        Args:
            scene_id (str | int): Identifier embedded into the scene output.

        Returns:
            Mapping[str, Any]: Scene dictionary that passes schema validation.

        """
        fps = int(self.cfg.fps)
        frames_total = max(1, int(round(self.cfg.duration_sec * fps)))
        cameras = self._build_cameras(int(self.cfg.num_cameras))
        players = self._build_players(frames_total)
        court3d = court_keypoints_3d()

        frames: list[dict[str, Any]] = []
        for t_idx in range(frames_total):
            frame_payload: dict[str, Any] = {}
            player_joints_frame = [p.joints_3d[t_idx] for p in players]
            racket_frame = [p.racket_3d[t_idx] for p in players]
            frame_payload["player_joints_3d"] = [
                j.tolist() for j in player_joints_frame
            ]
            frame_payload["racket_points_3d"] = [r.tolist() for r in racket_frame]
            frame_payload["num_players"] = len(players)
            for cam_idx, cam_entry in enumerate(cameras):
                cam = cam_entry["_cam_internal"]
                # Project 3D
                uv_court, mask_court = project_points(cam, court3d)
                vis_court = mask_court.to(torch.uint8)

                player_uv_list: list[list[list[float]]] = []
                player_vis_list: list[list[int]] = []
                racket_uv_list: list[list[list[float]]] = []
                racket_vis_list: list[list[int]] = []
                for joints3d, racket3d in zip(
                    player_joints_frame,
                    racket_frame,
                    strict=True,
                ):
                    uv_person, mask_person = project_points(cam, joints3d)
                    uv_racket, mask_racket = project_points(cam, racket3d)
                    player_uv_list.append(uv_person.tolist())
                    player_vis_list.append(mask_person.to(torch.uint8).tolist())
                    racket_uv_list.append(uv_racket.tolist())
                    racket_vis_list.append(mask_racket.to(torch.uint8).tolist())

                cam_key = f"cam_{cam_idx}"
                frame_payload[cam_key] = {
                    "court_keypoints_2d": {
                        "points": uv_court.tolist(),
                        "visibility": vis_court.tolist(),
                    },
                    "player_keypoints_2d": {
                        "joints": player_uv_list,
                        "visibility": player_vis_list,
                    },
                    "racket_keypoints_2d": {
                        "points": racket_uv_list,
                        "visibility": racket_vis_list,
                    },
                }
            frames.append(frame_payload)

        # Serialize cameras without internal objects, but keep calibration.
        cameras_pub = [
            {
                "id": c["id"],
                "image_size": c["image_size"],
                "camera_C": c["camera_C"],
                "camera_R": c["camera_R"],
                "camera_intr": c["camera_intr"],
            }
            for c in cameras
        ]
        return {
            "scene_id": str(scene_id),
            "fps": fps,
            "num_cameras": len(cameras_pub),
            "cameras": cameras_pub,
            "frames": frames,
        }


def write_scene_json(path: Path | str, scene: Mapping[str, Any]) -> None:
    """Persist a validated scene dictionary to disk as UTF-8 JSON.

    Args:
        path (Path | str): Target path for the JSON file.
        scene (Mapping[str, Any]): Scene dictionary produced by the generator.

    Returns:
        None: Always returns None after writing the file.

    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)


def _rotation_matrix_z(yaw: float) -> Tensor:
    """Return a 3x3 rotation matrix for a Z-axis rotation.

    Args:
        yaw (float): Rotation angle in radians.

    Returns:
        Tensor: 3x3 rotation matrix.

    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

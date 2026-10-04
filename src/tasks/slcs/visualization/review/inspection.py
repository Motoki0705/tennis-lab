"""Inspect saved observations and pseudo teachers without reconstructing data."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.slcs.data.dataset import ClipArrays, SLCSDataConfig
from src.tasks.slcs.data.quality import build_label_masks, player_label_confidence
from src.tennis_scene.schema import SceneResult
from src.utils.geometry.triangulation import PinholeCamera, PointRejection
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.player import COCO17_SKELETON


def observation_state(uv: NDArray[Any], observed: bool) -> str:
    """An unobserved point is unknown, never evidence that the object is absent."""
    if not observed:
        return "unobserved"
    return "in_frame" if bool(((uv >= 0) & (uv <= 1)).all()) else "out_of_frame"


class ClipInspection:
    """Keep only small review arrays; do not retain SMPL vertices from the export."""

    def __init__(
        self, clip: ClipArrays, scene: SceneResult, config: SLCSDataConfig
    ) -> None:
        self.clip = clip
        self.metadata = scene.metadata
        # Derive the source slots with the reader's exact quality gates, then
        # verify the mapping against its published canonical arrays.
        masks = build_label_masks(
            human_kp_vis=np.asarray(scene.human_kp_vis, np.float32),
            ball_vis=np.asarray(scene.ball_vis, np.bool_),
            player_position=scene.player_position,
            player_yaw=scene.player_yaw,
            ball_3d=np.asarray(scene.ball_3d, np.float32),
            config=config.quality,
            player_reconstruction_valid=np.asarray(scene.player_valid, np.bool_),
            player_heading_valid=np.asarray(scene.player_heading_valid, np.bool_),
            ball_reconstruction_valid=np.asarray(scene.ball_3d_valid, np.bool_),
            teacher_quality=scene.metadata.get("label_quality"),
        )
        means = [
            scene.player_position[p, valid, 1].mean()
            for p, valid in enumerate(masks["player_label_valid"])
        ]
        self.order = np.argsort(means, kind="stable")
        if not np.array_equal(
            np.asarray(scene.human_kp_2d)[self.order], clip.human_kp_2d
        ):
            raise ValueError(
                "Review source slots disagree with the SLCS reader's player order."
            )
        self.player_valid = np.asarray(scene.player_valid)[self.order]
        self.heading_valid = np.asarray(scene.player_heading_valid)[self.order]
        self.player_rejection = np.asarray(scene.player_rejection_code)[self.order]
        self.ball_valid = np.asarray(scene.ball_3d_valid)
        self.ball_rejection = np.asarray(scene.ball_rejection_code)
        self.player_coverage = player_label_confidence(clip.human_kp_vis)
        self.config = config
        reference = scene.metadata.get("court_reference", {})
        self.cameras: dict[str, PinholeCamera] = {}
        self.fits: dict[str, dict[str, Any]] = {}
        fits = reference.get("camera_fits")
        if fits is not None:
            ids = reference.get("camera_ids", [])
            if len(fits) != len(ids) or tuple(ids) != clip.manifest.camera_ids:
                raise ValueError(
                    "Saved camera fits must align with the manifest camera IDs."
                )
            for camera_id, fit in zip(ids, fits, strict=True):
                self.cameras[camera_id] = PinholeCamera(
                    camera_id,
                    np.asarray(fit["K"], np.float64),
                    np.asarray(fit["R"], np.float64),
                    np.asarray(fit["t"], np.float64),
                )
                self.fits[camera_id] = fit

    def summary(self) -> dict[str, Any]:
        manifest = self.clip.manifest
        court = self.metadata.get("court_detection", {})
        observations = [
            frame
            for camera in court.get("cameras", [])
            for frame in camera.get("frames", [])
        ]
        return {
            "label_kind": "pseudo",
            "is_ground_truth": False,
            "schema_version": 2,
            "coordinate_frame": self.metadata.get("court_reference_provenance", {}).get(
                "target_frame_id", "unknown"
            ),
            "representation": self.metadata.get("representation", "unknown"),
            "pipeline_contract": self.metadata.get("pipeline_contract", "unknown"),
            "identity_scope": self.metadata.get("identity_scope", "unknown"),
            "court_temporal_policy": court.get("temporal_policy", "unknown"),
            "court_point_origin": "homography_fit"
            if observations
            and all(
                "homography_court_metres_to_image_pixels" in frame
                for frame in observations
            )
            else "unknown",
            "court_observed_frames": court.get("observed_frame_indices", []),
            "calibration_status": "saved_pinhole" if self.cameras else "unavailable",
            "calibration_note": "保存K/R/tによる近似pinhole投影。独立校正GTなし・歪み補正なし。"
            if self.cameras
            else "保存K/R/tなし。3D再投影は利用不可。",
            "player_source_slots": self.order.tolist(),
            "skeleton": [list(edge) for edge in COCO17_SKELETON],
            "source_cameras": [
                {
                    "camera_id": camera_id,
                    "media": manifest.video_paths[i],
                    "media_available": manifest.media_path(
                        camera_id, must_exist=False
                    ).is_file(),
                    "source": dict(manifest.cameras[i]) if manifest.cameras else None,
                    "calibration": self.fits.get(camera_id),
                }
                for i, camera_id in enumerate(manifest.camera_ids)
            ],
            "timeline": {
                "players": self.clip.player_label_valid.astype(int).tolist(),
                "ball": self.clip.ball_label_valid.astype(int).tolist(),
                "ball_observed_cameras": self.clip.ball_vis.sum(axis=0).tolist(),
            },
            "quality": {
                "min_player_confidence": self.config.quality.min_player_confidence,
                "min_ball_cameras": self.config.quality.min_ball_cameras,
                "label_weight_power": self.config.quality.label_weight_power,
                "teacher_quality_present": "label_quality" in self.metadata,
            },
        }

    def _player_reasons(self, player: int, frame: int) -> list[str]:
        reasons = []
        code = int(self.player_rejection[player, frame])
        if not self.player_valid[player, frame]:
            names = self.metadata.get("body_placement", {}).get("rejection_codes", {})
            reason = next(
                (name for name, value in names.items() if value == code), f"code {code}"
            )
            reasons.append(f"root棄却: {reason}")
        if not self.heading_valid[player, frame]:
            reasons.append("heading無効")
        if (
            self.player_coverage[player, frame]
            < self.config.quality.min_player_confidence
        ):
            reasons.append("2D coverage閾値未満")
        if not self.clip.player_label_valid[player, frame] and not reasons:
            reasons.append("教師quality / 非有限値により無効")
        return reasons

    def _project(
        self, camera_id: str, xyz: NDArray[Any], valid: bool
    ) -> dict[str, Any]:
        camera = self.cameras.get(camera_id)
        if camera is None:
            return {"state": "calibration_unavailable", "uv": None}
        if not valid:
            return {"state": "teacher_invalid", "uv": None}
        uv_px, front = camera.project(xyz)
        if not bool(front):
            return {"state": "behind_camera", "uv": None}
        uv = uv_px / np.asarray([self.clip.manifest.width, self.clip.manifest.height])
        return {"state": observation_state(uv, True), "uv": uv.tolist()}

    def frame(self, camera_id: str, frame: int) -> dict[str, Any]:
        clip = self.clip
        cam = clip.manifest.camera_index(camera_id)
        if not 0 <= frame < clip.num_frames:
            raise ValueError(f"frame must be in [0, {clip.num_frames - 1}].")
        scale = np.asarray(COURT_COORD_SCALE_XYZ, np.float32)
        players = []
        for p in range(clip.player_position_norm.shape[0]):
            valid = bool(clip.player_label_valid[p, frame])
            xyz = clip.player_position_norm[p, frame] * scale
            uv, confidence = (
                clip.human_kp_2d[p, cam, frame],
                clip.human_kp_vis[p, cam, frame],
            )
            # Hidden observations carry no coordinate through the API.
            joints = [
                point.tolist() if conf > 0 else None
                for point, conf in zip(uv, confidence, strict=True)
            ]
            players.append(
                {
                    "slot": p,
                    "source_slot": int(self.order[p]),
                    "pose_uv": joints,
                    "pose_confidence": confidence.tolist(),
                    "observation_states": [
                        observation_state(point, bool(conf > 0))
                        for point, conf in zip(uv, confidence, strict=True)
                    ],
                    "observed_joints": int((confidence > 0).sum()),
                    "label_valid": valid,
                    "weight": float(clip.player_label_weight[p, frame]),
                    "coverage": float(self.player_coverage[p, frame]),
                    "root_valid": bool(self.player_valid[p, frame]),
                    "heading_valid": bool(self.heading_valid[p, frame]),
                    "rejection_code": int(self.player_rejection[p, frame]),
                    "reasons": self._player_reasons(p, frame),
                    "position_m": xyz.tolist() if valid else None,
                    "yaw_rad": float(np.arctan2(*clip.player_rotation[p, frame, ::-1]))
                    if valid
                    else None,
                    "projection": self._project(camera_id, xyz, valid),
                }
            )
        ball_observed = bool(clip.ball_vis[cam, frame])
        ball_valid = bool(clip.ball_label_valid[frame])
        ball_xyz = clip.ball_position_norm[frame] * scale
        code = int(self.ball_rejection[frame])
        ball_reasons = []
        if not self.ball_valid[frame]:
            ball_reasons.append(
                PointRejection(code).name
                if code in PointRejection._value2member_map_
                else f"code {code}"
            )
        if int(clip.ball_vis[:, frame].sum()) < self.config.quality.min_ball_cameras:
            ball_reasons.append("観測カメラ数がquality閾値未満")
        if not ball_valid and not ball_reasons:
            ball_reasons.append("教師quality / 非有限値により無効")
        return {
            "clip_id": clip.manifest.clip_id,
            "camera_id": camera_id,
            "frame": frame,
            "time_seconds": frame / clip.fps,
            "image_size": [clip.manifest.width, clip.manifest.height],
            "uv_normalization": "image_width_height",
            "players": players,
            "court": {
                "uv": [
                    point.tolist() if conf > 0 else None
                    for point, conf in zip(
                        clip.court_kp[cam, frame],
                        clip.court_vis[cam, frame],
                        strict=True,
                    )
                ],
                "confidence": clip.court_vis[cam, frame].tolist(),
            },
            "ball": {
                "observation_state": observation_state(
                    clip.ball_uv[cam, frame], ball_observed
                ),
                "uv": clip.ball_uv[cam, frame].tolist() if ball_observed else None,
                "observed_cameras": int(clip.ball_vis[:, frame].sum()),
                "label_valid": ball_valid,
                "reconstruction_valid": bool(self.ball_valid[frame]),
                "weight": float(clip.ball_label_weight[frame]),
                "rejection_code": code,
                "reasons": ball_reasons,
                "position_m": ball_xyz.tolist() if ball_valid else None,
                "projection": self._project(camera_id, ball_xyz, ball_valid),
            },
            "calibration": self.fits.get(camera_id),
        }

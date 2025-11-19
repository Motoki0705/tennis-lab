"""Schema helpers for tennis pose scene data (JSON/NPZ-like dictionaries)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_scene_dict(scene: Mapping[str, Any]) -> None:
    """Validate the minimal shape of a scene dictionary.

    The function checks required fields and basic consistency between counts.
    It is intentionally permissive to allow incremental enrichment over phases.
    """
    # Top-level keys
    _require("scene_id" in scene, "missing scene_id")
    _require("fps" in scene, "missing fps")
    _require("num_cameras" in scene, "missing num_cameras")
    _require("cameras" in scene, "missing cameras")
    _require("frames" in scene, "missing frames")

    cams = scene["cameras"]
    frames = scene["frames"]
    num_cams = int(scene["num_cameras"])
    _require(isinstance(cams, Sequence), "cameras must be a sequence")
    _require(isinstance(frames, Sequence), "frames must be a sequence")
    _require(len(cams) == num_cams, "len(cameras) must equal num_cameras")

    # Minimal camera shape: id + image_size
    for cam in cams:
        _require(isinstance(cam, Mapping), "camera entry must be a mapping")
        _require("id" in cam, "camera missing id")
        _require("image_size" in cam, "camera missing image_size")
        size = cam["image_size"]
        _require(
            isinstance(size, Sequence) and len(size) == 2,
            "image_size must be [w, h]",
        )

    # Minimal frame payload: 2D keypoints for every camera + optional 3D GT
    for fr in frames:
        _require(isinstance(fr, Mapping), "frame must be a mapping")
        player_joints = fr.get("player_joints_3d")
        racket_points = fr.get("racket_points_3d")
        _require(
            isinstance(player_joints, Sequence), "player_joints_3d must be a sequence"
        )
        num_players = len(player_joints)
        _require(num_players > 0, "frame must contain at least one player")
        _require(
            isinstance(racket_points, Sequence), "racket_points_3d must be a sequence"
        )
        _require(len(racket_points) == num_players, "racket_points_3d length mismatch")
        if "num_players" in fr:
            _require(int(fr["num_players"]) == num_players, "num_players mismatch")
        for cam_idx in range(num_cams):
            key = f"cam_{cam_idx}"
            _require(key in fr, f"frame missing camera payload: {key}")
            cam_payload = fr[key]
            _require(isinstance(cam_payload, Mapping), "cam payload must be mapping")
            for req in (
                "court_keypoints_2d",
                "player_keypoints_2d",
                "racket_keypoints_2d",
            ):
                _require(req in cam_payload, f"cam payload missing {req}")
            # visibility presence (per spec) but not strictly shape-checked here
            court_bundle = cam_payload["court_keypoints_2d"]
            player_bundle = cam_payload["player_keypoints_2d"]
            racket_bundle = cam_payload["racket_keypoints_2d"]
            _require("visibility" in court_bundle, "missing court visibility")
            _require("points" in court_bundle, "missing court points")
            _require("joints" in player_bundle, "missing player joints")
            _require("visibility" in player_bundle, "missing player visibility")
            _require("points" in racket_bundle, "missing racket points")
            _require("visibility" in racket_bundle, "missing racket visibility")
            _require(
                isinstance(player_bundle["joints"], Sequence),
                "player joints must be a sequence",
            )
            _require(
                isinstance(player_bundle["visibility"], Sequence),
                "player visibility must be a sequence",
            )
            _require(
                isinstance(racket_bundle["points"], Sequence),
                "racket points must be a sequence",
            )
            _require(
                isinstance(racket_bundle["visibility"], Sequence),
                "racket visibility must be a sequence",
            )
            _require(
                len(player_bundle["joints"]) == num_players,
                "player joints length mismatch",
            )
            _require(
                len(player_bundle["visibility"]) == num_players,
                "player visibility length mismatch",
            )
            _require(
                len(racket_bundle["points"]) == num_players,
                "racket points length mismatch",
            )
            _require(
                len(racket_bundle["visibility"]) == num_players,
                "racket visibility length mismatch",
            )

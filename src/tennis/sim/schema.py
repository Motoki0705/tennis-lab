"""Schema helpers for tennis pose scene data (JSON/NPZ-like dictionaries)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


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
    num_cams = int(scene["num_cameras"])  # type: ignore[arg-type]
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
        for cam_idx in range(num_cams):
            key = f"cam_{cam_idx}"
            _require(key in fr, f"frame missing camera payload: {key}")
            cam_payload = fr[key]
            _require(isinstance(cam_payload, Mapping), "cam payload must be mapping")
            for req in ("court_keypoints_2d", "player_keypoints_2d", "racket_keypoints_2d"):
                _require(req in cam_payload, f"cam payload missing {req}")
            # visibility presence (per spec) but not strictly shape-checked here
            _require("visibility" in cam_payload["court_keypoints_2d"], "missing court visibility")
            _require("visibility" in cam_payload["player_keypoints_2d"], "missing player visibility")
            _require("visibility" in cam_payload["racket_keypoints_2d"], "missing racket visibility")


"""Video and crop conditions shared by CLIP planning and dataset publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.utils.video.reader import OpenCVDecodeIdentity, probe_video_decoder

APPEARANCE_MEDIA = "appearance_media"


class AppearanceMediaIdentity(TypedDict):
    crop: dict[str, object]
    decoders: dict[str, OpenCVDecodeIdentity]


def appearance_crop_settings() -> dict[str, object]:
    return {"opencv_build": document_digest(cv2.getBuildInformation()),
            "opencv_optimized": cv2.useOptimized(), "opencv_ipp": cv2.ipp.useIPP(),
            "numpy": np.__version__}


def appearance_media_identity(camera_ids: Sequence[str], paths: Sequence[Path]) -> AppearanceMediaIdentity:
    """Bind the runtime and the decoder actually selected for each source video."""
    return {"crop": appearance_crop_settings(), "decoders": {
        camera: probe_video_decoder(path) for camera, path in zip(camera_ids, paths, strict=True)}}


def require_appearance_media(descriptor: Mapping[str, Any], expected: object) -> None:
    """Check that an adopted artifact used the expected pixel conditions."""
    if expected is not None and descriptor["identity"].get("settings", {}).get(APPEARANCE_MEDIA) != json_value(expected):
        raise ValueError("Stored appearance decoder/preprocessing identity differs from the current conditions")


def require_current_appearance_crop(settings: Mapping[str, Any]) -> None:
    media = settings.get(APPEARANCE_MEDIA)
    if media is not None and media["crop"] != appearance_crop_settings():
        raise ValueError("Appearance crop settings changed after planning")

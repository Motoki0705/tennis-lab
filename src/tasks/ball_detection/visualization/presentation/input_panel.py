"""Explicit schematic icons or real RGB/MDD triplets, with input provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.visualization.presentation.configuration import InputClip
from src.tasks.base.visualization.presentation import style
from src.utils.checksum import dual_sha256


@dataclass(frozen=True)
class InputPanel:
    rgb: tuple[NDArray[np.uint8], ...]
    mdd: tuple[NDArray[np.uint8], ...]
    provenance: dict[str, Any]
    input_files: dict[str, str]


def schematic_panel() -> InputPanel:
    """Draw principles only; these icons are never reported as model results."""
    rgb, mdd = [], []
    background, fixed, foreground = (
        tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
        for color in (style.BACKGROUND, style.FIXED, style.FOREGROUND)
    )
    for index in range(3):
        image: NDArray[np.uint8] = np.full((90, 150, 3), background, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (140, 80), fixed, 1)
        cv2.line(image, (75, 10), (75, 80), fixed, 1)
        cv2.line(image, (10, 45), (140, 45), fixed, 1)
        x, y = 35 + index * 35, 62 - index * 15
        cv2.circle(image, (x, y), 5, foreground, -1)
        motion = np.zeros_like(image)
        cv2.circle(motion, (x - 12, y + 5), 5, (255, 0, 0), -1)
        cv2.circle(motion, (x, y), 5, (0, 255, 0), -1)
        rgb.append(image)
        mdd.append(motion)
    return InputPanel(
        tuple(rgb), tuple(mdd), {"origin": "schematic", "kind": "RGB/MDD icons"}, {}
    )


def real_clip_panel(clip: InputClip, input_contract: dict[str, Any]) -> InputPanel:
    """Use the canonical saved RGBToMDD contract on 32 native CPU RGB frames."""
    import torch

    from src.tasks.ball_detection.data.store import (
        INDEX_FILE,
        METADATA_FILE,
        BallFrameStore,
    )
    from src.tasks.ball_detection.preprocessing.mdd import RGBToMDD

    transform = RGBToMDD.from_contract(input_contract)
    store = BallFrameStore(clip.store_dir)
    record = store.clip_by_id(clip.clip_id)
    if clip.frame < 0 or clip.frame + 32 > record.frame_count:
        raise ValueError("input_clip.frame must start a complete 32-frame window")
    paths = [
        clip.store_dir / METADATA_FILE,
        clip.store_dir / INDEX_FILE,
        store.image_path(record),
    ]
    hashes = {str(path): dual_sha256(path) for path in paths}
    rgb = np.stack(
        [
            cv2.cvtColor(store.read_bgr(store.row_of(record, index)), cv2.COLOR_BGR2RGB)
            for index in range(clip.frame, clip.frame + 32)
        ]
    )
    with torch.inference_mode():
        features = transform(torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy())[None])[
            0
        ].numpy()
    indices = (1, 16, 31)
    motion = []
    for index in indices:
        image = np.zeros_like(rgb[index])
        image[..., 0] = (features[1, index] * 255).astype(np.uint8)
        image[..., 1] = (features[0, index] * 255).astype(np.uint8)
        motion.append(image)
    if hashes != {str(path): dual_sha256(path) for path in paths}:
        raise ValueError("Input clip changed while reading")
    return InputPanel(
        tuple(rgb[index] for index in indices),
        tuple(motion),
        {
            "origin": "measured",
            "store_dir": str(clip.store_dir),
            "clip_id": clip.clip_id,
            "window_start": clip.frame,
            "frames": [clip.frame + index for index in indices],
            "input_contract": input_contract,
            "decoder": "OpenCV/CPU",
            "mdd_channels": {"G": "brighten", "R": "darken"},
        },
        hashes,
    )

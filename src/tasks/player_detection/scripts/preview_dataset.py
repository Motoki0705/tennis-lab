"""Render evenly spaced stored frames of each split with their player labels.

Colors: green=observed, orange=inferred (occluded/truncated amodal box).
Unresolved players (no box) are listed in the header. Usage::

    .venv/bin/python -m src.tasks.player_detection.scripts.preview_dataset \
        paths.data_root=/abs/tennis-lab/data paths.output_root=/abs/tennis-lab/outputs
"""

from __future__ import annotations

import cv2
import numpy as np
from omegaconf import DictConfig

from src.tasks.player_detection.configuration import (
    PreviewConfig,
    validate_preview_boundary,
)
from src.tasks.player_detection.data.store import (
    BBOX_SOURCE_CODES,
    SPLIT_CODES,
    PlayerFrameStore,
    split_names,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "player_detection.preview_dataset"
register_boundary_validator(_BOUNDARY, validate_preview_boundary)
_COLORS = {BBOX_SOURCE_CODES["observed"]: (0, 200, 0), BBOX_SOURCE_CODES["inferred"]: (0, 150, 255)}


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="preview_dataset",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    config = PreviewConfig.from_config(cfg)
    store = PlayerFrameStore(config.dataset_dir)
    for split in split_names(list(SPLIT_CODES)):
        rows = store.split_frames(split)
        if rows.size == 0:
            continue
        directory = config.output_dir / split
        directory.mkdir(parents=True, exist_ok=True)
        picks = np.linspace(0, rows.size - 1, min(config.frames_per_split, rows.size)).round().astype(int)
        for row in rows[picks].tolist():
            clip = store.clip_of(row)
            image = store.read_bgr(row)
            instances = store.instances_of(row)
            unresolved = []
            for track, box, source in zip(
                instances.track_index.tolist(), instances.boxes_xyxy, instances.bbox_source.tolist(), strict=True
            ):
                name = clip.track_ids[track]
                if source == BBOX_SOURCE_CODES["unresolved"]:
                    unresolved.append(name)
                    continue
                x1, y1, x2, y2 = box.round().astype(int).tolist()
                cv2.rectangle(image, (x1, y1), (x2, y2), _COLORS[source], 3)
                cv2.putText(image, name, (max(x1, 4), max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _COLORS[source], 2)
            header = f"{store.frame_key(row)} unresolved={unresolved}"
            cv2.putText(image, header, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(directory / f"{store.frame_key(row).replace(':', '__')}.jpg"), image)
    print(f"Wrote previews to {config.output_dir}")


if __name__ == "__main__":
    main()

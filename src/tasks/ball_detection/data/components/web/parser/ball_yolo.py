"""Parser for Ball-YOLO tennis labels and mapped source videos."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path

from src.tasks.ball_detection.data.components.web.data_access_layer.writer import (
    WebFrameRecord,
)
from src.tasks.ball_detection.data.components.web.parser.base import (
    ParsedSource,
    WebDatasetParser,
)
from src.utils.data.splits import GroupSplitConfig, make_group_split_map
from src.utils.geometry.keypoints import clamp_pixel_coordinate
from src.utils.video import iter_selected_video_jpegs, probe_video_info


class BallYoloParser(WebDatasetParser):
    """Normalize explicit Ball-YOLO labels without inferring missing negatives."""

    def __init__(
        self,
        web_root: Path,
        jpeg_quality: int,
        split_config: GroupSplitConfig,
    ) -> None:
        self.web_root = web_root
        self.jpeg_quality = jpeg_quality
        self.split_config = split_config

    def sources(self) -> Iterator[ParsedSource]:
        """Yield the Ball-YOLO source."""
        split_map = make_group_split_map(
            self._group_weights(),
            self.split_config,
        )
        yield ParsedSource(
            name="ball_yolo",
            records=partial(self._records, split_map),
        )

    @property
    def _labels_root(self) -> Path:
        return self.web_root / "ball_yolo_sport_ball_labels" / "tennis" / "Labels"

    def _group_weights(self) -> dict[str, int]:
        return {
            f"ball_yolo:{folder.name}": sum(1 for _ in folder.glob("*.txt"))
            for folder in self._labels_root.iterdir()
            if folder.is_dir()
        }

    def _records(
        self,
        split_map: Mapping[str, str],
    ) -> Iterator[WebFrameRecord]:
        videos_dir = self.web_root / "sport_ball_detection_videos" / "tennis" / "Videos"
        mapping_csv = self.web_root / "ball_yolo_tennis_video_mapping.csv"
        mapping = {
            row["label_folder"]: row
            for row in csv.DictReader(mapping_csv.open(encoding="utf-8"))
        }
        for folder in sorted(self._labels_root.iterdir()):
            if not folder.is_dir() or folder.name not in mapping:
                continue
            parts = [
                videos_dir / name
                for name in mapping[folder.name]["official_video_files"].split(";")
            ]
            parts = [part for part in parts if part.exists()]
            if not parts:
                continue
            frame_boxes: dict[int, list[tuple[float, float, int]]] = {}
            for label_file in folder.glob("*.txt"):
                frame_index = int(label_file.stem.rsplit("_", 1)[1])
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) < 5:
                        continue
                    center_x, center_y = float(fields[1]), float(fields[2])
                    frame_boxes.setdefault(frame_index, []).append(
                        (center_x, center_y, 1)
                    )
            if not frame_boxes:
                continue
            video_info = probe_video_info(parts[0])
            width, height = video_info.width, video_info.height
            sequence = f"ball_yolo:{folder.name}"
            split = split_map[sequence]
            for index, jpeg in iter_selected_video_jpegs(
                parts,
                set(frame_boxes),
                quality=self.jpeg_quality,
            ):
                instances = [
                    (
                        clamp_pixel_coordinate(center_x * width, width),
                        clamp_pixel_coordinate(center_y * height, height),
                        visibility,
                    )
                    for center_x, center_y, visibility in frame_boxes[index]
                ]
                yield WebFrameRecord(
                    instances=instances,
                    orig_w=width,
                    orig_h=height,
                    temporal=1,
                    source="ball_yolo",
                    sequence=sequence,
                    frame_index=index,
                    split=split,
                    jpeg=jpeg,
                )


__all__ = ["BallYoloParser"]

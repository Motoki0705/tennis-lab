"""Parser for the Kaggle tennis back-view dataset."""

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


class KaggleParser(WebDatasetParser):
    """Normalize Kaggle CSV labels and sentinel-negative rows."""

    def __init__(
        self,
        web_root: Path,
        jpeg_quality: int,
        split_config: GroupSplitConfig,
        corner_fraction: float,
    ) -> None:
        self.web_root = web_root
        self.jpeg_quality = jpeg_quality
        self.split_config = split_config
        self.corner_fraction = corner_fraction

    def sources(self) -> Iterator[ParsedSource]:
        """Yield the Kaggle back-view source."""
        split_map = make_group_split_map(
            self._group_weights(),
            self.split_config,
        )
        yield ParsedSource(
            name="kaggle_backview",
            records=partial(self._records, split_map),
        )

    @property
    def _root(self) -> Path:
        return self.web_root / "kaggle_tenis_backview"

    def _group_weights(self) -> dict[str, int]:
        weights: dict[str, int] = {}
        for ball_csv in self._root.glob("video*_ball.csv"):
            video_id = ball_csv.name[: -len("_ball.csv")]
            row_count = sum(1 for _ in csv.DictReader(ball_csv.open(encoding="utf-8")))
            weights[f"kaggle_backview:{video_id}"] = row_count
        return weights

    def _records(
        self,
        split_map: Mapping[str, str],
    ) -> Iterator[WebFrameRecord]:
        for ball_csv in sorted(self._root.glob("video*_ball.csv")):
            video_id = ball_csv.name[: -len("_ball.csv")]
            video = self._root / f"{video_id}.mp4"
            if not video.exists():
                continue
            video_info = probe_video_info(video)
            width, height = video_info.width, video_info.height
            corner_x = width * (1.0 - self.corner_fraction)
            corner_y = height * self.corner_fraction
            frame_boxes: dict[int, list[tuple[float, float, int]]] = {}
            for row in csv.DictReader(ball_csv.open(encoding="utf-8")):
                try:
                    x = float(row["ball_x"])
                    y = float(row["ball_y"])
                except (TypeError, ValueError):
                    continue
                frame_index = int(str(row["frame"]).split("_")[-1])
                frame_boxes.setdefault(frame_index, [])
                if not (x >= corner_x and y <= corner_y):
                    frame_boxes[frame_index].append(
                        (
                            clamp_pixel_coordinate(x, width),
                            clamp_pixel_coordinate(y, height),
                            1,
                        )
                    )
            if not frame_boxes:
                continue
            sequence = f"kaggle_backview:{video_id}"
            split = split_map[sequence]
            for index, jpeg in iter_selected_video_jpegs(
                [video],
                set(frame_boxes),
                quality=self.jpeg_quality,
            ):
                yield WebFrameRecord(
                    instances=frame_boxes[index],
                    orig_w=width,
                    orig_h=height,
                    temporal=1,
                    source="kaggle_backview",
                    sequence=sequence,
                    frame_index=index,
                    split=split,
                    jpeg=jpeg,
                )


__all__ = ["KaggleParser"]

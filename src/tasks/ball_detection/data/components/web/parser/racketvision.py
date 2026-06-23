"""Parser for the RacketVision tennis tracking dataset."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from functools import partial
from pathlib import Path

from src.tasks.ball_detection.data.components.web.data_access_layer.web_store import (
    SPLIT_CODES,
)
from src.tasks.ball_detection.data.components.web.data_access_layer.writer import (
    WebFrameRecord,
)
from src.tasks.ball_detection.data.components.web.parser.base import (
    ParsedSource,
    WebDatasetParser,
)
from src.utils.geometry.keypoints import clamp_pixel_coordinate
from src.utils.io import load_json
from src.utils.video import iter_selected_video_jpegs, probe_video_info


class RacketVisionParser(WebDatasetParser):
    """Normalize RacketVision labels using its official sequence splits."""

    def __init__(self, web_root: Path, jpeg_quality: int) -> None:
        self.web_root = web_root
        self.jpeg_quality = jpeg_quality

    def sources(self) -> Iterator[ParsedSource]:
        """Yield the RacketVision source."""
        yield ParsedSource(
            name="racketvision",
            records=partial(self._records),
        )

    def _load_split_map(self, root: Path) -> dict[tuple[str, str], str]:
        split_map: dict[tuple[str, str], str] = {}
        for split in SPLIT_CODES:
            path = root / "info" / f"{split}.json"
            entries = load_json(path)
            for match_id, clip_id in entries:
                key = (str(match_id), str(clip_id))
                if key in split_map:
                    raise ValueError(f"RacketVision sequence appears twice: {key}.")
                split_map[key] = split
        return split_map

    def _records(self) -> Iterator[WebFrameRecord]:
        root = self.web_root / "racketvision_tennis" / "tennis"
        videos_dir = root / "videos"
        split_map = self._load_split_map(root)
        for match_dir in sorted((root / "all").iterdir()):
            if not match_dir.is_dir():
                continue
            match_id = match_dir.name
            for csv_path in sorted(match_dir.glob("csv/*_ball.csv")):
                clip_id = csv_path.stem.split("_")[0]
                video = videos_dir / f"{match_id}_{clip_id}.mp4"
                if not video.exists():
                    continue
                frame_boxes: dict[int, list[tuple[float, float, int]]] = {}
                for row in csv.DictReader(csv_path.open(encoding="utf-8")):
                    frame_index = int(row["Frame"])
                    frame_boxes.setdefault(frame_index, [])
                    if str(row.get("Visibility")) == "1":
                        frame_boxes[frame_index].append(
                            (float(row["X"]), float(row["Y"]), 1)
                        )
                if not frame_boxes:
                    continue
                video_info = probe_video_info(video)
                width, height = video_info.width, video_info.height
                split = split_map[(match_id, clip_id)]
                sequence = f"racketvision:{match_id}_{clip_id}"
                for index, jpeg in iter_selected_video_jpegs(
                    [video],
                    set(frame_boxes),
                    quality=self.jpeg_quality,
                ):
                    instances = [
                        (
                            clamp_pixel_coordinate(x, width),
                            clamp_pixel_coordinate(y, height),
                            visibility,
                        )
                        for x, y, visibility in frame_boxes[index]
                    ]
                    yield WebFrameRecord(
                        instances=instances,
                        orig_w=width,
                        orig_h=height,
                        temporal=1,
                        source="racketvision",
                        sequence=sequence,
                        frame_index=index,
                        split=split,
                        jpeg=jpeg,
                    )


__all__ = ["RacketVisionParser"]

"""Parser for Roboflow COCO exports used by ball detection."""

from __future__ import annotations

import re
from collections import Counter
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
from src.utils.io import load_json

_DATASETS = (
    "roboflow_tennis_ball_tracking_detection_h9rat_v1",
    "roboflow_tennis_ball_tracking_1wnxz_v2",
    "roboflow_tennis_ball_wafqb_v2",
)
_RAW_SPLITS = ("train", "valid", "test")


def roboflow_source_group(file_name: str) -> str:
    """Remove the Roboflow-generated content hash from an exported file name."""
    stem = Path(file_name).stem
    return re.sub(r"\.rf\.[^.]+$", "", stem)


class RoboflowParser(WebDatasetParser):
    """Normalize all configured Roboflow COCO exports."""

    def __init__(self, web_root: Path, split_config: GroupSplitConfig) -> None:
        self.web_root = web_root
        self.split_config = split_config

    def sources(self) -> Iterator[ParsedSource]:
        """Yield one logical source per Roboflow export."""
        for name in _DATASETS:
            split_map = make_group_split_map(
                self._group_weights(name),
                self.split_config,
            )
            yield ParsedSource(
                name=name,
                records=partial(self._records, name, split_map),
            )

    def _group_weights(self, name: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        dataset_dir = self.web_root / name
        for raw_split in _RAW_SPLITS:
            annotations = dataset_dir / raw_split / "_annotations.coco.json"
            if not annotations.exists():
                continue
            coco = load_json(annotations)
            counts.update(
                f"{name}:{roboflow_source_group(str(image['file_name']))}"
                for image in coco["images"]
            )
        return dict(counts)

    def _records(
        self,
        name: str,
        split_map: Mapping[str, str],
    ) -> Iterator[WebFrameRecord]:
        dataset_dir = self.web_root / name
        for raw_split in _RAW_SPLITS:
            split_dir = dataset_dir / raw_split
            annotations = split_dir / "_annotations.coco.json"
            if not annotations.exists():
                continue
            coco = load_json(annotations)
            ball_categories = {
                category["id"]
                for category in coco["categories"]
                if str(category.get("supercategory", "none")).lower() != "none"
            }
            boxes_by_image: dict[int, list[tuple[float, float, int]]] = {}
            for annotation in coco["annotations"]:
                if annotation["category_id"] not in ball_categories:
                    continue
                x, y, width, height = annotation["bbox"]
                boxes_by_image.setdefault(annotation["image_id"], []).append(
                    (x + width / 2.0, y + height / 2.0, 1)
                )
            for image in coco["images"]:
                width = int(image["width"])
                height = int(image["height"])
                instances = [
                    (
                        clamp_pixel_coordinate(center_x, width),
                        clamp_pixel_coordinate(center_y, height),
                        visibility,
                    )
                    for center_x, center_y, visibility in boxes_by_image.get(
                        image["id"],
                        [],
                    )
                ]
                group = roboflow_source_group(str(image["file_name"]))
                sequence = f"{name}:{group}"
                yield WebFrameRecord(
                    instances=instances,
                    orig_w=width,
                    orig_h=height,
                    temporal=0,
                    source=name,
                    sequence=sequence,
                    frame_index=-1,
                    split=split_map[sequence],
                    file_path=split_dir / image["file_name"],
                )


__all__ = ["RoboflowParser", "roboflow_source_group"]

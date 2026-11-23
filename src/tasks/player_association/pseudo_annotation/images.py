"""Render synchronized overviews and source-resolution box crops for annotation."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.pseudo_annotation.job import (
    Candidate,
    Frame,
    Packet,
    Panel,
    frame_numbers,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.checksum import dual_sha256
from src.utils.video.reader import RandomAccessVideoReader

_WIDTH, _HEIGHT = 704, 396
_CROP_WIDTH, _CROP_HEIGHT = 88, 126
_COLORS = ((58, 230, 255), (242, 202, 101), (129, 236, 146), (244, 156, 232))

Image: TypeAlias = NDArray[np.uint8]


def _text(
    image: Image, label: str, x: int, y: int, color: tuple[int, int, int], size: float
) -> None:
    cv2.putText(
        image,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (18, 18, 18),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA
    )


def _panel(image: Image, panel: Panel, frame: int) -> Image:
    height, width = image.shape[:2]
    overview: Image = cv2.resize(image, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_AREA)
    _text(overview, f"Frame {frame}   {panel.camera}", 10, 22, (255, 255, 255), 0.63)
    columns = _WIDTH // _CROP_WIDTH
    rows = (len(panel.candidates) + columns - 1) // columns
    result: Image = np.full(
        (_HEIGHT + rows * _CROP_HEIGHT, _WIDTH, 3), 27, dtype=np.uint8
    )
    for index, candidate in enumerate(panel.candidates):
        x1, y1, x2, y2 = candidate.box
        a, b, c, d = (
            round(x1 * _WIDTH / width),
            round(y1 * _HEIGHT / height),
            round(x2 * _WIDTH / width),
            round(y2 * _HEIGHT / height),
        )
        color = _COLORS[index % len(_COLORS)]
        cv2.rectangle(overview, (a, b), (c, d), color, 1)
        _text(
            overview,
            f"B{index}",
            max(0, min(a, _WIDTH - 35)),
            max(36, min(b - 2, _HEIGHT - 3)),
            color,
            0.47,
        )
        dx, dy = 0.13 * (x2 - x1), 0.10 * (y2 - y1)
        left, right = max(0, int(x1 - dx)), min(width, int(np.ceil(x2 + dx)))
        top, bottom = max(0, int(y1 - dy)), min(height, int(np.ceil(y2 + dy)))
        tile_x = (index % columns) * _CROP_WIDTH
        tile_y = _HEIGHT + (index // columns) * _CROP_HEIGHT
        if left < right and top < bottom:
            scale = min(
                (_CROP_WIDTH - 3) / (right - left), (_CROP_HEIGHT - 22) / (bottom - top)
            )
            crop = cv2.resize(
                image[top:bottom, left:right],
                (
                    max(1, round((right - left) * scale)),
                    max(1, round((bottom - top) * scale)),
                ),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
            crop_x = tile_x + (_CROP_WIDTH - crop.shape[1]) // 2
            result[
                tile_y + 20 : tile_y + 20 + crop.shape[0],
                crop_x : crop_x + crop.shape[1],
            ] = crop
        _text(result, f"B{index}", tile_x + 4, tile_y + 15, color, 0.5)
    result[:_HEIGHT] = overview
    return result


def _sheet(panels: list[Image]) -> Image:
    columns = min(2, len(panels))
    groups = [panels[i : i + columns] for i in range(0, len(panels), columns)]
    heights = [max(panel.shape[0] for panel in group) for group in groups]
    result: Image = np.full((sum(heights), _WIDTH * columns, 3), 12, dtype=np.uint8)
    offset = 0
    for group, height in zip(groups, heights, strict=True):
        for column, panel in enumerate(group):
            start = column * _WIDTH
            result[offset : offset + panel.shape[0], start : start + _WIDTH] = panel
        offset += height
    return result


def prepare_images(
    clip: ClipManifest,
    candidates: Mapping[int, dict[str, list[Candidate]]],
    strata: Mapping[int, list[str]],
    destination: Path,
    *,
    player_count: int,
    input_snapshot_sha256: str,
    show_prior_association: bool,
    expected_media: dict[str, str] | None = None,
) -> Packet:
    """Write a complete packet in one rename after the media and frame grid are checked."""
    frames = frame_numbers(sorted(candidates), clip.num_frames)
    if set(strata) != set(frames):
        raise ValueError("Every requested frame must have a stratum entry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    media = {camera: clip.media_path(camera) for camera in clip.camera_ids}
    media_sha256 = {camera: dual_sha256(path) for camera, path in media.items()}
    if expected_media is not None and expected_media != media_sha256:
        raise ValueError(f"Synchronized source media changed for {clip.clip_id}")
    rendered: list[Frame] = []
    with (
        tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".packet-"
        ) as temporary,
        ExitStack() as stack,
    ):
        root = Path(temporary) / "packet"
        root.mkdir()
        readers = {
            camera: stack.enter_context(RandomAccessVideoReader(path))
            for camera, path in media.items()
        }
        for frame in frames:
            if set(candidates[frame]) != set(clip.camera_ids):
                raise ValueError(f"Missing camera slots for frame {frame}")
            panels = [
                Panel(camera=camera, candidates=candidates[frame][camera])
                for camera in clip.camera_ids
            ]
            images = []
            for panel in panels:
                source = readers[panel.camera].read(frame)
                if source.shape[:2] != (clip.height, clip.width):
                    raise ValueError(
                        f"Decoded dimensions differ from {clip.clip_id} {panel.camera}"
                    )
                images.append(_panel(source, panel, frame))
            path = root / f"frame_{frame:06d}.jpg"
            if not cv2.imwrite(
                str(path), _sheet(images), [cv2.IMWRITE_JPEG_QUALITY, 91]
            ):
                raise OSError(f"Failed to write {path}")
            rendered.append(
                Frame(
                    index=frame,
                    panels=panels,
                    strata=strata[frame],
                    image_sha256=dual_sha256(path),
                )
            )
        packet: Packet = Packet.model_validate(
            {
                "schema_version": 1,
                "dataset_id": clip.dataset_id,
                "clip_id": clip.clip_id,
                "num_frames": clip.num_frames,
                "start": frames[0],
                "stop": frames[-1] + 1,
                "camera_ids": list(clip.camera_ids),
                "player_count": player_count,
                "show_prior_association": show_prior_association,
                "input_snapshot_sha256": input_snapshot_sha256,
                "clip_manifest_sha256": dual_sha256(clip.clip_dir / "clip.json"),
                "media_sha256": media_sha256,
                "frames": [frame.model_dump() for frame in rendered],
            }
        )
        (root / "packet.json").write_text(packet.model_dump_json() + "\n")
        root.rename(destination)
    return packet

"""Draw both paths from one source frame and leave every invalid joint masked."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import RowRejection
from src.tennis_scene.visualization.pose_comparison.inputs import (
    STOP_LABELS,
    ClipPlan,
    ComparisonInputs,
)
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)
from src.utils.schema.player import COCO17_SKELETON

VIEW_W, VIEW_H = 384, 216
COURT_W, HEADER_H, GUTTER = 170, 56, 12
SIDE_W = 2 * VIEW_W + COURT_W
VIDEO_W, VIDEO_H = 2 * SIDE_W + GUTTER, HEADER_H + 2 * VIEW_H
CORE_Y = round(HALF_LENGTH + 5.0, 3)
MAP_SCALE = 7.8  # Same scale (pixels per metre) on x and y.

Color = tuple[int, int, int]  # OpenCV BGR.
PALETTE: tuple[Color, ...] = (
    (255, 215, 87),
    (105, 220, 255),
    (152, 131, 255),
    (129, 238, 128),
    (255, 153, 220),
    (204, 236, 114),
    (95, 172, 255),
    (232, 182, 116),
    (170, 177, 255),
    (145, 255, 218),
    (225, 124, 177),
    (182, 229, 246),
)
INK: Color = (241, 240, 237)
MUTED: Color = (181, 173, 159)
AMBER: Color = (82, 197, 255)
BALL: Color = (77, 255, 221)
SHADOW: Color = (20, 20, 20)
ROW_STYLES: dict[RowRejection, tuple[str, str, Color]] = {
    RowRejection.INSUFFICIENT_VIEWS: ("V", "視点不足", (186, 167, 126)),
    RowRejection.DEGENERATE_RAYS: ("D", "レイが退化", (192, 153, 175)),
    RowRejection.INCONSISTENT_GEOMETRY: ("G", "幾何不一致", (135, 139, 203)),
    RowRejection.MIXED_POSE: ("M", "別人の関節が混入", (120, 162, 205)),
    RowRejection.CONFLICT: ("C", "割当の競合", (167, 183, 129)),
}


@dataclass(frozen=True)
class Poses:
    xyz: NDArray[np.float32]
    valid: NDArray[np.bool_]
    ids: tuple[int, ...]
    prefix: str
    ambiguous: NDArray[np.bool_]


def torso_xy(xyz: NDArray[np.float32], valid: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Prefer visible hips, then shoulders, then other valid joints."""
    chosen = valid[:, 11:13].any(axis=1)
    use_shoulders = ~chosen & valid[:, 5:7].any(axis=1)
    mask = valid.copy()
    mask[chosen, :11] = False
    mask[chosen, 13:] = False
    mask[use_shoulders, :5] = False
    mask[use_shoulders, 7:] = False
    count = mask.sum(axis=1, keepdims=True)
    xy: NDArray[np.float32] = np.full((len(xyz), 2), np.nan, dtype=np.float32)
    np.divide(
        np.where(mask[..., None], xyz[..., :2], 0).sum(axis=1),
        count,
        out=xy,
        where=count > 0,
    )
    return xy


def in_core(xy: NDArray[np.float32]) -> NDArray[np.bool_]:
    return (np.abs(xy[:, 0]) <= HALF_DOUBLES_WIDTH) & (np.abs(xy[:, 1]) <= CORE_Y)


def _label(
    image: NDArray[np.uint8],
    label: str,
    point: tuple[int, int],
    color: Color = INK,
    scale: float = 0.46,
) -> None:
    cv2.putText(
        image, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW, 3, cv2.LINE_AA
    )
    cv2.putText(
        image, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _pixel(xy: NDArray[np.floating]) -> tuple[int, int]:
    x, y = np.clip(np.rint(xy), -32768, 32768).astype(int)
    return int(x), int(y)


def _contains(image: NDArray[np.uint8], xy: tuple[int, int]) -> bool:
    height, width = image.shape[:2]
    return 0 <= xy[0] < int(width) and 0 <= xy[1] < int(height)


def draw_rejections(
    image: NDArray[np.uint8],
    boxes: NDArray[np.floating],
    codes: NDArray[np.uint8],
) -> None:
    for box, code in zip(boxes, codes, strict=True):
        if code == RowRejection.ASSIGNED:
            continue
        letter, _, color = ROW_STYLES[RowRejection(int(code))]
        p1, p2 = _pixel(box[:2]), _pixel(box[2:])
        cv2.rectangle(image, p1, p2, color, 1, cv2.LINE_AA)
        _label(image, letter, (max(1, p1[0]), max(10, p1[1] - 2)), color, 0.34)


def draw_poses(
    image: NDArray[np.uint8],
    uv: NDArray[np.float64],
    visible: NDArray[np.bool_],
    poses: Poses,
) -> None:
    for points, mask, identity, ambiguous in zip(
        uv, visible, poses.ids, poses.ambiguous, strict=True
    ):
        color = PALETTE[identity % len(PALETTE)]
        pixels = [
            (x, y)
            for x, y in np.clip(np.rint(points), -32768, 32768).astype(int).tolist()
        ]
        bones = [
            (pixels[a], pixels[b]) for a, b in COCO17_SKELETON if mask[a] and mask[b]
        ]
        for a, b in bones:
            cv2.line(image, a, b, SHADOW, 3, cv2.LINE_AA)
        for a, b in bones:
            cv2.line(image, a, b, color, 1, cv2.LINE_AA)
        for joint in np.flatnonzero(mask):
            point = pixels[joint]
            if _contains(image, point):
                cv2.circle(image, point, 1, color, -1, cv2.LINE_AA)
        inside = [
            point
            for point, valid in zip(pixels, mask, strict=True)
            if valid and _contains(image, point)
        ]
        if inside:
            x, y = min(inside, key=lambda point: point[1])
            label = f"{poses.prefix}{identity}"
            _label(image, label, (max(1, x - 2), max(11, y - 4)), color, 0.35)
            if ambiguous:
                _label(image, "?", (max(14, x + 14), max(11, y - 4)), AMBER, 0.50)


def _draw_ball(image: NDArray[np.uint8], point: tuple[int, int]) -> None:
    if _contains(image, point):
        cv2.circle(image, point, 5, SHADOW, -1, cv2.LINE_AA)
        cv2.circle(image, point, 3, BALL, -1, cv2.LINE_AA)


def _court_pixel(xy: NDArray[np.floating]) -> tuple[int, int]:
    return _pixel(xy * [MAP_SCALE, -MAP_SCALE] + [COURT_W / 2, VIEW_H])


def _court_background() -> NDArray[np.uint8]:
    image: NDArray[np.uint8] = np.full(
        (2 * VIEW_H, COURT_W, 3), (38, 44, 34), dtype=np.uint8
    )
    doubles, singles, baseline, service = (
        HALF_DOUBLES_WIDTH,
        HALF_SINGLES_WIDTH,
        HALF_LENGTH,
        SERVICE_LINE_DISTANCE,
    )

    def point(x: float, y: float) -> tuple[int, int]:
        return _court_pixel(np.array([x, y]))

    outline = (139, 166, 148)
    cv2.rectangle(
        image, point(-doubles, baseline), point(doubles, -baseline), (68, 83, 54), -1
    )
    cv2.rectangle(
        image,
        point(-doubles, baseline),
        point(doubles, -baseline),
        outline,
        1,
        cv2.LINE_AA,
    )
    for x in (-singles, singles):
        cv2.line(
            image, point(x, baseline), point(x, -baseline), outline, 1, cv2.LINE_AA
        )
    for y in (-service, service):
        cv2.line(image, point(-singles, y), point(singles, y), outline, 1, cv2.LINE_AA)
    cv2.line(image, point(0, service), point(0, -service), outline, 1, cv2.LINE_AA)
    cv2.line(image, point(-doubles, 0), point(doubles, 0), INK, 1, cv2.LINE_AA)
    left, top = point(-doubles, CORE_Y)
    right, bottom = point(doubles, -CORE_Y)
    for x in range(left, right, 8):
        for y in (top, bottom):
            cv2.line(image, (x, y), (min(x + 4, right), y), AMBER, 1, cv2.LINE_AA)
    for y in range(top, bottom, 8):
        for x in (left, right):
            cv2.line(image, (x, y), (x, min(y + 4, bottom)), AMBER, 1, cv2.LINE_AA)
    _label(image, "COURT / XY", (10, 19), INK)
    _label(image, "+Y", (8, top - 8), MUTED, 0.36)
    _label(image, "-Y", (8, bottom + 16), MUTED, 0.36)
    return image


def _draw_court(
    image: NDArray[np.uint8], poses: Poses, ball: NDArray[np.float32] | None
) -> None:
    xy = torso_xy(poses.xyz, poses.valid)
    off_map = 0
    for center, identity, ambiguous in zip(xy, poses.ids, poses.ambiguous, strict=True):
        p = _court_pixel(center)
        if not _contains(image, p):
            off_map += 1
            continue
        color = PALETTE[identity % len(PALETTE)]
        cv2.circle(image, p, 5, AMBER if ambiguous else SHADOW, -1, cv2.LINE_AA)
        cv2.circle(image, p, 3, color, -1, cv2.LINE_AA)
        _label(
            image,
            f"{identity}{'?' if ambiguous else ''}",
            (p[0] + 5, p[1] + 1),
            color,
            0.38,
        )
    if ball is not None:
        _draw_ball(image, _court_pixel(ball[:2]))
    _label(
        image,
        f"Core {np.count_nonzero(in_core(xy))}  /  Off map {off_map}",
        (10, 2 * VIEW_H - 13),
        INK,
        0.38,
    )


def _camera_tile(
    frame: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.int64]]:
    h, w = frame.shape[:2]
    factor = min(VIEW_W / w, VIEW_H / h)
    width, height = max(1, round(w * factor)), max(1, round(h * factor))
    x, y = (VIEW_W - width) // 2, (VIEW_H - height) // 2
    image: NDArray[np.uint8] = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    image[y : y + height, x : x + width] = cv2.resize(
        frame, (width, height), interpolation=cv2.INTER_AREA
    )
    return image, np.array([width / w, height / h]), np.array([x, y])


def _project(
    image: NDArray[np.uint8],
    camera: PinholeCamera,
    poses: Poses,
    scale: NDArray[np.float64],
    offset: NDArray[np.int64],
) -> None:
    uv, front = camera.project(poses.xyz)
    draw_poses(image, uv * scale + offset, poses.valid & front, poses)


class FrameRenderer:
    def __init__(self, comparison: ComparisonInputs, plan: ClipPlan) -> None:
        self.data = comparison
        self.court = _court_background()
        self.canvas: NDArray[np.uint8] = np.full(
            (VIDEO_H, VIDEO_W, 3), (27, 25, 23), dtype=np.uint8
        )
        assembly = comparison.provenance["legacy_scene_assembly"]
        self.old_id = assembly["artifact_id"][:12] if assembly else "none"
        self.stop = STOP_LABELS.get(
            plan.baseline_reason or "", ("", plan.baseline_reason or "")
        )[1]

    def render(
        self, frame: int, cameras_bgr: tuple[NDArray[np.uint8], ...]
    ) -> NDArray[np.uint8]:
        source, old, candidates = (
            self.data.source,
            self.data.legacy,
            self.data.candidates,
        )
        new_start, new_end = candidates.frame_offsets[frame : frame + 2]
        current = slice(new_start, new_end)
        new = Poses(
            candidates.positions_m[current],
            candidates.joint_valid[current],
            tuple(range(new_end - new_start)),
            "N",
            candidates.ambiguous[current],
        )
        old_poses = Poses(new.xyz[:0], new.valid[:0], (), "P", new.ambiguous[:0])
        ball = None
        if old is not None:
            keep = old.valid[:, frame].any(axis=1)
            old_poses = Poses(
                old.xyz[keep, frame],
                old.valid[keep, frame],
                tuple(old.player_ids[i] for i in np.flatnonzero(keep)),
                "P",
                np.zeros(keep.sum(), dtype=bool),
            )
            if old.ball_valid[frame]:
                ball = old.ball[frame]

        image = self.canvas.copy()
        new_x = SIDE_W + GUTTER
        for index, (camera, frame_bgr) in enumerate(
            zip(source.camera_ids, cameras_bgr, strict=True)
        ):
            tile, scale, offset = _camera_tile(frame_bgr)
            left, right = tile.copy(), tile
            if old is not None:
                _project(left, old.cameras[index], old_poses, scale, offset)
            if self.data.tracks:
                tracks = self.data.tracks[index]
                for track in np.flatnonzero(tracks.observed[:, frame]):
                    vertices = tracks.xyxy[track, frame].reshape(2, 2) * scale + offset
                    p1, p2 = map(_pixel, vertices)
                    cv2.rectangle(left, p1, p2, MUTED, 1, cv2.LINE_AA)
                    _label(
                        left,
                        f"T{tracks.ids[track]}",
                        (max(1, p1[0]), max(11, p1[1] - 2)),
                        MUTED,
                        0.35,
                    )
                _label(left, tracks.label, (8, VIEW_H - 8), MUTED, 0.34)
            start, end = candidates.detection_offsets[index][frame : frame + 2]
            boxes = self.data.boxes[index][start:end].reshape(-1, 2, 2) * scale + offset
            draw_rejections(
                right,
                boxes.reshape(-1, 4),
                candidates.row_rejection_codes[index][start:end],
            )
            _project(right, self.data.cameras[index], new, scale, offset)
            if ball is not None and old is not None:
                uv, front = old.cameras[index].project(ball)
                if front:
                    for side in (left, right):
                        _draw_ball(side, _pixel(uv * scale + offset))
            row, col = divmod(index, 2)
            y, x = HEADER_H + row * VIEW_H, col * VIEW_W
            for canvas_x, side in ((x, left), (new_x + x, right)):
                _label(side, camera, (8, 18))
                image[y : y + VIEW_H, canvas_x : canvas_x + VIEW_W] = side

        for x, poses in ((0, old_poses), (new_x, new)):
            court = self.court.copy()
            _draw_court(court, poses, ball)
            image[HEADER_H:, x + 2 * VIEW_W : x + SIDE_W] = court
        _label(
            image,
            f"OLD ROUTE  |  {len(old_poses.ids)} 3D  |  scene {self.old_id}",
            (12, 20),
            INK,
            0.57,
        )
        _label(
            image,
            f"NEW ROUTE  |  {len(new.ids)} 3D  |  {np.count_nonzero(new.ambiguous)} close calls",
            (new_x + 12, 20),
            INK,
            0.57,
        )
        old_note = self.stop if self.data.tracks else "Persistent player IDs"
        _label(
            image,
            f"f{frame:04d}  {frame / source.fps:6.2f} s  |  {old_note}",
            (12, 41),
            MUTED,
        )
        _label(
            image,
            "Frame-local IDs and colors  |  ? = close assignment",
            (new_x + 12, 41),
            MUTED,
        )
        return image

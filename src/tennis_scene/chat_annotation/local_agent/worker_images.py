"""Contact sheets, crops and coordinate rulers."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.chat_annotation.runtime.contracts import (
    loads_json,
)

from .common import (
    iter_frames,
)
from .worker_candidates import read_cache
from .worker_context import Ctx, dump, parse_frames

BALL_STATUS_BGR = {
    "visible": (0, 255, 255),
    "occluded": (0, 140, 255),
    "interpolated": (255, 0, 255),
}

PALETTE = [
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 255),
    (0, 165, 255),
    (203, 192, 255),
    (128, 255, 255),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

MAX_SHEET_W = 1600

MAX_SHEET_H = 1600

LABEL_H = 16

RULER_TOP = 14

RULER_LEFT = 36


def color_for(track: str) -> tuple[int, int, int]:
    digest = int(hashlib.md5(track.encode()).hexdigest(), 16)
    return PALETTE[digest % len(PALETTE)]


def put(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float = 0.42,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def dashed_rect(
    img: NDArray[np.uint8],
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    dashed: bool,
) -> None:
    if not dashed:
        cv2.rectangle(img, p1, p2, color, 1, cv2.LINE_AA)
        return
    x1, y1 = p1
    x2, y2 = p2
    for a, b in (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ):
        length = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1])))
        for s in range(0, length, 8):
            e = min(length, s + 4)
            pa = (
                int(a[0] + (b[0] - a[0]) * s / length),
                int(a[1] + (b[1] - a[1]) * s / length),
            )
            pb = (
                int(a[0] + (b[0] - a[0]) * e / length),
                int(a[1] + (b[1] - a[1]) * e / length),
            )
            cv2.line(img, pa, pb, color, 1, cv2.LINE_AA)


class Drawer:
    """Maps original pixels to a tile and draws annotation/candidate objects."""

    def __init__(self, ctx: Ctx, mode: str) -> None:
        self.mode = mode
        self.rows: dict[int, dict[str, Any]] = {}
        self.cands: dict[str, Any] = {}
        if mode == "annotation":
            data = loads_json(ctx.annotation_path.read_text(encoding="utf-8"))
            self.rows = {row["frame_index"]: row for row in data["frames"]}
        elif mode == "cands-ball":
            self.cands = read_cache(ctx.work / "cands_ball.json")["frames"]
        elif mode == "cands-players":
            self.cands = read_cache(ctx.work / "cands_players.json")["frames"]
        elif mode != "none":
            raise ValueError(f"unknown draw mode {mode}")

    def draw(
        self, tile: NDArray[np.uint8], index: int, x0: float, y0: float, s: float
    ) -> None:
        def m(x: float, y: float) -> tuple[int, int]:
            return int(round((x - x0) * s)), int(round((y - y0) * s))

        if self.mode == "annotation":
            row = self.rows.get(index, {})
            for player in row.get("players", []):
                if player["bbox_xyxy"] is None:
                    continue
                x1, y1, x2, y2 = player["bbox_xyxy"]
                color = color_for(player["track_id"])
                dashed_rect(
                    tile,
                    m(x1, y1),
                    m(x2, y2),
                    color,
                    player["bbox_source"] == "inferred",
                )
                px, py = m(x1, y1)
                put(tile, player["track_id"], (px + 2, max(10, py - 3)), 0.38, color)
            for ball in row.get("balls", []):
                if ball["center_px"] is None:
                    continue
                cx, cy = m(*ball["center_px"])
                color = BALL_STATUS_BGR.get(ball["status"], (255, 255, 255))
                radius = max(7, int(round(9 * s)))
                cv2.circle(tile, (cx, cy), radius, color, 1, cv2.LINE_AA)
                put(
                    tile,
                    f"{ball['track_id']}:{ball['status'][:3]}",
                    (cx + radius + 2, cy - radius),
                    0.36,
                    color,
                )
        elif self.mode == "cands-ball":
            for cand in self.cands.get(str(index), {}).get("candidates", []):
                cx, cy = m(*cand["center_px"])
                half = max(6, int(round(8 * s)))
                cv2.rectangle(
                    tile, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 1
                )
                put(
                    tile,
                    f"{cand['score']:.2f}",
                    (cx + half + 2, cy + half),
                    0.34,
                    (0, 255, 0),
                )
        elif self.mode == "cands-players":
            for cand in self.cands.get(str(index), []):
                x1, y1, x2, y2 = cand["box"]
                color = color_for(cand.get("tid", "?"))
                cv2.rectangle(tile, m(x1, y1), m(x2, y2), color, 1, cv2.LINE_AA)
                px, py = m(x1, y1)
                put(
                    tile,
                    f"{cand.get('tid', '?')} {cand['conf']:.2f}",
                    (px + 2, max(10, py - 3)),
                    0.36,
                    color,
                )


def ruler(
    tile: NDArray[np.uint8], x0: float, y0: float, s: float, grid: int
) -> NDArray[np.uint8]:
    """Add top/left margins with original-pixel tick labels (never over image content)."""
    h, w = tile.shape[:2]
    out = np.zeros((h + RULER_TOP, w + RULER_LEFT, 3), dtype=np.uint8)
    out[RULER_TOP:, RULER_LEFT:] = tile
    step_px = grid * s
    every = max(1, math.ceil(40 / max(step_px, 1e-6)))
    first = math.ceil(x0 / grid) * grid
    k = 0
    gx = first
    while (gx - x0) * s < w:
        u = int(round((gx - x0) * s)) + RULER_LEFT
        cv2.line(out, (u, RULER_TOP - 5), (u, RULER_TOP - 1), (200, 200, 200), 1)
        if k % every == 0:
            put(out, str(int(gx)), (u + 1, RULER_TOP - 5), 0.3, (200, 200, 200))
        gx += grid
        k += 1
    first = math.ceil(y0 / grid) * grid
    k = 0
    gy = first
    while (gy - y0) * s < h:
        v = int(round((gy - y0) * s)) + RULER_TOP
        cv2.line(out, (RULER_LEFT - 5, v), (RULER_LEFT - 1, v), (200, 200, 200), 1)
        if k % every == 0:
            put(out, str(int(gy)), (1, v + 4), 0.3, (200, 200, 200))
        gy += grid
        k += 1
    return out


def labeled(tile: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    h, w = tile.shape[:2]
    width = max(w, cv2.getTextSize(text, FONT, 0.42, 1)[0][0] + 6)
    out = np.zeros((h + LABEL_H, width, 3), dtype=np.uint8)
    out[LABEL_H:, :w] = tile
    put(out, text, (3, LABEL_H - 4))
    return out


def resize(img: NDArray[np.uint8], s: float) -> NDArray[np.uint8]:
    if abs(s - 1.0) < 1e-9:
        return img
    h, w = img.shape[:2]
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(img, size, interpolation=interp)


def write_sheets(
    ctx: Ctx,
    tiles: list[NDArray[np.uint8]],
    cols: int | None,
    header: list[str],
    name: str,
) -> list[str]:
    if not tiles:
        return []
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    gap = 4
    auto_cols = max(1, (MAX_SHEET_W + gap) // (tw + gap))
    cols = max(1, min(cols or auto_cols, auto_cols, len(tiles)))
    header_h = 18 * len(header) + 4
    rows_per = max(1, (MAX_SHEET_H - header_h + gap) // (th + gap))
    per_sheet = cols * rows_per
    out_dir = ctx.work / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for page, begin in enumerate(range(0, len(tiles), per_sheet)):
        chunk = tiles[begin : begin + per_sheet]
        rows = math.ceil(len(chunk) / cols)
        sheet = np.full(
            (header_h + rows * (th + gap), cols * (tw + gap), 3), 40, dtype=np.uint8
        )
        for i, line in enumerate(header):
            put(sheet, line, (4, 15 + 18 * i), 0.45)
        for i, tile in enumerate(chunk):
            r, c = divmod(i, cols)
            y = header_h + r * (th + gap)
            x = c * (tw + gap)
            sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        path = out_dir / f"{name}_p{page:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(str(path))
    return paths


def cmd_frames(ctx: Ctx, args: argparse.Namespace) -> int:
    start, stop = ctx.check_range(args.start, args.stop)
    if args.crop:
        x1, y1, x2, y2 = args.crop
        if not (0 <= x1 < x2 <= ctx.w and 0 <= y1 < y2 <= ctx.h):
            raise ValueError("crop must lie inside the image")
    else:
        x1, y1, x2, y2 = 0, 0, ctx.w, ctx.h
    scale = args.scale if args.scale else (0.25 if not args.crop else 1.0)
    drawer = Drawer(ctx, args.draw)
    tiles: list[NDArray[np.uint8]] = []
    for index, image in iter_frames(ctx.video, ctx.manifest, start, stop, ctx.work):
        if (index - start) % args.step:
            continue
        tile = resize(np.ascontiguousarray(image[y1:y2, x1:x2]), scale)
        drawer.draw(tile, index, x1, y1, scale)
        if args.ruler:
            tile = ruler(tile, x1, y1, scale, args.ruler)
        tiles.append(labeled(tile, f"f{index}"))
    name = args.name or f"frames_{start:04d}-{stop:04d}_s{args.step}"
    header = [
        f"{ctx.task['clip_id']} [{ctx.target}] frames {start}..{stop - 1} step {args.step} draw={args.draw}",
        f"crop origin=({x1},{y1}) size=({x2 - x1},{y2 - y1}) scale={scale:g}: original x = {x1} + u/{scale:g}, y = {y1} + v/{scale:g}"
        + (
            f" (u,v inside image area; ruler ticks = original px every {args.ruler})"
            if args.ruler
            else ""
        ),
    ]
    sheet_paths = write_sheets(ctx, tiles, args.cols, header, name)
    dump(
        {"sheets": sheet_paths, "tiles": len(tiles), "origin": [x1, y1], "scale": scale}
    )
    return 0


def parse_size(text: str) -> tuple[int, int]:
    if "x" in text:
        w, h = text.lower().split("x")
        return int(w), int(h)
    return int(text), int(text)


def crop_centered(
    image: NDArray[np.uint8], cx: float, cy: float, w: int, h: int
) -> tuple[NDArray[np.uint8], int, int]:
    H, W = image.shape[:2]
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    x0 = max(0, min(W - w, x0)) if w <= W else 0
    y0 = max(0, min(H - h, y0)) if h <= H else 0
    return np.ascontiguousarray(image[y0 : y0 + h, x0 : x0 + w]), x0, y0


def cmd_crops(ctx: Ctx, args: argparse.Namespace) -> int:
    """One tile per (frame, point); points come from a file, the annotation or candidates."""
    wanted: dict[int, list[tuple[float, float, str]]] = {}
    blobs: dict[int, list[tuple[float, float]]] = {}
    start, stop = ctx.check_range(args.start, args.stop)
    if args.points:
        raw = (
            loads_json(Path(args.points).read_text(encoding="utf-8"))
            if Path(args.points).exists()
            else loads_json(args.points)
        )
        items = raw.items() if isinstance(raw, dict) else [(p["frame"], p) for p in raw]
        for key, value in items:
            index = int(key)
            if isinstance(value, dict):
                wanted.setdefault(index, []).append(
                    (float(value["x"]), float(value["y"]), str(value.get("label", "")))
                )
            else:
                wanted.setdefault(index, []).append(
                    (float(value[0]), float(value[1]), "")
                )
    elif args.source == "annotation":
        data = loads_json(ctx.annotation_path.read_text(encoding="utf-8"))
        for row in data["frames"]:
            for obj in row.get("balls", []) + row.get("players", []):
                if args.track and obj["track_id"] != args.track:
                    continue
                if "center_px" in obj:
                    if obj["center_px"] is None:
                        continue
                    x, y = obj["center_px"]
                else:
                    if obj["bbox_xyxy"] is None:
                        continue
                    bx1, by1, bx2, by2 = obj["bbox_xyxy"]
                    x, y = (bx1 + bx2) / 2, (by1 + by2) / 2
                wanted.setdefault(row["frame_index"], []).append(
                    (x, y, obj["track_id"])
                )
    elif args.source == "cands-ball":
        frames = read_cache(ctx.work / "cands_ball.json")["frames"]
        for key, entry in frames.items():
            for rank, cand in enumerate(entry["candidates"][: args.top]):
                x, y = cand["center_px"]
                blob = cand.get("blob")
                label = f"c{rank}:{cand['score']:.2f}" + (
                    "" if "blob" not in cand else (" B" if blob else " noB")
                )
                wanted.setdefault(int(key), []).append((x, y, label))
                if blob:
                    blobs.setdefault(int(key), []).append(tuple(blob["center_px"]))
    elif args.source == "cands-players":
        frames = read_cache(ctx.work / "cands_players.json")["frames"]
        for key, cands in frames.items():
            for cand in cands:
                if args.track and cand.get("tid") != args.track:
                    continue
                bx1, by1, bx2, by2 = cand["box"]
                wanted.setdefault(int(key), []).append(
                    ((bx1 + bx2) / 2, (by1 + by2) / 2, cand.get("tid", ""))
                )
    else:
        raise ValueError("give --points or --source")
    default = (96, 96) if ctx.target == "ball" else (320, 440)
    w, h = parse_size(args.size) if args.size else default
    scale = args.scale if args.scale else (2.0 if ctx.target == "ball" else 0.6)
    if args.draw is None:
        args.draw = (
            "annotation"
            if (args.source == "annotation" and not args.points)
            else "none"
        )
    drawer = Drawer(ctx, args.draw)
    tiles: list[NDArray[np.uint8]] = []
    indices = sorted(
        i for i in wanted if start <= i < stop and (i - start) % args.step == 0
    )
    if args.frames:
        chosen = set(parse_frames(args.frames, ctx.n))
        indices = [i for i in indices if i in chosen]
    if not indices:
        dump({"sheets": [], "tiles": 0, "note": "no points in range"})
        return 0
    lookup = set(indices)
    for index, image in iter_frames(
        ctx.video, ctx.manifest, indices[0], indices[-1] + 1, ctx.work
    ):
        if index not in lookup:
            continue
        for x, y, label in wanted[index]:
            tile, x0, y0 = crop_centered(image, x, y, w, h)
            tile = resize(tile, scale)
            drawer.draw(tile, index, x0, y0, scale)
            if args.mark and args.draw == "none":
                u, v = int(round((x - x0) * scale)), int(round((y - y0) * scale))
                gap = max(6, int(10 * scale))
                for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(
                        tile,
                        (u + du * gap, v + dv * gap),
                        (u + du * (gap + 6), v + dv * (gap + 6)),
                        (0, 255, 0),
                        1,
                    )
                for bx, by in blobs.get(index, []):
                    bu, bv = (
                        int(round((bx - x0) * scale)),
                        int(round((by - y0) * scale)),
                    )
                    cv2.circle(
                        tile,
                        (bu, bv),
                        max(9, int(round(9 * scale))),
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
            if args.ruler:
                tile = ruler(tile, x0, y0, scale, args.ruler)
            text = f"f{index} o=({x0},{y0})" + (f" {label}" if label else "")
            tiles.append(labeled(tile, text))
    name = (
        args.name
        or f"crops_{args.source if not args.points else 'points'}_{indices[0]:04d}-{indices[-1]:04d}"
    )
    header = [
        f"{ctx.task['clip_id']} [{ctx.target}] crops {w}x{h} scale={scale:g} draw={args.draw}",
        f"tile label o=(x0,y0) is the crop origin: original x = x0 + u/{scale:g}, y = y0 + v/{scale:g}",
    ]
    sheet_paths = write_sheets(ctx, tiles, args.cols, header, name)
    dump(
        {
            "sheets": sheet_paths,
            "tiles": len(tiles),
            "frames": len(indices),
            "scale": scale,
            "size": [w, h],
        }
    )
    return 0

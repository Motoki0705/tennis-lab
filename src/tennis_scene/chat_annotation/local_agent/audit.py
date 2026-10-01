"""Independent quality audit of adopted ball annotations (reviewer = orchestrator, not the worker).

  audit.py sample --seed S --out qa/audit_<tag>     # stratified random sample + review sheets
  audit.py score qa/audit_<tag>                      # rates from verdicts.json (written by the reviewer)

Strata (per item one ball or one frame):
  visible / interpolated / occluded   centre given: 2x crop with an open crosshair at the centre
                                      (centre left uncovered) + frame thumbnail with the crop box
  unresolved / out_of_frame           no centre: frame at 0.5x + 2x crop at the position predicted
                                      from the nearest known centres (to judge "was it findable?")
  gap                                 balls=[] within 15 frames of a visible ball of the same clip
                                      (rally edges: missed balls would show up here)
Visible items are spread evenly over channels. The sample is reproducible from the seed.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .campaign_state import channel_map, processed_path, read_state
from .common import iter_frames, load_manifest, locate_video

QUOTA = {
    "visible": 72,
    "interpolated": 12,
    "occluded": 8,
    "unresolved": 16,
    "out_of_frame": 6,
    "gap": 16,
}
CROP, ZOOM, THUMB_W = 112, 2, 224


def final_annotation(task: dict[str, Any]) -> Path:
    return processed_path(task["target"], task["clip_id"])


def predicted(
    rows: dict[int, dict[str, Any]], index: int, track: str | None
) -> tuple[float, float] | None:
    """Linear prediction from the nearest known centres (same track when given) within 15 frames."""

    def known(i: int) -> tuple[float, float] | None:
        for ball in rows.get(i, {}).get("balls", []):
            if ball["center_px"] is not None and (
                track is None or ball["track_id"] == track
            ):
                return tuple(ball["center_px"])
        return None

    before = next(
        ((i, p) for i in range(index - 1, index - 16, -1) if (p := known(i))), None
    )
    after = next(
        ((i, p) for i in range(index + 1, index + 16) if (p := known(i))), None
    )
    if before and after:
        w = (index - before[0]) / (after[0] - before[0])
        return (
            before[1][0] + w * (after[1][0] - before[1][0]),
            before[1][1] + w * (after[1][1] - before[1][1]),
        )
    return (before or after or (None, None))[1]


def collect(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    channels = channel_map()
    pools: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen: set[str] = set()
    for task_id, task in state["tasks"].items():
        if task["status"] != "adopted" or task["clip_id"] in seen:
            continue
        seen.add(task["clip_id"])
        annotation = json.loads(final_annotation(task).read_text(encoding="utf-8"))
        rows = {r["frame_index"]: r for r in annotation["frames"]}
        channel = channels.get(task["clip_id"].split("__")[0], "?")
        visible_frames = sorted(
            i
            for i, r in rows.items()
            if any(b["status"] == "visible" for b in r.get("balls", []))
        )
        base = {
            "task_id": task_id,
            "clip_id": task["clip_id"],
            "channel": channel,
            "manifest": task["manifest"],
            "attempt_dir": task["attempts"][-1]["dir"],
        }
        for index, row in rows.items():
            for ball in row.get("balls", []):
                item = {
                    **base,
                    "frame": index,
                    "status": ball["status"],
                    "track": ball["track_id"],
                    "center": ball["center_px"],
                }
                if ball["center_px"] is None:
                    item["predicted"] = predicted(rows, index, ball["track_id"])
                pools[ball["status"]].append(item)
            if not row.get("balls") and visible_frames:
                near = min(abs(index - v) for v in visible_frames)
                if near <= 15:
                    pools["gap"].append(
                        {
                            **base,
                            "frame": index,
                            "status": "gap",
                            "track": None,
                            "center": None,
                            "predicted": predicted(rows, index, None),
                            "frames_from_visible": near,
                        }
                    )
    return pools


def draw_crosshair(img: NDArray[np.uint8], x: float, y: float) -> None:
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p1 = (int(round(x + dx * 9)), int(round(y + dy * 9)))
        p2 = (int(round(x + dx * 20)), int(round(y + dy * 20)))
        cv2.line(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)


def crop2x(
    frame: NDArray[np.uint8], cx: float, cy: float, mark: bool
) -> NDArray[np.uint8]:
    H, W = frame.shape[:2]
    x0 = int(round(min(max(cx - CROP / 2, 0), W - CROP)))
    y0 = int(round(min(max(cy - CROP / 2, 0), H - CROP)))
    tile: NDArray[np.uint8] = cv2.resize(
        frame[y0 : y0 + CROP, x0 : x0 + CROP],
        (CROP * ZOOM, CROP * ZOOM),
        interpolation=cv2.INTER_CUBIC,
    )
    if mark:
        draw_crosshair(tile, (cx - x0) * ZOOM, (cy - y0) * ZOOM)
    return tile


def thumb(
    frame: NDArray[np.uint8],
    width: int,
    box_at: tuple[float, float] | None,
    color: tuple[int, int, int] = (0, 0, 255),
) -> NDArray[np.uint8]:
    H, W = frame.shape[:2]
    s = width / W
    t: NDArray[np.uint8] = cv2.resize(
        frame, (width, int(round(H * s))), interpolation=cv2.INTER_AREA
    )
    if box_at is not None:
        cx, cy = box_at
        cv2.rectangle(
            t,
            (int((cx - CROP / 2) * s), int((cy - CROP / 2) * s)),
            (int((cx + CROP / 2) * s), int((cy + CROP / 2) * s)),
            color,
            1,
        )
    return t


def label(img: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    out: NDArray[np.uint8] = np.full((img.shape[0] + 18, img.shape[1], 3), 30, np.uint8)
    out[18:] = img
    cv2.putText(
        out,
        text,
        (3, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def grid(tiles: list[NDArray[np.uint8]], cols: int) -> NDArray[np.uint8]:
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet: NDArray[np.uint8] = np.full(
        (rows * (th + 6), cols * (tw + 6), 3), 60, np.uint8
    )
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[
            r * (th + 6) : r * (th + 6) + t.shape[0],
            c * (tw + 6) : c * (tw + 6) + t.shape[1],
        ] = t
    return sheet


def cmd_sample(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    pools = collect(read_state())
    sample: list[dict[str, Any]] = []
    by_channel = collections.defaultdict(list)
    for item in pools["visible"]:
        by_channel[item["channel"]].append(item)
    channels = sorted(by_channel)
    per = QUOTA["visible"] // len(channels) if channels else 0
    for ch in channels:
        sample += rng.sample(by_channel[ch], min(per, len(by_channel[ch])))
    for status in ("interpolated", "occluded", "unresolved", "out_of_frame", "gap"):
        sample += rng.sample(pools[status], min(QUOTA[status], len(pools[status])))
    rng.shuffle(sample)
    for i, item in enumerate(sample):
        item["id"] = f"a{i:03d}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_clip = collections.defaultdict(list)
    for item in sample:
        by_clip[item["clip_id"]].append(item)
    images: dict[str, NDArray[np.uint8]] = {}
    for _clip, items in by_clip.items():
        manifest = load_manifest(Path(items[0]["manifest"]))
        wanted = {it["frame"] for it in items}
        for index, frame in iter_frames(
            locate_video(manifest), manifest, min(wanted), max(wanted) + 1
        ):
            for it in items:
                if it["frame"] == index:
                    images[it["id"]] = frame.copy()
    centred, blind = [], []
    for item in sample:
        frame = images[item["id"]]
        text = f"{item['id']} {item['status'][:5]} {item['channel'][:10]}"
        if item["center"] is not None:
            cx, cy = item["center"]
            tile = np.vstack(
                [crop2x(frame, cx, cy, True), thumb(frame, THUMB_W, (cx, cy))]
            )
            centred.append(label(tile, text))
        else:
            p = item.get("predicted")
            parts = [thumb(frame, 448, p, (0, 200, 255))]
            if p is not None:
                crop = crop2x(frame, p[0], p[1], False)
                pad = np.full((parts[0].shape[0], crop.shape[1], 3), 30, np.uint8)
                pad[: min(crop.shape[0], pad.shape[0])] = crop[: pad.shape[0]]
                parts = [np.hstack([parts[0], pad])]
            blind.append(
                label(
                    parts[0],
                    text
                    + (
                        f" d={item.get('frames_from_visible')}"
                        if item["status"] == "gap"
                        else ""
                    ),
                )
            )
    sheets = []
    for k in range(0, len(centred), 16):
        path = out / f"centred_{k // 16:02d}.jpg"
        cv2.imwrite(
            str(path), grid(centred[k : k + 16], 4), [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        sheets.append(path.name)
    for k in range(0, len(blind), 8):
        path = out / f"nocentre_{k // 8:02d}.jpg"
        cv2.imwrite(
            str(path), grid(blind[k : k + 8], 2), [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        sheets.append(path.name)
    (out / "sample.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "pool_sizes": {k: len(v) for k, v in pools.items()},
                "items": [
                    {k: v for k, v in it.items() if k != "manifest"} for it in sample
                ],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                "items": len(sample),
                "sheets": sheets,
                "pool_sizes": {k: len(v) for k, v in pools.items()},
            }
        )
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    out = Path(args.dir)
    items = {
        it["id"]: it for it in json.loads((out / "sample.json").read_text())["items"]
    }
    verdicts = json.loads((out / "verdicts.json").read_text())
    table: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for vid, verdict in verdicts.items():
        table[items[vid]["status"]][verdict] += 1
    by_channel: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for vid, verdict in verdicts.items():
        if items[vid]["status"] == "visible":
            by_channel[items[vid]["channel"]][verdict] += 1
    print(
        json.dumps(
            {
                "by_status": {k: dict(v) for k, v in table.items()},
                "visible_by_channel": {k: dict(v) for k, v in by_channel.items()},
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("dir")
    args = parser.parse_args(argv)
    return {"sample": cmd_sample, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .selection import load_campaign
from .storage import clip_root, digest, write_json


def _color(track_id: int) -> tuple[int, int, int]:
    return (
        80 + track_id * 47 % 175,
        80 + track_id * 83 % 175,
        80 + track_id * 131 % 175,
    )


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"Image write failed: {path}")


def prepare(campaign: Path, index: int, directory: Path) -> dict[str, Any]:
    """Full frame coverage, with separately labelled identity crop timelines."""
    _, plan, store = load_campaign(campaign)
    root = clip_root(campaign, index)
    with np.load(root / "tracks.npz", allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    clip = store.clips[index]
    observed = raw["detection_rows"] >= 0
    directory.mkdir(parents=True, exist_ok=True)
    sheets = []
    width = 480
    height = int(round(clip.height * width / clip.width))
    # Every frame is shown. A frame label and raw ID label are never implicit.
    for start in range(0, clip.frame_count, 20):
        stop = min(start + 20, clip.frame_count)
        sheet: NDArray[np.uint8] = np.full(
            (4 * (height + 26), 5 * width, 3), 28, np.uint8
        )
        for frame in range(start, stop):
            original = store.read_bgr(store.row_of(clip, frame))
            image = cv2.resize(original, (width, height))
            for at in np.flatnonzero(observed[:, frame]):
                t = int(raw["track_ids"][at])
                box = raw["boxes"][at, frame] * (width / clip.width)
                a, b, c, d = np.rint(box).astype(int)
                cv2.rectangle(image, (a, b), (c, d), _color(t), 2)
                cv2.putText(
                    image,
                    f"ID {t}",
                    (max(a, 0), max(b - 3, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _color(t),
                    2,
                )
            r, c = divmod(frame - start, 5)
            y = r * (height + 26)
            x = c * width
            sheet[y : y + height, x : x + width] = image
            cv2.putText(
                sheet,
                f"frame {frame}",
                (x + 5, y + height + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
        name = f"frames-{start:06d}-{stop:06d}.jpg"
        _save(directory / name, sheet)
        sheets.append(name)
    tracks = []
    contacts = []
    for at, track_id in enumerate(raw["track_ids"]):
        frames = np.flatnonzero(observed[at])
        selected = frames[
            np.unique(np.linspace(0, len(frames) - 1, min(20, len(frames))).astype(int))
        ]
        sheet = np.full((4 * 226, 5 * 160, 3), 28, np.uint8)
        for slot, frame in enumerate(selected):
            image = store.read_bgr(store.row_of(clip, int(frame)))
            x1, y1, x2, y2 = raw["boxes"][at, frame]
            x1, x2 = np.clip([int(np.floor(x1)), int(np.ceil(x2))], 0, clip.width)
            y1, y2 = np.clip([int(np.floor(y1)), int(np.ceil(y2))], 0, clip.height)
            if x2 > x1 and y2 > y1:
                crop = cv2.resize(image[y1:y2, x1:x2], (160, 200))
                r, c = divmod(slot, 5)
                sheet[r * 226 : r * 226 + 200, c * 160 : (c + 1) * 160] = crop
                cv2.putText(
                    sheet,
                    f"f{frame}",
                    (c * 160 + 4, r * 226 + 219),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
        name = f"track-{int(track_id):06d}.jpg"
        _save(directory / name, sheet)
        contacts.append(name)
        tracks.append(
            {
                "raw_track_id": int(track_id),
                "observed_frames": frames.tolist(),
                "contact_sheet": name,
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
            }
        )
    packet = {
        "clip": plan["clips"][index],
        "raw_tracks_sha256": digest(root / "tracks.npz"),
        "required_frame_sheets": sheets,
        "track_contacts": contacts,
        "tracks": tracks,
        "frame_sheet_coverage": "all source frames; each sheet has dense, labelled frames",
        "court_policy": "disabled; select people from visual play context only",
    }
    write_json(directory / "packet.json", packet)
    return packet

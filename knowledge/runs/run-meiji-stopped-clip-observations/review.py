"""Extract synchronized evidence frames for the original 14 #2363 stops."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np
from inputs import CAMERAS, Archive, ArchivedClip, CameraRows, write_json
from omegaconf import OmegaConf

from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices

WIDTH = 960
HEIGHT = 540
HEADER = 30
COLORS = {
    "untracked": (204, 204, 204),
    "tracked": (245, 176, 29),
    "selected": (85, 215, 38),
    "focus": (15, 132, 255),
}


@dataclass(frozen=True)
class Focus:
    camera: str
    kind: str
    identity: int
    start: int
    end: int

    def observed(self, rows: CameraRows) -> np.ndarray:
        array, ids = (
            (rows.group_origins, rows.group_ids)
            if self.kind == "group"
            else (rows.track_rows, rows.track_ids)
        )
        (index,) = np.flatnonzero(ids == self.identity)
        return np.flatnonzero(array[index, self.start : self.end] >= 0) + self.start

    def matches(self, frame: int, row: dict[str, Any]) -> bool:
        return self.start <= frame < self.end and row[self.kind] == self.identity


def candidate(association: dict[str, Any], candidate_id: int) -> Focus:
    segment = association["segments"][association["candidates"][candidate_id]]
    return Focus(
        segment["camera"],
        "group",
        segment["track_id"],
        segment["start"],
        segment["end"],
    )


def figure_frames(
    clip: ArchivedClip, views: dict[str, CameraRows], max_frames: int
) -> tuple[dict[str, Any], list[Focus], list[int], list[int]]:
    reason = clip.stop["reason"]
    diagnostics = clip.stop["diagnostics"]
    count = clip.source_record["num_frames"]
    focus: list[Focus] = []
    decision: dict[str, Any] = {}
    if reason == "court_side_no_consistent_hypothesis":
        pool = sampled_frame_indices(
            count, clip.source_record["fps"], max_frames=max_frames
        )
        if len(pool) != diagnostics["sampled_frames"]:
            raise ValueError(
                f"{clip.name}: court-side sampling grid differs from #2363"
            )
        decision = {"stage": "court_side", **diagnostics}
    else:
        association = diagnostics["association"]
        identities = {row["identity"]: row for row in association["identities"]}
        decision["stage"] = "player_association"
        if reason == "player_association_ambiguous_players":
            (side,) = [
                side
                for side, row in association["selection"].items()
                if row["runner_up"] is not None
            ]
            selection = association["selection"][side]
            runner = identities[selection["runner_up"]]
            boundary = identities[selection["players"][-1]]
            focus = [candidate(association, i) for i in runner["candidates"]]
            decision.update(side=int(side), runner_up=runner, boundary=boundary)
        elif reason == "player_association_ambiguous_association":
            pair = min(association["ambiguous_pairs"], key=lambda pair: pair["margin"])
            focus = [candidate(association, pair[k]) for k in ("a", "b")]
            decision.update(
                pair=pair, ambiguous_pairs=len(association["ambiguous_pairs"])
            )
        elif reason == "player_association_player_not_found":
            (side,) = [
                side
                for side, row in association["selection"].items()
                if len(row["players"]) < 2
            ]
            if set(clip.alley_tracks) != set(views):
                raise ValueError(
                    f"{clip.name}: expected the four #2440 raw tracks for the short side"
                )
            focus = [
                Focus(camera, "track", clip.alley_tracks[camera], 0, count)
                for camera in views
            ]
            decision.update(
                side=int(side), selected=association["selection"][side]["players"]
            )
        else:
            raise ValueError(f"{clip.name}: unexpected stop reason {reason}")
        observed = [item.observed(views[item.camera]) for item in focus]
        if reason == "player_association_ambiguous_association":
            pool = np.intersect1d(*observed)
        else:
            pool = np.unique(np.concatenate(observed))
        if not len(pool):
            raise ValueError(f"{clip.name}: the decision has no observed frames")
        gaps = [
            (int(a), int(b), item.camera, item.identity)
            for item, times in zip(focus, observed, strict=True)
            for a, b in zip(times[:-1], times[1:], strict=True)
            if b - a >= 4
        ]
        if gaps:
            before, after, camera, track = max(
                gaps, key=lambda gap: (gap[1] - gap[0], -gap[0])
            )
            decision["longest_internal_gap"] = {
                "camera": camera,
                "id": track,
                "before": before,
                "after": after,
            }

    anchors = [int(pool[(len(pool) - 1) * percent // 100]) for percent in (20, 50, 80)]
    frames = {
        max(0, min(count - 1, anchor + offset))
        for anchor in anchors
        for offset in (-6, 0, 6)
    }
    gap = decision.get("longest_internal_gap")
    if gap is not None:
        before, after = gap["before"], gap["after"]
        frames.update((before, (before + after) // 2, after))
    decision.update(reason=reason, error=clip.stop["error"], eligible_frames=len(pool))
    return decision, focus, anchors, sorted(frames)


class FrameReader:
    """Read fixed source indices with one FFmpeg decoder thread per view."""

    def __init__(self, video: SourceVideo) -> None:
        self.video = video
        self.capture = cv2.VideoCapture(
            str(video.path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )

    def __enter__(self) -> FrameReader:
        if (
            not self.capture.isOpened()
            or int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) != self.video.num_frames
        ):
            self.capture.release()
            raise ValueError(
                f"{self.video.path}: could not open the pinned frame count"
            )
        return self

    def __exit__(self, *exc: object) -> None:
        self.capture.release()

    def decode(self, frame: int) -> np.ndarray:
        current = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        if 0 <= frame - current <= 24:
            for _ in range(current, frame):
                if not self.capture.grab():
                    raise ValueError(
                        f"{self.video.path}: skipped frame could not be read before {frame}"
                    )
        elif not self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
            raise ValueError(f"{self.video.path}: could not seek to {frame}")
        ok, image = self.capture.read()
        if (
            not ok
            or image.shape != (self.video.height, self.video.width, 3)
            or int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1
        ):
            raise ValueError(f"{self.video.path}: could not read source frame {frame}")
        return image


def render_overlay(
    image: np.ndarray, frame: int, rows: list[dict[str, Any]], focus: list[Focus]
) -> np.ndarray:
    overlay = image.copy()
    sx = image.shape[1] / 1920
    sy = image.shape[0] / 1080
    for row in sorted(
        rows,
        key=lambda row: (
            any(f.matches(frame, row) for f in focus),
            row["selected"],
            row["track"] is not None,
        ),
    ):
        if any(f.matches(frame, row) for f in focus):
            kind = "focus"
        elif row["selected"]:
            kind = "selected"
        else:
            kind = "tracked" if row["track"] is not None else "untracked"
        x1, y1, x2, y2 = (
            int(round(coordinate * scale))
            for coordinate, scale in zip(row["box"], (sx, sy, sx, sy), strict=True)
        )
        color = COLORS[kind]
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            2 if kind in {"focus", "selected"} else 1,
            cv2.LINE_AA,
        )
        if row["track"] is None:
            continue
        label = f"t{row['track']}"
        if row["group"] is not None:
            label += f" g{row['group']}"
        elif row["selected"]:
            label += " S"
        size = 0.45 if kind in {"focus", "selected"} else 0.33
        anchor = (max(0, x1), max(12, y1 - 2))
        cv2.putText(
            overlay,
            label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (14, 14, 14),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def mosaic(images: list[np.ndarray], clip: str, frame: int) -> np.ndarray:
    top = np.hstack(images[:2])
    bottom = np.hstack(images[2:])
    height, width = images[0].shape[:2]
    result = np.zeros((height * 2 + HEADER, width * 2, 3), dtype=np.uint8)
    result[HEADER : HEADER + height] = top
    result[HEADER + height :] = bottom
    cv2.putText(
        result,
        f"{clip} / source frame {frame}",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for i, camera in enumerate(CAMERAS):
        left, upper = (i % 2) * width, HEADER + (i // 2) * height
        cv2.rectangle(result, (left, upper), (left + 60, upper + 24), (18, 18, 18), -1)
        cv2.putText(
            result,
            camera,
            (left + 7, upper + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def save_jpg(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Could not write {path}")


def extract_clip(
    clip: ArchivedClip, *, output: Path, max_frames: int
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    store = clip.store()
    source = clip.source(store, verify_media=True)
    if source.size != (1920, 1080):
        raise ValueError(
            f"{clip.name}: review figures expect the archived native 1920x1080 media"
        )
    views = {
        camera: CameraRows.load(clip, store, camera) for camera in source.camera_ids
    }
    decision, focus, anchors, frames = figure_frames(clip, views, max_frames)
    clip_out = output / clip.name
    clip_out.mkdir()
    (clip_out / "native").mkdir()
    observations = {}
    with ExitStack() as stack:
        readers = [stack.enter_context(FrameReader(video)) for video in source.videos]
        for frame in frames:
            raw, overlay = [], []
            observations[frame] = {}
            for video, reader in zip(source.videos, readers, strict=True):
                camera = video.camera_id
                image = reader.decode(frame)
                save_jpg(clip_out / "native" / f"f{frame:04d}_{camera}.jpg", image)
                panel = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                raw.append(panel)
                rows = views[camera].rows_at(frame)
                observations[frame][camera] = rows
                overlay.append(
                    render_overlay(
                        panel,
                        frame,
                        rows,
                        [item for item in focus if item.camera == camera],
                    )
                )
            save_jpg(clip_out / f"f{frame:04d}_raw.jpg", mosaic(raw, clip.name, frame))
            save_jpg(
                clip_out / f"f{frame:04d}_overlay.jpg",
                mosaic(overlay, clip.name, frame),
            )
    write_json(clip_out / "observations.json", observations)
    artifacts = {}
    for video in source.videos:
        camera = video.camera_id
        artifacts[camera] = {"media_sha256": video.sha256}
        for name in ("person_detection", "person_tracking", "player_selection"):
            reference = store.references[f"{name}/{camera}"]
            artifacts[camera][name] = {
                key: reference[key] for key in ("artifact_id", "sha256")
            }
    result = {
        "clip": clip.name,
        "source_frames": source.num_frames,
        "decision": decision,
        "focus": [asdict(item) for item in focus],
        "anchors": anchors,
        "frames": frames,
        "artifacts": artifacts,
    }
    write_json(clip_out / "index.json", result)
    print(
        f"{clip.name}: {len(frames)} synchronized frames, {decision['reason']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("i2363_run", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--clip",
        action="append",
        help="Optional clip name such as clip_002 for a probe",
    )
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    start = monotonic()
    archive = Archive(args.i2363_run.resolve())
    config = OmegaConf.create(archive.progress["pipeline_config_yaml"])
    if args.clip and not set(args.clip) <= {
        clip.name for clip in archive.clips.values()
    }:
        parser.error("--clip must name a stopped #2363 clip")
    clips = [
        clip
        for clip in archive.clips.values()
        if not args.clip or clip.name in args.clip
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    render = partial(
        extract_clip,
        output=args.output,
        max_frames=int(config.frame_sampling.max_frames),
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(render, clips))
    write_json(
        args.output / "index.json", {"source": archive.receipt, "clips": results}
    )
    captures = sum(len(clip["frames"]) for clip in results)
    print(
        f"Completed {len(results)} clips, {captures} synchronized frames, {4 * captures} camera frames in {monotonic() - start:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

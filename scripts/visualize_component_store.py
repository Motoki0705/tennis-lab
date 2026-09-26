"""Render every published clip component into a local, static review gallery.

The gallery is derived only from scene.json's currently adopted artifacts. Run it
again after a pipeline run adds artifacts; inference is never invoked here.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

plt.switch_backend("Agg")


SCHEMAS = {
    "ball_detection": ("ball_detections", 1),
    "court_detection": ("court_observations", 2),
    "court_calibration": ("local_court_calibration", 1),
    "person_detection": ("person_detections", 1),
    "person_tracking": ("person_tracks", (2, 3)),
    "pose_estimation": ("person_poses", 1),
    "person_reid": ("person_identities", 1),
    "court_side": ("court_side", 1),
    "camera_alignment": ("aligned_cameras", 1),
    "player_triangulation": ("player_skeletons", 1),
    "ball_triangulation": ("ball_trajectory", 1),
    "ball_smoothing": ("smoothed_ball_trajectory", 1),
    "body_view_selection": ("body_view_selection", 1),
    "gvhmr": ("body_parameters", 1),
    "body_placement": ("placed_bodies", 1),
    "scene_assembly": ("scene_result", 2),
}
CAMERA_COMPONENTS = ("court_detection", "ball_detection", "person_detection", "person_tracking", "pose_estimation")
ORDER = ("court_detection", "court_calibration", "ball_detection", "person_detection",
         "person_tracking", "pose_estimation", "person_reid", "court_side", "camera_alignment",
         "player_triangulation", "ball_triangulation", "ball_smoothing", "body_view_selection", "gvhmr",
         "body_placement", "scene_assembly")
COLORS_BGR = ((48, 180, 255), (52, 220, 75), (230, 100, 255), (255, 170, 48))
COLORS_MPL = ("#ed8151", "#4bb985", "#9b70d7", "#4c92d5")
SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return cast(np.ndarray, value.detach().cpu().numpy())
    return cast(np.ndarray, np.asarray(value))


def _count(mask: Any) -> int:
    return int(np.count_nonzero(_array(mask)))


def _text(frame: np.ndarray, message: str, at: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, message, at, cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, message, at, cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2, cv2.LINE_AA)


def _save_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")
    return str(path.name)


def _save_plot(path: Path, draw: Callable[[Any], None], *, figsize: tuple[float, float] = (10, 5)) -> str:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    try:
        draw(ax)
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)
    return str(path.name)


def _court_axes(ax: Any) -> None:
    width, length = HALF_DOUBLES_WIDTH, HALF_LENGTH
    ax.plot([-width, width, width, -width, -width], [-length, -length, length, length, -length], color="#596b72")
    ax.plot([-width, width], [0, 0], color="#596b72", linestyle="--")
    ax.set_xlim(-width - 5, width + 5)
    ax.set_ylim(-length - 5, length + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("court x (m)")
    ax.set_ylabel("court y (m)")
    ax.grid(alpha=.15)


def _calibration_overlay(image: np.ndarray, _: int, *, points: np.ndarray, visible: np.ndarray,
                         polygon: Any, size: tuple[int, int]) -> None:
    if polygon is not None:
        cv2.polylines(image, [np.rint(polygon).astype(np.int32)], True, (70, 210, 90), 5)
    for point, score in zip(points, visible, strict=True):
        if score > 0:
            pixel = tuple(np.rint(point * (np.asarray(size) - 1)).astype(int))
            cv2.circle(image, pixel, 7, (60, 175, 255), -1)


def _dashed_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = np.rint(box).astype(int)
    for start, end in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                       ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = float(np.linalg.norm(np.subtract(end, start)))
        for step in range(0, int(np.ceil(length)), 18):
            a = step / length
            b = min(step + 9, length) / length
            p = tuple(np.rint(np.asarray(start) * (1 - a) + np.asarray(end) * a).astype(int))
            q = tuple(np.rint(np.asarray(start) * (1 - b) + np.asarray(end) * b).astype(int))
            cv2.line(image, p, q, color, 4)


class Review:
    def __init__(self, index: Path, output: Path, *, videos: bool = False) -> None:
        self.index_path, self.output = index.resolve(), output.resolve()
        document = json.loads(self.index_path.read_text())
        if document.get("schema") != "tennis_scene_index_v1":
            raise ValueError("Expected tennis_scene_index_v1 scene.json")
        self.source = document["source"]
        self.camera_ids = tuple(video["camera_id"] for video in self.source["videos"])
        self.videos = {video["camera_id"]: Path(video["path"]) for video in self.source["videos"]}
        self.frame_count = int(self.source["videos"][0]["num_frames"])
        self.size = (int(self.source["videos"][0]["width"]), int(self.source["videos"][0]["height"]))
        self.store = ClipStore(self.index_path.parent, self.source, memory_entries=0)
        self.references = self.store.references()
        self._active_by_artifact = {reference.artifact_id: node for node, reference in self.references.items()}
        self._staleness: dict[str, tuple[str, ...]] = {}
        self._frames: dict[tuple[str, int], np.ndarray] = {}
        self.make_videos = videos
        self.movies: dict[str, str] = {}
        self.raw_cosine_written = False
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "images").mkdir(exist_ok=True)
        if videos:
            (self.output / "videos").mkdir(exist_ok=True)

    def frame(self, camera: str, index: int) -> np.ndarray:
        key = (camera, index)
        if key not in self._frames:
            capture = cv2.VideoCapture(str(self.videos[camera]))
            try:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise OSError(f"Cannot seek {camera} frame {index}")
                okay, image = capture.read()
                if not okay:
                    raise OSError(f"Cannot decode {camera} frame {index}")
                self._frames[key] = image
            finally:
                capture.release()
        return self._frames[key].copy()

    def samples(self) -> tuple[int, ...]:
        standard = np.rint(np.linspace(0, self.frame_count - 1, 5)).astype(int).tolist()
        return tuple(sorted(set(min(self.frame_count - 1, index) for index in standard)))

    def sheet(self, node: str, camera: str, draw: Callable[[np.ndarray, int], None], *,
              frames: tuple[int, ...] | None = None) -> str:
        cells: list[np.ndarray] = []
        for index in self.samples() if frames is None else frames:
            image = self.frame(camera, index)
            draw(image, index)
            cell = cv2.resize(image, (720, 405), interpolation=cv2.INTER_AREA)
            cv2.rectangle(cell, (0, 0), (719, 43), (25, 33, 39), -1)
            _text(cell, f"{camera} | frame {index}", (12, 31))
            cells.append(cell)
        columns = min(3, len(cells))
        blank = np.full_like(cells[0], 245)
        while len(cells) % columns:
            cells.append(blank)
        sheet = np.vstack([np.hstack(cells[row:row + columns]) for row in range(0, len(cells), columns)])
        return _save_image(self.output / "images" / f"{node.replace('/', '_')}_frames.png", sheet)

    def movie(self, node: str, camera: str, draw: Callable[[np.ndarray, int], None]) -> None:
        if not self.make_videos:
            return
        source = next(video for video in self.source["videos"] if video["camera_id"] == camera)
        capture = cv2.VideoCapture(str(self.videos[camera]))
        filename = f"{node.replace('/', '_')}.mp4"
        destination = self.output / "videos" / filename
        partial = destination.with_name(f"{destination.stem}.partial.mp4")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", "1280x720", "-r", str(float(source["fps"])),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            if not capture.isOpened() or encoder.stdin is None or encoder.stderr is None:
                raise OSError(f"Could not open review video stream for {node}")
            for index in range(self.frame_count):
                okay, image = capture.read()
                if not okay:
                    raise OSError(f"Review video ended before frame {index}: {camera}")
                draw(image, index)
                encoder.stdin.write(cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA).tobytes())
            encoder.stdin.close()
            errors = encoder.stderr.read().decode(errors="replace")
            if encoder.wait() != 0:
                raise RuntimeError(f"H.264 review encoding failed for {node}: {errors}")
            os.replace(partial, destination)
        except BaseException:
            encoder.kill()
            encoder.wait()
            partial.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
        self.movies[node] = f"videos/{filename}"

    def track_samples(self, boxes: np.ndarray, observed: np.ndarray,
                      source_ids: list[list[int]], links: list[dict[str, Any]]) -> tuple[int, ...]:
        """Include the largest gaps and duplicate-box handoffs in the contact sheet."""
        chosen = set(self.samples())
        linked_ids = {int(item["earlier_id"]) for item in links} | {int(item["later_id"]) for item in links}
        for row, members in enumerate(source_ids):
            if not linked_ids.intersection(members):
                continue
            present: np.ndarray = observed[row].astype(bool)
            frames = np.flatnonzero(present)
            if len(frames) < 2:
                continue
            gaps = sorted(((int(a), int(b)) for a, b in zip(frames[:-1], frames[1:], strict=False) if b - a > 1),
                          key=lambda pair: pair[1] - pair[0], reverse=True)[:2]
            for before, after in gaps:
                chosen.update((before, (before + after) // 2, after))
            if any(item.get("overlap_span_frames", 0) for item in links):
                width = boxes[row, :, 2] - boxes[row, :, 0]
                pair_valid = present[:-1] & present[1:] & (width[:-1] > 0) & (width[1:] > 0)
                changes: np.ndarray = np.full(self.frame_count - 1, -1., np.float64)
                changes[pair_valid] = np.abs(np.log(width[1:][pair_valid] / width[:-1][pair_valid]))
                if pair_valid.any():
                    boundary = int(np.argmax(changes))
                    chosen.update((boundary - 3, boundary, boundary + 1, boundary + 3))
        return tuple(sorted(index for index in chosen if 0 <= index < self.frame_count))

    def payload(self, node: str) -> tuple[dict[str, Any], dict[str, Any]]:
        reference = self.references[node]
        name = node.split("/")[0]
        expected_schema, versions = SCHEMAS[name]
        supported_versions = versions if isinstance(versions, tuple) else (versions,)
        if reference.schema != expected_schema or reference.version not in supported_versions:
            raise ValueError(f"Unsupported {node} schema/version: {reference.schema} v{reference.version}")
        descriptor = self.store.descriptor(reference)
        location = (self.store.root / reference.path).resolve().parent
        payload = unpack_value(descriptor["payload"], location, descriptor["arrays"])
        if not isinstance(payload, dict):
            raise TypeError(f"{node} must have a structured component payload")
        return payload, descriptor

    def stale_dependencies(self, node: str, ancestors: frozenset[str] = frozenset()) -> tuple[str, ...]:
        """Do not show descendants of an artifact superseded in scene.json."""
        if node in self._staleness:
            return self._staleness[node]
        if node in ancestors:
            raise ValueError(f"Cyclic component artifact dependencies at {node}")
        descriptor = self.store.descriptor(self.references[node])
        reasons: list[str] = []
        for port, dependency in descriptor["dependencies"].items():
            producer = self._active_by_artifact.get(dependency["artifact_id"])
            if producer is None:
                reasons.append(f"{port}: upstream artifact {dependency['artifact_id'][:12]} was superseded")
            elif self.stale_dependencies(producer, ancestors | {node}):
                reasons.append(f"{port}: upstream component {producer} is stale")
        result = tuple(reasons)
        self._staleness[node] = result
        return result

    def render(self, node: str) -> tuple[list[str], list[tuple[str, str]]]:
        payload, descriptor = self.payload(node)
        name, _, camera = node.partition("/")
        renderer = getattr(self, f"render_{name}")
        images, details = renderer(node, camera, payload)
        details.insert(0, ("origin", str(descriptor["provenance"].get("origin", "unknown"))))
        details.insert(1, ("artifact", str(descriptor["artifact_id"])[:16]))
        details.insert(2, ("schema", f"{self.references[node].schema} v{self.references[node].version}"))
        return images, details

    def confirmed_track_labels(self, camera: str, local_ids: np.ndarray) -> dict[int, int] | None:
        """Only a confirmed import can label a raw person track as a target."""
        if "person_reid" not in self.references or self.stale_dependencies("person_reid"):
            return None
        reference = self.references["person_reid"]
        if self.store.descriptor(reference)["provenance"].get("origin") != "confirmed_person_association":
            return None
        value, _ = self.payload("person_reid")
        if value["result"] is None:
            raise ValueError("Confirmed person association has no output")
        view = value["camera_ids"].index(camera)
        raw_ids = _array(value["result"]["raw_track_ids"])
        if len(local_ids) > raw_ids.shape[1]:
            raise ValueError("Confirmed person assignments do not cover tracking rows")
        return {int(track_id): int(raw_ids[view, row]) for row, track_id in enumerate(local_ids)}

    def render_court_detection(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        points, visible = _array(value["keypoints"])[0, 0], _array(value["visibility"])[0, 0]
        if _array(value["frame_indices"]).tolist() != [0]:
            raise ValueError("Court reviewer expects an actual frame-zero observation")
        def draw(image: np.ndarray, _: int) -> None:
            for index, (point, score) in enumerate(zip(points, visible, strict=True)):
                if score <= 0:
                    continue
                pixel = tuple(np.rint(point * (np.asarray(self.size) - 1)).astype(int))
                cv2.circle(image, pixel, 8, (40, 210, 255), -1)
                _text(image, str(index), (pixel[0] + 9, pixel[1] - 9), (40, 210, 255))
        image = self.sheet(node, camera, draw, frames=(0,))
        diagnostics = value["diagnostics"]
        frame = diagnostics["cameras"][0]["frames"][0]
        return [image], [("observed frame", "0 only"), ("visible keypoints", f"{_count(visible)}/14"),
                         ("joint fitting", str(frame.get("status", "unknown")))]

    def render_court_calibration(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        images: list[str] = []
        court = value["court"]
        for view, camera_id in enumerate(self.camera_ids):
            points = _array(court["keypoints"])[view, 0]
            visible = _array(court["visibility"])[view, 0]
            polygon = value["footpoint_polygons"][camera_id]
            images.append(self.sheet(f"{node}_{camera_id}", camera_id,
                                     partial(_calibration_overlay, points=points, visible=visible, polygon=polygon, size=self.size)))
        views = value["calibration"]["views"]
        details = [("temporal policy", court["diagnostics"]["temporal_policy"]),
                   ("reference camera", value["reference_camera"])]
        details += [(f"{item['camera']['camera_id']} calibration RMSE", f"{item['rmse_px']:.2f} px; support={item['support_frames']}") for item in views]
        return images, details

    def render_ball_detection(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        position, kinds = _array(value["uv_px"]), _array(value["point_kind"])
        labels = ("absent", "observed", "interpolated", "occlusion estimate")
        colors = ((100, 100, 100), (30, 220, 255), (30, 140, 255), (200, 60, 255))
        def draw(image: np.ndarray, frame: int) -> None:
            kind = int(kinds[frame])
            if kind:
                point = tuple(np.rint(position[frame]).astype(int))
                cv2.circle(image, point, 15, colors[kind], 4)
                _text(image, labels[kind], (point[0] + 18, point[1]), colors[kind])
            else:
                _text(image, "ball unresolved", (60, 110))
        image = self.sheet(node, camera, draw)
        self.movie(node, camera, draw)
        path = self.output / "images" / f"{camera}_ball_timeline.png"
        def plot(ax: Any) -> None:
            for coordinate, label in enumerate(("x", "y")):
                ax.plot(np.arange(len(position)), position[:, coordinate], lw=1, alpha=.6, label=f"{label} px")
            for kind in (1, 2, 3):
                frames = np.flatnonzero(kinds == kind)
                ax.scatter(frames, position[frames, 1], s=5, label=labels[kind])
            ax.set(xlabel="source frame", ylabel="pixel coordinate", title=f"{camera} ball output")
            ax.legend(loc="upper right", ncol=2)
        timeline = _save_plot(path, plot)
        details = [(labels[kind], str(_count(kinds == kind))) for kind in range(4)]
        details.append(("score semantics", str(value["score_semantics"])))
        return [image, timeline], details

    def render_person_detection(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        offsets, boxes, scores = (_array(value[key]) for key in ("frame_offsets", "boxes_xyxy", "confidence"))
        def draw(image: np.ndarray, frame: int) -> None:
            for box, score in zip(boxes[offsets[frame]:offsets[frame + 1]], scores[offsets[frame]:offsets[frame + 1]], strict=True):
                cv2.rectangle(image, tuple(np.rint(box[:2]).astype(int)), tuple(np.rint(box[2:]).astype(int)), (70, 220, 85), 4)
                _text(image, f"{score:.2f}", tuple(np.rint(box[:2]).astype(int)), (70, 220, 85))
        image = self.sheet(node, camera, draw)
        self.movie(node, camera, draw)
        counts = np.diff(offsets)
        def plot(ax: Any) -> None:
            ax.plot(counts, lw=1, color=COLORS_MPL[1])
            ax.set(xlabel="source frame", ylabel="detected people", ylim=(-.1, max(4, int(counts.max()) + 1)), title=f"{camera} detections")
            ax.grid(alpha=.2)
        timeline = _save_plot(self.output / "images" / f"{camera}_detections_timeline.png", plot)
        return [image, timeline], [("detections", str(len(scores))), ("max in one frame", str(int(counts.max())))]

    def render_person_tracking(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        boxes, observed, ids = (_array(value[key]) for key in ("boxes_xyxy", "observed", "track_ids"))
        confirmed = self.confirmed_track_labels(camera, ids)
        def draw(image: np.ndarray, frame: int) -> None:
            for row, track_id in enumerate(ids):
                box = boxes[row, frame]
                global_id = None if confirmed is None else confirmed[int(track_id)]
                color = (150, 150, 150) if global_id == -1 else COLORS_BGR[row % len(COLORS_BGR)]
                label = f"track {track_id}" if global_id is None else (f"excluded track {track_id}" if global_id < 0 else f"player {global_id} / track {track_id}")
                if not observed[row, frame]:
                    frames = np.flatnonzero(observed[row])
                    if len(frames) and frames[0] < frame < frames[-1]:
                        _dashed_box(image, box, color)
                        _text(image, f"{label} interp", tuple(np.rint(box[:2]).astype(int)), color)
                    continue
                cv2.rectangle(image, tuple(np.rint(box[:2]).astype(int)), tuple(np.rint(box[2:]).astype(int)), color, 4)
                _text(image, label, tuple(np.rint(box[:2]).astype(int)), color)
        frames = self.track_samples(boxes, observed, value["source_track_ids"], value["tracklet_links"])
        image = self.sheet(node, camera, draw, frames=frames)
        self.movie(node, camera, draw)
        def plot(ax: Any) -> None:
            for row in range(len(ids)):
                frames = np.flatnonzero(observed[row])
                ax.scatter(frames, np.full(len(frames), row), marker="|", s=25, color=COLORS_MPL[row % len(COLORS_MPL)])
            ax.set(xlabel="source frame", ylabel="stable camera ID", yticks=np.arange(len(ids)), yticklabels=[str(v) for v in ids], title=f"{camera} observed tracks")
            ax.grid(axis="x", alpha=.2)
        timeline = _save_plot(self.output / "images" / f"{camera}_tracks_timeline.png", plot)
        details = [(f"ID {track_id}", f"{_count(observed[row])} observed frames; source tracklets {value['source_track_ids'][row]}") for row, track_id in enumerate(ids)]
        if confirmed is not None:
            details += [(f"track {track_id} target selection", "excluded from triangulation/GVHMR/scene" if confirmed[int(track_id)] < 0 else f"confirmed player {confirmed[int(track_id)]}") for track_id in ids]
        for item in value["tracklet_links"]:
            overlap = (f"overlap span {item['overlap_span_frames']} frames, {item['shared_observation_frames']} shared observations, "
                       f"containment {item['duplicate_containment']:.2f}; ") if item.get("overlap_span_frames") else ""
            details.append((f"link {item['earlier_id']} → {item['later_id']}",
                            f"gap {item['missing_frames']} frames; {overlap}location {item['center_distance_diagonals']:.2f} box diagonals; "
                            f"clothing ΔLab {item['appearance_lab_distance']:.1f}"))
        return [image, timeline], details

    def render_pose_estimation(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        points = _array(value["uv_px"])[0]
        confidence = _array(value["confidence"])[0]
        observed = _array(value["observed"])[0]
        ids = _array(value["local_track_ids"])[0]
        confirmed = self.confirmed_track_labels(camera, ids)
        def draw(image: np.ndarray, frame: int) -> None:
            for row, track_id in enumerate(ids):
                if not observed[frame, row]:
                    continue
                global_id = None if confirmed is None else confirmed[int(track_id)]
                color = (150, 150, 150) if global_id == -1 else COLORS_BGR[row % len(COLORS_BGR)]
                valid = confidence[frame, row] >= .15
                for a, b in SKELETON:
                    if valid[a] and valid[b]:
                        cv2.line(image, tuple(np.rint(points[frame, row, a]).astype(int)),
                                 tuple(np.rint(points[frame, row, b]).astype(int)), color, 4)
                for point in points[frame, row, valid]:
                    cv2.circle(image, tuple(np.rint(point).astype(int)), 6, color, -1)
                if valid.any():
                    anchor = tuple(np.rint(points[frame, row, valid][0]).astype(int))
                    label = f"track {track_id}" if global_id is None else (f"excluded track {track_id}" if global_id < 0 else f"player {global_id} / track {track_id}")
                    _text(image, label, anchor, color)
        image = self.sheet(node, camera, draw)
        self.movie(node, camera, draw)
        def plot(ax: Any) -> None:
            for row, track_id in enumerate(ids):
                mean = np.where(observed[:, row], confidence[:, row].mean(-1), np.nan)
                ax.plot(mean, label=f"ID {track_id}", color=COLORS_MPL[row % len(COLORS_MPL)])
            ax.set(xlabel="source frame", ylabel="mean COCO17 confidence", ylim=(0, 1), title=f"{camera} pose output")
            ax.legend()
        timeline = _save_plot(self.output / "images" / f"{camera}_pose_timeline.png", plot)
        return [image, timeline], [(f"ID {track_id}", f"{_count(observed[:, row])} observed pose frames") for row, track_id in enumerate(ids)]

    def render_person_reid(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        result = value["result"]
        details: list[tuple[str, str]] = []
        descriptor = self.store.descriptor(self.references[node])
        if "model_raw_track_ids" in descriptor["provenance"]:
            details.append(("model-inferred IDs before confirmation", str(descriptor["provenance"]["model_raw_track_ids"])))
            details.append(("model embedding artifact", str(descriptor["provenance"]["model_artifact_id"])))
            confirmation = descriptor["provenance"]["confirmation"]
            details.append(("confirmed target IDs", str(confirmation["confirmed_target_ids"])))
            details.append(("excluded raw tracks", str(confirmation["extra_tracks"])))
        def plot(ax: Any) -> None:
            if result is None:
                ax.text(.5, .5, "No valid person embeddings", ha="center", va="center", transform=ax.transAxes)
                return
            valid: np.ndarray = _array(result["track_valid"]).astype(bool)
            embeddings = _array(result["track_embedding"])[valid]
            local = _array(result["local_track_ids"])
            global_ids = _array(result["slot_global_ids"])
            names = [f"{value['camera_ids'][v]}:{local[v, p]} → {global_ids[v, p]}"
                     for v, p in np.argwhere(valid)]
            if not len(embeddings):
                ax.text(.5, .5, "No valid person embeddings", ha="center", va="center", transform=ax.transAxes)
                return
            similarity = embeddings @ embeddings.T
            image = ax.imshow(similarity, vmin=-1, vmax=1, cmap="coolwarm")
            ax.figure.colorbar(image, ax=ax, label="cosine similarity")
            for row in range(len(names)):
                for column in range(len(names)):
                    ax.text(column, row, f"{similarity[row, column]:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if similarity[row, column] > .65 or similarity[row, column] < -.65 else "black")
            ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
                   xticklabels=names, yticklabels=names, title="Raw model cosine similarity (IDs shown are current assignments)")
            ax.tick_params(axis="x", labelrotation=70)
            raw = {"embedding_artifact_id": descriptor["provenance"].get("model_artifact_id", self.references[node].artifact_id),
                   "cosine_threshold": float(result["cosine_threshold"]), "labels": names,
                   "similarity": similarity.tolist(),
                   "assignment_source": descriptor["provenance"].get("origin", "component")}
            (self.output / "reid_raw_cosine.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2))
            with (self.output / "reid_raw_cosine.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["camera:local → current_global", *names])
                writer.writerows([name, *[f"{number:.8f}" for number in row]] for name, row in zip(names, similarity, strict=True))
            self.raw_cosine_written = True
        image = _save_plot(self.output / "images" / "person_reid_similarity.png", plot, figsize=(10, 8))
        if result is not None:
            local = _array(result["local_track_ids"])
            global_ids = _array(result["slot_global_ids"])
            valid: np.ndarray = _array(result["track_valid"]).astype(bool)
            details += [(str(value["camera_ids"][v]), ", ".join(
                f"local {int(local[v, p])} → global {int(global_ids[v, p])}" if int(global_ids[v, p]) >= 0
                else f"local {int(local[v, p])} → excluded from target reconstruction"
                for p in np.flatnonzero(valid[v])))
                        for v in range(len(value["camera_ids"]))]
            details.append(("cosine threshold", str(result["cosine_threshold"])))
        images = [image]
        required = ["court_calibration", "court_side", *(f"person_tracking/{camera_id}" for camera_id in self.camera_ids)]
        if all(parent in self.references and not self.stale_dependencies(parent) for parent in required):
            ground_image, comparisons = self.ground_distance_matrix()
            images.append(ground_image)
            details.append(("ground-plane comparison", "bbox footpoints under ball-confirmed sides; diagnostic, not identity ground truth"))
            details += [(f"{a} ↔ {b}", f"median {distance:.2f} m over {frames} shared observed frames")
                        for a, b, distance, frames in comparisons if distance < 3.5]
        else:
            details.append(("ground-plane comparison", "unavailable: calibration, side, or tracking artifact is missing/stale"))
        return images, details

    def ground_distance_matrix(self) -> tuple[str, list[tuple[str, str, float, int]]]:
        """Compare camera-local bbox footpoints on the ball-confirmed court plane."""
        calibration, _ = self.payload("court_calibration")
        side, _ = self.payload("court_side")
        turns = dict(zip(side["camera_ids"], side["view_half_turns"], strict=True))
        tracks: list[tuple[str, np.ndarray, np.ndarray]] = []
        for view in calibration["calibration"]["views"]:
            camera = view["camera"]
            camera_id = camera["camera_id"]
            rotation: np.ndarray = _array(camera["rotation"]).astype(np.float64)
            if turns[camera_id]:
                rotation = rotation @ np.diag([-1., -1., 1.])
            translation: np.ndarray = _array(camera["translation"]).astype(np.float64)
            center = -rotation.T @ translation
            inverse_intrinsic = np.linalg.inv(_array(camera["intrinsic"]).astype(np.float64))
            tracking, _ = self.payload(f"person_tracking/{camera_id}")
            boxes, observed = _array(tracking["boxes_xyxy"]), _array(tracking["observed"])
            for row, track_id in enumerate(_array(tracking["track_ids"])):
                box = boxes[row]
                uv1 = np.column_stack(((box[:, 0] + box[:, 2]) / 2, box[:, 3], np.ones(len(box))))
                rays = (uv1 @ inverse_intrinsic.T) @ rotation
                with np.errstate(divide="ignore", invalid="ignore"):
                    scale = -center[2] / rays[:, 2]
                ground = center + scale[:, None] * rays
                valid = observed[row] & np.isfinite(ground).all(-1)
                tracks.append((f"{camera_id}:{int(track_id)}", ground[:, :2], valid))
        count = len(tracks)
        distances: np.ndarray = np.full((count, count), np.nan, np.float64)
        comparisons: list[tuple[str, str, float, int]] = []
        for a in range(count):
            for b in range(a + 1, count):
                if tracks[a][0].split(":")[0] == tracks[b][0].split(":")[0]:
                    continue
                shared = tracks[a][2] & tracks[b][2]
                frames = int(shared.sum())
                if frames < 30:
                    continue
                distance = float(np.median(np.linalg.norm(tracks[a][1][shared] - tracks[b][1][shared], axis=-1)))
                distances[a, b] = distances[b, a] = distance
                comparisons.append((tracks[a][0], tracks[b][0], distance, frames))
        labels = [item[0] for item in tracks]
        def plot(ax: Any) -> None:
            palette = plt.get_cmap("viridis").copy()
            palette.set_bad("#e7eaeb")
            image = ax.imshow(np.ma.masked_invalid(distances), vmin=0, vmax=10, cmap=palette)
            ax.figure.colorbar(image, ax=ax, label="median footpoint distance (m); color clipped at 10")
            ax.set(xticks=np.arange(count), yticks=np.arange(count), xticklabels=labels,
                   yticklabels=labels, title="Cross-camera court-plane footpoint distance")
            ax.tick_params(axis="x", labelrotation=70)
            for a in range(count):
                for b in range(count):
                    if np.isfinite(distances[a, b]):
                        ax.text(b, a, f"{distances[a, b]:.1f}", ha="center", va="center",
                                color="white" if distances[a, b] > 5 else "black", fontsize=8)
        image = _save_plot(self.output / "images" / "person_reid_ground_distance.png", plot, figsize=(10, 8))
        return image, comparisons

    def render_court_side(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        turns = value["view_half_turns"]
        if turns is None:
            raise ValueError("Court-side artifact has no confirmed/model half-turns")
        cells: list[np.ndarray] = []
        for camera_id, turn in zip(value["camera_ids"], turns, strict=True):
            image = cv2.resize(self.frame(camera_id, 0), (720, 405), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (719, 78), (25, 33, 39), -1)
            _text(image, f"{camera_id}: {'180 deg' if turn else '0 deg'}", (20, 49), (70, 220, 255))
            cells.append(image)
        image = _save_image(self.output / "images" / "court_side_cameras.png", np.hstack(cells))
        details = [(camera_id, "180°" if turn else "0°") for camera_id, turn in zip(value["camera_ids"], turns, strict=True)]
        details.append(("reference", value["reference_camera"]))
        details.append(("model logits", "not present (explicit import)" if value["side_logits"] is None else str(_array(value["side_logits"]).tolist())))
        return [image], details

    def render_camera_alignment(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        geometry = value["geometry"]
        if geometry is None:
            raise ValueError("Camera alignment artifact has no aligned geometry")
        def plot(ax: Any) -> None:
            _court_axes(ax)
            for item, turn in zip(geometry["cameras"], geometry["view_half_turns"], strict=True):
                rotation = _array(item["rotation"])
                translation = _array(item["translation"])
                center = -rotation.T @ translation
                direction = rotation.T @ np.array([0., 0., 1.])
                ax.scatter(center[0], center[1], s=90)
                ax.arrow(center[0], center[1], direction[0] * 3, direction[1] * 3,
                         head_width=.3, length_includes_head=True)
                ax.annotate(f"{item['camera_id']} ({'180°' if turn else '0°'})", center[:2], xytext=(4, 5), textcoords="offset points")
            ax.set_title("Aligned camera centers and view directions")
        image = _save_plot(self.output / "images" / "camera_alignment_topdown.png", plot, figsize=(7, 8))
        return [image], [("reference", geometry["reference_camera"]),
                         ("aligned cameras", ", ".join(geometry["camera_ids"]))]

    def render_player_triangulation(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        skeleton = value["skeleton"]
        if skeleton is None:
            raise ValueError("Player triangulation artifact has no skeleton output")
        positions, valid = _array(skeleton["positions"]), _array(skeleton["valid"])
        def plot(ax: Any) -> None:
            _court_axes(ax)
            for player in range(len(positions)):
                joint_mask = valid[player]
                count = joint_mask.sum(-1)
                xy = (positions[player, ..., :2] * joint_mask[..., None]).sum(1)
                xy = np.divide(xy, count[:, None], out=np.zeros_like(xy), where=count[:, None] > 0)
                ax.scatter(xy[count > 0, 0], xy[count > 0, 1], s=5, alpha=.5,
                           color=COLORS_MPL[player % len(COLORS_MPL)], label=f"player {player}")
            ax.set_title("Triangulated player joint centers")
            if len(positions):
                ax.legend()
        image = _save_plot(self.output / "images" / "player_triangulation_topdown.png", plot, figsize=(7, 8))
        details = [(f"player {player}", f"{_count(valid[player])} valid joints across {_count(valid[player].any(-1))} frames")
                   for player in range(len(positions))]
        return [image], details

    def render_ball_triangulation(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        ball = value["ball"]
        if ball is None:
            raise ValueError("Ball triangulation artifact has no trajectory")
        trajectory = ball["trajectory"]
        positions, valid = _array(trajectory["positions"]), _array(trajectory["valid"])
        def topdown(ax: Any) -> None:
            _court_axes(ax)
            if valid.any():
                dots = ax.scatter(positions[valid, 0], positions[valid, 1], c=positions[valid, 2],
                                  s=12, cmap="viridis", vmin=0, vmax=max(4., float(positions[valid, 2].max())))
                ax.figure.colorbar(dots, ax=ax, label="height (m)")
            ax.set_title("Triangulated ball positions")
        def height(ax: Any) -> None:
            ax.plot(np.flatnonzero(valid), positions[valid, 2], lw=1, color=COLORS_MPL[0])
            ax.set(xlabel="source frame", ylabel="height (m)", title="Ball height on valid frames")
            ax.grid(alpha=.2)
        images = [_save_plot(self.output / "images" / "ball_triangulation_topdown.png", topdown, figsize=(7, 8)),
                  _save_plot(self.output / "images" / "ball_triangulation_height.png", height)]
        return images, [("status", ball["status"]), ("valid frames", f"{_count(valid)}/{len(valid)}"),
                        ("inlier camera observations", str(_count(trajectory["inliers"])))]

    def render_ball_smoothing(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        ball = value["ball"]
        raw, _ = self.payload("ball_triangulation")
        if ball is None or raw["ball"] is None:
            raise ValueError("Ball smoothing review requires both raw and smoothed trajectories")
        smoothed = ball["trajectory"]
        baseline = raw["ball"]["trajectory"]
        positions, valid = _array(smoothed["positions"]), _array(smoothed["valid"])
        raw_positions, raw_valid = _array(baseline["positions"]), _array(baseline["valid"])
        if not np.array_equal(valid, raw_valid):
            raise ValueError("Ball smoothing changed the triangulation validity mask")
        delta = np.linalg.norm(positions - raw_positions, axis=-1)
        def topdown(ax: Any) -> None:
            _court_axes(ax)
            ax.scatter(raw_positions[valid, 0], raw_positions[valid, 1], s=6, alpha=.3, color="#8b9aa6", label="raw")
            path = np.where(valid[:, None], positions, np.nan)
            ax.plot(path[:, 0], path[:, 1], lw=.8, color=COLORS_MPL[0], label="smoothed")
            ax.set_title("Ball 3D: raw vs smoothed")
            ax.legend()
        def timeline(ax: Any) -> None:
            frames = np.arange(len(valid))
            ax.plot(frames, np.where(valid, raw_positions[:, 2], np.nan), lw=.75, alpha=.5, color="#8b9aa6", label="raw height")
            ax.plot(frames, np.where(valid, positions[:, 2], np.nan), lw=1, color=COLORS_MPL[0], label="smoothed height")
            ax.set(xlabel="source frame", ylabel="height (m)", title="Ball height on supported frames")
            ax.grid(alpha=.2)
            ax.legend()
        def displacement(ax: Any) -> None:
            ax.plot(np.arange(len(valid)), np.where(valid, delta, np.nan), lw=.8, color=COLORS_MPL[0])
            ax.set(xlabel="source frame", ylabel="change from raw (m)", title="Temporal smoothing displacement")
            ax.grid(alpha=.2)
        images = [_save_plot(self.output / "images" / "ball_smoothing_topdown.png", topdown, figsize=(7, 8)),
                  _save_plot(self.output / "images" / "ball_smoothing_height.png", timeline),
                  _save_plot(self.output / "images" / "ball_smoothing_displacement.png", displacement)]
        descriptor = self.store.descriptor(self.references[node])
        method = descriptor["identity"]["settings"]["config"]["method"]
        return images, [("method", str(method)), ("valid frames", f"{_count(valid)}/{len(valid)}"),
                        ("3D displacement p50 / p95 (m)", f"{np.median(delta[valid]):.3f} / {np.quantile(delta[valid], .95):.3f}" if valid.any() else "none")]

    def render_body_view_selection(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        selections = value["selections"]
        def plot(ax: Any) -> None:
            for row, selection in enumerate(selections):
                for request in selection["requests"]:
                    frames = _array(request["source_frames"])
                    ax.plot(frames, np.full(len(frames), row), marker="|", markersize=5,
                            color=COLORS_MPL[row % len(COLORS_MPL)])
            ax.set(xlabel="source frame", ylabel="selected player / camera", yticks=np.arange(len(selections)),
                   yticklabels=[f"{item['person_id']} / {item['camera_id']}" for item in selections],
                   title="GVHMR input view and frame segments")
            ax.grid(axis="x", alpha=.2)
        image = _save_plot(self.output / "images" / "body_view_selection_timeline.png", plot)
        return [image], [(f"player {item['person_id']}", f"{item['camera_id']}; {len(item['requests'])} segments; {sum(item['observed_samples'])} observed samples")
                         for item in selections]

    def render_gvhmr(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        bodies = value["bodies"]
        def plot(ax: Any) -> None:
            for row, body in enumerate(bodies):
                for segment in body["segments"]:
                    frames = _array(segment["source_frames"])
                    translation = _array(segment["parameters"]["transl"])
                    ax.plot(frames, translation[:, 2], lw=1,
                            color=COLORS_MPL[row % len(COLORS_MPL)], label=f"player {body['person_id']}" if frames[0] == _array(body["segments"][0]["source_frames"])[0] else None)
            ax.set(xlabel="source frame", ylabel="in-camera translation z (m)", title="GVHMR recovered body segments")
            if bodies:
                ax.legend()
            ax.grid(alpha=.2)
        image = _save_plot(self.output / "images" / "gvhmr_segments.png", plot)
        def pose_plot(ax: Any) -> None:
            for row, body in enumerate(bodies):
                for segment in body["segments"]:
                    frames = _array(segment["source_frames"])
                    parameters = segment["parameters"]
                    pose = np.linalg.norm(_array(parameters["body_pose"]), axis=-1)
                    orient = np.linalg.norm(_array(parameters["global_orient"]), axis=-1)
                    color = COLORS_MPL[row % len(COLORS_MPL)]
                    ax.plot(frames, pose, lw=1, color=color, label=f"player {body['person_id']} pose")
                    ax.plot(frames, orient, lw=1, linestyle="--", color=color, label=f"player {body['person_id']} orient")
            ax.set(xlabel="source frame", ylabel="rotation-vector norm (rad)", title="GVHMR body pose and global orientation")
            if bodies:
                handles, labels = ax.get_legend_handles_labels()
                unique = dict(zip(labels, handles, strict=True))
                ax.legend(unique.values(), unique.keys(), ncol=2)
            ax.grid(alpha=.2)
        pose_image = _save_plot(self.output / "images" / "gvhmr_pose_norms.png", pose_plot)
        details = [(f"player {body['person_id']}", f"{body['camera_id']}; {len(body['segments'])} recovered segments") for body in bodies]
        return [image, pose_image], details

    def render_body_placement(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        players = value["players"]
        if players is None:
            raise ValueError("Body placement artifact has no player output")
        position, yaw = _array(players["position"]), _array(players["yaw"])
        root_valid, smpl_valid = _array(players["root_valid"]), _array(players["smpl_valid"])
        def plot(ax: Any) -> None:
            _court_axes(ax)
            for player in range(len(position)):
                valid = root_valid[player]
                xy = position[player, valid, :2]
                ax.plot(xy[:, 0], xy[:, 1], lw=1, color=COLORS_MPL[player % len(COLORS_MPL)], label=f"player {player}")
                frames = np.flatnonzero(valid)[::max(1, _count(valid) // 20)]
                ax.quiver(position[player, frames, 0], position[player, frames, 1],
                          np.sin(yaw[player, frames]), np.cos(yaw[player, frames]),
                          color=COLORS_MPL[player % len(COLORS_MPL)], scale=24)
            ax.set_title("Placed bodies: root positions and yaw")
            if len(position):
                ax.legend()
        image = _save_plot(self.output / "images" / "body_placement_topdown.png", plot, figsize=(7, 8))
        images = [image]
        if players["vertices_local"] is not None and smpl_valid.any():
            vertices = _array(players["vertices_local"])
            fig = plt.figure(figsize=(8, 7), constrained_layout=True)
            axis = fig.add_subplot(111, projection="3d")
            try:
                for player in range(len(position)):
                    valid_frames = np.flatnonzero(smpl_valid[player])
                    if not len(valid_frames):
                        continue
                    frame = int(valid_frames[len(valid_frames) // 2])
                    sample = vertices[player, frame, ::max(1, vertices.shape[2] // 350)]
                    axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2,
                                 color=COLORS_MPL[player % len(COLORS_MPL)], label=f"player {player}, frame {frame}")
                axis.set(xlabel="local x (m)", ylabel="local y (m)", zlabel="local z (m)",
                         title="Body placement local SMPL vertex samples")
                axis.legend()
                destination = self.output / "images" / "body_placement_mesh_samples.png"
                fig.savefig(destination, dpi=140)
                images.append(destination.name)
            finally:
                plt.close(fig)
        details = [(f"player {player}", f"{_count(root_valid[player])} valid roots; {_count(smpl_valid[player])} valid SMPL frames")
                   for player in range(len(position))]
        details.append(("local mesh", "present" if players["vertices_local"] is not None else "absent"))
        return images, details

    def render_scene_assembly(self, node: str, camera: str, value: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
        positions = _array(value["player_position"])
        player_valid = _array(value["player_valid"])
        ball = _array(value["ball_3d"])
        ball_valid = _array(value["ball_3d_valid"])
        def plot(ax: Any) -> None:
            _court_axes(ax)
            for player in range(len(positions)):
                valid = player_valid[player]
                ax.plot(positions[player, valid, 0], positions[player, valid, 1],
                        lw=1.3, color=COLORS_MPL[player % len(COLORS_MPL)], label=f"player {player}")
            ax.scatter(ball[ball_valid, 0], ball[ball_valid, 1], s=4, color="#d0aa25", label="ball")
            ax.set_title("Assembled court scene with validity masks")
            ax.legend()
        image = _save_plot(self.output / "images" / "scene_assembly_topdown.png", plot, figsize=(7, 8))
        return [image], [("scene status", str(value["metadata"]["status"])),
                         ("valid ball frames", f"{_count(ball_valid)}/{self.frame_count}"),
                         ("valid player frames", str(_count(player_valid)))]

    def build(self) -> Path:
        nodes = [f"{name}/{camera}" if name in CAMERA_COMPONENTS else name
                 for name in ORDER for camera in (self.camera_ids if name in CAMERA_COMPONENTS else ("",))]
        # Global node names do not contain a camera suffix.
        nodes = [node[:-1] if node.endswith("/") else node for node in nodes]
        cards: list[str] = []
        manifest: dict[str, Any] = {"source": self.source["clip_id"], "scene_index": str(self.index_path), "components": {}}
        for node in nodes:
            if node not in self.references:
                cards.append(f'<section class="missing"><h2>{html.escape(node)}</h2><p>未生成（scene.json に成果物なし）</p></section>')
                manifest["components"][node] = {"status": "missing"}
                continue
            stale = self.stale_dependencies(node)
            if stale:
                reasons = "; ".join(stale)
                cards.append(f'<section class="missing"><h2>{html.escape(node)}</h2><p>旧入力に依存する成果物。現版の可視化から除外: {html.escape(reasons)}</p></section>')
                manifest["components"][node] = {"status": "stale", "reasons": stale,
                                                 "artifact_id": self.references[node].artifact_id}
                continue
            images, details = self.render(node)
            if not images:
                raise ValueError(f"No visualization generated for {node}")
            rows = "".join(f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>" for key, value in details)
            figures = "".join(f'<a href="images/{html.escape(name)}"><img src="images/{html.escape(name)}" alt="{html.escape(node)} visualization"></a>' for name in images)
            movie = self.movies.get(node)
            playback = "" if movie is None else f'<video controls preload="metadata" src="{html.escape(movie)}"></video>'
            downloads = '<p><a href="reid_raw_cosine.json">生のcosine値 JSON</a> · <a href="reid_raw_cosine.csv">CSV</a></p>' if node == "person_reid" and self.raw_cosine_written else ""
            cards.append(f'<section id="{html.escape(node.replace("/", "-"))}"><h2>{html.escape(node)}</h2><div class="figures">{figures}</div>{playback}{downloads}<table>{rows}</table></section>')
            manifest["components"][node] = {"status": "rendered", "images": [f"images/{name}" for name in images],
                                             "video": movie, "details": dict(details), "artifact_id": self.references[node].artifact_id}
        (self.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        navigation = "".join(f'<a href="#{html.escape(node.replace("/", "-"))}">{html.escape(node)}</a>' for node in nodes if node in self.references)
        page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Component review — {html.escape(self.source['clip_id'])}</title>
<style>body{{font:16px system-ui,sans-serif;color:#182631;background:#f2f5f5;margin:0}}header{{background:#19333d;color:white;padding:24px 3vw}}h1{{margin:0 0 6px}}nav{{display:flex;gap:8px;flex-wrap:wrap;padding:14px 3vw;background:white;position:sticky;top:0;z-index:2;box-shadow:0 2px 8px #0002}}nav a{{color:#245d70;text-decoration:none;font-size:13px;padding:4px 8px;border-radius:8px;background:#e8f1f2}}main{{padding:20px 3vw}}section{{background:white;border-radius:12px;margin:0 0 24px;padding:20px;box-shadow:0 2px 10px #0001}}section.missing{{opacity:.58}}h2{{margin:0 0 16px;font-size:22px}}.figures{{display:flex;gap:12px;overflow-x:auto;align-items:flex-start}}.figures a{{flex:0 0 auto;max-width:100%}}img{{width:min(100%,720px);max-height:620px;object-fit:contain;border:1px solid #d7e1e4;border-radius:8px}}video{{display:block;width:min(100%,1280px);margin-top:16px;border-radius:8px}}table{{border-collapse:collapse;margin-top:14px;max-width:100%}}th,td{{border-bottom:1px solid #e3e9eb;padding:6px 14px 6px 0;text-align:left;vertical-align:top}}th{{color:#536a72;white-space:nowrap}}</style>
<header><h1>Component review</h1><div>{html.escape(self.source['clip_id'])} · {len(self.references)} published artifacts · {self.frame_count} frames</div><small>各panelは保存済みcomponent出力から生成。画像をクリックすると原寸で開きます。未生成componentは明示します。</small></header>
<nav>{navigation}</nav><main>{''.join(cards)}</main></html>'''
        destination = self.output / "index.html"
        destination.write_text(page)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True, help="Structured clip directory")
    parser.add_argument("--output", type=Path, required=True, help="Gallery directory outside the component store")
    parser.add_argument("--videos", action="store_true", help="Also render complete 2D component overlay videos")
    args = parser.parse_args()
    index = args.clip / "annotations" / "tennis_scene" / "scene.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    print(Review(index, args.output, videos=args.videos).build())


if __name__ == "__main__":
    main()

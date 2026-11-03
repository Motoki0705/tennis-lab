"""CPU source overlays and court trajectories, with explicit gaps and frame IDs."""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.evaluation.ball3d.annotations import Ball2DAnnotations
from src.tennis_scene.evaluation.ball3d.contracts import MethodInput
from src.tennis_scene.evaluation.ball3d.runner import MethodRun
from src.tennis_scene.evaluation.ball3d.scoring import project_result
from src.tennis_scene.evaluation.ball3d.uncertainty import (
    ellipse_vertices,
    project_covariance,
)
from src.tennis_scene.visualization.rendering.multiview import sampled_frame_indices
from src.utils.checksum import dual_sha256
from src.utils.video.reader import (
    OpenCVVideoFrameReader,
    RandomAccessVideoReader,
    probe_video_info,
)


def line_segments(points: NDArray[Any], valid: NDArray[np.bool_]) -> NDArray[Any]:
    """Only adjacent valid pairs; singleton observations never acquire a line."""
    if points.ndim != 2 or valid.shape != (len(points),) or valid.dtype != bool:
        raise ValueError("Trajectory points and explicit mask must share a frame axis")
    return np.stack((points[:-1], points[1:]), axis=1)[valid[:-1] & valid[1:]]


def method_colors(names: tuple[str, ...]) -> dict[str, tuple[int, int, int]]:
    # BGR colors, deliberately separate from the green annotation cross.
    palette = [(10, 180, 255), (255, 130, 70), (220, 90, 220), (80, 240, 240)]
    colors = {}
    for i, name in enumerate(names):
        if i < len(palette):
            colors[name] = palette[i]
        else:
            hsv = np.array([[[int((i * 137) % 180), 180, 240]]], np.uint8)
            color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            colors[name] = (int(color[0]), int(color[1]), int(color[2]))
    return colors


class BallComparisonRenderer:
    """Three synchronized RGB views plus top/side court trajectories.

    Reuses the shared readers and source frame selection. MultiviewSceneRenderer
    requires SMPL meshes/SceneResult; this ball-only result schema has neither.
    """

    def __init__(
        self,
        inputs: MethodInput,
        runs: Mapping[str, MethodRun],
        videos: Mapping[str, Path],
        *,
        annotations: Ball2DAnnotations | None = None,
        width: int = 1200,
    ) -> None:
        if (
            len(inputs.views) != 3
            or set(videos) != set(inputs.camera_ids)
            or not runs
            or width < 640
            or width % 4
        ):
            raise ValueError(
                "Render three explicitly bound cameras, at least one method and width divisible by four"
            )
        if any(
            run.held_out_camera is not None
            or run.source_camera_ids != inputs.camera_ids
            for run in runs.values()
        ):
            raise ValueError(
                "Comparison rendering requires full-view results on the source camera axis"
            )
        for camera_id, path in videos.items():
            info = probe_video_info(path)
            if (
                (info.width, info.height) != inputs.source_size_wh
                or info.frame_count != inputs.num_frames
                or not math.isclose(info.fps, inputs.fps, abs_tol=1e-3, rel_tol=0)
            ):
                raise ValueError(f"Render video camera/frame/fps mismatch: {camera_id}")
        if annotations is not None:
            annotations.validate_for_inputs(inputs)
        self.inputs, self.runs, self.annotations = inputs, runs, annotations
        self.readers = {
            c: RandomAccessVideoReader(videos[c]) for c in inputs.camera_ids
        }
        self.colors = method_colors(tuple(runs))
        self.projections = {
            (name, camera): project_result(inputs, run.result, camera)
            for name, run in runs.items()
            for camera in inputs.camera_ids
        }
        self.projected_covariances = {
            (name, camera): project_covariance(inputs, run.result, camera)[0]
            for name, run in runs.items() if run.result.covariance is not None
            for camera in inputs.camera_ids
        }
        self.width, self.left_width = width, width // 2
        self.camera_height = round(
            self.left_width * inputs.source_size_wh[1] / inputs.source_size_wh[0]
        )
        self.header = 92 + 22 * ((len(runs) + 2) // 3)
        self.height = self.header + self.camera_height * 3
        self.height += self.height % 2
        all_points = np.concatenate(
            [run.result.positions[run.result.valid] for run in runs.values()]
        )
        self.lower = (
            np.minimum(np.min(all_points, axis=0), [-6.0, -12.0, 0.0]) - 1
            if len(all_points)
            else np.array([-7.0, -13.0, -1.0])
        )
        self.upper = (
            np.maximum(np.max(all_points, axis=0), [6.0, 12.0, 4.0]) + 1
            if len(all_points)
            else np.array([7.0, 13.0, 5.0])
        )

    def _trajectory(
        self,
        image: NDArray[np.uint8],
        frame: int,
        axes: tuple[int, int],
        rect: tuple[int, int, int, int],
        title: str,
    ) -> None:
        x, y, width, height = rect
        cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (70, 75, 80), 1)
        cv2.putText(
            image,
            title,
            (x + 12, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        def screen(points: NDArray[Any]) -> NDArray[np.int32]:
            xy = (points[..., list(axes)] - self.lower[list(axes)]) / (
                self.upper[list(axes)] - self.lower[list(axes)]
            )
            return np.rint(
                xy * [width - 40, -(height - 60)] + [x + 20, y + height - 20]
            ).astype(np.int32)

        if axes == (0, 1):
            court = np.array(
                [
                    [-5.485, -11.885, 0],
                    [5.485, -11.885, 0],
                    [5.485, 11.885, 0],
                    [-5.485, 11.885, 0],
                ]
            )
            cv2.polylines(image, [screen(court)], True, (115, 125, 115), 1, cv2.LINE_AA)
        else:
            cv2.line(
                image,
                tuple(screen(np.array([0, -11.885, 0]))),
                tuple(screen(np.array([0, 11.885, 0]))),
                (115, 125, 115),
                1,
            )
        start = max(0, frame - round(self.inputs.fps * 2))
        for name, run in self.runs.items():
            result = run.result
            segments = line_segments(
                result.positions[start : frame + 1], result.valid[start : frame + 1]
            )
            for pair in segments:
                cv2.line(
                    image,
                    tuple(screen(pair[0])),
                    tuple(screen(pair[1])),
                    self.colors[name],
                    2,
                    cv2.LINE_AA,
                )
            if result.valid[frame]:
                if result.covariance is not None:
                    selected = list(axes)
                    scales = np.array([width-40, -(height-60)]) / (self.upper[selected]-self.lower[selected])
                    covariance = result.covariance[frame][np.ix_(selected, selected)] * scales[:, None] * scales[None, :]
                    vertices = ellipse_vertices(screen(result.positions[frame]).astype(float), covariance)
                    cv2.polylines(image, [np.rint(vertices.clip(-1e6, 1e6)).astype(np.int32)], True,
                        self.colors[name], 1, cv2.LINE_AA)
                cv2.circle(
                    image,
                    tuple(screen(result.positions[frame])),
                    5,
                    self.colors[name],
                    -1,
                    cv2.LINE_AA,
                )

    def render(self, frame: int) -> NDArray[np.uint8]:
        if not 0 <= frame < self.inputs.num_frames:
            raise ValueError("Source frame out of range")
        image: NDArray[np.uint8] = np.full((self.height, self.width, 3), 23, np.uint8)
        text = f"frame {frame} | {frame / self.inputs.fps:.3f} s | "
        text += (
            "user reference (observed + approved estimated)"
            if self.annotations is not None
            else "GT unavailable: projections are predictions"
        )
        cv2.putText(
            image,
            text,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        for i, name in enumerate(self.runs):
            cv2.putText(
                image,
                name,
                (12 + (i % 3) * (self.width // 3), 54 + (i // 3) * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors[name],
                1,
                cv2.LINE_AA,
            )
        if self.annotations is not None:
            cv2.putText(
                image,
                "Reference: green + observed | white diamond estimated | rings: method reprojections",
                (12, self.header - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 230, 220),
                1,
                cv2.LINE_AA,
            )
        scale = self.left_width / self.inputs.source_size_wh[0]
        for i, camera in enumerate(self.inputs.camera_ids):
            bgr = self.readers[camera].read(frame)
            panel = cv2.resize(
                bgr, (self.left_width, self.camera_height), interpolation=cv2.INTER_AREA
            )
            for j, name in enumerate(self.runs):
                uv, valid = self.projections[name, camera]
                if (
                    valid[frame]
                    and np.all(uv[frame] >= 0)
                    and np.all(uv[frame] < self.inputs.source_size_wh)
                ):
                    point = tuple(np.rint(uv[frame] * scale).astype(int))
                    if (name, camera) in self.projected_covariances:
                        covariance = self.projected_covariances[name, camera][frame] * scale**2
                        vertices = ellipse_vertices(uv[frame] * scale, covariance)
                        cv2.polylines(panel, [np.rint(vertices.clip(-1e6, 1e6)).astype(np.int32)], True,
                            self.colors[name], 1, cv2.LINE_AA)
                    cv2.circle(
                        panel, point, 5 + j * 3, self.colors[name], 2, cv2.LINE_AA
                    )
            if (
                self.annotations is not None
                and self.annotations.valid_for_scoring[i, frame]
            ):
                point = tuple(
                    np.rint(self.annotations.uv_px[i, frame] * scale).astype(int)
                )
                observed = self.annotations.observed[i, frame]
                cv2.drawMarker(
                    panel,
                    point,
                    (60, 255, 60) if observed else (255, 255, 255),
                    cv2.MARKER_CROSS if observed else cv2.MARKER_DIAMOND,
                    18,
                    2,
                    cv2.LINE_AA,
                )
            caption = (
                camera
                if self.annotations is None
                else f"{camera} | {self.annotations.reference_kind[i, frame]}"
            )
            cv2.putText(
                panel,
                caption,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y = self.header + i * self.camera_height
            image[y : y + self.camera_height, : self.left_width] = panel
        body_height = self.height - self.header
        self._trajectory(
            image,
            frame,
            (0, 1),
            (
                self.left_width,
                self.header,
                self.width - self.left_width,
                body_height // 2,
            ),
            "Top: court X/Y (metres)",
        )
        self._trajectory(
            image,
            frame,
            (1, 2),
            (
                self.left_width,
                self.header + body_height // 2,
                self.width - self.left_width,
                body_height - body_height // 2,
            ),
            "Side: court Y/Z (metres)",
        )
        return image

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()

    def __enter__(self) -> BallComparisonRenderer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def validate_video(
    path: Path, *, frames: int, fps: float, size: tuple[int, int]
) -> dict[str, Any]:
    start = time.perf_counter()
    info = probe_video_info(path)
    count = 0
    for packet in OpenCVVideoFrameReader(path):
        if packet.original_size != size:
            raise ValueError("Decoded comparison frame size mismatch")
        count += 1
    if (
        count != frames
        or info.frame_count != frames
        or (info.width, info.height) != size
        or not math.isclose(info.fps, fps, abs_tol=1e-3, rel_tol=0)
    ):
        raise ValueError(
            "Comparison video failed full-frame decode/timeline validation"
        )
    return {
        "decoded_frames": count,
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "sha256": dual_sha256(path),
        "decode_seconds": time.perf_counter() - start,
    }


def render_video(
    renderer: BallComparisonRenderer, output: Path, *, step: int = 1
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Comparison video already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = sampled_frame_indices(
        renderer.inputs.num_frames, start=0, stop=renderer.inputs.num_frames, step=step
    )
    fps = renderer.inputs.fps / step
    start = time.perf_counter()
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{renderer.width}x{renderer.height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg input pipe unavailable")
        previews = {
            frames[i]
            for i in (
                0,
                len(frames) // 4,
                len(frames) // 2,
                3 * len(frames) // 4,
                len(frames) - 1,
            )
        }
        for frame in frames:
            image = renderer.render(frame)
            process.stdin.write(image.tobytes())
            if frame in previews and not cv2.imwrite(
                str(output.parent / f"preview-{frame:04d}.png"), image
            ):
                raise RuntimeError("Failed to save comparison preview")
        process.stdin.close()
        error = process.stderr.read().decode() if process.stderr is not None else ""
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.stderr is not None:
            process.stderr.close()
    validation = validate_video(
        output, frames=len(frames), fps=fps, size=(renderer.width, renderer.height)
    )
    return {
        **validation,
        "render_and_encode_seconds": time.perf_counter() - start,
        "source_frames": list(frames),
        "step": step,
    }

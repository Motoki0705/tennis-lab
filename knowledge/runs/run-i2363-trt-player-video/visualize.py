"""Render synchronized camera tracks and available 3D from saved #2363 artifacts."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from analyze import clip_store, progress_path
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.visualization.rendering.multiview import sampled_frame_indices
from src.tennis_scene.visualization.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.rendering.camera_view import CameraController, CameraView3D
from src.utils.video.reader import RandomAccessVideoReader
from src.utils.video.writer import VideoWriter

VIEW_WIDTH, VIEW_HEIGHT = 480, 270
COLORS = [
    (72, 204, 255),
    (235, 190, 65),
    (125, 231, 133),
    (241, 115, 240),
    (95, 145, 255),
]


def camera_labels(
    store: StoreReader, outcome: dict[str, Any]
) -> tuple[str, dict[tuple[str, int], np.ndarray]]:
    labels = {}
    if outcome["status"] == "generated":
        association, _ = store.payload("player_association")
        for camera, tracks, players in zip(
            association["camera_ids"],
            association["local_track_ids"],
            association["player_ids"],
            strict=True,
        ):
            labels.update(
                {
                    (camera, int(track)): ids
                    for track, ids in zip(tracks, players, strict=True)
                    if track >= 0
                }
            )
        return "P", labels

    association = outcome.get("diagnostics", {}).get("association", {})
    if association.get("clustering"):
        for candidate, identity in zip(
            association["candidates"],
            association["clustering"]["identity_of_candidate"],
            strict=True,
        ):
            segment = association["segments"][candidate]
            key = segment["camera"], segment["track_id"]
            timeline = labels.setdefault(
                key, np.full(store.frame_count, -1, dtype=np.int64)
            )
            timeline[segment["start"] : segment["end"]] = identity
    return "cluster ", labels


def label(
    image: np.ndarray,
    text: str,
    at: tuple[int, int],
    color: tuple[int, int, int] = (235, 240, 248),
) -> None:
    cv2.putText(
        image, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.53, (14, 17, 21), 3, cv2.LINE_AA
    )
    cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 1, cv2.LINE_AA)


def draw_camera(
    frame: np.ndarray,
    camera: str,
    tracks: dict[str, Any],
    index: int,
    prefix: str,
    labels: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    scale = np.array([VIEW_WIDTH / frame.shape[1], VIEW_HEIGHT / frame.shape[0]] * 2)
    image: np.ndarray = cv2.resize(
        frame, (VIEW_WIDTH, VIEW_HEIGHT), interpolation=cv2.INTER_AREA
    )
    for row in np.flatnonzero(tracks["observed"][:, index]):
        track = int(tracks["track_ids"][row])
        ids = labels.get((camera, track))
        identity = -1 if ids is None else int(ids[index])
        color = (170, 175, 184) if identity < 0 else COLORS[identity % len(COLORS)]
        x1, y1, x2, y2 = (
            np.rint(tracks["boxes_xyxy"][row, index] * scale).astype(int).tolist()
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"T{track}" if identity < 0 else f"{prefix}{identity} / T{track}"
        label(image, text, (max(0, x1), max(18, y1 - 3)), color)
    label(image, camera, (10, 20))
    return image


def render(
    run: Path,
    record: dict[str, Any],
    outcome: dict[str, Any],
    output: Path,
    frame_step: int,
    renderer: TennisSceneRenderer,
) -> None:
    store = clip_store(run, record)
    if len(store.camera_ids) != 4:
        raise ValueError("This report renders exactly four camera panels")
    frames = sampled_frame_indices(
        store.frame_count, start=0, stop=store.frame_count, step=frame_step
    )
    tracks = {
        camera: store.payload(f"player_selection/{camera}")[0]["tracks"]
        for camera in store.camera_ids
    }
    prefix, labels = camera_labels(store, outcome)
    fps = float(store.source["videos"][0]["fps"])
    scene = None
    if outcome["status"] == "generated":
        scene = load_scene_result(store.index_path)
        figure = Figure(figsize=(VIEW_WIDTH / 100, 2 * VIEW_HEIGHT / 100), dpi=100)
        figure.set_facecolor(renderer.theme.figure_color)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    width = VIEW_WIDTH * (2 if scene is None else 3)
    image: np.ndarray = np.zeros((2 * VIEW_HEIGHT + 52, width, 3), dtype=np.uint8)
    detail = (
        outcome["error"].removeprefix("ReconstructionUnavailable: ").split(" [[", 1)[0]
        if scene is None
        else ""
    )
    pending = output.with_suffix(".partial.mp4")
    try:
        with ExitStack() as resources:
            readers = {
                camera: resources.enter_context(
                    RandomAccessVideoReader(store.videos[camera])
                )
                for camera in store.camera_ids
            }
            writer = resources.enter_context(
                VideoWriter(pending, fps / frame_step, crf=21)
            )
            for index in frames:
                for panel, (camera, reader) in enumerate(readers.items()):
                    row, col = divmod(panel, 2)
                    y, x = row * VIEW_HEIGHT, col * VIEW_WIDTH
                    image[y : y + VIEW_HEIGHT, x : x + VIEW_WIDTH] = draw_camera(
                        reader.read(index),
                        camera,
                        tracks[camera],
                        index,
                        prefix,
                        labels,
                    )
                if scene is not None:
                    assert (
                        scene.player_valid is not None
                        and scene.player_smpl_valid is not None
                    )
                    axis.clear()
                    renderer.render_frame_3d(scene, index, ax=axis)
                    canvas.draw()
                    image[: 2 * VIEW_HEIGHT, 2 * VIEW_WIDTH :] = cv2.cvtColor(
                        np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR
                    )
                    detail = f"3D COCO-17   Root valid: {scene.player_valid[:, index].sum()}/4   SMPL valid: {scene.player_smpl_valid[:, index].sum()}/4"
                image[2 * VIEW_HEIGHT :] = (24, 26, 30)
                label(
                    image,
                    f"{record['clip_id']}    {outcome['status']}    frame {index}/{store.frame_count - 1}    {index / fps:.2f}s",
                    (12, 2 * VIEW_HEIGHT + 19),
                )
                label(image, detail, (12, 2 * VIEW_HEIGHT + 42))
                writer.write_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("clips", nargs="+", help="clip names within video_000")
    parser.add_argument("--frame-step", type=int, default=3)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    progress = json.loads(progress_path(args.run).read_bytes())
    records = {
        entry["record"]["clip_name"]: entry["record"] for entry in progress["targets"]
    }
    assets = Path("ckpt/body_models").resolve()
    renderer = TennisSceneRenderer(
        TennisSceneStyle(
            player_representation="skeleton",
            theme="dark",
            show_minimap=False,
            show_player_trail=False,
            show_bounces=False,
            show_hud=False,
        ),
        smpl_faces_path=assets / "smplh/neutral/model.npz",
        smpl_joint_regressor_path=assets / "gvhmr/smpl_neutral_J_regressor.pt",
        camera=CameraController(CameraView3D(elev=28.0, azim=-65.0, zoom=1.2)),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.clips:
        record = records[name]
        outcome = progress["clips"][record["clip_id"]]
        destination = args.output / f"{name}-{outcome['status']}.mp4"
        render(args.run, record, outcome, destination, args.frame_step, renderer)
        print(destination, flush=True)


if __name__ == "__main__":
    main()

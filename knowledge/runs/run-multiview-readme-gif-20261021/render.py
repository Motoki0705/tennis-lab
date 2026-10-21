"""Render the README demonstration from an existing scene, entirely on CPU.

The reusable compositor lives in src; this file pins this asset's shot and
exports PNGs plus provenance. FFmpeg palette generation is recorded in repro.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.visualization.rendering.multiview import (
    MultiviewSceneRenderer,
    sampled_frame_indices,
)
from src.tennis_scene.visualization.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)
from src.utils.rendering.ball_renderer import BallStyle
from src.utils.rendering.camera_view import CameraController, CameraView3D


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--video", nargs=2, action="append", metavar=("CAMERA", "PATH"), required=True)
    parser.add_argument("--smpl-faces", type=Path, required=True)
    parser.add_argument("--smpl-joint-regressor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    videos = {camera: Path(path).resolve() for camera, path in args.video}
    if len(videos) != len(args.video):
        raise ValueError("Duplicate --video camera ID")
    scene_path = args.scene.resolve()
    scene = load_scene_result(scene_path)
    frames = sampled_frame_indices(scene.num_frames, start=args.start_frame,
                                   stop=scene.num_frames if args.end_frame is None else args.end_frame,
                                   step=args.frame_step)
    style = TennisSceneStyle(
        player_representation="smpl", mesh_alpha=1., theme="dark",
        show_minimap=False, show_hud=False, show_bounces=False,
        show_player_trail=False, trail_length=60,
        ball_style=BallStyle(ball_size=40., trajectory_color="#CCFF00"),
    )
    renderer = TennisSceneRenderer(style, smpl_faces_path=args.smpl_faces.resolve(),
                                  smpl_joint_regressor_path=args.smpl_joint_regressor.resolve(),
                                  camera=CameraController(CameraView3D(elev=28., azim=-65., zoom=1.5)))
    inputs = {"scene": scene_path, "scene_metadata": scene_path.with_suffix(".metadata.json"),
              "smpl_faces": args.smpl_faces.resolve(), "smpl_joint_regressor": args.smpl_joint_regressor.resolve(),
              **videos}
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs": {name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                   for name, path in inputs.items()},
        "source_frames": frames, "source_fps": scene.fps,
        "output_fps": scene.fps / args.frame_step,
        "frame_step": args.frame_step,
        "player_smpl_frames": scene.player_smpl_valid.sum(axis=1).tolist(),
        "ball_3d_frames": int(scene.ball_3d_valid.sum()),
    }
    with MultiviewSceneRenderer(scene, renderer, videos, width=args.width) as view:
        # Never overwrite an old render or any source. Partial runs lack provenance.json.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        for output_index, source_index in enumerate(frames):
            image = cv2.cvtColor(view.render(source_index), cv2.COLOR_RGB2BGR)
            target = args.output_dir / f"{output_index:05d}.png"
            if not cv2.imwrite(str(target), image):
                raise OSError(f"Could not write {target}")
            if output_index % 10 == 0:
                print(f"{output_index + 1}/{len(frames)} source={source_index} elapsed={time.monotonic() - started:.1f}s", flush=True)
        provenance.update(width=view.width, height=view.height, render_seconds=time.monotonic() - started)
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

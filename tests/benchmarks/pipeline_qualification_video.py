"""Full-length synchronized three-view observations and final court-plane scene."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import file_identity
from src.tennis_scene.schema import SceneResult, validate_scene_result_arrays


def render(scene: SceneResult, source: ClipSource, output: Path) -> dict[str, Any]:
    validate_scene_result_arrays(scene)
    if output.exists():
        raise FileExistsError(output)
    if len(source.videos) != 3 or scene.num_frames != source.num_frames \
            or not np.isclose(scene.fps, source.fps) or scene.court_kp.shape[0] != 3:
        raise ValueError('Video requires three complete, synchronized scene cameras')
    assert scene.human_kp_2d is not None and scene.human_kp_vis is not None
    assert scene.ball_vis is not None and scene.ball_uv is not None and scene.ball_3d is not None
    assert scene.ball_3d_valid is not None and scene.player_valid is not None and scene.player_track_ids is not None
    captures = [cv2.VideoCapture(str(video.path)) for video in source.videos]
    if not all(c.isOpened() for c in captures):
        for capture in captures:
            capture.release()
        raise ValueError('Cannot open all source cameras')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', '1280x720', '-r', str(scene.fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-threads', '1',
               '-movflags', '+faststart', str(output)]
    colors = ((70, 230, 100), (255, 150, 70))

    def court_pixel(x: float, y: float) -> tuple[int, int]:
        return int(320 + x * 13), int(185 - y * 12)

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in range(scene.num_frames):
            panels = []
            for view, capture in enumerate(captures):
                ok, image = capture.read()
                if not ok or image.shape[:2] != (scene.height, scene.width):
                    raise ValueError(f'Source camera {view} ended/changed at frame {frame}')
                panel = cv2.resize(image, (640, 360))
                for player in range(len(scene.player_track_ids)):
                    color = colors[player % len(colors)]
                    points = scene.human_kp_2d[player, view, frame] * [640, 360]
                    visible = scene.human_kp_vis[player, view, frame] >= .3
                    for point in points[visible]:
                        cv2.circle(panel, tuple(np.rint(point).astype(int)), 2, color, -1)
                    if visible.any():
                        at = tuple(np.rint(points[visible].mean(0)).astype(int))
                        cv2.putText(panel, f'ID {scene.player_track_ids[player]}', at,
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                if scene.ball_vis[view, frame]:
                    point = tuple(np.rint(scene.ball_uv[view, frame] * [640, 360]).astype(int))
                    cv2.circle(panel, point, 4, (0, 255, 255), 1)
                cv2.rectangle(panel, (0, 0), (640, 25), (15, 15, 15), -1)
                cv2.putText(panel, f'{source.camera_ids[view]} | frame {frame}/{scene.num_frames - 1} | {frame / scene.fps:.2f}s',
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                panels.append(panel)
            court: np.ndarray = np.full((360, 640, 3), 28, np.uint8)
            for x in (-5.485, -4.115, 4.115, 5.485):
                cv2.line(court, court_pixel(x, -11.885), court_pixel(x, 11.885), (180, 180, 180))
            for y in (-11.885, -6.4, 0., 6.4, 11.885):
                cv2.line(court, court_pixel(-5.485, y), court_pixel(5.485, y), (180, 180, 180))
            for player, identity in enumerate(scene.player_track_ids):
                valid = scene.player_valid[player, frame]
                color = colors[player % len(colors)]
                if valid:
                    x, y, _ = scene.player_position[player, frame]
                    cv2.circle(court, court_pixel(float(x), float(y)), 5, color, -1)
                cv2.putText(court, f'ID {identity}: root {"valid" if valid else "missing"}', (10, 55 + 24 * player),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            if scene.ball_3d_valid[frame]:
                x, y, z = scene.ball_3d[frame]
                cv2.circle(court, court_pixel(float(x), float(y)), 4, (0, 255, 255), -1)
                ball_text = f'Ball z={z:.2f}m'
            else:
                ball_text = 'Ball 3D missing'
            cv2.putText(court, 'Final scene: court-plane view (valid roots/ball only)', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            cv2.putText(court, ball_text, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            canvas = np.concatenate((np.concatenate(panels[:2], axis=1),
                                     np.concatenate((panels[2], court), axis=1)), axis=0)
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait(timeout=60) != 0:
            raise RuntimeError('ffmpeg encoding failed')
    except BaseException:
        process.kill()
        process.wait(timeout=10)
        raise
    finally:
        for capture in captures:
            capture.release()
    read = cv2.VideoCapture(str(output))
    count = 0
    try:
        if not read.isOpened() or not np.isclose(read.get(cv2.CAP_PROP_FPS), scene.fps, atol=.01):
            raise ValueError('Encoded video FPS differs')
        while True:
            ok, frame_image = read.read()
            if not ok:
                break
            if frame_image.shape != (720, 1280, 3):
                raise ValueError('Encoded video dimensions differ')
            count += 1
    finally:
        read.release()
    if count != scene.num_frames:
        raise ValueError('Encoded video lost source frames')
    return {'file': file_identity(output), 'frames_written': scene.num_frames, 'frames_read': count,
            'fps': scene.fps, 'size': [1280, 720], 'camera_ids': list(source.camera_ids),
            'view': 'three synchronized 2D observations + final valid 3D roots/ball on court plane',
            'human_labels_used': False}

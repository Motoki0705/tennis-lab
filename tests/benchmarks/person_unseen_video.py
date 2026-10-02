"""Full-length three-camera prediction view, without annotation or tuning."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import file_identity


def render(source: ClipSource, arrays: dict[str, np.ndarray], output: Path, status: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if source.camera_ids != ('cam0', 'cam1', 'cam2'):
        raise ValueError('Exactly three cameras are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    captures = [cv2.VideoCapture(str(v.path)) for v in source.videos]
    if not all(c.isOpened() for c in captures):
        for capture in captures:
            capture.release()
        raise ValueError('Cannot open all three cameras')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', '1920x400', '-r', str(source.fps), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
               '-threads', '1', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    colors = ((70, 230, 100), (255, 150, 70), (220, 90, 230), (60, 210, 240))
    assert process.stdin is not None
    try:
        for frame in range(source.num_frames):
            panels = []
            for camera, capture in zip(source.camera_ids, captures, strict=True):
                ok, image = capture.read()
                if not ok or image.shape[:2] != (source.size[1], source.size[0]):
                    raise ValueError(f'Incomplete source {camera} frame {frame}')
                panel = cv2.resize(image, (640, 360))
                observed = arrays[f'{camera}_observed']
                for row in np.flatnonzero(observed[:, frame]):
                    identity = int(arrays[f'{camera}_ids'][row, frame])
                    selected = bool(arrays[f'{camera}_selected'][row, frame])
                    color = colors[identity % len(colors)] if identity >= 0 else (0, 220, 220) if selected else (160, 160, 160)
                    box = np.rint(arrays[f'{camera}_boxes'][row, frame] * [640/source.size[0], 360/source.size[1], 640/source.size[0], 360/source.size[1]]).astype(int)
                    cv2.rectangle(panel, tuple(box[:2]), tuple(box[2:]), color, 2 if selected else 1)
                    label = f'r{arrays[f"{camera}_track_ids"][row]} / p{identity}' if identity >= 0 else f'r{arrays[f"{camera}_track_ids"][row]}'
                    cv2.putText(panel, label, (int(box[0]), max(16, int(box[1]))),
                                cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
                cv2.rectangle(panel, (0, 0), (640, 25), (15, 15, 15), -1)
                cv2.putText(panel, f'{camera} | frame {frame}/{source.num_frames-1} | {frame/source.fps:.2f}s',
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1)
                panels.append(panel)
            footer: np.ndarray = np.full((40, 1920, 3), 20, np.uint8)
            cv2.putText(footer, f'{source.clip_id} | association: {status} | gray: raw  yellow: selected/unassigned  color: player ID | no GT',
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
            canvas = np.concatenate((np.concatenate(panels, axis=1), footer))
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait(timeout=60) != 0:
            raise RuntimeError('Video encoding failed')
    except BaseException:
        process.kill()
        process.wait(timeout=10)
        raise
    finally:
        for capture in captures:
            capture.release()
    capture = cv2.VideoCapture(str(output))
    count = 0
    try:
        if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), source.fps, atol=.01):
            raise ValueError('Encoded video FPS differs')
        while True:
            ok, decoded = capture.read()
            if not ok:
                break
            if decoded.shape != (400, 1920, 3):
                raise ValueError('Encoded video shape differs')
            count += 1
    finally:
        capture.release()
    if count != source.num_frames:
        raise ValueError('Encoded video lost frames')
    return {'file': file_identity(output), 'frames_written': source.num_frames, 'frames_read': count,
            'fps': source.fps, 'size': [1920, 400], 'cameras': list(source.camera_ids), 'labels_used': False}

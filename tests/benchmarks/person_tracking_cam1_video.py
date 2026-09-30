"""Render the three predeclared cam1-far diagnostic windows and matching evidence."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from person_tracking_linking import (  # type: ignore[import-not-found]
    CLIP,
    DEEP,
    checked,
    record_file,
)
from person_tracking_matrix_video import load_result  # type: ignore[import-not-found]

from src.tasks.person_tracking.archive import load_features
from src.tennis_scene.pipeline.artifacts import write_json_atomic


def render(diagnosis: Path, matrix: Path, output: Path) -> None:
    report = json.loads(diagnosis.read_text())
    identity = json.loads((matrix / 'identity.json').read_text())
    plan = json.loads(checked(identity['plan']).read_text())
    source = json.loads(checked(plan['sources']).read_text())
    manifest = json.loads(checked(identity['features']).read_text())
    if output.exists():
        raise FileExistsError(output)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', '1920x880', '-r', '15', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
               '-pix_fmt', 'yuv420p', '-threads', '1', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError('ffmpeg stdin unavailable')
    written = 0
    try:
        for index, window in enumerate(report['windows']):
            clip = window['clip']
            record = next(r for r in source['inputs'] if r['clip'] == clip and r['camera'] == 'cam1')
            features = load_features(checked(manifest['records'][CLIP][f'{clip}/cam1']))[0]
            result, _ = load_result(matrix, DEEP, clip)
            with np.load(checked(result['cameras']['cam1']['tracking']['arrays']), allow_pickle=False) as saved:
                arrays = dict(saved)
            with gzip.open(checked(report['clips'][clip]['details']), 'rt') as stream:
                details = {d['frame']: d for line in stream if (d := json.loads(line))}
            capture = cv2.VideoCapture(str(checked(record['video'])))
            if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, window['start']):
                raise ValueError('Cannot open/seek diagnostic video')
            try:
                for frame in range(window['start'], window['end']):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError('Video ended inside the fixed window')
                    if (frame - window['start']) % 4:
                        continue
                    detail = details.get(frame)
                    positions = features[frame]
                    ids = {int(arrays['origins'][row, frame]): int(arrays['track_ids'][row])
                           for row in np.flatnonzero(arrays['observed'][:, frame])}
                    annotated = image.copy()
                    for row, box in zip(positions.rows, positions.boxes, strict=True):
                        x1, y1, x2, y2 = np.rint(box).astype(int)
                        color = (0, 220, 255) if detail is not None and row == detail.get('detection_row') else (220, 180, 60)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f'row {row} ID {ids.get(int(row), "-")}', (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                    box = np.asarray(detail['box']) if detail is not None and 'box' in detail else np.array([480, 400, 600, 540])
                    center = (box[:2] + box[2:]) / 2
                    left = int(np.clip(center[0] - 240, 0, 1920 - 480))
                    top = int(np.clip(center[1] - 135, 0, 1080 - 270))
                    canvas: np.ndarray = np.zeros((880, 1920, 3), np.uint8)
                    canvas[40:580, :960] = cv2.resize(annotated, (960, 540))
                    canvas[40:580, 960:] = cv2.resize(annotated[top:top + 270, left:left + 480], (960, 540))
                    title = f'Window {index + 1}: {clip} cam1  frame {frame} | full view + far-player zoom'
                    cv2.putText(canvas, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 1, cv2.LINE_AA)
                    texts = ['Deep OC-SORT+pose / CLIP (run 8); yellow = GT-overlapping detection; blue = competitors; - = not emitted']
                    if detail is not None:
                        internal = detail.get('internal') or {}
                        texts += [f'GT {detail["person"]}: {detail["state"]} | selected ID={detail["track_id"]} | internal ID={internal.get("id")} hits={internal.get("hit_streak")}',
                                  f'detection IoU={detail["detection_iou"]:.3f} height={detail.get("height_px", 0):.1f}px joints>=.3={detail.get("pose_joints_ge03", 0)}',
                                  'stage / track ID / IoU / appearance similarity / appearance contribution / pose penalty / total / assigned']
                        matches = sorted(detail.get('matching', []), key=lambda m: (-m['iou'], m['stage'], m['track_id']))[:5]
                        texts += [f'{m["stage"]:5s} ID {m["track_id"]:3d}   {m["iou"]:.3f}   {m["appearance_similarity"]:.3f}   '
                                  f'{m["appearance_cost"]:.3f}   {m["pose_cost"]:.3f}   {m["similarity"]:.3f}   {m["accepted"]}' for m in matches]
                    for line, text in enumerate(texts):
                        cv2.putText(canvas, text, (15, 606 + 29 * line), cv2.FONT_HERSHEY_SIMPLEX, .64, (240, 240, 240), 1, cv2.LINE_AA)
                    process.stdin.write(canvas.tobytes())
                    if (frame - window['start']) == 4:
                        cv2.imwrite(str(output.parent / f'diagnosis-window-{index + 1}.jpg'), canvas)
                    written += 1
            finally:
                capture.release()
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    read = cv2.VideoCapture(str(output))
    count = 0
    while True:
        ok, frame = read.read()
        if not ok:
            break
        if frame.shape != (880, 1920, 3):
            raise ValueError('Diagnostic video dimensions changed')
        count += 1
    read.release()
    if count != written:
        raise ValueError('Diagnostic video read-back lost frames')
    write_json_atomic(output.with_suffix('.json'), {'video': record_file(output), 'diagnosis': record_file(diagnosis),
                                                   'frames_written': written, 'frames_read': count, 'windows': report['windows']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    render(args.diagnosis, args.matrix, args.output)

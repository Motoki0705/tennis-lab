"""Two adjacent-court failure cases, both saved sources, synchronized 3cam video."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from person_selection_cpu import (  # type: ignore[import-not-found]  # sibling CLI
    calibration,
)

from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH, HALF_SINGLES_WIDTH


def text_at(image: NDArray[np.uint8], text: str, x: int, y: int, scale: float = .6) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def polygon(half_width: float, half_length: float) -> np.ndarray:
    result: np.ndarray = np.array([[-half_width, -half_length], [half_width, -half_length],
                     [half_width, half_length], [-half_width, half_length]], np.float64)
    return result


def ground_panel(points: np.ndarray, active: np.ndarray, old: np.ndarray, new: np.ndarray) -> NDArray[np.uint8]:
    panel: NDArray[np.uint8] = np.zeros((360, 640, 3), np.uint8)
    def pixel(xy: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.round(xy * np.array([20., -7.8]) + [320, 180]).astype(np.int32)
        return result
    for width, length, colour in ((HALF_DOUBLES_WIDTH + 2.5, HALF_LENGTH + 5, (200, 80, 200)),
        (HALF_DOUBLES_WIDTH, HALF_LENGTH + 5, (255, 255, 255)),
        (HALF_SINGLES_WIDTH, HALF_LENGTH + 5, (255, 180, 0)),
        (HALF_DOUBLES_WIDTH, HALF_LENGTH, (80, 100, 80))):
        cv2.polylines(panel, [pixel(polygon(width, length))], True, colour, 1)
    cv2.line(panel, (210, 180), (430, 180), (130, 130, 130), 1)
    for row in np.flatnonzero(active):
        colour = (70, 210, 70) if new[row] else (60, 60, 255) if old[row] else (130, 130, 130)
        if np.abs(points[row]).max() <= 40:
            cv2.circle(panel, tuple(pixel(points[row])), 4, colour, -1)
    text_at(panel, 'cam0 calibrated footpoints (metres, z=0)', 8, 20, .5)
    text_at(panel, 'purple: old | white: outer | cyan: dwell core', 8, 345, .48)
    return panel


def render(report: Path) -> None:
    cv2.setNumThreads(2)
    result = json.loads((report / 'selection.json').read_text())
    diagnosis = json.loads((report / 'diagnosis.json').read_text())
    previous = Path(result['previous'])
    old = json.loads((previous / 'selection.json').read_text())
    sides = json.loads(Path(old['side_decisions']).read_text())
    target, temporary = report / 'adjacent_failure_3cam.mp4', report / 'adjacent_failure.raw.mp4'
    if target.exists() or temporary.exists():
        raise FileExistsError(target)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter.fourcc(*'mp4v'), 20., (1920, 720))
    if not writer.isOpened():
        raise RuntimeError('Video encoder failed')
    cases, previews = [], []
    count = 0
    try:
        for clip, start in (('video_000/clip_000', 750), ('video_001/clip_001', 60)):
            records = sorted((r for r in result['inputs'] if r['clip'] == clip), key=lambda r: r['camera'])
            side = next(s for s in sides['clips'] if s['clip_id'] == clip)
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            cameras = [calibration(r, turns[r['camera']]) for r in records]
            for source in ('ft_base_0.01', 'union_0.30'):
                data, captures = [], []
                for record, camera in zip(records, cameras, strict=True):
                    value = result['records'][source][f"{clip}/{record['camera']}"]
                    for prefix in ('', 'previous_'):
                        if dual_sha256(Path(value[prefix + 'path'])) != value[prefix + 'sha256']:
                            raise ValueError('Selection archive changed')
                    if dual_sha256(Path(record['video']['path'])) != record['video']['sha256']:
                        raise ValueError('Dev video changed')
                    with np.load(value['previous_path']) as a, np.load(value['path']) as b:
                        value = {**dict(a), **dict(b)}
                    points, valid = ground_footpoints(value['boxes'], value['observed'], camera, 1080, FootpointConfig())
                    value.update(points=points, valid=valid)
                    data.append(value)
                    captures.append(cv2.VideoCapture(record['video']['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]))
                evidence = diagnosis['records'][source][f'{clip}/cam0']
                adjacent = evidence['adjacent_units']
                frames = start + np.round(np.arange(80) * records[0]['video']['fps'] / 20).astype(int)
                cases.append({'source': source, 'clip': clip, 'source_frames': frames.tolist()})
                for capture in captures:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
                next_frame = int(frames[0])
                try:
                    for n, frame in enumerate(frames):
                        canvas: NDArray[np.uint8] = np.zeros((720, 1920, 3), np.uint8)
                        unit = min(adjacent, key=lambda u: abs(u['frame'] - frame))
                        for k, capture in enumerate(captures):
                            for _ in range(next_frame, int(frame)):
                                if not capture.grab():
                                    raise ValueError('Dev video skipped-frame decode failed')
                            ok, image = capture.read()
                            if not ok:
                                raise ValueError('Dev video decode failed')
                            d = data[k]
                            accepted_old = np.isin(np.arange(len(d['track_ids'])), d['chosen'])
                            for row in np.flatnonzero(d['observed'][:, frame]):
                                colour = (70, 210, 70) if d['selected'][row, frame] else (60, 60, 255) if accepted_old[row] else (160, 160, 160)
                                x1, y1, x2, y2 = np.round(d['boxes'][row, frame]).astype(int)
                                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                            canvas[:360, k * 640:(k + 1) * 640] = cv2.resize(image, (640, 360))
                            text_at(canvas, f'{records[k]["camera"]} frame {frame}', k * 640 + 8, 24)
                            if k == 0:
                                x1, y1, x2, y2 = unit['box']
                                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                left, top = int(np.clip(cx - 240, 0, 1440)), int(np.clip(cy - 135, 0, 810))
                                canvas[360:, :640] = cv2.resize(image[top:top + 270, left:left + 480], (640, 360))
                                text_at(canvas, f'cam0 detail | reviewed adjacent {unit["person"]}', 8, 386, .56)
                                canvas[360:, 640:1280] = ground_panel(d['points'][:, frame], d['valid'][:, frame], accepted_old, d['selected'][:, frame])
                        summary = next(t for t in evidence['tracks'] if t['track_id'] == unit['track_id'])
                        lines = [f'DEV {clip} | {source}', 'Red: old rule only | green: improved rule',
                            f'Adjacent raw track {unit["track_id"]}: {summary["composition"]}',
                            f'Player identity mixing: {summary["player_and_adjacent_mixed"]}',
                            f'Labelled adjacent units kept: {summary["adjacent_kept_units"]}',
                            f'Old region: {summary["adjacent_inside_old"]} | core: {summary["adjacent_inside_core"]}',
                            f'Nearest labelled foot xy: {unit["xy"][0]:.2f}, {unit["xy"][1]:.2f} m',
                            'Footpoint from box bottom, not ankle.',
                            'Labels identify failure cases only.',
                            'Union COCO component is still ROI-filtered.']
                        for line, text in enumerate(lines):
                            text_at(canvas, text, 1290, 388 + line * 31, .49)
                        writer.write(canvas)
                        next_frame = int(frame) + 1
                        if n == 35:
                            preview = report / f'adjacent_preview_{len(cases)}.jpg'
                            if not cv2.imwrite(str(preview), canvas):
                                raise RuntimeError('Preview write failed')
                            previews.append({'path': str(preview), 'sha256': dual_sha256(preview)})
                        count += 1
                finally:
                    for capture in captures:
                        capture.release()
    finally:
        writer.release()
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(temporary), '-c:v', 'libx264',
        '-threads', '2', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
    read = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    decoded = 0
    while True:
        ok, image = read.read()
        if not ok:
            break
        if image.shape != (720, 1920, 3):
            raise ValueError('Unexpected video dimensions')
        decoded += 1
    read.release()
    if count != decoded or decoded != 320:
        raise ValueError('Review video incomplete on readback')
    temporary.unlink()
    write_json_atomic(report / 'video.json', {'path': str(target), 'sha256': dual_sha256(target),
        'frames': count, 'fps': 20, 'shape': [720, 1920, 3], 'cases': cases, 'previews': previews,
        'selection_sha256': dual_sha256(report / 'selection.json'), 'readback': 'all 320 frames verified'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    render(parser.parse_args().report)

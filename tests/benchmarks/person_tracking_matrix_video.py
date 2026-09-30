"""Three-camera review of the predeclared largest-difference windows (CPU)."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from person_tracking_matrix import (  # type: ignore[import-not-found]
    CLIP,
    checked,
    record_file,
)

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tennis_scene.pipeline.artifacts import write_json_atomic


def recommendation(table: list[dict[str, Any]]) -> str:
    candidates = [r for r in table if '__' in r['variant'] and r['camera'] == r['near_far'] == 'all']
    best = max(r['idf1'] for r in candidates)
    tied = [r for r in candidates if best - r['idf1'] <= 1e-6]
    chosen: str = min(tied, key=lambda r: (r['id_switches'], r['fragments'], r['variant']))['variant']
    return chosen


def load_result(report: Path, variant: str, clip: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = json.loads((report / 'evaluation' / variant / clip / 'result.json').read_text())
    with gzip.open(checked(result['units']), 'rt') as src:
        units = [json.loads(line) for line in src]
    return result, units


def states(result: dict[str, Any], units: list[dict[str, Any]]) -> dict[tuple[str, int, str], str]:
    mapping = {(r['camera'], r['track_id']): r['person'] for r in result['metrics']['mapping']}
    return {(u['camera'], u['frame'], u['person']): 'miss' if u['track_id'] is None else
            'correct' if mapping.get((u['camera'], u['track_id'])) == u['person'] else 'wrong_identity'
            for u in units if u['role'] == 'player'}


def draw(image: np.ndarray, box: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = np.rint(box / 3).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (max(0, x1), max(16, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)


def render(report: Path) -> None:
    manifest = json.loads((report / 'comparison.json').read_text())
    chosen = recommendation(manifest['table'])
    identity = json.loads(checked(manifest['identity']).read_text())
    plan = json.loads(checked(identity['plan']).read_text())
    source = json.loads(checked(plan['sources']).read_text())
    output = report / 'review.mp4'
    if output.exists():
        raise FileExistsError(output)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', '1920x800', '-r', '15', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '25',
        '-pix_fmt', 'yuv420p', '-threads', '1', '-movflags', '+faststart', str(output)]
    video = subprocess.Popen(command, stdin=subprocess.PIPE)
    if video.stdin is None:
        raise RuntimeError('ffmpeg stdin unavailable')
    windows = []
    try:
        for clip in sorted({r['clip'] for r in source['inputs']}):
            records = sorted([r for r in source['inputs'] if r['clip'] == clip], key=lambda r: r['camera'])
            baseline, bu = load_result(report, 'new_old', clip)
            candidate, cu = load_result(report, chosen, clip)
            before, after = states(baseline, bu), states(candidate, cu)
            if set(before) != set(after):
                raise ValueError('Comparison units differ')
            differences = np.zeros(records[0]['video']['num_frames'], np.int64)
            for key in before:
                differences[key[1]] += before[key] != after[key]
            width = min(len(differences), round(5 * records[0]['video']['fps']))
            sums = np.convolve(differences, np.ones(width, np.int64), mode='valid')
            start = int(sums.argmax())
            end = start + width
            windows.append({'clip': clip, 'start': start, 'end': end, 'differing_player_units': int(sums[start]),
                            'baseline': 'new_old', 'candidate': chosen})
            arrays = []
            for result in (baseline, candidate):
                cams = []
                for record in records:
                    with np.load(checked(result['cameras'][record['camera']]['arrays']), allow_pickle=False) as a:
                        cams.append(dict(a))
                arrays.append(cams)
            labels = ClipLabels.load(Path(records[0]['label_path']))
            captures = [cv2.VideoCapture(str(checked(r['video']))) for r in records]
            for capture in captures:
                if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise ValueError('Video could not seek to selected window')
            try:
                for frame in range(start, end):
                    images = []
                    for capture in captures:
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError('Video ended inside review window')
                        images.append(image)
                    if (frame - start) % 4:
                        continue
                    canvas: np.ndarray = np.zeros((800, 1920, 3), np.uint8)
                    for stage, (result, units, title) in enumerate(((baseline, bu, 'COCO .30 + old BoT-SORT/Lab'),
                                                                  (candidate, cu, chosen.replace(CLIP, 'CLIP')))):
                        y = stage * 400
                        cv2.putText(canvas, f'{clip}  frame {frame}  {title}  | green=correct magenta=error orange=nonplayer',
                                    (10, y + 27), cv2.FONT_HERSHEY_SIMPLEX, .64, (255, 255, 255), 1, cv2.LINE_AA)
                        for view, record in enumerate(records):
                            cam, archive = record['camera'], arrays[stage][view]
                            tile = cv2.resize(images[view], (640, 360))
                            local = {u['track_id']: u for u in units if u['camera'] == cam and u['frame'] == frame and u['track_id'] is not None}
                            status = states(result, [u for u in units if u['camera'] == cam and u['frame'] == frame])
                            for row in np.flatnonzero(archive['selected'][:, frame]):
                                tid = int(archive['track_ids'][row])
                                unit = local.get(tid)
                                caption, color = f'ID {tid}', (255, 220, 0)
                                if unit is not None:
                                    caption += f' {unit["person"]}'
                                    color = (0, 210, 0) if status.get((cam, frame, unit['person'])) == 'correct' else (230, 30, 230)
                                    if unit['role'] != 'player':
                                        color = (0, 150, 255)
                                    for field, tag in (('switch_events', 'SWITCH'), ('fragment_events', 'FRAGMENT')):
                                        if any(e['camera'] == cam and e['person'] == unit['person'] and 0 <= frame - e['frame'] <= 18
                                               for e in result['metrics'][field]):
                                            caption += f' {tag}'
                                draw(tile, archive['boxes'][row, frame], caption, color)
                            for unit in (u for u in units if u['camera'] == cam and u['frame'] == frame and u['role'] == 'player' and u['track_id'] is None):
                                refs = labels.cameras[cam]
                                at = refs.at(frame)
                                person = next(i for i, p in enumerate(labels.people) if p.person_id == unit['person'])
                                boxes = refs.boxes_xyxy[at][refs.person_index[at] == person]
                                box = boxes[np.prod(boxes[:, 2:] - boxes[:, :2], axis=1).argmax()]
                                draw(tile, box, f'MISS {unit["person"]}', (230, 30, 230))
                            cv2.putText(tile, cam, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                            metrics = result['cameras'][cam]['metrics']
                            cv2.rectangle(tile, (0, 338), (470, 360), (0, 0, 0), -1)
                            cv2.putText(tile, f'clip IDF1 {metrics["idf1"]:.3f}  switches {metrics["id_switches"]}  fragments {metrics["fragments"]}',
                                        (8, 354), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
                            if result['cameras'][cam]['tracking']['status'] != 'ok':
                                cv2.putText(tile, 'TRACKER STOPPED', (120, 55), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
                            canvas[y + 40:y + 400, view * 640:(view + 1) * 640] = tile
                    video.stdin.write(canvas.tobytes())
            finally:
                for capture in captures:
                    capture.release()
    finally:
        video.stdin.close()
    if video.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    write_json_atomic(report / 'review.json', {'video': record_file(output), 'windows': windows,
        'comparison': record_file(report / 'comparison.json'), 'candidate': chosen,
        'event_overlay': 'switch/fragment persists 18 source frames for 15fps readability'})
    print(json.dumps({'candidate': chosen, 'video': str(output), 'windows': windows}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    cv2.setNumThreads(1)
    render(parser.parse_args().report)

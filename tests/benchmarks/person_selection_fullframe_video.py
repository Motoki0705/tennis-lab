"""Review the recommended source, including per-clip worst windows and wide runs.

Labels select diagnostic windows only after the comparison is frozen. No
label is an input to detection, tracking, dwell selection or association.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from person_selection_failure_video import text_at  # type: ignore[import-not-found]

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.fullframe_sources import SOURCES
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def windows(comparison: dict[str, Any], source: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for clip in DEV_CLIPS:
        value = comparison['results'][source][clip]
        with gzip.open(value['units']['path'], 'rt') as src:
            all_units = [json.loads(line) for line in src]
        units = [u for u in all_units if u['stage'] == 'selected']
        cross = next(r for r in comparison['association'] if r['source'] == source and r['clip'] == clip)
        length = max(u['frame'] for u in units) + 1
        width = min(240, length)  # four seconds at the fixed Meiji source rate
        for kind in ('worst', 'wide', 'adjacent', 'association', 'nonplayer'):
            loss = np.zeros(length, np.int64)
            for u in units:
                value = (u['role'] == 'player' and not u['selected_03']) or (u['role'] != 'player' and u['selected_03'])
                if kind == 'wide':
                    value = u['role'] == 'player' and u['reference_wide'] is True
                elif kind == 'adjacent':
                    value = u['kind'] == 'adjacent_court'
                elif kind == 'association':
                    value = False
                elif kind == 'nonplayer':
                    value = u['role'] != 'player' and u['selected_03']
                loss[u['frame']] += int(value)
            if kind == 'association' and cross['metrics']:
                for failure in cross['metrics']['0.3']['failed_frame_runs']:
                    loss[failure['start']:failure['end']] += 1
            sums = np.convolve(loss, np.ones(width, np.int64), mode='valid')
            start = int(np.argmax(sums))
            if kind == 'nonplayer' and loss.any():
                # Centre a short failure instead of placing it at the last
                # frame of a four-second window; the preview then shows it.
                start = int(np.clip(int(np.flatnonzero(loss)[len(np.flatnonzero(loss)) // 2]) - width // 2, 0, length - width))
            item = {'clip': clip, 'start': start, 'end': start + width, 'kind': kind, 'score': int(sums[start])}
            (result if kind == 'worst' else extra).append(item)
    for kind in ('wide', 'adjacent', 'association', 'nonplayer'):
        best = max((r for r in extra if r['kind'] == kind), key=lambda r: r['score'])
        if best['score'] and not any((r['clip'], r['start'], r['end']) == (best['clip'], best['start'], best['end']) for r in result):
            result.append(best)
    return result


def box(image: np.ndarray, xyxy: np.ndarray, color: tuple[int, int, int], label: str = '', *, scale: float = 1 / 3) -> None:
    x1, y1, x2, y2 = np.round(xyxy * scale).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if label else 1)
    if label:
        cv2.putText(image, label, (x1, max(15, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)


def render(report: Path, source: str, stem: str) -> None:
    cv2.setNumThreads(1)
    comparison_path = report / 'comparison.json'
    comparison = json.loads(comparison_path.read_text())
    inputs = json.loads((report / 'sources.json').read_text())['inputs']
    cases = windows(comparison, source)
    if not stem or Path(stem).name != stem:
        raise ValueError('Video stem must be a filename')
    target, raw_path = report / f'{stem}.mp4', report / f'{stem}.raw.mp4'
    if target.exists() or raw_path.exists():
        raise FileExistsError(target)
    shape, fps = (1920, 820), 20.
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter.fourcc(*'mp4v'), fps, shape)
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    frames_written, previews = 0, []
    try:
        for case_index, case in enumerate(cases):
            clip = case['clip']
            verdict = comparison['results'][source][clip]
            records = sorted((r for r in inputs if r['clip'] == clip), key=lambda r: r['camera'])
            labels = ClipLabels.load(Path(records[0]['label_path']))
            with gzip.open(verdict['units']['path'], 'rt') as handle:
                all_units = [json.loads(line) for line in handle]
            units = [u for u in all_units if u['stage'] == 'selected']
            unit_index = {(u['camera'], u['frame'], u['person']): u for u in units}
            associated_index = {(u['camera'], u['frame'], u['person']): u for u in all_units if u['stage'] == 'associated'}
            captures, data = [], []
            for record in records:
                saved = verdict['cameras'][record['camera']]
                if dual_sha256(Path(record['video']['path'])) != record['video']['sha256'] or dual_sha256(Path(saved['path'])) != saved['sha256']:
                    raise ValueError('Video/selection hash changed')
                with np.load(saved['path'], allow_pickle=False) as a:
                    arrays = {k: a[k] for k in ('boxes', 'observed', 'selected', 'track_ids')}
                arrays['player_ids'] = np.full(arrays['observed'].shape, -1, np.int64)
                if verdict['association']['status'] == 'ok':
                    ids = saved['identities']
                    if dual_sha256(Path(ids['path'])) != ids['sha256']:
                        raise ValueError('Association hash changed')
                    with np.load(ids['path'], allow_pickle=False) as a:
                        arrays['player_ids'] = a['player_ids']
                data.append(arrays)
                capture = cv2.VideoCapture(str(record['video']['path']))
                if not capture.isOpened():
                    raise RuntimeError('Cannot read dev video')
                captures.append(capture)
            source_fps = records[0]['video']['fps']
            frames = np.round(np.arange(80) * source_fps / fps).astype(int) + case['start']
            frames = frames[frames < case['end']]
            cursor = -1
            try:
                for local_index, frame in enumerate(frames):
                    originals = []
                    # Decode in order: random seeks were unreliable for these clips.
                    while cursor < frame:
                        cursor += 1
                        current = [cap.read() for cap in captures]
                        if not all(ok for ok, _ in current):
                            raise RuntimeError('Review video decode ended early')
                        if cursor == frame:
                            originals = [bgr for _, bgr in current]
                    canvas: np.ndarray = np.zeros((shape[1], shape[0], 3), np.uint8)
                    text_at(canvas, f'{source} | {clip} | {case["kind"]} | frame {frame} | association: {verdict["association"]["status"]}', 10, 25, .62)
                    text_at(canvas, 'Green/P*: selected / cross-camera ID | Grey: excluded | Red: missed ref | Amber: no ID | Magenta: kept non-player ref (COCO-derived)', 10, 50, .53)
                    for ci, (record, image, a) in enumerate(zip(records, originals, data, strict=True)):
                        panel = cv2.resize(image, (640, 360))
                        annotated = image.copy()
                        active = np.flatnonzero(a['observed'][:, frame])
                        for row in active:
                            selected = bool(a['selected'][row, frame])
                            identity = int(a['player_ids'][row, frame])
                            label = f'P{identity}' if identity >= 0 else f'S t{a["track_ids"][row]}' if selected else ''
                            color = (60, 220, 60) if selected else (140, 140, 140)
                            box(panel, a['boxes'][row, frame], color, label)
                            box(annotated, a['boxes'][row, frame], color, label, scale=1.)
                        ref = labels.cameras[record['camera']]
                        at = ref.at(int(frame))
                        refs = []
                        for person in np.unique(ref.person_index[at]):
                            if person < 0:
                                continue
                            name = labels.people[person].person_id
                            candidates = ref.boxes_xyxy[at][ref.person_index[at] == person]
                            xyxy = candidates[np.prod(candidates[:, 2:] - candidates[:, :2], axis=1).argmax()]
                            u = unit_index[record['camera'], int(frame), name]
                            missed = u['role'] == 'player' and not u['selected_03']
                            associated = associated_index.get((record['camera'], int(frame), name))
                            no_identity = bool(u['role'] == 'player' and u['selected_03'] and associated is not None and not associated['selected_03'])
                            nonplayer = u['role'] != 'player' and u['selected_03']
                            if missed:
                                box(panel, xyxy, (30, 30, 255), f'ref {name}')
                                box(annotated, xyxy, (30, 30, 255), f'ref {name}', scale=1.)
                            elif no_identity:
                                box(panel, xyxy, (0, 180, 255), f'no ID {name}')
                                box(annotated, xyxy, (0, 180, 255), f'no ID {name}', scale=1.)
                            elif nonplayer:
                                box(panel, xyxy, (255, 0, 255), f'kept ref {name}')
                                box(annotated, xyxy, (255, 0, 255), f'kept ref {name}', scale=1.)
                            priority = 10 * int(missed) + int(u['near_far'] == 'far')
                            if case['kind'] == 'association':
                                priority += 20 * int(no_identity)
                            if case['kind'] == 'nonplayer':
                                priority += 20 * int(nonplayer)
                            if case['kind'] == 'wide':
                                priority += 20 * int(u['reference_wide'] is True and u['role'] == 'player')
                            if case['kind'] == 'adjacent':
                                priority += 20 * int(u['kind'] == 'adjacent_court')
                            refs.append((priority, xyxy, name, u))
                        text_at(panel, record['camera'], 8, 24, .6)
                        canvas[60:420, ci * 640:(ci + 1) * 640] = panel
                        if refs:
                            _, xyxy, name, u = max(refs, key=lambda v: v[0])
                            centre = (xyxy[:2] + xyxy[2:]) / 2
                            width = max(320, int((xyxy[2] - xyxy[0]) * 3))
                            height = max(180, int((xyxy[3] - xyxy[1]) * 2))
                            width = min(width, 1920)
                            height = min(height, 1080)
                            x = int(np.clip(centre[0] - width / 2, 0, 1920 - width))
                            y = int(np.clip(centre[1] - height / 2, 0, 1080 - height))
                            crop = cv2.resize(annotated[y:y + height, x:x + width], (640, 340))
                            canvas[435:775, ci * 640:(ci + 1) * 640] = crop
                            status = 'KEPT' if u['selected_03'] else 'MISSED' if u['role']=='player' else 'EXCLUDED'
                            text_at(canvas, f'{record["camera"]}: reference {name} {u["kind"]} {status}', ci * 640 + 8, 803, .55)
                    writer.write(canvas)
                    frames_written += 1
                    if local_index == len(frames) // 2:
                        preview = report / f'{stem}_preview_{case_index}.jpg'
                        if not cv2.imwrite(str(preview), canvas):
                            raise RuntimeError('Preview write failed')
                        previews.append({'path': str(preview), 'sha256': dual_sha256(preview), 'case': case_index, 'source_frame': int(frame)})
            finally:
                for cap in captures:
                    cap.release()
            case['source_frames'] = frames.tolist()
    finally:
        writer.release()
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(raw_path), '-c:v', 'libx264', '-threads', '4', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
    read = cv2.VideoCapture(str(target))
    count = 0
    while True:
        ok, frame = read.read()
        if not ok:
            break
        if frame.shape != (820, 1920, 3):
            raise ValueError('Unexpected video shape')
        count += 1
    read.release()
    if count != frames_written:
        raise ValueError('Encoded review video failed full readback')
    raw_path.unlink()  # only this run's temporary intermediate
    write_json_atomic(report / f'{stem}.json', {'path': str(target), 'sha256': dual_sha256(target), 'bytes': target.stat().st_size,
        'source': source, 'comparison_sha256': dual_sha256(comparison_path), 'cases': cases, 'previews': previews,
        'fps': fps, 'frames': count, 'shape': [820, 1920, 3], 'readback': 'all frames verified',
        'worst_definition': 'per-clip max selected-stage missed-player + retained-nonplayer count over 240 source frames; add maximum-wide, adjacent, association-failed-frame (#933 group metric, IoU .3), and centred retained-nonplayer windows; identical windows appear once',
        'labels': 'post-hoc window selection and red review overlay only; COCO-derived and biased'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--stem', default='best_source_failures_3cam')
    args = parser.parse_args()
    render(args.report, args.source, args.stem)


if __name__ == '__main__':
    main()

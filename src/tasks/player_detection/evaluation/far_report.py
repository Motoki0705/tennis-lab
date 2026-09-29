"""CPU tables and review video for immutable #964 detector diagnosis archives."""
from __future__ import annotations

import csv
import gzip
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.submodules.models import PersonDetectionResult, filter_detections_by_footpoint
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.far_player import (
    add_counts,
    merge_extra,
    rates,
    threshold,
    unit_rows,
)
from src.tasks.player_detection.evaluation.partial_labels import PartialDetectionMetrics
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256


def load_baseline(record: dict[str, Any]) -> PersonDetectionOutput:
    root = Path(record['baseline_store'])
    source = json.loads((root / 'scene.json').read_text())['source']
    store = ClipStore(root, source)
    reference = store.active(f"person_detection/{record['camera']}")
    if reference is None or json_value(reference) != record['baseline_reference']:
        raise ValueError('Baseline reference changed')
    return store.load(reference, ArtifactCodec(PersonDetectionOutput))


def frame_variants(plan: dict[str, Any], record: dict[str, Any], archives: dict[str, DetectionArchive],
                   frame: int) -> tuple[dict[str, PersonDetectionResult], dict[str, float]]:
    predictions, times = {}, {}
    roi = tuple(tuple(point) for point in record['roi'])
    for score in plan['thresholds']:
        name = f'ft_s{score:.2f}'
        start = time.perf_counter()
        predictions[name] = filter_detections_by_footpoint(threshold(archives['ft_base'].at(frame), score), roi)
        times[name] = float(archives['ft_base'].milliseconds[frame]) + (time.perf_counter() - start) * 1000
    for raw in ('ft_1080', 'ft_largest', 'coco_base'):
        if raw == 'ft_largest' and raw not in archives:
            continue  # Explicit capacity result: largest supported trial was already 1080.
        name = f'{raw}_s0.30'
        start = time.perf_counter()
        predictions[name] = filter_detections_by_footpoint(threshold(archives[raw].at(frame), .3), roi)
        times[name] = float(archives[raw].milliseconds[frame]) + (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    tiled = filter_detections_by_footpoint(threshold(archives['ft_tiles'].at(frame), .3), roi)
    predictions['ft_plus_far_tiles_s0.30'] = merge_extra(predictions['ft_s0.30'], tiled, dedup_iou=plan['dedup_iou'])
    times['ft_plus_far_tiles_s0.30'] = times['ft_s0.30'] + float(archives['ft_tiles'].milliseconds[frame]) + (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    predictions['ft_union_coco_h64_s0.30'] = merge_extra(predictions['ft_s0.30'], predictions['coco_base_s0.30'],
        max_height=plan['union_max_height_at_1080p'] * record['video']['height'] / 1080, dedup_iou=plan['dedup_iou'])
    times['ft_union_coco_h64_s0.30'] = times['ft_s0.30'] + times['coco_base_s0.30'] + (time.perf_counter() - start) * 1000
    return predictions, times


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins = [.01, .05, .1, .2, .3, .5, 1.000001]
    result: dict[str, Any] = {'missed_old_player_units': len(rows), 'bins': bins}
    for field in ('best_score_before_roi', 'best_score_after_roi'):
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = {'no_overlap_candidate_at_floor_001': len(rows) - len(values),
                        'histogram': np.histogram(values, bins)[0].tolist(),
                        'p10_p50_p90': np.percentile(values, [10, 50, 90]).tolist() if values else None}
    return result


def summarize_report(report: Path, *, render: bool = True) -> dict[str, Any]:
    if (report / 'diagnosis.json').exists():
        raise FileExistsError(report / 'diagnosis.json')
    plan = json.loads((report / 'inference.json').read_text())
    if plan['status'] != 'ok':
        raise ValueError('All inference variants must complete before final comparison')
    required = {'ft_base', 'ft_1080', 'ft_tiles', 'coco_base'}
    if plan['capacity']['largest_successful_short_side'] > 1080:
        required.add('ft_largest')
    if set(plan['archives']) != required:
        raise ValueError('Required resolution/tile/COCO variant missing')
    expected_keys = {f"{r['clip']}/{r['camera']}" for r in plan['inputs']}
    if any(set(records) != expected_keys for records in plan['archives'].values()):
        raise ValueError('Inference variants did not cover identical inputs')
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    runtime: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0., 0.])
    missed, cases, per_clip = [], [], {}
    with gzip.open(report / 'missed_old_scores.jsonl.gz', 'wt') as handle:
        for record in plan['inputs']:
            key = f"{record['clip']}/{record['camera']}"
            label_path = Path(record['label_path'])
            if dual_sha256(label_path) != record['label_sha256']:
                raise ValueError('Dev labels changed')
            labels = ClipLabels.load(label_path)
            baseline = load_baseline(record)
            archives = {name: DetectionArchive.load(records[key]) for name, records in plan['archives'].items()}
            if any(len(a.milliseconds) != labels.num_frames for a in archives.values()):
                raise ValueError('Archive timeline differs from labels')
            metrics: dict[str, PartialDetectionMetrics] = {}
            camera_cases = []
            for frame in range(labels.num_frames):
                predictions, times = frame_variants(plan, record, archives, frame)
                start, end = baseline.frame_offsets[frame:frame + 2]
                predictions['stored_ft_s0.30'] = PersonDetectionResult(baseline.boxes_xyxy[start:end], baseline.confidence[start:end])
                reference = unit_rows(predictions['stored_ft_s0.30'], labels.cameras[record['camera']], labels.roles, frame)
                refs = {row['person']: row for row in reference}
                raw_rows = unit_rows(archives['ft_base'].at(frame), labels.cameras[record['camera']], labels.roles, frame)
                roi_rows = unit_rows(predictions['ft_s0.01'], labels.cameras[record['camera']], labels.roles, frame)
                for ref, raw, filtered in zip(reference, raw_rows, roi_rows, strict=True):
                    if ref['role'] == 'player' and not ref['matched']:
                        row = {'clip': record['clip'], 'camera': record['camera'], 'frame': frame,
                            'person': ref['person'], 'near_far': ref['near_far'], 'old_box': ref['old_box'],
                            'best_score_before_roi': raw['best_score_iou03'],
                            'best_score_after_roi': filtered['best_score_iou03']}
                        missed.append(row)
                        handle.write(json.dumps(row) + '\n')
                        if ref['near_far'] == 'far':
                            camera_cases.append(row)
                for name, prediction in predictions.items():
                    metrics.setdefault(name, PartialDetectionMetrics(.5, .0)).update(
                        prediction.boxes_xyxy, prediction.scores, labels=labels.cameras[record['camera']], roles=labels.roles, frame=frame)
                    for row in unit_rows(prediction, labels.cameras[record['camera']], labels.roles, frame):
                        for camera, side in ((record['camera'], row['near_far']), (record['camera'], 'all'), ('all', row['near_far']), ('all', 'all')):
                            add_counts(grouped[(name, camera, side)], row, refs[row['person']])
                    if name in times:
                        for camera in (record['camera'], 'all'):
                            totals = runtime[(name, camera)]
                            totals[0] += times[name]
                            totals[1] += 1
            if camera_cases:
                cases.append(camera_cases[len(camera_cases) // 2])
            per_clip[key] = {name: metric.compute() for name, metric in metrics.items()}
    table = []
    for (name, camera, side), counts in sorted(grouped.items()):
        timing = runtime.get((name, camera))
        table.append({'variant': name, 'camera': camera, 'near_far': side, **rates(counts),
                      'ms_per_frame': timing[0] / timing[1] if timing else None,
                      'timed_frames': int(timing[1]) if timing else 0})
    # Ranking is for dev review illustration only; it does not select deployment.
    ranked = sorted((row for row in table if row['camera'] == 'all' and row['near_far'] == 'far'
                     and row['variant'] not in {'stored_ft_s0.30', 'coco_base_s0.30', 'ft_s0.30'}),
                    key=lambda row: (-row['old_box_agreement'], row.get('added_non_player_units', 0), row['ms_per_frame']))
    best = [row['variant'] for row in ranked[:2]]
    result: dict[str, Any] = {'schema': 'far_player_diagnosis_summary_v1', 'status': 'ok',
        'inference_sha256': dual_sha256(report / 'inference.json'), 'scope': plan['scope'],
        'interpretation': plan['interpretation'], 'near_far': plan['near_far'], 'runtime_scope': plan['runtime_scope'],
        'capacity': plan['capacity'], 'table': table, 'per_clip_camera': per_clip,
        'missed_old_scores': {'all': _score_summary(missed), **{
            f'{camera}/{side}': _score_summary([r for r in missed if r['camera'] == camera and r['near_far'] == side])
            for camera in ('cam0', 'cam1', 'cam2') for side in ('near', 'far', 'unknown')}},
        'video_variants': best, 'video_selection': 'highest dev far old-box agreement; ties by added nonplayer hits and time; not adoption',
        'cases': cases, 'limits': 'Incomplete COCO-derived labels: unlabelled detections are not false positives; '
                                'added nonplayer units mean newly hit known person/frame units versus stored FT 0.3.'}
    fields = ['variant', 'camera', 'near_far', 'player_units', 'matched_player_units', 'old_box_agreement',
              'old_box_agreement_iou03', 'non_player_units', 'matched_non_player_units', 'added_non_player_units',
              'lost_non_player_units', 'non_player_hit_rate', 'ms_per_frame', 'timed_frames']
    with (report / 'diagnosis.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 遠側選手の旧box一致率診断', '',
        '同じ4開発clip・保存済みcourt ROIの比較。旧COCO boxとの一致（IoU≥0.5）であり検出recallではない。',
        '追加非選手は既知の人物×camera×frame単位。未ラベル予測は別集計で、全FPとは扱わない。',
        '近遠は画像内box下端順位。非選手は2選手の下端中点で層別。不明を捨てない。',
        'ms/frameは同期した推論＋ROI/union。動画decode・load・warmup・保存は除く。同camera値を近遠両行に表示。', '',
        '|camera|近遠|variant|一致/選手単位|一致率|追加非選手|非選手hit/単位|非選手hit率|ms/frame|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        if row['camera'] == 'all' or row['near_far'] == 'all':
            continue
        def percentage(value: float | None) -> str:
            return 'N/A' if value is None else f'{value * 100:.2f}%'
        ms = 'N/A (saved)' if row['ms_per_frame'] is None else f"{row['ms_per_frame']:.2f}"
        lines.append(f"|{row['camera']}|{row['near_far']}|{row['variant']}|"
                     f"{row.get('matched_player_units', 0)}/{row.get('player_units', 0)}|{percentage(row['old_box_agreement'])}|"
                     f"{row.get('added_non_player_units', 0)}|{row.get('matched_non_player_units', 0)}/{row.get('non_player_units', 0)}|"
                     f"{percentage(row['non_player_hit_rate'])}|{ms}|")
    (report / 'diagnosis.md').write_text('\n'.join(lines) + '\n')
    if render:
        video = render_video(report, plan, cases, best)
        result.update(video=str(video), video_sha256=dual_sha256(video))
    write_json_atomic(report / 'diagnosis.json', result)
    return result


def render_video(report: Path, plan: dict[str, Any], cases: list[dict[str, Any]], best: list[str]) -> Path:
    if len(best) != 2:
        raise ValueError('Expected two diagnostic video variants')
    raw_path = report / 'old_vs_variants.raw.mp4'
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter.fourcc(*'mp4v'), 15., (1280, 680))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    try:
        for index, case in enumerate(cases):
            record = next(r for r in plan['inputs'] if r['clip'] == case['clip'] and r['camera'] == case['camera'])
            key = f"{record['clip']}/{record['camera']}"
            if dual_sha256(Path(record['video']['path'])) != record['video']['sha256']:
                raise ValueError('Review video input changed')
            archives = {name: DetectionArchive.load(records[key]) for name, records in plan['archives'].items()}
            labels = ClipLabels.load(Path(record['label_path']))
            start = max(0, min(case['frame'] - 60, labels.num_frames - 120))
            end = min(labels.num_frames, start + 120)
            cap = cv2.VideoCapture(record['video']['path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            try:
                for frame in range(start, end):
                    ok, source_image = cap.read()
                    if not ok:
                        raise RuntimeError(f'Incomplete review video at {key}/{frame}')
                    if (frame - start) % 4:
                        continue
                    variants, _ = frame_variants(plan, record, archives, frame)
                    canvas: NDArray[np.uint8] = np.zeros((680, 1280, 3), np.uint8)
                    cv2.putText(canvas, f"{case['clip']} {case['camera']} f{frame} | ORANGE old player boxes / GREEN variant", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                    for col, name in enumerate(best):
                        image = source_image.copy()
                        labelled = labels.cameras[case['camera']]
                        selection = labelled.at(frame)
                        for box, person in zip(labelled.boxes_xyxy[selection], labelled.person_index[selection], strict=True):
                            if person >= 0 and labels.roles[person] == 'player':
                                x1, y1, x2, y2 = np.rint(box).astype(int).tolist()
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 160, 255), 3)
                        for box in variants[name].boxes_xyxy:
                            x1, y1, x2, y2 = np.rint(box).astype(int).tolist()
                            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 255, 80), 2)
                        left = col * 640
                        cv2.putText(canvas, name, (left + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
                        canvas[75:435, left:left + 640] = cv2.resize(image, (640, 360))
                        box = case['old_box']
                        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                        crop = image[max(0, cy - 90):min(image.shape[0], cy + 90), max(0, cx - 160):min(image.shape[1], cx + 160)]
                        canvas[445:675, left:left + 640] = cv2.resize(crop, (640, 230))
                    writer.write(canvas)
                    if frame == start:
                        cv2.imwrite(str(report / f'case_{index:02}.jpg'), canvas)
            finally:
                cap.release()
    finally:
        writer.release()
    path = report / 'old_vs_best_variants.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(raw_path), '-c:v', 'libx264', '-threads', '2',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], check=True)
    return path

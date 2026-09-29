"""CPU audit and short video of old-box disagreements in a saved detector comparison.

Reads immutable run-1 stores and relocated dataset labels. It never reruns a
model. Near/far is the explicitly labelled box-bottom image-space proxy.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.disagreements import (
    player_unit_rows,
    summarize,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    if (args.report / 'disagreements.json').exists():
        raise FileExistsError(args.report)
    comparison = json.loads(args.comparison.read_text())
    if comparison['status'] != 'ok':
        raise ValueError('Comparison must have completed')
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    totals: dict[str, Any] = {}
    cases = []
    for variant, record in comparison['variants'].items():
        variant_rows = []
        for clip, value in record['clips'].items():
            path = label_path(args.dataset, clip)
            labels = ClipLabels.load(path)
            expected_hash = comparison['labels'][clip]['sha256']
            migrated_hash = labels.provenance.get('migration', {}).get('source_label_sha256')
            if dual_sha256(path) != expected_hash and migrated_hash != expected_hash:
                raise ValueError(f'Label provenance mismatch: {path}')
            root = Path(value['store'])
            source = json.loads((root / 'scene.json').read_text())['source']
            store = ClipStore(root, source)
            clip_rows = []
            for video in source['videos']:
                camera = video['camera_id']
                reference = store.active(f'person_detection/{camera}')
                if reference is None:
                    raise ValueError(f'Missing detections: {clip} {camera}')
                detections = store.load(reference, ArtifactCodec(PersonDetectionOutput))
                # The comparison has already thresholded stored DINO detections at 0.3.
                if (detections.confidence < .3).any():
                    raise ValueError('Unexpected below-threshold stored detections')
                camera_rows = []
                for frame in range(labels.num_frames):
                    start, end = detections.frame_offsets[frame:frame + 2]
                    found = player_unit_rows(detections.boxes_xyxy[start:end], labels.cameras[camera], labels.roles,
                                             frame, min_iou=comparison['min_iou'])
                    for row in found:
                        row.update(variant=variant, clip=clip, camera=camera)
                    camera_rows.extend(found)
                clip_rows.extend(camera_rows)
                missing = [row for row in camera_rows if not row['matched']]
                if variant == 'player_ft' and missing:
                    # One reproducible two-second example per camera/clip; median unmatched frame.
                    case = missing[len(missing) // 2]
                    cases.append((case, video, labels, detections))
            expected = sum(c['metrics']['known_player_units'] - c['metrics']['matched_player_units'] for c in value['cameras'].values())
            if sum(not row['matched'] for row in clip_rows) != expected:
                raise ValueError(f'Diagnostic assignment differs from metric: {variant} {clip}')
            variant_rows.extend(clip_rows)
        rows.extend(variant_rows)
        totals[variant] = summarize(variant_rows)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['variant'] != 'player_ft':
            continue
        for key in ('camera', 'near_far', 'clip'):
            by_group[f'{key}/{row[key]}'].append(row)
        by_group[f"camera_near_far/{row['camera']}/{row['near_far']}"] .append(row)
    raw_video = args.report / 'unmatched_raw.mp4'
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter.fourcc(*'mp4v'), 15., (1440, 640))
    if not writer.isOpened():
        raise RuntimeError('Could not open video writer')
    try:
        for index, (case, video, labels, detections) in enumerate(cases):
            centre = case['frame']
            frame_start = max(0, min(centre - 60, labels.num_frames - 120))
            frame_end = min(labels.num_frames, frame_start + 120)
            examples.append({**case, 'start_frame': frame_start, 'end_frame': frame_end,
                             'source_video': video['path'], 'source_sha256': video['sha256']})
            cap = cv2.VideoCapture(video['path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            try:
                for frame in range(frame_start, frame_end):
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f'Video ended at {frame}')
                    if (frame - frame_start) % 4:
                        continue
                    old = labels.cameras[case['camera']]
                    selection = old.at(frame)
                    for box, person in zip(old.boxes_xyxy[selection], old.person_index[selection], strict=True):
                        if person >= 0 and labels.roles[person] == 'player':
                            x1, y1, x2, y2 = np.rint(box).astype(int).tolist()
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 160, 255), 2)
                    start, end = detections.frame_offsets[frame:frame + 2]
                    for box in detections.boxes_xyxy[start:end]:
                        x1, y1, x2, y2 = np.rint(box).astype(int).tolist()
                        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 255, 80), 1)
                    canvas: NDArray[np.uint8] = np.zeros((640, 1440, 3), np.uint8)
                    canvas[80:620, :960] = cv2.resize(image, (960, 540))
                    box = case['old_box']
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    radius = max(90, (box[3] - box[1]) * 1.5)
                    x1, x2 = max(0, int(cx-radius)), min(image.shape[1], int(cx+radius))
                    y1, y2 = max(0, int(cy-radius)), min(image.shape[0], int(cy+radius))
                    canvas[120:600, 960:] = cv2.resize(image[y1:y2, x1:x2], (480, 480))
                    text = f"{case['clip']} {case['camera']} f{frame} | case f{centre} best IoU={case['best_iou']:.3f}"
                    cv2.putText(canvas, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
                    cv2.putText(canvas, 'ORANGE: reviewed old boxes   GREEN: player FT | ROI comparison; old boxes are not independent GT',
                                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                    writer.write(canvas)
                    if frame == frame_start:
                        cv2.imwrite(str(args.report / f'case_{index:02}.jpg'), canvas)
            finally:
                cap.release()
    finally:
        writer.release()
    video_path = args.report / 'unmatched_old_vs_new.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(raw_video), '-c:v', 'libx264', '-threads', '2',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path)], check=True)
    with gzip.open(args.report / 'units.jsonl.gz', 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    output = {'scope': 'pipeline_court_roi', 'interpretation': 'agreement with biased old COCO boxes; not detection recall',
              'near_far_definition': 'image-space bottom rank of exactly two labelled players; otherwise unknown',
              'comparison': str(args.comparison), 'comparison_sha256': dual_sha256(args.comparison),
              'totals': totals, 'player_ft_groups': {k: summarize(v) for k, v in by_group.items()},
              'examples': examples, 'video': str(video_path), 'video_sha256': dual_sha256(video_path)}
    (args.report / 'disagreements.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()

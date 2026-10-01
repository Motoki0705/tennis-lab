"""Blind raw-track views and label materialization; never reads association output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from player_association_clips import track_sheet  # type: ignore[import-not-found]

from src.tasks.player_association.evaluation.labels import ReviewedTrack, materialize
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.definition import file_identity
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def load_raw(report: Path, record: dict[str, Any]) -> dict[str, PersonTrackingOutput]:
    root = report / record['clip']
    receipt = json.loads((root / 'person-execute.json').read_text())
    store = ClipStore(root / 'store', record['source'], memory_entries=0)
    return {video['camera_id']: store.load(
        ArtifactRef(**receipt['references'][f'person_tracking/{video["camera_id"]}']),
        ArtifactCodec(PersonTrackingOutput)) for video in record['source']['videos']}


def views(report: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    cv2.setNumThreads(1)
    records = json.loads((report / 'plan.json').read_text())['records']
    inventory: dict[str, Any] = {}
    for record in records:
        clip = record['clip']
        tracks = load_raw(report, record)
        inventory[clip] = {}
        for video in record['source']['videos']:
            camera = video['camera_id']
            raw = tracks[camera]
            output = destination / clip / camera
            output.mkdir(parents=True)
            sheet = track_sheet(Path(video['path']), raw, samples=12, crop_height=144)
            if sheet is None:
                raise ValueError('Expected observed raw tracks')
            for start in range(0, len(raw.track_ids), 6):
                cv2.imwrite(str(output / f'overview-{start:02d}.jpg'),
                            sheet[start*162:(start+6)*162], [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows = []
            for row, track in enumerate(raw.track_ids):
                frames = np.flatnonzero(raw.observed[row])
                centers = (raw.boxes_xyxy[row, frames, :2] + raw.boxes_xyxy[row, frames, 2:]) / 2
                heights = raw.boxes_xyxy[row, frames, 3] - raw.boxes_xyxy[row, frames, 1]
                jumps = np.flatnonzero(np.linalg.norm(np.diff(centers, axis=0), axis=1) > heights[1:]) + 1
                gaps = np.flatnonzero(np.diff(frames) > 2) + 1
                rows.append({'track': int(track), 'count': len(frames), 'first': int(frames[0]),
                             'last': int(frames[-1]), 'source_track_ids': raw.source_track_ids[row],
                             'inspect_boundaries': [[int(frames[i-1]), int(frames[i])] for i in np.union1d(jumps, gaps)]})
            inventory[clip][camera] = rows
            capture = cv2.VideoCapture(video['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            try:
                for frame in (0, video['num_frames']//2, video['num_frames']-1):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f'Cannot read {clip}/{camera}/{frame}')
                    for row in np.flatnonzero(raw.observed[:, frame]):
                        x1, y1, x2, y2 = np.rint(raw.boxes_xyxy[row, frame]).astype(int)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 230, 255), 2)
                        cv2.putText(image, f'r{raw.track_ids[row]}', (x1, max(20, y1)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 230, 255), 2)
                    cv2.imwrite(str(output / f'context-{frame:04d}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            finally:
                capture.release()
            print(clip, camera, len(rows), flush=True)
    (destination / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')


def labels(report: Path, reviews: Path) -> None:
    records = json.loads((report / 'plan.json').read_text())['records']
    for record in records:
        clip = record['clip']
        review_path = reviews / clip / 'review.yaml'
        review = yaml.safe_load(review_path.read_text())['clips'][clip]
        path = review_path.with_name('labels.json')
        if path.exists():
            raise FileExistsError(path)
        raw = load_raw(report, record)
        tracks = {camera: [ReviewedTrack(int(track), value.boxes_xyxy[row], value.observed[row])
                           for row, track in enumerate(value.track_ids)] for camera, value in raw.items()}
        result = materialize(clip, record['source']['videos'][0]['num_frames'], review, tracks,
                             {'review': file_identity(review_path), 'plan': file_identity(report / 'plan.json'),
                              'raw_receipt': file_identity(report / clip / 'person-execute.json'),
                              'blind_to_association': True, 'synthetic_gsi_included': False,
                              'scope': 'Partial reference from every raw observed track; not full detection recall'})
        result.save(path)
        print(file_identity(path), flush=True)


def detail_views(report: Path, requests: Path, destination: Path) -> None:
    """Explicit raw track/frame crops, with box coordinates and nearby context."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    cv2.setNumThreads(1)
    specification = json.loads(requests.read_text())
    records = json.loads((report / 'plan.json').read_text())['records']
    for record in records:
        clip = record['clip']
        if clip not in specification:
            continue
        tracks = load_raw(report, record)
        for video in record['source']['videos']:
            camera = video['camera_id']
            if camera not in specification[clip]:
                continue
            raw = tracks[camera]
            wanted: dict[int, list[tuple[int, int, int]]] = {}
            canvases: dict[int, np.ndarray] = {}
            for track, ranges in specification[clip][camera].items():
                row = int(np.flatnonzero(raw.track_ids == int(track))[0])
                frames = sorted({f for start, end, step in ranges for f in range(start, end, step) if raw.observed[row, f]})
                if not frames:
                    raise ValueError(f'No observed boxes for {clip}/{camera}/{track}')
                canvases[int(track)] = np.full((((len(frames)+11)//12)*174, 12*100, 3), 32, np.uint8)
                for cell, frame in enumerate(frames):
                    wanted.setdefault(frame, []).append((int(track), row, cell))
            capture = cv2.VideoCapture(video['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            try:
                for frame in range(max(wanted)+1):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f'Cannot read {clip}/{camera}/{frame}')
                    for track, row, cell in wanted.get(frame, []):
                        box = raw.boxes_xyxy[row, frame]
                        x1, y1, x2, y2 = np.rint(box).astype(int)
                        pad = max(4, int((y2-y1)*.2))
                        left, top = max(0,x1-pad), max(0,y1-pad)
                        crop = image[top:min(image.shape[0],y2+pad),left:min(image.shape[1],x2+pad)].copy()
                        cv2.rectangle(crop,(x1-left,y1-top),(x2-left,y2-top),(0,220,255),1)
                        scale = min(98/crop.shape[1], 140/crop.shape[0])
                        crop = cv2.resize(crop, (max(1,int(crop.shape[1]*scale)), max(1,int(crop.shape[0]*scale))))
                        y, x = (cell//12)*174, (cell%12)*100
                        canvas = canvases[track]
                        canvas[y:y+crop.shape[0],x:x+crop.shape[1]] = crop
                        for offset, text in ((153,f'r{track} f{frame}'),(168,f'x{x1} y{y1}')):
                            cv2.putText(canvas,text,(x,y+offset),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,220,255),1)
            finally:
                capture.release()
            root = destination / clip / camera
            root.mkdir(parents=True)
            for track, canvas in canvases.items():
                for page, start in enumerate(range(0, canvas.shape[0], 6*174)):
                    cv2.imwrite(str(root/f'r{track}-p{page}.jpg'),canvas[start:start+6*174],[cv2.IMWRITE_JPEG_QUALITY,94])
    (destination/'requests.json').write_bytes(requests.read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('views', 'details', 'labels'), required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=Path)
    args = parser.parse_args()
    if args.phase == 'details':
        if args.requests is None:
            parser.error('--requests is required for details')
        detail_views(args.report, args.requests, args.output)
    else:
        (views if args.phase == 'views' else labels)(args.report, args.output)

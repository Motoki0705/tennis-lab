"""Render frozen labels versus saved predictions after the single scoring receipt.

No inference, fitting, matching or scoring. Raw boxes must equal the committed
reference boxes in frame order. Display ID names come from the saved scoring
assignment, and must be consistent across cameras (no camera-specific remap).
Colors describe individual boxes; duplicate-box exclusions need not be metric errors.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from person_unseen import checked  # type: ignore[import-not-found]

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tennis_scene.pipeline.definition import file_identity

COLORS = {'agree': (80, 230, 80), 'disagree': (70, 70, 255),
          'prediction_only': (0, 190, 255), 'label_only': (240, 80, 240),
          'unknown': (160, 160, 160)}


def category(person: str | None, role: str | None, identity: int, mapping: dict[int, str]) -> str:
    if person is None:
        return 'prediction_only' if identity >= 0 else 'unknown'
    if role == 'non_player':
        return 'agree' if identity < 0 else 'disagree'
    if identity < 0:
        return 'label_only'
    return 'agree' if mapping.get(identity) == person else 'disagree'


def saved_mapping(summary: dict[str, Any], clip: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for row in summary['tracking']['associated']['mapping']:
        if row['clip'] != clip:
            continue
        identity, person = int(row['track_id']), str(row['person'])
        if mapping.setdefault(identity, person) != person:
            raise ValueError('Conflicting camera assignments; provide an explicit global display map')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Display map is not one-to-one')
    return mapping


def render_clip(record: dict[str, Any], arrays: dict[str, np.ndarray], labels: ClipLabels,
                mapping: dict[int, str], output: Path, status: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    videos = record['source']['videos']
    cameras = [v['camera_id'] for v in videos]
    if cameras != ['cam0', 'cam1', 'cam2'] or set(labels.cameras) != set(cameras):
        raise ValueError('Require all three cameras')
    frames, fps = videos[0]['num_frames'], videos[0]['fps']
    if labels.num_frames != frames or labels.clip_id != record['clip']:
        raise ValueError('Label timeline differs')
    cv2.setNumThreads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    captures = [cv2.VideoCapture(v['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]) for v in videos]
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', '2880x620', '-r', str(fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
               '-threads', '1', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in range(frames):
            panels = []
            for video, capture in zip(videos, captures, strict=True):
                camera = video['camera_id']
                ok, image = capture.read()
                if not ok or image.shape[:2] != (video['height'], video['width']):
                    raise ValueError(f'Missing source frame {camera}/{frame}')
                panel = cv2.resize(image, (960, 540))
                rows = np.flatnonzero(arrays[f'{camera}_observed'][:, frame])
                reference = labels.cameras[camera]
                at = reference.at(frame)
                boxes = reference.boxes_xyxy[at]
                # Labels were materialized from exactly these raw tracks. This
                # check prevents a wrong ordering/source from masquerading as GT.
                if boxes.shape != (len(rows), 4) or not np.allclose(
                    np.round(arrays[f'{camera}_boxes'][rows, frame].astype(float), 1), boxes, atol=.051, rtol=0):
                    raise ValueError('Reference and raw prediction boxes differ')
                for index, row in enumerate(rows):
                    person_index = int(reference.person_index[at][index])
                    person = labels.people[person_index] if person_index >= 0 else None
                    identity = int(arrays[f'{camera}_ids'][row, frame])
                    kind = category(None if person is None else person.person_id,
                                    None if person is None else person.role, identity, mapping)
                    color = COLORS[kind]
                    x1, y1, x2, y2 = np.rint(
                        boxes[index]*[960/video['width'],540/video['height'],960/video['width'],540/video['height']]).astype(int)
                    cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
                    truth = '?' if person is None else person.person_id
                    predicted = '-' if identity < 0 else f'{identity}:{mapping.get(identity,"?")}'
                    text = f'G:{truth} P:{predicted}'
                    cv2.putText(panel, text, (max(1,min(x1,840)), max(42,y1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, .43, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(panel, text, (max(1,min(x1,840)), max(42,y1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
                cv2.rectangle(panel, (0,0), (960,30), (20,20,20), -1)
                cv2.putText(panel, f'{camera} | {frame}/{frames-1} | {frame/fps:.2f}s | G=label P=prediction',
                            (10,21), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1)
                panels.append(panel)
            footer: np.ndarray = np.full((80,2880,3),20,np.uint8)
            cv2.putText(footer, f'{record["clip"]} | association: {status} | partial reference | saved predictions; no re-inference',
                        (14,24), cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1)
            legend = [('agree','ID agrees / non-player excluded'),('disagree','wrong ID / non-player kept'),
                      ('prediction_only','prediction without label'),('label_only','player label without ID'),
                      ('unknown','ambiguous and unassigned')]
            for index,(name,text) in enumerate(legend):
                cv2.putText(footer,text,(14+index*565,58),cv2.FONT_HERSHEY_SIMPLEX,.58,COLORS[name],1)
            process.stdin.write(np.concatenate((np.concatenate(panels,axis=1),footer)).tobytes())
        process.stdin.close()
        if process.wait(timeout=60) != 0:
            raise RuntimeError('Video encoder failed')
    except BaseException:
        process.kill()
        process.wait(timeout=10)
        raise
    finally:
        for capture in captures:
            capture.release()
    capture = cv2.VideoCapture(str(output),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    count = 0
    try:
        if not np.isclose(capture.get(cv2.CAP_PROP_FPS),fps,atol=.01):
            raise ValueError('Wrong review FPS')
        while True:
            ok,image = capture.read()
            if not ok:
                break
            if image.shape != (620,2880,3):
                raise ValueError('Wrong review dimensions')
            count += 1
    finally:
        capture.release()
    if count != frames:
        raise ValueError('Incomplete review video')
    return {'file':file_identity(output),'frames_read':count,'frames_written':frames,
            'fps':fps,'size':[2880,620],'display_mapping':mapping,'colors_bgr':COLORS,
            'classification_scope':'per raw box; duplicate exclusions may not be metric errors'}


def render_all(report: Path) -> None:
    summary = json.loads((report/'scoring/summary.json').read_text())
    attempt = json.loads((report/'scoring/attempt.json').read_text())
    if summary['scoring_batches'] != 1 or summary['status'] != 'ok':
        raise ValueError('Require the completed single scoring batch')
    plan = json.loads((report/'plan.json').read_text())
    complete = json.loads((report/'complete.json').read_text())
    receipt = {}
    for record in plan['records']:
        clip = record['clip']
        labels = ClipLabels.load(checked(attempt['labels'][clip]))
        prediction = json.loads(checked(complete['clips'][clip]['prediction']).read_text())
        with np.load(checked(prediction['arrays']), allow_pickle=False) as saved:
            arrays = {k:saved[k] for k in saved.files}
        receipt[clip] = render_clip(record,arrays,labels,saved_mapping(summary,clip),
                                   report/clip/'labels_vs_predictions_r18.mp4',prediction['status'])
        (report/'review-videos-r18.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(clip,receipt[clip],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    render_all(parser.parse_args().report)

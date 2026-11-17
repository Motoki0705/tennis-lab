"""CPU-only rendering and counts from the existing review; no inference/tuning."""

import collections
import csv
import hashlib
import json
import pathlib
import subprocess

import cv2
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
SOURCE = pathlib.Path('/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/meiji4cam-court-review-20261116')
DATASET = pathlib.Path('data/tennis_multiview/processed/meiji_4cam/dataset')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = json.loads((SOURCE / 'summary.json').read_text())
    manifest = json.loads((DATASET / 'dataset.json').read_text())
    assert len(rows) == 276
    assert len({(r['clip_id'], r['camera']) for r in rows}) == 276
    stats = {}
    table = []
    crops = [(40, 180, 780, 385), (215, 200, 900, 430), (100, 195, 740, 390), (245, 110, 960, 330)]
    for cam in range(4):
        camera = f'cam{cam}'
        subset = [r for r in rows if r['camera'] == camera]
        gates = collections.Counter()
        tiles = []
        for row in subset:
            best = max(row['candidates'], key=lambda c: (c['accepted'], c['inliers'], c['inlier_score']))
            assert best['accepted'] == row['accepted']
            flags = []
            if best['inliers'] < 8:
                flags.append('inliers')
            if not best['inside_content']:
                flags.append('outside')
            gate = '+'.join(flags) if flags else 'accepted'
            gates[gate] += 1
            clip = row['clip_id'].split('/')[-1]
            tile = cv2.imread(str(SOURCE / f'{clip}.jpg'))
            x, y = (cam % 2) * 960, (cam // 2) * 540
            tile = tile[y:y+540, x:x+960]
            (OUT / 'tiles').mkdir(exist_ok=True)
            cv2.imwrite(str(OUT / 'tiles' / f'{clip}_{camera}.jpg'), tile)
            x0, y0, x1, y1 = crops[cam]
            zoom = cv2.resize(tile[y0:y1, x0:x1], (720, 230))
            panel = np.zeros((265, 720, 3), np.uint8)
            panel[35:] = zoom
            text = f'{clip} {camera} {"ACCEPT" if row["accepted"] else "REJECT"} n={best["inliers"]} {gate}'
            cv2.putText(panel, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (235,235,235), 1, cv2.LINE_AA)
            tiles.append(panel)
            table.append({'clip_id': row['clip_id'], 'camera': camera, 'old_accepted': row['accepted'],
                          'best_inliers': best['inliers'], 'gate_class': gate,
                          'region_xyxy': json.dumps(best['region_xyxy']), 'visual_class': 'unreviewed',
                          'human_reference_status': 'not_reviewed'})
        for start in range(0, len(tiles), 12):
            part = tiles[start:start+12]
            while len(part) < 12:
                part.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(part[j:j+2]) for j in range(0, 12, 2)])
            cv2.imwrite(str(OUT / f'{camera}_{start:03d}_{min(start+11,68):03d}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        stats[camera] = {'total': len(subset), 'gates': dict(gates),
                         'best_inliers': dict(collections.Counter(r['best_inliers'] for r in subset)),
                         'candidate_status': dict(collections.Counter(c['status'] for r in subset for c in r['candidates']))}
    (OUT / 'gate_counts.json').write_text(json.dumps(stats, indent=2))
    with (OUT / 'visual_review.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    originals = OUT / 'frames'
    originals.mkdir(exist_ok=True)
    for clip_index in (0, 34, 68):
        clip = manifest['clips'][clip_index]
        for cam in range(4):
            source = DATASET / clip['path'] / 'media' / f'cam{cam}.mp4'
            video = cv2.VideoCapture(str(source))
            ok, frame = video.read()
            video.release()
            if not ok:
                raise RuntimeError(source)
            assert frame.shape[:2] == (1080, 1920)
            cv2.imwrite(str(originals / f'clip_{clip_index:03d}_cam{cam}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
    provenance = {'head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  'source_summary': str(SOURCE / 'summary.json'),
                  'source_summary_sha256': sha(SOURCE / 'summary.json'),
                  'source_script_sha256': sha(SOURCE / 'court_review.py'),
                  'dataset_manifest_sha256': sha(DATASET / 'dataset.json'),
                  'no_model_inference': True, 'no_reference_evaluation': True,
                  'visual_review_is_provisional': True}
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

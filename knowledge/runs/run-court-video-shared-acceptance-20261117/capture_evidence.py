"""Promote small summaries; retain full native images/predictions in outputs."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from build import save, sha


def inventory(root, folder, target):
    records = []
    for path in sorted((root / folder).rglob('*')):
        if path.is_file() and not path.name.endswith('.lock'):
            records.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(row) + '\n' for row in records))
    return {'root': str(root / folder), 'inventory_path': str(target), 'files': len(records),
            'bytes': sum(r['bytes'] for r in records), 'sha256': sha(target)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root, bundle = args.output, Path(__file__).resolve().parent
    # Refuse to call an unfinished compatibility run a complete evidence set.
    replay = json.loads((root / 'legacy-replay/summary.json').read_text())
    assert replay['images'] == 3545 and replay['regions'] == 8530 and not replay['mismatches']
    integrity = json.loads((root / 'input-integrity.json').read_text())
    assert all(row['all_hashes_match'] for row in integrity['groups'].values())
    for name in ('build-summary.json', 'evaluation-summary.json', 'cases.csv', 'drift-lines.csv',
                 'input-integrity.json', 'gallery-browser-check.json'):
        assert (root / name).stat().st_size <= 256 * 1024
        shutil.copyfile(root / name, bundle / name)
    shutil.copyfile(root / 'legacy-replay/summary.json', bundle / 'legacy-replay-summary.json')
    shutil.copyfile(root / 'legacy-replay/replay-source/protocol.json', bundle / 'legacy-replay-protocol.json')
    protocol = json.loads((root / 'build-protocol.json').read_text())
    save(bundle / 'build-protocol-summary.json', {k: v for k, v in protocol.items() if k != 'sources'} |
         {'full_protocol_sha256': sha(root / 'build-protocol.json'), 'source_files': len(protocol['sources'])})
    groups = {folder: inventory(root, folder, root / 'inventories' / (folder + '.jsonl'))
              for folder in ('store', 'inputs', 'review', 'synthetic', 'legacy-replay',
                             'pre-heldout-interrupted-build', 'ci-import-interrupted', 'cpu-thread-interrupted')}
    save(bundle / 'artifact-inventories.json', groups)

    split_path = Path('knowledge/runs/run-court-video-shared-stage1-20261117/frame-split.json')
    split = json.loads(split_path.read_text())
    references = json.loads(Path('knowledge/runs/run-court-white-line-reference-20261116/reference/frame-stability.json').read_text())
    clips = {p.parent.name: json.loads(p.read_text()) for p in (args.dataset / 'videos/video_000/clips').glob('*/clip.json')}
    def source_key(camera, clip, frame):
        data = clips[clip.split('/')[1]]
        view = next(c for c in data['cameras'] if c['camera_id'] == camera)
        return view['source_path'], view['source_frame_start'] + frame
    excluded = {source_key(c['camera'], r['clip_id'], r['frame_index']) for c in references for r in c['frames']}
    counts = {}
    for phase, frames in split.items():
        values = {source_key(r['camera_id'], r['clip_id'], r['frame_index']) for r in frames}
        assert len(values) == len(frames) and not values & excluded
        assert all(source_key(r['camera_id'], r['clip_id'], r['frame_index'])[1] == r['source_frame'] for r in frames)
        counts[phase] = {'frames': len(values), 'reference_source_frame_overlap': 0}
    save(bundle / 'split-integrity.json', {'reference_source_frames': len(excluded), 'phases': counts, 'split_sha256': sha(split_path)})

    evaluation = json.loads((root / 'evaluation-summary.json').read_text())
    rows = list(csv.DictReader((root / 'cases.csv').open()))
    cards = []
    for camera in sorted(evaluation['cameras']):
        metadata = evaluation['cameras'][camera]
        case = next(r for r in rows if r['camera'] == camera and r['clip_id'] == 'video_000/clip_001')
        photo = cv2.imread(str(root / 'review' / camera / 'clip_001/heldout/overlay.jpg'))
        photo = cv2.resize(photo, (480, 270), interpolation=cv2.INTER_AREA)
        photo[:14] = (25, 25, 25)  # replace the downscaled tiny caption below
        card = np.full((316, 480, 3), 25, np.uint8)
        card[46:] = photo
        cv2.putText(card, f'{camera} clip_001 / heldout / {case["clip_status"]}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        error = metadata['accuracy']
        cv2.putText(card, f'reference mean {error["mean_px"]:.2f}px / max {error["max_px"]:.2f}px', (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        cards.append(card)
    overview = np.vstack((np.hstack(cards[:2]), np.hstack(cards[2:])))
    Image.fromarray(cv2.cvtColor(overview, cv2.COLOR_BGR2RGB)).quantize(colors=256).save(bundle / 'overview.png', optimize=True)
    assert (bundle / 'overview.png').stat().st_size <= 512 * 1024
    print('Promoted summaries and image overview; full predictions/native images retained in outputs.')

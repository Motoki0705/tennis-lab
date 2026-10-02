"""Extract auditable cause examples from saved traces, without changing tracking."""
import gzip
import json
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent
BASE = Path('/home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix')
FEATURES = Path('/home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/dev_features/i964-features-r7-20260930')
VARIANT = 'deep_ocsort_pose__clipreid_vitb16_market1501'


def main():
    diag = json.loads((BASE / 'i964-linking-r9-20260930-v2/diagnosis.json').read_text())
    output = {'miss_accounting': {}, 'examples': {}, 'duplicate_detections': {}}
    for clip, value in diag['clips'].items():
        result = json.loads((BASE / f'i964-matrix-r8-20260930/evaluation/{VARIANT}/{clip}/result.json').read_text())
        camera = result['cameras']['cam1']
        with np.load(camera['arrays']['path'], allow_pickle=False) as data:
            ids, selected = data['track_ids'].tolist(), data['selected']
        with gzip.open(value['details']['path'], 'rt') as stream:
            details = {d['frame']: d for line in stream if (d := json.loads(line))}
        counts = {'not_emitted': 0, 'emitted_unselected': 0, 'emitted_selected_other_label': 0, 'missing_source_detection': 0}
        for item in details.values():
            if item['state'] != 'miss':
                continue
            if item['detection_iou'] < .5:
                counts['missing_source_detection'] += 1
            elif item.get('emitted_id') is None:
                counts['not_emitted'] += 1
            elif selected[ids.index(item['emitted_id']), item['frame']]:
                counts['emitted_selected_other_label'] += 1
            else:
                counts['emitted_unselected'] += 1
        output['miss_accounting'][clip] = counts
        events = sorted({e['frame'] for e in value['events']['switch_events']} | ({963, 966} if clip == 'video_001/clip_001' else set()))
        output['examples'][clip] = {str(frame): details[frame] for frame in events}
        birth = {'video_000/clip_000': 476, 'video_002/clip_013': 176}.get(clip)
        if birth is not None:
            with np.load(FEATURES / f'clipreid_vitb16_market1501/{clip}/cam1.features.npz', allow_pickle=False) as features:
                at = slice(*features['offsets'][birth:birth + 2])
                boxes = features['boxes'][at]
                left, right = boxes[1:3]
                intersection = np.maximum(np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]), 0).prod()
                iou = float(intersection / ((left[2:] - left[:2]).prod() + (right[2:] - right[:2]).prod() - intersection))
                output['duplicate_detections'][clip] = {'frame': birth, 'rows': features['rows'][at].tolist(),
                    'boxes': boxes.tolist(), 'scores': features['scores'][at].tolist(), 'far_pair_iou': iou}
    (BUNDLE / 'diagnosis-causes.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'miss_accounting': output['miss_accounting'], 'duplicates': output['duplicate_detections']}, indent=2))


if __name__ == '__main__':
    main()

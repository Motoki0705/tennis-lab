"""Post-hoc attribution of missing wide reference units, never selection input."""
import collections
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.partial_labels import _unit_matches
from src.utils.geometry.bbox import pairwise_iou

report = Path(sys.argv[1])
source = json.loads((report / 'sources.json').read_text())
losses = []
for path in sorted((report / 'selection').glob('*/*/*/result.json')):
    result = json.loads(path.read_text())
    label_path = next(r['label_path'] for r in source['inputs'] if r['clip'] == result['clip'])
    labels = ClipLabels.load(Path(label_path))
    with gzip.open(result['units']['path'], 'rt') as handle:
        units = [json.loads(line) for line in handle]
    for camera, saved in result['cameras'].items():
        with np.load(saved['path'], allow_pickle=False) as archive:
            ids, boxes, seen, chosen = (archive[k] for k in ('track_ids','boxes','observed','selected'))
        reference = labels.cameras[camera]
        for unit in units:
            if unit['stage'] != 'selected' or unit['camera'] != camera or unit['role'] != 'player' or not unit['reference_wide'] or unit['selected_03']:
                continue
            item = {k: unit[k] for k in ('source','clip','camera','frame','person','tracked_03','reference_x_m')}
            if not unit['tracked_03']:
                item['reason'] = 'no_tracked_box_match'
                losses.append(item)
                continue
            frame = unit['frame']
            rows = np.flatnonzero(seen[:,frame])
            at = reference.at(frame)
            people, refboxes = reference.person_index[at], reference.boxes_xyxy[at]
            people_units = np.unique(people[people >= 0])
            overlap = pairwise_iou(boxes[rows,frame],refboxes)
            iou = np.column_stack([overlap[:,people==p].max(1) for p in people_units])
            left,right = _unit_matches(iou,.3)
            person_index = next(i for i,p in enumerate(labels.people) if p.person_id==unit['person'])
            row = next(int(rows[l]) for l,r in zip(left,right,strict=True) if people_units[r]==person_index)
            item['track_id'] = int(ids[row])
            if chosen[row,frame]:
                item['reason'] = 'competing_match_after_selection'
            else:
                fi,fragment = next((i,f) for i,f in enumerate(saved['linked']['fragments']) if f['row']==row and f['start']<=frame<f['end'])
                item.update(fragment_start=fragment['start'],fragment_end=fragment['end'])
                if fragment['excluded_short_ambiguous']:
                    item['reason'] = 'short_ambiguous_fragment'
                else:
                    group = next(g for g in saved['linked']['groups'] if fi in g['fragments'])
                    item.update(reason=group['reason'],in_core_frames=group['in_core_frames'],required_frames=saved['linked']['required_frames'])
            losses.append(item)
counts = collections.Counter((r['source'],r['reason']) for r in losses)
output={'method':'same one-to-one IoU .3 reference-unit assignment as selection metric; then original fragment/group diagnostic; no label enters selection',
        'counts':[{'source':k[0],'reason':k[1],'units':v} for k,v in sorted(counts.items())], 'units':losses}
(report/'wide_losses.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output['counts'],indent=2))
